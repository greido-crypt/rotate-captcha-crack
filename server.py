import argparse
import asyncio
import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from rotate_captcha_crack.common import device
from rotate_captcha_crack.const import DEFAULT_CLS_NUM
from rotate_captcha_crack.model import QuantRotNetR, RotNetR, WhereIsMyModel
from rotate_captcha_crack.utils import process_captcha

# --- RTX 4050 laptop (6GB VRAM, Ada Lovelace) tuning ---
MAX_BATCH   = 4     # safe for 6GB; lower to 4 to reduce GPU spike
MAX_WAIT_MS = 30    # ms to wait before flushing a partial batch

parser = argparse.ArgumentParser()
parser.add_argument("--index", "-i", type=int, default=-1, help="Model index (-1 = latest)")
parser.add_argument("--quant", action="store_true", help="Use INT8 quantized model (CPU)")
parser.add_argument("--cpu", action="store_true", help="Force FP32 model on CPU (no GPU required)")
opts = parser.parse_args()

cls_num = DEFAULT_CLS_NUM


def _load_fp32_model():
    """Load standard RotNetR — GPU by default, CPU if --cpu flag is set."""
    target_device = torch.device("cpu") if opts.cpu else device

    m = RotNetR(cls_num=cls_num, train=False)
    model_path = WhereIsMyModel(m).with_index(opts.index).model_dir / "best.pth"
    m.load_state_dict(torch.load(model_path, map_location=target_device, weights_only=True))
    m = m.to(device=target_device)
    m.eval()

    if target_device.type == "cuda":
        try:
            import triton  # noqa: F401
            m = torch.compile(m, mode="reduce-overhead")
            print("INFO: torch.compile enabled")
        except ModuleNotFoundError:
            print("WARNING: triton not found — torch.compile skipped (not supported on Windows). "
                  "Run under WSL2 for full performance.")

    # Warmup
    with torch.inference_mode():
        _dummy = torch.zeros(1, 3, 224, 224, device=target_device)
        if target_device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                m(_dummy)
            torch.cuda.empty_cache()
        else:
            m(_dummy)
        del _dummy

    return m, target_device


def _load_quant_model():
    """Load INT8 quantized QuantRotNetR on CPU."""
    # Build the converted model skeleton (same steps as quant_RotNetR.py)
    m = QuantRotNetR(cls_num=cls_num, train=False)
    m.eval()
    m.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
    m = torch.ao.quantization.fuse_modules(m, [["conv", "bn", "relu"]])
    m = torch.ao.quantization.prepare_qat(m.train())
    m = torch.ao.quantization.convert(m)

    model_path = WhereIsMyModel(QuantRotNetR(cls_num=cls_num, train=False)).with_index(opts.index).model_dir / "quant.pth"
    m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    m.eval()

    # Warmup
    with torch.inference_mode():
        m(torch.zeros(1, 3, 224, 224))

    return m, torch.device("cpu")


if opts.quant:
    print("INFO: Loading INT8 quantized model (CPU)")
    model, infer_device = _load_quant_model()
elif opts.cpu:
    print("INFO: Loading FP32 model (CPU)")
    model, infer_device = _load_fp32_model()
else:
    print("INFO: Loading FP16 model (GPU)")
    model, infer_device = _load_fp32_model()

print("INFO: Model ready")

# --- CPU thread pool for parallel image preprocessing ---
_cpu_pool = ThreadPoolExecutor(max_workers=os.cpu_count(), thread_name_prefix="preproc")

# --- Shared request queue: items are (img_bytes, asyncio.Future) ---
_queue: asyncio.Queue = asyncio.Queue()


def _decode_and_preprocess(img_bytes: bytes) -> torch.Tensor:
    """Runs in ThreadPoolExecutor. Returns [3, 224, 224] float32 CPU tensor."""
    img = Image.open(io.BytesIO(img_bytes))
    return process_captcha(img)


async def _batcher():
    """
    Drains _queue into batches.
    Flushes when MAX_BATCH images are ready OR MAX_WAIT_MS have passed.
    """
    loop = asyncio.get_running_loop()
    use_autocast = (infer_device.type == "cuda")

    while True:
        first = await _queue.get()
        batch: list[tuple[bytes, asyncio.Future]] = [first]

        deadline = loop.time() + MAX_WAIT_MS / 1000
        while len(batch) < MAX_BATCH:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(_queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        # Preprocess all images in parallel on CPU
        preprocess_futs = [
            loop.run_in_executor(_cpu_pool, _decode_and_preprocess, img_bytes)
            for img_bytes, _ in batch
        ]
        results = await asyncio.gather(*preprocess_futs, return_exceptions=True)

        valid: list[tuple[torch.Tensor, asyncio.Future]] = []
        for (_, fut), result in zip(batch, results):
            if isinstance(result, Exception):
                fut.set_exception(result)
            else:
                valid.append((result, fut))

        if not valid:
            continue

        try:
            batch_ts = torch.stack([t for t, _ in valid]).to(device=infer_device)
            with torch.inference_mode():
                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(batch_ts)
                else:
                    logits = model(batch_ts)
            angles = logits.argmax(dim=1).tolist()
            for angle_idx, (_, fut) in zip(angles, valid):
                fut.set_result(angle_idx / cls_num * 360)
        except Exception as exc:
            for _, fut in valid:
                if not fut.done():
                    fut.set_exception(exc)


# --- FastAPI app ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.ensure_future(_batcher())
    yield
    task.cancel()
    _cpu_pool.shutdown(wait=False)


app = FastAPI(title="rotate-captcha-crack", lifespan=lifespan)


class InferRequest(BaseModel):
    image_base64: str


def _error(code: int, msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"err": {"code": code, "msg": msg}}, status_code=status)


async def _run_inference(img_bytes: bytes) -> JSONResponse:
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    await _queue.put((img_bytes, fut))
    try:
        pred = await fut
    except Exception as exc:
        return _error(1, str(exc))
    return JSONResponse({"err": {"code": 0}, "pred": pred})


@app.post("/")
async def infer_json(req: InferRequest):
    """Accept JSON with base64-encoded image."""
    try:
        img_bytes = base64.b64decode(req.image_base64, validate=True)
    except Exception:
        return _error(1, "invalid base64")
    return await _run_inference(img_bytes)


@app.post("/raw")
async def infer_raw(request: Request):
    """Accept raw binary image (application/octet-stream)."""
    img_bytes = await request.body()
    return await _run_inference(img_bytes)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=4396,
        log_level="info",
        access_log=True,
    )
