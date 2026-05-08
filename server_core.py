"""
server_core.py — importable server logic.
No side effects at import time. All state lives in ServerManager.
"""

import asyncio
import base64
import io
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from rotate_captcha_crack import const as _rcc_const
from rotate_captcha_crack.const import DEFAULT_CLS_NUM
from rotate_captcha_crack.model import RotNetR, WhereIsMyModel
from rotate_captcha_crack.utils import process_captcha

MAX_BATCH   = 4
MAX_WAIT_MS = 30
cls_num     = DEFAULT_CLS_NUM


# ── Runtime path helper ───────────────────────────────────────────────────────

def _base_dir() -> Path:
    """
    Returns directory that contains the 'models' folder.
    PyInstaller ≥6 puts bundled data inside _internal/ (sys._MEIPASS).
    """
    if getattr(sys, "frozen", False):
        meipass = Path(getattr(sys, "_MEIPASS", ""))
        if (meipass / "models").exists():
            return meipass
        return Path(sys.executable).parent
    return Path(__file__).parent


# Patch MODELS_DIR so WhereIsMyModel resolves correctly everywhere
_rcc_const.MODELS_DIR = str(_base_dir() / "models")


# ── Logging handler that routes records to a callback ─────────────────────────

class _CallbackHandler(logging.Handler):
    def __init__(self, cb: Callable[[str], None]):
        super().__init__()
        self._cb = cb

    def emit(self, record: logging.LogRecord):
        try:
            self._cb(self.format(record))
        except Exception:
            pass


# ── Image saving ──────────────────────────────────────────────────────────────

def _save_pair(img_bytes: bytes, angle_deg: float, save_dir: Path) -> None:
    """Save input image and solved (rotated) image as a named pair."""
    try:
        uid = f"{int(time.time() * 1000)}"
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(save_dir / f"{uid}_input.png")
        # rotate(-angle) recovers original orientation (per model convention)
        img.rotate(-angle_deg, expand=True).save(
            save_dir / f"{uid}_solved_{angle_deg:.1f}deg.png"
        )
    except Exception:
        pass   # never crash the inference on save failure


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_fp32_model(index: int, use_cpu: bool, log: Callable[[str], None]):
    from rotate_captcha_crack.common import device as cuda_device

    target = torch.device("cpu") if use_cpu else cuda_device
    log(f"INFO: target device → {target}")

    m = RotNetR(cls_num=cls_num, train=False)
    model_path = WhereIsMyModel(m).with_index(index).model_dir / "best.pth"
    log(f"INFO: loading weights from {model_path}")

    m.load_state_dict(torch.load(model_path, map_location=target, weights_only=True))
    m = m.to(device=target)
    m.eval()

    if target.type == "cuda":
        try:
            import triton  # noqa: F401
            m = torch.compile(m, mode="reduce-overhead")
            log("INFO: torch.compile enabled")
        except ModuleNotFoundError:
            log("WARNING: triton not found — torch.compile skipped (Windows limitation)")

    log("INFO: warmup…")
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 224, 224, device=target)
        if target.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                m(dummy)
            torch.cuda.empty_cache()
        else:
            m(dummy)
        del dummy

    return m, target


# ── ServerManager ─────────────────────────────────────────────────────────────

class ServerManager:
    """Manages the full lifecycle of the inference server."""

    def __init__(self):
        self._server:   Optional[uvicorn.Server] = None
        self._thread:   Optional[threading.Thread] = None
        self._log_cb:   Optional[Callable[[str], None]] = None
        self._cpu_pool: Optional[ThreadPoolExecutor] = None
        self._log_handler: Optional[_CallbackHandler] = None
        self.request_count = 0
        self.save_dir: Optional[Path] = None
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_log_callback(self, cb: Callable[[str], None]):
        self._log_cb = cb

    def start(self, mode: str = "gpu", port: int = 4396, index: int = -1):
        """mode: 'gpu' | 'cpu'. Blocks until server is up (call from background thread)."""
        with self._lock:
            if self.is_running:
                raise RuntimeError("Server is already running")

        self._log("=" * 50)
        self._log(f"INFO: starting server  mode={mode}  port={port}")
        self.request_count = 0

        if mode == "cpu":
            self._log("INFO: loading FP32 model (CPU)…")
            model, infer_device = _load_fp32_model(index, use_cpu=True,  log=self._log)
        else:
            self._log("INFO: loading FP16 model (GPU)…")
            model, infer_device = _load_fp32_model(index, use_cpu=False, log=self._log)

        self._log("INFO: model ready ✓")

        self._cpu_pool = ThreadPoolExecutor(
            max_workers=os.cpu_count(), thread_name_prefix="preproc"
        )
        fastapi_app = self._build_app(model, infer_device)

        # Attach callback handler to uvicorn loggers (suppress default config)
        fmt = logging.Formatter("%(levelname)s  %(message)s")
        handler = _CallbackHandler(self._log)
        handler.setFormatter(fmt)
        self._log_handler = handler
        for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
            lg = logging.getLogger(name)
            lg.handlers.clear()
            lg.addHandler(handler)
            lg.setLevel(logging.INFO)
            lg.propagate = False

        # minimal log_config so uvicorn doesn't try to reconfigure loggers
        _silent_log_cfg = {"version": 1, "disable_existing_loggers": False}

        config = uvicorn.Config(
            fastapi_app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True,
            log_config=_silent_log_cfg,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        self._log(f"INFO: listening on http://0.0.0.0:{port}")
        self._log(f"INFO: swagger → http://127.0.0.1:{port}/docs")

    def stop(self):
        with self._lock:
            if not self.is_running:
                return
            self._server.should_exit = True

        # Give uvicorn 3s to stop gracefully, then force-exit
        self._thread.join(timeout=3)
        if self._thread.is_alive():
            self._server.force_exit = True
            self._thread.join(timeout=5)

        # Clean up
        if self._cpu_pool:
            self._cpu_pool.shutdown(wait=False)
            self._cpu_pool = None
        if self._log_handler:
            for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
                logging.getLogger(name).removeHandler(self._log_handler)
            self._log_handler = None

        self._server = None
        self._thread = None
        self._log("INFO: server stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @staticmethod
    def gpu_info() -> dict:
        if not torch.cuda.is_available():
            return {"available": False}
        idx   = torch.cuda.current_device()
        name  = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        used  = torch.cuda.memory_allocated(idx) / 1024 ** 3
        return {"available": True, "name": name,
                "total_gb": round(total, 1), "used_gb": round(used, 2)}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self._log_cb:
            self._log_cb(msg)

    def _build_app(self, model, infer_device) -> FastAPI:
        queue:       asyncio.Queue = asyncio.Queue()
        cpu_pool     = self._cpu_pool
        use_autocast = (infer_device.type == "cuda")
        mgr          = self

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            task = asyncio.ensure_future(_batcher())
            yield
            task.cancel()

        app = FastAPI(title="rotate-captcha-crack", lifespan=lifespan)

        @app.middleware("http")
        async def _count(request: Request, call_next):
            mgr.request_count += 1
            return await call_next(request)

        async def _batcher():
            loop = asyncio.get_running_loop()
            while True:
                first = await queue.get()
                batch: list[tuple[bytes, asyncio.Future]] = [first]

                deadline = loop.time() + MAX_WAIT_MS / 1000
                while len(batch) < MAX_BATCH:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=remaining)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                # Preprocess (CPU, parallel)
                futs_pre = [
                    loop.run_in_executor(cpu_pool, _preprocess, raw)
                    for raw, _ in batch
                ]
                results = await asyncio.gather(*futs_pre, return_exceptions=True)

                # Separate errors from valid; keep raw bytes for saving
                valid: list[tuple[bytes, torch.Tensor, asyncio.Future]] = []
                for (raw, fut), res in zip(batch, results):
                    if isinstance(res, Exception):
                        fut.set_exception(res)
                    else:
                        valid.append((raw, res, fut))

                if not valid:
                    continue

                # GPU inference
                try:
                    batch_ts = torch.stack([t for _, t, _ in valid]).to(device=infer_device)
                    with torch.inference_mode():
                        if use_autocast:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                logits = model(batch_ts)
                        else:
                            logits = model(batch_ts)
                    angles = logits.argmax(dim=1).tolist()

                    for angle_idx, (raw, _, fut) in zip(angles, valid):
                        angle_deg = angle_idx / cls_num * 360
                        fut.set_result(angle_deg)
                        # Save pair if save_dir is configured
                        if mgr.save_dir:
                            loop.run_in_executor(
                                cpu_pool, _save_pair, raw, angle_deg, mgr.save_dir
                            )

                except Exception as exc:
                    for _, _, fut in valid:
                        if not fut.done():
                            fut.set_exception(exc)

        def _preprocess(img_bytes: bytes) -> torch.Tensor:
            return process_captcha(Image.open(io.BytesIO(img_bytes)))

        def _err(code: int, msg: str, status: int = 400) -> JSONResponse:
            return JSONResponse({"err": {"code": code, "msg": msg}}, status_code=status)

        async def _infer(img_bytes: bytes) -> JSONResponse:
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            await queue.put((img_bytes, fut))
            try:
                pred = await fut
            except Exception as exc:
                return _err(1, str(exc))
            return JSONResponse({"err": {"code": 0}, "pred": pred})

        class InferRequest(BaseModel):
            image_base64: str

        @app.post("/")
        async def infer_json(req: InferRequest):
            try:
                img_bytes = base64.b64decode(req.image_base64, validate=True)
            except Exception:
                return _err(1, "invalid base64")
            return await _infer(img_bytes)

        @app.post("/raw")
        async def infer_raw(request: Request):
            return await _infer(await request.body())

        return app
