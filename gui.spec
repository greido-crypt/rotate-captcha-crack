# gui.spec — PyInstaller spec for rotate-captcha-crack GUI
# Run: pyinstaller gui.spec

import fnmatch
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# ── Collect data files ────────────────────────────────────────────────────────

datas = []
datas += collect_data_files("customtkinter")
datas += [("src/rotate_captcha_crack", "rotate_captcha_crack")]
datas += [("models", "models")]

# ── Hidden imports ────────────────────────────────────────────────────────────

hiddenimports = [
    "psutil",
    # uvicorn
    "uvicorn.logging",
    "uvicorn.loops", "uvicorn.loops.auto", "uvicorn.loops.asyncio",
    "uvicorn.protocols", "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto", "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan", "uvicorn.lifespan.on", "uvicorn.lifespan.off",
    # starlette / pydantic
    "starlette.routing", "starlette.middleware", "starlette.middleware.base",
    "pydantic", "pydantic.deprecated.class_validators", "pydantic_core",
    # torch / torchvision
    "torch", "torchvision", "torchvision.models", "torchvision.models.regnet",
    # PIL
    "PIL.Image", "PIL.JpegImagePlugin", "PIL.PngImagePlugin",
    # rotate_captcha_crack (inference-only subset)
    "rotate_captcha_crack",
    "rotate_captcha_crack.common", "rotate_captcha_crack.const",
    "rotate_captcha_crack.model", "rotate_captcha_crack.model.rotr",
    "rotate_captcha_crack.model.helper",
    "rotate_captcha_crack.utils",
    "rotate_captcha_crack.dataset.midware.imgproc",
    "rotate_captcha_crack.dataset.midware.normalizer",
    "rotate_captcha_crack.dataset.midware.totensor",
    # GUI
    "customtkinter", "tkinter", "tkinter.ttk",
]

# ── CUDA DLLs NOT needed for CNN inference ────────────────────────────────────
# Removing these saves ~1.2–1.5 GB without affecting GPU inference quality.
#
#  cufft     — Fast Fourier Transform        (audio/signal, not CNN)
#  cusolver  — Dense linear algebra solvers  (training/eigenvalues)
#  cusparse  — Sparse matrix operations      (NLP, not dense CNN)
#  nvrtc     — NVIDIA Runtime Compilation    (torch.jit / dynamic kernels)
#  caffe2    — Legacy Caffe2 runtime         (not used)
#  nvfuser   — Kernel fusion compiler        (training optimiser)
#  mkl_avx*  — Intel AVX math variants       (redundant with base mkl)

EXCLUDE_DLL_PATTERNS = [
    # FFT / sparse / dense-solver — not used in CNN inference
    # (verified safe: torch_cuda.dll does NOT import these directly)
    "cufft64*",
    "cufftw64*",
    "cusolver64*",
    "cusolverMg64*",
    "cusparse64*",

    # JIT / runtime compilation — we don't use torch.compile on Windows
    "nvrtc64*",
    "nvrtc-builtins64*",
    "nvJitLink*",
    "caffe2_nvrtc*",
    "nvfuser_codegen*",

    # cuDNN precompiled engine plans — cuDNN falls back to runtime
    # compilation via cudnn_engines_runtime_compiled64 (kept, 19 MB).
    # Safe: no other DLL imports from this one.
    "cudnn_engines_precompiled64*",

    # Standalone NVIDIA tools — not imported by any torch DLL
    "nvjpeg64*",        # NVIDIA JPEG encoder (we use Pillow)
    "cupti64*",         # CUDA Profiling Tools Interface
    "nvperf_host*",     # Nsight profiler host
    "nvperf_target*",   # Nsight profiler target

    # Per-arch CUDA kernels blob (kept torch_cuda.dll itself)
    "torch_cuda_cu*",

    # Redundant Intel MKL variants (base mkl + libiomp5md kept)
    "mkl_avx*",
    "mkl_def*",
    "mkl_mc*",
    "mkl_sequential*",
    "libmklml*",
]

def _excluded(filename: str) -> bool:
    name = os.path.basename(filename).lower()
    return any(fnmatch.fnmatch(name, p.lower()) for p in EXCLUDE_DLL_PATTERNS)

# ── Analysis ──────────────────────────────────────────────────────────────────

a = Analysis(
    ["gui_app.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Training-only Python packages
        "matplotlib", "tqdm", "IPython", "jupyter", "notebook", "pytest",
        # aiohttp — replaced by FastAPI/uvicorn, still installed in venv
        "aiohttp", "aiosignal", "frozenlist", "multidict", "yarl",
        # Training modules inside rotate_captcha_crack
        "rotate_captcha_crack.criterion",
        "rotate_captcha_crack.loss",
        "rotate_captcha_crack.lr",
        "rotate_captcha_crack.trainer",
        "rotate_captcha_crack.visualizer",
        "rotate_captcha_crack.dataset",
        # Unused torch subsystems
        "torch.distributions",
        "torch.testing",
        "torch.onnx",
        "torch.fx",
        "torch.ao",
        "torch.backends.xnnpack",
        "torch.optim",
        "torch.profiler",
        "torch.export",
        "torch._dynamo",
        "torch._inductor",
        # Unused stdlib
        "unittest", "pdb", "doctest", "difflib",
        "xml", "xmlrpc",
        "tkinter.test",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ── Strip unused CUDA/MKL binaries ────────────────────────────────────────────
before = len(a.binaries)
a.binaries = [(name, path, kind)
              for name, path, kind in a.binaries
              if not _excluded(name)]
after  = len(a.binaries)
print(f"\n[spec] Removed {before - after} unused binaries "
      f"({before} → {after})\n")

# ── Package ───────────────────────────────────────────────────────────────────

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="rotate-captcha-crack",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=["torch_cuda.dll", "cublas64_12.dll", "cudnn64_9.dll"],
    name="rotate-captcha-crack",
)
