# gui.spec — PyInstaller spec for rotate-captcha-crack GUI
# Run: pyinstaller gui.spec

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
        # Unused torch subsystems (Python-level only, DLLs untouched)
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
    upx_exclude=[
        # Large CUDA DLLs — already compressed, UPX won't help much
        # and risks corruption on some machines
        "torch_cuda.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudnn64_9.dll",
        "cudnn_ops64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
        "cufft64_11.dll",
        "cusparse64_12.dll",
        "cusolver64_11.dll",
    ],
    name="rotate-captcha-crack",
)
