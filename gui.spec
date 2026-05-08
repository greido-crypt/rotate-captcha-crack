# gui.spec — PyInstaller spec for rotate-captcha-crack GUI
# Run: pyinstaller gui.spec

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_all

block_cipher = None

# ── Collect data files ────────────────────────────────────────────────────────

datas = []

# customtkinter themes and assets (required)
datas += collect_data_files("customtkinter")

# rotate_captcha_crack package source
datas += [("src/rotate_captcha_crack", "rotate_captcha_crack")]

# Trained model weights
datas += [("models", "models")]

# ── Hidden imports ────────────────────────────────────────────────────────────

hiddenimports = [
    # FastAPI / uvicorn internals
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "starlette.routing",
    "starlette.middleware",
    "starlette.middleware.base",
    # Pydantic
    "pydantic",
    "pydantic.deprecated.class_validators",
    "pydantic_core",
    # PyTorch
    "torch",
    "torchvision",
    "torchvision.models",
    "torchvision.models.regnet",
    # PIL
    "PIL.Image",
    "PIL.JpegImagePlugin",
    "PIL.PngImagePlugin",
    # rotate_captcha_crack
    "rotate_captcha_crack",
    "rotate_captcha_crack.common",
    "rotate_captcha_crack.const",
    "rotate_captcha_crack.model",
    "rotate_captcha_crack.model.rotr",
    "rotate_captcha_crack.model.rotr_quant",
    "rotate_captcha_crack.model.helper",
    "rotate_captcha_crack.utils",
    "rotate_captcha_crack.dataset",
    "rotate_captcha_crack.dataset.midware",
    "rotate_captcha_crack.dataset.midware.imgproc",
    "rotate_captcha_crack.dataset.midware.normalizer",
    "rotate_captcha_crack.dataset.midware.totensor",
    # customtkinter
    "customtkinter",
    "tkinter",
    "tkinter.ttk",
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
        # Exclude training-only heavy deps
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

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
    upx=True,                   # compress binaries (install upx for smaller size)
    console=False,              # no console window — GUI only
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,                  # replace with "icon.ico" if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="rotate-captcha-crack",   # output folder name in dist/
)
