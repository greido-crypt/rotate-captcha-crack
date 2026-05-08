@echo off
setlocal EnableDelayedExpansion
title rotate-captcha-crack — Build

echo.
echo  ============================================================
echo   rotate-captcha-crack  ^|  EXE Builder
echo  ============================================================
echo.

:: ── Activate venv ────────────────────────────────────────────────────────────
if not exist ".venv\Scripts\activate.bat" (
    echo  [ERROR] .venv not found. Run first:
    echo          uv venv --python 3.11
    echo          uv pip install -e ".[server]"
    pause & exit /b 1
)
call .venv\Scripts\activate.bat
echo  [OK] venv activated

:: ── Install build deps ────────────────────────────────────────────────────────
echo  [..] Installing build dependencies...
uv pip install pyinstaller customtkinter --quiet
if %errorlevel% neq 0 ( echo  [ERROR] pip install failed & pause & exit /b 1 )
echo  [OK] build deps ready

:: ── Clean previous build ─────────────────────────────────────────────────────
echo  [..] Cleaning previous build...
if exist "dist\rotate-captcha-crack" rmdir /s /q "dist\rotate-captcha-crack"
if exist "build\rotate-captcha-crack" rmdir /s /q "build\rotate-captcha-crack"
echo  [OK] cleaned

:: ── Build ────────────────────────────────────────────────────────────────────
echo.
echo  [..] Building EXE (this takes 3-10 minutes)...
echo.
pyinstaller gui.spec --noconfirm
if %errorlevel% neq 0 ( echo. & echo  [ERROR] Build failed! & pause & exit /b 1 )

:: ── Check output ─────────────────────────────────────────────────────────────
if not exist "dist\rotate-captcha-crack\rotate-captcha-crack.exe" (
    echo  [ERROR] EXE not found after build
    pause & exit /b 1
)

:: ── Show size ─────────────────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   BUILD SUCCESSFUL
echo  ============================================================
echo.
echo   Output: dist\rotate-captcha-crack\
echo.

for /f "tokens=3" %%a in ('dir "dist\rotate-captcha-crack" /s /-c ^| find "File(s)"') do (
    set SIZE=%%a
)
echo   Folder size: !SIZE! bytes
echo.
echo   To run:
echo     dist\rotate-captcha-crack\rotate-captcha-crack.exe
echo.
echo   To share: zip the entire dist\rotate-captcha-crack\ folder
echo.

pause
