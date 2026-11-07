@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: ---------------------------------------------------------------
:: Portable setup: venv, uv cache, Hugging Face models and ffmpeg
:: all live inside this project folder.
:: ---------------------------------------------------------------
set "UV_CACHE_DIR=%~dp0uv-cache"
set "HF_HOME=%~dp0models_cache"
set "PATH=%~dp0tools\ffmpeg\bin;%USERPROFILE%\.local\bin;%PATH%"

:: Set the CUDA_PATH to your 12.8 installation directory
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

:: Prepend CUDA 12.8 bin and libnvvp folders to PATH so they take precedence
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

:: 1. uv (the uv installer puts it in %USERPROFILE%\.local\bin)
where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv was not found. Install it with:
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    pause
    exit /b 1
)

:: 2. Python packages. uv skips anything already installed and reuses the cache.
uv sync
if errorlevel 1 (
    echo [ERROR] uv sync failed. Run this file again: the download cache is reused.
    pause
    exit /b 1
)

:: 3. ffmpeg and Hugging Face models: downloads what is missing, skips what is present.
".venv\Scripts\python.exe" tools\preflight_check.py
if errorlevel 1 (
    echo.
    echo [WARN] Some items are still missing. See MISSING_MANUAL.txt in this folder.
    pause
    exit /b 1
)

:: Count how many GPUs are present
set GPU_COUNT=0
for /f %%i in ('nvidia-smi -L ^| find /c "GPU"') do set GPU_COUNT=%%i

:: If more than 1 GPU, ask the user
if %GPU_COUNT% GTR 1 (
    echo =================================================
    echo Multiple GPUs detected:
    nvidia-smi --query-gpu=index,name --format=csv,noheader
    echo =================================================
    set /p GPU_ID="Enter the ID of the GPU you want to use (e.g. 0 or 1): "

    :: Set the environment variable for this session
    set CUDA_VISIBLE_DEVICES=!GPU_ID!
    echo Script will now run using ONLY GPU !GPU_ID!.
) else (
    echo Only %GPU_COUNT% GPU detected. Running on default.
)

:: 4. Run the app with the project's own interpreter
".venv\Scripts\python.exe" depthcrafter_gui_seg.py
