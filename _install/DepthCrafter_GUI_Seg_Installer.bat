@echo off
REM DepthCrafter_GUI_Seg Installer Script (UV Edition)
REM Version: 2.0
REM Logs output to install_log.txt for debugging

REM Initialize log file
echo [%date% %time%] Starting installation > install_log.txt

REM -----------------------------------------------------------------
REM 1. Check for Git
REM -----------------------------------------------------------------
echo Checking for Git...
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo git is not installed or not available in PATH. >> install_log.txt
    echo Error: Git is not found. Please install it from https://git-scm.com/
    pause
    exit /b 1
)

REM -----------------------------------------------------------------
REM 2. Check/Install uv
REM -----------------------------------------------------------------
echo Checking for uv...
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo uv not found. Installing uv...
    echo uv not found. Installing uv... >> install_log.txt
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    
    REM Refresh environment variables for this session so we can use uv immediately
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)

REM Verify uv installation
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Failed to install or locate uv. >> install_log.txt
    echo Error: Failed to install 'uv'. Please install manually: https://docs.astral.sh/uv/
    pause
    exit /b 1
)
echo uv is ready. >> install_log.txt

REM -----------------------------------------------------------------
REM 3. Handle Existing Directory
REM -----------------------------------------------------------------
if exist "DepthCrafter_GUI_Seg" (
    echo The DepthCrafter_GUI_Seg directory already exists. >> install_log.txt
    echo The DepthCrafter_GUI_Seg directory already exists.
    set /p user_choice="Do you want to remove it and continue? (Y/N): "
    if /i "%user_choice%"=="Y" (
        rmdir /s /q DepthCrafter_GUI_Seg
        if %errorlevel% neq 0 (
            echo Failed to remove existing directory. >> install_log.txt
            echo Error: Failed to remove existing directory.
            pause
            exit /b %errorlevel%
        )
    ) else (
        echo Aborting installation. >> install_log.txt
        echo Aborting installation.
        pause
        exit /b 0
    )
)

REM -----------------------------------------------------------------
REM 4. Clone Repository
REM -----------------------------------------------------------------
echo Cloning repository... >> install_log.txt
echo Cloning repository...
git clone https://github.com/Billynom8/DepthCrafter_GUI_Seg.git
if %errorlevel% neq 0 (
    echo Failed to clone repository. >> install_log.txt
    pause
    exit /b %errorlevel%
)

cd DepthCrafter_GUI_Seg

REM -----------------------------------------------------------------
REM 5. CUDA Check (Kept for hardware verification)
REM -----------------------------------------------------------------
echo Checking for CUDA Toolkit... >> install_log.txt
nvcc --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: NVIDIA CUDA Toolkit [nvcc] not found. 
    echo UV will install PyTorch with CUDA support, but compiling custom CUDA kernels might fail without the Toolkit.
    echo Proceeding... >> install_log.txt
) else (
    nvcc --version >> install_log.txt
    echo CUDA Toolkit detected.
)

REM -----------------------------------------------------------------
REM 6. Setup Environment with uv
REM -----------------------------------------------------------------
echo Setting up virtual environment (.venv)... >> install_log.txt

:: Force the project to use Python 3.12 to avoid 3.13 incompatibilities
echo Pinning Python version to 3.12...
uv python pin 3.12

REM Option A: If pyproject.toml exists
if exist "pyproject.toml" (
    echo Found pyproject.toml. Syncing dependencies with uv...
    uv sync
    if %errorlevel% neq 0 (
        echo UV Sync failed. >> install_log.txt
        echo Error: UV Sync failed. Check log.
        pause
        exit /b 1
    )
) else (
    REM Option B: Legacy requirements.txt Method
    echo pyproject.toml not found. Falling back to requirements.txt... >> install_log.txt
    
    REM Create .venv explicitly
    uv venv .venv
    
    REM Install from requirements
    if exist "requirements.txt" (
        echo Installing from requirements.txt...
        uv pip install -r requirements.txt
    ) else (
        echo Error: Neither pyproject.toml nor requirements.txt found! >> install_log.txt
        echo Error: No dependency file found.
        pause
        exit /b 1
    )
)

echo.
echo =========================================================
echo Installation Complete!
echo =========================================================
echo.
echo To run the application:
echo 1. cd DepthCrafter_GUI_Seg
echo 2. uv run depthcrafter_gui_seg.py
echo.
echo (Or double-click the run script if you have one)
echo.
pause