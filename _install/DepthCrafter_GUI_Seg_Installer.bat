@echo off
REM DepthCrafter_GUI_Seg Installer Script
REM Version: 1.0
REM Logs output to install_log.txt for debugging

REM Initialize log file
echo [%date% %time%] Starting installation > install_log.txt

REM Check if git is installed
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo git is not installed or not available in PATH. >> install_log.txt
    echo git is not installed or not available in PATH.
    echo Please install git from https://git-scm.com/ and ensure it is in your PATH.
    pause
    exit /b 1
)

REM If the DepthCrafter_GUI_Seg directory exists, prompt user
if exist "DepthCrafter_GUI_Seg" (
    echo The DepthCrafter_GUI_Seg directory already exists. >> install_log.txt
    echo The DepthCrafter_GUI_Seg directory already exists.
    set /p user_choice="Do you want to remove it and continue? (Y/N): "
    if /i "%user_choice%"=="Y" (
        rmdir /s /q DepthCrafter_GUI_Seg
        if %errorlevel% neq 0 (
            echo Failed to remove existing DepthCrafter_GUI_Seg directory. >> install_log.txt
            echo Failed to remove existing DepthCrafter_GUI_Seg directory.
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


REM Clone the DepthCrafter_GUI_Seg repository
echo Cloning repository... >> install_log.txt
git clone https://github.com/Billynom8/DepthCrafter_GUI_Seg.git
if %errorlevel% neq 0 (
    echo Failed to clone the DepthCrafter_GUI_Seg repository. >> install_log.txt
    echo Failed to clone the DepthCrafter_GUI_Seg repository.
    pause
    exit /b %errorlevel%
)

REM Verify directory exists before changing
if not exist "DepthCrafter_GUI_Seg" (
    echo Cloned directory DepthCrafter_GUI_Seg not found. >> install_log.txt
    echo Cloned directory DepthCrafter_GUI_Seg not found.
    pause
    exit /b 1
)

cd DepthCrafter_GUI_Seg
if %errorlevel% neq 0 (
    echo Failed to change directory into DepthCrafter_GUI_Seg. >> install_log.txt
    echo Failed to change directory into DepthCrafter_GUI_Seg.
    pause
    exit /b %errorlevel%
)

REM Check for CUDA Toolkit and version 12.8 or 12.9
echo Checking for CUDA 12.8 or 12.9 Toolkit... >> install_log.txt
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo NVIDIA CUDA Toolkit [nvcc] not found in PATH. >> install_log.txt
    echo NVIDIA CUDA Toolkit [nvcc] not found in PATH.
    echo Please install CUDA Toolkit 12.8 or 12.9 from https://developer.nvidia.com/cuda-toolkit and ensure it is in your PATH.
    pause
    exit /b 1
)

REM Log raw nvcc --version output for debugging
echo Raw nvcc --version output: >> install_log.txt
nvcc --version >> install_log.txt

REM Parse the output of nvcc --version to get the version number
set "CUDA_VERSION="
for /f "tokens=5 delims= " %%v in ('nvcc --version ^| findstr "release"') do (
    for /f "tokens=1 delims=," %%c in ("%%v") do set CUDA_VERSION=%%c
)

echo CUDA version detected: %CUDA_VERSION%

REM Check if CUDA_VERSION was successfully set
if not defined CUDA_VERSION (
    echo Failed to determine CUDA version. Check install_log.txt for nvcc output. >> install_log.txt
    echo Failed to determine CUDA version.
    echo Please ensure CUDA Toolkit 12.8 or 12.9 is correctly installed and nvcc is functioning.
    pause
    exit /b 1
)

echo Found CUDA version: %CUDA_VERSION% >> install_log.txt
echo Found CUDA version: %CUDA_VERSION%

if not "%CUDA_VERSION%"=="12.8" if not "%CUDA_VERSION%"=="12.9" (
    echo Incorrect CUDA version detected. This script requires version 12.8 or 12.9, but found %CUDA_VERSION%. >> install_log.txt
    echo Incorrect CUDA version detected. This script requires version 12.8 or 12.9, but found %CUDA_VERSION%.
    pause
    exit /b 1
)
echo CUDA %CUDA_VERSION% Toolkit found. >> install_log.txt

REM Check if Python is installed and verify version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not added to PATH. >> install_log.txt
    echo Python is not installed or not added to PATH.
    echo Please install Python from https://www.python.org/.
    pause
    exit /b 1
)

REM Check Python version (example: require 3.8 or higher)
for /f "tokens=2 delims= " %%v in ('python --version') do set py_version=%%v
echo Python version: %py_version% >> install_log.txt
for /f "tokens=1,2 delims=." %%a in ("%py_version%") do (
    set major=%%a
    set minor=%%b
)
if %major% lss 3 (
    echo Python version %py_version% is not supported. Requires Python 3.8 or higher. >> install_log.txt
    echo Python version %py_version% is not supported. Requires Python 3.8 or higher.
    pause
    exit /b 1
)
if %major% equ 3 if %minor% lss 8 (
    echo Python version %py_version% is not supported. Requires Python 3.8 or higher. >> install_log.txt
    echo Python version %py_version% is not supported. Requires Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check for requirements.txt
if not exist "requirements.txt" (
    echo requirements.txt not found in DepthCrafter_GUI_Seg directory. >> install_log.txt
    echo requirements.txt not found in DepthCrafter_GUI_Seg directory.
    pause
    exit /b 1
)

REM Create a virtual environment
echo Creating virtual environment... >> install_log.txt
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment. >> install_log.txt
    echo Failed to create virtual environment.
    pause
    exit /b %errorlevel%
)

REM Activate the virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment activation script not found. >> install_log.txt
    echo Virtual environment activation script not found.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment. >> install_log.txt
    echo Failed to activate virtual environment.
    pause
    exit /b %errorlevel%
)

REM Upgrade pip
echo Upgrading pip... >> install_log.txt
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Failed to upgrade pip. >> install_log.txt
    echo Failed to upgrade pip.
    pause
    exit /b %errorlevel%
)

REM Install dependencies from requirements.txt
echo Installing dependencies from requirements.txt... >> install_log.txt
python -m pip install --upgrade -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies from requirements.txt. >> install_log.txt
    echo Failed to install dependencies from requirements.txt.
    pause
    exit /b %errorlevel%
)

REM Install xformers
echo Installing xformers... >> install_log.txt
python -m pip install -U xformers --index-url https://download.pytorch.org/whl/cu128
if %errorlevel% neq 0 (
    echo Failed to install xformers. >> install_log.txt
    echo Failed to install xformers.
    pause
    exit /b %errorlevel%
)
echo All dependencies installed successfully. >> install_log.txt
echo All dependencies installed successfully.
echo Installation log saved to install_log.txt
pause