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

REM Set environment variable before cloning
set GIT_CLONE_PROTECTION_ACTIVE=false

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

REM Check for NVIDIA GPU and CUDA support (optional)
echo Checking for CUDA support... >> install_log.txt
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())" >> install_log.txt 2>&1
if %errorlevel% neq 0 (
    echo Failed to check CUDA support. Installing CPU version of PyTorch. >> install_log.txt
    echo Failed to check CUDA support. Installing CPU version of PyTorch.
    python -m pip install -U torch torchvision torchaudio
) else (
    REM Install specific PyTorch with CUDA 12.8
    echo Installing PyTorch with CUDA 12.8... >> install_log.txt
    python -m pip install -U torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    if %errorlevel% neq 0 (
        echo Failed to install PyTorch packages. >> install_log.txt
        echo Failed to install PyTorch packages.
        pause
        exit /b %errorlevel%
    )
)

REM Install additional packages
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