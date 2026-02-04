@echo off
echo ==========================================
echo Lumina Studio - GPU Version Build Script
echo ==========================================
echo.
echo This script builds an EXE with CUDA support
echo Requires: NVIDIA GPU + CUDA-capable PyTorch
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)

echo Step 1: Installing CUDA version of PyTorch...
echo Uninstalling CPU version if present...
pip uninstall torch torchvision torchaudio -y 2>nul

echo Installing CUDA 12.1 version...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch CUDA
    pause
    exit /b 1
)

echo.
echo Step 2: Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo Step 3: Installing PyInstaller...
pip install pyinstaller
if errorlevel 1 (
    echo ERROR: Failed to install PyInstaller
    pause
    exit /b 1
)

echo.
echo Step 4: Building GPU-accelerated executable...
echo This may take 10-30 minutes depending on your system...
echo.

python -m PyInstaller LuminaStudio_GPU.spec --clean --noconfirm

if errorlevel 1 (
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Build completed successfully!
echo ==========================================
echo.
echo The executable is located at:
echo dist\LuminaStudio_GPU.exe
echo.
echo IMPORTANT: This EXE requires NVIDIA GPU + drivers
echo For computers without NVIDIA GPU, use build_cpu.bat instead
echo.
pause
