@echo off
echo ==========================================
echo Lumina Studio - CPU Version Build Script
echo ==========================================
echo.
echo This script builds a CPU-only EXE
echo Works on: Any Windows computer (Intel/AMD/NVIDIA)
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)

echo Step 1: Installing CPU version of PyTorch...
echo Uninstalling CUDA version if present...
pip uninstall torch torchvision -y 2>nul

echo Installing CPU version...
pip install torch torchvision
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch CPU
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
echo Step 4: Building CPU-only executable...
echo This may take 10-30 minutes depending on your system...
echo.

python -m PyInstaller LuminaStudio_CPU.spec --clean --noconfirm

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
echo dist\LuminaStudio_CPU.exe
echo.
echo This EXE works on ANY Windows computer:
echo - Intel integrated graphics
echo - AMD graphics
echo - NVIDIA graphics
echo - No dedicated graphics
echo.
echo Note: Processing will be slower than GPU version
echo.
pause
