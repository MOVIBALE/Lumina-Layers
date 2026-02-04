@echo off
chcp 65001 >nul
title Lumina Studio Launcher
cls

echo ==========================================
echo    Lumina Studio
echo    Multi-Material 3D Print Color System
echo ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Upgrade pip first
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Auto-detect GPU and install appropriate PyTorch version
echo [INFO] Auto-detecting GPU and installing optimal PyTorch version...

REM Check for NVIDIA GPU via nvidia-smi
python -c "import subprocess; result = subprocess.run(['nvidia-smi'], capture_output=True); exit(0 if result.returncode == 0 else 1)" 2>nul
if errorlevel 1 (
    echo [INFO] No NVIDIA GPU detected. Installing CPU version...
    set HAS_GPU=0
) else (
    echo [OK] NVIDIA GPU detected!
    set HAS_GPU=1
)

if %HAS_GPU% == 1 (
    REM Check if PyTorch CUDA is already installed
    python -c "import torch; exit(0 if 'cu' in torch.__version__ else 1)" 2>nul
    if errorlevel 1 (
        echo [INFO] Installing CUDA version of PyTorch...
        pip uninstall torch torchvision torchaudio -y
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        if errorlevel 1 (
            echo [WARN] Failed to install CUDA version. Installing CPU version instead...
            pip install torch torchvision
        ) else (
            echo [OK] CUDA version installed!
        )
    ) else (
        echo [OK] CUDA version already installed.
    )
) else (
    REM No GPU - ensure CPU version is installed
    python -c "import torch" 2>nul
    if errorlevel 1 (
        echo [INFO] Installing CPU version of PyTorch...
        pip install torch torchvision
        echo [OK] CPU version installed!
    ) else (
        REM Check if it's already CPU version
        python -c "import torch; exit(0 if 'cu' not in torch.__version__ else 1)" 2>nul
        if errorlevel 1 (
            echo [INFO] Converting from CUDA to CPU version for better compatibility...
            pip uninstall torch torchvision torchaudio -y
            pip install torch torchvision
            echo [OK] CPU version installed!
        ) else (
            echo [OK] CPU version already installed.
        )
    )
)

echo.

REM Install other dependencies
echo [INFO] Installing other dependencies...
pip install -r requirements.txt
echo.

echo [OK] All dependencies ready
echo.

REM Check final GPU status
echo [INFO] Final GPU status:
python -c "import torch; print('[INFO] PyTorch:', torch.__version__); print('[INFO] CUDA available:', torch.cuda.is_available()); print('[INFO] GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Using CPU')" 2>&1
echo.

REM Start the application
echo [INFO] Starting Lumina Studio...
echo [INFO] The application will open in your web browser
echo [INFO] Press Ctrl+C to stop the server
echo.

python main.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application crashed!
    pause
)
