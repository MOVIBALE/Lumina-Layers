@echo off
echo === Python Diagnostic ===
echo.
echo Checking Python installation...
where python 2>nul
if %errorlevel% neq 0 (
    echo Python not found in PATH
    goto :end
)

echo.
echo Python version:
python --version 2>&1

echo.
echo Python path:
python -c "import sys; print(sys.executable)" 2>&1

echo.
echo Checking pip...
python -m pip --version 2>&1

echo.
echo Checking key modules...
python -c "import numpy; print('NumPy:', numpy.__version__)" 2>&1
python -c "import cv2; print('OpenCV:', cv2.__version__)" 2>&1
python -c "import gradio; print('Gradio:', gradio.__version__)" 2>&1

echo.
echo === End Diagnostic ===
pause
