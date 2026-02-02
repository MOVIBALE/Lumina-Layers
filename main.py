"""
Lumina Studio v1.4.2
Multi-Material 3D Print Color System

Main Entry Point
"""

import os

# Set Gradio temp directory to project directory
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_GRADIO_TEMP = os.path.join(_PROJECT_ROOT, "output", ".gradio_cache")
os.makedirs(_GRADIO_TEMP, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = _GRADIO_TEMP
import sys
import time
import threading
import webbrowser
import socket
import gradio as gr     # type:ignore
from ui.layout_new import create_app
from ui.styles import CUSTOM_CSS

HAS_DISPLAY = False  # Disable system tray to avoid conflicts
LuminaTray = None  # type: ignore
try:
    from core.tray import LuminaTray
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False
        
def find_available_port(start_port=7860, max_attempts=1000):
    import socket
    for i in range(max_attempts):
        port = start_port + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No available port found after {max_attempts} attempts")

def start_browser(port):
    """Launch the default web browser after a short delay."""
    time.sleep(2)
    webbrowser.open(f"http://127.0.0.1:{port}")

if __name__ == "__main__":

    # 1. Initialize System Tray
    tray = None
    PORT = 7860 # Default fallback
    try:
        PORT = find_available_port(7860)
        if LuminaTray is not None:
            tray = LuminaTray(port=PORT)
    except Exception as e:
        print(f"[WARN] Failed to initialize tray: {e}")

    # 2. Start Browser Thread
    threading.Thread(target=start_browser, args=(PORT,), daemon=True).start()

    # 3. Launch Gradio App
    print(f"[INFO] Lumina Studio is running on http://127.0.0.1:{PORT}")
    app = create_app()

    try:
        app.launch(
            inbrowser=False,
            server_name="0.0.0.0",
            server_port=PORT,
            show_error=True,
            prevent_thread_lock=True,
            favicon_path="icon.ico" if os.path.exists("icon.ico") else None,
            css=CUSTOM_CSS,
            theme=gr.themes.Soft()
        )
    except Exception as e:
        raise
    except BaseException as e:
        raise

    # 4. Start System Tray or Keep Alive
    if tray:
        try:
            print("[INFO] Starting System Tray...")
            tray.run()
        except Exception as e:
            print(f"[WARN] System tray crashed: {e}")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    else:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    print("[INFO] Stopping...")
    os._exit(0)
