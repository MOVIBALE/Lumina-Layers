"""
Lumina Studio - Simple Launcher
Simple launcher script without system tray
"""

import os
import sys

# 设置 Gradio 临时目录到项目目录
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_GRADIO_TEMP = os.path.join(_PROJECT_ROOT, "output", ".gradio_cache")
os.makedirs(_GRADIO_TEMP, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = _GRADIO_TEMP

import gradio as gr
from ui.layout_new import create_app
from ui.styles import CUSTOM_CSS

if __name__ == "__main__":
    PORT = 7860
    
    print(f"✨ Lumina Studio is starting on http://127.0.0.1:{PORT}")
    print("🌐 Open your browser and navigate to the URL above")
    print("⚠️  Press Ctrl+C to stop the server")
    
    app = create_app()
    
    app.launch(
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=PORT,
        show_error=True,
        prevent_thread_lock=False,
        favicon_path="icon.ico" if os.path.exists("icon.ico") else None,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
        share=False
    )
