import sys
import traceback

try:
    print("Step 1: Importing basic modules...")
    import os
    print("✓ os imported")
    
    print("Step 2: Setting up environment...")
    _PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    _GRADIO_TEMP = os.path.join(_PROJECT_ROOT, "output", ".gradio_cache")
    os.makedirs(_GRADIO_TEMP, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = _GRADIO_TEMP
    print(f"✓ Environment set: {_GRADIO_TEMP}")
    
    print("Step 3: Importing gradio...")
    import gradio as gr
    print(f"✓ Gradio imported, version: {gr.__version__}")
    
    print("Step 4: Importing UI modules...")
    from ui.layout_new import create_app
    from ui.styles import CUSTOM_CSS
    print("✓ UI modules imported")
    
    print("Step 5: Creating app...")
    app = create_app()
    print("✓ App created")
    
    print("Step 6: Launching server...")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        prevent_thread_lock=False,
        inbrowser=True
    )
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()
    input("\nPress Enter to exit...")
