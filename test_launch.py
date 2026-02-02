import sys
sys.path.insert(0, '.')

print('Testing imports...')

try:
    import gradio as gr
    print('✓ Gradio imported')
except Exception as e:
    print('✗ Gradio failed:', e)
    sys.exit(1)

try:
    from ui.layout_new import create_app
    print('✓ Layout module imported')
except Exception as e:
    print('✗ Layout module failed:', e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('All imports successful!')
print('Creating app...')
app = create_app()
print('App created successfully!')
print('Starting server on http://127.0.0.1:7860')
app.launch(server_name='127.0.0.1', server_port=7860, show_error=True)
