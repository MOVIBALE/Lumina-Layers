"""
Lumina Studio Launcher with GPU Auto-Detection
This script launches Lumina Studio with automatic CPU/GPU mode selection
"""

import os
import sys
import subprocess

def check_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        # Method 1: Try to import torch and check CUDA
        import torch
        if torch.cuda.is_available():
            print(f"[OK] NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
            return True
    except:
        pass
    
    try:
        # Method 2: Check for nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] NVIDIA driver detected via nvidia-smi")
            return True
    except:
        pass
    
    print("[INFO] No NVIDIA GPU detected, will use CPU mode")
    return False

def main():
    """Main entry point."""
    print("=" * 50)
    print("Lumina Studio - GPU Auto-Detection Launcher")
    print("=" * 50)
    print()
    
    # Check GPU availability
    has_gpu = check_gpu()
    
    if not has_gpu:
        # Force CPU mode
        os.environ['LUMINA_FORCE_CPU'] = '1'
        print("[INFO] Starting in CPU mode...")
    else:
        print("[INFO] Starting with GPU acceleration...")
    
    print()
    
    # Import and run main application
    try:
        import main
    except Exception as e:
        print(f"[ERROR] Failed to start application: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
