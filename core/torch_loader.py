"""
Lumina Studio - PyTorch Loader with Auto GPU Detection
This module handles PyTorch import with automatic GPU/CPU fallback
"""

import sys
import os

# Set environment variable to suppress PyTorch warnings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Try to import torch with comprehensive error handling
TORCH_AVAILABLE = False
CUDA_AVAILABLE = False
torch = None

# First, check if we should force CPU mode
FORCE_CPU = os.environ.get('LUMINA_FORCE_CPU', '0') == '1'

try:
    import torch
    TORCH_AVAILABLE = True
    
    if not FORCE_CPU:
        # Check CUDA availability
        try:
            CUDA_AVAILABLE = torch.cuda.is_available()
            if CUDA_AVAILABLE:
                # Test actual CUDA functionality
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.synchronize()
                print(f"[OK] CUDA is available: {torch.cuda.get_device_name(0)}")
            else:
                print("[INFO] CUDA not available, using CPU mode")
        except Exception as e:
            print(f"[WARN] CUDA check failed: {e}")
            CUDA_AVAILABLE = False
    else:
        print("[INFO] CPU mode forced by environment variable")
        CUDA_AVAILABLE = False
        
except ImportError as e:
    print(f"[WARN] PyTorch import failed: {e}")
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
except Exception as e:
    print(f"[ERROR] Unexpected error importing PyTorch: {e}")
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# Export the variables
__all__ = ['torch', 'TORCH_AVAILABLE', 'CUDA_AVAILABLE', 'FORCE_CPU']
