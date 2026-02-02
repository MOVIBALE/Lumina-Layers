"""
Lumina Studio - PyTorch CUDA Accelerated Image Processing Core
GPU-accelerated image processing module using PyTorch CUDA
"""

import numpy as np
import cv2
from PIL import Image
from scipy.spatial import KDTree
import torch

from config import PrinterConfig


class LuminaImageProcessorCUDA:
    """
    CUDA-accelerated Image processor class using PyTorch
    Handles LUT loading, image processing, and color matching with GPU acceleration
    """
    
    def __init__(self, lut_path, color_mode):
        """
        Initialize image processor with CUDA support
        
        Args:
            lut_path: LUT file path (.npy)
            color_mode: Color mode string (CMYW/RYBW)
        """
        self.color_mode = color_mode
        self.lut_rgb = None
        self.ref_stacks = None
        self.kdtree = None
        
        # Check CUDA availability
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        if self.use_cuda:
            print(f"[CUDA] Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"[CUDA] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("[CUDA] GPU not available, using CPU")
        
        # Load and validate LUT
        self._load_lut(lut_path)
    
    def _load_lut(self, lut_path):
        """Load and validate LUT file"""
        try:
            lut_grid = np.load(lut_path)
            measured_colors = lut_grid.reshape(-1, 3)
        except Exception as e:
            raise ValueError(f"LUT file corrupted: {e}")
        
        valid_rgb, valid_stacks = [], []
        base_blue = np.array([30, 100, 200])
        dropped = 0
        
        # Filter outliers
        for i in range(1024):
            digits = []
            temp = i
            for _ in range(5):
                digits.append(temp % 4)
                temp //= 4
            stack = digits[::-1]
            
            real_rgb = measured_colors[i]
            dist = np.linalg.norm(real_rgb - base_blue)
            
            # Filter out anomalies: close to blue but doesn't contain blue
            if dist < 60 and 3 not in stack:
                dropped += 1
                continue
            
            valid_rgb.append(real_rgb)
            valid_stacks.append(stack)
        
        self.lut_rgb = np.array(valid_rgb)
        self.ref_stacks = np.array(valid_stacks)
        self.kdtree = KDTree(self.lut_rgb)
        
        # Move LUT to GPU if CUDA available
        if self.use_cuda:
            try:
                self.lut_rgb_gpu = torch.from_numpy(self.lut_rgb).float().to(self.device)
                self.ref_stacks_gpu = torch.from_numpy(self.ref_stacks).long().to(self.device)
                print(f"[CUDA] LUT loaded on GPU (filtered {dropped} outliers)")
            except Exception as e:
                print(f"[CUDA] Failed to load LUT on GPU: {e}")
                self.use_cuda = False
        else:
            print(f"[CPU] LUT loaded (filtered {dropped} outliers)")
    
    def process_image(self, image_path, target_width_mm, modeling_mode,
                     quantize_colors, auto_bg, bg_tol,
                     blur_kernel=0, smooth_sigma=10):
        """
        Main image processing method with CUDA acceleration
        """
        # Normalize modeling mode
        mode_str = str(modeling_mode).lower()
        use_high_fidelity = "high-fidelity" in mode_str or "高保真" in mode_str
        use_pixel = "pixel" in mode_str or "像素" in mode_str
        
        # Determine mode name
        if use_high_fidelity:
            mode_name = "High-Fidelity"
        elif use_pixel:
            mode_name = "Pixel Art"
        else:
            mode_name = "High-Fidelity"
            use_high_fidelity = True
        
        print(f"[IMAGE_PROCESSOR] Mode: {mode_name}")
        print(f"[IMAGE_PROCESSOR] Device: {'CUDA' if self.use_cuda else 'CPU'}")
        print(f"[IMAGE_PROCESSOR] Filter settings: blur_kernel={blur_kernel}, smooth_sigma={smooth_sigma}")
        
        # Load image
        img = Image.open(image_path).convert('RGBA')
        
        # Calculate target resolution
        if use_high_fidelity:
            PIXELS_PER_MM = 10
            target_w = int(target_width_mm * PIXELS_PER_MM)
            pixel_to_mm_scale = 1.0 / PIXELS_PER_MM
            print(f"[IMAGE_PROCESSOR] High-res mode: {PIXELS_PER_MM} px/mm")
        else:
            target_w = int(target_width_mm / PrinterConfig.NOZZLE_WIDTH)
            pixel_to_mm_scale = PrinterConfig.NOZZLE_WIDTH
            print(f"[IMAGE_PROCESSOR] Pixel mode: {1.0/pixel_to_mm_scale:.2f} px/mm")
        
        target_h = int(target_w * img.height / img.width)
        print(f"[IMAGE_PROCESSOR] Target: {target_w}x{target_h}px ({target_w*pixel_to_mm_scale:.1f}x{target_h*pixel_to_mm_scale:.1f}mm)")
        
        img = img.resize((target_w, target_h), Image.Resampling.NEAREST)
        img_arr = np.array(img)
        rgb_arr = img_arr[:, :, :3]
        alpha_arr = img_arr[:, :, 3]
        
        # Identify transparent pixels
        mask_transparent_initial = alpha_arr < 10
        print(f"[IMAGE_PROCESSOR] Found {np.sum(mask_transparent_initial)} transparent pixels (alpha<10)")
        
        # Color processing and matching
        debug_data = None
        if use_high_fidelity:
            matched_rgb, material_matrix, bg_reference, debug_data = self._process_high_fidelity_mode_cuda(
                rgb_arr, target_h, target_w, quantize_colors, blur_kernel, smooth_sigma
            )
        else:
            matched_rgb, material_matrix, bg_reference = self._process_pixel_mode_cuda(
                rgb_arr, target_h, target_w
            )
        
        # Background removal
        mask_transparent = mask_transparent_initial.copy()
        if auto_bg:
            bg_color = bg_reference[0, 0]
            diff = np.sum(np.abs(bg_reference - bg_color), axis=-1)
            mask_transparent = np.logical_or(mask_transparent, diff < bg_tol)
        
        # Apply transparency mask
        material_matrix[mask_transparent] = -1
        mask_solid = ~mask_transparent
        
        result = {
            'matched_rgb': matched_rgb,
            'material_matrix': material_matrix,
            'mask_solid': mask_solid,
            'dimensions': (target_w, target_h),
            'pixel_scale': pixel_to_mm_scale,
            'mode_info': {
                'name': mode_name,
                'use_high_fidelity': use_high_fidelity,
                'use_pixel': use_pixel
            }
        }
        
        if debug_data is not None:
            result['debug_data'] = debug_data
        
        return result
    
    def _process_high_fidelity_mode_cuda(self, rgb_arr, target_h, target_w, quantize_colors,
                                         blur_kernel, smooth_sigma):
        """
        CUDA-accelerated high-fidelity mode image processing using PyTorch
        """
        print(f"[IMAGE_PROCESSOR] Starting {'CUDA' if self.use_cuda else 'CPU'} processing...")
        
        # Step 1: Bilateral filter (edge-preserving smoothing)
        if smooth_sigma > 0:
            print(f"[IMAGE_PROCESSOR] Applying bilateral filter (sigma={smooth_sigma})...")
            rgb_processed = cv2.bilateralFilter(
                rgb_arr.astype(np.uint8), 
                d=9,
                sigmaColor=smooth_sigma, 
                sigmaSpace=smooth_sigma
            )
        else:
            print(f"[IMAGE_PROCESSOR] Bilateral filter disabled (sigma=0)")
            rgb_processed = rgb_arr.astype(np.uint8)
        
        # Step 2: Optional median filter
        if blur_kernel > 0:
            kernel_size = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            print(f"[IMAGE_PROCESSOR] Applying median blur (kernel={kernel_size})...")
            rgb_processed = cv2.medianBlur(rgb_processed, kernel_size)
        
        # Step 3: Sharpening
        sharpen_kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0]
        ])
        print(f"[IMAGE_PROCESSOR] Applying subtle sharpening...")
        rgb_sharpened = cv2.filter2D(rgb_processed, -1, sharpen_kernel)
        rgb_sharpened = np.clip(rgb_sharpened, 0, 255).astype(np.uint8)
        
        # Step 4: K-Means quantization with CUDA if available
        print(f"[IMAGE_PROCESSOR] K-Means quantization to {quantize_colors} colors...")
        h, w = rgb_sharpened.shape[:2]
        pixels = rgb_sharpened.reshape(-1, 3).astype(np.float32)
        
        if self.use_cuda and len(pixels) > 10000:
            # Use PyTorch for large images
            try:
                quantized_image, centers = self._kmeans_torch(pixels, quantize_colors, h, w)
                print(f"[IMAGE_PROCESSOR] PyTorch K-Means complete!")
            except Exception as e:
                print(f"[CUDA] PyTorch K-Means failed: {e}. Falling back to CPU.")
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                flags = cv2.KMEANS_PP_CENTERS
                _, labels, centers = cv2.kmeans(pixels, quantize_colors, None, criteria, 10, flags)
                centers = centers.astype(np.uint8)
                quantized_pixels = centers[labels.flatten()]
                quantized_image = quantized_pixels.reshape(h, w, 3)
        else:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            flags = cv2.KMEANS_PP_CENTERS
            _, labels, centers = cv2.kmeans(pixels, quantize_colors, None, criteria, 10, flags)
            centers = centers.astype(np.uint8)
            quantized_pixels = centers[labels.flatten()]
            quantized_image = quantized_pixels.reshape(h, w, 3)
        
        # Find unique colors
        unique_colors = np.unique(quantized_image.reshape(-1, 3), axis=0)
        print(f"[IMAGE_PROCESSOR] Found {len(unique_colors)} unique colors")
        
        # Match to LUT with CUDA if available
        print(f"[IMAGE_PROCESSOR] Matching colors to LUT...")
        if self.use_cuda and len(unique_colors) > 50:
            try:
                unique_indices = self._match_colors_torch(unique_colors)
                print(f"[IMAGE_PROCESSOR] PyTorch color matching complete!")
            except Exception as e:
                print(f"[CUDA] PyTorch color matching failed: {e}. Falling back to CPU.")
                _, unique_indices = self.kdtree.query(unique_colors.astype(float))
        else:
            _, unique_indices = self.kdtree.query(unique_colors.astype(float))
        
        # Build color mapping
        color_to_stack = {}
        color_to_rgb = {}
        for i, color in enumerate(unique_colors):
            color_key = tuple(color)
            color_to_stack[color_key] = self.ref_stacks[unique_indices[i]]
            color_to_rgb[color_key] = self.lut_rgb[unique_indices[i]]
        
        # Map back to full image
        print(f"[IMAGE_PROCESSOR] Mapping to full image...")
        matched_rgb = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        material_matrix = np.zeros((target_h, target_w, PrinterConfig.COLOR_LAYERS), dtype=int)
        
        for y in range(target_h):
            for x in range(target_w):
                color_key = tuple(quantized_image[y, x])
                matched_rgb[y, x] = color_to_rgb[color_key]
                material_matrix[y, x] = color_to_stack[color_key]
        
        print(f"[IMAGE_PROCESSOR] Color matching complete!")
        
        # Prepare debug data
        debug_data = {
            'quantized_image': quantized_image.copy(),
            'num_colors': len(unique_colors),
            'bilateral_filtered': rgb_processed.copy(),
            'sharpened': rgb_sharpened.copy(),
            'filter_settings': {
                'blur_kernel': blur_kernel,
                'smooth_sigma': smooth_sigma
            }
        }
        
        return matched_rgb, material_matrix, quantized_image, debug_data
    
    def _kmeans_torch(self, pixels, k, h, w):
        """
        PyTorch CUDA-accelerated K-Means clustering
        """
        # Transfer data to GPU
        pixels_torch = torch.from_numpy(pixels).float().to(self.device)
        
        # Initialize centroids randomly
        torch.manual_seed(42)
        indices = torch.randperm(len(pixels))[:k]
        centroids = pixels_torch[indices].clone()
        
        # K-Means iterations
        max_iter = 100
        tol = 0.2
        
        for iteration in range(max_iter):
            # Calculate distances and assign labels
            distances = torch.cdist(pixels_torch, centroids)
            labels = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for i in range(k):
                mask = labels == i
                if mask.any():
                    new_centroids[i] = pixels_torch[mask].mean(dim=0)
                else:
                    new_centroids[i] = centroids[i]
            
            # Check convergence
            shift = torch.norm(new_centroids - centroids)
            centroids = new_centroids
            
            if shift < tol:
                print(f"[CUDA] K-Means converged at iteration {iteration + 1}")
                break
        
        # Transfer results back to CPU
        labels_cpu = labels.cpu().numpy()
        centroids_cpu = centroids.cpu().numpy().astype(np.uint8)
        
        # Build quantized image
        quantized_pixels = centroids_cpu[labels_cpu]
        quantized_image = quantized_pixels.reshape(h, w, 3)
        
        return quantized_image, centroids_cpu
    
    def _match_colors_torch(self, unique_colors):
        """
        PyTorch CUDA-accelerated color matching
        """
        # Transfer data to GPU
        unique_colors_torch = torch.from_numpy(unique_colors.astype(float)).float().to(self.device)
        lut_rgb_torch = self.lut_rgb_gpu
        
        # Calculate all pairwise distances using broadcasting
        # unique_colors: (N, 3), lut_rgb: (M, 3)
        # distances: (N, M)
        distances = torch.cdist(unique_colors_torch, lut_rgb_torch)
        
        # Find nearest neighbors
        indices = torch.argmin(distances, dim=1)
        
        return indices.cpu().numpy()
    
    def _process_pixel_mode_cuda(self, rgb_arr, target_h, target_w):
        """
        CUDA-accelerated pixel art mode image processing
        """
        print(f"[IMAGE_PROCESSOR] Direct pixel-level matching (Pixel Art mode)...")
        
        flat_rgb = rgb_arr.reshape(-1, 3)
        
        if self.use_cuda and len(flat_rgb) > 10000:
            try:
                # Use PyTorch for color matching
                print(f"[IMAGE_PROCESSOR] Using PyTorch CUDA for color matching...")
                flat_rgb_torch = torch.from_numpy(flat_rgb.astype(float)).float().to(self.device)
                lut_rgb_torch = self.lut_rgb_gpu
                
                # Calculate distances in batches to avoid memory issues
                batch_size = 100000
                indices_list = []
                
                for i in range(0, len(flat_rgb_torch), batch_size):
                    batch = flat_rgb_torch[i:i+batch_size]
                    distances = torch.cdist(batch, lut_rgb_torch)
                    batch_indices = torch.argmin(distances, dim=1)
                    indices_list.append(batch_indices)
                
                indices = torch.cat(indices_list)
                indices_cpu = indices.cpu().numpy()
                print(f"[IMAGE_PROCESSOR] PyTorch CUDA color matching complete!")
            except Exception as e:
                print(f"[CUDA] PyTorch matching failed: {e}. Falling back to CPU.")
                _, indices_cpu = self.kdtree.query(flat_rgb)
        else:
            _, indices_cpu = self.kdtree.query(flat_rgb)
        
        matched_rgb = self.lut_rgb[indices_cpu].reshape(target_h, target_w, 3)
        material_matrix = self.ref_stacks[indices_cpu].reshape(
            target_h, target_w, PrinterConfig.COLOR_LAYERS
        )
        
        print(f"[IMAGE_PROCESSOR] Direct matching complete!")
        
        return matched_rgb, material_matrix, rgb_arr


# Backward compatibility alias
LuminaImageProcessor = LuminaImageProcessorCUDA
