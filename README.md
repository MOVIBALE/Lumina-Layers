# Lumina Studio

Physics-Based Multi-Material FDM Color System

[📖 中文文档 / Chinese Version](README_CN.md)

---

## Project Status

**Current Version**: v1.5.0  
**License**: CC BY-NC-SA 4.0 (with Commercial Exemption)  
**Nature**: Non-profit independent implementation, open-source community project

---
Inspiration and Technical Statements

### Acknowledgements to Pioneers

This project owes its existence to the open sharing of the following technologies:

- **HueForge** - The first tool to introduce optical color mixing to the FDM community, demonstrating that layering transparent materials can achieve rich colors through light transmission.

- **AutoForge** - An automated color matching workflow, making multi-material color printing easy to use.

- **CMYK Printing Theory** - A layer-by-layer transmission adaptation of the classic subtractive color model in 3D printing.

### Technical Differentiation and Positioning

Traditional tools rely on theoretical calculations (such as TD1/TD0 transmission distance values), but these parameters are highly susceptible to failure due to various objective factors.

 **Lumina Studio employs an exhaustive search approach:**

1. Print a 1024-color physical calibration board (4 colors x 5 layers, full permutation)

2. Scan the board by photograph and extract the actual RGB data

3. Create a "LUT" (Learning Unknown Test Table)

4. Use a nearest neighbor algorithm for matching (similar to the matching in Bambulab's keychain generator).

### Prior Art Declaration

The core principle of FDM multilayer overlay was publicly disclosed by software such as HueForge between 2022 and 2023, and is considered **prior art**.

The HueForge authors have also clearly stated that this technology has entered the public domain, and in most countries and regions, if the patent office carefully examines it, a principle patent would certainly be rejected.

The pioneers have chosen to remain open to help the community develop; therefore, this technology is generally **not patentable**.

Lumina Studio will continue to maintain its open-source, collaborative, and non-profit positioning, and we welcome everyone's supervision.
Lumina Studio will continue to operate on an open-source, collaborative, and non-profit basis, and we welcome your feedback.

- This project is an open-source, non-profit project. There will be no bundled sales, and no features will be made into paid features.
- If you or your company wish to support the project's continued development, please contact us. Sponsored products will only be used for software development, testing, and optimization.
- Sponsorship represents support for the project only and does not constitute any commercial binding.
- We reject any sponsorship collaborations that could influence technical decisions or open-source licenses.

Lumina Studio did not refer to any patent applications because such patents usually only contain specifications and the technical code is not disclosed in the short term. Blindly referring to these patents would affect its own development process.

**Special thanks to the HueForge team for their support and understanding of open source!**  **
---

## Open Ecosystem

### About .npy Calibration Files

All calibration presets (`.npy` files) are **completely free and open**, adhering to the following principles:

- **No Vendor Lock-in:** We have never, currently, and will never force users to use specific consumable brands, nor will we require manufacturers to produce specific "compatible consumables" that meet our requirements. This violates the spirit of open source.

- **Community Collaboration:** We welcome all users, organizations, and consumable manufacturers to submit PRs and synchronize calibration presets. Your printer data can help others.

- No other testing tools are needed; all you need is a 3D printer and a mobile phone.

**Open Data = Democratization of Technology**

---

## License

### Core License: CC BY-NC-SA 4.0

- ✅ **Attribution**: You must give appropriate credit
- ❌ **NonCommercial**: You may not sell the source code or close it
- 🔄 **ShareAlike**: If you modify it, you must distribute it under the same license

### Commercial Exemption ("Street Vendor" Special Authorization)

**For individual creators, street vendors, and small private businesses**:

You **do NOT need to ask for permission**. You automatically have the right to:
- Use this software to generate models
- Sell physical prints (keychains, reliefs, etc.)
- Sell at night markets, fairs, and small online shops

**Go set up your stall and make money! This is your right.**

*Note: Batch industrial production, SaaS platform operations, and OEM branding still require commercial licensing from the author.*

---

Lumina Studio v1.5.0 integrates three major modules into a unified interface:

### 📐 Module 1: Calibration Generator

Generates precision calibration boards to physically test filament mixing.

- **1024-Color Matrix**: Full permutation of 4 base filaments across 5 layers (0.4mm total)
- **Dual Color Modes**: Supports both CMYW (Cyan/Magenta/Yellow/White) and RYBW (Red/Yellow/Blue/White)
- **Face-Down Optimization**: Viewing surface prints directly on the build plate for a smooth finish
- **Solid Backing**: Automatically generates a 1.6mm opaque backing to ensure color consistency and structural rigidity
- **Anti-Overlap Geometry**: Applies 0.02mm micro-shrinkage to voxels to prevent slicer line-width conflicts

### 🎨 Module 2: Color Extractor

Digitizes the physical reality of your printer.

- **Computer Vision**: Perspective warp + lens distortion correction for automatic grid alignment
- **Mode-Aware Alignment**: Corner markers follow the correct color sequence based on your selected mode (CMYW vs RYBW)
- **Digital Twin**: Extracts RGB values from the print and generates a .npy LUT file
- **Human-in-the-Loop**: Interactive probe tools allow manual verification/correction of specific color block readings (e.g., removing glare/shadows)

### 💎 Module 3: Image Converter

Converts images into printable 3D models using calibrated data.

- **KD-Tree Color Matching**: Maps image pixels to actual printable colors found in your LUT
- **Live 3D Preview**: Interactive WebGL preview with true matched colors—rotate, zoom, and inspect before printing
- **Keychain Loop Generator**: Automatically adds functional hanging loops with:
  - Smart color detection (matches nearby model colors)
  - Customizable dimensions (width, length, hole diameter)
  - Rectangle base + semicircle top + hollow hole geometry
  - 2D preview shows loop placement
- **Structure Options**: Double-sided (keychain) or Single-sided (relief) modes
- **Smart Background Removal**: Automatic transparency detection with adjustable tolerance
- **Correct 3MF Naming**: Objects are named by color (e.g., "Cyan", "Magenta") instead of "geometry_0" for easy slicer identification

---

## What's New in v1.5.0 🚀

### GPU Acceleration Support 🎮

**NVIDIA and AMD GPU users can now enjoy significant performance boosts:**
- **K-Means Color Quantization**: Up to 10x faster processing on large images
- **Color Matching**: Significant speedup for high-resolution image conversion
- **Automatic Detection**: Software automatically uses GPU if available, falls back to CPU otherwise
- **Easy Setup**: Just install PyTorch with CUDA support (see Installation section)

Supported GPUs:
- **NVIDIA**: GTX/RTX series with CUDA 11.8 or 12.1 ✅ Tested
- **AMD**: RX 6000/7000 series with ROCm 6.2 (Linux only) ⚠️ Not tested

### Code Standardization

- ✅ **English-only Comments** - All code comments translated to English for better international collaboration
- ✅ **Documentation Standards** - Unified Google-style docstrings across codebase
- ✅ **Code Cleanup** - Removed redundant comments, kept essential algorithm explanations

---

## Installation

### Clone the repository

```bash
git clone https://github.com/MOVIBALE/Lumina-Layers.git
cd Lumina-Layers
```

### Quick Start (Recommended)

**For Windows users**, simply run:

```bash
start.bat
```

This script will:
- ✅ Auto-detect your GPU (NVIDIA or not)
- ✅ Install the appropriate PyTorch version (CUDA for NVIDIA, CPU for others)
- ✅ Install all other dependencies
- ✅ Launch the application

### Manual Installation

If you prefer manual installation or are on Linux/Mac:

**1. Clone and navigate:**
```bash
git clone https://github.com/MOVIBALE/Lumina-Layers.git
cd Lumina-Layers
```

**2. Install dependencies:**

**Option A: With NVIDIA GPU (CUDA 12.1 recommended):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Option B: Without NVIDIA GPU (CPU only):**
```bash
pip install torch torchvision
pip install -r requirements.txt
```

### GPU Acceleration Details

**Supported Configurations:**

| Platform | GPU Type | PyTorch Version |
|----------|----------|----------------|
| Windows | NVIDIA (RTX 20/30/40 series) | CUDA 12.1 |
| Windows | NVIDIA (GTX 10 series) | CUDA 11.8 |
| Windows | AMD/Intel/No GPU | CPU |
| Linux | NVIDIA | CUDA 12.1 |
| Linux | AMD (RX 6000/7000) | ROCm 6.2 |

> ⚠️ **Note**: ROCm (AMD GPU on Linux) support is implemented but **not tested** by developers (no AMD hardware available). If you encounter issues, please report on GitHub Issues.

**Verify GPU support:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Using CPU')"
```

When CUDA is available, the software automatically uses GPU for:
- K-Means color quantization (up to 10x faster on large images)
- Color matching to LUT (significant speedup for high-resolution images)

---

## Usage Guide

### Quick Start

**Windows - With Python installed:**
```bash
start.bat
```

**Windows - Standalone EXE (No Python required):**

We provide two pre-built EXE versions:

| Version | File | Requirements |
|---------|------|--------------|
| **GPU** | `LuminaStudio_GPU.exe` | NVIDIA GPU + Drivers |
| **CPU** | `LuminaStudio_CPU.exe` | Any Windows PC |

**Build your own EXE:**
```bash
# For NVIDIA GPU users
build_gpu.bat

# For all users (no GPU required)
build_cpu.bat
```

**Linux/Mac:**
```bash
python main.py
```

This launches the web interface with all three modules in tabs.

---

### Step 1: Generate Calibration Board

1. Open the **📐 Calibration** tab
2. Select your color mode:
   - **RYBW** (Red/Yellow/Blue/White) - Traditional primaries
   - **CMYW** (Cyan/Magenta/Yellow/White) - Print colors, wider gamut
3. Adjust block size (default: 5mm) and gap (default: 0.82mm)
4. Click **Generate** and download the `.3mf` file

**Print Settings**:

- Layer height: 0.08mm (color layers), backing can use 0.2mm
- Filament slots must match your selected mode

| Mode | Slot 1 | Slot 2 | Slot 3 | Slot 4 |
|------|--------|--------|--------|--------|
| RYBW | White | Red | Yellow | Blue |
| CMYW | White | Cyan | Magenta | Yellow |

---

### Step 2: Extract Colors

1. Print the calibration board and photograph it (face-up, even lighting)
2. Open the **🎨 Color Extractor** tab
3. Select the same color mode as your calibration board
4. Upload your photo
5. Click the four corner blocks in order:

| Mode | Corner 1 (TL) | Corner 2 (TR) | Corner 3 (BR) | Corner 4 (BL) |
|------|---------------|---------------|---------------|---------------|
| RYBW | ⬜ White | 🟥 Red | 🟦 Blue | 🟨 Yellow |
| CMYW | ⬜ White | 🔵 Cyan | 🟣 Magenta | 🟨 Yellow |

6. Adjust correction sliders if needed (white balance, vignette, distortion)
7. Click **Extract** and download `my_printer_lut.npy`

---

### Step 3: Convert Image

1. Open the **💎 Image Converter** tab
2. Upload your `.npy` LUT file
3. Upload your image
4. Select the same color mode as your LUT
5. **Choose Modeling Mode**:
   - **High-Fidelity (Smooth)** - Recommended for logos, photos, portraits, illustrations
   - **Pixel Art (Blocky)** - Recommended for pixel art and 8-bit style graphics
6. Adjust **Color Detail** slider (8-256 colors, default 64):
   - 8-32 colors: Minimalist style, fast generation
   - 64-128 colors: Balanced detail & speed (recommended)
   - 128-256 colors: Photographic detail, slower generation
7. Click **👁️ Generate Preview** to see the result
8. (Optional) Add Keychain Loop:
   - Click on the 2D preview where you want the loop attached
   - Enable "启用挂孔" checkbox
   - Adjust loop width, length, and hole diameter
   - The loop color is automatically detected from nearby pixels
9. Choose structure type:
   - **Double-sided** - For keychains (image on both sides)
   - **Single-sided** - For relief/lithophane style
10. Click **🚀 Generate 3MF**
11. Preview in the interactive 3D viewer
12. Download the `.3mf` file

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Core Logic | Python (NumPy for voxel manipulation) |
| Geometry Engine | Trimesh (Mesh generation & Export) |
| UI Framework | Gradio 4.0+ |
| Vision Stack | OpenCV (Perspective & Color Extraction) |
| Color Matching | SciPy KDTree / PyTorch CUDA (GPU accelerated) |
| Color Quantization | K-Means (OpenCV CPU / PyTorch CUDA GPU) |
| 3D Preview | Gradio Model3D (GLB format) |
| GPU Acceleration | PyTorch 2.0+ with CUDA 11.8/12.1 or ROCm 6.2 |

---

## How It Works

### Why Calibration Matters

Theoretical TD values assume:
- Perfectly consistent filament dye concentration
- Identical nozzle temperatures across all materials
- Uniform layer adhesion

In reality, these vary significantly between:
- Different filament brands/batches
- Printer models and nozzle designs
- Environmental humidity and temperature

The LUT-based approach solves this by measuring actual printed colors and matching them via nearest-neighbor search in RGB space.

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0).

- ✅ **Attribution**: You must give appropriate credit
- ❌ **NonCommercial**: You may not use this for commercial purposes (at the source code level)
- 🔄 **ShareAlike**: If you modify it, you must distribute it under the same license

**Commercial Exemption**: Individual creators, street vendors, and small private businesses may freely use this software to generate models and sell physical prints.

---

## Acknowledgments

Special thanks to:

- **HueForge** - For pioneering optical color mixing in FDM printing
- **AutoForge** - For democratizing multi-color workflows
- **The 3D printing community** - For continuous innovation

---

Made with ❤️ by [MIN]

⭐ Star this repo if you find it useful!
