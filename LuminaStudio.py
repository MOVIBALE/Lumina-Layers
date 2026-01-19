"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          LUMINA STUDIO v1.3                                   ║
║                    Multi-Material 3D Print Color System                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Author: [MIN]                                                                ║
║  License: CC BY-NC-SA 4.0                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import tempfile
import zipfile
import io
import re
from typing import List, Tuple, Optional
from datetime import datetime

import gradio as gr
import numpy as np
import trimesh
from PIL import Image
import cv2
from scipy.spatial import KDTree


def _safe_fix_3mf_names(filepath: str, slot_names: List[str], create_assembly: bool = True):
    """
    Fix object names in 3MF file and optionally create an assembly.
    Maps objects to slot_names in the order they appear in the file.

    Args:
        filepath: 3MF文件路径
        slot_names: 对象名称列表
        create_assembly: 是否创建组合体
    """
    try:
        # Read original 3MF
        with zipfile.ZipFile(filepath, 'r') as zf_in:
            files_data = {}
            for name in zf_in.namelist():
                files_data[name] = zf_in.read(name)

        # Find the 3D model file
        model_file = None
        for name in files_data:
            if name.endswith('.model') and '3D/' in name:
                model_file = name
                break

        if model_file and model_file in files_data:
            content = files_data[model_file].decode('utf-8')

            # Find all <object> tags with their IDs (in order of appearance)
            object_pattern = re.compile(r'<object\s+([^>]*)>', re.IGNORECASE)

            # Track which objects we've seen
            obj_info = []  # List of (start_pos, end_pos, full_tag, id)

            for match in object_pattern.finditer(content):
                attrs = match.group(1)
                id_match = re.search(r'\bid="(\d+)"', attrs)
                if id_match:
                    obj_id = id_match.group(1)
                    obj_info.append((match.start(), match.end(), match.group(0), obj_id))

            # Collect object IDs for assembly
            object_ids = [info[3] for info in obj_info]
            print(f"[DEBUG] Found {len(object_ids)} objects in 3MF: {object_ids}")

            # Process in reverse order to preserve positions (for name fixing)
            for idx, (start, end, old_tag, obj_id) in enumerate(reversed(obj_info)):
                real_idx = len(obj_info) - 1 - idx
                if real_idx >= len(slot_names):
                    continue

                color_name = slot_names[real_idx]

                # Remove existing name attribute and add new one
                new_tag = re.sub(r'\s+name="[^"]*"', '', old_tag)
                new_tag = new_tag[:-1] + f' name="{color_name}">'

                content = content[:start] + new_tag + content[end:]

            # Create assembly if requested
            if create_assembly and len(object_ids) > 1:
                # Find the maximum object ID
                max_id = max(int(oid) for oid in object_ids)
                assembly_id = max_id + 1

                # Create assembly object XML
                components_xml = '\n'.join([f'      <component objectid="{oid}" />' for oid in object_ids])
                assembly_xml = f'''
  <object id="{assembly_id}" type="model" name="Lumina_Model">
    <components>
{components_xml}
    </components>
  </object>
'''

                # Insert assembly before </resources>
                resources_end = content.find('</resources>')
                if resources_end != -1:
                    content = content[:resources_end] + assembly_xml + content[resources_end:]
                    print(f"[DEBUG] Created assembly with id={assembly_id}, containing {len(object_ids)} components")

                # Modify <build> section to only reference the assembly
                # Find and replace the build section
                build_pattern = re.compile(r'<build>.*?</build>', re.DOTALL)
                build_match = build_pattern.search(content)
                if build_match:
                    new_build = f'<build>\n    <item objectid="{assembly_id}" />\n  </build>'
                    content = content[:build_match.start()] + new_build + content[build_match.end():]
                    print(f"[DEBUG] Updated build section to reference assembly")

            files_data[model_file] = content.encode('utf-8')

        # Write back
        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf_out:
            for name, data in files_data.items():
                zf_out.writestr(name, data)

        print(f"[DEBUG] 3MF file updated successfully: {filepath}")

    except Exception as e:
        print(f"Warning: Could not fix 3MF names: {e}")

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                           SHARED CONFIGURATION                                ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class PrinterConfig:
    """Physical printer parameters."""
    LAYER_HEIGHT: float = 0.08
    NOZZLE_WIDTH: float = 0.42
    COLOR_LAYERS: int = 5
    BACKING_MM: float = 1.6
    SHRINK_OFFSET: float = 0.02


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                           INTERNATIONALIZATION                                ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class I18N:
    """Internationalization support for Chinese and English."""

    TEXTS = {
        # Header
        'app_title': {'zh': '✨ Lumina Studio', 'en': '✨ Lumina Studio'},
        'app_subtitle': {'zh': '多材料3D打印色彩系统', 'en': 'Multi-Material 3D Print Color System'},

        # Stats
        'stats_total': {'zh': '📊 累计生成:', 'en': '📊 Total Generated:'},
        'stats_calibrations': {'zh': '校准板', 'en': 'Calibrations'},
        'stats_extractions': {'zh': '颜色提取', 'en': 'Extractions'},
        'stats_conversions': {'zh': '模型转换', 'en': 'Conversions'},

        # Tabs
        'tab_calibration': {'zh': '📐 校准板生成', 'en': '📐 Calibration'},
        'tab_extractor': {'zh': '🎨 颜色提取', 'en': '🎨 Color Extractor'},
        'tab_converter': {'zh': '💎 图像转换', 'en': '💎 Image Converter'},
        'tab_about': {'zh': 'ℹ️ 关于', 'en': 'ℹ️ About'},

        # Tab 1: Calibration
        'cal_title': {'zh': '### 第一步：生成校准板', 'en': '### Step 1: Generate Calibration Board'},
        'cal_desc': {'zh': '生成1024种颜色的校准板，打印后用于提取打印机的实际色彩数据。',
                     'en': 'Generate a 1024-color calibration board. Print it to extract your printer\'s actual color data.'},
        'cal_params': {'zh': '#### ⚙️ 参数设置', 'en': '#### ⚙️ Parameters'},
        'cal_mode': {'zh': '色彩模式', 'en': 'Color Mode'},
        'cal_mode_cmyw': {'zh': 'CMYW (青/品红/黄)', 'en': 'CMYW (Cyan/Magenta/Yellow)'},
        'cal_mode_rybw': {'zh': 'RYBW (红/黄/蓝)', 'en': 'RYBW (Red/Yellow/Blue)'},
        'cal_block_size': {'zh': '色块尺寸 (mm)', 'en': 'Block Size (mm)'},
        'cal_gap': {'zh': '间隙 (mm)', 'en': 'Gap (mm)'},
        'cal_backing': {'zh': '底板颜色', 'en': 'Backing Color'},
        'cal_generate': {'zh': '🚀 生成校准板', 'en': '🚀 Generate'},
        'cal_status': {'zh': '状态', 'en': 'Status'},
        'cal_preview': {'zh': '#### 👁️ 预览', 'en': '#### 👁️ Preview'},
        'cal_download': {'zh': '下载 3MF 文件', 'en': 'Download 3MF'},
        'cal_success': {'zh': '✅ 校准板已生成！对象名称:', 'en': '✅ Calibration board generated! Object names:'},

        # Tab 2: Extractor
        'ext_title': {'zh': '### 第二步：提取颜色数据', 'en': '### Step 2: Extract Color Data'},
        'ext_desc': {'zh': '拍摄打印好的校准板照片，提取真实的色彩数据生成 LUT 文件。',
                     'en': 'Take a photo of your printed calibration board to extract real color data and generate a LUT file.'},
        'ext_upload': {'zh': '#### 📸 上传照片', 'en': '#### 📸 Upload Photo'},
        'ext_color_mode': {'zh': '🎨 校准板的色彩模式', 'en': '🎨 Calibration Board Color Mode'},
        'ext_photo': {'zh': '校准板照片', 'en': 'Calibration Photo'},
        'ext_rotate': {'zh': '↺ 旋转', 'en': '↺ Rotate'},
        'ext_reset': {'zh': '🗑️ 重置点位', 'en': '🗑️ Reset Points'},
        'ext_correction': {'zh': '#### 🔧 校正参数', 'en': '#### 🔧 Correction'},
        'ext_wb': {'zh': '自动白平衡', 'en': 'Auto White Balance'},
        'ext_vignette': {'zh': '暗角校正', 'en': 'Vignette Fix'},
        'ext_zoom': {'zh': '缩放', 'en': 'Zoom'},
        'ext_distortion': {'zh': '畸变', 'en': 'Distortion'},
        'ext_offset_x': {'zh': 'X偏移', 'en': 'Offset X'},
        'ext_offset_y': {'zh': 'Y偏移', 'en': 'Offset Y'},
        'ext_extract': {'zh': '🚀 提取颜色', 'en': '🚀 Extract Colors'},
        'ext_hint_white': {'zh': '#### 👉 点击: **白色色块 (左上角)**', 'en': '#### 👉 Click: **White Block (Top-Left)**'},
        'ext_hint_done': {'zh': '#### ✅ 定位完成，可以提取颜色了！', 'en': '#### ✅ Positioning complete, ready to extract!'},
        'ext_sampling': {'zh': '#### 📍 采样预览', 'en': '#### 📍 Sampling Preview'},
        'ext_reference': {'zh': '#### 🎯 参考对照', 'en': '#### 🎯 Reference'},
        'ext_result': {'zh': '#### 📊 提取结果 (点击修正)', 'en': '#### 📊 Result (Click to Fix)'},
        'ext_manual_fix': {'zh': '#### 🛠️ 手动修正', 'en': '#### 🛠️ Manual Fix'},
        'ext_click_cell': {'zh': '点击左侧色块查看...', 'en': 'Click a cell on the left...'},
        'ext_override': {'zh': '替换颜色', 'en': 'Override Color'},
        'ext_apply': {'zh': '🔧 应用修正', 'en': '🔧 Apply Fix'},
        'ext_download_npy': {'zh': '下载 .npy', 'en': 'Download .npy'},
        'ext_success': {'zh': '✅ 提取完成！LUT已保存', 'en': '✅ Extraction complete! LUT saved'},
        'ext_no_image': {'zh': '❌ 请先上传图片', 'en': '❌ Please upload an image first'},
        'ext_need_4_points': {'zh': '❌ 请点击4个角点', 'en': '❌ Please click 4 corner points'},

        # Tab 3: Converter
        'conv_title': {'zh': '### 第三步：转换图像', 'en': '### Step 3: Convert Image'},
        'conv_desc': {'zh': '使用校准数据将图像转换为多层 3D 模型，实现精准色彩还原。',
                      'en': 'Convert images to multi-layer 3D models using calibration data for accurate color reproduction.'},
        'conv_input': {'zh': '#### 📁 输入文件', 'en': '#### 📁 Input Files'},
        'conv_lut': {'zh': '1. 校准数据 (.npy)', 'en': '1. Calibration Data (.npy)'},
        'conv_image': {'zh': '2. 输入图像', 'en': '2. Input Image'},
        'conv_params': {'zh': '#### ⚙️ 参数设置', 'en': '#### ⚙️ Parameters'},
        'conv_color_mode': {'zh': '🎨 色彩模式（需与校准板一致）', 'en': '🎨 Color Mode (must match calibration)'},
        'conv_structure': {'zh': '结构类型', 'en': 'Structure Type'},
        'conv_double': {'zh': '双面 (钥匙扣)', 'en': 'Double-sided (Keychain)'},
        'conv_single': {'zh': '单面 (浮雕)', 'en': 'Single-sided (Relief)'},
        'conv_auto_bg': {'zh': '自动移除背景', 'en': 'Auto Background Removal'},
        'conv_tolerance': {'zh': '背景容差', 'en': 'Background Tolerance'},
        'conv_width': {'zh': '目标宽度 (mm)', 'en': 'Target Width (mm)'},
        'conv_thickness': {'zh': '背板厚度 (mm)', 'en': 'Backing Thickness (mm)'},
        'conv_generate': {'zh': '🚀 生成模型', 'en': '🚀 Generate Model'},
        'conv_3d_preview': {'zh': '#### 🎮 3D 预览（可拖拽旋转/滚轮缩放）', 'en': '#### 🎮 3D Preview (Drag to rotate / Scroll to zoom)'},
        'conv_color_preview': {'zh': '#### 🎨 色彩预览', 'en': '#### 🎨 Color Preview'},
        'conv_download': {'zh': '#### 📁 下载', 'en': '#### 📁 Download'},
        'conv_download_3mf': {'zh': '下载 3MF 文件', 'en': 'Download 3MF'},
        'conv_success': {'zh': '✅ 转换完成！分辨率:', 'en': '✅ Conversion complete! Resolution:'},
        'conv_no_image': {'zh': '❌ 请上传图片', 'en': '❌ Please upload an image'},
        'conv_no_lut': {'zh': '⚠️ 请上传 .npy 校准文件！', 'en': '⚠️ Please upload a .npy calibration file!'},

        # Footer
        'footer_tip': {'zh': '💡 提示: 使用高质量的PLA/PETG透光材料可获得最佳效果',
                       'en': '💡 Tip: Use high-quality translucent PLA/PETG for best results'},

        # Language
        'lang_label': {'zh': '🌐 语言', 'en': '🌐 Language'},
        'lang_zh': {'zh': '中文', 'en': '中文'},
        'lang_en': {'zh': 'English', 'en': 'English'},
    }

    @staticmethod
    def get(key: str, lang: str = 'zh') -> str:
        """Get translated text for a key."""
        if key in I18N.TEXTS:
            return I18N.TEXTS[key].get(lang, I18N.TEXTS[key].get('zh', key))
        return key


class ColorSystem:
    """Color model definitions for CMYW and RYBW."""

    CMYW = {
        'name': 'CMYW',
        'slots': ["White", "Cyan", "Magenta", "Yellow"],
        'preview': {
            0: [255, 255, 255, 255],
            1: [0, 134, 214, 255],
            2: [236, 0, 140, 255],
            3: [244, 238, 42, 255]
        },
        'map': {"White": 0, "Cyan": 1, "Magenta": 2, "Yellow": 3},
        # 定位点顺序: TL, TR, BR, BL
        'corner_labels': ["白色 (左上)", "青色 (右上)", "品红 (右下)", "黄色 (左下)"],
        'corner_labels_en': ["White (TL)", "Cyan (TR)", "Magenta (BR)", "Yellow (BL)"]
    }

    RYBW = {
        'name': 'RYBW',
        'slots': ["White", "Red", "Yellow", "Blue"],
        'preview': {
            0: [255, 255, 255, 255],
            1: [220, 20, 60, 255],
            2: [255, 230, 0, 255],
            3: [0, 100, 240, 255]
        },
        'map': {"White": 0, "Red": 1, "Yellow": 2, "Blue": 3},
        # 定位点顺序: TL, TR, BR, BL
        'corner_labels': ["白色 (左上)", "红色 (右上)", "黄色 (右下)", "蓝色 (左下)"],
        'corner_labels_en': ["White (TL)", "Red (TR)", "Yellow (BR)", "Blue (BL)"]
    }

    @staticmethod
    def get(mode: str):
        return ColorSystem.CMYW if "CMYW" in mode else ColorSystem.RYBW


# Usage statistics (local counter)
class Stats:
    _file = os.path.join(tempfile.gettempdir(), "lumina_stats.txt")

    @staticmethod
    def increment(key: str) -> int:
        data = Stats._load()
        data[key] = data.get(key, 0) + 1
        Stats._save(data)
        return data[key]

    @staticmethod
    def get_all() -> dict:
        return Stats._load()

    @staticmethod
    def _load() -> dict:
        try:
            with open(Stats._file, 'r') as f:
                lines = f.readlines()
                return {l.split(':')[0]: int(l.split(':')[1]) for l in lines if ':' in l}
        except:
            return {"calibrations": 0, "extractions": 0, "conversions": 0}

    @staticmethod
    def _save(data: dict):
        try:
            with open(Stats._file, 'w') as f:
                for k, v in data.items():
                    f.write(f"{k}:{v}\n")
        except:
            pass


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                     MODULE 1: CALIBRATION GENERATOR                           ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def _generate_voxel_mesh(voxel_matrix: np.ndarray, material_index: int,
                          grid_h: int, grid_w: int) -> Optional[trimesh.Trimesh]:
    """Generate mesh for a specific material from voxel data."""
    scale_x = PrinterConfig.NOZZLE_WIDTH
    scale_y = PrinterConfig.NOZZLE_WIDTH
    scale_z = PrinterConfig.LAYER_HEIGHT
    shrink = PrinterConfig.SHRINK_OFFSET

    vertices, faces = [], []
    total_z_layers = voxel_matrix.shape[0]

    for z in range(total_z_layers):
        z_bottom, z_top = z * scale_z, (z + 1) * scale_z
        layer_mask = (voxel_matrix[z] == material_index)
        if not np.any(layer_mask):
            continue

        for y in range(grid_h):
            world_y = y * scale_y
            row = layer_mask[y]
            padded_row = np.pad(row, (1, 1), mode='constant')
            diff = np.diff(padded_row.astype(int))
            starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]

            for start, end in zip(starts, ends):
                x0, x1 = start * scale_x + shrink, end * scale_x - shrink
                y0, y1 = world_y + shrink, world_y + scale_y - shrink

                base_idx = len(vertices)
                vertices.extend([
                    [x0, y0, z_bottom], [x1, y0, z_bottom], [x1, y1, z_bottom], [x0, y1, z_bottom],
                    [x0, y0, z_top], [x1, y0, z_top], [x1, y1, z_top], [x0, y1, z_top]
                ])
                cube_faces = [
                    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                    [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                    [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]
                ]
                faces.extend([[v + base_idx for v in f] for f in cube_faces])

    if not vertices:
        return None

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    return mesh





def generate_calibration_board(color_mode: str, block_size_mm: float,
                                gap_mm: float, backing_color: str):
    """Generate a 1024-color calibration board as 3MF."""

    color_conf = ColorSystem.get(color_mode)
    slot_names = color_conf['slots']
    preview_colors = color_conf['preview']
    color_map = color_conf['map']

    backing_id = color_map.get(backing_color, 0)

    # Grid setup
    grid_dim, padding = 32, 1
    total_w = total_h = grid_dim + (padding * 2)

    pixels_per_block = max(1, int(block_size_mm / PrinterConfig.NOZZLE_WIDTH))
    pixels_gap = max(1, int(gap_mm / PrinterConfig.NOZZLE_WIDTH))

    voxel_w = total_w * (pixels_per_block + pixels_gap)
    voxel_h = total_h * (pixels_per_block + pixels_gap)

    backing_layers = int(PrinterConfig.BACKING_MM / PrinterConfig.LAYER_HEIGHT)
    total_layers = PrinterConfig.COLOR_LAYERS + backing_layers

    full_matrix = np.full((total_layers, voxel_h, voxel_w), backing_id, dtype=int)

    # Generate 1024 permutations
    for i in range(1024):
        digits = []
        temp = i
        for _ in range(5):
            digits.append(temp % 4)
            temp //= 4
        stack = digits[::-1]

        row = (i // grid_dim) + padding
        col = (i % grid_dim) + padding
        px = col * (pixels_per_block + pixels_gap)
        py = row * (pixels_per_block + pixels_gap)

        for z in range(PrinterConfig.COLOR_LAYERS):
            full_matrix[z, py:py+pixels_per_block, px:px+pixels_per_block] = stack[z]

    # Corner markers - 根据模式设置不同的角点颜色
    # 角点位置: (row, col, mat_id)
    # row=0是顶部, row=total_h-1是底部
    # col=0是左边, col=total_w-1是右边
    if "RYBW" in color_mode:
        # RYBW: slots = [White(0), Red(1), Yellow(2), Blue(3)]
        # corner_labels: TL=White, TR=Red, BR=Blue, BL=Yellow
        corners = [
            (0, 0, 0),              # TL = White
            (0, total_w-1, 1),      # TR = Red
            (total_h-1, total_w-1, 3),  # BR = Blue
            (total_h-1, 0, 2)       # BL = Yellow
        ]
    else:
        # CMYW: slots = [White(0), Cyan(1), Magenta(2), Yellow(3)]
        # corner_labels: TL=White, TR=Cyan, BR=Magenta, BL=Yellow
        corners = [
            (0, 0, 0),              # TL = White
            (0, total_w-1, 1),      # TR = Cyan
            (total_h-1, total_w-1, 2),  # BR = Magenta
            (total_h-1, 0, 3)       # BL = Yellow
        ]

    for r, c, mat_id in corners:
        px = c * (pixels_per_block + pixels_gap)
        py = r * (pixels_per_block + pixels_gap)
        for z in range(PrinterConfig.COLOR_LAYERS):
            full_matrix[z, py:py+pixels_per_block, px:px+pixels_per_block] = mat_id

    # Build 3MF
    scene = trimesh.Scene()
    for mat_id in range(4):
        mesh = _generate_voxel_mesh(full_matrix, mat_id, voxel_h, voxel_w)
        if mesh:
            mesh.visual.face_colors = preview_colors[mat_id]
            name = slot_names[mat_id]
            # Set multiple name attributes to increase compatibility
            mesh.metadata['name'] = name
            scene.add_geometry(mesh, node_name=name, geom_name=name)

    # Export
    mode_tag = color_conf['name']
    output_path = os.path.join(tempfile.gettempdir(), f"Lumina_Calibration_{mode_tag}.3mf")
    scene.export(output_path)

    # Fix object names in 3MF for better slicer compatibility
    _safe_fix_3mf_names(output_path, slot_names)

    # Preview
    bottom_layer = full_matrix[0].astype(np.uint8)
    preview_arr = np.zeros((voxel_h, voxel_w, 3), dtype=np.uint8)
    for mat_id, rgba in preview_colors.items():
        preview_arr[bottom_layer == mat_id] = rgba[:3]

    Stats.increment("calibrations")

    return output_path, Image.fromarray(preview_arr), f"✅ 校准板已生成！已组合为一个对象 | 颜色: {', '.join(slot_names)}"


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                      MODULE 2: COLOR EXTRACTOR                                ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

PHYSICAL_GRID_SIZE = 34
DATA_GRID_SIZE = 32
DST_SIZE = 1000
CELL_SIZE = DST_SIZE / PHYSICAL_GRID_SIZE
LUT_FILE_PATH = os.path.join(tempfile.gettempdir(), "lumina_lut.npy")


def generate_simulated_reference():
    """Generate reference image for visual comparison."""
    colors = {
        0: np.array([250, 250, 250]),
        1: np.array([220, 20, 60]),
        2: np.array([255, 230, 0]),
        3: np.array([0, 100, 240])
    }

    ref_img = np.zeros((DATA_GRID_SIZE, DATA_GRID_SIZE, 3), dtype=np.uint8)
    for i in range(1024):
        digits = []
        temp = i
        for _ in range(5):
            digits.append(temp % 4)
            temp //= 4
        stack = digits[::-1]

        mixed = sum(colors[mid] for mid in stack) / 5.0
        ref_img[i // DATA_GRID_SIZE, i % DATA_GRID_SIZE] = mixed.astype(np.uint8)

    return cv2.resize(ref_img, (512, 512), interpolation=cv2.INTER_NEAREST)


def rotate_image(img, direction):
    if img is None:
        return None
    if direction == "左旋 90°":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == "右旋 90°":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def draw_corner_points(img, points, color_mode: str):
    """Draw corner points with mode-specific colors and labels."""
    if img is None:
        return None

    vis = img.copy()
    color_conf = ColorSystem.get(color_mode)
    labels = color_conf['corner_labels']

    # Define colors for drawing (BGR for OpenCV)
    if "CMYW" in color_mode:
        draw_colors = [
            (255, 255, 255),  # White
            (214, 134, 0),    # Cyan (BGR)
            (140, 0, 236),    # Magenta (BGR)
            (42, 238, 244)    # Yellow (BGR)
        ]
    else:  # RYBW
        draw_colors = [
            (255, 255, 255),  # White
            (60, 20, 220),    # Red (BGR)
            (240, 100, 0),    # Blue (BGR)
            (0, 230, 255)     # Yellow (BGR)
        ]

    for i, pt in enumerate(points):
        color = draw_colors[i] if i < 4 else (0, 255, 0)

        # Draw filled circle
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 15, color, -1)
        # Draw outline
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 15, (0, 0, 0), 2)
        # Draw number
        cv2.putText(vis, str(i + 1), (int(pt[0]) + 20, int(pt[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Draw label
        if i < 4:
            cv2.putText(vis, labels[i], (int(pt[0]) + 20, int(pt[1]) + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return vis


def apply_auto_white_balance(img):
    h, w, _ = img.shape
    m = 50
    corners = [img[0:m, 0:m], img[0:m, w-m:w], img[h-m:h, 0:m], img[h-m:h, w-m:w]]
    avg_white = sum(c.mean(axis=(0, 1)) for c in corners) / 4.0
    gain = np.array([255, 255, 255]) / (avg_white + 1e-5)
    return np.clip(img.astype(float) * gain, 0, 255).astype(np.uint8)


def apply_brightness_correction(img):
    h, w, _ = img.shape
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)

    m = 50
    tl, tr = l[0:m, 0:m].mean(), l[0:m, w-m:w].mean()
    bl, br = l[h-m:h, 0:m].mean(), l[h-m:h, w-m:w].mean()

    top = np.linspace(tl, tr, w)
    bot = np.linspace(bl, br, w)
    mask = np.array([top * (1 - y/h) + bot * (y/h) for y in range(h)])

    target = (tl + tr + bl + br) / 4.0
    l_new = np.clip(l.astype(float) * (target / (mask + 1e-5)), 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([l_new, a, b]), cv2.COLOR_LAB2RGB)


def run_extraction(img, points, offset_x, offset_y, zoom, barrel, wb, bright):
    """Main extraction pipeline."""
    if img is None:
        return None, None, None, "❌ 请先上传图片"
    if len(points) != 4:
        return None, None, None, "❌ 请点击4个角点"

    # Perspective transform
    half = CELL_SIZE / 2.0
    src = np.float32(points)
    dst = np.float32([
        [half, half], [DST_SIZE - half, half],
        [DST_SIZE - half, DST_SIZE - half], [half, DST_SIZE - half]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (DST_SIZE, DST_SIZE))

    if wb:
        warped = apply_auto_white_balance(warped)
    if bright:
        warped = apply_brightness_correction(warped)

    # Sampling
    extracted = np.zeros((DATA_GRID_SIZE, DATA_GRID_SIZE, 3), dtype=np.uint8)
    vis = warped.copy()

    for r in range(DATA_GRID_SIZE):
        for c in range(DATA_GRID_SIZE):
            phys_r, phys_c = r + 1, c + 1
            nx = (phys_c + 0.5) / PHYSICAL_GRID_SIZE * 2 - 1
            ny = (phys_r + 0.5) / PHYSICAL_GRID_SIZE * 2 - 1

            rad = np.sqrt(nx**2 + ny**2)
            k = 1 + barrel * (rad**2)
            dx, dy = nx * k * zoom, ny * k * zoom

            cx = (dx + 1) / 2 * DST_SIZE + offset_x
            cy = (dy + 1) / 2 * DST_SIZE + offset_y

            if 0 <= cx < DST_SIZE and 0 <= cy < DST_SIZE:
                x0, y0 = int(max(0, cx - 4)), int(max(0, cy - 4))
                x1, y1 = int(min(DST_SIZE, cx + 4)), int(min(DST_SIZE, cy + 4))
                reg = warped[y0:y1, x0:x1]
                avg = reg.mean(axis=(0, 1)).astype(int) if reg.size > 0 else [0, 0, 0]
                cv2.drawMarker(vis, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
            else:
                avg = [0, 0, 0]
            extracted[r, c] = avg

    np.save(LUT_FILE_PATH, extracted)
    prev = cv2.resize(extracted, (512, 512), interpolation=cv2.INTER_NEAREST)

    Stats.increment("extractions")

    return vis, prev, LUT_FILE_PATH, "✅ 提取完成！LUT已保存"


def probe_lut_cell(evt: gr.SelectData):
    if not os.path.exists(LUT_FILE_PATH):
        return "⚠️ 无数据", None, None
    try:
        lut = np.load(LUT_FILE_PATH)
    except:
        return "⚠️ 数据损坏", None, None

    x, y = evt.index
    scale = 512 / DATA_GRID_SIZE
    c = min(max(int(x / scale), 0), DATA_GRID_SIZE - 1)
    r = min(max(int(y / scale), 0), DATA_GRID_SIZE - 1)

    rgb = lut[r, c]
    hex_c = '#{:02x}{:02x}{:02x}'.format(*rgb)

    html = f"""
    <div style='background:#1a1a2e; padding:10px; border-radius:8px; color:white;'>
        <b>行 {r+1} / 列 {c+1}</b><br>
        <div style='background:{hex_c}; width:60px; height:30px; border:2px solid white; 
             display:inline-block; vertical-align:middle; border-radius:4px;'></div>
        <span style='margin-left:10px; font-family:monospace;'>{hex_c}</span>
    </div>
    """
    return html, hex_c, (r, c)


def manual_fix_cell(coord, color_input):
    if not coord or not os.path.exists(LUT_FILE_PATH):
        return None, "⚠️ 错误"

    try:
        lut = np.load(LUT_FILE_PATH)
        r, c = coord
        new_color = [0, 0, 0]

        color_str = str(color_input)
        if color_str.startswith('rgb'):
            clean = color_str.replace('rgb', '').replace('a', '').replace('(', '').replace(')', '')
            parts = clean.split(',')
            if len(parts) >= 3:
                new_color = [int(float(p.strip())) for p in parts[:3]]
        elif color_str.startswith('#'):
            hex_s = color_str.lstrip('#')
            new_color = [int(hex_s[i:i+2], 16) for i in (0, 2, 4)]
        else:
            new_color = [int(color_str[i:i+2], 16) for i in (0, 2, 4)]

        lut[r, c] = new_color
        np.save(LUT_FILE_PATH, lut)
        return cv2.resize(lut, (512, 512), interpolation=cv2.INTER_NEAREST), "✅ 已修正"
    except Exception as e:
        return None, f"❌ 格式错误: {color_input}"


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                      MODULE 3: IMAGE CONVERTER                                ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def create_keychain_loop(width_mm, length_mm, hole_dia_mm, thickness_mm, attach_x_mm, attach_y_mm):
    """
    创建钥匙扣挂孔 - 手动构建网格，无需额外依赖

    Args:
        width_mm: 挂孔宽度（也是顶部圆形的直径）
        length_mm: 挂孔总长度
        hole_dia_mm: 孔洞直径
        thickness_mm: 挂孔厚度
        attach_x_mm: 连接点X坐标
        attach_y_mm: 连接点Y坐标（模型顶部）
    """
    print(f"[DEBUG] create_keychain_loop called: width={width_mm}, length={length_mm}, hole={hole_dia_mm}, thick={thickness_mm}, x={attach_x_mm}, y={attach_y_mm}")

    half_w = width_mm / 2
    circle_radius = half_w
    hole_radius = min(hole_dia_mm / 2, circle_radius * 0.8)

    # 矩形部分高度
    rect_height = max(0.2, length_mm - circle_radius)

    # 圆心Y坐标（相对于底部）
    circle_center_y = rect_height

    # ========== 创建外轮廓点 ==========
    n_arc = 32  # 半圆的细分数
    outer_pts = []

    # 底边左
    outer_pts.append((-half_w, 0))
    # 底边右
    outer_pts.append((half_w, 0))
    # 右边
    outer_pts.append((half_w, rect_height))

    # 半圆顶部（从右到左，0°到180°）
    for i in range(1, n_arc):
        angle = np.pi * i / n_arc
        x = circle_radius * np.cos(angle)
        y = circle_center_y + circle_radius * np.sin(angle)
        outer_pts.append((x, y))

    # 左边
    outer_pts.append((-half_w, rect_height))

    outer_pts = np.array(outer_pts)
    n_outer = len(outer_pts)

    # ========== 创建孔洞轮廓点 ==========
    n_hole = 32
    hole_pts = []
    for i in range(n_hole):
        angle = 2 * np.pi * i / n_hole
        x = hole_radius * np.cos(angle)
        y = circle_center_y + hole_radius * np.sin(angle)
        hole_pts.append((x, y))
    hole_pts = np.array(hole_pts)
    n_hole_pts = len(hole_pts)

    # ========== 手动三角化顶面和底面 ==========
    # 使用扇形三角化：从外轮廓中心向各边连接
    # 这是一个简化的方法，对于凸多边形有效

    # 计算外轮廓的质心
    outer_center = outer_pts.mean(axis=0)
    hole_center = np.array([0, circle_center_y])

    # 构建顶点数组
    vertices = []
    faces = []

    # 底面顶点 (z=0)
    # 外轮廓
    for pt in outer_pts:
        vertices.append([pt[0], pt[1], 0])
    # 孔洞轮廓
    for pt in hole_pts:
        vertices.append([pt[0], pt[1], 0])

    # 顶面顶点 (z=thickness)
    # 外轮廓
    for pt in outer_pts:
        vertices.append([pt[0], pt[1], thickness_mm])
    # 孔洞轮廓
    for pt in hole_pts:
        vertices.append([pt[0], pt[1], thickness_mm])

    # 索引偏移
    bottom_outer_start = 0
    bottom_hole_start = n_outer
    top_outer_start = n_outer + n_hole_pts
    top_hole_start = n_outer + n_hole_pts + n_outer

    # ========== 外轮廓侧面 ==========
    for i in range(n_outer):
        i_next = (i + 1) % n_outer
        # 底面到顶面的四边形，分成两个三角形
        bi = bottom_outer_start + i
        bi_next = bottom_outer_start + i_next
        ti = top_outer_start + i
        ti_next = top_outer_start + i_next
        faces.append([bi, bi_next, ti_next])
        faces.append([bi, ti_next, ti])

    # ========== 孔洞侧面（法线向内） ==========
    for i in range(n_hole_pts):
        i_next = (i + 1) % n_hole_pts
        bi = bottom_hole_start + i
        bi_next = bottom_hole_start + i_next
        ti = top_hole_start + i
        ti_next = top_hole_start + i_next
        # 反向绕序使法线向内
        faces.append([bi, ti, ti_next])
        faces.append([bi, ti_next, bi_next])

    # ========== 顶面和底面三角化 ==========
    # 对于带孔的环形区域，我们使用径向三角化
    # 将外轮廓和孔洞轮廓连接起来

    # 找到最近的点对来开始连接
    def connect_rings(outer_indices, hole_indices, vertices_arr, is_top=True):
        """连接外轮廓和孔洞，生成三角形"""
        ring_faces = []
        n_o = len(outer_indices)
        n_h = len(hole_indices)

        # 使用双指针方法连接两个环
        oi = 0  # 外轮廓索引
        hi = 0  # 孔洞索引

        # 获取3D顶点（只用x,y）
        def get_2d(idx):
            return np.array([vertices_arr[idx][0], vertices_arr[idx][1]])

        # 连接所有点
        total_steps = n_o + n_h
        for _ in range(total_steps):
            o_curr = outer_indices[oi % n_o]
            o_next = outer_indices[(oi + 1) % n_o]
            h_curr = hole_indices[hi % n_h]
            h_next = hole_indices[(hi + 1) % n_h]

            # 决定是移动外轮廓还是孔洞
            # 计算两种选择的三角形质量
            dist_o = np.linalg.norm(get_2d(o_next) - get_2d(h_curr))
            dist_h = np.linalg.norm(get_2d(o_curr) - get_2d(h_next))

            if oi >= n_o:
                # 外轮廓已遍历完，只移动孔洞
                if is_top:
                    ring_faces.append([o_curr, h_next, h_curr])
                else:
                    ring_faces.append([o_curr, h_curr, h_next])
                hi += 1
            elif hi >= n_h:
                # 孔洞已遍历完，只移动外轮廓
                if is_top:
                    ring_faces.append([o_curr, o_next, h_curr])
                else:
                    ring_faces.append([o_curr, h_curr, o_next])
                oi += 1
            elif dist_o < dist_h:
                # 移动外轮廓
                if is_top:
                    ring_faces.append([o_curr, o_next, h_curr])
                else:
                    ring_faces.append([o_curr, h_curr, o_next])
                oi += 1
            else:
                # 移动孔洞
                if is_top:
                    ring_faces.append([o_curr, h_next, h_curr])
                else:
                    ring_faces.append([o_curr, h_curr, h_next])
                hi += 1

        return ring_faces

    vertices_arr = np.array(vertices)

    # 底面（法线向下，需要反向绕序）
    bottom_outer_idx = list(range(bottom_outer_start, bottom_outer_start + n_outer))
    bottom_hole_idx = list(range(bottom_hole_start, bottom_hole_start + n_hole_pts))
    bottom_faces = connect_rings(bottom_outer_idx, bottom_hole_idx, vertices_arr, is_top=False)
    faces.extend(bottom_faces)

    # 顶面（法线向上）
    top_outer_idx = list(range(top_outer_start, top_outer_start + n_outer))
    top_hole_idx = list(range(top_hole_start, top_hole_start + n_hole_pts))
    top_faces = connect_rings(top_outer_idx, top_hole_idx, vertices_arr, is_top=True)
    faces.extend(top_faces)

    # ========== 平移到正确位置 ==========
    vertices_arr = np.array(vertices)
    vertices_arr[:, 0] += attach_x_mm
    vertices_arr[:, 1] += attach_y_mm

    # 创建mesh
    mesh = trimesh.Trimesh(vertices=vertices_arr, faces=np.array(faces))
    mesh.fix_normals()

    print(f"[DEBUG] Mesh created: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")

    return mesh


def load_calibrated_lut(npy_path):
    """Load and validate LUT file."""
    try:
        lut_grid = np.load(npy_path)
        measured_colors = lut_grid.reshape(-1, 3)
    except:
        return None, None, "❌ LUT文件损坏"

    valid_rgb, valid_stacks = [], []
    base_blue = np.array([30, 100, 200])
    dropped = 0

    for i in range(1024):
        digits = []
        temp = i
        for _ in range(5):
            digits.append(temp % 4)
            temp //= 4
        stack = digits[::-1]

        real_rgb = measured_colors[i]
        dist = np.linalg.norm(real_rgb - base_blue)

        if dist < 60 and 3 not in stack:
            dropped += 1
            continue

        valid_rgb.append(real_rgb)
        valid_stacks.append(stack)

    return np.array(valid_rgb), np.array(valid_stacks), f"✅ LUT已加载 (过滤了{dropped}个异常点)"


def create_slab_mesh(voxel_matrix, mat_id, height):
    """Generate optimized mesh from voxel data."""
    vertices, faces = [], []
    shrink = 0.05

    for z in range(voxel_matrix.shape[0]):
        z_bottom, z_top = z, z + 1
        mask = (voxel_matrix[z] == mat_id)
        if not np.any(mask):
            continue

        for y in range(height):
            world_y = (height - 1 - y)
            row = mask[y]
            padded = np.pad(row, (1, 1), mode='constant')
            diff = np.diff(padded.astype(int))
            starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]

            for start, end in zip(starts, ends):
                x0, x1 = start + shrink, end - shrink
                y0, y1 = world_y + shrink, world_y + 1 - shrink

                base_idx = len(vertices)
                vertices.extend([
                    [x0, y0, z_bottom], [x1, y0, z_bottom], [x1, y1, z_bottom], [x0, y1, z_bottom],
                    [x0, y0, z_top], [x1, y0, z_top], [x1, y1, z_top], [x0, y1, z_top]
                ])
                cube_faces = [
                    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                    [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                    [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]
                ]
                faces.extend([[v + base_idx for v in f] for f in cube_faces])

    if not vertices:
        return None
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    return mesh


def create_preview_mesh(matched_rgb, mask_solid, total_layers):
    """
    Create a colored preview mesh using the actual matched colors.
    Each pixel becomes a colored column with its LUT-matched color.
    """
    height, width = matched_rgb.shape[:2]
    vertices = []
    faces = []
    face_colors = []

    shrink = 0.05

    for y in range(height):
        for x in range(width):
            if not mask_solid[y, x]:
                continue

            # Get the matched color for this pixel
            rgb = matched_rgb[y, x]
            rgba = [int(rgb[0]), int(rgb[1]), int(rgb[2]), 255]

            # Create a column for this pixel
            world_y = (height - 1 - y)
            x0, x1 = x + shrink, x + 1 - shrink
            y0, y1 = world_y + shrink, world_y + 1 - shrink
            z0, z1 = 0, total_layers

            base_idx = len(vertices)
            vertices.extend([
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
            ])

            # 12 triangles for a cube (6 faces × 2 triangles)
            cube_faces = [
                [0, 2, 1], [0, 3, 2],  # bottom
                [4, 5, 6], [4, 6, 7],  # top
                [0, 1, 5], [0, 5, 4],  # front
                [1, 2, 6], [1, 6, 5],  # right
                [2, 3, 7], [2, 7, 6],  # back
                [3, 0, 4], [3, 4, 7]   # left
            ]

            for f in cube_faces:
                faces.append([v + base_idx for v in f])
                face_colors.append(rgba)

    if not vertices:
        return None

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.face_colors = np.array(face_colors, dtype=np.uint8)
    return mesh


def convert_image_to_3d(image_path, lut_path, target_width_mm, spacer_thick,
                         structure_mode, auto_bg, bg_tol, color_mode,
                         add_loop, loop_width, loop_length, loop_hole, loop_pos):
    """Main image conversion pipeline with optional keychain loop.

    Args:
        loop_pos: 挂孔位置元组 (x, y) 像素坐标，或 None 表示自动放置
    """
    if image_path is None:
        return None, None, None, "❌ 请上传图片"
    if lut_path is None:
        return None, None, None, "⚠️ 请上传 .npy 校准文件！"

    # Get color configuration based on mode
    color_conf = ColorSystem.get(color_mode)

    # Load LUT
    lut_rgb, ref_stacks, msg = load_calibrated_lut(lut_path.name)
    if lut_rgb is None:
        return None, None, None, msg
    tree = KDTree(lut_rgb)

    # Image preprocessing
    img = Image.open(image_path).convert('RGBA')
    target_w = int(target_width_mm / PrinterConfig.NOZZLE_WIDTH)
    target_h = int(target_w * img.height / img.width)

    img = img.resize((target_w, target_h), Image.Resampling.NEAREST)
    img_arr = np.array(img)
    rgb_arr, alpha_arr = img_arr[:, :, :3], img_arr[:, :, 3]

    # Color matching
    flat_rgb = rgb_arr.reshape(-1, 3)
    _, indices = tree.query(flat_rgb)

    matched_rgb = lut_rgb[indices].reshape(target_h, target_w, 3)
    best_stacks = ref_stacks[indices].reshape(target_h, target_w, PrinterConfig.COLOR_LAYERS)

    # Transparency handling
    mask_transparent = alpha_arr < 10
    if auto_bg:
        bg_color = rgb_arr[0, 0]
        diff = np.sum(np.abs(rgb_arr - bg_color), axis=-1)
        mask_transparent = np.logical_or(mask_transparent, diff < bg_tol)

    best_stacks[mask_transparent] = -1

    # Preview
    preview_rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    mask_solid = ~mask_transparent
    preview_rgba[mask_solid, :3] = matched_rgb[mask_solid]
    preview_rgba[mask_solid, 3] = 255

    # 挂孔相关变量
    loop_info = None
    loop_color_id = 0  # 默认白色

    print(f"[DEBUG] add_loop={add_loop}, loop_pos={loop_pos}, loop_width={loop_width}, loop_length={loop_length}, loop_hole={loop_hole}")

    if add_loop:
        # 确定挂孔连接位置
        solid_rows = np.any(mask_solid, axis=1)
        if np.any(solid_rows):
            # 检查是否有用户点击的位置
            if loop_pos is not None and len(loop_pos) == 2:
                # 使用用户点击的位置 (注意：预览图的坐标需要缩放)
                click_x, click_y = loop_pos

                # 预览图可能被缩放过，需要根据实际图像大小换算
                # 这里click_x, click_y是在预览图上的像素坐标
                # 假设预览图已经是target_w x target_h大小
                attach_col = int(click_x)
                attach_row = int(click_y)

                # 限制范围
                attach_col = max(0, min(target_w - 1, attach_col))
                attach_row = max(0, min(target_h - 1, attach_row))

                # 找到该列最近的实体像素
                col_mask = mask_solid[:, attach_col]
                if np.any(col_mask):
                    solid_rows_in_col = np.where(col_mask)[0]
                    # 找到点击位置附近最近的实体像素
                    distances = np.abs(solid_rows_in_col - attach_row)
                    nearest_idx = np.argmin(distances)
                    top_row = solid_rows_in_col[nearest_idx]
                else:
                    # 该列没有实体，使用最近的有实体的列
                    top_row = np.argmax(solid_rows)
                    solid_cols_in_top = np.where(mask_solid[top_row])[0]
                    if len(solid_cols_in_top) > 0:
                        distances = np.abs(solid_cols_in_top - attach_col)
                        nearest_idx = np.argmin(distances)
                        attach_col = solid_cols_in_top[nearest_idx]
            else:
                # 使用默认位置：模型顶部中心
                top_row = np.argmax(solid_rows)
                solid_cols_in_top = np.where(mask_solid[top_row])[0]
                if len(solid_cols_in_top) > 0:
                    attach_col = int(np.mean(solid_cols_in_top))
                else:
                    attach_col = target_w // 2

            attach_col = max(0, min(target_w - 1, attach_col))

            # 自动检测挂孔位置附近的颜色
            search_area = best_stacks[max(0, top_row-2):top_row+3,
                                     max(0, attach_col-3):attach_col+4]
            search_area = search_area[search_area >= 0]  # 排除透明
            if len(search_area) > 0:
                # 找最常见的非白色材料
                unique, counts = np.unique(search_area, return_counts=True)
                for mat_id in unique[np.argsort(-counts)]:
                    if mat_id != 0:  # 不是白色
                        loop_color_id = int(mat_id)
                        break

            # 保存挂孔信息用于3D生成
            loop_info = {
                'attach_x_mm': attach_col * PrinterConfig.NOZZLE_WIDTH,
                'attach_y_mm': (target_h - 1 - top_row) * PrinterConfig.NOZZLE_WIDTH,
                'width_mm': loop_width,
                'length_mm': loop_length,
                'hole_dia_mm': loop_hole,
                'color_id': loop_color_id
            }

            # 在2D预览中绘制挂孔
            from PIL import ImageDraw
            preview_pil = Image.fromarray(preview_rgba, mode='RGBA')
            draw = ImageDraw.Draw(preview_pil)

            # 挂孔颜色
            loop_color_rgba = tuple(color_conf['preview'][loop_color_id][:3]) + (255,)

            # 计算挂孔在预览中的位置（像素坐标）
            loop_w_px = int(loop_width / PrinterConfig.NOZZLE_WIDTH)
            loop_h_px = int(loop_length / PrinterConfig.NOZZLE_WIDTH)
            hole_r_px = int(loop_hole / 2 / PrinterConfig.NOZZLE_WIDTH)
            circle_r_px = loop_w_px // 2

            # 挂孔位置（顶部在top_row上方）
            loop_bottom = top_row
            loop_top = top_row - loop_h_px
            loop_left = attach_col - loop_w_px // 2
            loop_right = attach_col + loop_w_px // 2

            # 矩形部分高度
            rect_h_px = loop_h_px - circle_r_px
            rect_bottom = loop_bottom
            rect_top = loop_bottom - rect_h_px

            # 圆心位置
            circle_center_y = rect_top
            circle_center_x = attach_col

            # 绘制矩形部分
            if rect_h_px > 0:
                draw.rectangle([loop_left, rect_top, loop_right, rect_bottom], fill=loop_color_rgba)

            # 绘制圆形顶部
            draw.ellipse([circle_center_x - circle_r_px, circle_center_y - circle_r_px,
                          circle_center_x + circle_r_px, circle_center_y + circle_r_px],
                         fill=loop_color_rgba)

            # 绘制孔（透明）
            hole_center_y = circle_center_y
            draw.ellipse([circle_center_x - hole_r_px, hole_center_y - hole_r_px,
                          circle_center_x + hole_r_px, hole_center_y + hole_r_px],
                         fill=(0, 0, 0, 0))

            preview_rgba = np.array(preview_pil)

    preview_img = Image.fromarray(preview_rgba, mode='RGBA')

    # Voxel construction
    bottom_voxels = np.transpose(best_stacks, (2, 0, 1))
    spacer_layers = max(1, int(round(spacer_thick / PrinterConfig.LAYER_HEIGHT)))

    if "双面" in structure_mode:
        top_voxels = np.transpose(best_stacks[..., ::-1], (2, 0, 1))
        total_layers = 5 + spacer_layers + 5
        full_matrix = np.full((total_layers, target_h, target_w), -1, dtype=int)
        full_matrix[0:5] = bottom_voxels

        spacer = np.full((target_h, target_w), -1, dtype=int)
        spacer[~mask_transparent] = 0
        for z in range(5, 5 + spacer_layers):
            full_matrix[z] = spacer
        full_matrix[5 + spacer_layers:] = top_voxels
    else:
        total_layers = 5 + spacer_layers
        full_matrix = np.full((total_layers, target_h, target_w), -1, dtype=int)
        full_matrix[0:5] = bottom_voxels

        spacer = np.full((target_h, target_w), -1, dtype=int)
        spacer[~mask_transparent] = 0
        for z in range(5, total_layers):
            full_matrix[z] = spacer

    # Mesh generation
    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[0, 0] = PrinterConfig.NOZZLE_WIDTH
    transform[1, 1] = PrinterConfig.NOZZLE_WIDTH
    transform[2, 2] = PrinterConfig.LAYER_HEIGHT

    # Use colors and names from the selected color mode
    preview_colors = color_conf['preview']
    slot_names = color_conf['slots']

    for mat_id in range(4):
        mesh = create_slab_mesh(full_matrix, mat_id, target_h)
        if mesh:
            mesh.apply_transform(transform)
            mesh.visual.face_colors = preview_colors[mat_id]
            mesh.metadata['name'] = slot_names[mat_id]
            scene.add_geometry(mesh, node_name=slot_names[mat_id], geom_name=slot_names[mat_id])

    # 添加挂孔
    loop_added = False
    print(f"[DEBUG] Before loop creation: add_loop={add_loop}, loop_info={loop_info}")
    if add_loop and loop_info is not None:
        try:
            # 计算挂孔厚度（与模型相同）
            loop_thickness = total_layers * PrinterConfig.LAYER_HEIGHT
            print(f"[DEBUG] Creating loop mesh with thickness={loop_thickness}")

            loop_mesh = create_keychain_loop(
                width_mm=loop_info['width_mm'],
                length_mm=loop_info['length_mm'],
                hole_dia_mm=loop_info['hole_dia_mm'],
                thickness_mm=loop_thickness,
                attach_x_mm=loop_info['attach_x_mm'],
                attach_y_mm=loop_info['attach_y_mm']
            )

            print(f"[DEBUG] loop_mesh created: {loop_mesh is not None}")

            if loop_mesh is not None:
                loop_mesh.visual.face_colors = preview_colors[loop_info['color_id']]
                loop_mesh.metadata['name'] = "Keychain_Loop"
                scene.add_geometry(loop_mesh, node_name="Keychain_Loop", geom_name="Keychain_Loop")
                slot_names_with_loop = slot_names + ["Keychain_Loop"]
                loop_added = True
                print(f"[DEBUG] Loop added to scene successfully")
        except Exception as e:
            print(f"挂孔创建失败: {e}")
            import traceback
            traceback.print_exc()

    # Export 3MF for printing
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(tempfile.gettempdir(), f"{base_name}_Lumina.3mf")
    scene.export(out_path)

    # Create colored preview mesh using actual matched colors
    preview_mesh = create_preview_mesh(matched_rgb, mask_solid, total_layers)

    if preview_mesh:
        # 先对preview_mesh应用transform（从像素转为mm）
        preview_mesh.apply_transform(transform)

    # 如果有挂孔，也添加到预览mesh中
    print(f"[DEBUG] preview_mesh={preview_mesh is not None}, loop_added={loop_added}, loop_info={loop_info is not None}")
    if preview_mesh and loop_added and loop_info is not None:
        try:
            # 创建预览用的挂孔（已经是mm单位，不需要transform）
            loop_thickness = total_layers * PrinterConfig.LAYER_HEIGHT
            preview_loop = create_keychain_loop(
                width_mm=loop_info['width_mm'],
                length_mm=loop_info['length_mm'],
                hole_dia_mm=loop_info['hole_dia_mm'],
                thickness_mm=loop_thickness,
                attach_x_mm=loop_info['attach_x_mm'],
                attach_y_mm=loop_info['attach_y_mm']
            )
            print(f"[DEBUG] preview_loop created: {preview_loop is not None}")
            if preview_loop is not None:
                # 设置挂孔颜色
                loop_color = preview_colors[loop_info['color_id']]
                preview_loop.visual.face_colors = [loop_color] * len(preview_loop.faces)

                # 合并mesh（两者都已经是mm单位）
                preview_mesh = trimesh.util.concatenate([preview_mesh, preview_loop])
                print(f"[DEBUG] preview_mesh merged with loop")
        except Exception as e:
            print(f"预览挂孔创建失败: {e}")
            import traceback
            traceback.print_exc()

    if preview_mesh:
        glb_path = os.path.join(tempfile.gettempdir(), f"{base_name}_Preview.glb")
        preview_mesh.export(glb_path)
    else:
        glb_path = None

    # Fix object names in 3MF for better slicer compatibility
    names_to_fix = slot_names_with_loop if loop_added else slot_names
    _safe_fix_3mf_names(out_path, names_to_fix)

    Stats.increment("conversions")

    # 构建返回消息
    msg = f"✅ 转换完成！分辨率: {target_w}×{target_h}px | 已组合为一个对象"
    if loop_added:
        msg += f" | 挂孔: {slot_names[loop_info['color_id']]}"

    return out_path, glb_path, preview_img, msg


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              UI LAYOUT                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

CUSTOM_CSS = """
/* Global Theme */
.gradio-container {
    max-width: 1400px !important;
    margin: auto;
}

/* Header Styling */
.header-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px 30px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
}

.header-banner h1 {
    color: white !important;
    font-size: 2.5em !important;
    margin: 0 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.header-banner p {
    color: rgba(255,255,255,0.9) !important;
    margin: 5px 0 0 0 !important;
}

/* Stats Bar */
.stats-bar {
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
    padding: 12px 20px;
    border-radius: 10px;
    color: #a0a0ff;
    font-family: 'Courier New', monospace;
    text-align: center;
    margin-bottom: 15px;
}

/* Tab Styling */
.tab-nav button {
    font-size: 1.1em !important;
    padding: 12px 24px !important;
    border-radius: 10px 10px 0 0 !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

/* Card Styling */
.input-card, .output-card {
    background: #fafafa;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #e0e0e0;
}

/* Button Styling */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-size: 1.1em !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
}

/* Mode indicator */
.mode-indicator {
    background: #f0f0ff;
    border: 2px solid #667eea;
    border-radius: 8px;
    padding: 10px;
    margin: 10px 0;
    font-weight: bold;
}

/* Language Indicator */
.lang-indicator {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: bold;
}

/* Footer */
.footer {
    text-align: center;
    padding: 20px;
    color: #888;
    font-size: 0.9em;
}
"""


def create_app():
    with gr.Blocks(title="Lumina Studio", css=CUSTOM_CSS, theme=gr.themes.Soft()) as app:

        # Header with Language Indicator
        with gr.Row():
            with gr.Column(scale=10):
                gr.HTML("""
                <div class="header-banner">
                    <h1>✨ Lumina Studio</h1>
                    <p>多材料3D打印色彩系统 | Multi-Material 3D Print Color System | v1.3</p>
                </div>
                """)
            with gr.Column(scale=1, min_width=120):
                gr.HTML("""
                <div style="text-align:right; padding:10px;">
                    <span style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                 color:white; padding:5px 15px; border-radius:20px; font-weight:bold;">
                        🌐 中文 | EN
                    </span>
                </div>
                """)

        # Stats Bar
        stats = Stats.get_all()
        stats_html = gr.HTML(f"""
        <div class="stats-bar">
            📊 累计生成 Total: 
            <strong>{stats.get('calibrations', 0)}</strong> 校准板 Calibrations | 
            <strong>{stats.get('extractions', 0)}</strong> 颜色提取 Extractions | 
            <strong>{stats.get('conversions', 0)}</strong> 模型转换 Conversions
        </div>
        """)

        # Main Tabs
        with gr.Tabs() as tabs:

            # ═══════════════════════════════════════════════════════════════
            # TAB 1: Calibration Generator
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("📐 校准板 Calibration", id=0):
                cal_desc = gr.Markdown("""
                ### 第一步：生成校准板 | Step 1: Generate Calibration Board
                生成1024种颜色的校准板，打印后用于提取打印机的实际色彩数据。
                Generate a 1024-color calibration board to extract your printer's actual color data.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### ⚙️ 参数 Parameters")
                        cal_mode = gr.Radio(
                            choices=["CMYW (Cyan/Magenta/Yellow)", "RYBW (Red/Yellow/Blue)"],
                            value="RYBW (Red/Yellow/Blue)",
                            label="色彩模式 Color Mode"
                        )
                        cal_block_size = gr.Slider(3, 10, 5, step=1, label="色块尺寸 Block Size (mm)")
                        cal_gap = gr.Slider(0.4, 2.0, 0.82, step=0.02, label="间隙 Gap (mm)")
                        cal_backing = gr.Dropdown(
                            choices=["White", "Cyan", "Magenta", "Yellow", "Red", "Blue"],
                            value="White",
                            label="底板颜色 Backing Color"
                        )
                        cal_btn = gr.Button("🚀 生成 Generate", variant="primary", elem_classes=["primary-btn"])
                        cal_log = gr.Textbox(label="状态 Status", interactive=False)

                    with gr.Column(scale=1):
                        gr.Markdown("#### 👁️ 预览 Preview")
                        cal_preview = gr.Image(label="Calibration Preview", show_label=False)
                        cal_file = gr.File(label="下载 Download 3MF")

                cal_btn.click(
                    generate_calibration_board,
                    inputs=[cal_mode, cal_block_size, cal_gap, cal_backing],
                    outputs=[cal_file, cal_preview, cal_log]
                )

            # ═══════════════════════════════════════════════════════════════
            # TAB 2: Color Extractor
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("🎨 颜色提取 Extractor", id=1):
                gr.Markdown("""
                ### 第二步：提取颜色数据 | Step 2: Extract Color Data
                拍摄打印好的校准板照片，提取真实的色彩数据生成 LUT 文件。
                Take a photo of your printed calibration board to extract real color data.
                """)

                ext_state_img = gr.State(None)
                ext_state_pts = gr.State([])
                ext_curr_coord = gr.State(None)
                ref_img = generate_simulated_reference()

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📸 上传照片 Upload Photo")

                        ext_color_mode = gr.Radio(
                            choices=["CMYW (Cyan/Magenta/Yellow)", "RYBW (Red/Yellow/Blue)"],
                            value="RYBW (Red/Yellow/Blue)",
                            label="🎨 色彩模式 Color Mode"
                        )

                        ext_img_in = gr.Image(label="校准板照片 Calibration Photo", type="numpy", interactive=True)

                        with gr.Row():
                            ext_rot_btn = gr.Button("↺ 旋转 Rotate")
                            ext_clear_btn = gr.Button("🗑️ 重置 Reset")

                        gr.Markdown("#### 🔧 校正参数 Correction")
                        with gr.Row():
                            ext_wb = gr.Checkbox(label="自动白平衡 Auto WB", value=True)
                            ext_bf = gr.Checkbox(label="暗角校正 Vignette", value=False)

                        ext_zoom = gr.Slider(0.8, 1.2, 1.0, step=0.005, label="缩放 Zoom")
                        ext_barrel = gr.Slider(-0.2, 0.2, 0.0, step=0.01, label="畸变 Distortion")
                        ext_off_x = gr.Slider(-30, 30, 0, step=1, label="X偏移 Offset X")
                        ext_off_y = gr.Slider(-30, 30, 0, step=1, label="Y偏移 Offset Y")

                        ext_run_btn = gr.Button("🚀 提取 Extract", variant="primary", elem_classes=["primary-btn"])
                        ext_log = gr.Textbox(label="状态 Status", interactive=False)

                    with gr.Column(scale=1):
                        ext_hint = gr.Markdown("#### 👉 点击 Click: **White (左上 Top-Left)**")
                        ext_work_img = gr.Image(label="标记图 Marked", show_label=False, interactive=True)

                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### 📍 采样预览 Sampling")
                                ext_warp_view = gr.Image(show_label=False)
                            with gr.Column():
                                gr.Markdown("#### 🎯 参考 Reference")
                                ext_ref_view = gr.Image(show_label=False, value=ref_img, interactive=False)

                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### 📊 结果 Result (点击修正 Click to fix)")
                                ext_lut_view = gr.Image(show_label=False, interactive=True)
                            with gr.Column():
                                gr.Markdown("#### 🛠️ 手动修正 Manual Fix")
                                ext_probe_html = gr.HTML("点击左侧色块 Click cell on left...")
                                ext_picker = gr.ColorPicker(label="替换颜色 Override", value="#FF0000")
                                ext_fix_btn = gr.Button("🔧 应用 Apply")
                                ext_dl_btn = gr.File(label="下载 Download .npy")

                # 根据模式获取定位点顺序的函数（双语）
                def get_first_hint(mode):
                    conf = ColorSystem.get(mode)
                    label_zh = conf['corner_labels'][0]
                    label_en = conf['corner_labels_en'][0]
                    return f"#### 👉 点击 Click: **{label_zh} / {label_en}**"

                def get_next_hint(mode, pts_count):
                    conf = ColorSystem.get(mode)
                    if pts_count >= 4:
                        return "#### ✅ 定位完成！Ready to extract!"
                    label_zh = conf['corner_labels'][pts_count]
                    label_en = conf['corner_labels_en'][pts_count]
                    return f"#### 👉 点击 Click: **{label_zh} / {label_en}**"

                # Event handlers for extractor
                def on_upload(i, mode):
                    hint = get_first_hint(mode)
                    return i, i, [], None, hint

                ext_img_in.upload(
                    on_upload,
                    [ext_img_in, ext_color_mode],
                    [ext_state_img, ext_work_img, ext_state_pts, ext_curr_coord, ext_hint]
                )

                def on_mode_change(img, mode):
                    hint = get_first_hint(mode)
                    return [], hint, img

                ext_color_mode.change(
                    on_mode_change,
                    [ext_state_img, ext_color_mode],
                    [ext_state_pts, ext_hint, ext_work_img]
                )

                def on_rotate(i, mode):
                    if i is None:
                        return None, None, [], get_first_hint(mode)
                    r = rotate_image(i, "左旋 90°")
                    return r, r, [], get_first_hint(mode)

                ext_rot_btn.click(
                    on_rotate,
                    [ext_state_img, ext_color_mode],
                    [ext_state_img, ext_work_img, ext_state_pts, ext_hint]
                )

                def on_click(img, pts, mode, evt: gr.SelectData):
                    if len(pts) >= 4:
                        return img, pts, "#### ✅ 定位完成 Complete!"
                    n = pts + [[evt.index[0], evt.index[1]]]
                    vis = draw_corner_points(img, n, mode)
                    hint = get_next_hint(mode, len(n))
                    return vis, n, hint

                ext_work_img.select(
                    on_click,
                    [ext_state_img, ext_state_pts, ext_color_mode],
                    [ext_work_img, ext_state_pts, ext_hint]
                )

                def on_clear(img, mode):
                    hint = get_first_hint(mode)
                    return img, [], hint

                ext_clear_btn.click(
                    on_clear,
                    [ext_state_img, ext_color_mode],
                    [ext_work_img, ext_state_pts, ext_hint]
                )

                extract_inputs = [ext_state_img, ext_state_pts, ext_off_x, ext_off_y,
                                  ext_zoom, ext_barrel, ext_wb, ext_bf]
                extract_outputs = [ext_warp_view, ext_lut_view, ext_dl_btn, ext_log]

                ext_run_btn.click(run_extraction, extract_inputs, extract_outputs)

                for s in [ext_off_x, ext_off_y, ext_zoom, ext_barrel]:
                    s.release(run_extraction, extract_inputs, extract_outputs)

                ext_lut_view.select(probe_lut_cell, [], [ext_probe_html, ext_picker, ext_curr_coord])
                ext_fix_btn.click(manual_fix_cell, [ext_curr_coord, ext_picker], [ext_lut_view, ext_log])

            # ═══════════════════════════════════════════════════════════════
            # TAB 3: Image Converter
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("💎 图像转换 Converter", id=2):
                gr.Markdown("""
                ### 第三步：转换图像 | Step 3: Convert Image
                **流程**: 设置参数 → 预览 → 点击图片放置挂孔(暂不推荐使用) → 调整参数 → 生成
                """)

                # 状态变量
                conv_loop_pos = gr.State(None)  # 挂孔位置 (x, y)
                conv_preview_cache = gr.State(None)  # 缓存预览数据

                with gr.Row():
                    # 左侧：输入和参数
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📁 输入")
                        conv_lut = gr.File(label="校准数据 (.npy)", file_types=['.npy'])
                        conv_img = gr.Image(label="输入图像", type="filepath")

                        gr.Markdown("#### ⚙️ 参数")
                        conv_color_mode = gr.Radio(
                            choices=["CMYW (Cyan/Magenta/Yellow)", "RYBW (Red/Yellow/Blue)"],
                            value="RYBW (Red/Yellow/Blue)",
                            label="色彩模式"
                        )
                        conv_structure = gr.Radio(
                            ["双面 (钥匙扣)", "单面 (浮雕)"],
                            value="双面 (钥匙扣)",
                            label="结构"
                        )
                        with gr.Row():
                            conv_auto_bg = gr.Checkbox(label="移除背景", value=True)
                            conv_tol = gr.Slider(0, 150, 40, label="容差")
                        conv_width = gr.Slider(20, 150, 60, label="宽度 (mm)")
                        conv_thick = gr.Slider(0.2, 2.0, 1.2, step=0.08, label="背板 (mm)")

                        conv_preview_btn = gr.Button("👁️👁️ 生成预览", variant="secondary", size="lg")

                    # 中间：预览编辑区
                    with gr.Column(scale=2):
                        gr.Markdown("#### 🎨 2D预览 - 点击图片放置挂孔位置（暂不推荐使用）")

                        # 预览图 - 不可交互上传，但可点击
                        conv_preview = gr.Image(
                            label="",
                            type="numpy",
                            height=500,
                            interactive=False,  # 禁止拖拽上传
                            show_label=False
                        )

                        # 挂孔设置
                        with gr.Group():
                            gr.Markdown("##### 🔗 挂孔设置")
                            with gr.Row():
                                conv_add_loop = gr.Checkbox(label="启用挂孔", value=False)
                                conv_remove_loop = gr.Button("🗑️ 移除挂孔", size="sm")
                            with gr.Row():
                                conv_loop_width = gr.Slider(2, 10, 4, step=0.5, label="宽度(mm)")
                                conv_loop_length = gr.Slider(4, 15, 8, step=0.5, label="长度(mm)")
                                conv_loop_hole = gr.Slider(1, 5, 2.5, step=0.25, label="孔径(mm)")
                            with gr.Row():
                                conv_loop_angle = gr.Slider(-180, 180, 0, step=5, label="旋转角度°")
                                conv_loop_info = gr.Textbox(label="挂孔位置", interactive=False, scale=2)

                        conv_log = gr.Textbox(label="状态", lines=1, interactive=False)

                    # 右侧：输出
                    with gr.Column(scale=1):
                        conv_btn = gr.Button("🚀 生成3MF", variant="primary", size="lg")
                        gr.Markdown("#### 🎮 3D预览")
                        conv_3d_preview = gr.Model3D(
                            label="3D",
                            clear_color=[0.9, 0.9, 0.9, 1.0],
                            height=280
                        )
                        gr.Markdown("#### 📁 下载【务必合并对象后再切片】")
                        conv_file = gr.File(label="3MF文件")

                # ===== 核心函数 =====
                PREVIEW_SCALE = 2  # 固定预览缩放倍数
                PREVIEW_MARGIN = 30  # 预览图边距（显示坐标轴用）

                def generate_preview_cached(image_path, lut_path, target_width_mm,
                                           auto_bg, bg_tol, color_mode):
                    """生成预览并缓存数据"""
                    if image_path is None:
                        return None, None, "❌ 请上传图片"
                    if lut_path is None:
                        return None, None, "⚠️ 请上传校准文件"

                    color_conf = ColorSystem.get(color_mode)
                    lut_rgb, ref_stacks, msg = load_calibrated_lut(lut_path.name)
                    if lut_rgb is None:
                        return None, None, msg
                    tree = KDTree(lut_rgb)

                    img = Image.open(image_path).convert('RGBA')
                    target_w = int(target_width_mm / PrinterConfig.NOZZLE_WIDTH)
                    target_h = int(target_w * img.height / img.width)

                    img = img.resize((target_w, target_h), Image.Resampling.NEAREST)
                    img_arr = np.array(img)
                    rgb_arr, alpha_arr = img_arr[:, :, :3], img_arr[:, :, 3]

                    flat_rgb = rgb_arr.reshape(-1, 3)
                    _, indices = tree.query(flat_rgb)
                    matched_rgb = lut_rgb[indices].reshape(target_h, target_w, 3)
                    best_stacks = ref_stacks[indices].reshape(target_h, target_w, PrinterConfig.COLOR_LAYERS)

                    mask_transparent = alpha_arr < 10
                    if auto_bg:
                        bg_color = rgb_arr[0, 0]
                        diff = np.sum(np.abs(rgb_arr - bg_color), axis=-1)
                        mask_transparent = np.logical_or(mask_transparent, diff < bg_tol)

                    mask_solid = ~mask_transparent

                    # 创建预览图
                    preview_rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)
                    preview_rgba[mask_solid, :3] = matched_rgb[mask_solid]
                    preview_rgba[mask_solid, 3] = 255

                    # 缓存数据
                    cache = {
                        'target_w': target_w, 'target_h': target_h,
                        'mask_solid': mask_solid, 'best_stacks': best_stacks,
                        'matched_rgb': matched_rgb, 'preview_rgba': preview_rgba.copy(),
                        'color_conf': color_conf
                    }

                    # 缩放显示
                    display = render_preview(preview_rgba, None, 0, 0, 0, 0, False, color_conf)

                    return display, cache, f"✅ 预览 ({target_w}×{target_h}px) | 点击图片放置挂孔"

                def render_preview(preview_rgba, loop_pos, loop_width, loop_length, loop_hole, loop_angle, loop_enabled, color_conf):
                    """渲染带挂孔和坐标网格的预览图"""
                    from PIL import ImageDraw, ImageFont

                    h, w = preview_rgba.shape[:2]
                    new_w, new_h = w * PREVIEW_SCALE, h * PREVIEW_SCALE

                    # 边距（用于显示坐标轴）
                    margin = PREVIEW_MARGIN
                    canvas_w = new_w + margin
                    canvas_h = new_h + margin

                    # 创建带背景的画布
                    canvas = Image.new('RGBA', (canvas_w, canvas_h), (240, 240, 245, 255))
                    draw = ImageDraw.Draw(canvas)

                    # 绘制网格背景
                    grid_color = (220, 220, 225, 255)
                    grid_color_main = (200, 200, 210, 255)

                    # 网格间距（每10个像素一条线，每50个像素一条主线）
                    grid_step = 10 * PREVIEW_SCALE
                    main_step = 50 * PREVIEW_SCALE

                    # 绘制次网格线
                    for x in range(margin, canvas_w, grid_step):
                        draw.line([(x, margin), (x, canvas_h)], fill=grid_color, width=1)
                    for y in range(margin, canvas_h, grid_step):
                        draw.line([(margin, y), (canvas_w, y)], fill=grid_color, width=1)

                    # 绘制主网格线
                    for x in range(margin, canvas_w, main_step):
                        draw.line([(x, margin), (x, canvas_h)], fill=grid_color_main, width=1)
                    for y in range(margin, canvas_h, main_step):
                        draw.line([(margin, y), (canvas_w, y)], fill=grid_color_main, width=1)

                    # 绘制坐标轴
                    axis_color = (100, 100, 120, 255)
                    draw.line([(margin, margin), (margin, canvas_h)], fill=axis_color, width=2)  # Y轴
                    draw.line([(margin, canvas_h - 1), (canvas_w, canvas_h - 1)], fill=axis_color, width=2)  # X轴

                    # 绘制刻度标签
                    label_color = (80, 80, 100, 255)
                    try:
                        font = ImageFont.load_default()
                    except:
                        font = None

                    # X轴刻度（每50像素）
                    for i, x in enumerate(range(margin, canvas_w, main_step)):
                        px_value = i * 50
                        if font:
                            draw.text((x - 5, canvas_h - margin + 5), str(px_value), fill=label_color, font=font)

                    # Y轴刻度
                    for i, y in enumerate(range(margin, canvas_h, main_step)):
                        px_value = i * 50
                        if font:
                            draw.text((5, y - 5), str(px_value), fill=label_color, font=font)

                    # 缩放预览图
                    pil_img = Image.fromarray(preview_rgba, mode='RGBA')
                    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.NEAREST)

                    # 将预览图粘贴到画布上
                    canvas.paste(pil_img, (margin, 0), pil_img)

                    # 绘制挂孔
                    if loop_enabled and loop_pos is not None:
                        canvas = draw_loop_on_image(canvas, loop_pos, loop_width, loop_length, loop_hole, loop_angle, color_conf, margin)

                    return np.array(canvas)

                def draw_loop_on_image(pil_img, loop_pos, loop_width, loop_length, loop_hole, loop_angle, color_conf, margin=None):
                    """在图像上绘制挂孔"""
                    from PIL import ImageDraw

                    if margin is None:
                        margin = PREVIEW_MARGIN

                    # 计算像素尺寸（放大后）
                    loop_w_px = int(loop_width / PrinterConfig.NOZZLE_WIDTH * PREVIEW_SCALE)
                    loop_h_px = int(loop_length / PrinterConfig.NOZZLE_WIDTH * PREVIEW_SCALE)
                    hole_r_px = int(loop_hole / 2 / PrinterConfig.NOZZLE_WIDTH * PREVIEW_SCALE)
                    circle_r_px = loop_w_px // 2

                    # 挂孔位置（放大后的坐标，加上边距偏移）
                    cx = int(loop_pos[0] * PREVIEW_SCALE) + margin
                    cy = int(loop_pos[1] * PREVIEW_SCALE)

                    # 创建挂孔图层
                    loop_size = max(loop_w_px, loop_h_px) * 2 + 20
                    loop_layer = Image.new('RGBA', (loop_size, loop_size), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(loop_layer)

                    lc = loop_size // 2
                    rect_h = max(1, loop_h_px - circle_r_px)

                    # 挂孔颜色（红色便于识别）
                    loop_color = (220, 60, 60, 200)
                    outline_color = (255, 255, 255, 255)

                    # 矩形部分
                    draw.rectangle([lc - loop_w_px//2, lc, lc + loop_w_px//2, lc + rect_h],
                                  fill=loop_color, outline=outline_color, width=2)

                    # 圆形顶部
                    draw.ellipse([lc - circle_r_px, lc - circle_r_px,
                                 lc + circle_r_px, lc + circle_r_px],
                                fill=loop_color, outline=outline_color, width=2)

                    # 孔洞
                    draw.ellipse([lc - hole_r_px, lc - hole_r_px,
                                 lc + hole_r_px, lc + hole_r_px],
                                fill=(0, 0, 0, 0))

                    # 旋转
                    if loop_angle != 0:
                        loop_layer = loop_layer.rotate(-loop_angle, center=(lc, lc),
                                                       expand=False, resample=Image.BICUBIC)

                    # 粘贴
                    paste_x = cx - lc
                    paste_y = cy - lc - rect_h // 2
                    pil_img.paste(loop_layer, (paste_x, paste_y), loop_layer)

                    return pil_img

                def on_preview_click(cache, loop_pos, evt: gr.SelectData):
                    """点击预览图设置挂孔位置"""
                    if evt is None or cache is None:
                        return loop_pos, False, "点击无效 - 请先生成预览"

                    # 获取点击坐标（带margin的画布坐标）
                    click_x, click_y = evt.index

                    # 减去左边距，转换回图像坐标
                    click_x = click_x - PREVIEW_MARGIN

                    # 转换回原始坐标
                    orig_x = click_x / PREVIEW_SCALE
                    orig_y = click_y / PREVIEW_SCALE

                    # 限制范围
                    target_w = cache['target_w']
                    target_h = cache['target_h']
                    orig_x = max(0, min(target_w - 1, orig_x))
                    orig_y = max(0, min(target_h - 1, orig_y))

                    pos_info = f"位置: ({orig_x:.1f}, {orig_y:.1f}) px"
                    return (orig_x, orig_y), True, pos_info

                def update_preview_with_loop(cache, loop_pos, add_loop,
                                            loop_width, loop_length, loop_hole, loop_angle):
                    """更新带挂孔的预览"""
                    if cache is None:
                        return None

                    preview_rgba = cache['preview_rgba'].copy()
                    color_conf = cache['color_conf']

                    display = render_preview(
                        preview_rgba,
                        loop_pos if add_loop else None,
                        loop_width, loop_length, loop_hole, loop_angle,
                        add_loop, color_conf
                    )
                    return display

                def on_remove_loop():
                    """移除挂孔"""
                    return None, False, 0, "已移除挂孔"

                def generate_final_model(image_path, lut_path, target_width_mm, spacer_thick,
                                        structure_mode, auto_bg, bg_tol, color_mode,
                                        add_loop, loop_width, loop_length, loop_hole, loop_pos):
                    """生成最终3MF模型"""
                    return convert_image_to_3d(
                        image_path, lut_path, target_width_mm, spacer_thick,
                        structure_mode, auto_bg, bg_tol, color_mode,
                        add_loop, loop_width, loop_length, loop_hole, loop_pos
                    )

                # ===== 事件绑定 =====

                # 生成预览
                conv_preview_btn.click(
                    generate_preview_cached,
                    inputs=[conv_img, conv_lut, conv_width, conv_auto_bg, conv_tol, conv_color_mode],
                    outputs=[conv_preview, conv_preview_cache, conv_log]
                )

                # 点击预览图放置挂孔
                conv_preview.select(
                    on_preview_click,
                    inputs=[conv_preview_cache, conv_loop_pos],
                    outputs=[conv_loop_pos, conv_add_loop, conv_loop_info]
                ).then(
                    update_preview_with_loop,
                    inputs=[conv_preview_cache, conv_loop_pos, conv_add_loop,
                           conv_loop_width, conv_loop_length, conv_loop_hole, conv_loop_angle],
                    outputs=[conv_preview]
                )

                # 移除挂孔
                conv_remove_loop.click(
                    on_remove_loop,
                    outputs=[conv_loop_pos, conv_add_loop, conv_loop_angle, conv_loop_info]
                ).then(
                    update_preview_with_loop,
                    inputs=[conv_preview_cache, conv_loop_pos, conv_add_loop,
                           conv_loop_width, conv_loop_length, conv_loop_hole, conv_loop_angle],
                    outputs=[conv_preview]
                )

                # 挂孔参数变化时实时更新预览
                loop_params = [conv_loop_width, conv_loop_length, conv_loop_hole, conv_loop_angle]
                for param in loop_params:
                    param.change(
                        update_preview_with_loop,
                        inputs=[conv_preview_cache, conv_loop_pos, conv_add_loop,
                               conv_loop_width, conv_loop_length, conv_loop_hole, conv_loop_angle],
                        outputs=[conv_preview]
                    )

                # 生成最终模型
                conv_btn.click(
                    generate_final_model,
                    inputs=[conv_img, conv_lut, conv_width, conv_thick,
                            conv_structure, conv_auto_bg, conv_tol, conv_color_mode,
                            conv_add_loop, conv_loop_width, conv_loop_length, conv_loop_hole, conv_loop_pos],
                    outputs=[conv_file, conv_3d_preview, conv_preview, conv_log]
                )

            # ═══════════════════════════════════════════════════════════════
            # TAB 4: About
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("ℹ️ 关于 About", id=3):
                gr.Markdown("""
                ## 🌟 Lumina Studio v1.3
                
                **多材料3D打印色彩系统** | Multi-Material 3D Print Color System
                
                让FDM打印也能拥有精准的色彩还原 | Accurate color reproduction for FDM printing
                
                ---
                
                ### 📖 使用流程 Workflow
                
                1. **生成校准板 Generate Calibration** → 打印1024色校准网格 Print 1024-color grid
                2. **提取颜色 Extract Colors** → 拍照并提取打印机实际色彩 Photo → extract real colors
                3. **转换图像 Convert Image** → 将图片转为多层3D模型 Image → multi-layer 3D model
                
                ---
                
                ### 🎨 色彩模式定位点顺序 Color Mode Corner Order
                
                | 模式 Mode | 左上 TL | 右上 TR | 右下 BR | 左下 BL |
                |-----------|---------|---------|---------|---------|
                | **RYBW** | ⬜ White | 🟥 Red | 🟦 Blue | 🟨 Yellow |
                | **CMYW** | ⬜ White | 🔵 Cyan | 🟣 Magenta | 🟨 Yellow |
                
                ---
                
                ### 🔬 技术原理 Technology
                
                - **Beer-Lambert 光学混色** Optical Color Mixing
                - **KD-Tree 色彩匹配** Color Matching
                - **Integer Slab 几何优化** Geometry Optimization
                
                ---
                
                ### 📝 v1.3 更新日志 Changelog
                
                - ✅ **新增钥匙扣挂孔** Added keychain loop feature
                - ✅ 挂孔颜色自动检测 Auto-detect loop color from nearby pixels
                - ✅ 2D预览显示挂孔 2D preview shows loop
                - ✅ 修复3MF对象命名 Fixed 3MF object naming
                - ✅ 颜色提取/转换添加模式选择 Added color mode selection
                - ✅ 默认间隙改为0.82mm Default gap changed to 0.82mm
                - ✅ **新增3D实时预览** Added 3D preview with true colors
                
                ---
                
                ### 🚧 开发路线图 Roadmap
                
                - [✅] 4色基础模式 4-color base mode
                - [✅] 钥匙扣挂孔 Keychain loop
                - [ ] 6色扩展模式 6-color extended mode
                - [ ] 8色专业模式 8-color professional mode
                - [ ] 版画模式 Woodblock print mode
                - [ ] 拼豆模式 Perler bead mode
                
                ---
                
                ### 📄 许可证 License
                
                **CC BY-NC-SA 4.0** - Attribution-NonCommercial-ShareAlike
                
                ---
                
                <div style="text-align:center; color:#888; margin-top:20px;">
                    Made with ❤️ by [MIN]<br>
                    v1.3.0 | 2025
                </div>
                """)

        # Footer
        gr.HTML("""
        <div class="footer">
            <p>💡 提示 Tip: 使用高质量的PLA/PETG basic材料可获得最佳效果 | Use high-quality translucent PLA/PETG basic for best results</p>
        </div>
        """)

    return app


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN ENTRY                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    app = create_app()
    app.launch(
        inbrowser=True,
        server_port=7860,
        share=False,
        show_error=True
    )
