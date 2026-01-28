"""
Lumina Studio - Image Converter Coordinator (Refactored)
图像转换协调器 - 重构版本
负责协调各个模块完成图像到3D模型的转换
"""

import os
import tempfile
import numpy as np
import cv2
import trimesh
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

from config import PrinterConfig, ColorSystem, PREVIEW_SCALE, PREVIEW_MARGIN
from utils import Stats, safe_fix_3mf_names

# 导入重构后的模块
from core.image_processing import LuminaImageProcessor
from core.mesh_generators import get_mesher
from core.geometry_utils import create_keychain_loop


# ========== 主转换函数 ==========

def convert_image_to_3d(image_path, lut_path, target_width_mm, spacer_thick,
                         structure_mode, auto_bg, bg_tol, color_mode,
                         add_loop, loop_width, loop_length, loop_hole, loop_pos,
                         modeling_mode="vector", quantize_colors=32):
    """
    主转换函数：将图像转换为3D模型
    
    这是重构后的协调器函数，负责：
    1. 调用 LuminaImageProcessor 处理图像
    2. 调用 get_mesher 获取网格生成器
    3. 生成各材质的网格
    4. 添加挂孔（如果需要）
    5. 导出3MF文件
    """
    # 输入验证
    if image_path is None:
        return None, None, None, "❌ 请上传图片"
    if lut_path is None:
        return None, None, None, "⚠️ 请上传 .npy 校准文件！"
    
    print(f"[CONVERTER] Starting conversion...")
    print(f"[CONVERTER] Mode: {modeling_mode}, Quantize: {quantize_colors}")
    
    # 获取色彩配置
    color_conf = ColorSystem.get(color_mode)
    slot_names = color_conf['slots']
    preview_colors = color_conf['preview']
    
    # ========== 步骤1: 图像处理 ==========
    try:
        processor = LuminaImageProcessor(lut_path.name, color_mode)
        result = processor.process_image(
            image_path=image_path,
            target_width_mm=target_width_mm,
            modeling_mode=modeling_mode,
            quantize_colors=quantize_colors,
            auto_bg=auto_bg,
            bg_tol=bg_tol
        )
    except Exception as e:
        return None, None, None, f"❌ 图像处理失败: {e}"
    
    # 提取处理结果
    matched_rgb = result['matched_rgb']
    material_matrix = result['material_matrix']
    mask_solid = result['mask_solid']
    target_w, target_h = result['dimensions']
    pixel_scale = result['pixel_scale']
    mode_info = result['mode_info']
    
    print(f"[CONVERTER] Image processed: {target_w}×{target_h}px, scale={pixel_scale}mm/px")
    
    # ========== 步骤2: 生成预览图像 ==========
    preview_rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    preview_rgba[mask_solid, :3] = matched_rgb[mask_solid]
    preview_rgba[mask_solid, 3] = 255
    
    # ========== 步骤3: 处理挂孔 ==========
    loop_info = None
    if add_loop and loop_pos is not None:
        loop_info = _calculate_loop_info(
            loop_pos, loop_width, loop_length, loop_hole,
            mask_solid, material_matrix, target_w, target_h, pixel_scale
        )
        
        if loop_info:
            # 在预览图上绘制挂孔
            preview_rgba = _draw_loop_on_preview(
                preview_rgba, loop_info, color_conf, pixel_scale
            )
    
    preview_img = Image.fromarray(preview_rgba, mode='RGBA')
    
    # ========== 步骤4: 构建体素矩阵 ==========
    full_matrix = _build_voxel_matrix(
        material_matrix, mask_solid, spacer_thick, structure_mode
    )
    
    total_layers = full_matrix.shape[0]
    print(f"[CONVERTER] Voxel matrix: {full_matrix.shape} (Z×H×W)")
    
    # ========== 步骤5: 生成3D网格 ==========
    scene = trimesh.Scene()
    
    # 创建变换矩阵 (像素 → 毫米)
    transform = np.eye(4)
    transform[0, 0] = pixel_scale  # X
    transform[1, 1] = pixel_scale  # Y
    transform[2, 2] = PrinterConfig.LAYER_HEIGHT  # Z
    
    print(f"[CONVERTER] Transform: XY={pixel_scale}mm/px, Z={PrinterConfig.LAYER_HEIGHT}mm/layer")

    # 获取网格生成器
    mesher = get_mesher(modeling_mode)
    print(f"[CONVERTER] Using mesher: {mesher.__class__.__name__}")

    unique_materials = set(np.unique(material_matrix)) - {-1}
    unique_materials = sorted([m for m in unique_materials if m >= 0])

    # 为每种材质生成网格
    used_slot_names = []
    for mat_id in unique_materials:
        mesh = mesher.generate_mesh(full_matrix, mat_id, target_h)
        if mesh:
            mesh.apply_transform(transform)
            mesh.visual.face_colors = preview_colors[mat_id]
            mesh.metadata['name'] = slot_names[mat_id]
            scene.add_geometry(
                mesh, 
                node_name=slot_names[mat_id], 
                geom_name=slot_names[mat_id]
            )
            used_slot_names.append(slot_names[mat_id])
            print(f"[CONVERTER] Added mesh for {slot_names[mat_id]}")
    
    # ========== 步骤6: 添加挂孔 ==========
    loop_added = False
    slot_names_with_loop = used_slot_names[:]
    
    if add_loop and loop_info is not None:
        try:
            loop_thickness = total_layers * PrinterConfig.LAYER_HEIGHT
            loop_mesh = create_keychain_loop(
                width_mm=loop_info['width_mm'],
                length_mm=loop_info['length_mm'],
                hole_dia_mm=loop_info['hole_dia_mm'],
                thickness_mm=loop_thickness,
                attach_x_mm=loop_info['attach_x_mm'],
                attach_y_mm=loop_info['attach_y_mm']
            )
            
            if loop_mesh is not None:
                loop_mesh.visual.face_colors = preview_colors[loop_info['color_id']]
                loop_mesh.metadata['name'] = "Keychain_Loop"
                scene.add_geometry(
                    loop_mesh, 
                    node_name="Keychain_Loop", 
                    geom_name="Keychain_Loop"
                )
                slot_names_with_loop.append("Keychain_Loop")
                loop_added = True
                print(f"[CONVERTER] Loop added successfully")
        except Exception as e:
            print(f"[CONVERTER] Loop creation failed: {e}")
    
    # ========== 步骤7: 导出3MF ==========
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(tempfile.gettempdir(), f"{base_name}_Lumina.3mf")
    scene.export(out_path)
    
    # 修复3MF对象名称
    safe_fix_3mf_names(out_path, slot_names_with_loop)
    
    print(f"[CONVERTER] 3MF exported: {out_path}")
    
    # ========== 步骤8: 生成3D预览 ==========
    preview_mesh = _create_preview_mesh(matched_rgb, mask_solid, total_layers)
    
    if preview_mesh:
        preview_mesh.apply_transform(transform)
        
        # 添加挂孔到预览
        if loop_added and loop_info:
            try:
                preview_loop = create_keychain_loop(
                    width_mm=loop_info['width_mm'],
                    length_mm=loop_info['length_mm'],
                    hole_dia_mm=loop_info['hole_dia_mm'],
                    thickness_mm=loop_thickness,
                    attach_x_mm=loop_info['attach_x_mm'],
                    attach_y_mm=loop_info['attach_y_mm']
                )
                if preview_loop:
                    loop_color = preview_colors[loop_info['color_id']]
                    preview_loop.visual.face_colors = [loop_color] * len(preview_loop.faces)
                    preview_mesh = trimesh.util.concatenate([preview_mesh, preview_loop])
            except Exception as e:
                print(f"[CONVERTER] Preview loop failed: {e}")
    
    if preview_mesh:
        glb_path = os.path.join(tempfile.gettempdir(), f"{base_name}_Preview.glb")
        preview_mesh.export(glb_path)
    else:
        glb_path = None
    
    # ========== 步骤9: 生成状态消息 ==========
    Stats.increment("conversions")
    
    mode_name = mode_info['name']
    msg = f"✅ 转换完成 ({mode_name})！分辨率: {target_w}×{target_h}px"
    
    if loop_added:
        msg += f" | 挂孔: {slot_names[loop_info['color_id']]}"
    
    total_pixels = target_w * target_h
    if glb_path is None and total_pixels > 2_000_000:
        msg += " | ⚠️ 模型过大，已禁用3D预览"
    elif glb_path and total_pixels > 500_000:
        msg += " | ℹ️ 3D预览已简化"
    
    return out_path, glb_path, preview_img, msg



# ========== 辅助函数 ==========

def _calculate_loop_info(loop_pos, loop_width, loop_length, loop_hole,
                         mask_solid, material_matrix, target_w, target_h, pixel_scale):
    """计算挂孔信息"""
    solid_rows = np.any(mask_solid, axis=1)
    if not np.any(solid_rows):
        return None
    
    click_x, click_y = loop_pos
    attach_col = int(click_x)
    attach_row = int(click_y)
    attach_col = max(0, min(target_w - 1, attach_col))
    attach_row = max(0, min(target_h - 1, attach_row))
    
    # 找到最近的实体行
    col_mask = mask_solid[:, attach_col]
    if np.any(col_mask):
        solid_rows_in_col = np.where(col_mask)[0]
        distances = np.abs(solid_rows_in_col - attach_row)
        nearest_idx = np.argmin(distances)
        top_row = solid_rows_in_col[nearest_idx]
    else:
        top_row = np.argmax(solid_rows)
        solid_cols_in_top = np.where(mask_solid[top_row])[0]
        if len(solid_cols_in_top) > 0:
            distances = np.abs(solid_cols_in_top - attach_col)
            nearest_idx = np.argmin(distances)
            attach_col = solid_cols_in_top[nearest_idx]
        else:
            attach_col = target_w // 2
    
    attach_col = max(0, min(target_w - 1, attach_col))
    
    # 检测挂孔颜色
    loop_color_id = 0
    search_area = material_matrix[
        max(0, top_row-2):top_row+3,
        max(0, attach_col-3):attach_col+4
    ]
    search_area = search_area[search_area >= 0]
    if len(search_area) > 0:
        unique, counts = np.unique(search_area, return_counts=True)
        for mat_id in unique[np.argsort(-counts)]:
            if mat_id != 0:
                loop_color_id = int(mat_id)
                break
    
    return {
        'attach_x_mm': attach_col * pixel_scale,
        'attach_y_mm': (target_h - 1 - top_row) * pixel_scale,
        'width_mm': loop_width,
        'length_mm': loop_length,
        'hole_dia_mm': loop_hole,
        'color_id': loop_color_id
    }


def _draw_loop_on_preview(preview_rgba, loop_info, color_conf, pixel_scale):
    """在预览图上绘制挂孔"""
    preview_pil = Image.fromarray(preview_rgba, mode='RGBA')
    draw = ImageDraw.Draw(preview_pil)
    
    loop_color_rgba = tuple(color_conf['preview'][loop_info['color_id']][:3]) + (255,)
    
    # 转换为像素坐标
    attach_col = int(loop_info['attach_x_mm'] / pixel_scale)
    attach_row = int((preview_rgba.shape[0] - 1) - loop_info['attach_y_mm'] / pixel_scale)
    
    loop_w_px = int(loop_info['width_mm'] / pixel_scale)
    loop_h_px = int(loop_info['length_mm'] / pixel_scale)
    hole_r_px = int(loop_info['hole_dia_mm'] / 2 / pixel_scale)
    circle_r_px = loop_w_px // 2
    
    # 计算位置
    loop_bottom = attach_row
    loop_left = attach_col - loop_w_px // 2
    loop_right = attach_col + loop_w_px // 2
    
    rect_h_px = loop_h_px - circle_r_px
    rect_bottom = loop_bottom
    rect_top = loop_bottom - rect_h_px
    
    circle_center_y = rect_top
    circle_center_x = attach_col
    
    # 绘制矩形
    if rect_h_px > 0:
        draw.rectangle(
            [loop_left, rect_top, loop_right, rect_bottom], 
            fill=loop_color_rgba
        )
    
    # 绘制半圆
    draw.ellipse(
        [circle_center_x - circle_r_px, circle_center_y - circle_r_px,
         circle_center_x + circle_r_px, circle_center_y + circle_r_px],
        fill=loop_color_rgba
    )
    
    # 绘制孔洞
    draw.ellipse(
        [circle_center_x - hole_r_px, circle_center_y - hole_r_px,
         circle_center_x + hole_r_px, circle_center_y + hole_r_px],
        fill=(0, 0, 0, 0)
    )
    
    return np.array(preview_pil)


def _build_voxel_matrix(material_matrix, mask_solid, spacer_thick, structure_mode):
    """构建完整的体素矩阵"""
    target_h, target_w = material_matrix.shape[:2]
    mask_transparent = ~mask_solid
    
    # 转置材质矩阵 (H, W, Layers) → (Layers, H, W)
    bottom_voxels = np.transpose(material_matrix, (2, 0, 1))
    
    # 计算背板层数
    spacer_layers = max(1, int(round(spacer_thick / PrinterConfig.LAYER_HEIGHT)))
    
    if "双面" in structure_mode:
        # 双面模式
        top_voxels = np.transpose(material_matrix[..., ::-1], (2, 0, 1))
        total_layers = 5 + spacer_layers + 5
        full_matrix = np.full((total_layers, target_h, target_w), -1, dtype=int)
        
        # 底层
        full_matrix[0:5] = bottom_voxels
        
        # 背板
        spacer = np.full((target_h, target_w), -1, dtype=int)
        spacer[~mask_transparent] = 0
        for z in range(5, 5 + spacer_layers):
            full_matrix[z] = spacer
        
        # 顶层
        full_matrix[5 + spacer_layers:] = top_voxels
    else:
        # 单面模式
        total_layers = 5 + spacer_layers
        full_matrix = np.full((total_layers, target_h, target_w), -1, dtype=int)
        
        # 底层
        full_matrix[0:5] = bottom_voxels
        
        # 背板
        spacer = np.full((target_h, target_w), -1, dtype=int)
        spacer[~mask_transparent] = 0
        for z in range(5, total_layers):
            full_matrix[z] = spacer
    
    return full_matrix


def _create_preview_mesh(matched_rgb, mask_solid, total_layers):
    """
    创建简化的3D预览网格
    用于在浏览器中显示
    """
    height, width = matched_rgb.shape[:2]
    total_pixels = width * height
    
    DISABLE_THRESHOLD = 2_000_000
    SIMPLIFY_THRESHOLD = 500_000
    TARGET_PIXELS = 300_000
    
    # 检查是否需要禁用预览
    if total_pixels > DISABLE_THRESHOLD:
        print(f"[PREVIEW] Model too large ({total_pixels:,} pixels)")
        print(f"[PREVIEW] 3D preview disabled to prevent crash")
        return None
    
    # 检查是否需要降采样
    if total_pixels > SIMPLIFY_THRESHOLD:
        scale_factor = int(np.sqrt(total_pixels / TARGET_PIXELS))
        scale_factor = max(2, min(scale_factor, 16))
        
        print(f"[PREVIEW] Downsampling by {scale_factor}×")
        
        new_height = height // scale_factor
        new_width = width // scale_factor
        
        matched_rgb = cv2.resize(
            matched_rgb, (new_width, new_height), 
            interpolation=cv2.INTER_AREA
        )
        mask_solid = cv2.resize(
            mask_solid.astype(np.uint8), (new_width, new_height),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        height, width = new_height, new_width
        shrink = 0.05 * scale_factor
    else:
        shrink = 0.05
    
    # 生成体素网格
    vertices = []
    faces = []
    face_colors = []
    
    for y in range(height):
        for x in range(width):
            if not mask_solid[y, x]:
                continue
            
            rgb = matched_rgb[y, x]
            rgba = [int(rgb[0]), int(rgb[1]), int(rgb[2]), 255]
            
            world_y = (height - 1 - y)
            x0, x1 = x + shrink, x + 1 - shrink
            y0, y1 = world_y + shrink, world_y + 1 - shrink
            z0, z1 = 0, total_layers
            
            base_idx = len(vertices)
            vertices.extend([
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
            ])
            
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
    
    print(f"[PREVIEW] Generated: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    
    return mesh



# ========== 预览相关函数 ==========

def generate_preview_cached(image_path, lut_path, target_width_mm,
                            auto_bg, bg_tol, color_mode):
    """
    生成预览并缓存数据
    用于2D预览界面
    """
    if image_path is None:
        return None, None, "❌ 请上传图片"
    if lut_path is None:
        return None, None, "⚠️ 请上传校准文件"
    
    color_conf = ColorSystem.get(color_mode)
    
    # 使用简化的处理流程（像素模式）
    try:
        processor = LuminaImageProcessor(lut_path.name, color_mode)
        result = processor.process_image(
            image_path=image_path,
            target_width_mm=target_width_mm,
            modeling_mode="voxel",  # 预览使用像素模式
            quantize_colors=16,
            auto_bg=auto_bg,
            bg_tol=bg_tol
        )
    except Exception as e:
        return None, None, f"❌ 预览生成失败: {e}"
    
    matched_rgb = result['matched_rgb']
    material_matrix = result['material_matrix']
    mask_solid = result['mask_solid']
    target_w, target_h = result['dimensions']
    
    # 生成预览图像
    preview_rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    preview_rgba[mask_solid, :3] = matched_rgb[mask_solid]
    preview_rgba[mask_solid, 3] = 255
    
    # 缓存数据
    cache = {
        'target_w': target_w,
        'target_h': target_h,
        'mask_solid': mask_solid,
        'material_matrix': material_matrix,
        'matched_rgb': matched_rgb,
        'preview_rgba': preview_rgba.copy(),
        'color_conf': color_conf
    }
    
    # 渲染带网格的预览
    display = render_preview(
        preview_rgba, None, 0, 0, 0, 0, False, color_conf
    )
    
    return display, cache, f"✅ 预览 ({target_w}×{target_h}px) | 点击图片放置挂孔"


def render_preview(preview_rgba, loop_pos, loop_width, loop_length, 
                   loop_hole, loop_angle, loop_enabled, color_conf):
    """
    渲染带挂孔和坐标网格的预览图
    """
    h, w = preview_rgba.shape[:2]
    new_w, new_h = w * PREVIEW_SCALE, h * PREVIEW_SCALE
    
    margin = PREVIEW_MARGIN
    canvas_w = new_w + margin
    canvas_h = new_h + margin
    
    # 创建画布
    canvas = Image.new('RGBA', (canvas_w, canvas_h), (240, 240, 245, 255))
    draw = ImageDraw.Draw(canvas)
    
    # 绘制网格
    grid_color = (220, 220, 225, 255)
    grid_color_main = (200, 200, 210, 255)
    
    grid_step = 10 * PREVIEW_SCALE
    main_step = 50 * PREVIEW_SCALE
    
    # 细网格
    for x in range(margin, canvas_w, grid_step):
        draw.line([(x, margin), (x, canvas_h)], fill=grid_color, width=1)
    for y in range(margin, canvas_h, grid_step):
        draw.line([(margin, y), (canvas_w, y)], fill=grid_color, width=1)
    
    # 粗网格
    for x in range(margin, canvas_w, main_step):
        draw.line([(x, margin), (x, canvas_h)], fill=grid_color_main, width=1)
    for y in range(margin, canvas_h, main_step):
        draw.line([(margin, y), (canvas_w, y)], fill=grid_color_main, width=1)
    
    # 坐标轴
    axis_color = (100, 100, 120, 255)
    draw.line([(margin, margin), (margin, canvas_h)], fill=axis_color, width=2)
    draw.line([(margin, canvas_h - 1), (canvas_w, canvas_h - 1)], fill=axis_color, width=2)
    
    # 坐标标签
    label_color = (80, 80, 100, 255)
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    for i, x in enumerate(range(margin, canvas_w, main_step)):
        px_value = i * 50
        if font:
            draw.text((x - 5, canvas_h - margin + 5), str(px_value), 
                     fill=label_color, font=font)
    
    for i, y in enumerate(range(margin, canvas_h, main_step)):
        px_value = i * 50
        if font:
            draw.text((5, y - 5), str(px_value), fill=label_color, font=font)
    
    # 粘贴图像
    pil_img = Image.fromarray(preview_rgba, mode='RGBA')
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.NEAREST)
    canvas.paste(pil_img, (margin, 0), pil_img)
    
    # 绘制挂孔
    if loop_enabled and loop_pos is not None:
        canvas = _draw_loop_on_canvas(
            canvas, loop_pos, loop_width, loop_length, 
            loop_hole, loop_angle, color_conf, margin
        )
    
    return np.array(canvas)


def _draw_loop_on_canvas(pil_img, loop_pos, loop_width, loop_length, 
                         loop_hole, loop_angle, color_conf, margin):
    """在画布上绘制挂孔标记"""
    loop_w_px = int(loop_width / PrinterConfig.NOZZLE_WIDTH * PREVIEW_SCALE)
    loop_h_px = int(loop_length / PrinterConfig.NOZZLE_WIDTH * PREVIEW_SCALE)
    hole_r_px = int(loop_hole / 2 / PrinterConfig.NOZZLE_WIDTH * PREVIEW_SCALE)
    circle_r_px = loop_w_px // 2
    
    cx = int(loop_pos[0] * PREVIEW_SCALE) + margin
    cy = int(loop_pos[1] * PREVIEW_SCALE)
    
    # 创建挂孔图层
    loop_size = max(loop_w_px, loop_h_px) * 2 + 20
    loop_layer = Image.new('RGBA', (loop_size, loop_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(loop_layer)
    
    lc = loop_size // 2
    rect_h = max(1, loop_h_px - circle_r_px)
    
    loop_color = (220, 60, 60, 200)
    outline_color = (255, 255, 255, 255)
    
    # 绘制矩形
    draw.rectangle(
        [lc - loop_w_px//2, lc, lc + loop_w_px//2, lc + rect_h],
        fill=loop_color, outline=outline_color, width=2
    )
    
    # 绘制半圆
    draw.ellipse(
        [lc - circle_r_px, lc - circle_r_px,
         lc + circle_r_px, lc + circle_r_px],
        fill=loop_color, outline=outline_color, width=2
    )
    
    # 绘制孔洞
    draw.ellipse(
        [lc - hole_r_px, lc - hole_r_px,
         lc + hole_r_px, lc + hole_r_px],
        fill=(0, 0, 0, 0)
    )
    
    # 旋转
    if loop_angle != 0:
        loop_layer = loop_layer.rotate(
            -loop_angle, center=(lc, lc),
            expand=False, resample=Image.BICUBIC
        )
    
    # 粘贴到画布
    paste_x = cx - lc
    paste_y = cy - lc - rect_h // 2
    pil_img.paste(loop_layer, (paste_x, paste_y), loop_layer)
    
    return pil_img


def on_preview_click(cache, loop_pos, evt: gr.SelectData):
    """处理预览图点击事件"""
    if evt is None or cache is None:
        return loop_pos, False, "点击无效 - 请先生成预览"
    
    click_x, click_y = evt.index
    click_x = click_x - PREVIEW_MARGIN
    
    # 转换为原始坐标
    orig_x = click_x / PREVIEW_SCALE
    orig_y = click_y / PREVIEW_SCALE
    
    target_w = cache['target_w']
    target_h = cache['target_h']
    orig_x = max(0, min(target_w - 1, orig_x))
    orig_y = max(0, min(target_h - 1, orig_y))
    
    pos_info = f"位置: ({orig_x:.1f}, {orig_y:.1f}) px"
    return (orig_x, orig_y), True, pos_info


def update_preview_with_loop(cache, loop_pos, add_loop,
                            loop_width, loop_length, loop_hole, loop_angle):
    """更新预览图（带挂孔）"""
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
                        add_loop, loop_width, loop_length, loop_hole, loop_pos,
                        modeling_mode="vector", quantize_colors=64):
    """
    生成最终模型的包装函数
    直接调用主转换函数
    """
    return convert_image_to_3d(
        image_path, lut_path, target_width_mm, spacer_thick,
        structure_mode, auto_bg, bg_tol, color_mode,
        add_loop, loop_width, loop_length, loop_hole, loop_pos,
        modeling_mode, quantize_colors
    )
