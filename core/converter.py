"""
Lumina Studio - Image Converter Module
图像转换模块
"""

import os
import tempfile
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from scipy.spatial import KDTree

from config import (
    PrinterConfig,
    ColorSystem,
    PREVIEW_SCALE,
    PREVIEW_MARGIN
)
from utils import Stats, safe_fix_3mf_names


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
        top_voxels = np.transpose(best_stacks, (2, 0, 1))
        spacer = np.full((target_h, target_w), -1, dtype=int)
        spacer[~mask_transparent] = 0
        for z in range(0, spacer_layers):
            full_matrix[z] = spacer
        full_matrix[spacer_layers:] = top_voxels

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
    safe_fix_3mf_names(out_path, names_to_fix)

    Stats.increment("conversions")

    # 构建返回消息
    msg = f"✅ 转换完成！分辨率: {target_w}×{target_h}px | 已组合为一个对象"
    if loop_added:
        msg += f" | 挂孔: {slot_names[loop_info['color_id']]}"

    return out_path, glb_path, preview_img, msg


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
