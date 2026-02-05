"""
Lumina Studio - UI Layout
UI layout definition
"""

import gradio as gr     # type:ignore

from config import ColorSystem
from utils import Stats, LUTManager
from core.calibration import generate_calibration_board
from core.extractor import (
    rotate_image,
    draw_corner_points,
    run_extraction,
    probe_lut_cell,
    manual_fix_cell,
    generate_simulated_reference
)
from core.converter import (
    generate_preview_cached,
    render_preview,
    on_preview_click,
    update_preview_with_loop,
    on_remove_loop,
    generate_final_model
)
from .styles import CUSTOM_CSS
from .callbacks import (
    get_first_hint,
    get_next_hint,
    on_extractor_upload,
    on_extractor_mode_change,
    on_extractor_rotate,
    on_extractor_click,
    on_extractor_clear,
    on_lut_select,
    on_lut_upload_save,
    on_apply_color_replacement,
    on_clear_color_replacements,
    on_preview_generated_update_palette
)


def create_app():
    """Create Gradio application interface"""
    with gr.Blocks(title="Lumina Studio") as app:

        # Header with Language Indicator
        with gr.Row():
            with gr.Column(scale=10):
                gr.HTML("""
                <div class="header-banner">
                    <h1>✨ Lumina Studio</h1>
                    <p>Multi-Material 3D Print Color System | v1.5.3</p>
                </div>
                """)
            with gr.Column(scale=1, min_width=120):
                gr.HTML("""
                <div style="text-align:right; padding:10px;">
                    <span style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                 color:white; padding:5px 15px; border-radius:20px; font-weight:bold; white-space: nowrap;">
                        🌐 EN | CN
                    </span>
                </div>
                """)

        # Stats Bar
        stats = Stats.get_all()
        stats_html = gr.HTML(f"""
        <div class="stats-bar">
            📊 Total Generated: 
            <strong>{stats.get('calibrations', 0)}</strong> Calibrations | 
            <strong>{stats.get('extractions', 0)}</strong> Extractions | 
            <strong>{stats.get('conversions', 0)}</strong> Conversions
        </div>
        """)

        # Main Tabs
        with gr.Tabs() as tabs:

            # ═══════════════════════════════════════════════════════════════
            # TAB 1: Image Converter (MOVED TO FIRST)
            # ═══════════════════════════════════════════════════════════════
            create_converter_tab()

            # ═══════════════════════════════════════════════════════════════
            # TAB 2: Calibration Generator
            # ═══════════════════════════════════════════════════════════════
            create_calibration_tab()

            # ═══════════════════════════════════════════════════════════════
            # TAB 3: Color Extractor
            # ═══════════════════════════════════════════════════════════════
            create_extractor_tab()

            # ═══════════════════════════════════════════════════════════════
            # TAB 4: About
            # ═══════════════════════════════════════════════════════════════
            create_about_tab()

        # Footer
        gr.HTML("""
        <div class="footer">
            <p>💡 Tip: Use high-quality translucent PLA/PETG basic for best results</p>
        </div>
        """)

    return app


def create_calibration_tab():
    """创建校准板生成Tab"""
    with gr.TabItem("📐 校准板 Calibration", id=1):
        cal_desc = gr.Markdown("""
        ### 第二步：生成校准板 | Step 2: Generate Calibration Board
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


def create_extractor_tab():
    """创建颜色提取Tab"""
    with gr.TabItem("🎨 颜色提取 Extractor", id=2):
        gr.Markdown("""
        ### 第三步：提取颜色数据 | Step 3: Extract Color Data
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

        # Event handlers for extractor
        ext_img_in.upload(
            on_extractor_upload,
            [ext_img_in, ext_color_mode],
            [ext_state_img, ext_work_img, ext_state_pts, ext_curr_coord, ext_hint]
        )

        ext_color_mode.change(
            on_extractor_mode_change,
            [ext_state_img, ext_color_mode],
            [ext_state_pts, ext_hint, ext_work_img]
        )

        ext_rot_btn.click(
            on_extractor_rotate,
            [ext_state_img, ext_color_mode],
            [ext_state_img, ext_work_img, ext_state_pts, ext_hint]
        )

        ext_work_img.select(
            on_extractor_click,
            [ext_state_img, ext_state_pts, ext_color_mode],
            [ext_work_img, ext_state_pts, ext_hint]
        )

        ext_clear_btn.click(
            on_extractor_clear,
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


def create_converter_tab():
    """创建图像转换Tab"""
    with gr.TabItem("💎 图像转换 Converter", id=0):
        gr.Markdown("""
        ### 第一步：转换图像 | Step 1: Convert Image
        **两种建模模式**：高保真（RLE无缝拼接）、像素艺术（方块风格）
        
        **流程**: 上传LUT和图像 → 选择建模模式 → 调整色彩细节 → 预览 → 生成
        """)

        # State variables
        conv_loop_pos = gr.State(None)  # Loop position (x, y)
        conv_preview_cache = gr.State(None)  # Cache preview data

        with gr.Row():
            # Left: Input and parameters
            with gr.Column(scale=1):
                gr.Markdown("#### 📁 输入")
                
                # ========== NEW: LUT Preset Selector ==========
                with gr.Group():
                    gr.Markdown("**校准数据 Calibration Data (.npy)**")
                    
                    # LUT selection dropdown
                    conv_lut_dropdown = gr.Dropdown(
                        choices=LUTManager.get_lut_choices(),
                        label="选择预设 Select Preset",
                        value=None,
                        interactive=True,
                        info="从预设库中选择LUT | Select from library"
                    )
                    
                    # Micro upload area (auto-save)
                    conv_lut_upload = gr.File(
                        label="",
                        show_label=False,
                        file_types=['.npy'],
                        height=60,
                        elem_classes=["micro-upload"]
                    )
                    
                    # Status hint
                    conv_lut_status = gr.Markdown(
                        value="💡 拖放.npy文件自动添加 | Drop .npy to add",
                        visible=True
                    )
                
                # Hidden State to store actual LUT path
                conv_lut_path = gr.State(None)
                # ========== END NEW LUT SELECTOR ==========
                
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

                # ========== NEW: Modeling Mode Controls ==========
                conv_modeling_mode = gr.Radio(
                    choices=[
                        "高保真 (细节优先) High-Fidelity (Detail)",
                        "像素艺术 (方块风格) Pixel Art (Blocky)"
                    ],
                    value="高保真 (细节优先) High-Fidelity (Detail)",
                    label="🎨 建模模式 Modeling Mode",
                    info="高保真：RLE无缝拼接，水密模型 | 像素艺术：经典方块美学"
                )

                conv_quantize_count = gr.Slider(
                    minimum=8, maximum=256, step=8, value=64,
                    label="🎨 色彩细节 Color Detail",
                    info="颜色数量越多细节越丰富，但生成越慢 | Higher = More detail, Slower"
                )
                # ========== END NEW CONTROLS ==========

                conv_auto_bg = gr.Checkbox(label="🗑️ 移除背景 Remove Background", value=True,
                                          info="自动移除图像背景色 | Auto remove background")
                conv_tol = gr.Slider(0, 150, 40, label="容差 Tolerance",
                                    info="背景容差值 (0-150)，值越大移除越多 | Higher = Remove more")

                conv_width = gr.Slider(20, 400, 60, label="宽度 Width (mm)")
                conv_thick = gr.Slider(0.2, 3.5, 1.2, step=0.08, label="背板 (mm)")

                conv_preview_btn = gr.Button("👁️👁️ 生成预览", variant="secondary", size="lg")

            # Middle: Preview edit area
            with gr.Column(scale=2):
                gr.Markdown("#### 🎨 2D预览 - 点击图片放置挂孔位置（暂不推荐使用）")

                # Preview image - not interactive for upload, but clickable
                conv_preview = gr.Image(
                    label="",
                    type="numpy",
                    height=400,
                    interactive=False,  # 禁止拖拽上传
                    show_label=False
                )

                # ========== NEW: Color Palette & Replacement ==========
                with gr.Accordion("🎨 颜色调色板 Color Palette", open=False):
                    # State for color replacement
                    conv_selected_color = gr.State(None)  # Currently selected color hex
                    conv_replacement_map = gr.State({})   # {original_hex: replacement_hex}
                    
                    # Palette display
                    conv_palette_html = gr.HTML(
                        value="<p style='color:#888;'>生成预览后显示调色板 | Generate preview to see palette</p>",
                        label=""
                    )
                    
                    # Color replacement controls
                    with gr.Row():
                        conv_color_picker = gr.ColorPicker(
                            label="选择替换颜色 Replacement Color",
                            value="#FF0000",
                            scale=1
                        )
                        conv_selected_display = gr.Textbox(
                            label="已选颜色 Selected",
                            value="未选择",
                            interactive=False,
                            scale=1
                        )
                    
                    with gr.Row():
                        conv_apply_replacement = gr.Button("✅ 应用替换 Apply", size="sm")
                        conv_clear_replacements = gr.Button("🗑️ 清除所有 Clear All", size="sm")
                # ========== END NEW ==========

                # Loop settings
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

                conv_log = gr.Textbox(label="状态", lines=4, interactive=False, max_lines=6, show_label=True)

            # Right: Output
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

        # ===== Event Binding =====
        
        # LUT selection event
        conv_lut_dropdown.change(
            on_lut_select,
            inputs=[conv_lut_dropdown],
            outputs=[conv_lut_path, conv_lut_status]
        )
        
        # LUT upload event (auto-save)
        conv_lut_upload.upload(
            on_lut_upload_save,
            inputs=[conv_lut_upload],
            outputs=[conv_lut_dropdown, conv_lut_status]
        )

        # Generate preview and update palette
        conv_preview_btn.click(
            generate_preview_cached,
            inputs=[conv_img, conv_lut_path, conv_width, conv_auto_bg, conv_tol, conv_color_mode, conv_modeling_mode, conv_quantize_count],
            outputs=[conv_preview, conv_preview_cache, conv_log]
        ).then(
            on_preview_generated_update_palette,
            inputs=[conv_preview_cache],
            outputs=[conv_palette_html]
        )

        # Color replacement: Apply replacement
        conv_apply_replacement.click(
            on_apply_color_replacement,
            inputs=[
                conv_preview_cache, conv_selected_color, conv_color_picker,
                conv_replacement_map, conv_loop_pos, conv_add_loop,
                conv_loop_width, conv_loop_length, conv_loop_hole, conv_loop_angle
            ],
            outputs=[conv_preview, conv_preview_cache, conv_palette_html, conv_replacement_map, conv_log]
        )

        # Color replacement: Clear all replacements
        conv_clear_replacements.click(
            on_clear_color_replacements,
            inputs=[
                conv_preview_cache, conv_loop_pos, conv_add_loop,
                conv_loop_width, conv_loop_length, conv_loop_hole, conv_loop_angle
            ],
            outputs=[conv_preview, conv_preview_cache, conv_palette_html, conv_replacement_map, conv_log]
        )

        # Click preview image to place loop
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

        # Remove loop
        conv_remove_loop.click(
            on_remove_loop,
            outputs=[conv_loop_pos, conv_add_loop, conv_loop_angle, conv_loop_info]
        ).then(
            update_preview_with_loop,
            inputs=[conv_preview_cache, conv_loop_pos, conv_add_loop,
                   conv_loop_width, conv_loop_length, conv_loop_hole, conv_loop_angle],
            outputs=[conv_preview]
        )

        # Update preview in real-time when loop parameters change
        loop_params = [conv_loop_width, conv_loop_length, conv_loop_hole, conv_loop_angle]
        for param in loop_params:
            param.change(
                update_preview_with_loop,
                inputs=[conv_preview_cache, conv_loop_pos, conv_add_loop,
                       conv_loop_width, conv_loop_length, conv_loop_hole, conv_loop_angle],
                outputs=[conv_preview]
            )

        # Generate final model
        conv_btn.click(
            generate_final_model,
            inputs=[conv_img, conv_lut_path, conv_width, conv_thick,
                    conv_structure, conv_auto_bg, conv_tol, conv_color_mode,
                    conv_add_loop, conv_loop_width, conv_loop_length, conv_loop_hole, conv_loop_pos,
                    conv_modeling_mode, conv_quantize_count, conv_replacement_map],  # Added color_replacements
            outputs=[conv_file, conv_3d_preview, conv_preview, conv_log]
        )


def create_about_tab():
    """创建关于Tab"""
    with gr.TabItem("ℹ️ 关于 About", id=3):
        gr.Markdown("""
        ## 🌟 Lumina Studio v1.5.3
        
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
        - **RLE 几何生成** Run-Length Encoding for Geometry
        - **K-Means 色彩量化** Color Quantization for Detail Preservation
        
        ---
        
        ### 📝 v1.5.3 更新日志 Changelog
        
        #### 🔄 版本更新 Version Update
        - 更新版本号至 v1.5.3 Updated version to v1.5.3
        
        ---
        
        ### 📝 v1.5.0 更新日志 Changelog
        
        #### 🎨 代码标准化 Code Standardization
        
        - **注释统一为英文** English-only Comments
        - **文档规范化** Documentation Standards
        - **代码清理** Code Cleanup
        
        ---
        
        ### 📝 v1.4.1 更新日志 Changelog
        
        #### 🚀 建模模式整合 Modeling Mode Consolidation
        - **高保真模式取代矢量和版画模式** High-Fidelity Mode Replaces Vector & Woodblock
        - **语言切换功能** Language Switching Feature
        
        ---
        
        ### 📝 v1.4 更新日志 Changelog
        
        #### 🚀 核心功能：两大建模模式
        
        - ✅ **高保真模式（High-Fidelity）** - RLE算法，无缝拼接，水密模型（10 px/mm）
        - ✅ **像素艺术模式（Pixel Art）** - 经典方块美学，像素艺术风格
        
        #### 🔧 架构重构
    
        - 合并Vector和Woodblock为统一的High-Fidelity模式
        - RLE（Run-Length Encoding）几何生成引擎
        - 零间隙、完美边缘对齐（shrink=0.0）
        - 性能优化：支持100k+面片即时生成
        
        #### 🎨 色彩量化架构
        
        - K-Means聚类（8-256色可调，默认64色）
        - "先聚类，后匹配"（速度提升1000×）
        - 双边滤波 + 中值滤波（消除碎片化区域）
        
        #### 其他改进
        
        - 📏 分辨率解耦（高保真10px/mm，像素艺术2.4px/mm）
        - 🎮 3D预览智能降采样（大模型自动简化）
        - 🚫 浏览器崩溃保护（检测复杂度，超200万像素禁用预览）
        
        ---
        
        ### 📝 v1.3 更新日志 Previous Changelog
        
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
        - [✅] 两种建模模式 Two modeling modes (High-Fidelity/Pixel Art)
        - [✅] RLE几何引擎 RLE geometry engine
        - [✅] 钥匙扣挂孔 Keychain loop
        - [🚧] 漫画模式 Manga mode (Ben-Day dots simulation)
        - [ ] 6色扩展模式 6-color extended mode
        - [ ] 8色专业模式 8-color professional mode
        - [ ] 拼豆模式 Perler bead mode
        
        ---
        
        ### 📄 许可证 License
        
        **CC BY-NC-SA 4.0** - Attribution-NonCommercial-ShareAlike
        
        **商业豁免 Commercial Exemption**: 个人创作者、街边摊贩、小型私营企业可免费使用本软件生成模型并销售实体打印品。
        
        Individual creators, street vendors, and small businesses may freely use this software to generate models and sell physical prints.
        
        ---
        
        ### 🙏 致谢 Acknowledgments
        
        特别感谢 Special thanks to:
        - **HueForge** - 在FDM打印中开创光学混色技术 Pioneering optical color mixing
        - **AutoForge** - 让多色工作流民主化 Democratizing multi-color workflows
        - **3D打印社区** - 持续创新 Continuous innovation
        
        ---
        
        <div style="text-align:center; color:#888; margin-top:20px;">
            Made with ❤️ by [MIN]<br>
            v1.5.3 | 2025
        </div>
        """)




