"""
Lumina Studio - UI Layout (Refactored with i18n)
UI layout definition - Refactored version with language switching support
"""

import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import gradio as gr
from PIL import Image as PILImage

from core.i18n import I18n
from config import ColorSystem
from utils import Stats, LUTManager
from core.calibration import generate_calibration_board, generate_smart_board
from core.extractor import (
    rotate_image,
    draw_corner_points,
    run_extraction,
    probe_lut_cell,
    manual_fix_cell,
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
    on_undo_color_replacement,
    on_preview_generated_update_palette,
    on_color_swatch_click,
    on_lut_change_update_colors,
    on_lut_color_swatch_click,
    on_replacement_color_select,
    on_preview_update_lut_colors,
    on_highlight_color_change,
    on_clear_highlight
)

# Runtime-injected i18n keys (avoids editing core/i18n.py).
if hasattr(I18n, 'TEXTS'):
    I18n.TEXTS.update({
        'conv_advanced': {'zh': '🛠️ 高级设置', 'en': '🛠️ Advanced Settings'},
        'conv_stop':     {'zh': '🛑 停止生成', 'en': '🛑 Stop Generation'},
        'conv_batch_mode':      {'zh': '📦 批量模式', 'en': '📦 Batch Mode'},
        'conv_batch_mode_info': {'zh': '一次生成多个模型 (参数共享)', 'en': 'Generate multiple models (Shared Settings)'},
        'conv_batch_input':     {'zh': '📤 批量上传图片', 'en': '📤 Batch Upload Images'},
        'conv_lut_status': {'zh': '💡 拖放.npy文件自动添加', 'en': '💡 Drop .npy file to load'},
    })

CONFIG_FILE = "user_settings.json"


def load_last_lut_setting():
    """Load the last selected LUT name from the user settings file.

    Returns:
        str | None: LUT name if found, else None.
    """
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("last_lut", None)
        except Exception as e:
            print(f"Failed to load settings: {e}")
    return None


def save_last_lut_setting(lut_name):
    """Persist the current LUT selection to the user settings file.

    Args:
        lut_name: Display name of the selected LUT (or None to clear).
    """
    data = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            pass

    data["last_lut"] = lut_name

    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save settings: {e}")


# ---------- Header and layout CSS ----------
HEADER_CSS = """
/* Full-width container */
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding-left: 20px !important;
    padding-right: 20px !important;
}

/* Header row with rounded corners */
.header-row {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 15px 20px;
    margin-left: 0 !important;
    margin-right: 0 !important;
    width: 100% !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    margin-bottom: 15px !important;
    align-items: center;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2) !important;
}

.header-row h1 {
    color: white !important;
    margin: 0 !important;
    font-size: 24px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.header-row p {
    color: rgba(255,255,255,0.8) !important;
    margin: 0 !important;
    font-size: 14px;
}

/* Left sidebar */
.left-sidebar {
    background-color: #f9fafb;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    height: 100%;
}

.compact-row {
    margin-top: -10px !important;
    margin-bottom: -10px !important;
    gap: 10px;
}

.micro-upload {
    min-height: 40px !important;
}

/* Workspace area */
.workspace-area {
    padding: 0 10px;
}

/* Action buttons */
.action-buttons {
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Upload box height aligned with dropdown row */
.tall-upload {
    height: 84px !important;
    min-height: 84px !important;
    max-height: 84px !important;
    background-color: #ffffff !important;
    border-radius: 8px !important;
    border: 1px dashed #e5e7eb !important;
    overflow: hidden !important;
    padding: 0 !important;
}

/* Inner layout for upload area */
.tall-upload .wrap {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    padding: 2px !important;
    height: 100% !important;
}

/* Smaller font in upload area */
.tall-upload .icon-wrap { display: none !important; }
.tall-upload span,
.tall-upload div {
    font-size: 12px !important;
    line-height: 1.3 !important;
    color: #6b7280 !important;
    text-align: center !important;
    margin: 0 !important;
}

/* LUT status card style */
.lut-status {
    margin-top: 10px !important;
    padding: 8px 12px !important;
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    color: #4b5563 !important;
    font-size: 13px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    min-height: 36px !important;
    display: flex !important;
    align-items: center !important;
}
.lut-status p {
    margin: 0 !important;
}

/* Transparent group (no box) */
.clean-group {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Modeling mode radio text color (avoid theme override) */
.vertical-radio label span {
    color: #374151 !important;
    font-weight: 500 !important;
}

/* Selected state text color */
.vertical-radio input:checked + span,
.vertical-radio label.selected span {
    color: #1f2937 !important;
}
"""


# ---------- Image size and aspect-ratio helpers ----------

def _get_image_size(img):
    """Get image dimensions (width, height). Supports file path or numpy array.

    Args:
        img: File path (str) or numpy array (H, W, C).

    Returns:
        tuple[int, int] | None: (width, height) in pixels, or None.
    """
    if img is None:
        return None

    try:
        if isinstance(img, str):
            if img.lower().endswith('.svg'):
                try:
                    from svglib.svglib import svg2rlg
                    drawing = svg2rlg(img)
                    return (drawing.width, drawing.height)
                except ImportError:
                    print("⚠️ svglib not installed, cannot read SVG size")
                    return None
                except Exception as e:
                    print(f"⚠️ Error reading SVG size: {e}")
                    return None
            
            with PILImage.open(img) as i:
                return i.size

        elif hasattr(img, 'shape'):
            return (img.shape[1], img.shape[0])
    except Exception as e:
        print(f"Error getting image size: {e}")
        return None
    
    return None


def calc_height_from_width(width, img):
    """Compute height (mm) from width (mm) preserving aspect ratio.

    Args:
        width: Target width in mm.
        img: Image path or array for dimensions.

    Returns:
        float | gr.update: Height in mm, or gr.update() if unknown.
    """
    size = _get_image_size(img)
    if size is None or width is None:
        return gr.update()
    
    w_px, h_px = size
    if w_px == 0:
        return 0
    
    ratio = h_px / w_px
    return round(width * ratio, 1)


def calc_width_from_height(height, img):
    """Compute width (mm) from height (mm) preserving aspect ratio.

    Args:
        height: Target height in mm.
        img: Image path or array for dimensions.

    Returns:
        float | gr.update: Width in mm, or gr.update() if unknown.
    """
    size = _get_image_size(img)
    if size is None or height is None:
        return gr.update()
    
    w_px, h_px = size
    if h_px == 0:
        return 0
    
    ratio = w_px / h_px
    return round(height * ratio, 1)


def init_dims(img):
    """Compute default width/height (mm) from image aspect ratio.

    Args:
        img: Image path or array.

    Returns:
        tuple[float, float]: (default_width_mm, default_height_mm).
    """
    size = _get_image_size(img)
    if size is None:
        return 60, 60
    
    w_px, h_px = size
    default_w = 60
    default_h = round(default_w * (h_px / w_px), 1)
    return default_w, default_h


# ---------- Mode mapping ----------

def map_modeling_mode(mode_display_text):
    """Map UI modeling mode label to internal mode identifier.

    Args:
        mode_display_text: Label from UI (e.g. "High-Fidelity", "Pixel", "Vector").

    Returns:
        str: One of "high-fidelity", "pixel", "vector_native".
    """
    mode_lower = (mode_display_text or "").lower()
    if "vector" in mode_lower or "矢量" in mode_lower:
        return "vector_native"
    if "pixel" in mode_lower or "像素" in mode_lower:
        return "pixel"
    return "high-fidelity"


def generate_final_model_with_mapping(image_path, lut_path, target_width_mm, spacer_thick,
                                      structure_mode, auto_bg, bg_tol, color_mode,
                                      add_loop, loop_width, loop_length, loop_hole, loop_pos,
                                      modeling_mode_display, quantize_colors, color_replacements=None):
    """Run final model generation with UI mode label mapped to internal mode.

    Returns:
        tuple: (out_path, glb_path, preview_img, status_msg) from generate_final_model.
    """
    modeling_mode = map_modeling_mode(modeling_mode_display)
    return generate_final_model(
        image_path, lut_path, target_width_mm, spacer_thick,
        structure_mode, auto_bg, bg_tol, color_mode,
        add_loop, loop_width, loop_length, loop_hole, loop_pos,
        modeling_mode, quantize_colors, color_replacements
    )


def process_batch_generation(batch_files, is_batch, single_image, lut_path, target_width_mm,
                             spacer_thick, structure_mode, auto_bg, bg_tol, color_mode,
                             add_loop, loop_width, loop_length, loop_hole, loop_pos,
                             modeling_mode_display, quantize_colors, color_replacements=None, progress=gr.Progress()):
    """Dispatch to single-image or batch generation; batch writes a ZIP of 3MFs.

    Returns:
        tuple: (file_or_zip_path, model3d_value, preview_image, status_text).
    """
    args = (lut_path, target_width_mm, spacer_thick, structure_mode, auto_bg, bg_tol,
            color_mode, add_loop, loop_width, loop_length, loop_hole, loop_pos,
            modeling_mode_display, quantize_colors, color_replacements)

    if not is_batch:
        return generate_final_model_with_mapping(single_image, *args)

    if not batch_files:
        return None, None, None, "❌ 请先上传图片 / Please upload images first"

    generated_files = []
    total_files = len(batch_files)
    logs = []

    output_dir = os.path.join("outputs", f"batch_{int(time.time())}")
    os.makedirs(output_dir, exist_ok=True)

    logs.append(f"🚀 开始批量处理 {total_files} 张图片...")

    for i, file_obj in enumerate(batch_files):
        path = getattr(file_obj, 'name', file_obj) if file_obj else None
        if not path or not os.path.isfile(path):
            continue
        filename = os.path.basename(path)
        progress(i / total_files, desc=f"Processing {filename}...")
        logs.append(f"[{i+1}/{total_files}] 正在生成: {filename}")

        try:
            result_3mf, _, _, _ = generate_final_model_with_mapping(path, *args)

            if result_3mf and os.path.exists(result_3mf):
                new_name = os.path.splitext(filename)[0] + ".3mf"
                dest_path = os.path.join(output_dir, new_name)
                shutil.copy2(result_3mf, dest_path)
                generated_files.append(dest_path)
        except Exception as e:
            logs.append(f"❌ 失败 {filename}: {str(e)}")
            print(f"Batch error on {filename}: {e}")

    if generated_files:
        zip_path = os.path.join("outputs", f"Lumina_Batch_{int(time.time())}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for f in generated_files:
                zipf.write(f, os.path.basename(f))
        logs.append(f"✅ Batch done: {len(generated_files)} model(s).")
        return zip_path, None, None, "\n".join(logs)
    return None, None, None, "❌ Batch failed: no valid models.\n" + "\n".join(logs)


def create_app():
    """Build the Gradio app (tabs, i18n, events) and return the Blocks instance."""
    with gr.Blocks(title="Lumina Studio") as app:
        lang_state = gr.State(value="zh")

        # Header
        with gr.Row(elem_classes=["header-row"], equal_height=True):
            with gr.Column(scale=10):
                app_title_html = gr.HTML(
                    value=f"<h1>✨ Lumina Studio</h1><p>{I18n.get('app_subtitle', 'zh')}</p>",
                    elem_id="app-header"
                )
            with gr.Column(scale=1, min_width=120):
                lang_btn = gr.Button(
                    value="🌐 English",
                    size="sm",
                    elem_id="lang-btn"
                )
        
        stats = Stats.get_all()
        stats_html = gr.HTML(
            value=_get_stats_html("zh", stats),
            elem_id="stats-bar"
        )
        
        tab_components = {}
        with gr.Tabs() as tabs:
            components = {}

            # Converter tab
            with gr.TabItem(label=I18n.get('tab_converter', "zh"), id=0) as tab_conv:
                conv_components = create_converter_tab_content("zh")
                components.update(conv_components)
            tab_components['tab_converter'] = tab_conv
            
            with gr.TabItem(label=I18n.get('tab_calibration', "zh"), id=1) as tab_cal:
                cal_components = create_calibration_tab_content("zh")
                components.update(cal_components)
            tab_components['tab_calibration'] = tab_cal
            
            with gr.TabItem(label=I18n.get('tab_extractor', "zh"), id=2) as tab_ext:
                ext_components = create_extractor_tab_content("zh")
                components.update(ext_components)
            tab_components['tab_extractor'] = tab_ext
            
            with gr.TabItem(label=I18n.get('tab_about', "zh"), id=3) as tab_about:
                about_components = create_about_tab_content("zh")
                components.update(about_components)
            tab_components['tab_about'] = tab_about
        
        footer_html = gr.HTML(
            value=_get_footer_html("zh"),
            elem_id="footer"
        )
        
        def change_language(current_lang):
            """Switch UI language and return updates for all i18n components."""
            new_lang = "en" if current_lang == "zh" else "zh"
            updates = []
            updates.append(gr.update(value=I18n.get('lang_btn_zh' if new_lang == "zh" else 'lang_btn_en', new_lang)))
            updates.append(gr.update(value=_get_header_html(new_lang)))
            stats = Stats.get_all()
            updates.append(gr.update(value=_get_stats_html(new_lang, stats)))
            updates.append(gr.update(label=I18n.get('tab_converter', new_lang)))
            updates.append(gr.update(label=I18n.get('tab_calibration', new_lang)))
            updates.append(gr.update(label=I18n.get('tab_extractor', new_lang)))
            updates.append(gr.update(label=I18n.get('tab_about', new_lang)))
            updates.extend(_get_all_component_updates(new_lang, components))
            updates.append(gr.update(value=_get_footer_html(new_lang)))
            updates.append(new_lang)
            return updates

        output_list = [
            lang_btn,
            app_title_html,
            stats_html,
            tab_components['tab_converter'],
            tab_components['tab_calibration'],
            tab_components['tab_extractor'],
            tab_components['tab_about'],
        ]
        output_list.extend(_get_component_list(components))
        output_list.extend([footer_html, lang_state])

        lang_btn.click(
            change_language,
            inputs=[lang_state],
            outputs=output_list
        )

        app.load(
            fn=on_lut_select,
            inputs=[components['dropdown_conv_lut_dropdown']],
            outputs=[components['state_conv_lut_path'], components['md_conv_lut_status']]
        ).then(
            fn=on_lut_change_update_colors,
            inputs=[components['state_conv_lut_path']],
            outputs=[components['html_conv_lut_colors']]
        )

    return app


# ---------- Helpers for i18n updates ----------

def _get_header_html(lang: str) -> str:
    """Return header HTML (title + subtitle) for the given language."""
    return f"<h1>✨ Lumina Studio</h1><p>{I18n.get('app_subtitle', lang)}</p>"


def _get_stats_html(lang: str, stats: dict) -> str:
    """Return stats bar HTML (calibrations / extractions / conversions)."""
    return f"""
    <div class="stats-bar">
        {I18n.get('stats_total', lang)}: 
        <strong>{stats.get('calibrations', 0)}</strong> {I18n.get('stats_calibrations', lang)} | 
        <strong>{stats.get('extractions', 0)}</strong> {I18n.get('stats_extractions', lang)} | 
        <strong>{stats.get('conversions', 0)}</strong> {I18n.get('stats_conversions', lang)}
    </div>
    """


def _get_footer_html(lang: str) -> str:
    """Return footer HTML for the given language."""
    return f"""
    <div class="footer">
        <p>{I18n.get('footer_tip', lang)}</p>
    </div>
    """


def _get_all_component_updates(lang: str, components: dict) -> list:
    """Build a list of gr.update() for all components to apply i18n.

    Skips dynamic status components (md_conv_lut_status, textbox_conv_status)
    so their runtime text is not overwritten.

    Args:
        lang: Target language code ('zh' or 'en').
        components: Dict of component key -> Gradio component.

    Returns:
        list: One gr.update() per component, in dict iteration order.
    """
    updates = []
    for key, component in components.items():
        if key == 'md_conv_lut_status' or key == 'textbox_conv_status':
            updates.append(gr.update())
            continue

        if key.startswith('md_'):
            updates.append(gr.update(value=I18n.get(key[3:], lang)))
        elif key.startswith('lbl_'):
            updates.append(gr.update(label=I18n.get(key[4:], lang)))
        elif key.startswith('btn_'):
            updates.append(gr.update(value=I18n.get(key[4:], lang)))
        elif key.startswith('radio_'):
            choice_key = key[6:]
            if choice_key == 'conv_color_mode' or choice_key == 'cal_color_mode' or choice_key == 'ext_color_mode':
                updates.append(gr.update(
                    label=I18n.get(choice_key, lang),
                    choices=[
                        (I18n.get('conv_color_mode_cmyw', lang), I18n.get('conv_color_mode_cmyw', 'en')),
                        (I18n.get('conv_color_mode_rybw', lang), I18n.get('conv_color_mode_rybw', 'en')),
                        ("6-Color (Smart 1296)", "6-Color (Smart 1296)")
                    ]
                ))
            elif choice_key == 'conv_structure':
                updates.append(gr.update(
                    label=I18n.get(choice_key, lang),
                    choices=[
                        (I18n.get('conv_structure_double', lang), I18n.get('conv_structure_double', 'en')),
                        (I18n.get('conv_structure_single', lang), I18n.get('conv_structure_single', 'en'))
                    ]
                ))
            elif choice_key == 'conv_modeling_mode':
                updates.append(gr.update(
                    label=I18n.get(choice_key, lang),
                    info=I18n.get('conv_modeling_mode_info', lang),
                    choices=[
                        (I18n.get('conv_modeling_mode_hifi', lang), I18n.get('conv_modeling_mode_hifi', 'en')),
                        (I18n.get('conv_modeling_mode_pixel', lang), I18n.get('conv_modeling_mode_pixel', 'en')),
                        (I18n.get('conv_modeling_mode_vector', lang), I18n.get('conv_modeling_mode_vector', 'en'))
                    ]
                ))
        elif key.startswith('slider_'):
            slider_key = key[7:]
            updates.append(gr.update(label=I18n.get(slider_key, lang)))
        elif key.startswith('checkbox_'):
            checkbox_key = key[9:]
            info_key = checkbox_key + '_info'
            if info_key in I18n.TEXTS:
                updates.append(gr.update(
                    label=I18n.get(checkbox_key, lang),
                    info=I18n.get(info_key, lang)
                ))
            else:
                updates.append(gr.update(label=I18n.get(checkbox_key, lang)))
        elif key.startswith('dropdown_'):
            dropdown_key = key[9:]
            info_key = dropdown_key + '_info'
            if info_key in I18n.TEXTS:
                updates.append(gr.update(
                    label=I18n.get(dropdown_key, lang),
                    info=I18n.get(info_key, lang)
                ))
            else:
                updates.append(gr.update(label=I18n.get(dropdown_key, lang)))
        elif key.startswith('image_'):
            image_key = key[6:]
            updates.append(gr.update(label=I18n.get(image_key, lang)))
        elif key.startswith('file_'):
            file_key = key[5:]
            updates.append(gr.update(label=I18n.get(file_key, lang)))
        elif key.startswith('textbox_'):
            textbox_key = key[8:]
            updates.append(gr.update(label=I18n.get(textbox_key, lang)))
        elif key.startswith('num_'):
            num_key = key[4:]
            updates.append(gr.update(label=I18n.get(num_key, lang)))
        elif key == 'html_crop_modal':
            from ui.crop_extension import get_crop_modal_html
            updates.append(gr.update(value=get_crop_modal_html(lang)))
        elif key.startswith('html_'):
            html_key = key[5:]
            updates.append(gr.update(value=I18n.get(html_key, lang)))
        elif key.startswith('accordion_'):
            acc_key = key[10:]
            updates.append(gr.update(label=I18n.get(acc_key, lang)))
        else:
            updates.append(gr.update())
    
    return updates


def _get_component_list(components: dict) -> list:
    """Return component values in dict order (for Gradio outputs)."""
    return list(components.values())


def get_extractor_reference_image(mode_str):
    """Load or generate reference image for color extractor (disk-cached).

    Uses assets/ with filenames ref_6color_smart.png, ref_cmyw_standard.png,
    or ref_rybw_standard.png. Generates via calibration board logic if missing.

    Args:
        mode_str: Color mode label (e.g. "6-Color", "CMYW", "RYBW").

    Returns:
        PIL.Image.Image | None: Reference image or None on error.
    """
    cache_dir = "assets"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    if "6-Color" in mode_str or "1296" in mode_str:
        filename = "ref_6color_smart.png"
        gen_mode = "6-Color"
    elif "CMYW" in mode_str:
        filename = "ref_cmyw_standard.png"
        gen_mode = "CMYW"
    else:
        filename = "ref_rybw_standard.png"
        gen_mode = "RYBW"

    filepath = os.path.join(cache_dir, filename)

    if os.path.exists(filepath):
        try:
            print(f"[UI] Loading reference from cache: {filepath}")
            return PILImage.open(filepath)
        except Exception as e:
            print(f"Error loading cache, regenerating: {e}")

    print(f"[UI] Generating new reference for {gen_mode}...")
    try:
        block_size = 10
        gap = 0
        backing = "White"

        if gen_mode == "6-Color":
            from core.calibration import generate_smart_board
            _, img, _ = generate_smart_board(block_size, gap)
        else:
            from core.calibration import generate_calibration_board
            _, img, _ = generate_calibration_board(gen_mode, block_size, gap, backing)

        if img:
            if not isinstance(img, PILImage.Image):
                import numpy as np
                img = PILImage.fromarray(img.astype('uint8'), 'RGB')

            img.save(filepath)
            print(f"[UI] Cached reference saved to {filepath}")

        return img

    except Exception as e:
        print(f"Error generating reference: {e}")
        return None


# ---------- Tab builders ----------

def create_converter_tab_content(lang: str) -> dict:
    """Build converter tab UI and events. Returns component dict for i18n.

    Args:
        lang: Initial language code ('zh' or 'en').

    Returns:
        dict: Mapping from component key to Gradio component (and state refs).
    """
    components = {}
    conv_loop_pos = gr.State(None)
    conv_preview_cache = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1, min_width=320, elem_classes=["left-sidebar"]):
            components['md_conv_input_section'] = gr.Markdown(I18n.get('conv_input_section', lang))

            saved_lut = load_last_lut_setting()
            current_choices = LUTManager.get_lut_choices()
            default_lut_value = saved_lut if saved_lut in current_choices else None

            with gr.Row():
                components['dropdown_conv_lut_dropdown'] = gr.Dropdown(
                    choices=current_choices,
                    label="校准数据 (.npy) / Calibration Data",
                    value=default_lut_value,
                    interactive=True,
                    scale=2
                )
                conv_lut_upload = gr.File(
                    label="",
                    show_label=False,
                    file_types=['.npy'],
                    height=84,
                    min_width=100,
                    scale=1,
                    elem_classes=["tall-upload"]
                )
            
            components['md_conv_lut_status'] = gr.Markdown(
                value=I18n.get('conv_lut_status_default', lang),
                visible=True,
                elem_classes=["lut-status"]
            )
            conv_lut_path = gr.State(None)

            with gr.Row():
                components['checkbox_conv_batch_mode'] = gr.Checkbox(
                    label=I18n.get('conv_batch_mode', lang),
                    value=False,
                    info=I18n.get('conv_batch_mode_info', lang)
                )
            
            # ========== Image Crop Extension (Non-invasive) ==========
            # Hidden state for preprocessing
            preprocess_img_width = gr.State(0)
            preprocess_img_height = gr.State(0)
            preprocess_processed_path = gr.State(None)
            
            # Crop data states (used by JavaScript via hidden inputs)
            crop_data_state = gr.State({"x": 0, "y": 0, "w": 100, "h": 100})
            
            # Hidden textbox for JavaScript to pass crop data to Python (use CSS to hide)
            crop_data_json = gr.Textbox(
                value='{"x":0,"y":0,"w":100,"h":100,"autoColor":true}',
                elem_id="crop-data-json",
                visible=True,
                elem_classes=["hidden-crop-component"]
            )
            
            # Hidden buttons for JavaScript to trigger Python callbacks (use CSS to hide)
            use_original_btn = gr.Button("use_original", elem_id="use-original-hidden-btn", elem_classes=["hidden-crop-component"])
            confirm_crop_btn = gr.Button("confirm_crop", elem_id="confirm-crop-hidden-btn", elem_classes=["hidden-crop-component"])
            
            # Cropper.js Modal HTML (JS is loaded via head parameter in main.py)
            from ui.crop_extension import get_crop_modal_html
            cropper_modal_html = gr.HTML(
                get_crop_modal_html(lang),
                elem_classes=["crop-modal-container"]
            )
            components['html_crop_modal'] = cropper_modal_html
            
            # Hidden HTML element to store dimensions for JavaScript
            preprocess_dimensions_html = gr.HTML(
                value='<div id="preprocess-dimensions-data" data-width="0" data-height="0" style="display:none;"></div>',
                visible=True,
                elem_classes=["hidden-crop-component"]
            )
            # ========== END Image Crop Extension ==========
            
            components['image_conv_image_label'] = gr.Image(
                label=I18n.get('conv_image_label', lang),
                type="filepath",
                image_mode=None,  # Auto-detect mode to support both JPEG and PNG
                height=240,
                visible=True,
                elem_id="conv-image-input"
            )
            components['file_conv_batch_input'] = gr.File(
                label=I18n.get('conv_batch_input', lang),
                file_count="multiple",
                file_types=["image"],
                visible=False
            )
            components['md_conv_params_section'] = gr.Markdown(I18n.get('conv_params_section', lang))

            with gr.Row(elem_classes=["compact-row"]):
                components['slider_conv_width'] = gr.Slider(
                    minimum=10, maximum=400, value=60, step=1,
                    label=I18n.get('conv_width', lang),
                    interactive=True
                )
                components['slider_conv_height'] = gr.Slider(
                    minimum=10, maximum=400, value=60, step=1,
                    label=I18n.get('conv_height', lang),
                    interactive=True
                )
                components['slider_conv_thickness'] = gr.Slider(
                    0.2, 3.5, 1.2, step=0.08,
                    label=I18n.get('conv_thickness', lang)
                )
            conv_target_height_mm = components['slider_conv_height']

            with gr.Row(elem_classes=["compact-row"]):
                components['radio_conv_color_mode'] = gr.Radio(
                    choices=[
                        (I18n.get('conv_color_mode_cmyw', lang), I18n.get('conv_color_mode_cmyw', 'en')),
                        (I18n.get('conv_color_mode_rybw', lang), I18n.get('conv_color_mode_rybw', 'en')),
                        ("6-Color (Smart 1296)", "6-Color (Smart 1296)")
                    ],
                    value=I18n.get('conv_color_mode_rybw', 'en'),
                    label=I18n.get('conv_color_mode', lang)
                )
                
                components['radio_conv_structure'] = gr.Radio(
                    choices=[
                        (I18n.get('conv_structure_double', lang), I18n.get('conv_structure_double', 'en')),
                        (I18n.get('conv_structure_single', lang), I18n.get('conv_structure_single', 'en'))
                    ],
                    value=I18n.get('conv_structure_double', 'en'),
                    label=I18n.get('conv_structure', lang)
                )

            with gr.Row(elem_classes=["compact-row"]):
                components['radio_conv_modeling_mode'] = gr.Radio(
                    choices=[
                        (I18n.get('conv_modeling_mode_hifi', lang), I18n.get('conv_modeling_mode_hifi', 'en')),
                        (I18n.get('conv_modeling_mode_pixel', lang), I18n.get('conv_modeling_mode_pixel', 'en')),
                        (I18n.get('conv_modeling_mode_vector', lang), I18n.get('conv_modeling_mode_vector', 'en'))
                    ],
                    value=I18n.get('conv_modeling_mode_hifi', 'en'),
                    label=I18n.get('conv_modeling_mode', lang),
                    info=I18n.get('conv_modeling_mode_info', lang),
                    elem_classes=["vertical-radio"],
                    scale=2
                )
                
                components['checkbox_conv_auto_bg'] = gr.Checkbox(
                    label=I18n.get('conv_auto_bg', lang),
                    value=True,
                    info=I18n.get('conv_auto_bg_info', lang),
                    scale=1
                )
            with gr.Accordion(label=I18n.get('conv_advanced', lang), open=False) as conv_advanced_acc:
                components['accordion_conv_advanced'] = conv_advanced_acc
                with gr.Row():
                    components['slider_conv_quantize_colors'] = gr.Slider(
                        minimum=8, maximum=256, step=8, value=64,
                        label=I18n.get('conv_quantize_colors', lang),
                        info=I18n.get('conv_quantize_info', lang),
                        scale=3
                    )
                    components['btn_conv_auto_color'] = gr.Button(
                        I18n.get('conv_auto_color_btn', lang),
                        variant="secondary",
                        size="sm",
                        scale=1
                    )
                with gr.Row():
                    components['slider_conv_tolerance'] = gr.Slider(
                        0, 150, 40,
                        label=I18n.get('conv_tolerance', lang),
                        info=I18n.get('conv_tolerance_info', lang)
                    )
            gr.Markdown("---")
            with gr.Row(elem_classes=["action-buttons"]):
                components['btn_conv_preview_btn'] = gr.Button(
                    I18n.get('conv_preview_btn', lang),
                    variant="secondary",
                    size="lg"
                )
                components['btn_conv_generate_btn'] = gr.Button(
                    I18n.get('conv_generate_btn', lang),
                    variant="primary",
                    size="lg"
                )
            
        with gr.Column(scale=3, elem_classes=["workspace-area"]):
            with gr.Row():
                with gr.Column(scale=1):
                    components['md_conv_preview_section'] = gr.Markdown(
                        I18n.get('conv_preview_section', lang)
                    )
                        
                    conv_preview = gr.Image(
                        label="",
                        type="numpy",
                        height=600,
                        interactive=False,
                        show_label=False
                    )
                    
                    # ========== Color Palette & Replacement ==========
                    with gr.Accordion("🎨 颜色调色板 Color Palette", open=False):
                        # State for color replacement
                        conv_selected_color = gr.State(None)  # Currently selected color hex
                        conv_replacement_map = gr.State({})   # {original_hex: replacement_hex}
                        conv_replacement_history = gr.State([])  # History stack for undo
                        conv_replacement_color_state = gr.State(None)  # Selected replacement color from LUT
                        
                        # Palette display (clickable swatches)
                        conv_palette_html = gr.HTML(
                            value="<p style='color:#888;'>生成预览后显示调色板 | Generate preview to see palette</p>",
                            label=""
                        )
                        
                        # Hidden textbox to receive color selection from JavaScript
                        # Note: Must be visible in DOM for JavaScript to update, but hidden via CSS
                        conv_color_selected_hidden = gr.Textbox(
                            value="",
                            visible=True,
                            interactive=True,
                            elem_id="conv-color-selected-hidden",
                            elem_classes=["hidden-textbox-trigger"],
                            label="",
                            show_label=False,
                            container=False
                        )
                        
                        # Hidden textbox to receive highlight color from JavaScript (for preview highlight)
                        conv_highlight_color_hidden = gr.Textbox(
                            value="",
                            visible=True,
                            interactive=True,
                            elem_id="conv-highlight-color-hidden",
                            elem_classes=["hidden-textbox-trigger"],
                            label="",
                            show_label=False,
                            container=False
                        )
                        
                        # Hidden button to trigger highlight (clicked by JavaScript)
                        conv_highlight_trigger_btn = gr.Button(
                            "trigger_highlight",
                            elem_id="conv-highlight-trigger-btn",
                            elem_classes=["hidden-textbox-trigger"],
                            visible=True
                        )
                        
                        # Color replacement controls
                        with gr.Row():
                            conv_selected_display = gr.Textbox(
                                label="已选颜色 Selected",
                                value="未选择",
                                interactive=False,
                                scale=1
                            )
                        
                        # LUT available colors - clickable grid (no dropdown)
                        gr.Markdown("### 🎨 替换为 LUT 可用颜色 | Replace with LUT Color")
                        gr.Markdown("*点击色块选择替换颜色 | Click a color swatch to select*", elem_classes=["text-muted"])
                        
                        # Visual clickable grid of available LUT colors
                        conv_lut_colors_html = gr.HTML(
                            value="<p style='color:#888;'>选择 LUT 后显示可用颜色 | Select LUT to see available colors</p>",
                            label=""
                        )
                        
                        # Hidden textbox to receive LUT color selection from JavaScript
                        conv_lut_color_selected_hidden = gr.Textbox(
                            value="",
                            visible=True,
                            interactive=True,
                            elem_id="conv-lut-color-selected-hidden",
                            elem_classes=["hidden-textbox-trigger"],
                            label="",
                            show_label=False,
                            container=False
                        )
                        
                        # Hidden button to trigger LUT color selection (clicked by JavaScript)
                        conv_lut_color_trigger_btn = gr.Button(
                            "trigger_lut_color",
                            elem_id="conv-lut-color-trigger-btn",
                            elem_classes=["hidden-textbox-trigger"],
                            visible=True
                        )
                        
                        # Hidden button to trigger palette color selection (clicked by JavaScript)
                        conv_color_trigger_btn = gr.Button(
                            "trigger_color",
                            elem_id="conv-color-trigger-btn",
                            elem_classes=["hidden-textbox-trigger"],
                            visible=True
                        )
                        
                        conv_replacement_display = gr.Textbox(
                            label="替换颜色 Replacement",
                            value="未选择替换颜色",
                            interactive=False,
                            scale=1
                        )
                        
                        with gr.Row():
                            conv_apply_replacement = gr.Button("✅ 应用替换 Apply", size="sm")
                            conv_undo_replacement = gr.Button("↩️ 撤销 Undo", size="sm")
                            conv_clear_replacements = gr.Button("🗑️ 清除所有 Clear All", size="sm")
                    # ========== END Color Palette ==========
                    
                    with gr.Group():
                        components['md_conv_loop_section'] = gr.Markdown(
                            I18n.get('conv_loop_section', lang)
                        )
                            
                        with gr.Row():
                            components['checkbox_conv_loop_enable'] = gr.Checkbox(
                                label=I18n.get('conv_loop_enable', lang),
                                value=False
                            )
                            components['btn_conv_loop_remove'] = gr.Button(
                                I18n.get('conv_loop_remove', lang),
                                size="sm"
                            )
                            
                        with gr.Row():
                            components['slider_conv_loop_width'] = gr.Slider(
                                2, 10, 4, step=0.5,
                                label=I18n.get('conv_loop_width', lang)
                            )
                            components['slider_conv_loop_length'] = gr.Slider(
                                4, 15, 8, step=0.5,
                                label=I18n.get('conv_loop_length', lang)
                            )
                            components['slider_conv_loop_hole'] = gr.Slider(
                                1, 5, 2.5, step=0.25,
                                label=I18n.get('conv_loop_hole', lang)
                            )
                            
                        with gr.Row():
                            components['slider_conv_loop_angle'] = gr.Slider(
                                -180, 180, 0, step=5,
                                label=I18n.get('conv_loop_angle', lang)
                            )
                            components['textbox_conv_loop_info'] = gr.Textbox(
                                label=I18n.get('conv_loop_info', lang),
                                interactive=False,
                                scale=2
                            )
                    components['textbox_conv_status'] = gr.Textbox(
                        label=I18n.get('conv_status', lang),
                        lines=3,
                        interactive=False,
                        max_lines=10,
                        show_label=True
                    )
                with gr.Column(scale=1):
                    components['md_conv_3d_preview'] = gr.Markdown(
                        I18n.get('conv_3d_preview', lang)
                    )
                        
                    conv_3d_preview = gr.Model3D(
                        label="3D",
                        clear_color=[0.9, 0.9, 0.9, 1.0],
                        height=600
                    )
                        
                    components['md_conv_download_section'] = gr.Markdown(
                        I18n.get('conv_download_section', lang)
                    )
                        
                    components['file_conv_download_file'] = gr.File(
                        label=I18n.get('conv_download_file', lang)
                    )
                    components['btn_conv_stop'] = gr.Button(
                        value=I18n.get('conv_stop', lang),
                        variant="stop",
                        size="lg"
                    )
    
    # Event binding
    def toggle_batch_mode(is_batch):
        return [
            gr.update(visible=not is_batch),
            gr.update(visible=is_batch)
        ]

    components['checkbox_conv_batch_mode'].change(
        fn=toggle_batch_mode,
        inputs=[components['checkbox_conv_batch_mode']],
        outputs=[components['image_conv_image_label'], components['file_conv_batch_input']]
    )

    # ========== Image Crop Extension Events (Non-invasive) ==========
    from core.image_preprocessor import ImagePreprocessor
    
    def on_image_upload_process_with_html(image_path):
        """When image is uploaded, process and prepare for crop modal (不分析颜色)"""
        if image_path is None:
            return (
                0, 0, None,
                '<div id="preprocess-dimensions-data" data-width="0" data-height="0" style="display:none;"></div>'
            )
        
        try:
            info = ImagePreprocessor.process_upload(image_path)
            # 不在这里分析颜色，等用户确认裁剪后再分析
            dimensions_html = f'<div id="preprocess-dimensions-data" data-width="{info.width}" data-height="{info.height}" style="display:none;"></div>'
            return (info.width, info.height, info.processed_path, dimensions_html)
        except Exception as e:
            print(f"Image upload error: {e}")
            return (0, 0, None, '<div id="preprocess-dimensions-data" data-width="0" data-height="0" style="display:none;"></div>')
    
    # JavaScript to open crop modal (不传递颜色推荐，弹窗中不显示)
    open_crop_modal_js = """
    () => {
        setTimeout(() => {
            const dimElement = document.querySelector('#preprocess-dimensions-data');
            if (dimElement) {
                const width = parseInt(dimElement.dataset.width) || 0;
                const height = parseInt(dimElement.dataset.height) || 0;
                if (width > 0 && height > 0) {
                    const imgContainer = document.querySelector('#conv-image-input');
                    if (imgContainer) {
                        const img = imgContainer.querySelector('img');
                        if (img && img.src && typeof window.openCropModal === 'function') {
                            window.openCropModal(img.src, width, height, 0, 0);
                        }
                    }
                }
            }
        }, 300);
    }
    """
    
    components['image_conv_image_label'].upload(
        on_image_upload_process_with_html,
        inputs=[components['image_conv_image_label']],
        outputs=[preprocess_img_width, preprocess_img_height, preprocess_processed_path, preprocess_dimensions_html]
    ).then(
        fn=None,
        inputs=None,
        outputs=None,
        js=open_crop_modal_js
    )
    
    def use_original_image_simple(processed_path, w, h, crop_json):
        """Use original image without cropping"""
        print(f"[DEBUG] use_original_image_simple called: {processed_path}")
        if processed_path is None:
            return None
        try:
            result_path = ImagePreprocessor.convert_to_png(processed_path)
            return result_path
        except Exception as e:
            print(f"Use original error: {e}")
            return None
    
    use_original_btn.click(
        use_original_image_simple,
        inputs=[preprocess_processed_path, preprocess_img_width, preprocess_img_height, crop_data_json],
        outputs=[components['image_conv_image_label']]
    )
    
    def confirm_crop_image_simple(processed_path, crop_json):
        """Crop image with specified region"""
        print(f"[DEBUG] confirm_crop_image_simple called: {processed_path}, {crop_json}")
        if processed_path is None:
            return None
        try:
            import json
            data = json.loads(crop_json) if crop_json else {"x": 0, "y": 0, "w": 100, "h": 100}
            x = int(data.get("x", 0))
            y = int(data.get("y", 0))
            w = int(data.get("w", 100))
            h = int(data.get("h", 100))
            
            result_path = ImagePreprocessor.crop_image(processed_path, x, y, w, h)
            return result_path
        except Exception as e:
            print(f"Crop error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    confirm_crop_btn.click(
        confirm_crop_image_simple,
        inputs=[preprocess_processed_path, crop_data_json],
        outputs=[components['image_conv_image_label']]
    )
    
    # ========== Auto Color Detection Button ==========
    # 用于触发 toast 的隐藏 HTML 组件
    color_toast_trigger = gr.HTML(value="", visible=True, elem_classes=["hidden-crop-component"])
    
    # JavaScript to show color recommendation toast
    show_toast_js = """
    () => {
        setTimeout(() => {
            const trigger = document.querySelector('#color-rec-trigger');
            if (trigger) {
                const recommended = parseInt(trigger.dataset.recommended) || 0;
                const maxSafe = parseInt(trigger.dataset.maxsafe) || 0;
                if (recommended > 0 && typeof window.showColorRecommendationToast === 'function') {
                    const lang = document.documentElement.lang || 'zh';
                    let msg;
                    if (lang === 'en') {
                        msg = '💡 Color detail set to <b>' + recommended + '</b> (max safe: ' + maxSafe + ')';
                    } else {
                        msg = '💡 色彩细节已设置为 <b>' + recommended + '</b>（最大安全值: ' + maxSafe + '）';
                    }
                    window.showColorRecommendationToast(msg);
                }
                trigger.remove();
            }
        }, 100);
    }
    """
    
    def auto_detect_colors(image_path, target_width_mm):
        """自动检测推荐的色彩细节值"""
        if image_path is None:
            return gr.update(), ""
        try:
            import time
            print(f"[AutoColor] 开始分析: {image_path}, 目标宽度: {target_width_mm}mm")
            color_analysis = ImagePreprocessor.analyze_recommended_colors(image_path, target_width_mm)
            recommended = color_analysis.get('recommended', 24)
            max_safe = color_analysis.get('max_safe', 32)
            print(f"[AutoColor] 分析完成: recommended={recommended}, max_safe={max_safe}")
            # 添加时间戳确保每次返回值不同，触发 .then() 中的 JavaScript
            timestamp = int(time.time() * 1000)
            toast_html = f'<div id="color-rec-trigger" data-recommended="{recommended}" data-maxsafe="{max_safe}" data-ts="{timestamp}" style="display:none;"></div>'
            return gr.update(value=recommended), toast_html
        except Exception as e:
            print(f"[AutoColor] 分析失败: {e}")
            import traceback
            traceback.print_exc()
            return gr.update(), ""
    
    components['btn_conv_auto_color'].click(
        auto_detect_colors,
        inputs=[components['image_conv_image_label'], components['slider_conv_width']],
        outputs=[components['slider_conv_quantize_colors'], color_toast_trigger]
    ).then(
        fn=None,
        inputs=None,
        outputs=None,
        js=show_toast_js
    )
    # ========== END Image Crop Extension Events ==========

    components['dropdown_conv_lut_dropdown'].change(
            on_lut_select,
            inputs=[components['dropdown_conv_lut_dropdown']],
            outputs=[conv_lut_path, components['md_conv_lut_status']]
    ).then(
            fn=save_last_lut_setting,
            inputs=[components['dropdown_conv_lut_dropdown']],
            outputs=None
    ).then(
            fn=on_lut_change_update_colors,
            inputs=[conv_lut_path],
            outputs=[conv_lut_colors_html]
    )

    conv_lut_upload.upload(
            on_lut_upload_save,
            inputs=[conv_lut_upload],
            outputs=[components['dropdown_conv_lut_dropdown'], components['md_conv_lut_status']]
    )
    
    components['image_conv_image_label'].change(
            fn=init_dims,
            inputs=[components['image_conv_image_label']],
            outputs=[components['slider_conv_width'], conv_target_height_mm]
    )
    components['slider_conv_width'].input(
            fn=calc_height_from_width,
            inputs=[components['slider_conv_width'], components['image_conv_image_label']],
            outputs=[conv_target_height_mm]
    )
    conv_target_height_mm.input(
            fn=calc_width_from_height,
            inputs=[conv_target_height_mm, components['image_conv_image_label']],
            outputs=[components['slider_conv_width']]
    )
    components['btn_conv_preview_btn'].click(
            generate_preview_cached,
            inputs=[
                components['image_conv_image_label'],
                conv_lut_path,
                components['slider_conv_width'],
                components['checkbox_conv_auto_bg'],
                components['slider_conv_tolerance'],
                components['radio_conv_color_mode'],
                components['radio_conv_modeling_mode'],
                components['slider_conv_quantize_colors']
            ],
            outputs=[conv_preview, conv_preview_cache, components['textbox_conv_status']]
    ).then(
            on_preview_generated_update_palette,
            inputs=[conv_preview_cache],
            outputs=[conv_palette_html, conv_selected_color]
    ).then(
            on_preview_update_lut_colors,
            inputs=[conv_preview_cache, conv_lut_path],
            outputs=[conv_lut_colors_html]
    )
    
    # Hidden textbox receives color selection from JavaScript click (palette)
    # Use button click instead of textbox change for more reliable triggering
    conv_color_trigger_btn.click(
            on_color_swatch_click,
            inputs=[conv_color_selected_hidden],
            outputs=[conv_selected_color, conv_selected_display]
    )
    
    # Hidden textbox receives highlight color from JavaScript click (triggers preview highlight)
    # Use button click instead of textbox change for more reliable triggering
    conv_highlight_trigger_btn.click(
            on_highlight_color_change,
            inputs=[
                conv_highlight_color_hidden, conv_preview_cache, conv_loop_pos,
                components['checkbox_conv_loop_enable'],
                components['slider_conv_loop_width'], components['slider_conv_loop_length'],
                components['slider_conv_loop_hole'], components['slider_conv_loop_angle']
            ],
            outputs=[conv_preview, components['textbox_conv_status']]
    )
    
    # Hidden textbox receives LUT color selection from JavaScript click
    # Use button click instead of textbox change for more reliable triggering
    conv_lut_color_trigger_btn.click(
            on_lut_color_swatch_click,
            inputs=[conv_lut_color_selected_hidden],
            outputs=[conv_replacement_color_state, conv_replacement_display]
    )
    
    # Color replacement: Apply replacement
    conv_apply_replacement.click(
            on_apply_color_replacement,
            inputs=[
                conv_preview_cache, conv_selected_color, conv_replacement_color_state,
                conv_replacement_map, conv_replacement_history, conv_loop_pos, components['checkbox_conv_loop_enable'],
                components['slider_conv_loop_width'], components['slider_conv_loop_length'],
                components['slider_conv_loop_hole'], components['slider_conv_loop_angle']
            ],
            outputs=[conv_preview, conv_preview_cache, conv_palette_html, conv_replacement_map, conv_replacement_history, components['textbox_conv_status']]
    )
    
    # Color replacement: Undo last replacement
    conv_undo_replacement.click(
            on_undo_color_replacement,
            inputs=[
                conv_preview_cache, conv_replacement_map, conv_replacement_history,
                conv_loop_pos, components['checkbox_conv_loop_enable'],
                components['slider_conv_loop_width'], components['slider_conv_loop_length'],
                components['slider_conv_loop_hole'], components['slider_conv_loop_angle']
            ],
            outputs=[conv_preview, conv_preview_cache, conv_palette_html, conv_replacement_map, conv_replacement_history, components['textbox_conv_status']]
    )
    
    # Color replacement: Clear all replacements
    conv_clear_replacements.click(
            on_clear_color_replacements,
            inputs=[
                conv_preview_cache, conv_replacement_map, conv_replacement_history,
                conv_loop_pos, components['checkbox_conv_loop_enable'],
                components['slider_conv_loop_width'], components['slider_conv_loop_length'],
                components['slider_conv_loop_hole'], components['slider_conv_loop_angle']
            ],
            outputs=[conv_preview, conv_preview_cache, conv_palette_html, conv_replacement_map, conv_replacement_history, components['textbox_conv_status']]
    )
    
    conv_preview.select(
            on_preview_click,
            inputs=[conv_preview_cache, conv_loop_pos],
            outputs=[conv_loop_pos, components['checkbox_conv_loop_enable'], components['textbox_conv_loop_info']]
    ).then(
            update_preview_with_loop,
            inputs=[
                conv_preview_cache, conv_loop_pos, components['checkbox_conv_loop_enable'],
                components['slider_conv_loop_width'], components['slider_conv_loop_length'],
                components['slider_conv_loop_hole'], components['slider_conv_loop_angle']
            ],
            outputs=[conv_preview]
    )
    components['btn_conv_loop_remove'].click(
            on_remove_loop,
            outputs=[conv_loop_pos, components['checkbox_conv_loop_enable'], 
                    components['slider_conv_loop_angle'], components['textbox_conv_loop_info']]
    ).then(
            update_preview_with_loop,
            inputs=[
                conv_preview_cache, conv_loop_pos, components['checkbox_conv_loop_enable'],
                components['slider_conv_loop_width'], components['slider_conv_loop_length'],
                components['slider_conv_loop_hole'], components['slider_conv_loop_angle']
            ],
            outputs=[conv_preview]
    )
    loop_params = [
            components['slider_conv_loop_width'],
            components['slider_conv_loop_length'],
            components['slider_conv_loop_hole'],
            components['slider_conv_loop_angle']
    ]
    for param in loop_params:
            param.change(
                update_preview_with_loop,
                inputs=[
                    conv_preview_cache, conv_loop_pos, components['checkbox_conv_loop_enable'],
                    components['slider_conv_loop_width'], components['slider_conv_loop_length'],
                    components['slider_conv_loop_hole'], components['slider_conv_loop_angle']
                ],
                outputs=[conv_preview]
            )
    generate_event = components['btn_conv_generate_btn'].click(
            fn=process_batch_generation,
            inputs=[
                components['file_conv_batch_input'],
                components['checkbox_conv_batch_mode'],
                components['image_conv_image_label'],
                conv_lut_path,
                components['slider_conv_width'],
                components['slider_conv_thickness'],
                components['radio_conv_structure'],
                components['checkbox_conv_auto_bg'],
                components['slider_conv_tolerance'],
                components['radio_conv_color_mode'],
                components['checkbox_conv_loop_enable'],
                components['slider_conv_loop_width'],
                components['slider_conv_loop_length'],
                components['slider_conv_loop_hole'],
                conv_loop_pos,
                components['radio_conv_modeling_mode'],
                components['slider_conv_quantize_colors'],
                conv_replacement_map
            ],
            outputs=[
                components['file_conv_download_file'],
                conv_3d_preview,
                conv_preview,
                components['textbox_conv_status']
            ]
    )
    components['btn_conv_stop'].click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[generate_event]
    )
    components['state_conv_lut_path'] = conv_lut_path
    components['html_conv_lut_colors'] = conv_lut_colors_html
    return components



def create_calibration_tab_content(lang: str) -> dict:
    """Build calibration board tab UI and events. Returns component dict."""
    components = {}
    
    with gr.Row():
        with gr.Column(scale=1):
            components['md_cal_params'] = gr.Markdown(I18n.get('cal_params', lang))
                
            components['radio_cal_color_mode'] = gr.Radio(
                choices=[
                    (I18n.get('conv_color_mode_cmyw', lang), I18n.get('conv_color_mode_cmyw', 'en')),
                    (I18n.get('conv_color_mode_rybw', lang), I18n.get('conv_color_mode_rybw', 'en')),
                    ("6-Color (Smart 1296)", "6-Color (Smart 1296)")
                ],
                value=I18n.get('conv_color_mode_rybw', 'en'),
                label=I18n.get('cal_color_mode', lang)
            )
                
            components['slider_cal_block_size'] = gr.Slider(
                3, 10, 5, step=1,
                label=I18n.get('cal_block_size', lang)
            )
                
            components['slider_cal_gap'] = gr.Slider(
                0.4, 2.0, 0.82, step=0.02,
                label=I18n.get('cal_gap', lang)
            )
                
            components['dropdown_cal_backing'] = gr.Dropdown(
                choices=["White", "Cyan", "Magenta", "Yellow", "Red", "Blue"],
                value="White",
                label=I18n.get('cal_backing', lang)
            )
                
            components['btn_cal_generate_btn'] = gr.Button(
                I18n.get('cal_generate_btn', lang),
                variant="primary",
                elem_classes=["primary-btn"]
            )
                
            components['textbox_cal_status'] = gr.Textbox(
                label=I18n.get('cal_status', lang),
                interactive=False
            )
            
        with gr.Column(scale=1):
            components['md_cal_preview'] = gr.Markdown(I18n.get('cal_preview', lang))
                
            cal_preview = gr.Image(
                label="Calibration Preview",
                show_label=False
            )
                
            components['file_cal_download'] = gr.File(
                label=I18n.get('cal_download', lang)
            )
    
    # Event binding - Call different generator based on mode
    def generate_board_wrapper(color_mode, block_size, gap, backing):
        """Wrapper function to call appropriate generator based on mode"""
        if "6-Color" in color_mode:
            # Call Smart 1296 generator
            return generate_smart_board(block_size, gap)
        else:
            # Call traditional 4-color generator
            return generate_calibration_board(color_mode, block_size, gap, backing)
    
    components['btn_cal_generate_btn'].click(
            generate_board_wrapper,
            inputs=[
                components['radio_cal_color_mode'],
                components['slider_cal_block_size'],
                components['slider_cal_gap'],
                components['dropdown_cal_backing']
            ],
            outputs=[
                components['file_cal_download'],
                cal_preview,
                components['textbox_cal_status']
            ]
    )
    
    return components


def create_extractor_tab_content(lang: str) -> dict:
    """Build color extractor tab UI and events. Returns component dict."""
    components = {}
    ext_state_img = gr.State(None)
    ext_state_pts = gr.State([])
    ext_curr_coord = gr.State(None)
    default_mode = I18n.get('conv_color_mode_rybw', 'en')
    ref_img = get_extractor_reference_image(default_mode)

    with gr.Row():
        with gr.Column(scale=1):
            components['md_ext_upload_section'] = gr.Markdown(
                I18n.get('ext_upload_section', lang)
            )
                
            components['radio_ext_color_mode'] = gr.Radio(
                choices=[
                    (I18n.get('conv_color_mode_cmyw', lang), I18n.get('conv_color_mode_cmyw', 'en')),
                    (I18n.get('conv_color_mode_rybw', lang), I18n.get('conv_color_mode_rybw', 'en')),
                    ("6-Color (Smart 1296)", "6-Color (Smart 1296)")
                ],
                value=I18n.get('conv_color_mode_rybw', 'en'),
                label=I18n.get('ext_color_mode', lang)
            )
                
            ext_img_in = gr.Image(
                label=I18n.get('ext_photo', lang),
                type="numpy",
                interactive=True
            )
                
            with gr.Row():
                components['btn_ext_rotate_btn'] = gr.Button(
                    I18n.get('ext_rotate_btn', lang)
                )
                components['btn_ext_reset_btn'] = gr.Button(
                    I18n.get('ext_reset_btn', lang)
                )
                
            components['md_ext_correction_section'] = gr.Markdown(
                I18n.get('ext_correction_section', lang)
            )
                
            with gr.Row():
                components['checkbox_ext_wb'] = gr.Checkbox(
                    label=I18n.get('ext_wb', lang),
                    value=True
                )
                components['checkbox_ext_vignette'] = gr.Checkbox(
                    label=I18n.get('ext_vignette', lang),
                    value=False
                )
                
            components['slider_ext_zoom'] = gr.Slider(
                0.8, 1.2, 1.0, step=0.005,
                label=I18n.get('ext_zoom', lang)
            )
                
            components['slider_ext_distortion'] = gr.Slider(
                -0.2, 0.2, 0.0, step=0.01,
                label=I18n.get('ext_distortion', lang)
            )
                
            components['slider_ext_offset_x'] = gr.Slider(
                -30, 30, 0, step=1,
                label=I18n.get('ext_offset_x', lang)
            )
                
            components['slider_ext_offset_y'] = gr.Slider(
                -30, 30, 0, step=1,
                label=I18n.get('ext_offset_y', lang)
            )
                
            components['btn_ext_extract_btn'] = gr.Button(
                I18n.get('ext_extract_btn', lang),
                variant="primary",
                elem_classes=["primary-btn"]
            )
                
            components['textbox_ext_status'] = gr.Textbox(
                label=I18n.get('ext_status', lang),
                interactive=False
            )
            
        with gr.Column(scale=1):
            ext_hint = gr.Markdown(I18n.get('ext_hint_white', lang))
                
            ext_work_img = gr.Image(
                label=I18n.get('ext_marked', lang),
                show_label=False,
                interactive=True
            )
                
            with gr.Row():
                with gr.Column():
                    components['md_ext_sampling'] = gr.Markdown(
                        I18n.get('ext_sampling', lang)
                    )
                    ext_warp_view = gr.Image(show_label=False)
                    
                with gr.Column():
                    components['md_ext_reference'] = gr.Markdown(
                        I18n.get('ext_reference', lang)
                    )
                    ext_ref_view = gr.Image(
                        show_label=False,
                        value=ref_img,
                        interactive=False
                    )
                
            with gr.Row():
                with gr.Column():
                    components['md_ext_result'] = gr.Markdown(
                        I18n.get('ext_result', lang)
                    )
                    ext_lut_view = gr.Image(
                        show_label=False,
                        interactive=True
                    )
                    
                with gr.Column():
                    components['md_ext_manual_fix'] = gr.Markdown(
                        I18n.get('ext_manual_fix', lang)
                    )
                    ext_probe_html = gr.HTML(I18n.get('ext_click_cell', lang))
                        
                    ext_picker = gr.ColorPicker(
                        label=I18n.get('ext_override', lang),
                        value="#FF0000"
                    )
                        
                    components['btn_ext_apply_btn'] = gr.Button(
                        I18n.get('ext_apply_btn', lang)
                    )
                        
                    components['file_ext_download_npy'] = gr.File(
                        label=I18n.get('ext_download_npy', lang)
                    )
    
    ext_img_in.upload(
            on_extractor_upload,
            [ext_img_in, components['radio_ext_color_mode']],
            [ext_state_img, ext_work_img, ext_state_pts, ext_curr_coord, ext_hint]
    )
    
    components['radio_ext_color_mode'].change(
            on_extractor_mode_change,
            [ext_state_img, components['radio_ext_color_mode']],
            [ext_state_pts, ext_hint, ext_work_img]
    )

    components['radio_ext_color_mode'].change(
        fn=get_extractor_reference_image,
        inputs=[components['radio_ext_color_mode']],
        outputs=[ext_ref_view]
    )

    components['btn_ext_rotate_btn'].click(
            on_extractor_rotate,
            [ext_state_img, components['radio_ext_color_mode']],
            [ext_state_img, ext_work_img, ext_state_pts, ext_hint]
    )
    
    ext_work_img.select(
            on_extractor_click,
            [ext_state_img, ext_state_pts, components['radio_ext_color_mode']],
            [ext_work_img, ext_state_pts, ext_hint]
    )
    
    components['btn_ext_reset_btn'].click(
            on_extractor_clear,
            [ext_state_img, components['radio_ext_color_mode']],
            [ext_work_img, ext_state_pts, ext_hint]
    )
    
    extract_inputs = [
            ext_state_img, ext_state_pts,
            components['slider_ext_offset_x'], components['slider_ext_offset_y'],
            components['slider_ext_zoom'], components['slider_ext_distortion'],
            components['checkbox_ext_wb'], components['checkbox_ext_vignette'],
            components['radio_ext_color_mode']
    ]
    extract_outputs = [
            ext_warp_view, ext_lut_view,
            components['file_ext_download_npy'], components['textbox_ext_status']
    ]
    
    components['btn_ext_extract_btn'].click(run_extraction, extract_inputs, extract_outputs)
    
    for s in [components['slider_ext_offset_x'], components['slider_ext_offset_y'],
                  components['slider_ext_zoom'], components['slider_ext_distortion']]:
            s.release(run_extraction, extract_inputs, extract_outputs)
    
    ext_lut_view.select(probe_lut_cell, [], [ext_probe_html, ext_picker, ext_curr_coord])
    components['btn_ext_apply_btn'].click(
            manual_fix_cell,
            [ext_curr_coord, ext_picker],
            [ext_lut_view, components['textbox_ext_status']]
    )
    
    return components



def create_about_tab_content(lang: str) -> dict:
    """Build About tab content from i18n. Returns component dict."""
    components = {}
    
    # About page content (from i18n)
    components['md_about_content'] = gr.Markdown(I18n.get('about_content', lang))
    
    return components
