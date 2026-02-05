"""
Lumina Studio - UI Callbacks
UI event handling callback functions
"""

import gradio as gr

from config import ColorSystem
from core.extractor import generate_simulated_reference
from utils import LUTManager


# ═══════════════════════════════════════════════════════════════
# LUT Management Callbacks
# ═══════════════════════════════════════════════════════════════

def on_lut_select(display_name):
    """
    When user selects LUT from dropdown
    
    Returns:
        tuple: (lut_path, status_message)
    """
    if not display_name:
        return None, ""
    
    lut_path = LUTManager.get_lut_path(display_name)
    
    if lut_path:
        return lut_path, f"✅ Selected: {display_name}"
    else:
        return None, f"❌ File not found: {display_name}"


def on_lut_upload_save(uploaded_file):
    """
    Save uploaded LUT file (auto-save, no custom name needed)
    
    Returns:
        tuple: (new_dropdown, status_message)
    """
    success, message, new_choices = LUTManager.save_uploaded_lut(uploaded_file, custom_name=None)
    
    return gr.Dropdown(choices=new_choices), message


# ═══════════════════════════════════════════════════════════════
# Extractor Callbacks
# ═══════════════════════════════════════════════════════════════

def get_first_hint(mode):
    """Get first corner point hint based on mode"""
    conf = ColorSystem.get(mode)
    label_zh = conf['corner_labels'][0]
    label_en = conf['corner_labels_en'][0]
    return f"#### 👉 点击 Click: **{label_zh} / {label_en}**"


def get_next_hint(mode, pts_count):
    """Get next corner point hint based on mode"""
    conf = ColorSystem.get(mode)
    if pts_count >= 4:
        return "#### ✅ Positioning complete! Ready to extract!"
    label_zh = conf['corner_labels'][pts_count]
    label_en = conf['corner_labels_en'][pts_count]
    return f"#### 👉 点击 Click: **{label_zh} / {label_en}**"


def on_extractor_upload(i, mode):
    """Handle image upload"""
    hint = get_first_hint(mode)
    return i, i, [], None, hint


def on_extractor_mode_change(img, mode):
    """Handle color mode change"""
    hint = get_first_hint(mode)
    return [], hint, img


def on_extractor_rotate(i, mode):
    """Rotate image"""
    from core.extractor import rotate_image
    if i is None:
        return None, None, [], get_first_hint(mode)
    r = rotate_image(i, "Rotate Left 90°")
    return r, r, [], get_first_hint(mode)


def on_extractor_click(img, pts, mode, evt: gr.SelectData):
    """Set corner point by clicking image"""
    from core.extractor import draw_corner_points
    if len(pts) >= 4:
        return img, pts, "#### ✅ 定位完成 Complete!"
    n = pts + [[evt.index[0], evt.index[1]]]
    vis = draw_corner_points(img, n, mode)
    hint = get_next_hint(mode, len(n))
    return vis, n, hint


def on_extractor_clear(img, mode):
    """Clear corner points"""
    hint = get_first_hint(mode)
    return img, [], hint


# ═══════════════════════════════════════════════════════════════
# Color Replacement Callbacks
# ═══════════════════════════════════════════════════════════════

def on_palette_color_select(palette_html, evt: gr.SelectData):
    """
    Handle palette color selection from HTML display.
    
    Note: This is a placeholder - Gradio HTML components don't support
    click events directly. The actual selection is done via JavaScript
    or by clicking on the palette display area.
    
    Args:
        palette_html: Current palette HTML
        evt: Selection event data
    
    Returns:
        tuple: (selected_color_hex, display_text)
    """
    # In practice, color selection would be handled differently
    # since Gradio HTML doesn't support click events
    return None, "点击调色板选择颜色 | Click palette to select"


def on_apply_color_replacement(cache, selected_color, replacement_color, 
                               replacement_map, replacement_history, loop_pos, add_loop,
                               loop_width, loop_length, loop_hole, loop_angle):
    """
    Apply a color replacement to the preview.
    
    Args:
        cache: Preview cache from generate_preview_cached
        selected_color: Currently selected original color (hex string)
        replacement_color: New color to replace with (hex string from ColorPicker)
        replacement_map: Current replacement map dict
        replacement_history: History stack for undo
        loop_pos: Loop position tuple
        add_loop: Whether loop is enabled
        loop_width: Loop width in mm
        loop_length: Loop length in mm
        loop_hole: Loop hole diameter in mm
        loop_angle: Loop rotation angle
    
    Returns:
        tuple: (preview_image, updated_cache, palette_html, updated_replacement_map, 
                updated_history, status)
    """
    from core.converter import update_preview_with_replacements, generate_palette_html
    
    if cache is None:
        return None, None, "", replacement_map, replacement_history, "❌ 请先生成预览 | Generate preview first"
    
    if not selected_color:
        return None, cache, "", replacement_map, replacement_history, "❌ 请先选择要替换的颜色 | Select a color first"
    
    # Save current state to history before applying new replacement
    new_history = replacement_history.copy() if replacement_history else []
    new_history.append(replacement_map.copy() if replacement_map else {})
    
    # Update replacement map
    new_map = replacement_map.copy() if replacement_map else {}
    new_map[selected_color] = replacement_color
    
    # Apply replacements and update preview
    display, updated_cache, palette_html = update_preview_with_replacements(
        cache, new_map, loop_pos, add_loop,
        loop_width, loop_length, loop_hole, loop_angle
    )
    
    return display, updated_cache, palette_html, new_map, new_history, f"✅ 已替换 {selected_color} → {replacement_color}"


def on_clear_color_replacements(cache, replacement_map, replacement_history,
                                loop_pos, add_loop,
                                loop_width, loop_length, loop_hole, loop_angle):
    """
    Clear all color replacements and restore original preview.
    
    Args:
        cache: Preview cache
        replacement_map: Current replacement map dict
        replacement_history: History stack for undo
        loop_pos: Loop position tuple
        add_loop: Whether loop is enabled
        loop_width: Loop width in mm
        loop_length: Loop length in mm
        loop_hole: Loop hole diameter in mm
        loop_angle: Loop rotation angle
    
    Returns:
        tuple: (preview_image, updated_cache, palette_html, empty_replacement_map, 
                updated_history, status)
    """
    from core.converter import update_preview_with_replacements, generate_palette_html
    
    if cache is None:
        return None, None, "", {}, [], "❌ 请先生成预览 | Generate preview first"
    
    # Save current state to history before clearing
    new_history = replacement_history.copy() if replacement_history else []
    if replacement_map:
        new_history.append(replacement_map.copy())
    
    # Clear replacements by passing empty dict
    display, updated_cache, palette_html = update_preview_with_replacements(
        cache, {}, loop_pos, add_loop,
        loop_width, loop_length, loop_hole, loop_angle
    )
    
    return display, updated_cache, palette_html, {}, new_history, "✅ 已清除所有颜色替换 | All replacements cleared"


def on_preview_generated_update_palette(cache):
    """
    Update palette display after preview is generated.
    
    Args:
        cache: Preview cache from generate_preview_cached
    
    Returns:
        tuple: (palette_html, selected_color_state)
    """
    from core.converter import generate_palette_html
    
    if cache is None:
        return (
            "<p style='color:#888;'>生成预览后显示调色板 | Generate preview to see palette</p>",
            None  # selected_color state
        )
    
    palette = cache.get('color_palette', [])
    palette_html = generate_palette_html(palette, {}, None)
    
    return (
        palette_html,
        None  # Reset selected color
    )


def on_color_swatch_click(selected_hex):
    """
    Handle color selection from clicking palette swatch.
    
    Args:
        selected_hex: The hex color value from hidden textbox (set by JavaScript)
    
    Returns:
        tuple: (selected_color_state, display_text)
    """
    if not selected_hex or selected_hex.strip() == "":
        return None, "未选择"
    
    # Clean up the hex value
    hex_color = selected_hex.strip()
    
    return hex_color, f"✅ {hex_color}"


def on_color_dropdown_select(selected_value):
    """
    Handle color selection from dropdown.
    
    Args:
        selected_value: The hex color value selected from dropdown
    
    Returns:
        tuple: (selected_color_state, display_text)
    """
    if not selected_value:
        return None, "未选择"
    
    return selected_value, f"✅ {selected_value}"


def on_lut_change_update_colors(lut_path, cache=None):
    """
    Update available replacement colors when LUT selection changes.
    
    This callback extracts all available colors from the selected LUT
    and updates the LUT color grid HTML display, grouping by used/unused.
    
    Args:
        lut_path: Path to the selected LUT file
        cache: Optional preview cache containing color_palette
    
    Returns:
        str: HTML preview of LUT colors
    """
    from core.converter import generate_lut_color_dropdown_html
    
    if not lut_path:
        return "<p style='color:#888;'>请先选择 LUT | Select LUT first</p>"
    
    # Extract used colors from cache if available
    used_colors = set()
    if cache and 'color_palette' in cache:
        for entry in cache['color_palette']:
            used_colors.add(entry['hex'])
    
    html_preview = generate_lut_color_dropdown_html(lut_path, used_colors=used_colors)
    
    return html_preview


def on_preview_update_lut_colors(cache, lut_path):
    """
    Update LUT color display after preview is generated.
    
    Groups colors into "used in image" and "other available" sections.
    
    Args:
        cache: Preview cache containing color_palette
        lut_path: Path to the selected LUT file
    
    Returns:
        str: HTML preview of LUT colors with grouping
    """
    from core.converter import generate_lut_color_dropdown_html
    
    if not lut_path:
        return "<p style='color:#888;'>请先选择 LUT | Select LUT first</p>"
    
    # Extract used colors from cache
    used_colors = set()
    if cache and 'color_palette' in cache:
        for entry in cache['color_palette']:
            used_colors.add(entry['hex'])
    
    html_preview = generate_lut_color_dropdown_html(lut_path, used_colors=used_colors)
    
    return html_preview


def on_lut_color_swatch_click(selected_hex):
    """
    Handle LUT color selection from clicking color swatch.
    
    Args:
        selected_hex: The hex color value from hidden textbox (set by JavaScript)
    
    Returns:
        tuple: (selected_color_state, display_text)
    """
    if not selected_hex or selected_hex.strip() == "":
        return None, "未选择替换颜色"
    
    # Clean up the hex value
    hex_color = selected_hex.strip()
    
    return hex_color, f"替换为: {hex_color}"


def on_replacement_color_select(selected_value):
    """
    Handle replacement color selection from LUT color dropdown.
    
    Args:
        selected_value: The hex color value selected from dropdown
    
    Returns:
        str: Display text showing selected color
    """
    if not selected_value:
        return "未选择替换颜色"
    
    return f"替换为: {selected_value}"


# ═══════════════════════════════════════════════════════════════
# Color Highlight Callbacks
# ═══════════════════════════════════════════════════════════════

def on_highlight_color_change(highlight_hex, cache, loop_pos, add_loop,
                              loop_width, loop_length, loop_hole, loop_angle):
    """
    Handle color highlight request from palette click.
    
    When user clicks a color in the palette, this callback generates
    a preview with that color highlighted (other colors dimmed).
    
    Args:
        highlight_hex: Hex color to highlight (from hidden textbox)
        cache: Preview cache from generate_preview_cached
        loop_pos: Loop position tuple
        add_loop: Whether loop is enabled
        loop_width: Loop width in mm
        loop_length: Loop length in mm
        loop_hole: Loop hole diameter in mm
        loop_angle: Loop rotation angle
    
    Returns:
        tuple: (preview_image, status_message)
    """
    from core.converter import generate_highlight_preview
    
    if not highlight_hex or highlight_hex.strip() == "":
        # No highlight - return normal preview
        from core.converter import clear_highlight_preview
        return clear_highlight_preview(
            cache, loop_pos, add_loop,
            loop_width, loop_length, loop_hole, loop_angle
        )
    
    return generate_highlight_preview(
        cache, highlight_hex.strip(),
        loop_pos, add_loop,
        loop_width, loop_length, loop_hole, loop_angle
    )


def on_clear_highlight(cache, loop_pos, add_loop,
                       loop_width, loop_length, loop_hole, loop_angle):
    """
    Clear color highlight and restore normal preview.
    
    Args:
        cache: Preview cache from generate_preview_cached
        loop_pos: Loop position tuple
        add_loop: Whether loop is enabled
        loop_width: Loop width in mm
        loop_length: Loop length in mm
        loop_hole: Loop hole diameter in mm
        loop_angle: Loop rotation angle
    
    Returns:
        tuple: (preview_image, status_message, cleared_highlight_state)
    """
    from core.converter import clear_highlight_preview
    
    print(f"[ON_CLEAR_HIGHLIGHT] Called with cache={cache is not None}, loop_pos={loop_pos}, add_loop={add_loop}")
    
    display, status = clear_highlight_preview(
        cache, loop_pos, add_loop,
        loop_width, loop_length, loop_hole, loop_angle
    )
    
    print(f"[ON_CLEAR_HIGHLIGHT] Returning display={display is not None}, status={status}")
    
    return display, status, ""  # Clear the highlight state


# ═══════════════════════════════════════════════════════════════
# Undo Color Replacement Callback
# ═══════════════════════════════════════════════════════════════

def on_undo_color_replacement(cache, replacement_map, replacement_history,
                               loop_pos, add_loop, loop_width, loop_length, 
                               loop_hole, loop_angle):
    """
    Undo the last color replacement operation.
    
    Args:
        cache: Preview cache from generate_preview_cached
        replacement_map: Current replacement map dict
        replacement_history: History stack of previous states
        loop_pos: Loop position tuple
        add_loop: Whether loop is enabled
        loop_width: Loop width in mm
        loop_length: Loop length in mm
        loop_hole: Loop hole diameter in mm
        loop_angle: Loop rotation angle
    
    Returns:
        tuple: (preview_image, updated_cache, palette_html, updated_replacement_map, 
                updated_history, status)
    """
    from core.converter import update_preview_with_replacements, generate_palette_html
    
    if cache is None:
        return None, None, "", replacement_map, replacement_history, "❌ 请先生成预览 | Generate preview first"
    
    if not replacement_history:
        return None, cache, "", replacement_map, replacement_history, "❌ 没有可撤销的操作 | Nothing to undo"
    
    # Pop the last state from history
    new_history = replacement_history.copy()
    previous_map = new_history.pop()
    
    # Apply the previous replacement map
    display, updated_cache, palette_html = update_preview_with_replacements(
        cache, previous_map, loop_pos, add_loop,
        loop_width, loop_length, loop_hole, loop_angle
    )
    
    return display, updated_cache, palette_html, previous_map, new_history, "↩️ 已撤销 | Undone"
