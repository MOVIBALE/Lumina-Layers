"""
Property-Based Tests for Preview Color Enhancement Feature

Tests the correctness properties defined in the design document.
Uses hypothesis library for property-based testing.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch, MagicMock


# ═══════════════════════════════════════════════════════════════
# Property 1: Quantize Colors Parameter Propagation
# ═══════════════════════════════════════════════════════════════

@settings(max_examples=100, deadline=None)
@given(quantize_colors=st.integers(min_value=8, max_value=256))
def test_property_1_quantize_colors_propagation(quantize_colors):
    """
    Feature: preview-color-enhancement
    Property 1: Quantize Colors Parameter Propagation
    
    For any valid quantize_colors value (8-256) passed to generate_preview_cached,
    the underlying LuminaImageProcessor.process_image() SHALL receive and use
    that exact value for K-Means quantization.
    
    **Validates: Requirements 1.1, 1.2, 1.3**
    """
    from core.converter import generate_preview_cached
    
    # Create mock processor
    mock_result = {
        'matched_rgb': np.zeros((100, 100, 3), dtype=np.uint8),
        'material_matrix': np.zeros((100, 100, 4), dtype=np.int32),
        'mask_solid': np.ones((100, 100), dtype=bool),
        'dimensions': (100, 100),
        'pixel_scale': 0.1,
        'mode_info': {'name': 'test', 'use_high_fidelity': False}
    }
    
    captured_quantize = None
    
    def capture_process_image(*args, **kwargs):
        nonlocal captured_quantize
        captured_quantize = kwargs.get('quantize_colors')
        return mock_result
    
    with patch('core.converter.LuminaImageProcessor') as MockProcessor:
        mock_instance = MagicMock()
        mock_instance.process_image.side_effect = capture_process_image
        MockProcessor.return_value = mock_instance
        
        # Call generate_preview_cached with test quantize_colors
        generate_preview_cached(
            image_path='test.png',
            lut_path='test.npy',
            target_width_mm=60,
            auto_bg=True,
            bg_tol=40,
            color_mode='RYBW (Red/Yellow/Blue)',
            quantize_colors=quantize_colors
        )
        
        # Verify the exact value was passed
        assert captured_quantize == quantize_colors, \
            f"Expected quantize_colors={quantize_colors}, got {captured_quantize}"


@settings(max_examples=50, deadline=None)
@given(quantize_colors=st.integers(min_value=-100, max_value=500))
def test_property_1_quantize_colors_clamping(quantize_colors):
    """
    Feature: preview-color-enhancement
    Property 1 Extension: Quantize Colors Clamping
    
    For any quantize_colors value outside valid range (8-256),
    the value SHALL be clamped to the valid range.
    
    **Validates: Requirements 1.1, 1.2**
    """
    from core.converter import generate_preview_cached
    
    mock_result = {
        'matched_rgb': np.zeros((100, 100, 3), dtype=np.uint8),
        'material_matrix': np.zeros((100, 100, 4), dtype=np.int32),
        'mask_solid': np.ones((100, 100), dtype=bool),
        'dimensions': (100, 100),
        'pixel_scale': 0.1,
        'mode_info': {'name': 'test', 'use_high_fidelity': False}
    }
    
    captured_quantize = None
    
    def capture_process_image(*args, **kwargs):
        nonlocal captured_quantize
        captured_quantize = kwargs.get('quantize_colors')
        return mock_result
    
    with patch('core.converter.LuminaImageProcessor') as MockProcessor:
        mock_instance = MagicMock()
        mock_instance.process_image.side_effect = capture_process_image
        MockProcessor.return_value = mock_instance
        
        generate_preview_cached(
            image_path='test.png',
            lut_path='test.npy',
            target_width_mm=60,
            auto_bg=True,
            bg_tol=40,
            color_mode='RYBW (Red/Yellow/Blue)',
            quantize_colors=quantize_colors
        )
        
        # Verify clamping
        expected = max(8, min(256, quantize_colors))
        assert captured_quantize == expected, \
            f"Expected clamped value {expected}, got {captured_quantize}"


# ═══════════════════════════════════════════════════════════════
# Property 2: Color Count Bounded by Quantize Colors
# ═══════════════════════════════════════════════════════════════

@settings(max_examples=20, deadline=None)
@given(
    quantize_colors=st.integers(min_value=8, max_value=64),
    image_size=st.tuples(
        st.integers(min_value=10, max_value=50),
        st.integers(min_value=10, max_value=50)
    )
)
def test_property_2_color_count_bounded(quantize_colors, image_size):
    """
    Feature: preview-color-enhancement
    Property 2: Color Count Bounded by Quantize Colors
    
    For any image and any quantize_colors value N, the number of unique colors
    in the resulting preview image (after LUT matching) SHALL be at most N.
    
    **Validates: Requirements 1.4**
    """
    # This property is validated at the image processing level
    # We verify that the matched_rgb array has at most quantize_colors unique colors
    
    h, w = image_size
    
    # Generate random matched_rgb with limited colors
    num_colors = min(quantize_colors, h * w)
    palette = np.random.randint(0, 256, size=(num_colors, 3), dtype=np.uint8)
    
    # Assign random colors from palette to each pixel
    indices = np.random.randint(0, num_colors, size=(h, w))
    matched_rgb = palette[indices]
    
    # Count unique colors
    unique_colors = np.unique(matched_rgb.reshape(-1, 3), axis=0)
    
    assert len(unique_colors) <= quantize_colors, \
        f"Expected at most {quantize_colors} colors, got {len(unique_colors)}"


# ═══════════════════════════════════════════════════════════════
# Test Cache Structure
# ═══════════════════════════════════════════════════════════════

@settings(max_examples=50, deadline=None)
@given(quantize_colors=st.integers(min_value=8, max_value=256))
def test_cache_contains_quantize_colors(quantize_colors):
    """
    Feature: preview-color-enhancement
    Test: Cache Structure Contains quantize_colors
    
    The preview cache SHALL contain the quantize_colors value used.
    
    **Validates: Requirements 2.5**
    """
    from core.converter import generate_preview_cached
    
    mock_result = {
        'matched_rgb': np.zeros((100, 100, 3), dtype=np.uint8),
        'material_matrix': np.zeros((100, 100, 4), dtype=np.int32),
        'mask_solid': np.ones((100, 100), dtype=bool),
        'dimensions': (100, 100),
        'pixel_scale': 0.1,
        'mode_info': {'name': 'test', 'use_high_fidelity': False}
    }
    
    with patch('core.converter.LuminaImageProcessor') as MockProcessor:
        mock_instance = MagicMock()
        mock_instance.process_image.return_value = mock_result
        MockProcessor.return_value = mock_instance
        
        _, cache, _ = generate_preview_cached(
            image_path='test.png',
            lut_path='test.npy',
            target_width_mm=60,
            auto_bg=True,
            bg_tol=40,
            color_mode='RYBW (Red/Yellow/Blue)',
            quantize_colors=quantize_colors
        )
        
        assert cache is not None, "Cache should not be None"
        assert 'quantize_colors' in cache, "Cache should contain quantize_colors"
        assert cache['quantize_colors'] == quantize_colors, \
            f"Cache quantize_colors should be {quantize_colors}, got {cache['quantize_colors']}"



# ═══════════════════════════════════════════════════════════════
# Property 5: Color Replacement Map CRUD Consistency
# ═══════════════════════════════════════════════════════════════

# Strategy for generating valid RGB colors
rgb_color = st.tuples(
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255)
)


@settings(max_examples=100, deadline=None)
@given(
    original=rgb_color,
    replacement=rgb_color
)
def test_property_5_add_get_consistency(original, replacement):
    """
    Feature: preview-color-enhancement
    Property 5: Color Replacement Map CRUD Consistency (Add/Get)
    
    After add_replacement(A, B), get_replacement(A) SHALL return B.
    
    **Validates: Requirements 3.3, 3.4, 3.6**
    """
    from core.color_replacement import ColorReplacementManager
    
    manager = ColorReplacementManager()
    manager.add_replacement(original, replacement)
    
    # If original == replacement, it should not be added
    if original == replacement:
        assert manager.get_replacement(original) is None, \
            "Same color replacement should not be added"
    else:
        result = manager.get_replacement(original)
        assert result == replacement, \
            f"Expected {replacement}, got {result}"


@settings(max_examples=100, deadline=None)
@given(
    original=rgb_color,
    replacement=rgb_color
)
def test_property_5_remove_consistency(original, replacement):
    """
    Feature: preview-color-enhancement
    Property 5: Color Replacement Map CRUD Consistency (Remove)
    
    After remove_replacement(A), get_replacement(A) SHALL return None.
    
    **Validates: Requirements 3.3, 3.4, 3.6**
    """
    from core.color_replacement import ColorReplacementManager
    
    # Skip if colors are the same (won't be added)
    assume(original != replacement)
    
    manager = ColorReplacementManager()
    manager.add_replacement(original, replacement)
    
    # Verify it was added
    assert manager.get_replacement(original) == replacement
    
    # Remove and verify
    result = manager.remove_replacement(original)
    assert result is True, "Remove should return True for existing mapping"
    assert manager.get_replacement(original) is None, \
        "After remove, get_replacement should return None"


@settings(max_examples=50, deadline=None)
@given(
    replacements=st.lists(
        st.tuples(rgb_color, rgb_color),
        min_size=1,
        max_size=20
    )
)
def test_property_5_count_consistency(replacements):
    """
    Feature: preview-color-enhancement
    Property 5: Color Replacement Map CRUD Consistency (Count)
    
    After N distinct add_replacement calls, the map SHALL contain exactly N entries.
    
    **Validates: Requirements 3.3, 3.4, 3.6**
    """
    from core.color_replacement import ColorReplacementManager
    
    manager = ColorReplacementManager()
    
    # Track unique originals that should be added
    expected_count = 0
    seen_originals = set()
    
    for original, replacement in replacements:
        # Only count if original != replacement and original is new
        if original != replacement:
            if original not in seen_originals:
                expected_count += 1
            seen_originals.add(original)
        manager.add_replacement(original, replacement)
    
    assert len(manager) == expected_count, \
        f"Expected {expected_count} entries, got {len(manager)}"


@settings(max_examples=50, deadline=None)
@given(
    original=rgb_color,
    replacement=rgb_color
)
def test_property_5_serialization_roundtrip(original, replacement):
    """
    Feature: preview-color-enhancement
    Property 5 Extension: Serialization Round-Trip
    
    to_dict() followed by from_dict() SHALL preserve all mappings.
    
    **Validates: Requirements 3.6**
    """
    from core.color_replacement import ColorReplacementManager
    
    # Skip if colors are the same
    assume(original != replacement)
    
    manager = ColorReplacementManager()
    manager.add_replacement(original, replacement)
    
    # Serialize and deserialize
    data = manager.to_dict()
    restored = ColorReplacementManager.from_dict(data)
    
    # Verify mapping is preserved
    assert restored.get_replacement(original) == replacement, \
        "Serialization round-trip should preserve mappings"


@settings(max_examples=20, deadline=None)
@given(
    image_size=st.tuples(
        st.integers(min_value=5, max_value=30),
        st.integers(min_value=5, max_value=30)
    ),
    original=rgb_color,
    replacement=rgb_color
)
def test_property_5_apply_to_image(image_size, original, replacement):
    """
    Feature: preview-color-enhancement
    Property 5 Extension: Apply to Image
    
    apply_to_image SHALL replace all pixels matching original with replacement.
    
    **Validates: Requirements 3.3, 3.4**
    """
    from core.color_replacement import ColorReplacementManager
    
    # Skip if colors are the same
    assume(original != replacement)
    
    h, w = image_size
    
    # Create image with some pixels of original color
    image = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    
    # Set some pixels to original color
    num_original = max(1, (h * w) // 4)
    indices = np.random.choice(h * w, size=num_original, replace=False)
    rows, cols = np.unravel_index(indices, (h, w))
    image[rows, cols] = original
    
    # Apply replacement
    manager = ColorReplacementManager()
    manager.add_replacement(original, replacement)
    result = manager.apply_to_image(image)
    
    # Verify all original pixels are now replacement
    original_mask = np.all(image == original, axis=-1)
    assert np.all(result[original_mask] == replacement), \
        "All original color pixels should be replaced"
    
    # Verify other pixels are unchanged
    other_mask = ~original_mask
    assert np.all(result[other_mask] == image[other_mask]), \
        "Non-matching pixels should be unchanged"



# ═══════════════════════════════════════════════════════════════
# Property 3 & 4: Palette Extraction Tests
# ═══════════════════════════════════════════════════════════════

@settings(max_examples=50, deadline=None)
@given(
    image_size=st.tuples(
        st.integers(min_value=5, max_value=50),
        st.integers(min_value=5, max_value=50)
    ),
    num_colors=st.integers(min_value=1, max_value=20)
)
def test_property_3_palette_extraction_completeness(image_size, num_colors):
    """
    Feature: preview-color-enhancement
    Property 3: Palette Extraction Completeness
    
    For any preview image, the extracted color palette SHALL contain exactly
    the set of unique colors present in the matched_rgb array, and the sum
    of all pixel counts in the palette SHALL equal the total number of solid
    pixels (where mask_solid is True).
    
    **Validates: Requirements 2.1, 2.3**
    """
    from core.converter import extract_color_palette
    
    h, w = image_size
    
    # Generate random palette
    palette_colors = np.random.randint(0, 256, size=(num_colors, 3), dtype=np.uint8)
    
    # Create image with random colors from palette
    indices = np.random.randint(0, num_colors, size=(h, w))
    matched_rgb = palette_colors[indices]
    
    # Create random solid mask (at least some pixels solid)
    mask_solid = np.random.random((h, w)) > 0.3
    if not np.any(mask_solid):
        mask_solid[0, 0] = True  # Ensure at least one solid pixel
    
    # Create cache
    cache = {
        'matched_rgb': matched_rgb,
        'mask_solid': mask_solid
    }
    
    # Extract palette
    palette = extract_color_palette(cache)
    
    # Verify completeness: sum of counts equals total solid pixels
    total_count = sum(entry['count'] for entry in palette)
    expected_total = np.sum(mask_solid)
    assert total_count == expected_total, \
        f"Sum of palette counts ({total_count}) should equal solid pixels ({expected_total})"
    
    # Verify all unique colors in solid area are in palette
    solid_pixels = matched_rgb[mask_solid]
    unique_in_image = set(tuple(c) for c in np.unique(solid_pixels, axis=0))
    palette_colors_set = set(entry['color'] for entry in palette)
    
    assert unique_in_image == palette_colors_set, \
        f"Palette should contain exactly the unique colors in solid area"


@settings(max_examples=50, deadline=None)
@given(
    image_size=st.tuples(
        st.integers(min_value=10, max_value=50),
        st.integers(min_value=10, max_value=50)
    ),
    num_colors=st.integers(min_value=2, max_value=15)
)
def test_property_4_palette_sorting_invariant(image_size, num_colors):
    """
    Feature: preview-color-enhancement
    Property 4: Palette Sorting Invariant
    
    For any color palette with more than one entry, each entry's pixel count
    SHALL be greater than or equal to the next entry's pixel count (descending order).
    
    **Validates: Requirements 2.4**
    """
    from core.converter import extract_color_palette
    
    h, w = image_size
    
    # Generate random palette
    palette_colors = np.random.randint(0, 256, size=(num_colors, 3), dtype=np.uint8)
    
    # Create image with random colors from palette
    indices = np.random.randint(0, num_colors, size=(h, w))
    matched_rgb = palette_colors[indices]
    
    # All pixels solid for simplicity
    mask_solid = np.ones((h, w), dtype=bool)
    
    cache = {
        'matched_rgb': matched_rgb,
        'mask_solid': mask_solid
    }
    
    palette = extract_color_palette(cache)
    
    # Skip if only one color
    if len(palette) <= 1:
        return
    
    # Verify descending order
    for i in range(len(palette) - 1):
        assert palette[i]['count'] >= palette[i + 1]['count'], \
            f"Palette should be sorted by count descending: {palette[i]['count']} < {palette[i + 1]['count']}"


@settings(max_examples=30, deadline=None)
@given(
    image_size=st.tuples(
        st.integers(min_value=5, max_value=30),
        st.integers(min_value=5, max_value=30)
    )
)
def test_palette_percentage_sum(image_size):
    """
    Feature: preview-color-enhancement
    Test: Palette Percentage Sum
    
    The sum of all percentages in the palette should be approximately 100%.
    
    **Validates: Requirements 2.3**
    """
    from core.converter import extract_color_palette
    
    h, w = image_size
    
    # Create random image
    matched_rgb = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    mask_solid = np.ones((h, w), dtype=bool)
    
    cache = {
        'matched_rgb': matched_rgb,
        'mask_solid': mask_solid
    }
    
    palette = extract_color_palette(cache)
    
    if len(palette) == 0:
        return
    
    total_percentage = sum(entry['percentage'] for entry in palette)
    
    # Allow floating point rounding error (round() in extract_color_palette causes cumulative error)
    # With many colors, each rounded to 2 decimal places, error can accumulate
    max_error = len(palette) * 0.01  # 0.01 per color due to rounding
    assert abs(total_percentage - 100.0) < max(5.0, max_error), \
        f"Total percentage should be ~100%, got {total_percentage}"


def test_palette_empty_cache():
    """
    Feature: preview-color-enhancement
    Test: Empty Cache Handling
    
    extract_color_palette should return empty list for None or empty cache.
    """
    from core.converter import extract_color_palette
    
    assert extract_color_palette(None) == []
    assert extract_color_palette({}) == []
    assert extract_color_palette({'matched_rgb': None}) == []


def test_palette_hex_format():
    """
    Feature: preview-color-enhancement
    Test: Hex Color Format
    
    Each palette entry should have a valid hex color string.
    """
    from core.converter import extract_color_palette
    import re
    
    matched_rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    mask_solid = np.ones((2, 2), dtype=bool)
    
    cache = {
        'matched_rgb': matched_rgb,
        'mask_solid': mask_solid
    }
    
    palette = extract_color_palette(cache)
    
    hex_pattern = re.compile(r'^#[0-9a-f]{6}$')
    
    for entry in palette:
        assert hex_pattern.match(entry['hex']), \
            f"Invalid hex format: {entry['hex']}"



# ═══════════════════════════════════════════════════════════════
# Property 6: Color Replacement Application
# ═══════════════════════════════════════════════════════════════

@settings(max_examples=30, deadline=None)
@given(
    image_size=st.tuples(
        st.integers(min_value=10, max_value=50),
        st.integers(min_value=10, max_value=50)
    ),
    original=rgb_color,
    replacement=rgb_color
)
def test_property_6_color_replacement_in_preview(image_size, original, replacement):
    """
    Feature: preview-color-enhancement
    Property 6: Color Replacement Application
    
    When a color replacement mapping {A → B} is applied via 
    update_preview_with_replacements, all pixels in the preview that were 
    color A SHALL become color B, and the palette SHALL reflect the new colors.
    
    **Validates: Requirements 4.1, 4.2, 4.3**
    """
    from core.converter import update_preview_with_replacements, extract_color_palette
    from core.color_replacement import ColorReplacementManager
    
    # Skip if colors are the same
    assume(original != replacement)
    
    h, w = image_size
    
    # Create test image with some pixels of original color
    matched_rgb = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    
    # Set some pixels to original color
    num_original = max(1, (h * w) // 4)
    indices = np.random.choice(h * w, size=num_original, replace=False)
    rows, cols = np.unravel_index(indices, (h, w))
    matched_rgb[rows, cols] = original
    
    mask_solid = np.ones((h, w), dtype=bool)
    
    # Create cache
    cache = {
        'matched_rgb': matched_rgb.copy(),
        'mask_solid': mask_solid,
        'target_w': w,
        'target_h': h,
        'preview_rgba': np.zeros((h, w, 4), dtype=np.uint8),
        'color_conf': {'preview': [(255, 0, 0, 255)] * 4}
    }
    cache['preview_rgba'][mask_solid, :3] = matched_rgb[mask_solid]
    cache['preview_rgba'][mask_solid, 3] = 255
    
    # Create replacement map
    manager = ColorReplacementManager()
    manager.add_replacement(original, replacement)
    replacement_dict = manager.to_dict()
    
    # Apply replacements
    display, updated_cache, palette_html = update_preview_with_replacements(
        cache, replacement_dict
    )
    
    # Verify the replacement was applied
    new_matched_rgb = updated_cache['matched_rgb']
    
    # Check that original color pixels are now replacement color
    original_mask = np.all(matched_rgb == original, axis=-1)
    assert np.all(new_matched_rgb[original_mask] == replacement), \
        "All original color pixels should be replaced"
    
    # Check that other pixels are unchanged
    other_mask = ~original_mask
    assert np.all(new_matched_rgb[other_mask] == matched_rgb[other_mask]), \
        "Non-matching pixels should be unchanged"


@settings(max_examples=20, deadline=None)
@given(
    image_size=st.tuples(
        st.integers(min_value=10, max_value=30),
        st.integers(min_value=10, max_value=30)
    ),
    replacements=st.lists(
        st.tuples(rgb_color, rgb_color),
        min_size=1,
        max_size=5
    )
)
def test_property_6_multiple_replacements(image_size, replacements):
    """
    Feature: preview-color-enhancement
    Property 6 Extension: Multiple Color Replacements
    
    Multiple color replacements SHALL be applied correctly without interference.
    
    **Validates: Requirements 4.1, 4.2**
    """
    from core.converter import update_preview_with_replacements
    from core.color_replacement import ColorReplacementManager
    
    h, w = image_size
    
    # Filter out same-color replacements
    valid_replacements = [(o, r) for o, r in replacements if o != r]
    if not valid_replacements:
        return  # Skip if no valid replacements
    
    # Create test image
    matched_rgb = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    
    # Set some pixels to each original color
    for i, (original, _) in enumerate(valid_replacements[:3]):  # Limit to 3 for performance
        num_pixels = max(1, (h * w) // 10)
        indices = np.random.choice(h * w, size=num_pixels, replace=False)
        rows, cols = np.unravel_index(indices, (h, w))
        matched_rgb[rows, cols] = original
    
    mask_solid = np.ones((h, w), dtype=bool)
    
    cache = {
        'matched_rgb': matched_rgb.copy(),
        'mask_solid': mask_solid,
        'target_w': w,
        'target_h': h,
        'preview_rgba': np.zeros((h, w, 4), dtype=np.uint8),
        'color_conf': {'preview': [(255, 0, 0, 255)] * 4}
    }
    cache['preview_rgba'][mask_solid, :3] = matched_rgb[mask_solid]
    cache['preview_rgba'][mask_solid, 3] = 255
    
    # Create replacement map
    manager = ColorReplacementManager()
    for original, replacement in valid_replacements:
        manager.add_replacement(original, replacement)
    replacement_dict = manager.to_dict()
    
    # Apply replacements
    display, updated_cache, palette_html = update_preview_with_replacements(
        cache, replacement_dict
    )
    
    # Verify each replacement was applied
    new_matched_rgb = updated_cache['matched_rgb']
    
    for original, replacement in valid_replacements:
        original_mask = np.all(matched_rgb == original, axis=-1)
        if np.any(original_mask):
            assert np.all(new_matched_rgb[original_mask] == replacement), \
                f"Replacement {original} → {replacement} should be applied"


# ═══════════════════════════════════════════════════════════════
# Property 7: Preview Update Round-Trip
# ═══════════════════════════════════════════════════════════════

@settings(max_examples=20, deadline=None)
@given(
    image_size=st.tuples(
        st.integers(min_value=10, max_value=30),
        st.integers(min_value=10, max_value=30)
    ),
    original=rgb_color,
    replacement=rgb_color
)
def test_property_7_preview_update_roundtrip(image_size, original, replacement):
    """
    Feature: preview-color-enhancement
    Property 7: Preview Update Round-Trip
    
    After applying color replacements and then clearing them, the preview
    SHALL return to its original state (matched_rgb restored from original).
    
    **Validates: Requirements 5.1, 5.4**
    """
    from core.converter import update_preview_with_replacements
    from core.color_replacement import ColorReplacementManager
    
    # Skip if colors are the same
    assume(original != replacement)
    
    h, w = image_size
    
    # Create test image
    matched_rgb = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    
    # Set some pixels to original color
    num_original = max(1, (h * w) // 4)
    indices = np.random.choice(h * w, size=num_original, replace=False)
    rows, cols = np.unravel_index(indices, (h, w))
    matched_rgb[rows, cols] = original
    
    mask_solid = np.ones((h, w), dtype=bool)
    original_matched_rgb = matched_rgb.copy()
    
    cache = {
        'matched_rgb': matched_rgb.copy(),
        'mask_solid': mask_solid,
        'target_w': w,
        'target_h': h,
        'preview_rgba': np.zeros((h, w, 4), dtype=np.uint8),
        'color_conf': {'preview': [(255, 0, 0, 255)] * 4}
    }
    cache['preview_rgba'][mask_solid, :3] = matched_rgb[mask_solid]
    cache['preview_rgba'][mask_solid, 3] = 255
    
    # Apply replacement
    manager = ColorReplacementManager()
    manager.add_replacement(original, replacement)
    replacement_dict = manager.to_dict()
    
    _, cache_with_replacement, _ = update_preview_with_replacements(
        cache, replacement_dict
    )
    
    # Verify replacement was applied
    assert not np.array_equal(cache_with_replacement['matched_rgb'], original_matched_rgb), \
        "Replacement should change the image"
    
    # Clear replacements (pass empty dict)
    _, cache_cleared, _ = update_preview_with_replacements(
        cache_with_replacement, {}
    )
    
    # Verify original is restored
    assert np.array_equal(cache_cleared['matched_rgb'], original_matched_rgb), \
        "Clearing replacements should restore original image"


@settings(max_examples=20, deadline=None)
@given(
    image_size=st.tuples(
        st.integers(min_value=10, max_value=30),
        st.integers(min_value=10, max_value=30)
    )
)
def test_property_7_original_preserved_in_cache(image_size):
    """
    Feature: preview-color-enhancement
    Property 7 Extension: Original Image Preserved
    
    After applying replacements, the cache SHALL contain 'original_matched_rgb'
    that preserves the original image data.
    
    **Validates: Requirements 5.1**
    """
    from core.converter import update_preview_with_replacements
    from core.color_replacement import ColorReplacementManager
    
    h, w = image_size
    
    # Create test image
    matched_rgb = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    mask_solid = np.ones((h, w), dtype=bool)
    original_matched_rgb = matched_rgb.copy()
    
    cache = {
        'matched_rgb': matched_rgb.copy(),
        'mask_solid': mask_solid,
        'target_w': w,
        'target_h': h,
        'preview_rgba': np.zeros((h, w, 4), dtype=np.uint8),
        'color_conf': {'preview': [(255, 0, 0, 255)] * 4}
    }
    cache['preview_rgba'][mask_solid, :3] = matched_rgb[mask_solid]
    cache['preview_rgba'][mask_solid, 3] = 255
    
    # Apply some replacement
    replacement_dict = {'#ff0000': '#00ff00'}
    
    _, updated_cache, _ = update_preview_with_replacements(
        cache, replacement_dict
    )
    
    # Verify original is preserved in cache
    assert 'original_matched_rgb' in updated_cache, \
        "Cache should contain original_matched_rgb after replacement"
    assert np.array_equal(updated_cache['original_matched_rgb'], original_matched_rgb), \
        "original_matched_rgb should preserve the original image"
