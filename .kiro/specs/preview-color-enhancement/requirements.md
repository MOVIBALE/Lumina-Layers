# Requirements Document

## Introduction

本功能增强 Lumina Studio 的预览和颜色管理能力，包含两个核心改进：

1. **预览色彩细节同步**：修复预览图片不使用用户选择的"色彩细节"值的问题，确保预览效果与最终 3MF 一致
2. **颜色置换功能**：允许用户在预览后查看和替换特定颜色，实现精细的颜色微调

## Glossary

- **Preview_Generator**: 负责生成 2D 预览图像的模块，位于 `generate_preview_cached` 函数
- **Color_Detail_Slider**: UI 中的"色彩细节"滑块控件，控制 K-Means 量化的颜色数量（8-256）
- **Quantize_Colors**: K-Means 聚类算法使用的目标颜色数量参数
- **Color_Palette**: 预览图像中使用的所有唯一颜色的集合
- **Color_Replacement_Map**: 存储原始颜色到替换颜色映射关系的数据结构
- **LUT_Matcher**: 使用 KD-Tree 将颜色匹配到打印机校准数据的组件
- **Material_Matrix**: 存储每个像素对应材料堆叠顺序的三维数组

## Requirements

### Requirement 1: 预览使用用户选择的色彩细节值

**User Story:** As a user, I want the preview image to use my selected color detail value, so that I can accurately preview how the final 3MF will look.

#### Acceptance Criteria

1. WHEN the user clicks "生成预览" button, THE Preview_Generator SHALL use the value from Color_Detail_Slider as the quantize_colors parameter
2. WHEN the Color_Detail_Slider value changes, THE Preview_Generator SHALL reflect this change in subsequent preview generations
3. THE Preview_Generator SHALL accept quantize_colors values in the range of 8 to 256
4. WHEN a preview is generated with a specific quantize_colors value, THE resulting preview image SHALL contain at most that many unique colors (after LUT matching)

### Requirement 2: 显示预览图像的颜色调色板

**User Story:** As a user, I want to see all colors used in the preview image, so that I can understand the color composition and identify colors I want to change.

#### Acceptance Criteria

1. WHEN a preview is successfully generated, THE system SHALL extract and display all unique colors from the preview image
2. THE Color_Palette display SHALL show each color as a clickable color swatch
3. WHEN displaying the Color_Palette, THE system SHALL show the count of pixels using each color
4. THE Color_Palette SHALL be sorted by pixel count in descending order (most used colors first)
5. WHEN the preview is regenerated, THE Color_Palette SHALL update to reflect the new colors

### Requirement 3: 颜色选择和替换

**User Story:** As a user, I want to select a color from the palette and replace it with another color, so that I can fine-tune the color output.

#### Acceptance Criteria

1. WHEN a user clicks on a color swatch in the Color_Palette, THE system SHALL mark that color as selected for replacement
2. WHEN a color is selected, THE system SHALL display a color picker allowing the user to choose a replacement color
3. WHEN the user confirms a color replacement, THE system SHALL store the mapping in Color_Replacement_Map
4. THE system SHALL allow multiple color replacements to be defined before generating the final 3MF
5. WHEN a color replacement is defined, THE system SHALL visually indicate the replacement in the Color_Palette (showing original → new)
6. THE system SHALL provide a way to remove a color replacement from the Color_Replacement_Map

### Requirement 4: 应用颜色置换到最终模型

**User Story:** As a user, I want my color replacements to be applied when generating the final 3MF, so that the printed result matches my customizations.

#### Acceptance Criteria

1. WHEN generating the final 3MF with color replacements defined, THE system SHALL apply the Color_Replacement_Map before LUT matching
2. FOR ALL pixels matching a replaced color, THE system SHALL substitute the replacement color before material assignment
3. WHEN color replacements are applied, THE system SHALL re-match the replacement colors to the LUT to find the closest printable color
4. IF a replacement color has no close match in the LUT, THE system SHALL use the nearest available color and log a warning
5. THE system SHALL preserve the Color_Replacement_Map in the preview cache so it persists across preview regenerations

### Requirement 5: 预览更新反映颜色置换

**User Story:** As a user, I want to see the effect of my color replacements in the preview, so that I can verify the changes before generating the final model.

#### Acceptance Criteria

1. WHEN a color replacement is added or modified, THE system SHALL update the preview image to show the replacement color
2. WHEN the preview is updated with replacements, THE system SHALL re-match replacement colors to the LUT
3. THE preview update with color replacements SHALL complete within 2 seconds for images up to 1000x1000 pixels
4. WHEN a color replacement is removed, THE system SHALL revert the preview to show the original color
