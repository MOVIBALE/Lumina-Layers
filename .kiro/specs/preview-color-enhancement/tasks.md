# Implementation Plan: Preview Color Enhancement

## Overview

本实现计划将预览颜色增强功能分解为可执行的编码任务。采用渐进式实现策略：先修复核心问题（预览色彩细节同步），再添加新功能（颜色置换系统）。

## Tasks

- [x] 1. 修复预览色彩细节同步问题
  - [x] 1.1 修改 `generate_preview_cached` 函数接收 `quantize_colors` 参数
    - 在 `core/converter.py` 中更新函数签名
    - 将硬编码的 `quantize_colors=16` 替换为参数传递
    - 确保参数正确传递给 `processor.process_image()`
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 1.2 更新 UI 绑定传递 `quantize_colors` 参数
    - 在 `ui/layout.py` 中修改 `conv_preview_btn.click()` 的 inputs
    - 添加 `conv_quantize_count` 到预览按钮的输入参数列表
    - _Requirements: 1.1, 1.2_

  - [x] 1.3 编写属性测试验证参数传递
    - **Property 1: Quantize Colors Parameter Propagation**
    - **Validates: Requirements 1.1, 1.2, 1.3**

- [x] 2. Checkpoint - 验证预览色彩细节修复
  - 确保所有测试通过，如有问题请询问用户

- [x] 3. 实现 ColorReplacementManager 类
  - [x] 3.1 创建 `core/color_replacement.py` 文件
    - 实现 `ColorReplacementManager` 类
    - 包含 `add_replacement`, `remove_replacement`, `get_replacement` 方法
    - 实现 `apply_to_image` 方法用于批量替换
    - 实现 `to_dict` 和 `from_dict` 序列化方法
    - _Requirements: 3.3, 3.4, 3.6_

  - [x] 3.2 编写属性测试验证 CRUD 操作
    - **Property 5: Color Replacement Map CRUD Consistency**
    - **Validates: Requirements 3.3, 3.4, 3.6**

- [x] 4. 实现调色板提取功能
  - [x] 4.1 添加 `extract_color_palette` 函数
    - 在 `core/converter.py` 中实现
    - 从 `matched_rgb` 数组提取唯一颜色
    - 计算每种颜色的像素数量和百分比
    - 按像素数量降序排序
    - _Requirements: 2.1, 2.3, 2.4_

  - [x] 4.2 更新 `generate_preview_cached` 返回调色板数据
    - 在缓存中添加 `color_palette` 字段
    - 在缓存中添加 `quantize_colors` 字段
    - _Requirements: 2.1, 2.5_

  - [x] 4.3 编写属性测试验证调色板提取
    - **Property 3: Palette Extraction Completeness**
    - **Property 4: Palette Sorting Invariant**
    - **Validates: Requirements 2.1, 2.3, 2.4**

- [x] 5. 实现颜色置换应用逻辑
  - [x] 5.1 在 `generate_preview_cached` 中集成颜色置换
    - 添加 `color_replacement_map` 参数
    - 在 LUT 匹配前应用颜色替换
    - 保存原始 `matched_rgb` 到缓存
    - _Requirements: 4.1, 4.2, 4.3, 4.5_

  - [x] 5.2 在 `generate_final_model` 中集成颜色置换
    - 添加 `color_replacement_map` 参数
    - 确保最终模型使用替换后的颜色
    - _Requirements: 4.1, 4.2_

  - [x] 5.3 编写属性测试验证颜色置换应用
    - **Property 6: Color Replacement Application**
    - **Validates: Requirements 4.1, 4.2, 4.3**

- [x] 6. Checkpoint - 验证核心逻辑实现
  - 确保所有测试通过，如有问题请询问用户

- [x] 7. 实现 UI 组件
  - [x] 7.1 添加调色板显示区域
    - 在 `ui/layout.py` 的转换器 Tab 中添加调色板 Gallery
    - 显示颜色色块、十六进制值和像素数量
    - 实现点击色块选择颜色的交互
    - _Requirements: 2.1, 2.2, 3.1_

  - [x] 7.2 添加颜色替换控制面板
    - 添加 ColorPicker 组件选择替换颜色
    - 添加"应用替换"和"清除所有"按钮
    - 添加 State 组件存储选中颜色和替换映射
    - _Requirements: 3.2, 3.5, 3.6_

  - [x] 7.3 实现 UI 回调函数
    - 在 `ui/callbacks.py` 中添加颜色选择回调
    - 添加颜色替换应用回调
    - 添加清除替换回调
    - _Requirements: 3.1, 3.3, 5.1_

  - [x] 7.4 绑定 UI 事件
    - 连接调色板点击事件到选择回调
    - 连接替换按钮到应用回调
    - 连接预览更新到颜色置换逻辑
    - _Requirements: 5.1, 5.4_

- [x] 8. 实现预览实时更新
  - [x] 8.1 添加 `update_preview_with_replacements` 函数
    - 在 `core/converter.py` 中实现
    - 基于缓存数据和替换映射快速更新预览
    - 不重新处理图像，只替换颜色并重新渲染
    - _Requirements: 5.1, 5.2, 5.4_

  - [x] 8.2 编写属性测试验证预览更新
    - **Property 7: Preview Update Round-Trip**
    - **Validates: Requirements 5.1, 5.4**

- [x] 9. Final Checkpoint - 完整功能验证
  - 确保所有测试通过，如有问题请询问用户

## Notes

- 所有任务均为必需，包括属性测试
- 每个任务引用具体的需求条款以确保可追溯性
- Checkpoint 任务用于增量验证，确保每个阶段的正确性
- 属性测试使用 hypothesis 库，每个测试运行 100 次迭代
