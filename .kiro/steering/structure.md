# Lumina Studio - 项目结构

## 目录布局

```
lumina-studio/
├── main.py                 # 应用入口点
├── config.py               # 全局配置（打印机参数、色彩系统、i18n）
├── requirements.txt        # Python 依赖
├── user_settings.json      # 用户设置持久化
│
├── core/                   # 核心算法模块
│   ├── __init__.py         # 模块导出
│   ├── calibration.py      # 校准板生成
│   ├── extractor.py        # 颜色提取
│   ├── converter.py        # 图像转换主逻辑
│   ├── image_processing.py # 图像处理（LuminaImageProcessor）
│   ├── image_preprocessor.py # 图像预处理
│   ├── mesh_generators.py  # 网格生成器（VoxelMesher, HighFidelityMesher）
│   ├── geometry_utils.py   # 几何工具（钥匙扣挂孔等）
│   ├── color_replacement.py # 颜色替换逻辑
│   ├── vector_engine.py    # SVG 矢量引擎
│   ├── tray.py             # 系统托盘
│   └── i18n.py             # 国际化
│
├── ui/                     # 用户界面模块
│   ├── __init__.py
│   ├── layout.py           # 旧版布局
│   ├── layout_new.py       # 新版布局（当前使用）
│   ├── callbacks.py        # UI 回调函数
│   ├── styles.py           # CSS 样式
│   ├── crop_extension.py   # 裁剪扩展
│   └── palette_extension.py # 调色板扩展
│
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── helpers.py          # 通用辅助函数
│   ├── stats.py            # 统计功能
│   └── lut_manager.py      # LUT 文件管理
│
├── tests/                  # 测试目录
│   └── test_*.py           # pytest 测试文件
│
├── assets/                 # 静态资源
│   └── ref_rybw_standard.png # 参考图像
│
├── output/                 # 输出目录（生成的文件）
│
├── lut-npy预设/            # LUT 预设文件
│   ├── bambulab/
│   ├── Creality/
│   └── ...
│
└── .kiro/                  # Kiro 配置
    ├── specs/              # 功能规格文档
    └── steering/           # 引导规则
```

## 模块职责

### core/ - 核心算法

- **calibration.py**: 生成 1024 色校准板的 3MF 文件
- **extractor.py**: 从照片提取颜色数据，生成 LUT
- **converter.py**: 图像到 3D 模型的转换流程
- **image_processing.py**: `LuminaImageProcessor` 类，处理图像预处理和色彩匹配
- **mesh_generators.py**: `VoxelMesher` 和 `HighFidelityMesher` 网格生成策略

### ui/ - 用户界面

- **layout_new.py**: 主界面布局（Gradio Blocks）
- **callbacks.py**: 按钮点击等事件处理
- **styles.py**: 自定义 CSS

### utils/ - 工具

- **lut_manager.py**: LUT 文件的加载、保存、管理
- **stats.py**: 使用统计

## 设计模式

1. **策略模式**: `mesh_generators.py` 中的 `get_mesher()` 根据模式返回不同的网格生成器
2. **单例模式**: `Stats` 统计类
3. **工厂模式**: 色彩系统配置 `ColorSystem.get(mode)`

## 数据流

```
图像 → 预处理 → K-Means量化 → KD-Tree匹配 → 网格生成 → 3MF导出
                    ↑
                   LUT（校准数据）
```
