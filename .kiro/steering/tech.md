# Lumina Studio - 技术栈

## 核心技术

| 组件       | 技术                   |
| ---------- | ---------------------- |
| 语言       | Python 3.x             |
| UI 框架    | Gradio 6.0+            |
| 数值计算   | NumPy                  |
| 几何引擎   | Trimesh 4.0+           |
| 计算机视觉 | OpenCV (opencv-python) |
| 色彩匹配   | SciPy KDTree           |
| 图像处理   | Pillow                 |
| 矢量处理   | svgelements, shapely   |

## 依赖项

```
gradio>=6.0.0
numpy
opencv-python
Pillow
trimesh>=4.0.0
scipy
pystray
pytz
networkx
lxml
svglib
reportlab
svgelements>=1.9.0
shapely>=2.0.0
```

## 常用命令

### 启动应用

```bash
python main.py
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
pytest tests/
```

## 配置

- **config.py**: 全局配置（打印机参数、色彩系统、i18n）
- **user_settings.json**: 用户设置持久化

## 输出格式

- **.3mf**: 3D 模型文件（供切片器使用）
- **.npy**: NumPy 数组格式的 LUT 校准数据
- **.glb**: 3D 预览格式

## 关键算法

1. **色彩匹配**: KD-Tree 最近邻搜索
2. **图像量化**: K-Means 聚类
3. **网格生成**: RLE（游程编码）优化的高保真网格
4. **透视校正**: OpenCV 四点变换

## 要求

1. 每次更新了代码必须重启项目，不然不生效
2. layout_new.py 才是现在使用的布局文件！！
