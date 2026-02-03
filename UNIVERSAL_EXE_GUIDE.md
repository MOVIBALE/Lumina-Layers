# Lumina Studio - 通用 EXE 打包方案

## 问题分析

**当前问题：**
- 您使用的是 CUDA 版本的 PyTorch (`torch==2.5.1+cu121`)
- CUDA 版本的 PyTorch 在导入时就会尝试加载 NVIDIA CUDA 运行时库
- 在没有 NVIDIA 显卡或驱动的电脑上，这会导致程序崩溃
- **无论代码如何处理，只要打包了 CUDA 版本的 PyTorch，EXE 就无法在没有 NVIDIA 的电脑上运行**

## 解决方案

### 方案 1：使用 CPU 版本的 PyTorch 打包（推荐，最兼容）

**步骤：**

```bash
# 1. 卸载 CUDA 版本的 PyTorch
pip uninstall torch torchvision -y

# 2. 安装 CPU 版本的 PyTorch
pip install torch torchvision

# 3. 使用 CPU 配置打包
python -m PyInstaller LuminaStudio_CPU.spec --clean --noconfirm
```

**优点：**
- ✅ 可以在任何 Windows 电脑上运行
- ✅ 文件体积更小（约 500MB-800MB）
- ✅ 不需要 NVIDIA 显卡

**缺点：**
- ❌ 不支持 GPU 加速（处理速度较慢）

### 方案 2：创建两个版本的 EXE

**版本 A：CUDA 版本**（给有 NVIDIA 显卡的用户）
```bash
# 保持当前的 CUDA 版本 PyTorch
python -m PyInstaller LuminaStudio.spec --clean --noconfirm
# 输出：LuminaStudio.exe
```

**版本 B：CPU 版本**（给所有用户）
```bash
# 切换到 CPU 版本 PyTorch
pip uninstall torch torchvision -y
pip install torch torchvision
python -m PyInstaller LuminaStudio_CPU.spec --clean --noconfirm
# 输出：LuminaStudio_CPU.exe
```

**分发建议：**
- 让用户根据自己的硬件选择合适的版本
- 或者提供在线检测工具，自动下载合适的版本

### 方案 3：使用启动脚本（最灵活，但不是 EXE）

对于没有 NVIDIA 显卡的用户，推荐直接使用 `start.bat`：

```batch
start.bat
```

这个脚本会自动：
1. 检测系统是否有 NVIDIA 显卡
2. 自动安装合适的 PyTorch 版本（CPU 或 CUDA）
3. 启动应用

**优点：**
- ✅ 自动适配任何硬件
- ✅ 始终保持最新代码
- ✅ 可以自动更新依赖

**缺点：**
- ❌ 需要安装 Python
- ❌ 不是单个 EXE 文件

## 推荐做法

### 对于开发者（您）：

**日常使用**：使用 CUDA 版本（速度快）
```bash
# 安装 CUDA 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python main.py
```

**分享给朋友**：提供两个选项
1. **有 NVIDIA 显卡的朋友**：分享 CUDA 版本的 EXE
2. **没有 NVIDIA 显卡的朋友**：分享 CPU 版本的 EXE，或者让他们使用 `start.bat`

### 对于最终用户：

**选项 1：使用 EXE（简单）**
- 下载适合自己电脑的版本（CUDA 版或 CPU 版）
- 双击运行

**选项 2：使用启动脚本（推荐）**
- 安装 Python 3.11+
- 运行 `start.bat`
- 自动适配硬件

## 为什么不能像 start.bat 一样自动判断？

**关键区别：**

| 方式 | PyTorch 安装 | 运行时机 | 可行性 |
|------|-------------|---------|--------|
| `start.bat` | 运行时动态安装 | 可以检测 GPU 后安装对应版本 | ✅ 可行 |
| `EXE 打包` | 打包时固定 | 已经包含了特定版本的 PyTorch | ❌ 不可行 |

**原因：**
- `start.bat` 在运行时才安装 PyTorch，可以根据检测结果选择安装 CPU 或 CUDA 版本
- EXE 打包时已经将 PyTorch 打包进去了，无法运行时更换
- CUDA 版本的 PyTorch 包含 CUDA 运行时库，这些库在没有 NVIDIA 驱动的电脑上会导致崩溃

## 总结

**要实现"一个 EXE 在所有电脑上运行"，唯一的方法是使用 CPU 版本的 PyTorch 打包。**

如果您想要 GPU 加速，必须提供两个版本：
1. `LuminaStudio.exe` - CUDA 版本（需要 NVIDIA 显卡）
2. `LuminaStudio_CPU.exe` - CPU 版本（所有电脑可用）

或者，**推荐使用 `start.bat` 启动脚本**，它可以自动检测并安装合适的 PyTorch 版本。

---

## 快速操作指南

### 创建 CPU 版本的 EXE（通用版本）：

```batch
cd C:\叠色\Lumina-Layers-beta

# 安装 CPU 版本的 PyTorch
pip uninstall torch torchvision -y
pip install torch torchvision

# 打包
python -m PyInstaller LuminaStudio_CPU.spec --clean --noconfirm

# 输出文件：dist/LuminaStudio_CPU.exe
```

这个 EXE 可以在任何 Windows 电脑上运行！
