# Lumina Studio - Windows EXE 打包说明

## 重要提示

### 关于 CUDA 支持

当前打包的 EXE 使用的是 **CUDA 版本的 PyTorch** (`torch==2.5.1+cu121`)。

**这意味着：**
- ✅ 在**有 NVIDIA 显卡**的电脑上：CUDA 加速正常工作
- ⚠️ 在**没有 NVIDIA 显卡**的电脑上：可能会报错或无法启动

### 解决方案

#### 方案 1：创建两个版本的 EXE（推荐）

**版本 A：CUDA 版本**（给有 NVIDIA 显卡的用户）
- 使用当前配置打包
- 文件较大（约 1-2GB）
- 支持 GPU 加速

**版本 B：CPU 版本**（给所有用户）
- 需要安装 CPU 版本的 PyTorch：
  ```bash
  pip uninstall torch torchvision -y
  pip install torch torchvision
  ```
- 然后重新打包
- 文件较小（约 500MB-1GB）
- 仅支持 CPU 计算（速度较慢）

#### 方案 2：使用启动脚本（最兼容）

对于没有 NVIDIA 显卡的用户，推荐使用 `start.bat`：

```batch
start.bat
```

这个脚本会自动：
1. 检查 Python 是否安装
2. 安装依赖（自动选择合适的 PyTorch 版本）
3. 检测 GPU 可用性
4. 启动应用

#### 方案 3：动态检测（代码已实现）

代码已经实现了动态检测 GPU 的功能：
- 如果有 GPU：使用 CUDA 加速
- 如果没有 GPU：自动回退到 CPU 模式

**但问题在于：** CUDA 版本的 PyTorch 在导入时就会尝试加载 CUDA 库，导致在没有 NVIDIA 驱动的电脑上崩溃。

### 如何为不同用户分发

#### 给有 NVIDIA 显卡的用户：
```
LuminaStudio_CUDA.exe  (使用 build_exe.bat 打包)
```

#### 给没有 NVIDIA 显卡的用户：
```
1. 安装 Python 3.11+
2. 运行：pip install -r requirements.txt
3. 运行：python main.py
```

或者使用便携版 Python + 启动脚本。

### 推荐的最终方案

**创建安装程序**，在安装时检测 GPU 并安装相应版本：

1. 使用 Inno Setup 或 NSIS 创建安装包
2. 安装程序检测是否有 NVIDIA 显卡
3. 如果有：安装 CUDA 版本
4. 如果没有：安装 CPU 版本

这样可以确保每个用户都获得最适合的版本。

---

## 当前 EXE 的使用限制

⚠️ **当前打包的 EXE 要求目标电脑必须有：**
- Windows 10/11
- **NVIDIA 显卡** + 驱动程序
- 或者安装有 CUDA Runtime

**如果没有 NVIDIA 显卡，EXE 将无法运行。**

---

## 快速解决方案

如果您需要将程序分享给没有 NVIDIA 显卡的朋友：

### 方法 1：分享源代码
```
1. 将整个项目文件夹压缩
2. 让对方安装 Python 3.11+
3. 运行：pip install -r requirements.txt
4. 运行：python main.py
```

### 方法 2：使用 CPU 版本打包
```bash
# 在打包前，先安装 CPU 版本的 PyTorch
pip uninstall torch torchvision -y
pip install torch torchvision

# 然后打包
build_exe.bat
```

这样打包出来的 EXE 可以在任何 Windows 电脑上运行，但不支持 GPU 加速。

---

## 总结

| 版本 | 文件大小 | 需要 NVIDIA 显卡 | 速度 | 适用场景 |
|------|---------|----------------|------|---------|
| CUDA 版本 | 1-2GB | ✅ 需要 | 快（GPU 加速） | 有 NVIDIA 显卡的用户 |
| CPU 版本 | 500MB-1GB | ❌ 不需要 | 慢（纯 CPU） | 所有 Windows 用户 |

**建议：** 根据目标用户的硬件情况选择合适的版本。
