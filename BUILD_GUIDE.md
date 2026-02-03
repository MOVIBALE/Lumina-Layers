# Lumina Studio - Windows 打包指南

## 方案 1：使用 PyInstaller 打包成单个 EXE（推荐用于分发）

### 步骤：

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```

2. **运行打包脚本**
   ```bash
   build_exe.bat
   ```
   或者手动运行：
   ```bash
   pyinstaller LuminaStudio.spec --clean --noconfirm
   ```

3. **等待完成**
   - 这个过程可能需要 **10-30 分钟**
   - 生成的文件位于 `dist/LuminaStudio.exe`

4. **测试运行**
   - 双击 `dist/LuminaStudio.exe`
   - 确保 CUDA 加速正常工作

### 注意事项：

- **文件大小**：包含 PyTorch 的 EXE 文件会很大（约 1-2GB）
- **CUDA 支持**：打包后的 EXE 仍然需要目标电脑安装 NVIDIA 驱动
- **首次运行**：首次启动可能需要几分钟来解压文件

---

## 方案 2：使用启动脚本（推荐用于开发/自用）

### 使用方法：

直接双击 `start.bat` 文件即可运行。

### 功能：
- ✅ 自动检查 Python 安装
- ✅ 自动安装缺失的依赖
- ✅ 自动检测 GPU（CUDA）支持
- ✅ 一键启动应用

### 优点：
- 文件体积小（仅源代码）
- 启动速度快
- 易于更新（直接替换代码文件）

---

## 方案 3：创建便携版（高级）

### 步骤：

1. **下载便携版 Python**
   - 从 https://www.python.org/downloads/windows/ 下载 "Windows embeddable package"
   - 解压到项目目录的 `python` 文件夹

2. **安装依赖到便携版**
   ```bash
   python\python.exe -m pip install -r requirements.txt
   ```

3. **创建启动脚本**
   创建 `run_portable.bat`：
   ```batch
   @echo off
   python\python.exe main.py
   ```

4. **打包分发**
   将整个文件夹压缩成 ZIP，用户解压后即可运行

---

## CUDA 支持说明

### 打包后的 EXE 如何保持 CUDA 支持：

1. **开发环境**
   - 在安装了 CUDA 的电脑上打包
   - PyInstaller 会自动包含必要的 CUDA 运行时文件

2. **目标电脑要求**
   - 必须安装 NVIDIA 显卡驱动
   - 不需要安装 CUDA Toolkit
   - 不需要安装 Python

3. **验证 CUDA 工作**
   运行时会显示：
   ```
   [CUDA] Using NVIDIA GPU: NVIDIA GeForce RTX 3050 Laptop GPU
   [CUDA] Memory: 4.00 GB
   ```

---

## 常见问题

### Q: 打包后的 EXE 文件太大？
A: 这是正常的，因为 PyTorch 本身就很大。可以使用 UPX 压缩：
```bash
pyinstaller LuminaStudio.spec --upx-dir=C:\upx
```

### Q: 打包后 CUDA 不工作？
A: 确保：
1. 开发电脑安装了 PyTorch CUDA 版本
2. 目标电脑安装了 NVIDIA 驱动
3. 检查 `torch.cuda.is_available()` 返回 True

### Q: 杀毒软件报毒？
A: PyInstaller 打包的 EXE 有时会被误报。可以：
1. 将 EXE 添加到杀毒软件白名单
2. 使用代码签名证书签名（商业用途）
3. 向用户说明这是误报

### Q: 启动速度很慢？
A: 首次启动需要解压文件，后续会快一些。可以考虑使用 `--onedir` 模式代替 `--onefile`。

---

## 推荐的分发方式

对于不同场景：

| 场景 | 推荐方案 | 说明 |
|------|---------|------|
| **个人使用** | start.bat | 最简单，保持更新方便 |
| **分享给朋友** | PyInstaller EXE | 双击即用，无需安装 Python |
| **正式发布** | 便携版 | 最专业，可自定义图标 |
| **商业分发** | 安装程序 | 使用 Inno Setup 或 NSIS 创建安装包 |

---

## 构建安装包（可选）

使用 Inno Setup 创建专业安装程序：

1. 下载并安装 Inno Setup
2. 创建脚本文件 `setup.iss`
3. 编译生成 `LuminaStudio_Setup.exe`

示例脚本见 `installer/setup.iss`
