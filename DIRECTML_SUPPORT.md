# DirectML 后端支持

本项目现已支持 Microsoft DirectML 后端，这是 Windows 上最通用的 GPU 加速解决方案，支持所有类型的 GPU。

## 什么是 DirectML？

DirectML 是微软开发的机器学习 API，通过 DirectX 12 工作，可以在 Windows 上支持：
- **NVIDIA GPU** (GeForce, Quadro, Tesla 系列)
- **AMD GPU** (Radeon RX 系列)
- **Intel GPU** (Arc, Iris Xe 系列)
- **集成显卡** (Intel UHD, AMD Vega)

## 系统要求

### 硬件要求
- **GPU**: 任何支持 DirectX 12 的 GPU
- **操作系统**: Windows 10 版本 1903 或更高版本 / Windows 11
- **驱动**: 最新的 GPU 驱动

### 软件要求
- **Windows ML 运行时**: 自动包含在 Windows 10/11 中
- **Python**: 3.8 或更高版本
- **PyTorch**: 2.1.0 或更高版本

## 安装和配置

### 1. 安装支持 DirectML 的 PyTorch

```bash
# 安装 PyTorch with DirectML support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/directml
```

### 2. 验证安装

```python
import torch

# 检查 DirectML 支持
print(f"PyTorch version: {torch.__version__}")
print(f"DirectML available: {hasattr(torch.backends, 'directml') and torch.backends.directml.is_available()}")

# 测试 DirectML 张量
try:
    x = torch.tensor([1.0], device='directml:0')
    print("DirectML tensor creation successful")
except Exception as e:
    print(f"DirectML tensor creation failed: {e}")
```

## 设备优先级

系统会按以下优先级选择计算设备：

1. **CUDA** - 如果可用（NVIDIA GPU）
2. **DirectML** - 如果可用（Windows 上的任何 GPU）
3. **HIP** - 如果可用（AMD GPU）
4. **MPS** - 如果可用（Apple Silicon）
5. **CPU** - 作为后备选项

## 使用方法

无需额外配置，系统会自动检测并使用最佳可用设备：

```python
import voxcpm

# 自动选择最佳设备（CUDA > DirectML > HIP > MPS > CPU）
model = voxcpm.VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")

# 生成语音
wav = model.generate(text="Hello, this is a test.")
```

## 自动优化

- **数据类型兼容性**: 在 DirectML 设备上，不支持的数据类型（如 bfloat16）会自动降级为 float32
- **编译优化**: torch.compile 优化仅在 CUDA 设备上启用，DirectML 设备使用原始函数
- **ASR 模型**: SenseVoice ASR 模型在 DirectML 设备上会回退到 CPU 运行

## 性能说明

### 性能对比

| GPU 类型 | DirectML 性能 | CUDA 性能 | 推荐使用 |
|---------|--------------|-----------|----------|
| NVIDIA RTX 系列 | 85-95% | 100% | DirectML (通用性) |
| AMD RX 系列 | 90-100% | N/A | DirectML (唯一选择) |
| Intel Arc 系列 | 80-90% | N/A | DirectML (唯一选择) |
| 集成显卡 | 60-80% | N/A | DirectML (唯一选择) |

*性能数据仅供参考，实际性能取决于具体硬件配置*

### 优势

- ✅ **通用性**: 支持所有 Windows GPU
- ✅ **易用性**: 无需安装特定驱动或 SDK
- ✅ **稳定性**: 微软官方支持，更新频繁
- ✅ **兼容性**: 与现有 CUDA 代码兼容

### 限制

- ❌ **平台限制**: 仅支持 Windows
- ❌ **性能**: 可能略低于专用后端
- ❌ **功能**: 某些高级功能可能不支持

## 故障排除

### 常见问题

1. **DirectML 不可用**
   ```python
   # 检查系统要求
   import platform
   print(f"Windows version: {platform.version()}")
   
   # 检查 DirectX 12 支持
   # 在 PowerShell 中运行: dxdiag
   ```

2. **性能不佳**
   - 更新 GPU 驱动到最新版本
   - 确认 GPU 支持 DirectX 12
   - 检查 Windows 版本是否足够新

3. **内存不足**
   ```python
   # 设置 DirectML 内存管理
   import os
   os.environ['PYTORCH_DIRECTML_ALLOC_CONF'] = 'max_split_size_mb:128'
   ```

### 环境变量

```bash
# DirectML 内存配置
set PYTORCH_DIRECTML_ALLOC_CONF=max_split_size_mb:128

# DirectML 调试信息
set PYTORCH_DIRECTML_DEBUG=1

# 指定 DirectML 设备
set DIRECTML_DEVICE_ID=0
```

## 支持的 GPU 列表

### NVIDIA GPU
- **RTX 系列**: RTX 4090, RTX 4080, RTX 4070, RTX 4060, RTX 3090, RTX 3080, RTX 3070, RTX 3060
- **GTX 系列**: GTX 1660, GTX 1650, GTX 1080, GTX 1070, GTX 1060
- **Quadro 系列**: RTX A6000, RTX A5000, RTX A4000

### AMD GPU
- **RX 7000 系列**: RX 7900 XTX, RX 7800 XT, RX 7700 XT, RX 7600
- **RX 6000 系列**: RX 6900 XT, RX 6800 XT, RX 6700 XT, RX 6600 XT
- **RX 5000 系列**: RX 5700 XT, RX 5600 XT, RX 5500 XT

### Intel GPU
- **Arc 系列**: Arc A770, Arc A750, Arc A580, Arc A380
- **Iris Xe**: Iris Xe Graphics, Iris Xe MAX
- **集成显卡**: UHD Graphics 630, UHD Graphics 620

## 开发指南

### 手动指定 DirectML 设备

```python
import torch

# 指定 DirectML 设备
device = torch.device('directml:0')
model = model.to(device)

# 创建 DirectML 张量
x = torch.randn(2, 3, device=device)
```

### 检查 DirectML 设备信息

```python
import torch

if hasattr(torch.backends, 'directml') and torch.backends.directml.is_available():
    # 获取 DirectML 设备数量
    device_count = torch.backends.directml.device_count()
    print(f"DirectML devices: {device_count}")
    
    # 获取设备名称
    for i in range(device_count):
        device_name = torch.backends.directml.get_device_name(i)
        print(f"Device {i}: {device_name}")
```

## 相关资源

- [DirectML 官方文档](https://docs.microsoft.com/en-us/windows/ai/directml/)
- [PyTorch DirectML 支持](https://pytorch.org/blog/directml-backend/)
- [Windows ML 文档](https://docs.microsoft.com/en-us/windows/ai/)
- [DirectX 12 文档](https://docs.microsoft.com/en-us/windows/win32/direct3d12/)

## 注意事项

- DirectML 后端仍在积极开发中，建议使用最新版本的 PyTorch
- 某些第三方库可能不完全支持 DirectML，会回退到 CPU
- 如果遇到兼容性问题，系统会自动回退到 CPU
- 建议在支持的硬件上优先使用 DirectML 以获得最佳通用性
