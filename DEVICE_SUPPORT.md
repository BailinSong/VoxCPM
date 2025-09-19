# 设备后端支持

VoxCPM 项目现在支持多种 GPU 后端，可以根据您的硬件自动选择最佳的加速设备。

## 支持的后端

### 1. CUDA (NVIDIA GPU)
- **硬件要求**: NVIDIA GPU（支持 CUDA 计算能力 3.5+）
- **操作系统**: Windows, Linux, macOS
- **优势**: 最佳性能和兼容性
- **文档**: 详见 [CUDA 官方文档](https://docs.nvidia.com/cuda/)

### 2. DirectML (Windows 通用 GPU)
- **硬件要求**: 任何支持 DirectX 12 的 GPU
- **操作系统**: Windows 10/11
- **优势**: 支持所有 GPU 厂商，通用性最佳
- **文档**: 详见 [DIRECTML_SUPPORT.md](DIRECTML_SUPPORT.md)

### 3. HIP (AMD GPU)
- **硬件要求**: AMD RDNA 2/3 架构 GPU（RX 6000/7000 系列）
- **操作系统**: Windows, Linux
- **优势**: AMD GPU 加速，开源生态
- **文档**: 详见 [HIP_SUPPORT.md](HIP_SUPPORT.md)

### 4. MPS (Apple Silicon)
- **硬件要求**: Apple Silicon 芯片（M1、M2、M3 等）
- **操作系统**: macOS 12.3+
- **优势**: Apple 原生加速，低功耗
- **文档**: 详见 [MPS_SUPPORT.md](MPS_SUPPORT.md)

### 5. CPU
- **硬件要求**: 任何 x86_64 或 ARM64 CPU
- **操作系统**: 所有平台
- **优势**: 通用兼容性，无需额外配置

## 设备优先级

系统会自动按以下优先级选择计算设备：

```
CUDA > DirectML > HIP > MPS > CPU
```

## 自动优化特性

### 数据类型兼容性
- **CUDA**: 支持所有数据类型（float32, float16, bfloat16）
- **DirectML**: 自动降级 bfloat16 为 float32（兼容性考虑）
- **HIP**: 自动降级 bfloat16 为 float32（兼容性考虑）
- **MPS**: 自动降级 bfloat16 为 float32（硬件限制）
- **CPU**: 支持所有数据类型

### 编译优化
- **CUDA**: 启用 torch.compile 优化（需要 triton）
- **其他设备**: 使用原始函数（兼容性考虑）

### ASR 模型适配
- **CUDA**: 使用 GPU 加速
- **DirectML/HIP/MPS**: 自动回退到 CPU（第三方库兼容性）

## 使用方法

无需额外配置，系统会自动检测并使用最佳设备：

```python
import voxcpm

# 自动选择最佳设备（CUDA > DirectML > HIP > MPS > CPU）
model = voxcpm.VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")

# 生成语音
wav = model.generate(text="Hello, this is a test.")
```

## 性能对比

| 设备类型 | 相对性能 | 功耗 | 兼容性 | 推荐场景 |
|---------|---------|------|--------|----------|
| CUDA    | 100%    | 高   | 最佳   | 高性能计算 |
| DirectML | 85-95%  | 中等 | 最佳   | Windows 通用 |
| HIP     | 80-90%  | 中等 | 良好   | AMD GPU 用户 |
| MPS     | 60-80%  | 低   | 良好   | Apple 用户 |
| CPU     | 10-20%  | 低   | 最佳   | 通用兼容 |

*性能数据仅供参考，实际性能取决于具体硬件配置和模型大小*

## 故障排除

### 常见问题

1. **设备检测失败**
   ```python
   import torch
   print(f"CUDA: {torch.cuda.is_available()}")
   print(f"DirectML: {hasattr(torch.backends, 'directml') and torch.backends.directml.is_available()}")
   print(f"HIP: {hasattr(torch.version, 'hip') and torch.version.hip is not None}")
   print(f"MPS: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
   ```

2. **性能不佳**
   - 确认使用了正确的设备
   - 检查驱动版本
   - 验证内存配置

3. **兼容性问题**
   - 系统会自动回退到 CPU
   - 查看控制台警告信息
   - 参考具体设备的文档

### 环境变量

```bash
# CUDA 配置
export CUDA_VISIBLE_DEVICES=0

# HIP 配置
export HIP_VISIBLE_DEVICES=0
export ROCM_PATH=/opt/rocm

# MPS 配置（自动检测）
# 无需额外配置
```

## 开发指南

### 添加新设备支持

1. 在 `_is_hip_available()` 类似的函数中添加设备检测
2. 在设备检测逻辑中添加新设备的优先级
3. 在数据类型处理中添加兼容性检查
4. 在优化函数中添加设备特定逻辑
5. 更新文档和测试

### 测试设备支持

```python
# 测试特定设备
import torch
device = "cuda"  # 或 "hip", "mps", "cpu"
x = torch.tensor([1.0], device=device)
print(f"Device {device} works: {x.device}")
```

## 相关文档

- [CUDA 支持](https://pytorch.org/docs/stable/notes/cuda.html)
- [DirectML 支持文档](DIRECTML_SUPPORT.md)
- [HIP 支持文档](HIP_SUPPORT.md)
- [MPS 支持文档](MPS_SUPPORT.md)
- [PyTorch 设备选择](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
