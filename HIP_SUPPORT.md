# AMD HIP 后端支持

本项目现已支持 AMD 显卡的 HIP（Heterogeneous-Computing Interface for Portability）后端，可以在配备 AMD GPU 的 Windows 系统上加速模型推理。

## 系统要求

### 硬件要求
- **GPU**: AMD RDNA 2/3 架构显卡（如 RX 6000/7000 系列）
- **操作系统**: Windows 10/11 或 Linux

### 软件要求
- **ROCm**: 5.4.0 或更高版本
- **Python**: 3.8 或更高版本
- **PyTorch**: 2.1.0 或更高版本（支持 HIP 的版本）

## 安装 ROCm 和 PyTorch

### Windows 安装

1. **安装 ROCm**
   ```bash
   # 从 AMD 官网下载并安装 ROCm
   # https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html
   ```

2. **安装支持 HIP 的 PyTorch**
   ```bash
   # 安装 PyTorch with ROCm support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
   ```

### Linux 安装

1. **安装 ROCm**
   ```bash
   # Ubuntu/Debian
   wget https://repo.radeon.com/amdgpu-install/5.6/ubuntu/jammy/amdgpu-install_5.6.50600-1_all.deb
   sudo dpkg -i amdgpu-install_5.6.50600-1_all.deb
   sudo amdgpu-install --usecase=rocm
   ```

2. **安装支持 HIP 的 PyTorch**
   ```bash
   # 安装 PyTorch with ROCm support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
   ```

## 设备优先级

系统会按以下优先级选择计算设备：

1. **CUDA** - 如果可用（NVIDIA GPU）
2. **HIP** - 如果可用（AMD GPU）
3. **MPS** - 如果可用（Apple Silicon）
4. **CPU** - 作为后备选项

## 使用方法

无需额外配置，系统会自动检测并使用最佳可用设备：

```python
import voxcpm

# 自动选择最佳设备（CUDA > HIP > MPS > CPU）
model = voxcpm.VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")

# 生成语音
wav = model.generate(text="Hello, this is a test.")
```

## 自动优化

- **数据类型兼容性**: 在 HIP 设备上，不支持的数据类型（如 bfloat16）会自动降级为 float32
- **编译优化**: torch.compile 优化仅在 CUDA 设备上启用，HIP 设备使用原始函数
- **ASR 模型**: SenseVoice ASR 模型在 HIP 设备上会回退到 CPU 运行

## 性能说明

- 在 AMD GPU 上，HIP 后端可以显著提升推理速度
- 性能提升取决于具体的 GPU 型号和模型大小
- 对于大型模型，HIP 相比 CPU 通常有 3-8x 的性能提升
- RDNA 3 架构的 GPU 通常比 RDNA 2 有更好的性能

## 环境变量

可以通过以下环境变量配置 ROCm：

```bash
# 设置 ROCm 路径
export ROCM_PATH=/opt/rocm

# 设置 HIP 设备可见性
export HIP_VISIBLE_DEVICES=0

# 设置 HIP 内存池
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
```

## 故障排除

### 常见问题

1. **HIP 设备不可用**
   - 确认已安装 ROCm
   - 检查 `hipinfo` 命令是否可用
   - 验证 GPU 驱动是否正确安装

2. **PyTorch 不支持 HIP**
   - 确认安装的是支持 HIP 的 PyTorch 版本
   - 检查 PyTorch 版本：`torch.version.hip`

3. **性能不佳**
   - 确认使用的是 RDNA 2/3 架构的 GPU
   - 检查 HIP_VISIBLE_DEVICES 设置
   - 尝试调整内存池配置

### 验证安装

```python
import torch

# 检查 HIP 支持
print(f"PyTorch version: {torch.__version__}")
print(f"HIP version: {torch.version.hip}")
print(f"HIP available: {torch.cuda.is_available() and torch.version.hip is not None}")

# 测试 HIP 张量
try:
    x = torch.tensor([1.0], device='hip:0')
    print("HIP tensor creation successful")
except Exception as e:
    print(f"HIP tensor creation failed: {e}")
```

## 注意事项

- HIP 后端仍在积极开发中，某些操作可能不如 CUDA 稳定
- 建议使用较新的 AMD GPU（RDNA 2/3 架构）以获得最佳性能
- 如果遇到兼容性问题，系统会自动回退到 CPU
- 某些第三方库可能不完全支持 HIP，会回退到 CPU 运行

## 支持的 GPU 列表

### 推荐使用的 GPU
- **RDNA 3**: RX 7900 系列、RX 7800 系列、RX 7700 系列
- **RDNA 2**: RX 6900 系列、RX 6800 系列、RX 6700 系列

### 可能支持的 GPU
- **RDNA 1**: RX 5700 系列（支持有限）
- **Vega**: RX Vega 系列（支持有限）

详细的支持列表请参考 [ROCm 官方文档](https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html#supported-gpus)。
