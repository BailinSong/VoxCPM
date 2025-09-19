# macOS MPS 后端支持

本项目现已支持 macOS 的 MPS（Metal Performance Shaders）后端，可以在配备 Apple Silicon 芯片（M1、M2、M3 等）的 Mac 上加速模型推理。

## 系统要求

- **硬件**: Apple Silicon 芯片（M1、M2、M3 等）
- **操作系统**: macOS 12.3 或更高版本
- **Python**: 3.8 或更高版本
- **PyTorch**: 2.1.0 或更高版本

## 设备优先级

系统会按以下优先级选择计算设备：

1. **CUDA** - 如果可用（NVIDIA GPU）
2. **HIP** - 如果可用（AMD GPU）
3. **MPS** - 如果可用（Apple Silicon）
4. **CPU** - 作为后备选项

## 自动优化

- **数据类型兼容性**: 在 MPS 设备上，不支持的数据类型（如 bfloat16）会自动降级为 float32
- **编译优化**: torch.compile 优化仅在 CUDA 设备上启用，MPS 设备使用原始函数
- **ASR 模型**: SenseVoice ASR 模型在 MPS 设备上会回退到 CPU 运行

## 使用方法

无需额外配置，系统会自动检测并使用最佳可用设备：

```python
import voxcpm

# 自动选择最佳设备（CUDA > MPS > CPU）
model = voxcpm.VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")

# 生成语音
wav = model.generate(text="Hello, this is a test.")
```

## 性能说明

- 在 Apple Silicon Mac 上，MPS 后端可以显著提升推理速度
- 性能提升取决于具体的 Mac 型号和模型大小
- 对于大型模型，MPS 相比 CPU 通常有 2-5x 的性能提升

## 故障排除

如果遇到问题：

1. 确认您的 Mac 使用 Apple Silicon 芯片（不是 Intel）
2. 检查 macOS 版本是否 >= 12.3
3. 确认 PyTorch 版本 >= 2.1.0
4. 查看控制台输出的设备检测信息

## 注意事项

- MPS 后端仍在开发中，某些操作可能不如 CUDA 稳定
- 如果遇到兼容性问题，系统会自动回退到 CPU
- 建议在 Intel Mac 上使用 CPU 后端
