# GPU Compatibility Guide for PoT Framework

## Quick Check

Run this command to check your GPU status:
```bash
python scripts/detect_gpu.py
```

## GPU Support Matrix

| GPU Type | PyTorch Support | PoT Framework | Notes |
|----------|----------------|---------------|-------|
| NVIDIA (CUDA) | ✅ Full | ✅ Full | Best performance |
| Apple Silicon (M1/M2/M3) | ✅ MPS | ✅ Full | Good performance on macOS |
| AMD (ROCm) | ⚠️ Limited | ⚠️ Limited | Linux only, specific models |
| AMD (FirePro/Radeon) | ❌ None | ❌ CPU only | Use CPU mode |
| Intel (integrated) | ❌ None | ❌ CPU only | Use CPU mode |

## Platform-Specific Information

### macOS

#### 2013 Mac Pro (AMD FirePro)
Your Mac Pro has AMD FirePro GPUs (D300/D500/D700) which:
- ❌ **Cannot run CUDA** (NVIDIA-only technology)
- ❌ **Cannot use PyTorch GPU acceleration**
- ✅ **Can run the PoT framework in CPU mode**
- ✅ **Support OpenCL** (not used by PyTorch)

**Recommendation**: Use CPU-only mode:
```bash
pip install -r requirements-cpu.txt
```

#### Apple Silicon Macs (M1/M2/M3)
- ✅ Use MPS (Metal Performance Shaders) backend
- Automatic detection in PyTorch 1.12+
- Good performance for most operations

#### Intel Macs (2015-2019)
- Most have AMD GPUs (no CUDA support)
- Some have NVIDIA GPUs (no modern CUDA support on macOS)
- Use CPU-only mode

### Linux

#### NVIDIA GPUs
- ✅ Full CUDA support
- Install CUDA toolkit and cuDNN
- Best performance option

#### AMD GPUs
- ⚠️ ROCm support (specific models only)
- Check [ROCm compatibility](https://docs.amd.com/bundle/ROCm-Installation-Guide)
- Alternative: Use CPU mode

### Windows

#### NVIDIA GPUs
- ✅ Full CUDA support
- Install CUDA toolkit and cuDNN
- Follow [PyTorch installation guide](https://pytorch.org/get-started/locally/)

#### AMD GPUs
- ❌ No PyTorch GPU support on Windows
- Use CPU-only mode

## Installation by GPU Type

### For NVIDIA GPUs (CUDA)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### For Apple Silicon (MPS)
```bash
pip install torch torchvision
pip install -r requirements.txt
```

### For CPU-Only (including AMD GPUs)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
```

## Performance Expectations

| Configuration | Verification Speed | Training Speed | Recommended For |
|--------------|-------------------|----------------|-----------------|
| NVIDIA GPU | Fast (10-100x) | Fast | Production, Research |
| Apple Silicon | Good (5-20x) | Good | Development, Testing |
| CPU (modern) | Baseline | Slow | Testing, Small models |
| CPU (older) | Slow | Very slow | Basic testing only |

## Troubleshooting

### "CUDA not available" on system with NVIDIA GPU
1. Check CUDA installation: `nvidia-smi`
2. Verify PyTorch CUDA version matches system CUDA
3. Reinstall PyTorch with correct CUDA version

### Poor performance on Mac
1. Ensure using latest PyTorch (1.12+ for MPS)
2. For 2013 Mac Pro: CPU-only is the only option
3. Consider cloud GPU instances for heavy workloads

### Memory errors
1. Reduce batch size in configs
2. Use gradient checkpointing
3. Switch to CPU if GPU memory insufficient

## Cloud GPU Options

If local GPU is insufficient:
- **Google Colab**: Free tier with GPU
- **AWS EC2**: P3/P4 instances
- **Google Cloud**: GPU-enabled VMs
- **Paperspace**: Gradient notebooks
- **Lambda Labs**: GPU cloud

## Checking Your Configuration

Run this diagnostic:
```bash
python -c "
import torch
import platform

print(f'System: {platform.system()} {platform.machine()}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if hasattr(torch.backends, 'mps'):
    print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')
"
```

## Summary

- **2013 Mac Pro**: Use CPU-only mode, CUDA is not possible with AMD GPUs
- **Modern Macs**: Use MPS if available, otherwise CPU
- **Linux/Windows with NVIDIA**: Use CUDA for best performance
- **All AMD GPUs (except ROCm)**: Use CPU-only mode

The PoT framework works perfectly in CPU-only mode, just with reduced performance for large-scale operations.