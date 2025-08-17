#!/usr/bin/env python3
"""
GPU Detection Script for PoT Framework
Detects available GPU acceleration options across different platforms
"""

import sys
import platform

def detect_gpu():
    """Detect available GPU acceleration options."""
    
    gpu_info = {
        'cuda': False,
        'mps': False,
        'rocm': False,
        'device': 'cpu',
        'description': 'CPU only'
    }
    
    try:
        import torch
        
        # Check for CUDA (NVIDIA)
        if torch.cuda.is_available():
            gpu_info['cuda'] = True
            gpu_info['device'] = 'cuda'
            gpu_info['description'] = f'CUDA GPU: {torch.cuda.get_device_name(0)}'
            return gpu_info
        
        # Check for MPS (Apple Silicon/Metal)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info['mps'] = True
            gpu_info['device'] = 'mps'
            gpu_info['description'] = 'Apple Metal Performance Shaders'
            return gpu_info
            
        # Check for ROCm (AMD on Linux)
        if hasattr(torch, 'hip') and torch.hip.is_available():
            gpu_info['rocm'] = True
            gpu_info['device'] = 'cuda'  # ROCm uses CUDA interface
            gpu_info['description'] = 'AMD ROCm GPU'
            return gpu_info
            
    except ImportError:
        gpu_info['description'] = 'PyTorch not installed'
    
    # Platform-specific checks
    system = platform.system()
    machine = platform.machine()
    
    if system == 'Darwin':  # macOS
        # Check if it's a Mac with potential GPU
        import subprocess
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                 capture_output=True, text=True, timeout=5)
            output = result.stdout.lower()
            
            if 'amd' in output and 'firepro' in output:
                gpu_info['description'] = 'AMD FirePro GPU detected (OpenCL capable, no PyTorch acceleration)'
            elif 'amd' in output:
                gpu_info['description'] = 'AMD GPU detected (Metal capable on macOS 10.14+)'
            elif 'nvidia' in output:
                gpu_info['description'] = 'NVIDIA GPU detected (legacy, no CUDA on modern macOS)'
            elif 'apple' in output and ('m1' in output or 'm2' in output or 'm3' in output):
                gpu_info['description'] = 'Apple Silicon GPU (MPS not available in PyTorch)'
        except:
            pass
    
    return gpu_info

def main():
    """Main function for standalone execution."""
    info = detect_gpu()
    
    print(f"GPU Acceleration Status:")
    print(f"  Device: {info['device']}")
    print(f"  Description: {info['description']}")
    
    if info['cuda']:
        print("  ✓ CUDA available (NVIDIA GPU acceleration)")
    elif info['mps']:
        print("  ✓ MPS available (Apple Metal acceleration)")
    elif info['rocm']:
        print("  ✓ ROCm available (AMD GPU acceleration)")
    else:
        print("  ✗ No PyTorch GPU acceleration available")
        
        if '2013' in info['description'] or 'FirePro' in info['description']:
            print("\n  Note: Your 2013 Mac Pro has AMD FirePro GPUs which don't support CUDA.")
            print("  Options:")
            print("  1. Use CPU-only mode (fully supported)")
            print("  2. Use OpenCL for specific operations (requires custom code)")
            print("  3. Consider upgrading to NVIDIA GPU for CUDA support")
    
    return 0 if info['device'] != 'cpu' else 1

if __name__ == '__main__':
    sys.exit(main())