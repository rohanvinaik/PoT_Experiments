#!/usr/bin/env python3
"""
Quick restart script for Zephyr-7B download
Run this when you're on new WiFi
"""
import os
import sys

print("🚀 Restarting Zephyr-7B Download")
print("=" * 50)

# Set environment for optimal download
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '7200'

from huggingface_hub import snapshot_download

print("📊 Current status:")
print("  - 11GB already cached (78% complete)")
print("  - ~3GB remaining to download")
print("  - Will resume from cached position")
print("")

try:
    print("📡 Starting download with default parallelism...")
    local_dir = snapshot_download(
        "HuggingFaceH4/zephyr-7b-beta",
        resume_download=True,
        # Using default workers for better speed
    )
    
    print(f"\n✅ Download complete! Model at: {local_dir}")
    
    # Auto-run the test
    print("\n🔬 Running Mistral vs Zephyr comparison test...")
    exec(open("run_mistral_zephyr.py").read())
    
except KeyboardInterrupt:
    print("\n⚠️ Download paused by user")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)