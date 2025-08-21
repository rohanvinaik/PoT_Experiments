#!/usr/bin/env python3
"""
Qwen2.5-72B Download Tracker
Monitors progress for the Q4_K_M quantized version
"""
import os
import sys
import time
from pathlib import Path

def format_size(bytes):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"

def get_download_progress():
    """Check download progress for Qwen model"""
    # Check cache directory
    cache_dir = Path("/Users/rohanvinaik/LLM_Models/Qwen2.5-72B-Q4/.cache/huggingface/download")
    
    if not cache_dir.exists():
        return None, 0
    
    # Find incomplete file
    incomplete_files = list(cache_dir.glob("*.incomplete"))
    if not incomplete_files:
        # Check final location
        final_file = Path("/Users/rohanvinaik/LLM_Models/Qwen2.5-72B-Q4/Qwen2.5-72B-Instruct-Q4_K_M.gguf")
        if final_file.exists():
            size = final_file.stat().st_size
            return "COMPLETE", size
        return None, 0
    
    # Get size of downloading file
    total_size = 0
    for file in incomplete_files:
        total_size += file.stat().st_size
    
    return "DOWNLOADING", total_size

def monitor_download():
    """Monitor download with progress updates"""
    print("🤖 Qwen2.5-72B-Instruct Q4_K_M Download Monitor")
    print("=" * 60)
    print("📊 Expected size: ~45-50 GB")
    print("📁 Location: /Users/rohanvinaik/LLM_Models/Qwen2.5-72B-Q4/")
    print("=" * 60)
    
    last_size = 0
    start_time = time.time()
    
    while True:
        status, current_size = get_download_progress()
        
        if status is None:
            print("⏳ Waiting for download to start...")
        elif status == "DOWNLOADING":
            # Calculate speed
            elapsed = time.time() - start_time
            if elapsed > 0 and current_size > last_size:
                speed = (current_size - last_size) / elapsed
                speed_str = format_size(speed) + "/s"
            else:
                speed_str = "calculating..."
            
            # Estimate progress (assuming 48GB total)
            estimated_total = 48 * 1024 * 1024 * 1024  # 48GB
            progress = (current_size / estimated_total) * 100
            
            # Progress bar
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            print(f"\r📥 [{bar}] {progress:.1f}% | {format_size(current_size)} | {speed_str}", end="", flush=True)
            
            last_size = current_size
            start_time = time.time()
            
        elif status == "COMPLETE":
            print(f"\n✅ Download complete! Size: {format_size(current_size)}")
            print(f"📍 Model ready at: /Users/rohanvinaik/LLM_Models/Qwen2.5-72B-Q4/")
            break
        
        time.sleep(2)  # Update every 2 seconds

def main():
    try:
        monitor_download()
    except KeyboardInterrupt:
        print("\n\n⏸️ Monitoring stopped. Download continues in background.")
        print("💡 Run this script again to check progress.")

if __name__ == "__main__":
    main()