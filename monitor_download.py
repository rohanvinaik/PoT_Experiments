#!/usr/bin/env python3
"""
Monitor HuggingFace model download progress
"""
import os
import time
import sys
from pathlib import Path

def get_folder_size(folder):
    """Get total size of folder in bytes"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    except:
        pass
    return total

def format_bytes(bytes):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.1f}TB"

def format_time(seconds):
    """Format seconds to human readable"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def monitor_download():
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    zephyr_dir = cache_dir / "models--HuggingFaceH4--zephyr-7b-beta"
    
    # Expected size (approximate)
    expected_size = 14 * 1024 * 1024 * 1024  # 14GB
    
    print("MONITORING ZEPHYR MODEL DOWNLOAD")
    print("=" * 60)
    print(f"Cache location: {zephyr_dir}")
    print(f"Expected size: ~14GB")
    print("=" * 60)
    print()
    
    start_time = time.time()
    last_size = 0
    last_time = start_time
    stall_count = 0
    
    while True:
        try:
            # Get current size
            current_size = get_folder_size(zephyr_dir)
            current_time = time.time()
            
            # Calculate progress
            progress = (current_size / expected_size) * 100
            progress = min(progress, 100)  # Cap at 100%
            
            # Calculate speed
            time_delta = current_time - last_time
            size_delta = current_size - last_size
            
            if time_delta > 0:
                speed = size_delta / time_delta  # bytes per second
            else:
                speed = 0
            
            # Calculate ETA
            if speed > 0:
                remaining = expected_size - current_size
                eta_seconds = remaining / speed
                eta_str = format_time(eta_seconds)
            else:
                eta_str = "calculating..."
            
            # Check if stalled
            if size_delta == 0 and time_delta >= 5:
                stall_count += 1
                status = "⚠️  STALLED"
            else:
                stall_count = 0
                status = "⬇️  DOWNLOADING"
            
            # Create progress bar
            bar_width = 40
            filled = int(bar_width * progress / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            # Print progress
            sys.stdout.write("\r")
            sys.stdout.write(f"{status} [{bar}] {progress:.1f}% ")
            sys.stdout.write(f"| {format_bytes(current_size)}/{format_bytes(expected_size)} ")
            sys.stdout.write(f"| {format_bytes(speed)}/s ")
            sys.stdout.write(f"| ETA: {eta_str} ")
            
            # Check for completion
            if current_size >= expected_size * 0.95:  # 95% is close enough
                print("\n\n✅ Download appears complete!")
                break
            
            # Check for long stall
            if stall_count > 12:  # 1 minute of no progress
                print("\n\n⚠️  Download stalled for >1 minute")
                print("The download may have failed or been rate limited")
            
            # Update for next iteration
            last_size = current_size
            last_time = current_time
            
            # Wait before next check
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_download()