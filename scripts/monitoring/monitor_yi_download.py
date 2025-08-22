#!/usr/bin/env python3
"""
Download monitor for Yi-34B models
Tracks download progress and estimates completion time
"""

import os
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

def get_dir_size(path):
    """Get total size of directory in GB."""
    if not os.path.exists(path):
        return 0
    
    result = subprocess.run(f"du -s {path}", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        size_kb = int(result.stdout.split()[0])
        return size_kb / (1024 * 1024)  # Convert to GB
    return 0

def format_size(gb):
    """Format size nicely."""
    if gb < 1:
        return f"{gb*1024:.1f} MB"
    return f"{gb:.2f} GB"

def format_time(seconds):
    """Format time duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def check_download_processes():
    """Check if download processes are still running."""
    result = subprocess.run("ps aux | grep -E 'huggingface-cli download.*[Yy]i-34[Bb]' | grep -v grep", 
                          shell=True, capture_output=True, text=True)
    return len(result.stdout.strip().split('\n')) > 0 if result.stdout.strip() else False

def monitor_downloads():
    """Monitor Yi-34B model downloads."""
    base_dir = "/Users/rohanvinaik/LLM_Models"
    models = {
        "Yi-34B Base": f"{base_dir}/yi-34b",
        "Yi-34B Chat": f"{base_dir}/yi-34b-chat"
    }
    
    # Expected sizes (approximate)
    expected_size_gb = 68  # Each model is ~68GB
    
    print("=" * 70)
    print("YI-34B MODEL DOWNLOAD MONITOR")
    print("=" * 70)
    print(f"Expected size per model: ~{expected_size_gb} GB")
    print(f"Total expected: ~{expected_size_gb * 2} GB")
    print("=" * 70)
    print("")
    
    start_time = time.time()
    last_sizes = {}
    download_rates = {}
    
    while True:
        os.system('clear')  # Clear screen for clean display
        
        current_time = datetime.now().strftime("%H:%M:%S")
        elapsed = time.time() - start_time
        
        print("=" * 70)
        print(f"YI-34B DOWNLOAD MONITOR | {current_time} | Elapsed: {format_time(elapsed)}")
        print("=" * 70)
        
        # Check system resources
        mem_result = subprocess.run("top -l 1 -n 0 | grep PhysMem", 
                                  shell=True, capture_output=True, text=True)
        if mem_result.stdout:
            print(f"System Memory: {mem_result.stdout.strip()}")
        
        # Check network activity
        net_result = subprocess.run("netstat -ib | grep -E '^en0' | tail -1", 
                                   shell=True, capture_output=True, text=True)
        if net_result.stdout:
            fields = net_result.stdout.split()
            if len(fields) >= 7:
                print(f"Network (en0): RX {fields[6]} | TX {fields[9]}")
        
        print("-" * 70)
        
        total_downloaded = 0
        all_complete = True
        
        for name, path in models.items():
            current_size = get_dir_size(path)
            total_downloaded += current_size
            
            # Calculate download rate
            if name in last_sizes:
                rate = (current_size - last_sizes[name]) / 5  # GB per 5 seconds
                if name not in download_rates:
                    download_rates[name] = []
                download_rates[name].append(rate)
                # Keep only last 12 samples (1 minute)
                download_rates[name] = download_rates[name][-12:]
                avg_rate = sum(download_rates[name]) / len(download_rates[name]) if download_rates[name] else 0
            else:
                avg_rate = 0
            
            last_sizes[name] = current_size
            
            # Progress calculation
            progress = (current_size / expected_size_gb) * 100 if expected_size_gb > 0 else 0
            if progress > 100:
                progress = 100
            
            # Status determination
            config_exists = os.path.exists(f"{path}/config.json")
            if progress >= 95 and config_exists:
                status = "✅ COMPLETE"
                status_color = "\033[92m"  # Green
            elif current_size > 0.1:
                status = "⏳ DOWNLOADING"
                status_color = "\033[93m"  # Yellow
                all_complete = False
            else:
                status = "⏸️  WAITING"
                status_color = "\033[90m"  # Gray
                all_complete = False
            
            # ETA calculation
            if avg_rate > 0 and progress < 100:
                remaining_gb = expected_size_gb - current_size
                eta_seconds = (remaining_gb / avg_rate) * 5  # Convert back to seconds
                eta_str = f"ETA: {format_time(eta_seconds)}"
            else:
                eta_str = "ETA: --"
            
            # Progress bar
            bar_length = 30
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            # Display
            print(f"\n{status_color}{name}:{' '*(15-len(name))} {status}\033[0m")
            print(f"  Size: {format_size(current_size)} / {expected_size_gb} GB")
            print(f"  [{bar}] {progress:.1f}%")
            
            if avg_rate > 0:
                print(f"  Speed: {avg_rate*1024*5:.1f} MB/s | {eta_str}")
            
            # Check for specific files
            if os.path.exists(path):
                files = os.listdir(path)
                safetensors = [f for f in files if f.endswith('.safetensors')]
                print(f"  Files: {len(files)} total, {len(safetensors)} safetensors")
        
        print("-" * 70)
        print(f"Total Downloaded: {format_size(total_downloaded)} / ~{expected_size_gb * 2} GB")
        print(f"Overall Progress: {(total_downloaded / (expected_size_gb * 2)) * 100:.1f}%")
        
        # Check if downloads are still running
        if not check_download_processes():
            print("\n⚠️  No active download processes detected")
            if not all_complete:
                print("Downloads may have stalled or failed. Check logs.")
        
        if all_complete:
            print("\n" + "=" * 70)
            print("🎉 ALL DOWNLOADS COMPLETE!")
            print("=" * 70)
            print(f"Total time: {format_time(elapsed)}")
            print(f"Total size: {format_size(total_downloaded)}")
            print("\nModels ready for testing at:")
            for name, path in models.items():
                print(f"  - {path}")
            break
        
        print("\n[Press Ctrl+C to stop monitoring]")
        
        try:
            time.sleep(5)  # Update every 5 seconds
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
            break

if __name__ == "__main__":
    monitor_downloads()