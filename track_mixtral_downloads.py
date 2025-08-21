#!/usr/bin/env python3
"""
Mixtral-8x22B Download Tracker
Monitors both Base and Instruct model downloads
"""
import os
import time
from pathlib import Path

def format_size(bytes):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"

def check_model_progress(model_dir, model_name):
    """Check download progress for a specific model"""
    cache_dir = Path(f"/Users/rohanvinaik/LLM_Models/{model_dir}/.cache/huggingface/download")
    final_dir = Path(f"/Users/rohanvinaik/LLM_Models/{model_dir}")
    
    incomplete_files = []
    complete_files = []
    total_incomplete = 0
    total_complete = 0
    
    # Check cache for incomplete downloads
    if cache_dir.exists():
        for file in cache_dir.glob("*.incomplete"):
            size = file.stat().st_size
            incomplete_files.append((file.name[:30] + "...", size))
            total_incomplete += size
    
    # Check for completed GGUF files
    for file in final_dir.glob("*.gguf"):
        size = file.stat().st_size
        complete_files.append((file.name, size))
        total_complete += size
    
    return {
        'name': model_name,
        'incomplete': incomplete_files,
        'complete': complete_files,
        'total_incomplete': total_incomplete,
        'total_complete': total_complete,
        'total': total_incomplete + total_complete
    }

def display_progress():
    """Display current progress for both models"""
    print("\n" + "=" * 70)
    print("🚀 Mixtral-8x22B Download Progress Monitor")
    print("=" * 70)
    
    # Check both models
    base_progress = check_model_progress("Mixtral-8x22B-Base-Q4", "Base Model")
    instruct_progress = check_model_progress("Mixtral-8x22B-Instruct-Q4", "Instruct Model")
    
    for model in [base_progress, instruct_progress]:
        print(f"\n📦 {model['name']}:")
        print("-" * 50)
        
        if model['complete']:
            print("✅ Completed files:")
            for name, size in model['complete']:
                print(f"   {name}: {format_size(size)}")
            print(f"   Total complete: {format_size(model['total_complete'])}")
        
        if model['incomplete']:
            print("⏳ Downloading:")
            for name, size in model['incomplete']:
                print(f"   {name}: {format_size(size)}")
            print(f"   Total downloading: {format_size(model['total_incomplete'])}")
        
        if not model['complete'] and not model['incomplete']:
            print("   ⏸️ Not started or waiting...")
        
        # Progress estimate (Q4_K_M is approximately 95GB total per model)
        expected_size = 95 * 1024**3  # 95GB
        progress = (model['total'] / expected_size) * 100 if expected_size > 0 else 0
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * progress / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"\n   [{bar}] {progress:.1f}%")
        print(f"   Total: {format_size(model['total'])} / ~95GB")
    
    # Combined stats
    total_downloaded = base_progress['total'] + instruct_progress['total']
    print(f"\n{'=' * 70}")
    print(f"📊 Combined Progress: {format_size(total_downloaded)} / ~190GB")
    print(f"   ({(total_downloaded / (190 * 1024**3) * 100):.1f}% of both models)")

def main():
    print("Starting Mixtral-8x22B download monitor...")
    print("Press Ctrl+C to stop monitoring (downloads continue in background)")
    
    try:
        while True:
            display_progress()
            time.sleep(10)  # Update every 10 seconds
            print("\033[2J\033[H")  # Clear screen
    except KeyboardInterrupt:
        print("\n\n⏸️ Monitoring stopped. Downloads continue in background.")
        display_progress()  # Show final status

if __name__ == "__main__":
    main()