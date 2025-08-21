#!/usr/bin/env python3
"""
DeepSeek-R1 Download Tracker and Resumer
Monitors progress and resumes downloads on interruption
"""
import os
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download

def get_file_sizes():
    """Get current download progress"""
    cache_dir = Path("/Users/rohanvinaik/LLM_Models/DeepSeek-R1-UD-IQ1_M/.cache/huggingface/download/DeepSeek-R1-UD-IQ1_M")
    if not cache_dir.exists():
        return {}
    
    files = {}
    for file in cache_dir.glob("*.incomplete"):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            files[file.name] = size_mb
    return files

def show_progress():
    """Display current download progress"""
    files = get_file_sizes()
    if not files:
        print("📋 No download cache found - starting fresh download")
        return
    
    print("📊 Current Download Progress:")
    print("=" * 50)
    total_mb = 0
    for filename, size_mb in files.items():
        print(f"  {filename[:30]}... {size_mb:8.1f} MB")
        total_mb += size_mb
    
    print(f"\n📈 Total downloaded: {total_mb:,.1f} MB ({total_mb/1024:.1f} GB)")
    
    # Estimate based on typical GGUF model sizes
    estimated_total = 25000  # ~25GB for IQ1_M quantized 671B model
    progress_pct = (total_mb / estimated_total) * 100
    print(f"🎯 Estimated progress: {progress_pct:.1f}%")
    
    return total_mb

def resume_download():
    """Resume the DeepSeek download"""
    print("\n🚀 Resuming DeepSeek-R1 Download")
    print("=" * 50)
    
    # Set optimal download environment
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '7200'  # 2 hours timeout
    os.environ['HF_HUB_DOWNLOAD_WORKERS'] = '4'      # Parallel downloads
    
    target_dir = "/Users/rohanvinaik/LLM_Models/DeepSeek-R1-UD-IQ1_M"
    
    try:
        print("📡 Starting download with optimized settings...")
        print(f"📁 Target directory: {target_dir}")
        print("🔄 Resume mode: Enabled")
        print("⚡ Workers: 4 parallel downloads")
        print("")
        
        # Use snapshot_download for better progress tracking
        local_dir = snapshot_download(
            repo_id="unsloth/DeepSeek-R1-GGUF",
            allow_patterns=["DeepSeek-R1-UD-IQ1_M/*"],
            local_dir=target_dir,
            resume_download=True,
            max_workers=4
        )
        
        print(f"\n✅ Download complete! Model at: {local_dir}")
        
        # Verify files
        model_files = list(Path(target_dir).glob("**/*.gguf"))
        print(f"📁 Downloaded {len(model_files)} GGUF files:")
        for file in model_files:
            size_gb = file.stat().st_size / (1024**3)
            print(f"  - {file.name} ({size_gb:.1f} GB)")
            
    except KeyboardInterrupt:
        print("\n⚠️ Download paused by user")
        print("💡 Run this script again to resume from current position")
        return False
    except Exception as e:
        print(f"\n❌ Download error: {e}")
        print("🔄 Retrying in 10 seconds...")
        time.sleep(10)
        return False
    
    return True

def main():
    print("🤖 DeepSeek-R1 Download Tracker")
    print("=" * 50)
    
    # Show current progress
    current_size = show_progress()
    
    if current_size and current_size > 0:
        print(f"\n🔄 Resuming download from {current_size:,.1f} MB...")
    else:
        print("\n🆕 Starting fresh download...")
    
    # Resume download
    success = resume_download()
    
    if success:
        print("\n🎉 DeepSeek-R1 is ready for use!")
        print(f"📍 Location: /Users/rohanvinaik/LLM_Models/DeepSeek-R1-UD-IQ1_M")
    else:
        print("\n⏸️ Download incomplete - run script again to continue")

if __name__ == "__main__":
    main()