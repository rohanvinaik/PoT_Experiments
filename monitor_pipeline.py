#!/usr/bin/env python3
"""
Real-time monitor for PoT Pipeline execution
Shows progress, ETA, and performance metrics
"""

import time
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import re

def clear_screen():
    os.system('clear')

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def main():
    # Find the latest log file
    log_dir = Path("experimental_results")
    log_files = list(log_dir.glob("qwen_pipeline_*.log"))
    
    if not log_files:
        print("No pipeline log files found!")
        return
    
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    
    # Find checkpoint file
    checkpoint_files = list(log_dir.glob("qwen_pipeline_checkpoint_*.json"))
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime) if checkpoint_files else None
    
    print(f"Monitoring: {latest_log.name}")
    if latest_checkpoint:
        print(f"Checkpoint: {latest_checkpoint.name}")
    print("Press Ctrl+C to exit\n")
    
    start_time = time.time()
    last_checkpoint_data = {}
    
    try:
        while True:
            clear_screen()
            
            # Header
            print("="*80)
            print("POT FRAMEWORK PIPELINE MONITOR - QWEN 72B")
            print("="*80)
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"Runtime: {format_time(time.time() - start_time)}")
            print("="*80)
            
            # Read log file
            with open(latest_log, 'r') as f:
                lines = f.readlines()
            
            # Extract progress information
            current_prompt = 0
            total_prompts = 5000
            avg_time = 0
            eta_hours = 0
            mean_diff = 0
            current_phase = "Unknown"
            
            for line in reversed(lines):
                if "Progress:" in line and "/" in line:
                    match = re.search(r'Progress: (\d+)/(\d+)', line)
                    if match:
                        current_prompt = int(match.group(1))
                        total_prompts = int(match.group(2))
                
                if "Avg time:" in line:
                    match = re.search(r'Avg time: ([\d.]+)s', line)
                    if match:
                        avg_time = float(match.group(1))
                
                if "ETA:" in line:
                    match = re.search(r'ETA: ([\d.]+) hours', line)
                    if match:
                        eta_hours = float(match.group(1))
                
                if "Mean diff so far:" in line:
                    match = re.search(r'Mean diff so far: ([\d.]+)', line)
                    if match:
                        mean_diff = float(match.group(1))
                
                if "PHASE" in line:
                    current_phase = line.strip()
            
            # Read checkpoint if available
            if latest_checkpoint and latest_checkpoint.exists():
                try:
                    with open(latest_checkpoint, 'r') as f:
                        checkpoint_data = json.load(f)
                        if checkpoint_data != last_checkpoint_data:
                            last_checkpoint_data = checkpoint_data
                            if 'last_completed' in checkpoint_data:
                                current_prompt = checkpoint_data['last_completed'] + 1
                            if 'stats' in checkpoint_data:
                                stats = checkpoint_data['stats']
                                if 'mean_diff' in stats:
                                    mean_diff = stats['mean_diff']
                except:
                    pass
            
            # Display current phase
            print(f"\n📍 CURRENT PHASE: {current_phase}")
            
            # Progress bar
            if current_prompt > 0:
                progress = current_prompt / total_prompts
                bar_width = 50
                filled = int(bar_width * progress)
                bar = '█' * filled + '░' * (bar_width - filled)
                
                print(f"\n📊 BEHAVIORAL VERIFICATION PROGRESS:")
                print(f"[{bar}] {progress*100:.1f}%")
                print(f"Prompts: {current_prompt:,}/{total_prompts:,}")
                
                # Performance metrics
                print(f"\n⚡ PERFORMANCE:")
                if avg_time > 0:
                    print(f"  Speed: {avg_time:.2f}s per prompt")
                    print(f"  Throughput: {1/avg_time:.2f} prompts/sec")
                    tokens_per_sec = (50 * 2) / avg_time  # 50 tokens, 2 generations
                    print(f"  Token rate: {tokens_per_sec:.1f} tokens/sec")
                
                # Time estimates
                print(f"\n⏱️  TIME ESTIMATES:")
                if eta_hours > 0:
                    eta_time = datetime.now() + timedelta(hours=eta_hours)
                    print(f"  ETA: {eta_time.strftime('%H:%M:%S')} ({eta_hours:.1f} hours)")
                
                elapsed = time.time() - start_time
                if current_prompt > 10:
                    total_estimated = (elapsed / current_prompt) * total_prompts
                    print(f"  Total estimated: {format_time(total_estimated)}")
                
                # Verification results
                print(f"\n✅ VERIFICATION:")
                print(f"  Mean difference: {mean_diff:.8f}")
                print(f"  Status: {'IDENTICAL' if mean_diff < 0.001 else 'DIFFERENCES DETECTED'}")
            
            # Show recent log lines
            print(f"\n📝 RECENT LOG:")
            print("-" * 80)
            recent_lines = [l.strip() for l in lines[-10:] if l.strip()]
            for line in recent_lines[-5:]:
                if len(line) > 78:
                    line = line[:75] + "..."
                print(f"  {line}")
            
            # Comparison preview
            if current_prompt > 100:
                print(f"\n🎯 PROJECTED COMPARISON (at current rate):")
                if avg_time > 0:
                    projected_total = avg_time * 5000
                    print(f"  PoT on M1 Max: {projected_total/3600:.1f} hours")
                    print(f"  A100 baseline: 3.0 hours")
                    print(f"  Ratio: {projected_total/10800:.1f}× slower")
                    print(f"  Efficiency: ~{30/(projected_total/10800):.0f}× better than compute ratio")
            
            # Memory usage
            print(f"\n💾 SYSTEM:")
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                    if 'python' in proc.info['name'].lower() and 'qwen' in ' '.join(proc.cmdline()).lower():
                        mem_gb = proc.memory_info().rss / 1024 / 1024 / 1024
                        print(f"  Memory: {mem_gb:.1f} GB ({proc.info['memory_percent']:.1f}%)")
                        break
            except:
                pass
            
            print("\n" + "="*80)
            print("Press Ctrl+C to exit monitor (pipeline continues running)")
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitor stopped. Pipeline continues in background.")
        print(f"Check log: {latest_log}")

if __name__ == "__main__":
    main()