#!/usr/bin/env python3
"""Monitor the 5000-prompt test progress."""

import json
import os
import time
from datetime import datetime, timedelta

checkpoint_file = "experimental_results/qwen_5000_checkpoint.json"
log_file = "experimental_results/qwen_5000_output.log"
pid_file = "experimental_results/qwen_5000_pid.txt"

print("="*70)
print("MONITORING 5000-PROMPT TEST")
print("="*70)

# Check if process is running
if os.path.exists(pid_file):
    with open(pid_file, 'r') as f:
        pid = f.read().strip()
    
    # Check if process exists
    try:
        os.kill(int(pid), 0)
        print(f"✓ Process {pid} is running")
    except ProcessLookupError:
        print(f"✗ Process {pid} has stopped")

# Check log file
if os.path.exists(log_file):
    # Get file size and last modified time
    file_size = os.path.getsize(log_file)
    mod_time = os.path.getmtime(log_file)
    last_modified = datetime.fromtimestamp(mod_time)
    age = datetime.now() - last_modified
    
    print(f"\nLog file:")
    print(f"  Size: {file_size} bytes")
    print(f"  Last updated: {age.total_seconds():.0f} seconds ago")
    
    # Read last non-metal lines
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Filter out metal init messages
    content_lines = [l for l in lines if 'ggml_metal' not in l and 'llama_context' not in l]
    if content_lines:
        print(f"\nLast content (non-Metal):")
        for line in content_lines[-5:]:
            print(f"  {line.rstrip()}")
    else:
        print("\nNo content yet (still loading model)")

# Check checkpoint
if os.path.exists(checkpoint_file):
    print(f"\n✓ Checkpoint found!")
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    
    stats = checkpoint.get('stats', {})
    print(f"\nProgress:")
    print(f"  Prompts completed: {stats.get('prompts_completed', 0)}/5000")
    print(f"  Mean difference: {stats.get('mean_diff', 0):.8f}")
    print(f"  Total time: {stats.get('total_time', 0):.1f}s")
    
    if stats.get('prompts_completed', 0) > 0:
        rate = stats['prompts_completed'] / stats['total_time']
        remaining = 5000 - stats['prompts_completed']
        eta_seconds = remaining / rate
        eta = datetime.now() + timedelta(seconds=eta_seconds)
        
        print(f"\nPerformance:")
        print(f"  Rate: {rate:.2f} prompts/second")
        print(f"  ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Time remaining: {eta_seconds/3600:.1f} hours")
else:
    print("\n⚠ No checkpoint yet - test hasn't started saving progress")

print("\n" + "="*70)
print("ESTIMATED COMPLETION")
print("="*70)

# Based on our earlier tests, estimate completion
print("\nBased on observed performance:")
print("  ~4-10 seconds per prompt for 72B model")
print("  5000 prompts × 7 seconds (avg) = 35,000 seconds")
print("  Expected total time: ~9.7 hours")
print(f"  Expected completion: {(datetime.now() + timedelta(hours=9.7)).strftime('%Y-%m-%d %H:%M:%S')}")

print("\nTo continue monitoring:")
print("  Run this script again: python scripts/monitor_5000_test.py")
print("  Watch log: tail -f experimental_results/qwen_5000_output.log | grep -v ggml_metal")
print("  Check checkpoint: cat experimental_results/qwen_5000_checkpoint.json | python -m json.tool")
print("\n" + "="*70)