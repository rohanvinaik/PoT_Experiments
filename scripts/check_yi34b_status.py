#!/usr/bin/env python3
"""Quick check of Yi-34B test status."""

import os
import json
import psutil
from datetime import datetime

print(f"\n{'='*60}")
print(f"Yi-34B Test Status Check")
print(f"Time: {datetime.now()}")
print(f"{'='*60}\n")

# Check for running process
yi_process = None
for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
    try:
        if 'yi34b' in str(proc.info['cmdline']).lower():
            yi_process = proc
            break
    except:
        pass

if yi_process:
    print("✅ Yi-34B test STILL RUNNING")
    print(f"   PID: {yi_process.pid}")
    print(f"   CPU: {yi_process.cpu_percent()}%")
    print(f"   Memory: {yi_process.memory_percent():.1f}%")
    
    # Calculate runtime
    create_time = datetime.fromtimestamp(yi_process.create_time())
    runtime = datetime.now() - create_time
    print(f"   Runtime: {runtime.total_seconds()/60:.1f} minutes")
    
    # Memory details
    mem_info = yi_process.memory_info()
    print(f"   RSS Memory: {mem_info.rss/1e9:.1f}GB")
    print(f"   VMS Memory: {mem_info.vms/1e9:.1f}GB")
else:
    print("❌ Yi-34B test NOT running")

# Check for output files
print("\n📁 Output files:")
results_dir = "experimental_results"
yi_files = [f for f in os.listdir(results_dir) if 'yi34b' in f.lower()]
for f in sorted(yi_files)[-5:]:
    file_path = os.path.join(results_dir, f)
    size = os.path.getsize(file_path)
    print(f"   {f} ({size} bytes)")

# Check system memory
print("\n💾 System Memory:")
vm = psutil.virtual_memory()
print(f"   Total: {vm.total/1e9:.1f}GB")
print(f"   Available: {vm.available/1e9:.1f}GB")
print(f"   Used: {vm.percent}%")

print("\n🔍 Analysis:")
if yi_process:
    runtime_mins = runtime.total_seconds()/60
    if runtime_mins > 20:
        print("   ⚠️ Test running longer than expected (>20 min)")
        print("   Possible reasons:")
        print("   - Yi-34B inference is extremely slow on CPU")
        print("   - Each token takes 5-10 seconds to generate")
        print("   - With 20 tokens, could take 100-200 seconds")
        print("   - Memory bandwidth limitations with 64GB model")
    else:
        print("   Test duration within expected range")
        print("   34B model inference typically takes 15-25 minutes on CPU")
    
    print(f"\n   Estimated completion: 5-10 more minutes")
    print("   (Based on typical 34B CPU inference rates)")
else:
    print("   Test completed or was terminated")
    print("   Check output files for results")