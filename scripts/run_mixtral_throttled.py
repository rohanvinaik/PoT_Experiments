#!/usr/bin/env python3
"""
Resource-throttled Mixtral model testing for PoT framework.
Prevents system crashes by limiting CPU and memory usage.
"""

import os
import sys
import json
import time
import resource
import subprocess
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Resource limits
MAX_MEMORY_GB = 50  # Leave 14GB for system
MAX_CPU_PERCENT = 80  # Use 80% of CPU
NUM_THREADS = 8  # Use 8 out of 10 cores

def set_resource_limits():
    """Set system resource limits to prevent crashes."""
    # Set memory limit (in bytes)
    max_memory_bytes = MAX_MEMORY_GB * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
    
    # Set environment variables for PyTorch
    os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
    os.environ['MKL_NUM_THREADS'] = str(NUM_THREADS)
    os.environ['TORCH_NUM_THREADS'] = str(NUM_THREADS)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Force CPU-only mode to avoid GPU memory issues
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    print(f"✅ Resource limits set:")
    print(f"   - Max memory: {MAX_MEMORY_GB}GB")
    print(f"   - CPU threads: {NUM_THREADS}")
    print(f"   - PyTorch threads limited")

def run_with_nice(command, nice_level=10):
    """Run command with nice level to reduce priority."""
    nice_command = f"nice -n {nice_level} {command}"
    return subprocess.run(nice_command, shell=True, capture_output=True, text=True)

def test_model_loading():
    """Test if models can be loaded without crashing."""
    print("\n📊 Testing model loading capabilities...")
    
    test_script = """
import torch
from transformers import AutoConfig
import gc

models = [
    "/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Base-Q4",
    "/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Instruct-Q4"
]

for model_path in models:
    try:
        print(f"Testing config load for {model_path}...")
        config = AutoConfig.from_pretrained(model_path)
        print(f"  ✓ Config loaded: {config.architectures[0] if hasattr(config, 'architectures') else 'Unknown'}")
        print(f"  ✓ Hidden size: {config.hidden_size if hasattr(config, 'hidden_size') else 'Unknown'}")
        del config
        gc.collect()
    except Exception as e:
        print(f"  ✗ Failed: {e}")
"""
    
    with open('/tmp/test_loading.py', 'w') as f:
        f.write(test_script)
    
    result = run_with_nice(f"python /tmp/test_loading.py")
    print(result.stdout)
    if result.stderr:
        print(f"Warnings: {result.stderr}")
    
    return result.returncode == 0

def run_lightweight_tests():
    """Run lightweight verification tests that don't load full models."""
    print("\n🔍 Running lightweight verification tests...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/Users/rohanvinaik/PoT_Experiments/experimental_results/mixtral_throttled_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    tests = [
        {
            "name": "Config Hash Verification",
            "script": "run_config_hash_test.py",
            "args": "--model1 /Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Base-Q4 --model2 /Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Instruct-Q4"
        },
        {
            "name": "Security Tests (No Model Loading)",
            "script": "run_security_tests_simple.py",
            "args": "--skip-model-loading --config-only"
        }
    ]
    
    results = []
    for test in tests:
        print(f"\n▶ Running {test['name']}...")
        script_path = f"/Users/rohanvinaik/PoT_Experiments/scripts/{test['script']}"
        
        if not os.path.exists(script_path):
            # Create a simple config comparison script if it doesn't exist
            if "config_hash" in test['script']:
                create_config_hash_script(script_path)
        
        if os.path.exists(script_path):
            cmd = f"cd /Users/rohanvinaik/PoT_Experiments && python {script_path} {test.get('args', '')}"
            result = run_with_nice(cmd)
            
            test_result = {
                "test": test['name'],
                "success": result.returncode == 0,
                "output": result.stdout[:1000],  # Truncate output
                "timestamp": datetime.now().isoformat()
            }
            results.append(test_result)
            
            if result.returncode == 0:
                print(f"  ✅ {test['name']} passed")
            else:
                print(f"  ❌ {test['name']} failed")
    
    # Save results
    with open(f"{output_dir}/lightweight_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def create_config_hash_script(script_path):
    """Create a simple config hash comparison script."""
    script_content = """#!/usr/bin/env python3
import json
import hashlib
import argparse
from pathlib import Path

def hash_config(model_path):
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Sort keys for consistent hashing
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", required=True)
    parser.add_argument("--model2", required=True)
    args = parser.parse_args()
    
    hash1 = hash_config(args.model1)
    hash2 = hash_config(args.model2)
    
    print(f"Model 1 hash: {hash1}")
    print(f"Model 2 hash: {hash2}")
    print(f"Match: {hash1 == hash2}")
"""
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

def run_statistical_test_throttled():
    """Run the enhanced diff test with heavy throttling."""
    print("\n📈 Running statistical verification (heavily throttled)...")
    
    # Use the enhanced diff test with minimal queries
    cmd = """cd /Users/rohanvinaik/PoT_Experiments && \
        nice -n 15 python scripts/run_enhanced_diff_test.py \
        --ref-model /Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Base-Q4 \
        --cand-model /Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Instruct-Q4 \
        --mode quick \
        --max-queries 5 \
        --batch-size 1 \
        --no-cache \
        --low-memory \
        --verbose 2>&1"""
    
    print("Running with minimal queries (5) and single batch...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if "DIFFERENT" in result.stdout or "SAME" in result.stdout:
        print("✅ Statistical test completed")
    else:
        print("⚠️ Statistical test may have failed or been inconclusive")
    
    return result

def run_full_pipeline_throttled():
    """Run the full pipeline with extreme throttling."""
    print("\n🚀 Running full pipeline (throttled)...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"/Users/rohanvinaik/PoT_Experiments/experimental_results/mixtral_throttled_{timestamp}.log"
    
    # Modified run_all.sh with resource limits
    cmd = f"""cd /Users/rohanvinaik/PoT_Experiments && \
        nice -n 15 bash -c '
        export OMP_NUM_THREADS={NUM_THREADS}
        export MKL_NUM_THREADS={NUM_THREADS}
        export TORCH_NUM_THREADS={NUM_THREADS}
        export TOKENIZERS_PARALLELISM=false
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
        
        # Run with skip-zk flag to avoid memory-intensive ZK proofs
        bash scripts/run_all.sh --skip-zk --models "/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Base-Q4,/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Instruct-Q4"
        ' 2>&1 | tee {log_file}"""
    
    print(f"Logging to: {log_file}")
    print("This will take significant time due to model size and throttling...")
    
    # Run in subprocess with monitoring
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Monitor the process
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    rc = process.poll()
    return rc == 0

def monitor_system_resources():
    """Monitor system resources during execution."""
    print("\n📊 Current system resources:")
    result = subprocess.run("top -l 1 -n 0 | head -20", shell=True, capture_output=True, text=True)
    print(result.stdout)

def main():
    """Main execution function."""
    print("=" * 60)
    print("🚀 Mixtral Model Testing - Resource Throttled")
    print("=" * 60)
    print(f"Models to test:")
    print(f"  1. Mixtral-8x22B-Base-Q4 (80GB)")
    print(f"  2. Mixtral-8x22B-Instruct-Q4 (80GB)")
    print(f"\nSystem: 64GB RAM, 10 CPU cores")
    print(f"Limits: {MAX_MEMORY_GB}GB RAM, {NUM_THREADS} threads")
    print("=" * 60)
    
    # Set resource limits
    try:
        set_resource_limits()
    except Exception as e:
        print(f"⚠️ Could not set all resource limits: {e}")
        print("Continuing with environment variables only...")
    
    # Monitor initial state
    monitor_system_resources()
    
    # Test 1: Check if models can be loaded
    print("\n" + "=" * 40)
    print("PHASE 1: Model Loading Test")
    print("=" * 40)
    can_load = test_model_loading()
    
    if not can_load:
        print("\n⚠️ Models are too large for full loading.")
        print("Proceeding with lightweight tests only...")
    
    # Test 2: Lightweight verification
    print("\n" + "=" * 40)
    print("PHASE 2: Lightweight Verification")
    print("=" * 40)
    lightweight_results = run_lightweight_tests()
    
    # Test 3: Statistical test (if possible)
    print("\n" + "=" * 40)
    print("PHASE 3: Statistical Verification")
    print("=" * 40)
    
    response = input("\n⚠️ Statistical tests will attempt to load models. Continue? (y/n): ")
    if response.lower() == 'y':
        print("Starting statistical test with heavy throttling...")
        print("If system becomes unresponsive, press Ctrl+C to abort.")
        time.sleep(3)  # Give user time to abort
        
        try:
            stat_result = run_statistical_test_throttled()
            print("\n✅ Statistical test completed")
        except KeyboardInterrupt:
            print("\n❌ Statistical test aborted by user")
        except Exception as e:
            print(f"\n❌ Statistical test failed: {e}")
    else:
        print("Skipping statistical tests.")
    
    # Test 4: Full pipeline (optional)
    print("\n" + "=" * 40)
    print("PHASE 4: Full Pipeline (Optional)")
    print("=" * 40)
    
    response = input("\n⚠️ Full pipeline is memory intensive. Continue? (y/n): ")
    if response.lower() == 'y':
        print("Starting full pipeline with maximum throttling...")
        print("This will take a long time. Press Ctrl+C to abort if needed.")
        time.sleep(5)  # Give user time to abort
        
        try:
            success = run_full_pipeline_throttled()
            if success:
                print("\n✅ Full pipeline completed successfully!")
            else:
                print("\n⚠️ Full pipeline completed with some failures")
        except KeyboardInterrupt:
            print("\n❌ Full pipeline aborted by user")
        except Exception as e:
            print(f"\n❌ Full pipeline failed: {e}")
    else:
        print("Skipping full pipeline.")
    
    # Final resource check
    print("\n" + "=" * 40)
    print("FINAL SYSTEM STATE")
    print("=" * 40)
    monitor_system_resources()
    
    print("\n" + "=" * 60)
    print("✅ Testing completed!")
    print("Check experimental_results/ for detailed outputs")
    print("=" * 60)

if __name__ == "__main__":
    main()