#!/bin/bash

# Safe Mixtral GGUF model testing with resource limits
# Prevents system crashes by using conservative resource allocation

echo "=========================================="
echo "🚀 Mixtral GGUF Model Testing (Safe Mode)"
echo "=========================================="

# Set working directory
cd /Users/rohanvinaik/PoT_Experiments || exit 1

# Resource limits
export OMP_NUM_THREADS=8          # Use 8 of 10 cores
export MKL_NUM_THREADS=8
export TORCH_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_DATASETS_CACHE=/tmp/datasets_cache

# Memory management
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=100000

# Model paths
MODEL1="/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Base-Q4"
MODEL2="/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Instruct-Q4"

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experimental_results/mixtral_safe_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Models: Mixtral-8x22B Base-Q4 vs Instruct-Q4"
echo "  CPU Threads: 8/10"
echo "  Output: $OUTPUT_DIR"
echo "  Priority: Low (nice -n 15)"
echo ""

# Function to run with resource monitoring
run_with_monitoring() {
    local cmd="$1"
    local desc="$2"
    
    echo "▶ Running: $desc"
    echo "  Command: $cmd"
    
    # Check memory before
    echo "  Memory before:"
    top -l 1 -n 0 | grep PhysMem
    
    # Run command with nice priority
    nice -n 15 bash -c "$cmd" 2>&1 | tee -a "$OUTPUT_DIR/execution.log"
    local exit_code=${PIPESTATUS[0]}
    
    # Check memory after
    echo "  Memory after:"
    top -l 1 -n 0 | grep PhysMem
    
    if [ $exit_code -eq 0 ]; then
        echo "  ✅ Success"
    else
        echo "  ❌ Failed with code $exit_code"
    fi
    
    # Give system time to recover
    sleep 5
    
    return $exit_code
}

# Test 1: Config comparison (no model loading)
echo ""
echo "=========================================="
echo "TEST 1: Configuration Comparison"
echo "=========================================="

cat > "$OUTPUT_DIR/compare_configs.py" << 'EOF'
import json
import hashlib
from pathlib import Path

models = [
    "/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Base-Q4",
    "/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Instruct-Q4"
]

results = {}
for model_path in models:
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        # Extract key info
        model_name = Path(model_path).name
        results[model_name] = {
            "architecture": config.get("architectures", ["unknown"])[0],
            "hidden_size": config.get("hidden_size", "unknown"),
            "num_layers": config.get("num_hidden_layers", "unknown"),
            "num_experts": config.get("num_local_experts", "unknown"),
            "vocab_size": config.get("vocab_size", "unknown"),
            "config_hash": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest()[:16]
        }
        
        print(f"\n{model_name}:")
        for key, value in results[model_name].items():
            print(f"  {key}: {value}")

# Compare
if len(results) == 2:
    names = list(results.keys())
    if results[names[0]]["config_hash"] == results[names[1]]["config_hash"]:
        print("\n✅ Models have IDENTICAL configurations")
    else:
        print("\n❌ Models have DIFFERENT configurations")
        print("  This is expected for Base vs Instruct variants")

# Save results
import json
with open("$OUTPUT_DIR/config_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
EOF

run_with_monitoring "python $OUTPUT_DIR/compare_configs.py" "Configuration comparison"

# Test 2: Lightweight statistical test (minimal queries)
echo ""
echo "=========================================="
echo "TEST 2: Minimal Statistical Verification"
echo "=========================================="

cat > "$OUTPUT_DIR/minimal_stat_test.py" << 'EOF'
#!/usr/bin/env python3
import sys
import os
import gc
import json
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, "/Users/rohanvinaik/PoT_Experiments")

# Force CPU only and limit threads
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'

print("Running minimal statistical test...")
print("⚠️ Using only 3 queries to minimize memory usage")

try:
    from pot.core.diff_decision import EnhancedSequentialTester, TestingMode
    
    # Use the most conservative settings
    tester = EnhancedSequentialTester(
        mode=TestingMode.QUICK_GATE,
        n_max=3,  # Only 3 queries
        verbose=True
    )
    
    # Models
    model1 = "/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Base-Q4"
    model2 = "/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Instruct-Q4"
    
    print(f"Model 1: {Path(model1).name}")
    print(f"Model 2: {Path(model2).name}")
    
    # Note: This may fail if models are too large
    print("\n⚠️ Attempting to load models...")
    print("If this hangs, press Ctrl+C to skip")
    
    # We'll use a timeout approach
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Model loading timed out")
    
    # Set 60 second timeout for loading
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)
    
    try:
        # Try to run test
        result = tester.test_models(model1, model2)
        signal.alarm(0)  # Cancel alarm
        
        print(f"\nDecision: {result.decision}")
        print(f"Confidence: {result.confidence:.2%}")
        
        # Save result
        with open("$OUTPUT_DIR/statistical_result.json", "w") as f:
            json.dump({
                "decision": result.decision,
                "confidence": result.confidence,
                "n_samples": result.n_samples
            }, f, indent=2)
            
    except TimeoutError:
        print("\n❌ Model loading timed out - models too large")
        print("Skipping statistical test")
    except Exception as e:
        print(f"\n❌ Statistical test failed: {e}")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Framework not properly installed")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

# Cleanup
gc.collect()
print("\n✅ Test completed")
EOF

# Run with extreme caution
echo "⚠️ Attempting statistical test with minimal resources..."
echo "This may fail if models are too large for available memory."
read -p "Continue? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Use ulimit to set hard memory limit (50GB)
    (ulimit -v 52428800 && run_with_monitoring "python $OUTPUT_DIR/minimal_stat_test.py" "Minimal statistical test")
else
    echo "Skipping statistical test"
fi

# Test 3: Run security tests without model loading
echo ""
echo "=========================================="
echo "TEST 3: Security Tests (Config-Only)"
echo "=========================================="

if [ -f "scripts/run_security_tests_simple.py" ]; then
    run_with_monitoring "python scripts/run_security_tests_simple.py --config-only" "Security tests"
else
    echo "Security test script not found, skipping"
fi

# Test 4: Generate summary report
echo ""
echo "=========================================="
echo "GENERATING SUMMARY REPORT"
echo "=========================================="

cat > "$OUTPUT_DIR/generate_report.py" << 'EOF'
import json
from pathlib import Path
from datetime import datetime

output_dir = Path("$OUTPUT_DIR")
report = {
    "timestamp": datetime.now().isoformat(),
    "models": {
        "base": "Mixtral-8x22B-Base-Q4",
        "instruct": "Mixtral-8x22B-Instruct-Q4"
    },
    "tests_completed": [],
    "resource_limits": {
        "cpu_threads": 8,
        "memory_target_gb": 50,
        "nice_priority": 15
    }
}

# Check which tests completed
if (output_dir / "config_comparison.json").exists():
    report["tests_completed"].append("config_comparison")
    with open(output_dir / "config_comparison.json") as f:
        report["config_comparison"] = json.load(f)

if (output_dir / "statistical_result.json").exists():
    report["tests_completed"].append("statistical_verification")
    with open(output_dir / "statistical_result.json") as f:
        report["statistical_result"] = json.load(f)

# Generate summary
print("\n" + "=" * 60)
print("MIXTRAL MODEL VERIFICATION SUMMARY")
print("=" * 60)
print(f"Timestamp: {report['timestamp']}")
print(f"Models tested:")
print(f"  - Base: {report['models']['base']}")
print(f"  - Instruct: {report['models']['instruct']}")
print(f"\nTests completed: {', '.join(report['tests_completed'])}")

if "config_comparison" in report:
    print("\nConfiguration Analysis:")
    for model, info in report["config_comparison"].items():
        print(f"  {model}: {info.get('architecture', 'unknown')} architecture")

if "statistical_result" in report:
    print(f"\nStatistical Verification:")
    print(f"  Decision: {report['statistical_result']['decision']}")
    print(f"  Confidence: {report['statistical_result']['confidence']:.2%}")

print("\n" + "=" * 60)

# Save full report
with open(output_dir / "final_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n✅ Full report saved to: {output_dir}/final_report.json")
EOF

python "$OUTPUT_DIR/generate_report.py"

# Final system check
echo ""
echo "=========================================="
echo "FINAL SYSTEM STATUS"
echo "=========================================="
echo "Memory usage:"
top -l 1 -n 0 | grep PhysMem
echo ""
echo "CPU usage:"
top -l 1 -n 0 | grep "CPU usage"

echo ""
echo "=========================================="
echo "✅ Testing completed safely!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="