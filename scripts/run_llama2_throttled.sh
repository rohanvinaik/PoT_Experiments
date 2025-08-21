#!/bin/bash

# Resource-throttled testing for Llama-2-7B models
# Tests base vs chat variants with CPU/RAM limits to prevent crashes

echo "=========================================="
echo "🚀 Llama-2-7B Model Testing (Throttled)"
echo "=========================================="

# Set working directory
cd /Users/rohanvinaik/PoT_Experiments || exit 1

# Resource limits - conservative for 7B models
export OMP_NUM_THREADS=6          # Use 6 of 10 cores
export MKL_NUM_THREADS=6
export TORCH_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_DATASETS_CACHE=/tmp/datasets_cache
export CUDA_VISIBLE_DEVICES=""    # Force CPU only

# Memory management
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=100000

# Models to test
MODEL1="/Users/rohanvinaik/LLM_Models/llama-2-7b-hf"
MODEL2="/Users/rohanvinaik/LLM_Models/llama-2-7b-chat-hf"

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experimental_results/llama2_test_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/execution.log"
mkdir -p "$OUTPUT_DIR"

echo "Configuration:" | tee "$LOG_FILE"
echo "  Models: Llama-2-7B Base vs Chat" | tee -a "$LOG_FILE"
echo "  CPU Threads: 6/10" | tee -a "$LOG_FILE"
echo "  Memory: ~30GB target" | tee -a "$LOG_FILE"
echo "  Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "  Started: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check initial memory
echo "Initial system state:" | tee -a "$LOG_FILE"
top -l 1 -n 0 | grep PhysMem | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to monitor memory during execution
monitor_memory() {
    while true; do
        echo "[$(date +%H:%M:%S)] Memory: $(top -l 1 -n 0 | grep PhysMem)" >> "$OUTPUT_DIR/memory_monitor.log"
        sleep 10
    done
}

# Start memory monitoring in background
monitor_memory &
MONITOR_PID=$!
echo "Memory monitor started (PID: $MONITOR_PID)" | tee -a "$LOG_FILE"

# Trap to ensure monitor is killed on exit
trap "kill $MONITOR_PID 2>/dev/null" EXIT

echo "========================================" | tee -a "$LOG_FILE"
echo "PHASE 1: Configuration Analysis" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Compare configurations
python3 << EOF 2>&1 | tee -a "$LOG_FILE"
import json
import hashlib
from pathlib import Path

models = {
    "base": "$MODEL1",
    "chat": "$MODEL2"
}

print("\nComparing model configurations...")
configs = {}

for name, path in models.items():
    config_path = Path(path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        configs[name] = {
            "architecture": config.get("architectures", ["unknown"])[0],
            "hidden_size": config.get("hidden_size"),
            "num_layers": config.get("num_hidden_layers"),
            "num_heads": config.get("num_attention_heads"),
            "vocab_size": config.get("vocab_size"),
            "max_position": config.get("max_position_embeddings"),
            "hash": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest()[:16]
        }
        
        print(f"\n{name.upper()} model:")
        for key, value in configs[name].items():
            if key != "hash":
                print(f"  {key}: {value}")

# Compare
if len(configs) == 2:
    if configs["base"]["hash"] == configs["chat"]["hash"]:
        print("\n✅ Configurations are IDENTICAL")
    else:
        print("\n⚠️ Configurations DIFFER (expected for base vs chat)")
        
# Save configs
import json
with open("$OUTPUT_DIR/config_comparison.json", "w") as f:
    json.dump(configs, f, indent=2)

print("\nConfig comparison saved to $OUTPUT_DIR/config_comparison.json")
EOF

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "PHASE 2: Statistical Verification" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Run the enhanced diff test with heavy throttling
echo "Running statistical test with minimal resources..." | tee -a "$LOG_FILE"
echo "Using enhanced diff decision framework..." | tee -a "$LOG_FILE"

# Use nice to lower priority and limit resources
nice -n 15 python3 scripts/run_enhanced_diff_test.py \
    --ref-model "$MODEL1" \
    --cand-model "$MODEL2" \
    --mode quick \
    --prf-key deadbeefcafebabe1234567890abcdef \
    --verbose 2>&1 | tee -a "$LOG_FILE"

# Check if test completed
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Statistical test completed successfully" | tee -a "$LOG_FILE"
else
    echo "⚠️ Statistical test encountered issues" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "PHASE 3: Security Tests" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Run security tests if available
if [ -f "scripts/run_security_tests_simple.py" ]; then
    echo "Running security verification..." | tee -a "$LOG_FILE"
    nice -n 15 python3 scripts/run_security_tests_simple.py 2>&1 | tail -20 | tee -a "$LOG_FILE"
else
    echo "Security test script not found" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "PHASE 4: Full Pipeline (Optional)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Only run full pipeline if requested
if [ "$1" == "--full" ]; then
    echo "Running full validation pipeline..." | tee -a "$LOG_FILE"
    nice -n 15 bash scripts/run_all.sh --skip-zk 2>&1 | tee -a "$LOG_FILE"
else
    echo "Skipping full pipeline (use --full to enable)" | tee -a "$LOG_FILE"
fi

# Stop memory monitor
kill $MONITOR_PID 2>/dev/null

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "FINAL REPORT" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Generate summary report
python3 << EOF 2>&1 | tee -a "$LOG_FILE"
import json
from pathlib import Path
from datetime import datetime

output_dir = Path("$OUTPUT_DIR")
report = {
    "timestamp": datetime.now().isoformat(),
    "models": {
        "reference": "llama-2-7b-hf (Base)",
        "candidate": "llama-2-7b-chat-hf (Chat)"
    },
    "resource_limits": {
        "cpu_threads": 6,
        "memory_target_gb": 30,
        "nice_priority": 15
    }
}

# Check for test results
if (output_dir / "config_comparison.json").exists():
    with open(output_dir / "config_comparison.json") as f:
        report["config_comparison"] = json.load(f)
        
print("\nSUMMARY:")
print(f"  Models: Llama-2-7B Base vs Chat")
print(f"  Test completed: {report['timestamp']}")
print(f"  Resource limits: {report['resource_limits']['cpu_threads']} CPU threads")

# Save report
with open(output_dir / "final_report.json", "w") as f:
    json.dump(report, f, indent=2)
    
print(f"\n✅ Report saved to: {output_dir}/final_report.json")

# Check memory log
mem_log = output_dir / "memory_monitor.log"
if mem_log.exists():
    with open(mem_log) as f:
        lines = f.readlines()
    if lines:
        print(f"\nMemory usage samples: {len(lines)}")
        print(f"  First: {lines[0].strip()}")
        if len(lines) > 1:
            print(f"  Last:  {lines[-1].strip()}")
EOF

# Final system check
echo "" | tee -a "$LOG_FILE"
echo "Final system state:" | tee -a "$LOG_FILE"
top -l 1 -n 0 | grep PhysMem | tee -a "$LOG_FILE"
echo "CPU usage:" | tee -a "$LOG_FILE"
top -l 1 -n 0 | grep "CPU usage" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "✅ Testing completed!" | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"