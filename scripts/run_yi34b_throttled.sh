#!/bin/bash

# Resource-throttled testing for Yi-34B models (34B parameters)
# Extreme throttling to prevent system crashes with these large models

echo "============================================================"
echo "🚀 Yi-34B Model Testing (34B Parameters - Heavily Throttled)"
echo "============================================================"

# Set working directory
cd /Users/rohanvinaik/PoT_Experiments || exit 1

# Aggressive resource limits for 34B models
export OMP_NUM_THREADS=4          # Use only 4 of 10 cores
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=256
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_DATASETS_CACHE=/tmp/datasets_cache
export CUDA_VISIBLE_DEVICES=""    # Force CPU only
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Fallback for MPS issues

# Memory management
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=100000

# Models to test
MODEL1="/Users/rohanvinaik/LLM_Models/yi-34b"
MODEL2="/Users/rohanvinaik/LLM_Models/yi-34b-chat"

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experimental_results/yi34b_test_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/execution.log"
mkdir -p "$OUTPUT_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo "Configuration:" | tee -a "$LOG_FILE"
echo "  Models: Yi-34B Base vs Yi-34B-Chat" | tee -a "$LOG_FILE"
echo "  Model size: 34B parameters each (~68GB per model)" | tee -a "$LOG_FILE"
echo "  CPU Threads: 4/10 (heavily throttled)" | tee -a "$LOG_FILE"
echo "  Memory target: 50GB max" | tee -a "$LOG_FILE"
echo "  Priority: Very low (nice -n 19)" | tee -a "$LOG_FILE"
echo "  Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "  Started: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Check if models exist
echo "" | tee -a "$LOG_FILE"
echo "Checking model availability..." | tee -a "$LOG_FILE"
if [ ! -f "$MODEL1/config.json" ]; then
    echo "❌ Yi-34B base model not found at $MODEL1" | tee -a "$LOG_FILE"
    echo "Please wait for download to complete." | tee -a "$LOG_FILE"
    exit 1
fi
if [ ! -f "$MODEL2/config.json" ]; then
    echo "❌ Yi-34B chat model not found at $MODEL2" | tee -a "$LOG_FILE"
    echo "Please wait for download to complete." | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ Both models found" | tee -a "$LOG_FILE"

# Check initial memory
echo "" | tee -a "$LOG_FILE"
echo "Initial system state:" | tee -a "$LOG_FILE"
top -l 1 -n 0 | grep PhysMem | tee -a "$LOG_FILE"
df -h /tmp | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to monitor memory
monitor_memory() {
    while true; do
        MEM_INFO=$(top -l 1 -n 0 | grep PhysMem)
        echo "[$(date +%H:%M:%S)] $MEM_INFO" >> "$OUTPUT_DIR/memory_monitor.log"
        
        # Check if we're using too much memory
        USED_GB=$(echo "$MEM_INFO" | awk '{print $2}' | sed 's/G//')
        if (( $(echo "$USED_GB > 55" | bc -l) )); then
            echo "[WARNING] Memory usage critical: ${USED_GB}GB" >> "$OUTPUT_DIR/memory_monitor.log"
        fi
        sleep 15
    done
}

# Start memory monitoring
monitor_memory &
MONITOR_PID=$!
echo "Memory monitor started (PID: $MONITOR_PID)" | tee -a "$LOG_FILE"
trap "kill $MONITOR_PID 2>/dev/null" EXIT

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "PHASE 1: Configuration Comparison (No Model Loading)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Compare configurations without loading models
nice -n 19 python3 << 'EOF' 2>&1 | tee -a "$LOG_FILE"
import json
import hashlib
from pathlib import Path
import sys

models = {
    "base": "/Users/rohanvinaik/LLM_Models/yi-34b",
    "chat": "/Users/rohanvinaik/LLM_Models/yi-34b-chat"
}

print("\nAnalyzing Yi-34B model configurations...")
configs = {}

for name, path in models.items():
    config_path = Path(path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        configs[name] = {
            "architecture": config.get("architectures", ["unknown"])[0],
            "model_type": config.get("model_type", "unknown"),
            "hidden_size": config.get("hidden_size", 0),
            "intermediate_size": config.get("intermediate_size", 0),
            "num_layers": config.get("num_hidden_layers", 0),
            "num_heads": config.get("num_attention_heads", 0),
            "num_key_value_heads": config.get("num_key_value_heads", 0),
            "vocab_size": config.get("vocab_size", 0),
            "max_position": config.get("max_position_embeddings", 0),
            "rope_theta": config.get("rope_theta", 0),
            "hash": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest()[:16]
        }
        
        print(f"\n{name.upper()} Model Configuration:")
        print(f"  Architecture: {configs[name]['architecture']}")
        print(f"  Model type: {configs[name]['model_type']}")
        print(f"  Hidden size: {configs[name]['hidden_size']:,}")
        print(f"  Intermediate size: {configs[name]['intermediate_size']:,}")
        print(f"  Layers: {configs[name]['num_layers']}")
        print(f"  Attention heads: {configs[name]['num_heads']}")
        print(f"  Vocab size: {configs[name]['vocab_size']:,}")
        print(f"  Max positions: {configs[name]['max_position']:,}")
        
        # Calculate approximate parameter count
        h = configs[name]['hidden_size']
        l = configs[name]['num_layers']
        v = configs[name]['vocab_size']
        i = configs[name]['intermediate_size']
        
        # Rough estimation
        params = (v * h) + l * (4 * h * h + 2 * h * i)
        print(f"  Estimated parameters: ~{params/1e9:.1f}B")

# Compare
if len(configs) == 2:
    print("\n" + "="*40)
    print("COMPARISON RESULTS:")
    if configs["base"]["hash"] == configs["chat"]["hash"]:
        print("✅ Configurations are IDENTICAL")
    else:
        print("⚠️ Configurations DIFFER")
        # Show differences
        for key in configs["base"]:
            if key != "hash" and configs["base"][key] != configs["chat"][key]:
                print(f"  {key}: {configs['base'][key]} (base) vs {configs['chat'][key]} (chat)")

# Save to file
import json
output_dir = Path("/Users/rohanvinaik/PoT_Experiments/experimental_results").glob("yi34b_test_*")
latest_dir = max(output_dir, key=lambda p: p.stat().st_mtime)
with open(latest_dir / "config_comparison.json", "w") as f:
    json.dump(configs, f, indent=2)
print(f"\nConfig saved to {latest_dir}/config_comparison.json")
EOF

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "PHASE 2: Minimal Statistical Test (5 queries only)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "⚠️ WARNING: This will attempt to load 34B parameter models!" | tee -a "$LOG_FILE"
echo "Expected memory usage: ~50GB+" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check available memory
AVAIL_MEM=$(top -l 1 -n 0 | grep PhysMem | awk '{print $6}' | sed 's/G//')
echo "Available memory: ${AVAIL_MEM}GB" | tee -a "$LOG_FILE"

if (( $(echo "$AVAIL_MEM < 40" | bc -l) )); then
    echo "❌ Insufficient memory (need at least 40GB free)" | tee -a "$LOG_FILE"
    echo "Skipping statistical test to prevent crash" | tee -a "$LOG_FILE"
else
    echo "Proceeding with statistical test..." | tee -a "$LOG_FILE"
    
    # Run with extreme nice level and memory limits
    ulimit -v 52428800  # 50GB virtual memory limit
    
    nice -n 19 python3 scripts/run_enhanced_diff_test.py \
        --ref-model "$MODEL1" \
        --cand-model "$MODEL2" \
        --mode quick \
        --max-queries 5 \
        --prf-key deadbeefcafebabe1234567890abcdef \
        --verbose 2>&1 | tee -a "$LOG_FILE"
    
    TEST_EXIT=${PIPESTATUS[0]}
    
    if [ $TEST_EXIT -eq 0 ]; then
        echo "✅ Statistical test completed" | tee -a "$LOG_FILE"
    else
        echo "⚠️ Statistical test failed or was killed (exit code: $TEST_EXIT)" | tee -a "$LOG_FILE"
    fi
fi

# Kill memory monitor
kill $MONITOR_PID 2>/dev/null

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "FINAL SUMMARY" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Generate final report
python3 << EOF 2>&1 | tee -a "$LOG_FILE"
import json
from pathlib import Path
from datetime import datetime

output_dir = Path("$OUTPUT_DIR")
report = {
    "timestamp": datetime.now().isoformat(),
    "models": {
        "reference": "Yi-34B (Base)",
        "candidate": "Yi-34B-Chat"
    },
    "model_size": "34B parameters each",
    "resource_limits": {
        "cpu_threads": 4,
        "memory_limit_gb": 50,
        "nice_priority": 19
    },
    "tests_completed": []
}

# Check what tests completed
if (output_dir / "config_comparison.json").exists():
    report["tests_completed"].append("configuration_analysis")
    with open(output_dir / "config_comparison.json") as f:
        report["config_comparison"] = json.load(f)

# Check memory log
mem_log = output_dir / "memory_monitor.log"
if mem_log.exists():
    with open(mem_log) as f:
        lines = f.readlines()
    if lines:
        report["memory_monitoring"] = {
            "samples": len(lines),
            "first_reading": lines[0].strip() if lines else "N/A",
            "last_reading": lines[-1].strip() if lines else "N/A"
        }

print("\nTEST SUMMARY:")
print(f"  Models: Yi-34B Base vs Chat (34B params each)")
print(f"  Timestamp: {report['timestamp']}")
print(f"  Tests completed: {', '.join(report['tests_completed'])}")

if "memory_monitoring" in report:
    print(f"\nMemory monitoring:")
    print(f"  Samples collected: {report['memory_monitoring']['samples']}")
    print(f"  Last reading: {report['memory_monitoring']['last_reading']}")

# Save final report
with open(output_dir / "final_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n✅ Full report saved to: {output_dir}/final_report.json")
EOF

# Final system state
echo "" | tee -a "$LOG_FILE"
echo "Final system state:" | tee -a "$LOG_FILE"
top -l 1 -n 0 | grep PhysMem | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "✅ Yi-34B testing completed!" | tee -a "$LOG_FILE"
echo "Results directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"