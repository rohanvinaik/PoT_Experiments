#!/bin/bash
# Safe testing script for large models with memory protection

set -e

echo "=========================================="
echo "🛡️ SAFE LARGE MODEL TESTING SCRIPT"
echo "=========================================="

# Configuration
MAX_MEMORY_GB=${MAX_MEMORY_GB:-40}
MODEL1=${1:-""}
MODEL2=${2:-""}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if models are provided
if [ -z "$MODEL1" ] || [ -z "$MODEL2" ]; then
    echo -e "${RED}Error: Please provide two model paths${NC}"
    echo "Usage: $0 <model1_path> <model2_path>"
    echo "Example: $0 /path/to/yi-34b /path/to/yi-34b-chat"
    exit 1
fi

# Check system memory
TOTAL_MEM_GB=$(python3 -c "import psutil; print(int(psutil.virtual_memory().total / (1024**3)))")
echo "System RAM: ${TOTAL_MEM_GB}GB"

if [ $TOTAL_MEM_GB -lt 32 ]; then
    echo -e "${RED}⚠️ WARNING: System has less than 32GB RAM${NC}"
    echo "Large model testing may fail. Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Aborted"
        exit 1
    fi
fi

# Start memory monitor in background
echo -e "\n${GREEN}Starting memory monitor...${NC}"
python3 scripts/monitor_memory.py --warning 75 --critical 85 --kill 92 > /tmp/memory_monitor.log 2>&1 &
MONITOR_PID=$!
echo "Monitor PID: $MONITOR_PID"

# Trap to ensure monitor is killed on exit
trap "echo 'Cleaning up...'; kill $MONITOR_PID 2>/dev/null || true" EXIT

# Function to check if we should continue
check_memory() {
    local current_mem=$(python3 -c "import psutil; print(psutil.virtual_memory().percent)")
    if (( $(echo "$current_mem > 85" | bc -l) )); then
        echo -e "${RED}Memory usage critical: ${current_mem}%${NC}"
        echo "Aborting to prevent system crash"
        return 1
    fi
    return 0
}

# Wait a moment for monitor to start
sleep 2

# Run the throttled test
echo -e "\n${GREEN}Starting throttled model test...${NC}"
echo "Models:"
echo "  1: $(basename "$MODEL1")"
echo "  2: $(basename "$MODEL2")"
echo "Memory limit: ${MAX_MEMORY_GB}GB"
echo "=========================================="

# Set resource limits
ulimit -v $((MAX_MEMORY_GB * 1024 * 1024))  # Virtual memory limit in KB

# Export throttling environment variables
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# Check memory before starting
if ! check_memory; then
    exit 1
fi

# Run with nice to lower priority
echo -e "\n${YELLOW}Running test (this may take 10-30 minutes)...${NC}"
nice -n 10 python3 scripts/run_large_models_throttled.py \
    --model1 "$MODEL1" \
    --model2 "$MODEL2" \
    --max-memory $MAX_MEMORY_GB \
    --min-free 8 \
    --enable-8bit \
    --enable-offload \
    --threads 8 \
    --mode quick \
    2>&1 | tee /tmp/large_model_test.log

TEST_EXIT_CODE=${PIPESTATUS[0]}

# Check final memory state
echo -e "\n${GREEN}Test completed. Checking final state...${NC}"
python3 -c "
import psutil
mem = psutil.virtual_memory()
print(f'Final memory usage: {mem.percent:.1f}%')
print(f'Available: {mem.available / (1024**3):.1f}GB')
"

# Kill monitor
kill $MONITOR_PID 2>/dev/null || true

# Show results
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}✅ Test completed successfully!${NC}"
    echo "Check experimental_results/ for detailed results"
    
    # Show latest result file
    LATEST_RESULT=$(ls -t experimental_results/throttled_test_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_RESULT" ]; then
        echo -e "\nLatest result: $LATEST_RESULT"
        echo "Summary:"
        python3 -c "
import json
with open('$LATEST_RESULT') as f:
    data = json.load(f)
    print(f\"  Decision: {data.get('decision', 'N/A')}\")
    print(f\"  Peak memory: {data.get('peak_memory_gb', 0):.1f}GB\")
    if 'statistics' in data:
        print(f\"  Effect size: {data['statistics']['effect_size']:.3f}\")
        print(f\"  Queries: {data['statistics']['n_queries']}\")
"
    fi
else
    echo -e "\n${RED}❌ Test failed or was interrupted${NC}"
    echo "Check /tmp/large_model_test.log for details"
fi

echo -e "\n=========================================="
echo "Memory monitor log: /tmp/memory_monitor.log"
echo "Test log: /tmp/large_model_test.log"
echo "=========================================="