#!/bin/bash
# ONE-SHOT COMPLETE POT FRAMEWORK PIPELINE FOR QWEN 72B
# Expected runtime: 8-10 hours on M1 Max
# Run this script and let it complete overnight

echo "=========================================================================="
echo "COMPLETE POT FRAMEWORK PIPELINE - QWEN 72B"
echo "=========================================================================="
echo "This will run the FULL analytical pipeline including:"
echo "  1. Behavioral verification (5000 prompts)"
echo "  2. Statistical identity testing"
echo "  3. Challenge-response verification"
echo "  4. Fuzzy hash verification"
echo "  5. Performance benchmarking"
echo "  6. Comparison with baselines"
echo ""
echo "Expected runtime: 8-10 hours"
echo "=========================================================================="
echo ""

# Check if model exists
MODEL_PATH="/Users/rohanvinaik/LLM_Models/Qwen2.5-72B-Q4/Qwen2.5-72B-Instruct-Q4_K_M.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ ERROR: Model not found at $MODEL_PATH"
    echo "Please ensure the Qwen2.5-72B-Q4 model is downloaded"
    exit 1
fi

echo "✓ Model found: $(du -h "$MODEL_PATH" | cut -f1)"
echo ""

# Create results directory
mkdir -p experimental_results

# Get start time
START_TIME=$(date +%s)
START_DISPLAY=$(date "+%Y-%m-%d %H:%M:%S")

echo "Starting at: $START_DISPLAY"
echo "Process ID: $$"
echo ""
echo "You can monitor progress in another terminal with:"
echo "  tail -f experimental_results/qwen_pipeline_*.log"
echo ""
echo "=========================================================================="

# Run the complete pipeline
python3 scripts/run_complete_qwen_pipeline.py

# Get end time and calculate duration
END_TIME=$(date +%s)
END_DISPLAY=$(date "+%Y-%m-%d %H:%M:%S")
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "=========================================================================="
echo "PIPELINE COMPLETE!"
echo "=========================================================================="
echo "Started: $START_DISPLAY"
echo "Ended: $END_DISPLAY"
echo "Total duration: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved in experimental_results/"
echo "Look for files starting with: qwen_complete_pipeline_"
echo "=========================================================================="