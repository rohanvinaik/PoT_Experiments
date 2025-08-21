#!/bin/bash

# Test script to demonstrate run_all.sh auto-throttling feature
# Shows how the script automatically adjusts resources based on model size

echo "============================================"
echo "Testing run_all.sh Auto-Throttling Feature"
echo "============================================"
echo ""

# Test 1: Small models (should use default resources)
echo "TEST 1: Small Models (GPT-2 vs DistilGPT-2)"
echo "--------------------------------------------"
bash scripts/run_all.sh \
    --model1 /Users/rohanvinaik/LLM_Models/gpt2 \
    --model2 /Users/rohanvinaik/LLM_Models/distilgpt2 \
    --skip-zk \
    --help 2>&1 | head -20

echo ""
echo "To run: bash scripts/run_all.sh --model1 gpt2 --model2 distilgpt2 --skip-zk"
echo ""

# Test 2: Medium models (should use moderate throttling)
echo "TEST 2: Medium Models (GPT-Neo 1.3B)"
echo "-------------------------------------"
echo "Would configure for 1B-7B models with:"
echo "  - 75% CPU threads"
echo "  - Nice level 5"
echo "  - No memory limits"
echo ""

# Test 3: Large models (should use significant throttling)
echo "TEST 3: Large Models (Llama-2-7B)"
echo "----------------------------------"
echo "Would configure for 7B-30B models with:"
echo "  - 50% CPU threads"
echo "  - Nice level 10"
echo "  - 30GB memory target"
echo ""

# Test 4: Extra large models (should use maximum throttling)
echo "TEST 4: Extra Large Models (Yi-34B)"
echo "------------------------------------"
echo "Command:"
echo "bash scripts/run_all.sh \\"
echo "    --model1 /Users/rohanvinaik/LLM_Models/yi-34b \\"
echo "    --model2 /Users/rohanvinaik/LLM_Models/yi-34b-chat \\"
echo "    --skip-zk"
echo ""
echo "Would configure for >30B models with:"
echo "  - 33% CPU threads"
echo "  - Nice level 15"
echo "  - 50GB memory target"
echo "  - CPU-only mode (no CUDA)"
echo "  - Auto-skip ZK tests"
echo ""

# Test 5: Disable throttling
echo "TEST 5: Disable Auto-Throttling"
echo "--------------------------------"
echo "Use --no-throttle flag to disable automatic resource management:"
echo "bash scripts/run_all.sh --model1 yi-34b --model2 yi-34b-chat --no-throttle"
echo ""

echo "============================================"
echo "Key Features:"
echo "============================================"
echo "1. Automatic model size detection from config.json"
echo "2. Dynamic resource allocation based on parameters"
echo "3. Memory availability checking"
echo "4. Graceful degradation for large models"
echo "5. Override with --no-throttle if needed"
echo ""
echo "The script now handles models from <1B to >30B parameters!"