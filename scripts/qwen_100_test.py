#!/usr/bin/env python3
"""
Quick 100-prompt test to verify setup works
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import Llama

print("="*70)
print("100-PROMPT VERIFICATION TEST - QWEN 72B")
print("="*70)
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print("This is a reduced test to verify the setup works")
print("="*70)

model_path = "/Users/rohanvinaik/LLM_Models/Qwen2.5-72B-Q4/Qwen2.5-72B-Instruct-Q4_K_M.gguf"
results_file = f"experimental_results/qwen_100_test_{int(time.time())}.json"

# Test prompts
prompts = [
    f"The number {i} is" for i in range(100)
]

print(f"\nTest: 100 prompts, 20 tokens each")
print("Loading model with minimal configuration...")

# Try minimal configuration
load_start = time.time()
try:
    model = Llama(
        model_path=model_path,
        n_ctx=256,  # Minimal context
        n_threads=4,  # Fewer threads
        n_gpu_layers=-1,  # Use Metal
        verbose=True,  # Show what's happening
        seed=42,
        n_batch=64,  # Smaller batch
        use_mmap=True,
        use_mlock=False
    )
    load_time = time.time() - load_start
    print(f"\n✓ Model loaded in {load_time:.1f}s")
except Exception as e:
    print(f"\n✗ Failed to load model: {e}")
    sys.exit(1)

# Test verification
print(f"\nRunning verification on 100 prompts...")
print("-" * 50)

test_start = time.time()
differences = []

for i, prompt in enumerate(prompts):
    if i % 10 == 0:
        elapsed = time.time() - test_start
        if i > 0:
            rate = i / elapsed
            eta = (100 - i) / rate
            print(f"Progress: {i}/100 ({elapsed:.1f}s elapsed, {eta:.1f}s remaining)")
    
    try:
        # Generate twice with same seed
        out1 = model(prompt, max_tokens=20, temperature=0.0, seed=42)
        out2 = model(prompt, max_tokens=20, temperature=0.0, seed=42)
        
        text1 = out1['choices'][0]['text']
        text2 = out2['choices'][0]['text']
        
        diff = 0.0 if text1 == text2 else 1.0
        differences.append(diff)
        
    except Exception as e:
        print(f"\nError at prompt {i}: {e}")
        break

test_time = time.time() - test_start
total_time = time.time() - load_start

# Results
n_completed = len(differences)
if n_completed > 0:
    mean_diff = np.mean(differences)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Completed: {n_completed}/100 prompts")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Decision: {'SAME' if mean_diff < 0.01 else 'DIFFERENT'}")
    print(f"\nTiming:")
    print(f"  Model load: {load_time:.1f}s")
    print(f"  Verification: {test_time:.1f}s")
    print(f"  Total: {total_time:.1f}s")
    print(f"  Per prompt: {test_time/n_completed:.2f}s")
    print(f"\nExtrapolated to 5000 prompts:")
    print(f"  Expected time: {(test_time/n_completed * 5000)/3600:.1f} hours")
    print("="*70)
    
    # Save results
    results = {
        'test': '100-prompt verification',
        'model': 'Qwen2.5-72B-Q4',
        'prompts_completed': n_completed,
        'mean_difference': float(mean_diff),
        'timing': {
            'load_time': load_time,
            'test_time': test_time,
            'total_time': total_time,
            'per_prompt': test_time/n_completed
        },
        'extrapolation': {
            'prompts_5000_hours': (test_time/n_completed * 5000)/3600
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
else:
    print("\n✗ No prompts completed successfully")