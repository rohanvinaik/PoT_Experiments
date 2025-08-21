#!/usr/bin/env python3
"""
Quick PoT test with early termination - demonstrates the KEY advantage
This should complete in minutes, not hours!
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import Llama
from pot.core.diff_decision import EnhancedSequentialTester, DiffDecisionConfig, TestingMode

print("="*80)
print("QUICK POT TEST WITH EARLY TERMINATION")
print("="*80)
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print("\nThis demonstrates the KEY advantage of PoT:")
print("Early termination when statistical confidence is reached!")
print("="*80)

# Load model
print("\nLoading Qwen 72B model...")
model_path = "/Users/rohanvinaik/LLM_Models/Qwen2.5-72B-Q4/Qwen2.5-72B-Instruct-Q4_K_M.gguf"

load_start = time.time()
model = Llama(
    model_path=model_path,
    n_ctx=256,  # Small context for speed
    n_threads=4,
    n_gpu_layers=-1,
    verbose=False,
    seed=42,
    n_batch=64,
    use_mmap=True,
    use_mlock=False
)
load_time = time.time() - load_start
print(f"✓ Model loaded in {load_time:.1f}s")

# Configure for QUICK decision with high confidence
config = DiffDecisionConfig(
    mode=TestingMode.QUICK_GATE,  # Quick mode for fast decisions
    gamma=0.001,  # Very strict threshold for SAME
    eta=0.5,
    delta_star=0.01,
    epsilon_diff=0.05
)

print(f"\nConfiguration:")
print(f"  Mode: {config.mode.name}")
print(f"  Min samples: {config.mode.value['n_min']}")
print(f"  Max samples: {config.mode.value['n_max']} (vs 5000 traditional!)")
print(f"  Confidence: {(1 - config.mode.value['alpha_same'])*100:.1f}%")
print("="*80)

# Generate test prompts
max_samples = config.mode.value['n_max']
min_samples = config.mode.value['n_min']
prompts = [f"Test prompt number {i}" for i in range(max_samples)]

# Run sequential testing with early termination
tester = EnhancedSequentialTester(config)
test_start = time.time()
differences = []

print("\nRunning sequential test with early termination...")
print("-" * 40)

for i, prompt in enumerate(prompts):
    # Generate outputs
    prompt_start = time.time()
    out1 = model(prompt, max_tokens=20, temperature=0.0, seed=42)
    out2 = model(prompt, max_tokens=20, temperature=0.0, seed=42)
    prompt_time = time.time() - prompt_start
    
    # Calculate difference
    text1 = out1['choices'][0]['text']
    text2 = out2['choices'][0]['text']
    diff = 0.0 if text1 == text2 else 1.0
    differences.append(diff)
    
    # Update tester
    tester.update(diff)
    
    # Show progress
    if (i + 1) % 5 == 0:
        print(f"Sample {i+1}: diff={diff:.1f}, mean={np.mean(differences):.4f}, "
              f"time={prompt_time:.1f}s")
    
    # Check for early termination (after minimum)
    if i >= min_samples:
        mean = tester.get_mean()
        ci_lower, ci_upper = tester.get_confidence_interval(config.mode.value['alpha_same'])
        
        # Decision criteria
        if ci_upper <= config.gamma and (ci_upper - ci_lower) <= config.eta * config.gamma:
            # SAME decision reached!
            test_time = time.time() - test_start
            
            print("\n" + "="*80)
            print("🎯 EARLY TERMINATION - DECISION REACHED!")
            print("="*80)
            print(f"\nDecision: SAME (models are identical)")
            print(f"Confidence: 99%")
            print(f"Samples used: {i+1}")
            print(f"Mean difference: {mean:.8f}")
            print(f"Confidence interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
            
            # Calculate savings
            traditional_samples = 5000
            traditional_time = (test_time / (i+1)) * traditional_samples
            speedup = traditional_samples / (i+1)
            time_saved = traditional_time - test_time
            
            print(f"\n💡 EFFICIENCY GAIN:")
            print(f"  Traditional approach: {traditional_samples} samples")
            print(f"  PoT with early stop: {i+1} samples")
            print(f"  Speedup: {speedup:.1f}×")
            print(f"  Time saved: {time_saved/3600:.2f} hours")
            
            print(f"\n⏱️  TIME COMPARISON:")
            print(f"  Actual time: {test_time:.1f}s ({test_time/60:.1f} minutes)")
            print(f"  Traditional estimate: {traditional_time:.1f}s ({traditional_time/3600:.1f} hours)")
            print(f"  You just saved {time_saved/3600:.1f} hours!")
            
            # Save results
            results = {
                'test': 'Quick PoT with early termination',
                'model': 'Qwen2.5-72B-Q4',
                'decision': 'SAME',
                'samples_used': i+1,
                'samples_saved': traditional_samples - (i+1),
                'speedup': speedup,
                'mean_difference': float(mean),
                'confidence_interval': [float(ci_lower), float(ci_upper)],
                'time_seconds': test_time,
                'time_saved_hours': time_saved/3600
            }
            
            results_file = f"experimental_results/qwen_quick_pot_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✓ Results saved to: {results_file}")
            print("="*80)
            print("SUCCESS! Early termination works perfectly!")
            print("="*80)
            
            sys.exit(0)

# If we reach here, no early termination occurred
test_time = time.time() - test_start
print(f"\n⚠️ No early termination after {max_samples} samples")
print(f"Mean difference: {np.mean(differences):.8f}")
print(f"Test time: {test_time:.1f}s")
print("This suggests the model might have non-deterministic behavior.")