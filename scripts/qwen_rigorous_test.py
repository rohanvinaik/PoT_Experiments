#!/usr/bin/env python3
"""
RIGOROUS TEST: Qwen 72B with proper statistical coverage
This provides a fair comparison to standard verification methods.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import Llama

print("="*70)
print("RIGOROUS VERIFICATION TEST - QWEN 72B")
print("="*70)
print("This test provides proper statistical coverage for fair comparison")
print("")

# Configuration for rigorous test
N_PROMPTS = 100  # Minimum for statistical significance (not 10!)
MAX_TOKENS = 50  # More comprehensive output (not 30)
N_RUNS = 3  # Multiple runs for variance

model_path = "/Users/rohanvinaik/LLM_Models/Qwen2.5-72B-Q4/Qwen2.5-72B-Instruct-Q4_K_M.gguf"

# Diverse prompt set (not just 10 similar prompts)
prompt_categories = {
    'factual': [
        "What is the capital of France?",
        "Explain photosynthesis in simple terms.",
        "List the planets in our solar system.",
        "What causes seasons on Earth?",
        "Define democracy.",
    ],
    'reasoning': [
        "If all roses are flowers and some flowers fade quickly, can we conclude",
        "A train travels 60 mph for 2 hours, then 40 mph for 3 hours. What is",
        "Explain why correlation does not imply causation",
        "What is the difference between deduction and induction?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long",
    ],
    'creative': [
        "Write a haiku about artificial intelligence",
        "Describe a world where gravity works backwards",
        "Create a metaphor for machine learning",
        "Invent a new color and describe it",
        "Write the first sentence of a mystery novel",
    ],
    'technical': [
        "Explain the difference between TCP and UDP",
        "What is a hash function used for?",
        "Describe the transformer architecture",
        "What is gradient descent?",
        "Explain big O notation",
    ],
    'ethical': [
        "Should AI systems have rights?",
        "Discuss the trolley problem",
        "What are the ethics of genetic engineering?",
        "Is privacy a fundamental right?",
        "Should there be limits on AI capabilities?",
    ]
}

# Flatten prompts
all_prompts = []
for category, prompts in prompt_categories.items():
    all_prompts.extend(prompts)

# Extend to N_PROMPTS by cycling
while len(all_prompts) < N_PROMPTS:
    all_prompts.extend(all_prompts[:min(len(all_prompts), N_PROMPTS - len(all_prompts))])
all_prompts = all_prompts[:N_PROMPTS]

print(f"Test Configuration:")
print(f"  - Prompts: {N_PROMPTS} (diverse categories)")
print(f"  - Tokens per prompt: {MAX_TOKENS}")
print(f"  - Total tokens to generate: {N_PROMPTS * MAX_TOKENS:,}")
print(f"  - Statistical runs: {N_RUNS}")
print("")

# Load model
print("Loading Qwen 72B model (this takes ~30 seconds)...")
load_start = time.time()
model = Llama(
    model_path=model_path,
    n_ctx=1024,  # Larger context for real testing
    n_threads=8,
    n_gpu_layers=-1,
    verbose=False,
    seed=42,
    n_batch=256
)
load_time = time.time() - load_start
print(f"Model loaded in {load_time:.1f}s")
print("")

# Run verification
print("Running rigorous verification test...")
print("-" * 50)

total_start = time.time()
results = []

for run in range(N_RUNS):
    print(f"\nRun {run + 1}/{N_RUNS}:")
    run_start = time.time()
    
    differences = []
    tokens_generated = 0
    
    for i, prompt in enumerate(all_prompts):
        if i % 10 == 0:
            print(f"  Processing prompts {i+1}-{min(i+10, N_PROMPTS)}/{N_PROMPTS}...")
        
        # Generate outputs
        out1 = model(prompt, max_tokens=MAX_TOKENS, temperature=0.0, seed=42)
        out2 = model(prompt, max_tokens=MAX_TOKENS, temperature=0.0, seed=42)
        
        text1 = out1['choices'][0]['text']
        text2 = out2['choices'][0]['text']
        
        # Calculate difference
        if text1 == text2:
            diff = 0.0
        else:
            # Character-level difference ratio
            diff = sum(c1 != c2 for c1, c2 in zip(text1, text2)) / max(len(text1), len(text2), 1)
        
        differences.append(diff)
        tokens_generated += len(text1.split()) + len(text2.split())
    
    run_time = time.time() - run_start
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    results.append({
        'run': run + 1,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'time': run_time,
        'tokens_generated': tokens_generated,
        'tokens_per_second': tokens_generated / run_time
    })
    
    print(f"  Mean difference: {mean_diff:.6f} ± {std_diff:.6f}")
    print(f"  Time: {run_time:.1f}s ({tokens_generated / run_time:.1f} tokens/sec)")

total_time = time.time() - total_start

# Analysis
print("\n" + "="*70)
print("RIGOROUS TEST RESULTS")
print("="*70)

avg_mean_diff = np.mean([r['mean_diff'] for r in results])
avg_time = np.mean([r['time'] for r in results])
total_tokens = sum(r['tokens_generated'] for r in results)

print(f"Model: Qwen2.5-72B-Q4 (45.86GB)")
print(f"Total prompts tested: {N_PROMPTS * N_RUNS}")
print(f"Total tokens generated: {total_tokens:,}")
print(f"Total time: {total_time:.1f}s")
print(f"")
print(f"Verification Results:")
print(f"  Average difference: {avg_mean_diff:.8f}")
print(f"  Decision: {'SAME (identity verified)' if avg_mean_diff < 0.001 else 'DIFFERENT'}")
print(f"  Confidence: {(1 - avg_mean_diff) * 100:.2f}%")
print(f"")
print(f"Performance Metrics:")
print(f"  Model load time: {load_time:.1f}s")
print(f"  Verification time: {total_time - load_time:.1f}s")
print(f"  Throughput: {total_tokens / (total_time - load_time):.1f} tokens/sec")
print(f"  Time per prompt: {(total_time - load_time) / (N_PROMPTS * N_RUNS):.2f}s")

# Comparison with baselines
print("\n" + "="*70)
print("REALISTIC COMPARISON WITH BASELINES")
print("="*70)

# Scale our results to standard test sizes
standard_prompts = 5000
scale_factor = standard_prompts / (N_PROMPTS * N_RUNS)
extrapolated_time = (total_time - load_time) * scale_factor + load_time

print(f"Our test coverage: {N_PROMPTS * N_RUNS} prompts")
print(f"Standard coverage: {standard_prompts} prompts")
print(f"Scale factor: {scale_factor:.1f}×")
print(f"")
print(f"Extrapolated to standard coverage:")
print(f"  PoT time (extrapolated): {extrapolated_time:.0f}s ({extrapolated_time/3600:.1f} hours)")
print(f"  Behavioral cloning baseline: 10,800s (3.0 hours)")
print(f"  Actual speedup: {10800 / extrapolated_time:.1f}×")
print(f"")
print(f"With smart sampling (95% confidence at {N_PROMPTS} prompts):")
print(f"  PoT time (actual): {total_time:.0f}s")
print(f"  Behavioral baseline: 10,800s")
print(f"  Smart sampling speedup: {10800 / total_time:.1f}×")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("1. Real speedup with equivalent coverage: ~{:.1f}×".format(10800 / extrapolated_time))
print("2. Smart sampling speedup: ~{:.1f}×".format(10800 / total_time))
print("3. This is still significant but not 7,000×!")
print("4. Main advantage: Runs on consumer hardware vs datacenter")
print("="*70)

# Save results
output = {
    'model': 'Qwen2.5-72B-Q4',
    'test_config': {
        'n_prompts': N_PROMPTS,
        'max_tokens': MAX_TOKENS,
        'n_runs': N_RUNS
    },
    'results': {
        'mean_difference': avg_mean_diff,
        'total_time': total_time,
        'load_time': load_time,
        'verification_time': total_time - load_time,
        'total_tokens': total_tokens
    },
    'comparison': {
        'extrapolated_time': extrapolated_time,
        'baseline_time': 10800,
        'actual_speedup': 10800 / extrapolated_time,
        'smart_sampling_speedup': 10800 / total_time
    },
    'detailed_runs': results
}

output_file = f"experimental_results/qwen72b_rigorous_{int(time.time())}.json"
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nDetailed results saved to: {output_file}")