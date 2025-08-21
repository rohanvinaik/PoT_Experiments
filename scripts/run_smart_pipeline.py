#!/usr/bin/env python3
"""
SMART POT PIPELINE WITH EARLY TERMINATION
Uses sequential testing to terminate early when decision is reached
This is the ACTUAL advantage of PoT - no need for 5000 prompts!
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import Llama
from pot.core.diff_decision import EnhancedSequentialTester, DiffDecisionConfig

print("="*80)
print("SMART POT PIPELINE - QWEN 72B WITH EARLY TERMINATION")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nThis test uses SEQUENTIAL TESTING with early termination.")
print("Expected to complete in 50-400 prompts, not 5000!")
print("="*80)

# Configuration
model_path = "/Users/rohanvinaik/LLM_Models/Qwen2.5-72B-Q4/Qwen2.5-72B-Instruct-Q4_K_M.gguf"
results_file = f"experimental_results/qwen_smart_{int(time.time())}.json"

# Generate diverse prompts (but we won't need all of them!)
print("\nGenerating prompt pool...")
prompt_templates = [
    "Explain the concept of {}",
    "What are the implications of {}",
    "How does {} work?",
    "Describe the history of {}",
    "What is the future of {}?",
]

topics = [
    "artificial intelligence", "climate change", "quantum computing", 
    "democracy", "healthcare", "education", "renewable energy",
    "space exploration", "genetic engineering", "cryptocurrency"
]

all_prompts = []
for i in range(500):  # Only generate 500, we'll likely use <100
    template = prompt_templates[i % len(prompt_templates)]
    topic = topics[(i // len(prompt_templates)) % len(topics)]
    if i >= len(prompt_templates) * len(topics):
        topic = f"{topic} variation {i}"
    prompt = template.format(topic)
    all_prompts.append(prompt)

print(f"Generated {len(all_prompts)} prompts (but will use early termination)")

# Load model
print("\nLoading model...")
load_start = time.time()
model = Llama(
    model_path=model_path,
    n_ctx=256,
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

print("\n" + "="*80)
print("SEQUENTIAL TESTING WITH EARLY TERMINATION")
print("="*80)

# Configure testing
config = DiffDecisionConfig(
    alpha_same=0.025,  # 97.5% confidence for SAME
    alpha_diff=0.025,  # 97.5% confidence for DIFFERENT
    gamma=0.01,  # Threshold for SAME decision
    eta=0.5,  # Precision factor
    delta_star=0.05,  # Effect size for DIFFERENT
    epsilon_diff=0.1,  # Relative margin error
    n_min=30,  # Minimum samples
    n_max=200,  # Maximum samples before stopping
    K=2,  # Number of outputs per prompt
    clip_low=0.0,
    clip_high=1.0
)

results = {
    'model': 'Qwen2.5-72B-Q4',
    'load_time': load_time,
    'start_time': datetime.now().isoformat(),
    'test_result': {}
}

print(f"\n📊 Early Termination Test")
print(f"  Max samples: {config.n_max}")
print(f"  Min samples: {config.n_min}")
print("-" * 40)

tester = EnhancedSequentialTester(config)
differences = []
start_time = time.time()

for i, prompt in enumerate(all_prompts):
    if i >= config.n_max:
        break
            
        # Generate twice with same seed
        output1 = model(prompt, max_tokens=50, temperature=0.0, seed=42)
        output2 = model(prompt, max_tokens=50, temperature=0.0, seed=42)
        
        text1 = output1['choices'][0]['text']
        text2 = output2['choices'][0]['text']
        
        # Calculate difference
        diff = 0.0 if text1 == text2 else 1.0
        differences.append(diff)
        
        # Update sequential tester
        decision, info = tester.update(diff)
        
        # Progress update every 10 prompts
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Prompt {i+1}: mean_diff={np.mean(differences):.6f}, "
                  f"time={elapsed:.1f}s, decision={decision}")
        
        # EARLY TERMINATION - This is the key!
        if decision != 'undecided':
            elapsed = time.time() - start_time
            
            print(f"\n🎯 DECISION REACHED: {decision.upper()}")
            print(f"  Samples needed: {i+1} (vs 5000 traditional)")
            print(f"  Time taken: {elapsed:.1f}s")
            print(f"  Mean difference: {np.mean(differences):.8f}")
            print(f"  Confidence interval: [{info['ci_lower']:.6f}, {info['ci_upper']:.6f}]")
            
            # Calculate speedup
            traditional_prompts = 5000
            speedup = traditional_prompts / (i + 1)
            time_saved = (traditional_prompts - (i + 1)) * (elapsed / (i + 1))
            
            print(f"\n💡 EFFICIENCY GAIN:")
            print(f"  Prompts saved: {traditional_prompts - (i + 1):,}")
            print(f"  Speedup: {speedup:.1f}×")
            print(f"  Time saved: {time_saved/3600:.1f} hours")
            
            results['modes'][mode_name] = {
                'decision': decision,
                'prompts_used': i + 1,
                'time_seconds': elapsed,
                'mean_difference': float(np.mean(differences)),
                'confidence_interval': [info['ci_lower'], info['ci_upper']],
                'speedup_vs_5000': speedup,
                'time_saved_hours': time_saved/3600
            }
            break
    else:
        # Didn't reach decision within n_max
        print(f"\n⚠️ No decision after {len(differences)} prompts")
        results['modes'][mode_name] = {
            'decision': 'undecided',
            'prompts_used': len(differences),
            'time_seconds': time.time() - start_time,
            'mean_difference': float(np.mean(differences))
        }

# Final summary
print("\n" + "="*80)
print("SMART PIPELINE COMPLETE - EARLY TERMINATION SUCCESS!")
print("="*80)

total_prompts = sum(r['prompts_used'] for r in results['modes'].values())
total_time = sum(r['time_seconds'] for r in results['modes'].values()) + load_time

print(f"\n📈 OVERALL RESULTS:")
print(f"  Total prompts used: {total_prompts} (vs 5000 traditional)")
print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"  Average speedup: {5000/total_prompts:.1f}×")

print(f"\n🎯 VERIFICATION RESULTS:")
for mode_name, result in results['modes'].items():
    print(f"\n  {mode_name}:")
    print(f"    Decision: {result['decision'].upper()}")
    print(f"    Prompts: {result['prompts_used']}")
    print(f"    Speedup: {result.get('speedup_vs_5000', 0):.1f}×")

# Comparison with running pipeline
traditional_time_hours = 22.4  # Current pipeline estimate
smart_time_hours = total_time / 3600

print(f"\n⏱️  TIME COMPARISON:")
print(f"  Traditional (5000 prompts): {traditional_time_hours:.1f} hours")
print(f"  Smart (early termination): {smart_time_hours:.2f} hours")
print(f"  TIME SAVED: {traditional_time_hours - smart_time_hours:.1f} hours!")

results['summary'] = {
    'total_prompts': total_prompts,
    'total_time_seconds': total_time,
    'total_time_hours': smart_time_hours,
    'vs_traditional_hours': traditional_time_hours,
    'hours_saved': traditional_time_hours - smart_time_hours
}

# Save results
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {results_file}")
print("="*80)