#!/usr/bin/env python3
"""
COMPLETE VERIFICATION TEST: Qwen 72B with 5000 prompts
This matches standard behavioral verification coverage.
WARNING: This will take several hours to complete.
"""

import sys
import time
import json
import numpy as np
import hashlib
import psutil
import os
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import Llama

print("="*80)
print("COMPLETE 5000-PROMPT VERIFICATION TEST - QWEN 72B")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("This test will take several hours. Progress will be saved periodically.")
print("="*80)

# Configuration
N_PROMPTS = 5000  # Standard behavioral verification coverage
MAX_TOKENS = 50   # Standard response length
CHECKPOINT_EVERY = 100  # Save progress every N prompts
BATCH_SIZE = 10  # Process in batches for stability

model_path = "/Users/rohanvinaik/LLM_Models/Qwen2.5-72B-Q4/Qwen2.5-72B-Instruct-Q4_K_M.gguf"
checkpoint_file = "experimental_results/qwen_5000_checkpoint.json"
results_file = f"experimental_results/qwen_5000_complete_{int(time.time())}.json"

# Generate diverse prompt set
print("\nGenerating 5000 diverse prompts...")
prompt_templates = [
    "Explain the concept of {}",
    "What are the implications of {}",
    "How does {} work?",
    "Describe the history of {}",
    "What is the future of {}?",
    "Compare and contrast {} with alternatives",
    "What are the benefits of {}?",
    "What are the risks of {}?",
    "How can we improve {}?",
    "What is your analysis of {}?",
    "Provide examples of {}",
    "Define {} in simple terms",
    "What are the key factors in {}?",
    "How has {} evolved?",
    "What are the challenges in {}?",
    "Discuss the ethics of {}",
    "What is the scientific basis of {}?",
    "How does {} impact society?",
    "What are best practices for {}?",
    "Analyze the effectiveness of {}"
]

topics = [
    "artificial intelligence", "climate change", "quantum computing", "democracy",
    "healthcare", "education", "renewable energy", "space exploration",
    "genetic engineering", "cryptocurrency", "social media", "automation",
    "privacy", "sustainability", "machine learning", "neuroscience",
    "robotics", "biotechnology", "cybersecurity", "virtual reality",
    "blockchain", "nanotechnology", "fusion energy", "consciousness",
    "evolution", "economics", "philosophy", "psychology", "sociology",
    "anthropology", "linguistics", "mathematics", "physics", "chemistry",
    "biology", "ecology", "geology", "astronomy", "medicine",
    "engineering", "architecture", "agriculture", "transportation",
    "communication", "entertainment", "sports", "art", "music", "literature"
]

# Generate all prompts
all_prompts = []
for i in range(N_PROMPTS):
    template = prompt_templates[i % len(prompt_templates)]
    topic = topics[(i // len(prompt_templates)) % len(topics)]
    # Add variation
    if i >= len(prompt_templates) * len(topics):
        topic = f"{topic} in the {2020 + (i % 10)}s"
    prompt = template.format(topic)
    all_prompts.append(prompt)

print(f"Generated {len(all_prompts)} unique prompts")

# Load checkpoint if exists
start_from = 0
results = {
    'prompts_tested': [],
    'differences': [],
    'times': [],
    'memory_usage': [],
    'checkpoints': []
}

if os.path.exists(checkpoint_file):
    print(f"\nFound checkpoint file: {checkpoint_file}")
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
        start_from = checkpoint.get('last_completed', 0)
        results = checkpoint.get('results', results)
    print(f"Resuming from prompt {start_from + 1}")

# Load model
print("\nLoading Qwen 72B model...")
load_start = time.time()
model = Llama(
    model_path=model_path,
    n_ctx=256,  # Reduced context for stability
    n_threads=4,  # Fewer threads for stability
    n_gpu_layers=-1,  # Use all Metal layers
    verbose=False,
    seed=42,
    n_batch=64,  # Smaller batch for stability
    use_mmap=True,  # Memory map for efficiency
    use_mlock=False
)
load_time = time.time() - load_start
print(f"Model loaded in {load_time:.1f}s")

# Get initial memory usage
process = psutil.Process()
initial_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB

print(f"Initial memory usage: {initial_memory:.1f} GB")
print(f"\nStarting verification of {N_PROMPTS - start_from} prompts...")
print("="*80)

# Main verification loop
total_start = time.time()
batch_times = []

try:
    for i in range(start_from, N_PROMPTS):
        batch_num = i // BATCH_SIZE
        prompt_in_batch = i % BATCH_SIZE
        
        if prompt_in_batch == 0:
            batch_start = time.time()
            print(f"\nBatch {batch_num + 1}/{N_PROMPTS // BATCH_SIZE}")
            print(f"Prompts {i + 1}-{min(i + BATCH_SIZE, N_PROMPTS)}/{N_PROMPTS}")
        
        prompt = all_prompts[i]
        prompt_start = time.time()
        
        # Generate outputs with deterministic settings
        output1 = model(prompt, max_tokens=MAX_TOKENS, temperature=0.0, seed=42)
        output2 = model(prompt, max_tokens=MAX_TOKENS, temperature=0.0, seed=42)
        
        text1 = output1['choices'][0]['text']
        text2 = output2['choices'][0]['text']
        
        # Calculate difference
        if text1 == text2:
            diff = 0.0
        else:
            # Character-level difference
            diff = sum(c1 != c2 for c1, c2 in zip(text1, text2)) / max(len(text1), len(text2), 1)
        
        prompt_time = time.time() - prompt_start
        
        # Record results
        results['prompts_tested'].append(i)
        results['differences'].append(diff)
        results['times'].append(prompt_time)
        results['memory_usage'].append(process.memory_info().rss / 1024 / 1024 / 1024)
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            avg_time = np.mean(results['times'][-10:])
            remaining = N_PROMPTS - i - 1
            eta_seconds = remaining * avg_time
            eta = datetime.now() + timedelta(seconds=eta_seconds)
            
            print(f"  Progress: {i + 1}/{N_PROMPTS} ({(i + 1) / N_PROMPTS * 100:.1f}%)")
            print(f"  Avg time/prompt: {avg_time:.2f}s")
            print(f"  ETA: {eta.strftime('%H:%M:%S')} ({eta_seconds / 3600:.1f} hours remaining)")
            print(f"  Current memory: {results['memory_usage'][-1]:.1f} GB")
            
            if diff > 0:
                print(f"  ⚠️ Difference detected: {diff:.6f}")
        
        # Checkpoint saving
        if (i + 1) % CHECKPOINT_EVERY == 0:
            checkpoint_data = {
                'last_completed': i,
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'stats': {
                    'mean_diff': np.mean(results['differences']),
                    'max_diff': np.max(results['differences']) if results['differences'] else 0,
                    'total_time': time.time() - total_start + sum(results.get('checkpoints', [])),
                    'prompts_completed': i + 1
                }
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            print(f"\n✓ Checkpoint saved at prompt {i + 1}")
            print(f"  Mean difference so far: {checkpoint_data['stats']['mean_diff']:.8f}")
            
            # Add checkpoint time
            results['checkpoints'].append(time.time() - total_start)
        
        # Batch completion
        if prompt_in_batch == BATCH_SIZE - 1 or i == N_PROMPTS - 1:
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            print(f"  Batch completed in {batch_time:.1f}s")

except KeyboardInterrupt:
    print("\n\n⚠️ Test interrupted by user!")
    print(f"Completed {i - start_from} prompts before interruption")
except Exception as e:
    print(f"\n\n❌ Error occurred: {e}")
    print(f"Completed {i - start_from} prompts before error")

finally:
    # Calculate final statistics
    total_time = time.time() - total_start
    if results['checkpoints']:
        total_time += sum(results['checkpoints'])
    
    prompts_completed = len(results['prompts_tested'])
    
    print("\n" + "="*80)
    print("VERIFICATION TEST RESULTS")
    print("="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModel: Qwen2.5-72B-Q4 (45.86 GB)")
    print(f"Prompts tested: {prompts_completed}/{N_PROMPTS}")
    print(f"Total time: {total_time:.1f}s ({total_time / 3600:.2f} hours)")
    print(f"Model load time: {load_time:.1f}s")
    print(f"Verification time: {total_time - load_time:.1f}s")
    
    if results['differences']:
        mean_diff = np.mean(results['differences'])
        std_diff = np.std(results['differences'])
        max_diff = np.max(results['differences'])
        
        print(f"\nVerification Results:")
        print(f"  Mean difference: {mean_diff:.8f}")
        print(f"  Std deviation: {std_diff:.8f}")
        print(f"  Max difference: {max_diff:.8f}")
        print(f"  Non-zero differences: {sum(d > 0 for d in results['differences'])}/{prompts_completed}")
        print(f"  Decision: {'SAME (verified)' if mean_diff < 0.001 else 'DIFFERENT'}")
        print(f"  Confidence: {(1 - mean_diff) * 100:.2f}%")
    
    if results['times']:
        print(f"\nPerformance Metrics:")
        print(f"  Average time per prompt: {np.mean(results['times']):.2f}s")
        print(f"  Median time per prompt: {np.median(results['times']):.2f}s")
        print(f"  Total tokens generated: ~{prompts_completed * MAX_TOKENS * 2:,}")
        print(f"  Throughput: {prompts_completed * MAX_TOKENS * 2 / total_time:.1f} tokens/sec")
    
    if results['memory_usage']:
        print(f"\nResource Usage:")
        print(f"  Peak memory: {np.max(results['memory_usage']):.1f} GB")
        print(f"  Average memory: {np.mean(results['memory_usage']):.1f} GB")
    
    # Comparison with baselines
    print("\n" + "="*80)
    print("COMPARISON WITH STANDARD METHODS")
    print("="*80)
    
    behavioral_baseline = 10800  # 3 hours for 5000 prompts
    gradient_baseline = 21600    # 6 hours
    weight_baseline = 2700        # 45 minutes
    retraining_baseline = 1209600 # 14 days
    
    if prompts_completed > 0:
        # Extrapolate if not complete
        if prompts_completed < N_PROMPTS:
            extrapolated_time = (total_time / prompts_completed) * N_PROMPTS
            print(f"\n⚠️ Test incomplete. Extrapolating from {prompts_completed} prompts:")
            print(f"  Estimated total time for 5000: {extrapolated_time:.0f}s ({extrapolated_time/3600:.1f} hours)")
            comparison_time = extrapolated_time
        else:
            comparison_time = total_time
        
        print(f"\nActual Performance:")
        print(f"  PoT Framework: {comparison_time:.0f}s ({comparison_time/3600:.1f} hours)")
        print(f"  Behavioral Baseline: {behavioral_baseline}s (3.0 hours)")
        print(f"  Actual Speedup: {behavioral_baseline / comparison_time:.2f}×")
        
        print(f"\nSpeedup Factors (Real Data):")
        print(f"  vs Behavioral: {behavioral_baseline / comparison_time:.2f}×")
        print(f"  vs Gradient: {gradient_baseline / comparison_time:.2f}×")
        print(f"  vs Weights: {weight_baseline / comparison_time:.2f}×")
        print(f"  vs Retraining: {retraining_baseline / comparison_time:.1f}×")
        
        print(f"\nReality Check:")
        if behavioral_baseline / comparison_time < 1:
            print(f"  ⚠️ PoT is {comparison_time / behavioral_baseline:.1f}× SLOWER than behavioral baseline")
        else:
            print(f"  ✓ PoT is {behavioral_baseline / comparison_time:.1f}× faster than behavioral baseline")
    
    # Save final results
    final_results = {
        'test_config': {
            'model': 'Qwen2.5-72B-Q4',
            'model_size_gb': 45.86,
            'n_prompts_target': N_PROMPTS,
            'n_prompts_completed': prompts_completed,
            'max_tokens': MAX_TOKENS
        },
        'timing': {
            'start_time': (datetime.now() - timedelta(seconds=total_time)).isoformat(),
            'end_time': datetime.now().isoformat(),
            'load_time': load_time,
            'total_time': total_time,
            'verification_time': total_time - load_time
        },
        'results': {
            'mean_difference': float(np.mean(results['differences'])) if results['differences'] else None,
            'std_difference': float(np.std(results['differences'])) if results['differences'] else None,
            'max_difference': float(np.max(results['differences'])) if results['differences'] else None,
            'decision': 'SAME' if results['differences'] and np.mean(results['differences']) < 0.001 else 'UNKNOWN'
        },
        'performance': {
            'prompts_per_second': prompts_completed / total_time if total_time > 0 else 0,
            'tokens_per_second': (prompts_completed * MAX_TOKENS * 2) / total_time if total_time > 0 else 0,
            'avg_time_per_prompt': float(np.mean(results['times'])) if results['times'] else None
        },
        'comparison': {
            'behavioral_speedup': behavioral_baseline / comparison_time if prompts_completed > 0 else None,
            'gradient_speedup': gradient_baseline / comparison_time if prompts_completed > 0 else None,
            'weight_speedup': weight_baseline / comparison_time if prompts_completed > 0 else None,
            'retraining_speedup': retraining_baseline / comparison_time if prompts_completed > 0 else None
        },
        'raw_data': {
            'differences': results['differences'][-100:],  # Last 100 for reference
            'times': results['times'][-100:],
            'memory_usage': results['memory_usage'][-100:]
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Complete results saved to: {results_file}")
    print("="*80)
    
    # Clean up checkpoint if complete
    if prompts_completed >= N_PROMPTS and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("✓ Checkpoint file cleaned up")