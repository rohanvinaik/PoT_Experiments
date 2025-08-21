#!/usr/bin/env python3
"""
FULL POT FRAMEWORK PIPELINE - COMPARING TWO MIXTRAL MODELS
This tests:
1. Base Mixtral-8x22B vs Instruct Mixtral-8x22B
2. Should detect DIFFERENT (base vs instruction-tuned)
3. Uses early termination to avoid running 5000 prompts
"""

import sys
import time
import json
import numpy as np
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import Llama
from pot.core.diff_decision import EnhancedSequentialTester, DiffDecisionConfig, TestingMode
from pot.core.challenge import generate_challenges, ChallengeConfig
from pot.core.kdf_prompt_generator import KDFPromptGenerator
from pot.security.fuzzy_hash_verifier import FuzzyHashVerifier
from pot.security.provenance_auditor import ProvenanceAuditor
from pot.core.canonicalize import canonicalize_text
from pot.core.stats import empirical_bernstein_bound

print("="*80)
print("MIXTRAL MODEL COMPARISON - FULL POT FRAMEWORK")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nComparing:")
print("  Model 1: Mixtral-8x22B-Base (base model)")
print("  Model 2: Mixtral-8x22B-Instruct (instruction-tuned)")
print("\nExpected: DIFFERENT (base vs instruct should differ)")
print("="*80)

# Model paths - using the first file of each split model
model1_path = "/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Base-Q4/Mixtral-8x22B-v0.1-Q4_K_M-00001-of-00005.gguf"
model2_path = "/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Instruct-Q4/Mixtral-8x22B-Instruct-v0.1.Q4_K_M-00001-of-00002.gguf"

results_dir = Path("experimental_results")
results_dir.mkdir(exist_ok=True)
timestamp = int(time.time())
results_file = results_dir / f"mixtral_comparison_{timestamp}.json"

results = {
    'test': 'Mixtral Base vs Instruct Comparison',
    'model1': 'Mixtral-8x22B-Base',
    'model2': 'Mixtral-8x22B-Instruct',
    'timestamp': datetime.now().isoformat(),
    'phases': {},
    'metrics': {}
}

print("\n" + "="*80)
print("PHASE 1: LOADING MODELS")
print("="*80)

# Load Model 1 (Base)
print("\nLoading Model 1: Mixtral-8x22B-Base...")
load1_start = time.time()
try:
    model1 = Llama(
        model_path=model1_path,
        n_ctx=512,
        n_threads=4,
        n_gpu_layers=-1,  # Use all Metal layers
        verbose=False,
        seed=42,
        n_batch=128,
        use_mmap=True,
        use_mlock=False
    )
    load1_time = time.time() - load1_start
    print(f"✓ Model 1 loaded in {load1_time:.1f}s")
except Exception as e:
    print(f"✗ Failed to load Model 1: {e}")
    print("Note: Split GGUF models require all parts in same directory")
    sys.exit(1)

# Load Model 2 (Instruct)
print("\nLoading Model 2: Mixtral-8x22B-Instruct...")
load2_start = time.time()
try:
    model2 = Llama(
        model_path=model2_path,
        n_ctx=512,
        n_threads=4,
        n_gpu_layers=-1,
        verbose=False,
        seed=42,
        n_batch=128,
        use_mmap=True,
        use_mlock=False
    )
    load2_time = time.time() - load2_start
    print(f"✓ Model 2 loaded in {load2_time:.1f}s")
except Exception as e:
    print(f"✗ Failed to load Model 2: {e}")
    sys.exit(1)

total_load_time = load1_time + load2_time
results['phases']['model_loading'] = {
    'status': 'completed',
    'model1_time': load1_time,
    'model2_time': load2_time,
    'total_time': total_load_time
}

print("\n" + "="*80)
print("PHASE 2: KDF CHALLENGE GENERATION")
print("="*80)

kdf_start = time.time()
kdf_generator = KDFPromptGenerator(
    master_key="mixtral_comparison_2024",
    namespace="llm"
)

# Generate challenges
challenges = []
for i in range(100):
    challenges.append(kdf_generator.generate_prompt(i))

# Also generate framework challenges
challenge_config = ChallengeConfig(
    master_key_hex="6d69787472616c5f636f6d70617269736f6e",
    session_nonce_hex="3230323400000000000000000000000000000000000000",
    n=100,
    family="lm:templates",
    params={"template_type": "comparison"},
    model_id="mixtral"
)

framework_result = generate_challenges(challenge_config)
if 'challenges' in framework_result:
    for c in framework_result['challenges'][:50]:
        challenges.append(str(c.parameters.get('prompt', 'Test prompt')))

kdf_time = time.time() - kdf_start
print(f"✓ Generated {len(challenges)} challenges in {kdf_time:.1f}s")

results['phases']['challenge_generation'] = {
    'status': 'completed',
    'time_seconds': kdf_time,
    'num_challenges': len(challenges)
}

print("\n" + "="*80)
print("PHASE 3: SEQUENTIAL TESTING WITH EARLY TERMINATION")
print("="*80)

# Configure for detecting DIFFERENT models
config = DiffDecisionConfig(
    mode=TestingMode.AUDIT_GRADE,
    gamma=0.001,  # Threshold for SAME
    eta=0.5,
    delta_star=0.05,  # Threshold for DIFFERENT
    epsilon_diff=0.1,
    n_min=20,  # Lower minimum since we expect differences
    n_max=200  # Lower max since differences should be obvious
)

config.n_min = 20
config.n_max = 200

tester = EnhancedSequentialTester(config)
sequential_start = time.time()

print(f"Configuration:")
print(f"  Mode: {config.mode.name}")
print(f"  Min samples: {config.n_min}")
print(f"  Max samples: {config.n_max}")
print(f"  SAME threshold (γ): {config.gamma}")
print(f"  DIFFERENT threshold (δ*): {config.delta_star}")
print("-" * 40)

differences = []
model1_outputs = []
model2_outputs = []

for i, challenge in enumerate(challenges):
    if i >= config.n_max:
        break
    
    # Generate from both models
    prompt_start = time.time()
    
    # Model 1 output
    out1 = model1(challenge, max_tokens=30, temperature=0.0, seed=42)
    text1 = out1['choices'][0]['text']
    
    # Model 2 output
    out2 = model2(challenge, max_tokens=30, temperature=0.0, seed=42)
    text2 = out2['choices'][0]['text']
    
    prompt_time = time.time() - prompt_start
    
    # Compare outputs
    canon1 = canonicalize_text(text1)
    canon2 = canonicalize_text(text2)
    
    diff = 0.0 if canon1 == canon2 else 1.0
    differences.append(diff)
    model1_outputs.append(text1)
    model2_outputs.append(text2)
    
    # Update tester
    tester.update(diff)
    
    # Progress update
    if (i + 1) % 5 == 0:
        mean = tester.mean
        print(f"Sample {i+1}: diff={diff:.1f}, mean={mean:.4f}, time={prompt_time:.1f}s")
        if diff > 0 and (i + 1) <= 10:  # Show first few differences
            print(f"  Model1: {text1[:50]}...")
            print(f"  Model2: {text2[:50]}...")
    
    # Check for early termination
    if i >= config.n_min:
        mean = tester.mean
        (ci_lower, ci_upper), half_width = tester.ci()
        
        # DIFFERENT decision (more likely for base vs instruct)
        if mean >= config.delta_star:
            # Check relative margin of error
            if mean > 0:
                rme = half_width / mean
                if rme <= config.epsilon_diff:
                    sequential_time = time.time() - sequential_start
                    
                    print("\n" + "="*80)
                    print("🎯 EARLY TERMINATION - DECISION REACHED!")
                    print("="*80)
                    print(f"\nDecision: DIFFERENT (models are not identical)")
                    print(f"Samples used: {i+1} (vs 5000 traditional)")
                    print(f"Mean difference: {mean:.4f}")
                    print(f"Confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
                    print(f"Effect size: {mean:.4f} > {config.delta_star} (threshold)")
                    
                    # Efficiency gains
                    speedup = 5000 / (i + 1)
                    traditional_time = (sequential_time / (i + 1)) * 5000
                    time_saved = traditional_time - sequential_time
                    
                    print(f"\n💡 EFFICIENCY GAINS:")
                    print(f"  Speedup: {speedup:.1f}×")
                    print(f"  Time used: {sequential_time/60:.1f} minutes")
                    print(f"  Time saved: {time_saved/3600:.1f} hours")
                    
                    results['phases']['sequential_testing'] = {
                        'status': 'completed',
                        'decision': 'DIFFERENT',
                        'early_terminated': True,
                        'samples_used': i + 1,
                        'time_seconds': sequential_time,
                        'mean_difference': float(mean),
                        'confidence_interval': [float(ci_lower), float(ci_upper)],
                        'speedup': speedup
                    }
                    break
        
        # SAME decision (unlikely for these models)
        elif ci_upper <= config.gamma and (ci_upper - ci_lower) <= config.eta * config.gamma:
            sequential_time = time.time() - sequential_start
            print("\n🎯 EARLY TERMINATION - Models are SAME")
            results['phases']['sequential_testing'] = {
                'status': 'completed',
                'decision': 'SAME',
                'early_terminated': True,
                'samples_used': i + 1,
                'mean_difference': float(mean)
            }
            break
else:
    # No early termination
    sequential_time = time.time() - sequential_start
    mean = tester.mean if hasattr(tester, 'mean') else np.mean(differences)
    print(f"\n⚠️ No decision after {len(differences)} samples")
    print(f"Mean difference: {mean:.4f}")
    
    results['phases']['sequential_testing'] = {
        'status': 'completed',
        'decision': 'UNDECIDED',
        'early_terminated': False,
        'samples_used': len(differences),
        'mean_difference': float(mean)
    }

# Quick fuzzy hash of outputs
if len(model1_outputs) > 0:
    print("\n" + "="*80)
    print("PHASE 4: BEHAVIORAL FINGERPRINTING")
    print("="*80)
    
    behavior1 = "".join(model1_outputs[:20])
    behavior2 = "".join(model2_outputs[:20])
    
    hash1 = hashlib.sha256(behavior1.encode()).hexdigest()[:16]
    hash2 = hashlib.sha256(behavior2.encode()).hexdigest()[:16]
    
    print(f"Model 1 fingerprint: {hash1}")
    print(f"Model 2 fingerprint: {hash2}")
    print(f"Match: {'Yes' if hash1 == hash2 else 'No'}")
    
    results['phases']['fingerprinting'] = {
        'model1_hash': hash1,
        'model2_hash': hash2,
        'identical': hash1 == hash2
    }

# Final summary
print("\n" + "="*80)
print("MIXTRAL COMPARISON COMPLETE")
print("="*80)

total_time = time.time() - (load1_start)
results['summary'] = {
    'total_time_seconds': total_time,
    'total_time_minutes': total_time / 60,
    'decision': results['phases']['sequential_testing']['decision'],
    'samples_used': results['phases']['sequential_testing']['samples_used'],
    'mean_difference': results['phases']['sequential_testing']['mean_difference']
}

print(f"\n📊 FINAL RESULTS:")
print(f"  Decision: {results['summary']['decision']}")
print(f"  Mean difference: {results['summary']['mean_difference']:.4f}")
print(f"  Samples used: {results['summary']['samples_used']}")
print(f"  Total time: {results['summary']['total_time_minutes']:.1f} minutes")

if results['summary']['decision'] == 'DIFFERENT':
    print(f"\n✅ CORRECT! Base and Instruct models are indeed different")
elif results['summary']['decision'] == 'SAME':
    print(f"\n⚠️ UNEXPECTED! Base and Instruct should be different")

# Save results
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {results_file}")
print("="*80)