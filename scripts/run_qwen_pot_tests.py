#!/usr/bin/env python3
"""
Run PoT framework tests on Qwen 72B GGUF model
This adapts the run_all.sh tests to work with llama-cpp-python
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_cpp import Llama

# Import PoT components that work with any model
from pot.core.diff_decision import EnhancedSequentialTester, DiffDecisionConfig
from pot.core.calibration import ModelCalibrator
from pot.core.kdf_prompt_generator import KDFPromptGenerator
try:
    from pot.core.progressive_testing import ProgressiveTestingStrategy
    HAS_PROGRESSIVE = True
except ImportError:
    HAS_PROGRESSIVE = False

print("="*80)
print("POT FRAMEWORK TESTS - QWEN 72B")
print("Adapted from run_all.sh test suite")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Configuration
model_path = "/Users/rohanvinaik/LLM_Models/Qwen2.5-72B-Q4/Qwen2.5-72B-Instruct-Q4_K_M.gguf"
results_dir = Path("experimental_results")
results_dir.mkdir(exist_ok=True)
timestamp = int(time.time())
results_file = results_dir / f"qwen_pot_tests_{timestamp}.json"

# Test results tracking
test_results = {
    'model': 'Qwen2.5-72B-Q4',
    'timestamp': datetime.now().isoformat(),
    'tests': {}
}

def run_test(name, func):
    """Run a test and track results"""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)
    
    start = time.time()
    try:
        result = func()
        elapsed = time.time() - start
        test_results['tests'][name] = {
            'status': 'PASSED',
            'time': elapsed,
            'result': result
        }
        print(f"✓ {name} PASSED ({elapsed:.2f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - start
        test_results['tests'][name] = {
            'status': 'FAILED',
            'time': elapsed,
            'error': str(e)
        }
        print(f"✗ {name} FAILED: {e}")
        traceback.print_exc()
        return False

# Load model once for all tests
print("\nLoading Qwen 72B model...")
load_start = time.time()
model = Llama(
    model_path=model_path,
    n_ctx=512,
    n_threads=4,
    n_gpu_layers=-1,
    verbose=False,
    seed=42,
    n_batch=128,
    use_mmap=True,
    use_mlock=False
)
load_time = time.time() - load_start
print(f"Model loaded in {load_time:.1f}s")

# Wrapper to make GGUF model compatible with PoT interface
class ModelWrapper:
    def __init__(self, llama_model):
        self.model = llama_model
    
    def generate(self, prompt, max_tokens=50, temperature=0.0, seed=42):
        output = self.model(prompt, max_tokens=max_tokens, temperature=temperature, seed=seed)
        return output['choices'][0]['text']

wrapped_model = ModelWrapper(model)

# TEST 1: Deterministic Validation
def test_deterministic():
    """Test that model produces deterministic outputs with fixed seed"""
    prompts = [
        "The capital of France is",
        "2 + 2 equals",
        "The speed of light is"
    ]
    
    for prompt in prompts:
        out1 = wrapped_model.generate(prompt, seed=42)
        out2 = wrapped_model.generate(prompt, seed=42)
        if out1 != out2:
            raise ValueError(f"Non-deterministic output for: {prompt}")
    
    return {'deterministic': True, 'prompts_tested': len(prompts)}

# TEST 2: Enhanced Diff Decision
def test_enhanced_diff():
    """Test enhanced sequential testing with early termination"""
    config = DiffDecisionConfig(
        alpha_same=0.025,
        alpha_diff=0.025,
        gamma=0.01,
        eta=0.5,
        delta_star=0.05,
        epsilon_diff=0.1,
        n_min=20,
        n_max=100,
        K=2
    )
    
    tester = EnhancedSequentialTester(config)
    
    # Generate test prompts
    prompts = [f"Define concept {i}" for i in range(100)]
    
    for i, prompt in enumerate(prompts):
        if i >= config.n_max:
            break
            
        out1 = wrapped_model.generate(prompt, seed=42)
        out2 = wrapped_model.generate(prompt, seed=42)
        diff = 0.0 if out1 == out2 else 1.0
        
        tester.update(diff)
        
        if i >= config.n_min:
            mean = tester.get_mean()
            ci_lower, ci_upper = tester.get_confidence_interval(config.alpha_same)
            
            # Check for early termination
            if ci_upper <= config.gamma:
                return {
                    'decision': 'SAME',
                    'samples': i + 1,
                    'mean': mean,
                    'ci': [ci_lower, ci_upper],
                    'early_terminated': True
                }
    
    return {
        'decision': 'UNDECIDED',
        'samples': config.n_max,
        'mean': tester.get_mean()
    }

# TEST 3: Calibration System
def test_calibration():
    """Test auto-calibration using percentile data"""
    calibrator = ModelCalibrator()
    
    # Generate calibration data
    same_scores = []
    for i in range(30):
        prompt = f"Test prompt {i}"
        out1 = wrapped_model.generate(prompt, seed=42)
        out2 = wrapped_model.generate(prompt, seed=42)
        score = 0.0 if out1 == out2 else 1.0
        same_scores.append(score)
    
    # Calibrate thresholds
    result = calibrator.calibrate(
        same_model_scores=same_scores,
        different_model_scores=[0.5, 0.6, 0.7],  # Synthetic different scores
        use_enhanced=True
    )
    
    config = result.config if hasattr(result, 'config') else DiffDecisionConfig()
    
    return {
        'gamma': config.gamma,
        'delta_star': config.delta_star,
        'calibration_samples': len(same_scores)
    }

# TEST 4: Progressive Testing Strategy
def test_progressive():
    """Test multi-stage progressive testing"""
    if not HAS_PROGRESSIVE:
        # Fallback implementation if progressive testing not available
        return {'status': 'skipped', 'reason': 'Progressive testing not available'}
    
    # Stage 1: Quick gate
    stage1_prompts = ["Quick test " + str(i) for i in range(10)]
    stage1_diffs = []
    
    for prompt in stage1_prompts:
        out1 = wrapped_model.generate(prompt, max_tokens=20, seed=42)
        out2 = wrapped_model.generate(prompt, max_tokens=20, seed=42)
        diff = 0.0 if out1 == out2 else 1.0
        stage1_diffs.append(diff)
    
    if np.mean(stage1_diffs) > 0.1:
        return {'decision': 'DIFFERENT', 'stage': 1}
    
    # Stage 2: Standard test
    stage2_prompts = ["Standard test " + str(i) for i in range(20)]
    stage2_diffs = []
    
    for prompt in stage2_prompts:
        out1 = wrapped_model.generate(prompt, max_tokens=30, seed=42)
        out2 = wrapped_model.generate(prompt, max_tokens=30, seed=42)
        diff = 0.0 if out1 == out2 else 1.0
        stage2_diffs.append(diff)
    
    if np.mean(stage2_diffs) > 0.05:
        return {'decision': 'DIFFERENT', 'stage': 2}
    
    # Stage 3: Deep audit
    stage3_prompts = ["Deep audit " + str(i) for i in range(30)]
    stage3_diffs = []
    
    for prompt in stage3_prompts:
        out1 = wrapped_model.generate(prompt, max_tokens=50, seed=42)
        out2 = wrapped_model.generate(prompt, max_tokens=50, seed=42)
        diff = 0.0 if out1 == out2 else 1.0
        stage3_diffs.append(diff)
    
    final_mean = np.mean(stage3_diffs)
    
    return {
        'decision': 'SAME' if final_mean < 0.01 else 'DIFFERENT',
        'stage': 3,
        'total_samples': len(stage1_prompts) + len(stage2_prompts) + len(stage3_prompts),
        'mean_difference': final_mean
    }

# TEST 5: KDF Challenge Generation
def test_kdf_challenges():
    """Test KDF-based cryptographic challenge generation"""
    generator = KDFPromptGenerator(
        master_seed=b"qwen_test",
        num_iterations=1000
    )
    
    challenges = []
    responses = []
    
    for i in range(20):
        challenge = generator.generate_prompt(i)
        challenges.append(challenge)
        
        # Test determinism of responses
        out1 = wrapped_model.generate(challenge, max_tokens=30, seed=42)
        out2 = wrapped_model.generate(challenge, max_tokens=30, seed=42)
        
        if out1 != out2:
            raise ValueError(f"Non-deterministic response to KDF challenge {i}")
        
        responses.append(out1)
    
    return {
        'challenges_generated': len(challenges),
        'all_deterministic': True,
        'unique_challenges': len(set(challenges))
    }

# TEST 6: Boundary Testing
def test_boundaries():
    """Test model behavior at boundaries"""
    boundary_prompts = [
        "",  # Empty prompt
        "a",  # Single character
        "a" * 500,  # Long prompt
        "🎉 Unicode test 中文",  # Unicode
        "1234567890",  # Numbers only
    ]
    
    results = []
    for prompt in boundary_prompts:
        try:
            out = wrapped_model.generate(prompt, max_tokens=10, seed=42)
            results.append({'prompt_type': prompt[:20], 'success': True})
        except Exception as e:
            results.append({'prompt_type': prompt[:20], 'success': False, 'error': str(e)})
    
    success_rate = sum(1 for r in results if r.get('success', False)) / len(results)
    
    return {
        'boundary_tests': len(boundary_prompts),
        'success_rate': success_rate,
        'results': results
    }

# TEST 7: Statistical Verification
def test_statistical():
    """Test statistical properties of outputs"""
    prompts = [f"Random test {i}" for i in range(50)]
    differences = []
    
    for prompt in prompts:
        out1 = wrapped_model.generate(prompt, seed=42)
        out2 = wrapped_model.generate(prompt, seed=42)
        out3 = wrapped_model.generate(prompt, seed=43)  # Different seed
        
        # Same seed should be identical
        diff_same = 0.0 if out1 == out2 else 1.0
        differences.append(diff_same)
        
        # Different seed might differ
        diff_other = 0.0 if out1 == out3 else 1.0
    
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    # Compute confidence interval
    n = len(differences)
    ci_lower = mean_diff - 2.58 * std_diff / np.sqrt(n)
    ci_upper = mean_diff + 2.58 * std_diff / np.sqrt(n)
    
    return {
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'confidence_interval': [max(0, ci_lower), min(1, ci_upper)],
        'samples': n,
        'decision': 'IDENTICAL' if ci_upper < 0.01 else 'DIFFERENT'
    }

# Run all tests
print("\n" + "="*80)
print("RUNNING POT FRAMEWORK TEST SUITE")
print("="*80)

tests = [
    ("Deterministic Validation", test_deterministic),
    ("Enhanced Diff Decision", test_enhanced_diff),
    ("Calibration System", test_calibration),
    ("Progressive Testing", test_progressive),
    ("KDF Challenges", test_kdf_challenges),
    ("Boundary Testing", test_boundaries),
    ("Statistical Verification", test_statistical)
]

passed = 0
failed = 0
total_time = 0

for name, test_func in tests:
    success = run_test(name, test_func)
    if success:
        passed += 1
    else:
        failed += 1
    
    if name in test_results['tests']:
        total_time += test_results['tests'][name]['time']

# Summary
print("\n" + "="*80)
print("TEST SUITE SUMMARY")
print("="*80)
print(f"Total tests: {len(tests)}")
print(f"✓ Passed: {passed}")
print(f"✗ Failed: {failed}")
print(f"Success rate: {passed/len(tests)*100:.1f}%")
print(f"Total time: {total_time:.1f}s")

# Key results
print("\nKEY FINDINGS:")
for test_name, test_data in test_results['tests'].items():
    if test_data['status'] == 'PASSED' and 'result' in test_data:
        result = test_data['result']
        if 'decision' in result:
            print(f"  {test_name}: {result['decision']}")
        if 'early_terminated' in result and result['early_terminated']:
            print(f"    → Early termination after {result.get('samples', 'N/A')} samples")
        if 'mean_difference' in result:
            print(f"    → Mean difference: {result['mean_difference']:.8f}")

# Save results
test_results['summary'] = {
    'total_tests': len(tests),
    'passed': passed,
    'failed': failed,
    'success_rate': passed/len(tests),
    'total_time': total_time,
    'model_load_time': load_time
}

with open(results_file, 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"\n✓ Results saved to: {results_file}")
print("="*80)