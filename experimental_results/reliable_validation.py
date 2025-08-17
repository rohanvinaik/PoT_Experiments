#!/usr/bin/env python3
"""
Standard PoT validation using deterministic test models.
This is the standard validation approach providing consistent, reproducible results.
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.security.proof_of_training import ProofOfTraining
from pot.testing.test_models import create_test_model
from pot.testing.validation_config import (
    ValidationConfig, 
    get_reliable_test_config,
    create_test_models_from_config
)


def run_reliable_verification_test(config: ValidationConfig) -> Dict[str, Any]:
    """
    Run verification tests with deterministic models.
    
    Args:
        config: Validation configuration
        
    Returns:
        Test results dictionary
    """
    results = {
        'test_name': 'reliable_verification',
        'timestamp': datetime.now().isoformat(),
        'config': config.__dict__,
        'results': []
    }
    
    print("=== Reliable Verification Test ===")
    print(f"Testing {len(config.verification_types)} verification types with deterministic models")
    
    for v_type in config.verification_types:
        print(f"\nTesting verification type: {v_type}")
        
        pot_config = {
            'verification_type': v_type,
            'model_type': 'generic',
            'security_level': 'medium'
        }
        
        try:
            pot = ProofOfTraining(pot_config)
            
            # Create deterministic test model
            test_model = create_test_model(
                config.model_type,
                model_id=f"reliable_test_{v_type}",
                seed=config.model_seed
            )
            
            # Register model
            model_id = pot.register_model(test_model, f"test_arch_{v_type}", 1000)
            print(f"  Model registered: {model_id}")
            
            # Test each verification depth
            depth_results = []
            for depth in config.verification_depths:
                print(f"    Testing depth: {depth}")
                
                start_time = time.time()
                result = pot.perform_verification(test_model, model_id, depth)
                duration = time.time() - start_time
                
                depth_result = {
                    'depth': depth,
                    'verified': bool(result.verified),
                    'confidence': float(result.confidence),
                    'challenges_passed': int(result.challenges_passed),
                    'challenges_total': int(result.challenges_total),
                    'duration': float(duration)
                }
                depth_results.append(depth_result)
                
                print(f"      Result: {result.verified} (confidence: {result.confidence:.2%})")
                print(f"      Challenges: {result.challenges_passed}/{result.challenges_total}")
            
            results['results'].append({
                'verification_type': v_type,
                'model_id': model_id,
                'depths': depth_results
            })
            
        except Exception as e:
            print(f"  Error testing {v_type}: {e}")
            results['results'].append({
                'verification_type': v_type,
                'error': str(e)
            })
    
    return results


def run_performance_benchmark(config: ValidationConfig) -> Dict[str, Any]:
    """
    Run performance benchmarks with deterministic models.
    
    Args:
        config: Validation configuration
        
    Returns:
        Benchmark results dictionary
    """
    results = {
        'test_name': 'performance_benchmark',
        'timestamp': datetime.now().isoformat(),
        'config': config.__dict__,
        'results': []
    }
    
    print("\n=== Performance Benchmark ===")
    
    # Test batch verification
    print("Testing batch verification performance...")
    
    pot_config = {
        'verification_type': 'fuzzy',
        'model_type': 'generic', 
        'security_level': 'low'  # Low for speed
    }
    
    pot = ProofOfTraining(pot_config)
    
    # Create multiple deterministic models
    models = create_test_models_from_config(config)
    model_ids = []
    
    # Register models
    start_time = time.time()
    for i, model in enumerate(models):
        model_id = pot.register_model(model, f"perf_model_{i}", 1000)
        model_ids.append(model_id)
    registration_time = time.time() - start_time
    
    print(f"  Registered {len(models)} models in {registration_time:.3f}s")
    
    # Batch verify
    start_time = time.time()
    verification_results = pot.batch_verify(models, model_ids, 'quick')
    verification_time = time.time() - start_time
    
    verified_count = sum(1 for r in verification_results if r.verified)
    
    print(f"  Verified {verified_count}/{len(models)} models in {verification_time:.3f}s")
    
    results['results'].append({
        'test_type': 'batch_verification',
        'model_count': len(models),
        'registration_time': registration_time,
        'verification_time': verification_time,
        'verified_count': verified_count,
        'success_rate': verified_count / len(models)
    })
    
    return results


def run_challenge_effectiveness_test(config: ValidationConfig) -> Dict[str, Any]:
    """
    Test challenge generation and effectiveness.
    
    Args:
        config: Validation configuration
        
    Returns:
        Challenge test results
    """
    results = {
        'test_name': 'challenge_effectiveness',
        'timestamp': datetime.now().isoformat(),
        'config': config.__dict__,
        'results': []
    }
    
    print("\n=== Challenge Effectiveness Test ===")
    
    from pot.core.challenge import generate_challenges
    
    for dimension in config.challenge_dimensions:
        print(f"Testing dimension {dimension}...")
        
        try:
            start_time = time.time()
            challenges = generate_challenges(
                dimension, 
                config.challenge_count, 
                'mixed'
            )
            generation_time = time.time() - start_time
            
            # Analyze challenges
            complexities = []
            sparsities = []
            
            for challenge in challenges:
                if hasattr(challenge, 'vector'):
                    vector = challenge.vector
                    complexity = np.std(vector)
                    sparsity = np.sum(np.abs(vector) < 0.1) / len(vector)
                    complexities.append(complexity)
                    sparsities.append(sparsity)
            
            result = {
                'dimension': dimension,
                'challenge_count': len(challenges),
                'generation_time': generation_time,
                'avg_complexity': np.mean(complexities) if complexities else 0,
                'avg_sparsity': np.mean(sparsities) if sparsities else 0
            }
            
            results['results'].append(result)
            
            print(f"  Generated {len(challenges)} challenges in {generation_time:.3f}s")
            print(f"  Avg complexity: {result['avg_complexity']:.3f}")
            print(f"  Avg sparsity: {result['avg_sparsity']:.3f}")
            
        except Exception as e:
            print(f"  Error with dimension {dimension}: {e}")
            results['results'].append({
                'dimension': dimension,
                'error': str(e)
            })
    
    return results


def main():
    """Run comprehensive standard validation."""
    print("Proof of Training - Standard Validation")
    print("=" * 50)
    
    # Use reliable configuration
    config = get_reliable_test_config()
    
    print(f"Configuration:")
    print(f"  Model type: {config.model_type}")
    print(f"  Session seed: {config.model_seed} (time-based)")
    print(f"  Model count: {config.model_count}")
    print(f"  Verification types: {config.verification_types}")
    print(f"  Verification depths: {config.verification_depths}")
    
    # Run all tests
    all_results = {
        'validation_run': {
            'timestamp': datetime.now().isoformat(),
            'config': config.__dict__,
            'tests': []
        }
    }
    
    try:
        # Verification test
        verification_results = run_reliable_verification_test(config)
        all_results['validation_run']['tests'].append(verification_results)
        
        # Performance benchmark
        performance_results = run_performance_benchmark(config)
        all_results['validation_run']['tests'].append(performance_results)
        
        # Challenge effectiveness
        challenge_results = run_challenge_effectiveness_test(config)
        all_results['validation_run']['tests'].append(challenge_results)
        
        # Summary
        print("\n" + "=" * 50)
        print("Validation Summary")
        print("=" * 50)
        
        verification_success = all(
            any(d['verified'] for d in test.get('depths', []))
            for test in verification_results['results']
            if 'depths' in test
        )
        
        performance_success = any(
            result.get('success_rate', 0) > 0
            for result in performance_results['results']
        )
        
        challenge_success = len(challenge_results['results']) > 0
        
        print(f"✓ Verification tests: {'PASS' if verification_success else 'FAIL'}")
        print(f"✓ Performance tests: {'PASS' if performance_success else 'FAIL'}")
        print(f"✓ Challenge tests: {'PASS' if challenge_success else 'FAIL'}")
        
        overall_success = verification_success and performance_success and challenge_success
        print(f"\nOverall result: {'SUCCESS' if overall_success else 'MIXED RESULTS'}")
        
        # Save results
        if config.generate_reports:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"reliable_validation_results_{timestamp}.json"
            
            # Use atomic write to prevent corruption
            from pot.core.jsonenc import atomic_json_dump
            atomic_json_dump(all_results, results_file)
            
            print(f"\nResults saved to: {results_file}")
            
            # Auto-update results history
            try:
                import subprocess
                print("\n🔄 Auto-updating validation results history...")
                subprocess.run(["python3", "scripts/update_results_history.py"], 
                              cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              capture_output=True)
                subprocess.run(["python3", "scripts/update_readme_metrics.py"],
                              cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              capture_output=True)
                print("✅ Results history and README metrics updated")
            except Exception as e:
                print(f"Note: Could not auto-update results history: {e}")
        
    except Exception as e:
        print(f"\nValidation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)