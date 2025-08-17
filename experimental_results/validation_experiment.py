"""
Comprehensive experimental validation of the Proof-of-Training system
"""

import sys
import json
import time
import numpy as np
from datetime import datetime

# Add parent directory to path
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PoT components
try:
    from pot.security.proof_of_training import ProofOfTraining, ChallengeLibrary
    from pot.security.fuzzy_hash_verifier import FuzzyHashVerifier, ChallengeVector
    from pot.prototypes.training_provenance_auditor import (
        TrainingProvenanceAuditor,
        EventType,
        ProofType,
    )
    from pot.security.token_space_normalizer import TokenSpaceNormalizer, StochasticDecodingController, TokenizerType
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def run_experiments():
    """Run comprehensive validation experiments"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'experiments': []
    }
    
    print("\n=== EXPERIMENT 1: Verification Types Comparison ===")
    exp1_results = experiment_verification_types()
    results['experiments'].append(exp1_results)
    
    print("\n=== EXPERIMENT 2: Security Levels Analysis ===")
    exp2_results = experiment_security_levels()
    results['experiments'].append(exp2_results)
    
    print("\n=== EXPERIMENT 3: Model Type Coverage ===")
    exp3_results = experiment_model_types()
    results['experiments'].append(exp3_results)
    
    print("\n=== EXPERIMENT 4: Challenge Effectiveness ===")
    exp4_results = experiment_challenge_effectiveness()
    results['experiments'].append(exp4_results)
    
    print("\n=== EXPERIMENT 5: Performance Benchmarks ===")
    exp5_results = experiment_performance()
    results['experiments'].append(exp5_results)
    
    return results

def experiment_verification_types():
    """Compare different verification types"""
    print("Testing verification types: exact, fuzzy, statistical")
    
    results = {'name': 'Verification Types', 'data': []}
    
    for v_type in ['exact', 'fuzzy', 'statistical']:
        config = {
            'verification_type': v_type,
            'model_type': 'generic',
            'security_level': 'medium'
        }
        
        try:
            pot = ProofOfTraining(config)
            
            # Mock model
            class MockModel:
                def __init__(self):
                    # Set seed for consistent responses
                    np.random.seed(42)
                    
                def forward(self, x):
                    # Return deterministic output based on input
                    if isinstance(x, np.ndarray):
                        seed = int(np.sum(x) * 1000) % 2**32
                    else:
                        seed = hash(str(x)) % 2**32
                    np.random.seed(seed)
                    return np.random.randn(10)
                    
                def state_dict(self):
                    return {'layer': 'weights'}
            
            model = MockModel()
            model_id = pot.register_model(model, "test_arch", 1000)
            
            # Test different depths
            for depth in ['quick', 'standard', 'comprehensive']:
                start = time.time()
                result = pot.perform_verification(model, model_id, depth)
                duration = time.time() - start
                
                results['data'].append({
                    'type': v_type,
                    'depth': depth,
                    'verified': result.verified,
                    'confidence': float(result.confidence),
                    'duration': duration
                })
                
                print(f"  {v_type}/{depth}: verified={result.verified}, "
                      f"confidence={result.confidence:.2%}, time={duration:.3f}s")
        
        except Exception as e:
            print(f"  Error with {v_type}: {e}")
            results['data'].append({'type': v_type, 'error': str(e)})
    
    return results

def experiment_security_levels():
    """Test different security levels"""
    print("Testing security levels: low, medium, high")
    
    results = {'name': 'Security Levels', 'data': []}
    
    for level in ['low', 'medium', 'high']:
        config = {
            'verification_type': 'fuzzy',
            'model_type': 'generic',
            'security_level': level
        }
        
        try:
            pot = ProofOfTraining(config)
            stats = pot.get_statistics()
            
            # Check threshold settings
            if pot.fuzzy_verifier:
                threshold = pot.fuzzy_verifier.similarity_threshold
            else:
                threshold = 0.85
            
            results['data'].append({
                'level': level,
                'threshold': threshold,
                'components': stats['components']
            })
            
            print(f"  {level}: threshold={threshold}, components={stats['components']}")
            
        except Exception as e:
            print(f"  Error with {level}: {e}")
            results['data'].append({'level': level, 'error': str(e)})
    
    return results

def experiment_model_types():
    """Test different model types"""
    print("Testing model types: vision, language, multimodal, generic")
    
    results = {'name': 'Model Types', 'data': []}
    
    for model_type in ['vision', 'language', 'multimodal', 'generic']:
        print(f"\n  Testing {model_type} model:")
        
        # Test challenge generation
        if model_type == 'vision':
            challenges = ChallengeLibrary.get_vision_challenges(224, 3, 3)
            print(f"    Generated {len(challenges)} vision challenges")
            
        elif model_type == 'language':
            challenges = ChallengeLibrary.get_language_challenges(50000, 100, 3)
            print(f"    Generated {len(challenges)} language challenges")
            
        elif model_type == 'multimodal':
            challenges = ChallengeLibrary.get_multimodal_challenges(2)
            print(f"    Generated {len(challenges)} multimodal challenges")
            
        else:
            challenges = ChallengeLibrary.get_generic_challenges(100, 3)
            print(f"    Generated {len(challenges)} generic challenges")
        
        results['data'].append({
            'model_type': model_type,
            'num_challenges': len(challenges),
            'challenge_types': type(challenges[0]).__name__ if challenges else None
        })
    
    return results

def experiment_challenge_effectiveness():
    """Test challenge-response effectiveness"""
    print("Testing challenge effectiveness")
    
    results = {'name': 'Challenge Effectiveness', 'data': []}
    
    # Test ChallengeVector
    for topology in ['complex', 'sparse', 'normal']:
        for dimension in [100, 500, 1000]:
            try:
                challenge = ChallengeVector(dimension=dimension, topology=topology, seed=42)
                
                # Analyze challenge properties
                mean = float(np.mean(challenge.vector))
                std = float(np.std(challenge.vector))
                sparsity = float(np.sum(np.abs(challenge.vector) < 0.01) / dimension)
                
                results['data'].append({
                    'topology': topology,
                    'dimension': dimension,
                    'mean': mean,
                    'std': std,
                    'sparsity': sparsity
                })
                
                print(f"  {topology}/{dimension}D: mean={mean:.3f}, "
                      f"std={std:.3f}, sparsity={sparsity:.2%}")
                
            except Exception as e:
                print(f"  Error with {topology}/{dimension}: {e}")
    
    return results

def experiment_performance():
    """Benchmark performance metrics"""
    print("Running performance benchmarks")
    
    results = {'name': 'Performance', 'data': []}
    
    # Test fuzzy hash performance
    print("\n  Fuzzy Hash Performance:")
    verifier = FuzzyHashVerifier(similarity_threshold=0.85)
    
    for size in [100, 1000, 10000]:
        data = np.random.randn(size)
        
        start = time.time()
        hash_val = verifier.generate_fuzzy_hash(data)
        hash_time = time.time() - start
        
        start = time.time()
        verifier.verify_fuzzy(hash_val, hash_val)
        verify_time = time.time() - start
        
        results['data'].append({
            'operation': 'fuzzy_hash',
            'size': size,
            'hash_time': hash_time,
            'verify_time': verify_time
        })
        
        print(f"    Size {size}: hash={hash_time:.6f}s, verify={verify_time:.6f}s")
    
    # Test provenance auditor performance
    print("\n  Provenance Auditor Performance:")
    auditor = TrainingProvenanceAuditor(model_id="perf_test")
    
    for num_events in [10, 50, 100]:
        start = time.time()
        for i in range(num_events):
            auditor.log_training_event(
                epoch=i,
                metrics={'loss': 1.0/(i+1)},
                event_type=EventType.EPOCH_END
            )
        log_time = time.time() - start
        
        start = time.time()
        proof = auditor.generate_training_proof(0, num_events-1, ProofType.MERKLE)
        proof_time = time.time() - start
        
        results['data'].append({
            'operation': 'provenance',
            'num_events': num_events,
            'log_time': log_time,
            'proof_time': proof_time
        })
        
        print(f"    Events {num_events}: log={log_time:.6f}s, proof={proof_time:.6f}s")
    
    return results

def convert_numpy_types(obj):
    """Convert NumPy types to Python types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

if __name__ == "__main__":
    print("Starting comprehensive experimental validation...")
    results = run_experiments()
    
    # Convert NumPy types before saving
    results = convert_numpy_types(results)
    
    # Save results
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n=== VALIDATION COMPLETE ===")
    print(f"Results saved to validation_results.json")
    
    # Summary
    total_experiments = len(results['experiments'])
    successful = sum(1 for exp in results['experiments'] if 'data' in exp and exp['data'])
    
    print(f"\nSummary:")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Successful: {successful}")
    print(f"  Success rate: {successful/total_experiments*100:.1f}%")
    
    sys.exit(0 if successful == total_experiments else 1)
