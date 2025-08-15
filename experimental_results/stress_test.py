"""
Stress testing the PoT system
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.security.proof_of_training import ProofOfTraining

def stress_test_batch_verification():
    """Test batch verification with many models"""
    print("Stress testing batch verification...")
    
    config = {
        'verification_type': 'fuzzy',
        'model_type': 'generic',
        'security_level': 'low'  # Low for speed
    }
    
    pot = ProofOfTraining(config)
    
    # Create multiple mock models
    class MockModel:
        def __init__(self, seed):
            self.seed = seed
        def forward(self, x):
            np.random.seed(self.seed)
            return np.random.randn(10)
        def state_dict(self):
            return {'seed': self.seed}
    
    # Register many models
    num_models = 20
    models = []
    model_ids = []
    
    print(f"  Registering {num_models} models...")
    start = time.time()
    
    for i in range(num_models):
        model = MockModel(i)
        model_id = pot.register_model(model, f"model_{i}", 1000)
        models.append(model)
        model_ids.append(model_id)
    
    reg_time = time.time() - start
    print(f"  Registration time: {reg_time:.2f}s ({reg_time/num_models:.3f}s per model)")
    
    # Batch verify
    print(f"  Batch verifying {num_models} models...")
    start = time.time()
    
    results = pot.batch_verify(models, model_ids, 'quick')
    
    batch_time = time.time() - start
    print(f"  Batch verification time: {batch_time:.2f}s ({batch_time/num_models:.3f}s per model)")
    
    # Check results
    verified_count = sum(1 for r in results if r.verified)
    print(f"  Verified: {verified_count}/{num_models}")
    
    return verified_count == num_models

def stress_test_large_challenges():
    """Test with large challenge vectors"""
    print("Stress testing large challenges...")
    
    from pot.security.fuzzy_hash_verifier import ChallengeVector, FuzzyHashVerifier
    
    verifier = FuzzyHashVerifier()
    
    for dimension in [1000, 5000, 10000, 50000]:
        print(f"  Testing dimension {dimension}...")
        
        try:
            start = time.time()
            challenge = ChallengeVector(dimension=dimension, topology='complex')
            gen_time = time.time() - start
            
            start = time.time()
            hash_val = verifier.generate_fuzzy_hash(challenge.vector)
            hash_time = time.time() - start
            
            print(f"    Generation: {gen_time:.3f}s, Hashing: {hash_time:.3f}s")
            
        except Exception as e:
            print(f"    Error at dimension {dimension}: {e}")
            return False
    
    return True

def stress_test_provenance_history():
    """Test with large training histories"""
    print("Stress testing training history...")
    
    from pot.security.training_provenance_auditor import TrainingProvenanceAuditor, EventType, ProofType
    
    auditor = TrainingProvenanceAuditor(
        model_id="stress_test",
        max_history_size=100  # Test compression
    )
    
    num_epochs = 500
    print(f"  Logging {num_epochs} training events...")
    
    start = time.time()
    for epoch in range(num_epochs):
        auditor.log_training_event(
            epoch=epoch,
            metrics={
                'loss': 1.0/(epoch+1),
                'accuracy': min(0.99, epoch/100),
                'gradient_norm': np.random.random()
            },
            event_type=EventType.EPOCH_END
        )
    
    log_time = time.time() - start
    print(f"  Logging time: {log_time:.2f}s ({log_time/num_epochs:.6f}s per event)")
    
    # Generate proof
    print("  Generating Merkle proof...")
    start = time.time()
    proof = auditor.generate_training_proof(0, num_epochs-1, ProofType.MERKLE)
    proof_time = time.time() - start
    print(f"  Proof generation time: {proof_time:.3f}s")
    
    # Check compression
    stats = auditor.get_statistics()
    print(f"  Events in memory: {stats['total_events']}")
    print(f"  Compression active: {len(auditor.events) < num_epochs}")
    
    return True

if __name__ == "__main__":
    print("Starting stress tests...")
    
    tests = [
        ("Batch Verification", stress_test_batch_verification),
        ("Large Challenges", stress_test_large_challenges),
        ("Provenance History", stress_test_provenance_history)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                print(f"  ✓ {test_name} passed")
                passed += 1
            else:
                print(f"  ✗ {test_name} failed")
        except Exception as e:
            print(f"  ✗ {test_name} error: {e}")
    
    print(f"\nStress test results: {passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
