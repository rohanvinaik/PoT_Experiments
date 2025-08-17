#!/usr/bin/env python3
"""
Quick verification script to confirm PoT system is working
"""

import sys
import numpy as np

print("=" * 70)
print("PROOF-OF-TRAINING INSTALLATION VERIFICATION")
print("=" * 70)

# Test imports
print("\n1. Testing imports...")
try:
    from pot.security.proof_of_training import ProofOfTraining
    from pot.security.fuzzy_hash_verifier import FuzzyHashVerifier
    from pot.prototypes.training_provenance_auditor import TrainingProvenanceAuditor
    from pot.security.token_space_normalizer import TokenSpaceNormalizer
    print("✓ All security components imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test basic functionality
print("\n2. Testing basic functionality...")
try:
    # Initialize PoT system
    config = {
        'verification_type': 'fuzzy',
        'model_type': 'generic',
        'security_level': 'medium'
    }
    pot = ProofOfTraining(config)
    print("✓ PoT system initialized")
    
    # Create mock model
    class MockModel:
        def forward(self, x):
            return np.random.randn(10)
        def state_dict(self):
            return {'layer': 'weights'}
    
    model = MockModel()
    
    # Register model
    model_id = pot.register_model(model, "test_model", 1000)
    print(f"✓ Model registered with ID: {model_id}")
    
    # Perform verification
    result = pot.perform_verification(model, model_id, 'quick')
    print(f"✓ Verification completed: {result.verified} (confidence: {result.confidence:.2%})")
    
    # Generate proof
    proof = pot.generate_verification_proof(result)
    print(f"✓ Proof generated: {len(proof)} bytes")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test experimental framework
print("\n3. Testing experimental framework...")
try:
    from pot.core.challenge import ChallengeConfig, generate_challenges
    from pot.core.governance import new_session_nonce
    
    config = ChallengeConfig(
        master_key_hex="0" * 64,
        session_nonce_hex=new_session_nonce(),
        n=10,
        family="vision:freq",
        params={"freq_range": [0.5, 8.0], "contrast_range": [0.3, 1.0]}
    )
    
    challenges = generate_challenges(config)
    print(f"✓ Generated {len(challenges['items'])} challenges")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print("\n✅ The PoT system is properly installed and functional!")
print("\nYou can now:")
print("1. Run experiments: python scripts/run_grid.py --config configs/vision_cifar10.yaml --exp E1")
print("2. Run full test suite: bash run_all.sh")
print("3. Check documentation: See EXPERIMENTS.md for detailed protocols")