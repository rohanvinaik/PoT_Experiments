#!/usr/bin/env python3
"""Debug script to identify experimental validation issues."""

import sys
import os
import traceback

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

print(f"Python path: {sys.path[:3]}")
print(f"Current directory: {os.getcwd()}")
print(f"Script directory: {script_dir}")
print(f"Parent directory: {parent_dir}")
print()

# Test imports
tests = [
    ("Core challenge", "from pot.core.challenge import generate_challenges"),
    ("Sequential verifier", "from pot.core.sequential import SequentialVerifier"),
    ("Fingerprint", "from pot.core.fingerprint import compute_fingerprint"),
    ("Proof of Training", "from pot.security.proof_of_training import ProofOfTraining"),
    ("Fuzzy Hash Verifier", "from pot.security.fuzzy_hash_verifier import FuzzyHashVerifier"),
    ("Token Space Normalizer", "from pot.security.token_space_normalizer import TokenSpaceNormalizer"),
    ("Training Provenance Auditor", "from pot.prototypes.training_provenance_auditor import TrainingProvenanceAuditor"),
]

passed = 0
failed = 0

for name, import_stmt in tests:
    try:
        exec(import_stmt)
        print(f"✅ {name}: OK")
        passed += 1
    except ImportError as e:
        print(f"❌ {name}: {e}")
        failed += 1
    except Exception as e:
        print(f"⚠️ {name}: {type(e).__name__}: {e}")
        failed += 1

print(f"\nSummary: {passed} passed, {failed} failed")

# Try running a simple experiment
print("\n" + "="*60)
print("Testing E1 experiment setup...")
print("="*60)

try:
    from pot.core.challenge import generate_challenges
    
    # Generate some test challenges
    config = {
        'num_challenges': 10,
        'challenge_type': 'numeric',
        'seed': 42
    }
    
    challenges = generate_challenges(config)
    print(f"✅ Generated {len(challenges)} challenges")
    
    # Test sequential verification
    from pot.core.sequential import SequentialVerifier
    verifier = SequentialVerifier(alpha=0.05, beta=0.05, tau=0.5)
    print(f"✅ Created SequentialVerifier")
    
except Exception as e:
    print(f"❌ Error in experiment setup:")
    traceback.print_exc()

print("\nDebug complete!")