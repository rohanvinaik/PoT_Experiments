#!/usr/bin/env python3
"""Fix imports in experimental_report.py to use correct class/function names."""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

print("Testing correct imports...")

# Test what we can actually import
successful_imports = []
failed_imports = []

test_imports = [
    ("pot.core.challenge", "generate_challenges", "function"),
    ("pot.core.challenge", "ChallengeConfig", "class"),
    ("pot.core.sequential", "SequentialTester", "class"),
    ("pot.core.sequential", "sequential_verify", "function"),
    ("pot.core.fingerprint", "FingerprintConfig", "class"),
    ("pot.core.fingerprint", "fingerprint_model", "function"),
    ("pot.security.proof_of_training", "ProofOfTraining", "class"),
]

for module_name, item_name, item_type in test_imports:
    try:
        module = __import__(module_name, fromlist=[item_name])
        if hasattr(module, item_name):
            successful_imports.append(f"from {module_name} import {item_name}")
            print(f"✅ {item_name} from {module_name}")
        else:
            # Try to find what's available
            available = [x for x in dir(module) if not x.startswith('_') and (x[0].isupper() if item_type == "class" else x[0].islower())][:5]
            failed_imports.append(f"{module_name}.{item_name} - Available: {', '.join(available)}")
            print(f"❌ {item_name} not in {module_name}")
            print(f"   Available: {', '.join(available)}")
    except Exception as e:
        failed_imports.append(f"{module_name}.{item_name} - Error: {e}")
        print(f"❌ Error importing from {module_name}: {e}")

print("\n" + "="*60)
print("SUCCESSFUL IMPORTS TO USE:")
print("="*60)
for imp in successful_imports:
    print(imp)

print("\n" + "="*60)
print("FAILED IMPORTS (need alternatives):")
print("="*60)
for imp in failed_imports:
    print(imp)

# Now test a working configuration for challenges
print("\n" + "="*60)
print("TESTING CHALLENGE GENERATION:")
print("="*60)

try:
    from pot.core.challenge import generate_challenges, ChallengeConfig
    
    # Try with ChallengeConfig
    config = ChallengeConfig(
        num_challenges=10,
        challenge_type='numeric',
        master_key_hex='0' * 64,  # 32 bytes as hex
        model_id='test_model'
    )
    
    challenges = generate_challenges(config)
    print(f"✅ Successfully generated {len(challenges)} challenges with ChallengeConfig")
    
except Exception as e:
    print(f"❌ Failed with ChallengeConfig: {e}")
    
    # Try with dict
    try:
        config_dict = {
            'num_challenges': 10,
            'challenge_type': 'numeric',
            'master_key_hex': '0' * 64,
            'model_id': 'test_model'
        }
        challenges = generate_challenges(config_dict)
        print(f"✅ Successfully generated {len(challenges)} challenges with dict")
    except Exception as e2:
        print(f"❌ Failed with dict: {e2}")