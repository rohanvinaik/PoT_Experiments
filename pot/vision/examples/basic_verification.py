#!/usr/bin/env python3
"""
Basic Vision Verification Example

This script demonstrates the simplest way to verify a vision model
using the PoT vision verification framework.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from pot.vision.verifier import EnhancedVisionVerifier


def create_simple_model():
    """Create a simple CNN for demonstration."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 10)
    )


def main():
    """Run basic vision verification example."""
    
    print("Basic Vision Verification Example")
    print("=" * 50)
    
    # Step 1: Create or load a model
    print("\n1. Creating model...")
    model = create_simple_model()
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Parameters: {num_params:,}")
    
    # Step 2: Configure the verifier
    print("\n2. Configuring verifier...")
    config = {
        'device': 'cpu',  # Use 'cuda' for GPU acceleration
        'verification_method': 'batch',
        'temperature': 1.0,
        'normalization': 'softmax'
    }
    print(f"   Configuration: {config}")
    
    # Step 3: Create the verifier
    print("\n3. Creating verifier...")
    verifier = EnhancedVisionVerifier(model, config)
    print(f"   Verifier created successfully")
    print(f"   Device: {verifier.device}")
    
    # Step 4: Run verification
    print("\n4. Running verification...")
    print("   Generating challenges and testing model...")
    
    result = verifier.verify_session(
        num_challenges=5,  # Number of test challenges
        challenge_types=['frequency', 'texture']  # Types of challenges
    )
    
    # Step 5: Display results
    print("\n5. Verification Results:")
    print("=" * 30)
    
    # Main results
    status = "PASSED" if result['verified'] else "FAILED"
    print(f"   Status: {status}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Success Rate: {result['success_rate']:.2%}")
    print(f"   Challenges Used: {result['num_challenges']}")
    
    # Detailed results if available
    if 'results' in result and result['results']:
        print(f"\n   Challenge Details:")
        for i, challenge_result in enumerate(result['results']):
            challenge_type = challenge_result.get('challenge_type', 'unknown')
            success = '✓' if challenge_result.get('success', False) else '✗'
            print(f"     Challenge {i+1} ({challenge_type}): {success}")
    
    # Performance information
    if 'total_time' in result:
        print(f"\n   Performance:")
        print(f"     Total time: {result['total_time']:.2f}s")
        print(f"     Time per challenge: {result['total_time']/result['num_challenges']:.3f}s")
    
    # Step 6: Interpretation
    print("\n6. Interpretation:")
    if result['verified']:
        print("   ✓ Model verification PASSED")
        print("   ✓ Model appears to be genuine and behaves as expected")
        if result['confidence'] > 0.95:
            print("   ✓ High confidence in verification result")
        elif result['confidence'] > 0.85:
            print("   ~ Moderate confidence in verification result")
        else:
            print("   ⚠ Low confidence - consider running more challenges")
    else:
        print("   ✗ Model verification FAILED")
        print("   ✗ Model may be modified, wrapped, or substituted")
        print("   ⚠ Consider investigating model provenance")
    
    # Step 7: Next steps
    print("\n7. Next Steps:")
    if result['verified']:
        print("   • Run comprehensive verification for production use")
        print("   • Set up monitoring for ongoing verification")
        print("   • Consider calibrating with more samples")
    else:
        print("   • Investigate model source and modifications")
        print("   • Run diagnostic tests to identify issues")
        print("   • Consider re-training or obtaining verified model")
    
    print("\n" + "=" * 50)
    print("Basic verification example completed!")
    
    return 0 if result['verified'] else 1


if __name__ == "__main__":
    exit(main())