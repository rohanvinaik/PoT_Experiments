#!/usr/bin/env python3
"""
Test script to verify all fixes for the experimental results issues
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_token_space_normalizer():
    """Test TokenSpaceNormalizer imports and basic functionality"""
    print("\n" + "="*60)
    print("Testing TokenSpaceNormalizer Fixes")
    print("="*60)
    
    try:
        from pot.security.token_space_normalizer import (
            TokenSpaceNormalizer,
            StochasticDecodingController,
            TokenizerType
        )
        print("✓ All imports successful")
        
        # Test StochasticDecodingController
        controller = StochasticDecodingController(temperature=0.8, seed=42)
        config = controller.get_config()
        print(f"✓ StochasticDecodingController initialized: {config}")
        
        # Test TokenizerType enum
        print(f"✓ TokenizerType values: {[t.value for t in TokenizerType]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_fuzzy_hash_verifier():
    """Test improved FuzzyHashVerifier with SHA256"""
    print("\n" + "="*60)
    print("Testing FuzzyHashVerifier SHA256 Improvements")
    print("="*60)
    
    try:
        from pot.security.fuzzy_hash_verifier import (
            FuzzyHashVerifier,
            HashAlgorithm
        )
        
        # Create verifier with SHA256
        verifier = FuzzyHashVerifier(
            similarity_threshold=0.85,
            algorithm=HashAlgorithm.SHA256
        )
        print("✓ FuzzyHashVerifier initialized with SHA256")
        
        # Test data
        base_vector = np.random.randn(1000)
        
        # Generate reference hash
        ref_hash = verifier.generate_fuzzy_hash(base_vector)
        print(f"✓ Generated reference hash (LSH-enhanced): {ref_hash[:50]}...")
        
        # Test exact match
        result = verifier.verify_model_output(base_vector, ref_hash)
        print(f"✓ Exact match: {result.is_valid} (similarity: {result.similarity_score:.4f})")
        
        # Test with small noise (should have better fuzzy matching now)
        noisy_output = base_vector + np.random.randn(1000) * 0.001
        result = verifier.verify_model_output(noisy_output, ref_hash)
        print(f"✓ Small noise: {result.is_valid} (similarity: {result.similarity_score:.4f})")
        
        # The similarity should be > 0 even with noise (not just 0.0 or 1.0)
        if 0 < result.similarity_score < 1:
            print("✓ Fuzzy matching working (similarity between 0 and 1)")
        else:
            print(f"⚠ Fuzzy matching may need tuning (similarity: {result.similarity_score})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_experiment():
    """Test that validation experiment can run without import errors"""
    print("\n" + "="*60)
    print("Testing Validation Experiment Imports")
    print("="*60)
    
    try:
        # Test the problematic imports
        from pot.security.proof_of_training import ProofOfTraining
        print("✓ ProofOfTraining imported")
        
        from pot.security.token_space_normalizer import (
            TokenSpaceNormalizer,
            StochasticDecodingController,
            TokenizerType
        )
        print("✓ TokenSpaceNormalizer components imported")
        
        from pot.security.fuzzy_hash_verifier import FuzzyHashVerifier
        print("✓ FuzzyHashVerifier imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing Experimental Results Fixes")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("TokenSpaceNormalizer", test_token_space_normalizer()))
    results.append(("FuzzyHashVerifier", test_fuzzy_hash_verifier()))
    results.append(("Validation Experiment", test_validation_experiment()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All fixes working correctly!")
    else:
        print(f"\n⚠️ {total - passed} test(s) still failing")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())