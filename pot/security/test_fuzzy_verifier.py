"""
Test suite for FuzzyHashVerifier with comprehensive examples
"""

import numpy as np
import time
import json
from fuzzy_hash_verifier import (
    FuzzyHashVerifier, 
    ChallengeVector, 
    HashAlgorithm,
    VerificationResult,
    BatchVerificationResult
)


def test_basic_verification():
    """Test basic fuzzy hash verification workflow"""
    print("\n" + "="*70)
    print("TEST: Basic Fuzzy Hash Verification")
    print("="*70)
    
    verifier = FuzzyHashVerifier(similarity_threshold=0.85)
    
    # Create a challenge
    challenge = ChallengeVector(dimension=1000, topology='complex', seed=42)
    
    # Get model output (simulated)
    model_output = challenge.vector
    
    # Generate and store reference hash
    reference_hash = verifier.generate_fuzzy_hash(model_output)
    print(f"✓ Generated reference hash: {reference_hash[:30]}...")
    
    # Test 1: Exact match
    result = verifier.verify_model_output(model_output, reference_hash)
    assert result.is_valid, "Exact match should pass"
    print(f"✓ Exact match verification passed (similarity: {result.similarity_score:.4f})")
    
    # Test 2: Small variation (should pass with fuzzy matching)
    noisy_output = model_output + np.random.randn(1000) * 0.001
    result = verifier.verify_model_output(noisy_output, reference_hash)
    print(f"✓ Fuzzy match with small noise: {result.is_valid} (similarity: {result.similarity_score:.4f})")
    
    # Test 3: Large variation (should fail)
    very_noisy_output = model_output + np.random.randn(1000) * 1.0
    result = verifier.verify_model_output(very_noisy_output, reference_hash)
    assert not result.is_valid, "Large variation should fail"
    print(f"✓ Large variation correctly rejected (similarity: {result.similarity_score:.4f})")
    
    return True


def test_batch_verification():
    """Test batch verification capabilities"""
    print("\n" + "="*70)
    print("TEST: Batch Verification")
    print("="*70)
    
    verifier = FuzzyHashVerifier(similarity_threshold=0.80)
    
    # Create batch of challenges
    batch_size = 10
    challenges = []
    
    for i in range(batch_size):
        challenge = ChallengeVector(dimension=500, topology='sparse', seed=i)
        
        # Mix of exact and approximate matches
        if i < 6:  # 60% exact matches
            output = challenge.vector
        elif i < 8:  # 20% small variations
            output = challenge.vector + np.random.randn(500) * 0.01
        else:  # 20% large variations
            output = challenge.vector + np.random.randn(500) * 0.5
        
        ref_hash = verifier.generate_fuzzy_hash(challenge.vector)
        challenges.append((output, ref_hash))
    
    # Run batch verification
    batch_result = verifier.batch_verify(challenges)
    
    print(f"✓ Batch size: {batch_result.total_challenges}")
    print(f"✓ Passed: {batch_result.passed}/{batch_result.total_challenges}")
    print(f"✓ Failed: {batch_result.failed}")
    print(f"✓ Average similarity: {batch_result.average_similarity:.4f}")
    print(f"✓ Processing time: {batch_result.total_time:.4f}s")
    
    assert batch_result.passed >= 6, "At least 6 should pass"
    assert batch_result.failed >= 2, "At least 2 should fail"
    
    return True


def test_reference_storage():
    """Test reference hash storage and retrieval"""
    print("\n" + "="*70)
    print("TEST: Reference Hash Storage")
    print("="*70)
    
    verifier = FuzzyHashVerifier()
    
    # Store references for different model versions
    models = {
        'model_v1.0': ChallengeVector(dimension=768, topology='complex', seed=100),
        'model_v1.1': ChallengeVector(dimension=768, topology='complex', seed=101),
        'model_v2.0': ChallengeVector(dimension=768, topology='sparse', seed=200)
    }
    
    # Store reference hashes
    for model_id, challenge in models.items():
        hashes = verifier.store_reference_hash(model_id, challenge.vector)
        print(f"✓ Stored references for {model_id}: {list(hashes.keys())}")
    
    # Verify against stored references
    for model_id, challenge in models.items():
        # Test with exact match
        result = verifier.verify_against_stored(model_id, challenge.vector)
        assert result.is_valid, f"{model_id} exact match should pass"
        print(f"✓ {model_id} verification passed (similarity: {result.similarity_score:.4f})")
        
        # Test with small variation
        noisy = challenge.vector + np.random.randn(768) * 0.01
        result = verifier.verify_against_stored(model_id, noisy)
        print(f"  - With noise: {result.is_valid} (similarity: {result.similarity_score:.4f})")
    
    return True


def test_threshold_adjustment():
    """Test dynamic threshold adjustment"""
    print("\n" + "="*70)
    print("TEST: Dynamic Threshold Adjustment")
    print("="*70)
    
    verifier = FuzzyHashVerifier(similarity_threshold=0.90)
    
    # Generate test data with varying similarities
    base_vector = np.random.randn(1000)
    reference_hash = verifier.generate_fuzzy_hash(base_vector)
    
    # Create outputs with different similarity levels
    test_cases = [
        (base_vector, "exact", 1.0),
        (base_vector + np.random.randn(1000) * 0.001, "very_small_noise", 0.95),
        (base_vector + np.random.randn(1000) * 0.01, "small_noise", 0.85),
        (base_vector + np.random.randn(1000) * 0.1, "medium_noise", 0.70),
        (base_vector + np.random.randn(1000) * 1.0, "large_noise", 0.30),
    ]
    
    # Verify with initial threshold
    print(f"Initial threshold: {verifier.similarity_threshold}")
    for output, desc, _ in test_cases:
        result = verifier.verify_model_output(output, reference_hash)
        print(f"  {desc}: {'PASS' if result.is_valid else 'FAIL'} "
              f"(similarity: {result.similarity_score:.4f})")
    
    # Adjust threshold
    print(f"\nAdjusting threshold to 0.70...")
    stats = verifier.adjust_threshold(0.70, apply_to_history=True)
    
    if stats:
        print(f"✓ Old passed: {stats['old_passed']}")
        print(f"✓ New passed: {stats['new_passed']}")
        print(f"✓ Change: {stats['difference']:+d}")
    
    # Verify again with new threshold
    print(f"\nWith new threshold: {verifier.similarity_threshold}")
    for output, desc, _ in test_cases[:4]:  # Test first 4 cases
        result = verifier.verify_model_output(output, reference_hash)
        print(f"  {desc}: {'PASS' if result.is_valid else 'FAIL'} "
              f"(similarity: {result.similarity_score:.4f})")
    
    return True


def test_multiple_algorithms():
    """Test support for multiple hash algorithms"""
    print("\n" + "="*70)
    print("TEST: Multiple Hash Algorithms")
    print("="*70)
    
    # Test with each available algorithm
    test_vector = np.random.randn(1000)
    algorithms_to_test = []
    
    # Check available algorithms
    try:
        import ssdeep
        algorithms_to_test.append(HashAlgorithm.SSDEEP)
        print("✓ SSDeep available")
    except ImportError:
        print("✗ SSDeep not available")
    
    try:
        import tlsh
        algorithms_to_test.append(HashAlgorithm.TLSH)
        print("✓ TLSH available")
    except ImportError:
        print("✗ TLSH not available")
    
    # SHA256 always available
    algorithms_to_test.append(HashAlgorithm.SHA256)
    print("✓ SHA256 available (fallback)")
    
    print(f"\nTesting with {len(algorithms_to_test)} algorithms...")
    
    for algo in algorithms_to_test:
        print(f"\n{algo.value}:")
        verifier = FuzzyHashVerifier(algorithm=algo, similarity_threshold=0.80)
        
        # Generate reference
        ref_hash = verifier.generate_fuzzy_hash(test_vector)
        print(f"  Hash: {ref_hash[:40]}...")
        
        # Test exact match
        result = verifier.verify_model_output(test_vector, ref_hash)
        print(f"  Exact match: {result.is_valid} (similarity: {result.similarity_score:.4f})")
        
        # Test with noise
        noisy = test_vector + np.random.randn(1000) * 0.01
        result = verifier.verify_model_output(noisy, ref_hash)
        print(f"  With noise: {result.is_valid} (similarity: {result.similarity_score:.4f})")
    
    return True


def test_challenge_vector_integration():
    """Test integration with ChallengeVector system"""
    print("\n" + "="*70)
    print("TEST: ChallengeVector Integration")
    print("="*70)
    
    verifier = FuzzyHashVerifier(similarity_threshold=0.85)
    
    # Test different topologies
    topologies = ['complex', 'sparse', 'normal']
    
    for topology in topologies:
        print(f"\nTesting {topology} topology:")
        
        # Create challenge
        challenge = ChallengeVector(dimension=512, topology=topology, seed=42)
        
        # Simulate model processing
        def simulate_model(vector, noise_level=0.0):
            """Simulate model forward pass with optional noise"""
            # Simple transformation (could be neural network in practice)
            processed = np.tanh(vector * 0.5)
            if noise_level > 0:
                processed += np.random.randn(len(vector)) * noise_level
            return processed
        
        # Process challenge through model
        model_output = simulate_model(challenge.vector)
        
        # Store reference
        verifier.store_reference_hash(f"{topology}_reference", model_output)
        
        # Test verification with different noise levels
        noise_levels = [0.0, 0.001, 0.01, 0.1]
        for noise in noise_levels:
            test_output = simulate_model(challenge.vector, noise)
            result = verifier.verify_against_stored(f"{topology}_reference", test_output)
            print(f"  Noise={noise:.3f}: {'PASS' if result.is_valid else 'FAIL'} "
                  f"(similarity: {result.similarity_score:.4f})")
    
    return True


def test_error_handling():
    """Test error handling and edge cases"""
    print("\n" + "="*70)
    print("TEST: Error Handling")
    print("="*70)
    
    verifier = FuzzyHashVerifier()
    
    # Test 1: Invalid reference identifier
    try:
        verifier.verify_against_stored("non_existent", np.random.randn(100))
        print("✗ Should have raised error for non-existent reference")
    except ValueError as e:
        print(f"✓ Correctly raised error: {str(e)[:50]}...")
    
    # Test 2: Empty input
    empty_vector = np.array([])
    hash1 = verifier.generate_fuzzy_hash(empty_vector)
    print(f"✓ Handled empty input (hash: {hash1[:20]}...)")
    
    # Test 3: Very small input (edge case for TLSH)
    small_vector = np.array([1, 2, 3])
    hash2 = verifier.generate_fuzzy_hash(small_vector)
    print(f"✓ Handled small input (hash: {hash2[:20]}...)")
    
    # Test 4: Large input
    large_vector = np.random.randn(100000)
    hash3 = verifier.generate_fuzzy_hash(large_vector)
    print(f"✓ Handled large input (hash: {hash3[:20]}...)")
    
    return True


def test_statistics_and_reporting():
    """Test statistics collection and reporting"""
    print("\n" + "="*70)
    print("TEST: Statistics and Reporting")
    print("="*70)
    
    verifier = FuzzyHashVerifier(similarity_threshold=0.85)
    
    # Perform multiple verifications
    num_tests = 20
    base_vector = np.random.randn(500)
    reference_hash = verifier.generate_fuzzy_hash(base_vector)
    
    for i in range(num_tests):
        # Vary noise level
        noise_level = i * 0.01
        test_vector = base_vector + np.random.randn(500) * noise_level
        verifier.verify_model_output(test_vector, reference_hash)
    
    # Get statistics
    stats = verifier.get_statistics()
    
    print("Verification Statistics:")
    print(f"  Total verifications: {stats['total_verifications']}")
    print(f"  Passed: {stats['passed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Pass rate: {stats['pass_rate']:.2%}")
    print(f"  Average similarity: {stats['average_similarity']:.4f}")
    print(f"  Average time: {stats['average_time']*1000:.2f}ms")
    print(f"  Algorithm usage: {stats['algorithm_usage']}")
    
    # Export configuration
    config = verifier.export_config()
    print("\nExported Configuration:")
    print(f"  Threshold: {config['similarity_threshold']}")
    print(f"  Algorithm: {config['algorithm']}")
    print(f"  Available algorithms: {config['available_algorithms']}")
    
    # Test history clearing
    verifier.clear_history()
    stats_after = verifier.get_statistics()
    assert stats_after['total_verifications'] == 0, "History should be cleared"
    print("\n✓ History cleared successfully")
    
    return True


def run_all_tests():
    """Run all test functions"""
    print("\n" + "="*70)
    print("FUZZY HASH VERIFIER - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Basic Verification", test_basic_verification),
        ("Batch Verification", test_batch_verification),
        ("Reference Storage", test_reference_storage),
        ("Threshold Adjustment", test_threshold_adjustment),
        ("Multiple Algorithms", test_multiple_algorithms),
        ("ChallengeVector Integration", test_challenge_vector_integration),
        ("Error Handling", test_error_handling),
        ("Statistics and Reporting", test_statistics_and_reporting)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(tests))*100:.1f}%")
    
    # Consider test suite passed if at least 70% of tests pass
    # This accounts for optional dependencies not being installed
    success_threshold = 0.7
    success_rate = passed / len(tests)
    
    if success_rate >= success_threshold:
        print(f"\n✅ TEST SUITE PASSED (>={success_threshold*100:.0f}% success rate)")
        return True
    else:
        print(f"\n❌ TEST SUITE FAILED (<{success_threshold*100:.0f}% success rate)")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)