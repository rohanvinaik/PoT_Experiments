#!/usr/bin/env python3
"""
Test script for fingerprint comparison utilities
"""

import sys
import numpy as np
from typing import List

# Add parent directory to path
sys.path.append('.')

from pot.core.fingerprint import (
    FingerprintResult,
    compare_fingerprints,
    fingerprint_distance,
    is_behavioral_match,
    batch_compare_fingerprints,
    find_closest_match
)


def create_test_fingerprint(io_hash: str, jacobian_sketch: str = None, 
                           model_type: str = 'vision') -> FingerprintResult:
    """Create a test fingerprint with specified parameters"""
    return FingerprintResult(
        io_hash=io_hash,
        jacobian_sketch=jacobian_sketch,
        raw_outputs=[],
        timing_info=[],
        metadata={'model_type': model_type, 'num_challenges': 10}
    )


def test_compare_fingerprints():
    """Test compare_fingerprints function"""
    print("Testing compare_fingerprints...")
    
    # Test 1: Identical fingerprints
    fp1 = create_test_fingerprint("abcd1234" * 8)
    fp2 = create_test_fingerprint("abcd1234" * 8)
    similarity = compare_fingerprints(fp1, fp2)
    assert similarity == 1.0, f"Identical fingerprints should have similarity 1.0, got {similarity}"
    print("  ✓ Identical fingerprints: similarity = 1.0")
    
    # Test 2: Completely different fingerprints
    fp3 = create_test_fingerprint("efgh5678" * 8)
    similarity = compare_fingerprints(fp1, fp3)
    print(f"  ✓ Different fingerprints: similarity = {similarity:.4f}")
    
    # Test 3: With Jacobian sketches
    fp4 = create_test_fingerprint("abcd1234" * 8, "deadbeef" * 4)
    fp5 = create_test_fingerprint("abcd1234" * 8, "deadbeef" * 4)
    similarity = compare_fingerprints(fp4, fp5)
    assert similarity == 1.0, f"Identical fingerprints with Jacobian should have similarity 1.0, got {similarity}"
    print("  ✓ Identical with Jacobian: similarity = 1.0")
    
    # Test 4: Same IO hash, different Jacobian
    fp6 = create_test_fingerprint("abcd1234" * 8, "cafebabe" * 4)
    similarity = compare_fingerprints(fp4, fp6)
    print(f"  ✓ Same IO, different Jacobian: similarity = {similarity:.4f}")
    assert 0.6 <= similarity <= 1.0, f"Mixed match should have partial similarity, got {similarity}"
    
    # Test 5: None handling
    similarity = compare_fingerprints(None, fp1)
    assert similarity == 0.0, "None fingerprint should have similarity 0.0"
    print("  ✓ None handling: similarity = 0.0")
    
    print("  All tests passed!\n")


def test_fingerprint_distance():
    """Test fingerprint_distance function with different metrics"""
    print("Testing fingerprint_distance...")
    
    fp1 = create_test_fingerprint("abcd1234" * 8, "deadbeef" * 4)
    fp2 = create_test_fingerprint("abcd1234" * 8, "deadbeef" * 4)
    fp3 = create_test_fingerprint("efgh5678" * 8, "cafebabe" * 4)
    
    # Test combined metric
    distance = fingerprint_distance(fp1, fp2, metric='combined')
    assert distance == 0.0, f"Identical fingerprints should have distance 0.0, got {distance}"
    print(f"  ✓ Combined metric (identical): distance = {distance}")
    
    distance = fingerprint_distance(fp1, fp3, metric='combined')
    print(f"  ✓ Combined metric (different): distance = {distance:.4f}")
    
    # Test IO metric
    fp4 = create_test_fingerprint("abcd1234" * 8)
    fp5 = create_test_fingerprint("abcd1234" * 8)
    distance = fingerprint_distance(fp4, fp5, metric='io')
    assert distance == 0.0, f"Same IO hash should have distance 0.0, got {distance}"
    print(f"  ✓ IO metric (same hash): distance = {distance}")
    
    # Test Jacobian metric
    distance = fingerprint_distance(fp1, fp2, metric='jacobian')
    assert distance == 0.0, f"Same Jacobian should have distance 0.0, got {distance}"
    print(f"  ✓ Jacobian metric (same sketch): distance = {distance}")
    
    # Test missing Jacobian
    distance = fingerprint_distance(fp4, fp5, metric='jacobian')
    assert distance == 1.0, f"Missing Jacobian should have distance 1.0, got {distance}"
    print(f"  ✓ Jacobian metric (missing): distance = {distance}")
    
    # Test Hamming distance
    fp6 = create_test_fingerprint("abcd1234" * 8)
    fp7 = create_test_fingerprint("abcd1235" * 8)  # One character different
    distance = fingerprint_distance(fp6, fp7, metric='hamming')
    print(f"  ✓ Hamming metric: distance = {distance:.4f}")
    
    print("  All tests passed!\n")


def test_is_behavioral_match():
    """Test is_behavioral_match function"""
    print("Testing is_behavioral_match...")
    
    fp1 = create_test_fingerprint("abcd1234" * 8, "deadbeef" * 4)
    fp2 = create_test_fingerprint("abcd1234" * 8, "deadbeef" * 4)
    fp3 = create_test_fingerprint("efgh5678" * 8, "cafebabe" * 4)
    
    # Test exact match
    is_match = is_behavioral_match(fp1, fp2, threshold=0.95)
    assert is_match, "Identical fingerprints should match"
    print("  ✓ Identical fingerprints: MATCH")
    
    # Test non-match
    is_match = is_behavioral_match(fp1, fp3, threshold=0.95)
    assert not is_match, "Different fingerprints should not match"
    print("  ✓ Different fingerprints: NO MATCH")
    
    # Test with verbose output
    print("\n  Testing with verbose output:")
    is_match = is_behavioral_match(fp1, fp3, threshold=0.5, verbose=True)
    
    # Test threshold adjustment
    fp4 = create_test_fingerprint("abcd1234" * 8, "cafebabe" * 4)  # Same IO, different Jacobian
    is_match_high = is_behavioral_match(fp1, fp4, threshold=0.9)
    is_match_low = is_behavioral_match(fp1, fp4, threshold=0.5)
    print(f"\n  ✓ Threshold test: high={is_match_high}, low={is_match_low}")
    
    print("  All tests passed!\n")


def test_batch_compare_fingerprints():
    """Test batch_compare_fingerprints function"""
    print("Testing batch_compare_fingerprints...")
    
    # Create test fingerprints
    fingerprints = [
        create_test_fingerprint("aaaa" * 16),
        create_test_fingerprint("aaaa" * 16),  # Identical to first
        create_test_fingerprint("bbbb" * 16),
        create_test_fingerprint("cccc" * 16)
    ]
    
    # Test pairwise comparison
    similarity_matrix = batch_compare_fingerprints(fingerprints)
    assert similarity_matrix.shape == (4, 4), f"Expected 4x4 matrix, got {similarity_matrix.shape}"
    assert np.allclose(np.diag(similarity_matrix), 1.0), "Diagonal should be 1.0"
    assert similarity_matrix[0, 1] == 1.0, "Identical fingerprints should have similarity 1.0"
    print("  ✓ Pairwise comparison matrix created")
    print(f"    Shape: {similarity_matrix.shape}")
    print(f"    Diagonal: {np.diag(similarity_matrix)}")
    
    # Test comparison against reference
    reference = create_test_fingerprint("aaaa" * 16)
    similarities = batch_compare_fingerprints(fingerprints, reference=reference)
    assert similarities.shape == (4,), f"Expected 1D array of length 4, got {similarities.shape}"
    assert similarities[0] == 1.0 and similarities[1] == 1.0, "Matching fingerprints should have similarity 1.0"
    print("  ✓ Reference comparison vector created")
    print(f"    Similarities: {similarities}")
    
    print("  All tests passed!\n")


def test_find_closest_match():
    """Test find_closest_match function"""
    print("Testing find_closest_match...")
    
    query = create_test_fingerprint("abcd" * 16, "dead" * 8)
    candidates = [
        create_test_fingerprint("efgh" * 16, "beef" * 8),
        create_test_fingerprint("abcd" * 16, "dead" * 8),  # Exact match
        create_test_fingerprint("ijkl" * 16, "cafe" * 8),
        create_test_fingerprint("abcd" * 16, "babe" * 8)   # Partial match
    ]
    
    # Test finding best match
    best_idx, best_sim = find_closest_match(query, candidates)
    assert best_idx == 1, f"Should find exact match at index 1, got {best_idx}"
    assert best_sim == 1.0, f"Exact match should have similarity 1.0, got {best_sim}"
    print(f"  ✓ Found best match: index={best_idx}, similarity={best_sim}")
    
    # Test with all scores returned
    best_idx, best_sim, all_sims = find_closest_match(query, candidates, return_all_scores=True)
    assert len(all_sims) == 4, f"Should return 4 similarities, got {len(all_sims)}"
    print(f"  ✓ All similarities: {all_sims}")
    
    # Test empty candidates
    best_idx, best_sim = find_closest_match(query, [])
    assert best_idx == -1 and best_sim == 0.0, "Empty candidates should return -1, 0.0"
    print("  ✓ Empty candidates handled correctly")
    
    print("  All tests passed!\n")


def test_real_world_scenario():
    """Test a more realistic scenario with gradual changes"""
    print("Testing real-world scenario...")
    
    # Simulate model versions with gradual drift
    base_hash = "0123456789abcdef" * 4
    models = []
    
    # Original model
    models.append(create_test_fingerprint(base_hash, "original" * 2))
    
    # Minor update (99% similar)
    minor_hash = base_hash[:-1] + "e"
    models.append(create_test_fingerprint(minor_hash, "original" * 2))
    
    # Moderate update (90% similar)
    moderate_hash = base_hash[:-8] + "fedcba98"
    models.append(create_test_fingerprint(moderate_hash, "updated1" * 2))
    
    # Major update (70% similar)
    major_hash = base_hash[:32] + "fedcba9876543210" * 2
    models.append(create_test_fingerprint(major_hash, "updated2" * 2))
    
    # Completely different model
    models.append(create_test_fingerprint("deadbeefcafebabe" * 4, "different" * 2))
    
    # Compare all models pairwise
    similarity_matrix = batch_compare_fingerprints(models)
    
    print("  Similarity matrix (models vs models):")
    print("  " + " " * 10 + "Orig   Minor  Moder  Major  Diff")
    labels = ["Original ", "Minor    ", "Moderate ", "Major    ", "Different"]
    for i, label in enumerate(labels):
        row_str = f"  {label}"
        for j in range(5):
            row_str += f"{similarity_matrix[i, j]:6.3f} "
        print(row_str)
    
    # Test behavioral matching with different thresholds
    print("\n  Behavioral matching tests:")
    thresholds = [0.99, 0.95, 0.90, 0.80]
    for threshold in thresholds:
        matches = []
        for i in range(1, 5):
            is_match = is_behavioral_match(models[0], models[i], threshold=threshold)
            matches.append("MATCH" if is_match else "NO")
        print(f"    Threshold {threshold:.2f}: Minor={matches[0]}, Moderate={matches[1]}, "
              f"Major={matches[2]}, Different={matches[3]}")
    
    print("\n  Real-world scenario tests passed!\n")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Fingerprint Comparison Utilities")
    print("=" * 60 + "\n")
    
    try:
        test_compare_fingerprints()
        test_fingerprint_distance()
        test_is_behavioral_match()
        test_batch_compare_fingerprints()
        test_find_closest_match()
        test_real_world_scenario()
        
        print("=" * 60)
        print("✅ All fingerprint comparison tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)