#!/usr/bin/env python3
"""
Comprehensive test suite for Merkle tree implementation in the PoT audit system.

Tests Merkle tree construction, proof generation and verification, various tree sizes,
edge cases, and integration with training provenance auditing.
"""

import sys
import os
import hashlib
from typing import List, Tuple

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pot.prototypes.training_provenance_auditor import (
    MerkleNode, build_merkle_tree, compute_merkle_root, 
    generate_merkle_proof, verify_merkle_proof
)


class TestMerkleTreeImplementation:
    """Comprehensive test suite for Merkle tree implementation."""
    
    def test_merkle_node_basic_functionality(self):
        """Test basic MerkleNode functionality."""
        print("Testing MerkleNode basic functionality...")
        
        # Test leaf node creation
        leaf_data = b"test_leaf_data"
        leaf_node = MerkleNode(data=leaf_data)
        
        assert leaf_node.is_leaf()
        assert leaf_node.data == leaf_data
        assert leaf_node.left is None
        assert leaf_node.right is None
        assert leaf_node.hash == hashlib.sha256(leaf_data).digest()
        assert len(leaf_node.get_hex_hash()) == 64  # SHA256 hex length
        
        # Test internal node creation
        left_child = MerkleNode(data=b"left_data")
        right_child = MerkleNode(data=b"right_data")
        internal_node = MerkleNode(left=left_child, right=right_child)
        
        assert not internal_node.is_leaf()
        assert internal_node.data is None
        assert internal_node.left == left_child
        assert internal_node.right == right_child
        
        # Internal node hash should be hash of concatenated child hashes
        expected_hash = hashlib.sha256(left_child.hash + right_child.hash).digest()
        assert internal_node.hash == expected_hash
        
        print("  ✓ MerkleNode basic functionality tests passed")
    
    def test_merkle_tree_construction_small(self):
        """Test Merkle tree construction with small datasets."""
        print("Testing Merkle tree construction (small datasets)...")
        
        # Test single element tree
        single_data = [b"single_element"]
        single_tree = build_merkle_tree(single_data)
        
        assert single_tree.is_leaf()
        assert single_tree.data == b"single_element"
        assert single_tree.hash == hashlib.sha256(b"single_element").digest()
        
        # Test two element tree
        two_data = [b"first", b"second"]
        two_tree = build_merkle_tree(two_data)
        
        assert not two_tree.is_leaf()
        assert two_tree.left.data == b"first"
        assert two_tree.right.data == b"second"
        
        # Root hash should be hash of concatenated child hashes
        expected_root = hashlib.sha256(
            hashlib.sha256(b"first").digest() + 
            hashlib.sha256(b"second").digest()
        ).digest()
        assert two_tree.hash == expected_root
        
        # Test three element tree (requires padding)
        three_data = [b"first", b"second", b"third"]
        three_tree = build_merkle_tree(three_data)
        
        assert not three_tree.is_leaf()
        # Should pad to 4 elements (duplicate last)
        assert three_tree.left is not None
        assert three_tree.right is not None
        
        # Test four element tree (perfect binary tree)
        four_data = [b"first", b"second", b"third", b"fourth"]
        four_tree = build_merkle_tree(four_data)
        
        assert not four_tree.is_leaf()
        # Left subtree should have first two elements
        assert four_tree.left.left.data == b"first"
        assert four_tree.left.right.data == b"second"
        # Right subtree should have last two elements
        assert four_tree.right.left.data == b"third"
        assert four_tree.right.right.data == b"fourth"
        
        print("  ✓ Small dataset Merkle tree construction tests passed")
    
    def test_merkle_tree_construction_medium(self):
        """Test Merkle tree construction with medium datasets."""
        print("Testing Merkle tree construction (medium datasets)...")
        
        # Test 8 elements (perfect binary tree)
        eight_data = [f"element_{i}".encode() for i in range(8)]
        eight_tree = build_merkle_tree(eight_data)
        
        # Verify tree structure
        assert not eight_tree.is_leaf()
        
        # Count leaves to verify all elements are included
        def count_leaves(node):
            if node.is_leaf():
                return 1
            return count_leaves(node.left) + count_leaves(node.right)
        
        leaf_count = count_leaves(eight_tree)
        assert leaf_count == 8
        
        # Test 10 elements (requires padding to 16)
        ten_data = [f"item_{i:02d}".encode() for i in range(10)]
        ten_tree = build_merkle_tree(ten_data)
        
        leaf_count_ten = count_leaves(ten_tree)
        assert leaf_count_ten == 16  # Padded to next power of 2
        
        # Test 15 elements (requires padding to 16)
        fifteen_data = [f"data_{i:02d}".encode() for i in range(15)]
        fifteen_tree = build_merkle_tree(fifteen_data)
        
        leaf_count_fifteen = count_leaves(fifteen_tree)
        assert leaf_count_fifteen == 16  # Padded to next power of 2
        
        # Test 16 elements (perfect power of 2)
        sixteen_data = [f"block_{i:02d}".encode() for i in range(16)]
        sixteen_tree = build_merkle_tree(sixteen_data)
        
        leaf_count_sixteen = count_leaves(sixteen_tree)
        assert leaf_count_sixteen == 16  # No padding needed
        
        print("  ✓ Medium dataset Merkle tree construction tests passed")
    
    def test_merkle_tree_construction_large(self):
        """Test Merkle tree construction with large datasets."""
        print("Testing Merkle tree construction (large datasets)...")
        
        # Test 100 elements
        hundred_data = [f"training_event_{i:03d}".encode() for i in range(100)]
        hundred_tree = build_merkle_tree(hundred_data)
        
        assert not hundred_tree.is_leaf()
        
        # Verify tree depth is logarithmic
        def tree_depth(node):
            if node.is_leaf():
                return 1
            return 1 + max(tree_depth(node.left), tree_depth(node.right))
        
        depth_100 = tree_depth(hundred_tree)
        assert depth_100 <= 8  # log2(128) = 7, plus 1 for root
        
        # Test 1000 elements
        thousand_data = [f"epoch_{i:04d}_checkpoint".encode() for i in range(1000)]
        thousand_tree = build_merkle_tree(thousand_data)
        
        depth_1000 = tree_depth(thousand_tree)
        assert depth_1000 <= 11  # log2(1024) = 10, plus 1 for root
        
        # Test determinism
        thousand_tree_2 = build_merkle_tree(thousand_data)
        assert thousand_tree.hash == thousand_tree_2.hash
        
        print("  ✓ Large dataset Merkle tree construction tests passed")
    
    def test_merkle_root_computation(self):
        """Test direct Merkle root computation."""
        print("Testing Merkle root computation...")
        
        # Test various sizes
        test_sizes = [1, 2, 3, 4, 5, 8, 10, 16, 25, 32, 50, 100]
        
        for size in test_sizes:
            data_blocks = [f"block_{i}".encode() for i in range(size)]
            
            # Compute root using tree construction
            tree = build_merkle_tree(data_blocks)
            tree_root = tree.hash
            
            # Compute root directly
            direct_root = compute_merkle_root(data_blocks)
            
            # Should be identical
            assert tree_root == direct_root
        
        # Test determinism
        test_data = [b"determinism", b"test", b"data", b"blocks"]
        root1 = compute_merkle_root(test_data)
        root2 = compute_merkle_root(test_data)
        assert root1 == root2
        
        # Test order sensitivity
        reordered_data = [b"blocks", b"data", b"test", b"determinism"]
        root_reordered = compute_merkle_root(reordered_data)
        assert root1 != root_reordered  # Different order should give different root
        
        print("  ✓ Merkle root computation tests passed")
    
    def test_merkle_proof_generation(self):
        """Test Merkle proof generation for various tree sizes."""
        print("Testing Merkle proof generation...")
        
        # Test small tree proof generation
        small_data = [b"a", b"b", b"c", b"d"]
        small_tree = build_merkle_tree(small_data)
        
        for i in range(len(small_data)):
            proof = generate_merkle_proof(small_tree, i)
            assert isinstance(proof, list)
            assert len(proof) > 0
            
            # Each proof element should be (sibling_hash, is_right)
            for sibling_hash, is_right in proof:
                assert isinstance(sibling_hash, bytes)
                assert len(sibling_hash) == 32  # SHA256 hash length
                assert isinstance(is_right, bool)
        
        # Test medium tree proof generation
        medium_data = [f"training_step_{i}".encode() for i in range(20)]
        medium_tree = build_merkle_tree(medium_data)
        
        # Generate proofs for first, middle, and last elements
        test_indices = [0, 10, 19]
        proofs = {}
        
        for idx in test_indices:
            proof = generate_merkle_proof(medium_tree, idx)
            proofs[idx] = proof
            
            # Proof length should be logarithmic in tree size
            assert len(proof) <= 6  # log2(32) = 5, plus some margin
        
        # Test large tree proof generation
        large_data = [f"block_{i:04d}".encode() for i in range(500)]
        large_tree = build_merkle_tree(large_data)
        
        # Generate proofs for random indices
        import random
        random.seed(42)  # Deterministic test
        large_test_indices = random.sample(range(500), 10)
        
        for idx in large_test_indices:
            proof = generate_merkle_proof(large_tree, idx)
            # Proof length should be logarithmic
            assert len(proof) <= 10  # log2(512) = 9, plus some margin
        
        print("  ✓ Merkle proof generation tests passed")
    
    def test_merkle_proof_verification(self):
        """Test Merkle proof verification."""
        print("Testing Merkle proof verification...")
        
        # Test comprehensive proof verification
        test_data = [f"verification_test_{i}".encode() for i in range(16)]
        tree = build_merkle_tree(test_data)
        root_hash = tree.hash
        
        # Verify all elements
        for i, data_block in enumerate(test_data):
            # Generate proof
            proof = generate_merkle_proof(tree, i)
            
            # Compute leaf hash
            leaf_hash = hashlib.sha256(data_block).digest()
            
            # Verify proof
            is_valid = verify_merkle_proof(leaf_hash, proof, root_hash)
            assert is_valid, f"Proof verification failed for index {i}"
        
        # Test proof verification with wrong data
        wrong_data = b"wrong_data_block"
        wrong_leaf_hash = hashlib.sha256(wrong_data).digest()
        
        for i in range(len(test_data)):
            proof = generate_merkle_proof(tree, i)
            # Should fail with wrong leaf hash
            is_valid = verify_merkle_proof(wrong_leaf_hash, proof, root_hash)
            assert not is_valid, f"Proof should fail with wrong data for index {i}"
        
        # Test proof verification with tampered proof
        test_index = 5
        original_proof = generate_merkle_proof(tree, test_index)
        original_leaf_hash = hashlib.sha256(test_data[test_index]).digest()
        
        # Tamper with proof
        if len(original_proof) > 0:
            tampered_proof = original_proof.copy()
            # Change first sibling hash
            fake_sibling = hashlib.sha256(b"fake_sibling").digest()
            tampered_proof[0] = (fake_sibling, tampered_proof[0][1])
            
            # Should fail with tampered proof
            is_valid = verify_merkle_proof(original_leaf_hash, tampered_proof, root_hash)
            assert not is_valid, "Proof should fail with tampered sibling hash"
        
        # Test proof verification with wrong root
        wrong_root = hashlib.sha256(b"wrong_root").digest()
        correct_proof = generate_merkle_proof(tree, 0)
        correct_leaf = hashlib.sha256(test_data[0]).digest()
        
        is_valid = verify_merkle_proof(correct_leaf, correct_proof, wrong_root)
        assert not is_valid, "Proof should fail with wrong root hash"
        
        print("  ✓ Merkle proof verification tests passed")
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        print("Testing edge cases...")
        
        # Test empty data list
        try:
            build_merkle_tree([])
            assert False, "Should raise ValueError for empty data list"
        except ValueError as e:
            assert "empty list" in str(e)
        
        try:
            compute_merkle_root([])
            assert False, "Should raise ValueError for empty data list"
        except ValueError as e:
            assert "empty list" in str(e)
        
        # Test single element edge case
        single_element = [b"lonely_element"]
        single_tree = build_merkle_tree(single_element)
        
        # Single element tree should work
        assert single_tree.is_leaf()
        
        # Proof for single element should work
        single_proof = generate_merkle_proof(single_tree, 0)
        single_leaf_hash = hashlib.sha256(b"lonely_element").digest()
        single_root = single_tree.hash
        
        # Single element proof might be empty or minimal
        is_valid = verify_merkle_proof(single_leaf_hash, single_proof, single_root)
        assert is_valid
        
        # Test invalid index for proof generation
        test_data = [b"element1", b"element2", b"element3"]
        test_tree = build_merkle_tree(test_data)
        
        try:
            generate_merkle_proof(test_tree, 10)  # Index out of range
            assert False, "Should raise IndexError for invalid index"
        except (IndexError, ValueError):
            pass  # Either exception type is acceptable
        
        try:
            generate_merkle_proof(test_tree, -1)  # Negative index
            assert False, "Should raise error for negative index"
        except (IndexError, ValueError):
            pass
        
        # Test very large single proof
        large_single = [b"x" * 100000]  # Large data block
        large_single_tree = build_merkle_tree(large_single)
        large_single_proof = generate_merkle_proof(large_single_tree, 0)
        
        large_leaf_hash = hashlib.sha256(b"x" * 100000).digest()
        is_valid = verify_merkle_proof(large_leaf_hash, large_single_proof, large_single_tree.hash)
        assert is_valid
        
        print("  ✓ Edge case tests passed")
    
    def test_merkle_tree_properties(self):
        """Test mathematical properties of Merkle trees."""
        print("Testing Merkle tree mathematical properties...")
        
        # Test 1: Avalanche effect (small change causes large hash change)
        original_data = [f"data_{i}".encode() for i in range(10)]
        modified_data = original_data.copy()
        modified_data[5] = b"modified_data_5"  # Change one element
        
        original_root = compute_merkle_root(original_data)
        modified_root = compute_merkle_root(modified_data)
        
        assert original_root != modified_root
        
        # Compute Hamming distance in bits
        original_bits = bin(int.from_bytes(original_root, 'big'))
        modified_bits = bin(int.from_bytes(modified_root, 'big'))
        
        # Pad to same length for comparison
        max_len = max(len(original_bits), len(modified_bits))
        original_bits = original_bits.ljust(max_len, '0')
        modified_bits = modified_bits.ljust(max_len, '0')
        
        hamming_distance = sum(b1 != b2 for b1, b2 in zip(original_bits, modified_bits))
        # Should have significant bit differences (avalanche effect)
        assert hamming_distance > 10  # At least some bits should change
        
        # Test 2: Determinism
        test_data = [f"determinism_test_{i}".encode() for i in range(20)]
        
        root1 = compute_merkle_root(test_data)
        root2 = compute_merkle_root(test_data)
        root3 = compute_merkle_root(test_data)
        
        assert root1 == root2 == root3
        
        # Test 3: Collision resistance (different data should give different roots)
        data_set_1 = [f"set1_element_{i}".encode() for i in range(10)]
        data_set_2 = [f"set2_element_{i}".encode() for i in range(10)]
        
        root_set_1 = compute_merkle_root(data_set_1)
        root_set_2 = compute_merkle_root(data_set_2)
        
        assert root_set_1 != root_set_2
        
        # Test 4: Proof consistency (all proofs for same tree should verify)
        consistency_data = [f"consistency_{i}".encode() for i in range(32)]
        consistency_tree = build_merkle_tree(consistency_data)
        consistency_root = consistency_tree.hash
        
        all_proofs_valid = True
        for i in range(len(consistency_data)):
            proof = generate_merkle_proof(consistency_tree, i)
            leaf_hash = hashlib.sha256(consistency_data[i]).digest()
            is_valid = verify_merkle_proof(leaf_hash, proof, consistency_root)
            all_proofs_valid &= is_valid
        
        assert all_proofs_valid, "All proofs for valid tree should verify"
        
        print("  ✓ Merkle tree mathematical properties tests passed")
    
    def test_performance_characteristics(self):
        """Test performance characteristics of Merkle operations."""
        print("Testing performance characteristics...")
        
        import time
        
        # Test tree construction performance
        sizes = [100, 500, 1000]
        construction_times = {}
        
        for size in sizes:
            data = [f"perf_test_{i:06d}".encode() for i in range(size)]
            
            start_time = time.time()
            tree = build_merkle_tree(data)
            construction_time = time.time() - start_time
            construction_times[size] = construction_time
            
            print(f"  Tree construction ({size} elements): {construction_time:.3f}s")
            
            # Performance should be reasonable
            assert construction_time < 5.0, f"Tree construction too slow for {size} elements"
        
        # Test proof generation performance
        large_data = [f"proof_perf_{i:06d}".encode() for i in range(1000)]
        large_tree = build_merkle_tree(large_data)
        
        start_time = time.time()
        proofs = []
        for i in range(0, 1000, 100):  # Every 100th element
            proof = generate_merkle_proof(large_tree, i)
            proofs.append(proof)
        proof_time = time.time() - start_time
        
        print(f"  Proof generation (10 proofs from 1000 elements): {proof_time:.3f}s")
        assert proof_time < 1.0, "Proof generation should be fast"
        
        # Test proof verification performance
        start_time = time.time()
        for i, proof in enumerate(proofs):
            element_idx = i * 100
            leaf_hash = hashlib.sha256(large_data[element_idx]).digest()
            is_valid = verify_merkle_proof(leaf_hash, proof, large_tree.hash)
            assert is_valid
        verify_time = time.time() - start_time
        
        print(f"  Proof verification (10 proofs): {verify_time:.3f}s")
        assert verify_time < 0.5, "Proof verification should be very fast"
        
        # Test scaling properties
        print(f"  Scaling analysis:")
        for size in sizes:
            elements_per_second = size / construction_times[size]
            print(f"    {size} elements: {elements_per_second:.0f} elements/second")
        
        print("  ✓ Performance characteristics tests passed")
    
    def test_training_provenance_integration(self):
        """Test integration with training provenance auditing."""
        print("Testing training provenance integration...")
        
        # Simulate training events
        training_events = []
        for epoch in range(10):
            event_data = {
                "epoch": epoch,
                "loss": 1.0 / (epoch + 1),
                "accuracy": 0.5 + epoch * 0.05,
                "timestamp": f"2024-01-01T{epoch:02d}:00:00Z",
                "checkpoint_hash": f"checkpoint_{epoch:08x}"
            }
            # Convert to bytes for Merkle tree
            event_bytes = str(event_data).encode()
            training_events.append(event_bytes)
        
        # Build provenance tree
        provenance_tree = build_merkle_tree(training_events)
        provenance_root = provenance_tree.hash
        
        # Generate proof for specific epoch
        epoch_5_index = 5
        epoch_5_proof = generate_merkle_proof(provenance_tree, epoch_5_index)
        
        # Verify epoch 5 training event
        epoch_5_hash = hashlib.sha256(training_events[epoch_5_index]).digest()
        is_valid = verify_merkle_proof(epoch_5_hash, epoch_5_proof, provenance_root)
        assert is_valid
        
        # Simulate model distribution with embedded provenance
        model_metadata = {
            "model_id": "training_proven_model_v1",
            "training_provenance_root": provenance_root.hex(),
            "total_training_events": len(training_events),
            "training_complete": True
        }
        
        # Verify training event during model validation
        claimed_epoch_5_data = training_events[epoch_5_index]
        claimed_epoch_5_hash = hashlib.sha256(claimed_epoch_5_data).digest()
        
        # Verifier can check without access to full training data
        stored_root = bytes.fromhex(model_metadata["training_provenance_root"])
        verification_result = verify_merkle_proof(
            claimed_epoch_5_hash, epoch_5_proof, stored_root
        )
        
        assert verification_result, "Training provenance verification should pass"
        
        # Test with tampered training data
        tampered_epoch_5 = training_events[epoch_5_index].replace(b"epoch", b"TAMPERED")
        tampered_hash = hashlib.sha256(tampered_epoch_5).digest()
        
        tampered_verification = verify_merkle_proof(
            tampered_hash, epoch_5_proof, stored_root
        )
        assert not tampered_verification, "Should detect tampered training data"
        
        print("  ✓ Training provenance integration tests passed")


def run_all_tests():
    """Run all Merkle tree tests."""
    print("=" * 70)
    print("COMPREHENSIVE MERKLE TREE IMPLEMENTATION TEST SUITE")
    print("=" * 70)
    
    test_suite = TestMerkleTreeImplementation()
    
    test_methods = [
        test_suite.test_merkle_node_basic_functionality,
        test_suite.test_merkle_tree_construction_small,
        test_suite.test_merkle_tree_construction_medium,
        test_suite.test_merkle_tree_construction_large,
        test_suite.test_merkle_root_computation,
        test_suite.test_merkle_proof_generation,
        test_suite.test_merkle_proof_verification,
        test_suite.test_edge_cases,
        test_suite.test_merkle_tree_properties,
        test_suite.test_performance_characteristics,
        test_suite.test_training_provenance_integration
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_method()
            passed += 1
        except Exception as e:
            print(f"✗ {test_method.__name__} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("🎉 ALL MERKLE TREE TESTS PASSED!")
        print("\nMerkle tree implementation ready for production!")
        print("Features validated:")
        print("  ✓ Merkle tree construction for various sizes")
        print("  ✓ Efficient Merkle root computation")
        print("  ✓ Cryptographic proof generation")
        print("  ✓ Secure proof verification")
        print("  ✓ Edge case handling")
        print("  ✓ Mathematical security properties")
        print("  ✓ Production-grade performance")
        print("  ✓ Training provenance integration")
    else:
        print(f"❌ {failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)