#!/usr/bin/env python3
"""
Comprehensive test suite for Merkle tree implementation in training_provenance_auditor.py

Tests all core functionality with edge cases, performance benchmarks, and integration scenarios.
"""

import sys
import os
import time
import hashlib
import random
from typing import List, Tuple

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pot.prototypes.training_provenance_auditor import (
    MerkleNode, build_merkle_tree, compute_merkle_root,
    generate_merkle_proof, verify_merkle_proof,
    _collect_leaves, _get_subtree_size
)


def test_merkle_node_basic():
    """Test basic MerkleNode functionality."""
    print("Testing MerkleNode basic functionality...")
    
    # Test leaf node
    data = b"test data"
    leaf = MerkleNode(data=data)
    assert leaf.data == data
    assert leaf.left is None
    assert leaf.right is None
    assert leaf.is_leaf() == True
    assert len(leaf.hash) == 32  # SHA256 produces 32 bytes
    assert leaf.get_hex_hash() == hashlib.sha256(data).hexdigest()
    
    # Test internal node
    left_leaf = MerkleNode(data=b"left")
    right_leaf = MerkleNode(data=b"right")
    internal = MerkleNode(left=left_leaf, right=right_leaf)
    assert internal.data is None
    assert internal.left == left_leaf
    assert internal.right == right_leaf
    assert internal.is_leaf() == False
    
    # Verify internal node hash
    expected_hash = hashlib.sha256(left_leaf.hash + right_leaf.hash).digest()
    assert internal.hash == expected_hash
    
    print("✓ MerkleNode basic functionality passed")


def test_build_merkle_tree_single():
    """Test building Merkle tree with single element."""
    print("Testing build_merkle_tree with single element...")
    
    data_blocks = [b"single block"]
    root = build_merkle_tree(data_blocks)
    
    assert root.is_leaf() == True
    assert root.data == b"single block"
    assert root.hash == hashlib.sha256(b"single block").digest()
    
    print("✓ Single element tree passed")


def test_build_merkle_tree_two():
    """Test building Merkle tree with two elements."""
    print("Testing build_merkle_tree with two elements...")
    
    data_blocks = [b"block1", b"block2"]
    root = build_merkle_tree(data_blocks)
    
    assert root.is_leaf() == False
    assert root.left.data == b"block1"
    assert root.right.data == b"block2"
    
    # Verify root hash
    left_hash = hashlib.sha256(b"block1").digest()
    right_hash = hashlib.sha256(b"block2").digest()
    expected_root = hashlib.sha256(left_hash + right_hash).digest()
    assert root.hash == expected_root
    
    print("✓ Two element tree passed")


def test_build_merkle_tree_odd():
    """Test building Merkle tree with odd number of elements."""
    print("Testing build_merkle_tree with odd number of elements...")
    
    data_blocks = [b"block1", b"block2", b"block3"]
    root = build_merkle_tree(data_blocks)
    
    # Should have 3 leaves
    leaves = _collect_leaves(root)
    assert len(leaves) == 4  # Last node is duplicated for odd count
    assert leaves[0].data == b"block1"
    assert leaves[1].data == b"block2"
    assert leaves[2].data == b"block3"
    assert leaves[3].data == b"block3"  # Duplicate
    
    print("✓ Odd element tree passed")


def test_build_merkle_tree_power_of_two():
    """Test building Merkle tree with power of 2 elements."""
    print("Testing build_merkle_tree with power of 2 elements...")
    
    data_blocks = [f"block{i}".encode() for i in range(4)]
    root = build_merkle_tree(data_blocks)
    
    leaves = _collect_leaves(root)
    assert len(leaves) == 4
    
    # Verify tree structure
    assert root.left.left.data == b"block0"
    assert root.left.right.data == b"block1"
    assert root.right.left.data == b"block2"
    assert root.right.right.data == b"block3"
    
    print("✓ Power of 2 tree passed")


def test_build_merkle_tree_large():
    """Test building Merkle tree with large number of elements."""
    print("Testing build_merkle_tree with large number of elements...")
    
    # Test with 100 elements
    data_blocks = [f"block{i}".encode() for i in range(100)]
    root = build_merkle_tree(data_blocks)
    
    leaves = _collect_leaves(root)
    # Should have power of 2 number of leaves >= 100
    assert len(leaves) >= 100
    
    # Count how many of each original block appear
    block_counts = {}
    for leaf in leaves:
        if leaf.data in block_counts:
            block_counts[leaf.data] += 1
        else:
            block_counts[leaf.data] = 1
    
    # All original blocks should appear at least once
    for i in range(100):
        block_data = f"block{i}".encode()
        assert block_data in block_counts, f"block{i} missing from tree"
        assert block_counts[block_data] >= 1, f"block{i} appears {block_counts[block_data]} times"
    
    # Total unique original blocks should be 100
    original_blocks = [f"block{i}".encode() for i in range(100)]
    original_in_tree = [data for data in block_counts.keys() if data in original_blocks]
    assert len(original_in_tree) == 100, f"Expected 100 unique blocks, got {len(original_in_tree)}"
    
    print("✓ Large tree passed")


def test_compute_merkle_root():
    """Test compute_merkle_root function."""
    print("Testing compute_merkle_root...")
    
    # Test with various sizes
    for size in [1, 2, 3, 4, 5, 8, 16, 100]:
        data_blocks = [f"block{i}".encode() for i in range(size)]
        
        # Compare with build_merkle_tree
        tree = build_merkle_tree(data_blocks)
        root_hash = compute_merkle_root(data_blocks)
        
        assert root_hash == tree.hash
    
    print("✓ compute_merkle_root passed")


def test_compute_merkle_root_empty():
    """Test compute_merkle_root with empty input."""
    print("Testing compute_merkle_root with empty input...")
    
    try:
        compute_merkle_root([])
        assert False, "Should raise ValueError for empty input"
    except ValueError as e:
        assert "empty data blocks" in str(e)
    
    print("✓ Empty input handling passed")


def test_generate_merkle_proof_single():
    """Test generating Merkle proof for single element tree."""
    print("Testing generate_merkle_proof for single element...")
    
    data_blocks = [b"single"]
    tree = build_merkle_tree(data_blocks)
    
    proof = generate_merkle_proof(tree, 0)
    assert len(proof) == 0  # No siblings for single element
    
    # Verify proof
    leaf_hash = hashlib.sha256(b"single").digest()
    root_hash = tree.hash
    assert verify_merkle_proof(leaf_hash, proof, root_hash)
    
    print("✓ Single element proof passed")


def test_generate_merkle_proof_two():
    """Test generating Merkle proof for two element tree."""
    print("Testing generate_merkle_proof for two elements...")
    
    data_blocks = [b"left", b"right"]
    tree = build_merkle_tree(data_blocks)
    
    # Proof for left element (index 0)
    proof_left = generate_merkle_proof(tree, 0)
    assert len(proof_left) == 1
    
    # Should contain right sibling
    right_hash = hashlib.sha256(b"right").digest()
    assert proof_left[0] == (right_hash, True)  # Right sibling
    
    # Verify proof
    left_hash = hashlib.sha256(b"left").digest()
    assert verify_merkle_proof(left_hash, proof_left, tree.hash)
    
    # Proof for right element (index 1)
    proof_right = generate_merkle_proof(tree, 1)
    assert len(proof_right) == 1
    
    # Should contain left sibling
    left_hash = hashlib.sha256(b"left").digest()
    assert proof_right[0] == (left_hash, False)  # Left sibling
    
    # Verify proof
    right_hash = hashlib.sha256(b"right").digest()
    assert verify_merkle_proof(right_hash, proof_right, tree.hash)
    
    print("✓ Two element proof passed")


def test_generate_merkle_proof_four():
    """Test generating Merkle proof for four element tree."""
    print("Testing generate_merkle_proof for four elements...")
    
    data_blocks = [f"block{i}".encode() for i in range(4)]
    tree = build_merkle_tree(data_blocks)
    
    # Test proof for each element
    for i in range(4):
        proof = generate_merkle_proof(tree, i)
        assert len(proof) == 2  # Two levels for 4 elements
        
        # Verify proof
        leaf_hash = hashlib.sha256(f"block{i}".encode()).digest()
        assert verify_merkle_proof(leaf_hash, proof, tree.hash)
    
    print("✓ Four element proof passed")


def test_generate_merkle_proof_invalid_index():
    """Test generating Merkle proof with invalid index."""
    print("Testing generate_merkle_proof with invalid index...")
    
    data_blocks = [b"block1", b"block2"]
    tree = build_merkle_tree(data_blocks)
    
    # Test negative index
    try:
        generate_merkle_proof(tree, -1)
        assert False, "Should raise ValueError for negative index"
    except ValueError as e:
        assert "Invalid leaf index" in str(e)
    
    # Test index too large
    try:
        generate_merkle_proof(tree, 2)
        assert False, "Should raise ValueError for index too large"
    except ValueError as e:
        assert "Invalid leaf index" in str(e)
    
    print("✓ Invalid index handling passed")


def test_verify_merkle_proof_comprehensive():
    """Test comprehensive Merkle proof verification."""
    print("Testing comprehensive Merkle proof verification...")
    
    # Test with various tree sizes
    for size in [1, 2, 3, 4, 5, 8, 16, 31, 32, 63, 64]:
        data_blocks = [f"data{i}".encode() for i in range(size)]
        tree = build_merkle_tree(data_blocks)
        
        # Test proof for each valid index
        for i in range(size):
            proof = generate_merkle_proof(tree, i)
            leaf_hash = hashlib.sha256(f"data{i}".encode()).digest()
            
            # Valid proof should verify
            assert verify_merkle_proof(leaf_hash, proof, tree.hash)
            
            # Invalid leaf hash should not verify
            wrong_hash = hashlib.sha256(f"wrong{i}".encode()).digest()
            assert not verify_merkle_proof(wrong_hash, proof, tree.hash)
            
            # Invalid root hash should not verify
            wrong_root = hashlib.sha256(b"wrong_root").digest()
            assert not verify_merkle_proof(leaf_hash, proof, wrong_root)
    
    print("✓ Comprehensive proof verification passed")


def test_verify_merkle_proof_tampered():
    """Test Merkle proof verification with tampered proofs."""
    print("Testing Merkle proof verification with tampered proofs...")
    
    data_blocks = [f"block{i}".encode() for i in range(8)]
    tree = build_merkle_tree(data_blocks)
    
    # Get valid proof
    proof = generate_merkle_proof(tree, 3)
    leaf_hash = hashlib.sha256(b"block3").digest()
    
    # Valid proof should work
    assert verify_merkle_proof(leaf_hash, proof, tree.hash)
    
    # Tamper with proof by changing a sibling hash
    if len(proof) > 0:
        tampered_proof = proof.copy()
        wrong_hash = hashlib.sha256(b"tampered").digest()
        tampered_proof[0] = (wrong_hash, proof[0][1])
        assert not verify_merkle_proof(leaf_hash, tampered_proof, tree.hash)
    
    # Tamper with proof by changing direction
    if len(proof) > 0:
        tampered_proof = proof.copy()
        tampered_proof[0] = (proof[0][0], not proof[0][1])
        assert not verify_merkle_proof(leaf_hash, tampered_proof, tree.hash)
    
    print("✓ Tampered proof detection passed")


def test_collect_leaves():
    """Test _collect_leaves helper function."""
    print("Testing _collect_leaves helper function...")
    
    # Test with various tree sizes
    for size in [1, 2, 3, 4, 8, 15, 16]:
        data_blocks = [f"leaf{i}".encode() for i in range(size)]
        tree = build_merkle_tree(data_blocks)
        
        leaves = _collect_leaves(tree)
        
        # Check that all original leaves are present
        for i in range(size):
            assert leaves[i].data == f"leaf{i}".encode()
        
        # For odd sizes, check duplication
        if size % 2 == 1 and size > 1:
            # Should have duplicated last element to make even
            expected_leaves = size
            while expected_leaves != 1 and expected_leaves % 2 == 1:
                expected_leaves += 1
            
            # Find next power of 2
            power_of_2 = 1
            while power_of_2 < expected_leaves:
                power_of_2 *= 2
            
            assert len(leaves) == power_of_2
    
    print("✓ _collect_leaves passed")


def test_get_subtree_size():
    """Test _get_subtree_size helper function."""
    print("Testing _get_subtree_size helper function...")
    
    # Test with various tree sizes
    for size in [1, 2, 4, 8, 16]:
        data_blocks = [f"node{i}".encode() for i in range(size)]
        tree = build_merkle_tree(data_blocks)
        
        # Root should contain all leaves (including duplicates)
        total_size = _get_subtree_size(tree)
        
        # For single element
        if size == 1:
            assert total_size == 1
        else:
            # Should be next power of 2 due to duplication
            power_of_2 = 1
            while power_of_2 < size:
                power_of_2 *= 2
            assert total_size == power_of_2
    
    print("✓ _get_subtree_size passed")


def test_merkle_tree_determinism():
    """Test that Merkle tree construction is deterministic."""
    print("Testing Merkle tree determinism...")
    
    data_blocks = [f"deterministic{i}".encode() for i in range(10)]
    
    # Build tree multiple times
    trees = [build_merkle_tree(data_blocks) for _ in range(5)]
    
    # All roots should have same hash
    root_hashes = [tree.hash for tree in trees]
    assert all(h == root_hashes[0] for h in root_hashes)
    
    # All proofs should be identical
    for i in range(len(data_blocks)):
        proofs = [generate_merkle_proof(tree, i) for tree in trees]
        assert all(p == proofs[0] for p in proofs)
    
    print("✓ Determinism passed")


def test_merkle_tree_edge_cases():
    """Test Merkle tree with edge cases."""
    print("Testing Merkle tree edge cases...")
    
    # Empty data blocks
    try:
        build_merkle_tree([])
        assert False, "Should raise ValueError for empty input"
    except ValueError:
        pass
    
    # Very large data blocks
    large_block = b"x" * 1000000  # 1MB
    tree = build_merkle_tree([large_block])
    assert tree.data == large_block
    
    # Data blocks with special characters
    special_blocks = [
        b"",  # Empty block
        b"\x00\x01\x02",  # Binary data
        "unicode_ñ_🚀".encode('utf-8'),  # Unicode
        b"\xff" * 100  # All 0xFF bytes
    ]
    tree = build_merkle_tree(special_blocks)
    
    # Verify all proofs work
    for i, block in enumerate(special_blocks):
        proof = generate_merkle_proof(tree, i)
        leaf_hash = hashlib.sha256(block).digest()
        assert verify_merkle_proof(leaf_hash, proof, tree.hash)
    
    print("✓ Edge cases passed")


def test_merkle_tree_performance():
    """Test Merkle tree performance with large datasets."""
    print("Testing Merkle tree performance...")
    
    sizes = [100, 1000, 10000]
    
    for size in sizes:
        print(f"  Testing with {size} elements...")
        
        # Generate test data
        data_blocks = [f"performance_test_{i}".encode() for i in range(size)]
        
        # Time tree construction
        start_time = time.time()
        tree = build_merkle_tree(data_blocks)
        build_time = time.time() - start_time
        
        print(f"    Build time: {build_time:.4f}s")
        
        # Time proof generation
        start_time = time.time()
        proof = generate_merkle_proof(tree, size // 2)
        proof_time = time.time() - start_time
        
        print(f"    Proof generation time: {proof_time:.4f}s")
        
        # Time proof verification
        leaf_hash = hashlib.sha256(f"performance_test_{size // 2}".encode()).digest()
        start_time = time.time()
        is_valid = verify_merkle_proof(leaf_hash, proof, tree.hash)
        verify_time = time.time() - start_time
        
        print(f"    Proof verification time: {verify_time:.4f}s")
        assert is_valid
        
        # Verify proof length is logarithmic
        expected_depth = size.bit_length() - 1 if size > 1 else 0
        actual_depth = len(proof)
        print(f"    Proof depth: {actual_depth} (expected ~{expected_depth})")
        
        # Performance should be reasonable
        assert build_time < 1.0, f"Build time too slow: {build_time}s"
        assert proof_time < 0.1, f"Proof generation too slow: {proof_time}s"
        assert verify_time < 0.01, f"Proof verification too slow: {verify_time}s"
    
    print("✓ Performance tests passed")


def test_merkle_tree_integration():
    """Test Merkle tree integration with real-world scenarios."""
    print("Testing Merkle tree integration scenarios...")
    
    # Simulate blockchain-like scenario
    transactions = []
    for i in range(50):
        tx = {
            'id': f"tx_{i}",
            'amount': random.randint(1, 1000),
            'timestamp': time.time() + i
        }
        tx_bytes = str(tx).encode()
        transactions.append(tx_bytes)
    
    # Build Merkle tree for transactions
    tree = build_merkle_tree(transactions)
    root_hash = tree.hash
    
    # Verify random transactions
    for _ in range(10):
        tx_index = random.randint(0, len(transactions) - 1)
        proof = generate_merkle_proof(tree, tx_index)
        leaf_hash = hashlib.sha256(transactions[tx_index]).digest()
        
        assert verify_merkle_proof(leaf_hash, proof, root_hash)
    
    # Simulate audit scenario: verify specific transaction exists
    target_tx = transactions[25]
    proof = generate_merkle_proof(tree, 25)
    leaf_hash = hashlib.sha256(target_tx).digest()
    
    # Someone else can verify without seeing all transactions
    assert verify_merkle_proof(leaf_hash, proof, root_hash)
    
    # Simulate tampering detection
    fake_tx = b"fake_transaction"
    fake_hash = hashlib.sha256(fake_tx).digest()
    assert not verify_merkle_proof(fake_hash, proof, root_hash)
    
    print("✓ Integration scenarios passed")


def run_all_tests():
    """Run all Merkle tree tests."""
    print("=" * 70)
    print("COMPREHENSIVE MERKLE TREE TEST SUITE")
    print("=" * 70)
    
    test_functions = [
        test_merkle_node_basic,
        test_build_merkle_tree_single,
        test_build_merkle_tree_two,
        test_build_merkle_tree_odd,
        test_build_merkle_tree_power_of_two,
        test_build_merkle_tree_large,
        test_compute_merkle_root,
        test_compute_merkle_root_empty,
        test_generate_merkle_proof_single,
        test_generate_merkle_proof_two,
        test_generate_merkle_proof_four,
        test_generate_merkle_proof_invalid_index,
        test_verify_merkle_proof_comprehensive,
        test_verify_merkle_proof_tampered,
        test_collect_leaves,
        test_get_subtree_size,
        test_merkle_tree_determinism,
        test_merkle_tree_edge_cases,
        test_merkle_tree_performance,
        test_merkle_tree_integration
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
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
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"❌ {failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)