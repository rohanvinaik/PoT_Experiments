#!/usr/bin/env python3
"""
Simple test for Merkle tree implementation to verify basic functionality.
"""

import sys
import os
import hashlib

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pot.prototypes.training_provenance_auditor import (
    MerkleNode, build_merkle_tree, compute_merkle_root,
    generate_merkle_proof, verify_merkle_proof
)


def test_basic_functionality():
    """Test basic Merkle tree functionality."""
    print("Testing basic Merkle tree functionality...")
    
    # Test 1: Single element
    data1 = [b"single"]
    tree1 = build_merkle_tree(data1)
    root1 = compute_merkle_root(data1)
    
    assert tree1.hash == root1
    assert tree1.is_leaf()
    
    # Proof for single element should be empty
    proof1 = generate_merkle_proof(tree1, 0)
    assert len(proof1) == 0
    
    # Verify proof
    leaf_hash1 = hashlib.sha256(b"single").digest()
    assert verify_merkle_proof(leaf_hash1, proof1, root1)
    
    print("✓ Single element test passed")
    
    # Test 2: Two elements
    data2 = [b"left", b"right"]
    tree2 = build_merkle_tree(data2)
    root2 = compute_merkle_root(data2)
    
    assert tree2.hash == root2
    assert not tree2.is_leaf()
    
    # Test proofs for both elements
    for i in range(2):
        proof = generate_merkle_proof(tree2, i)
        leaf_hash = hashlib.sha256(data2[i]).digest()
        assert verify_merkle_proof(leaf_hash, proof, root2)
    
    print("✓ Two element test passed")
    
    # Test 3: Four elements
    data4 = [f"block{i}".encode() for i in range(4)]
    tree4 = build_merkle_tree(data4)
    root4 = compute_merkle_root(data4)
    
    assert tree4.hash == root4
    
    # Test proofs for all elements
    for i in range(4):
        proof = generate_merkle_proof(tree4, i)
        leaf_hash = hashlib.sha256(data4[i]).digest()
        is_valid = verify_merkle_proof(leaf_hash, proof, root4)
        print(f"    Proof {i}: {'valid' if is_valid else 'INVALID'}")
        assert is_valid
    
    print("✓ Four element test passed")
    
    # Test 4: Invalid proofs should fail
    wrong_hash = hashlib.sha256(b"wrong").digest()
    proof = generate_merkle_proof(tree4, 0)
    assert not verify_merkle_proof(wrong_hash, proof, root4)
    
    wrong_root = hashlib.sha256(b"wrong_root").digest()
    leaf_hash = hashlib.sha256(b"block0").digest()
    assert not verify_merkle_proof(leaf_hash, proof, wrong_root)
    
    print("✓ Invalid proof detection passed")
    
    print("🎉 All basic tests passed!")


if __name__ == "__main__":
    test_basic_functionality()