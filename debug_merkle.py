#!/usr/bin/env python3
"""
Debug Merkle tree implementation.
"""

import sys
import os
import hashlib

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pot.prototypes.training_provenance_auditor import (
    MerkleNode, build_merkle_tree, compute_merkle_root,
    generate_merkle_proof, verify_merkle_proof, _collect_leaves, _get_subtree_size
)


def debug_two_element_tree():
    """Debug two element tree."""
    print("Debugging two element tree...")
    
    data = [b"left", b"right"]
    tree = build_merkle_tree(data)
    
    print(f"Root hash: {tree.hash.hex()}")
    print(f"Root is leaf: {tree.is_leaf()}")
    print(f"Left child hash: {tree.left.hash.hex()}")
    print(f"Right child hash: {tree.right.hash.hex()}")
    print(f"Left data: {tree.left.data}")
    print(f"Right data: {tree.right.data}")
    
    # Generate proof for index 0
    proof = generate_merkle_proof(tree, 0)
    print(f"Proof for index 0: {[(h.hex(), is_right) for h, is_right in proof]}")
    
    # Manually verify
    leaf_hash = hashlib.sha256(b"left").digest()
    print(f"Leaf hash: {leaf_hash.hex()}")
    
    current_hash = leaf_hash
    for sibling_hash, is_right in proof:
        print(f"Combining {current_hash.hex()} with sibling {sibling_hash.hex()} (is_right={is_right})")
        if is_right:
            combined = current_hash + sibling_hash
        else:
            combined = sibling_hash + current_hash
        current_hash = hashlib.sha256(combined).digest()
        print(f"Result: {current_hash.hex()}")
    
    print(f"Final hash: {current_hash.hex()}")
    print(f"Root hash:  {tree.hash.hex()}")
    print(f"Match: {current_hash == tree.hash}")
    
    # Test verification function
    is_valid = verify_merkle_proof(leaf_hash, proof, tree.hash)
    print(f"verify_merkle_proof result: {is_valid}")


def debug_four_element_tree():
    """Debug four element tree."""
    print("\nDebugging four element tree...")
    
    data = [f"block{i}".encode() for i in range(4)]
    tree = build_merkle_tree(data)
    
    print("Tree structure:")
    print(f"Root: {tree.hash.hex()}")
    print(f"Left subtree: {tree.left.hash.hex()}")
    print(f"Right subtree: {tree.right.hash.hex()}")
    
    leaves = _collect_leaves(tree)
    print(f"Leaves: {[leaf.data for leaf in leaves]}")
    
    # Test proof for index 0
    proof = generate_merkle_proof(tree, 0)
    print(f"Proof for index 0: {[(h.hex(), is_right) for h, is_right in proof]}")
    
    leaf_hash = hashlib.sha256(b"block0").digest()
    is_valid = verify_merkle_proof(leaf_hash, proof, tree.hash)
    print(f"Proof 0 valid: {is_valid}")


if __name__ == "__main__":
    debug_two_element_tree()
    debug_four_element_tree()