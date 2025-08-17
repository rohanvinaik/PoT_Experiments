#!/usr/bin/env python3
"""
Detailed debug of four element Merkle tree.
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


def print_tree_structure(node, level=0, label="Root"):
    """Print tree structure recursively."""
    indent = "  " * level
    if node.is_leaf():
        print(f"{indent}{label}: LEAF {node.data} -> {node.hash.hex()}")
    else:
        print(f"{indent}{label}: INTERNAL -> {node.hash.hex()}")
        if node.left:
            print_tree_structure(node.left, level + 1, "L")
        if node.right:
            print_tree_structure(node.right, level + 1, "R")


def manual_verify_proof(leaf_hash, proof, root_hash):
    """Manually verify proof step by step."""
    print(f"Manual verification:")
    print(f"  Starting with leaf hash: {leaf_hash.hex()}")
    
    current_hash = leaf_hash
    for i, (sibling_hash, is_right) in enumerate(proof):
        print(f"  Step {i+1}:")
        print(f"    Current hash: {current_hash.hex()}")
        print(f"    Sibling hash: {sibling_hash.hex()}")
        print(f"    Sibling is right: {is_right}")
        
        if is_right:
            combined = current_hash + sibling_hash
            print(f"    Combining: current + sibling")
        else:
            combined = sibling_hash + current_hash
            print(f"    Combining: sibling + current")
        
        current_hash = hashlib.sha256(combined).digest()
        print(f"    Result: {current_hash.hex()}")
    
    print(f"  Final computed root: {current_hash.hex()}")
    print(f"  Expected root:       {root_hash.hex()}")
    print(f"  Match: {current_hash == root_hash}")
    
    return current_hash == root_hash


def debug_four_element_detailed():
    """Detailed debug of four element tree."""
    print("Detailed debug of four element tree...")
    
    data = [f"block{i}".encode() for i in range(4)]
    tree = build_merkle_tree(data)
    
    print("Data blocks:")
    for i, block in enumerate(data):
        print(f"  {i}: {block} -> {hashlib.sha256(block).hexdigest()}")
    
    print("\nTree structure:")
    print_tree_structure(tree)
    
    print("\nTesting proof for index 0 (block0):")
    proof = generate_merkle_proof(tree, 0)
    
    print(f"Generated proof: {[(h.hex(), is_right) for h, is_right in proof]}")
    
    leaf_hash = hashlib.sha256(b"block0").digest()
    manual_result = manual_verify_proof(leaf_hash, proof, tree.hash)
    
    auto_result = verify_merkle_proof(leaf_hash, proof, tree.hash)
    print(f"Auto verification result: {auto_result}")
    
    # Let's trace through the proof generation
    print("\nTracing proof generation:")
    print(f"Looking for index 0 in tree with {_get_subtree_size(tree)} leaves")
    
    current_node = tree
    current_index = 0
    step = 0
    
    while not current_node.is_leaf():
        step += 1
        left_size = _get_subtree_size(current_node.left)
        print(f"Step {step}: Left subtree size = {left_size}, current_index = {current_index}")
        
        if current_index < left_size:
            print(f"  Going left, sibling is right subtree: {current_node.right.hash.hex()}")
            current_node = current_node.left
        else:
            print(f"  Going right, sibling is left subtree: {current_node.left.hash.hex()}")
            current_node = current_node.right
            current_index -= left_size
    
    print(f"Final node data: {current_node.data}")


if __name__ == "__main__":
    debug_four_element_detailed()