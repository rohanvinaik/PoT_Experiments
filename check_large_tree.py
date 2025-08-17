#!/usr/bin/env python3
"""
Check what the large tree structure actually looks like.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pot.prototypes.training_provenance_auditor import build_merkle_tree, _collect_leaves

# Test with 100 elements
data_blocks = [f"block{i}".encode() for i in range(100)]
root = build_merkle_tree(data_blocks)

leaves = _collect_leaves(root)
print(f"Number of leaves: {len(leaves)}")
print(f"First 5 leaves: {[leaf.data for leaf in leaves[:5]]}")
print(f"Last 5 leaves: {[leaf.data for leaf in leaves[-5:]]}")

# Check duplicates
unique_count = 0
for i, leaf in enumerate(leaves):
    if i < 100:
        expected = f"block{i}".encode()
        if leaf.data == expected:
            unique_count += 1
        else:
            print(f"Mismatch at index {i}: expected {expected}, got {leaf.data}")

print(f"Unique blocks correctly placed: {unique_count}")

# Check what's beyond index 99
print(f"Block at index 100: {leaves[100].data if len(leaves) > 100 else 'N/A'}")
print(f"Block at index 127: {leaves[127].data if len(leaves) > 127 else 'N/A'}")