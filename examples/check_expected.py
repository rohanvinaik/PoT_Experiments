#!/usr/bin/env python3
"""
Check what the expected proof should be.
"""

import hashlib

# Expected computation for block0 proof:
block0_hash = bytes.fromhex('3da2892d37823d9298e1d5011d7dcfaaf2d9d9a6d465e99be33af5be1d87c12b')
block1_hash = bytes.fromhex('9a59c5f8229aab55e9f855173ef94485aab8497eea0588f365c871d6d0561722')
right_subtree_hash = bytes.fromhex('e552509f927b6a1e352a66ecf16f77a841e50381a21939a76c6a12dd2d0b9328')
expected_root = bytes.fromhex('cc1b728b558eb8e0a3831fa5f1465881ec944c1ea3504ea5b07ae61f67795175')

print("Expected computation:")
print(f"block0: {block0_hash.hex()}")
print(f"block1: {block1_hash.hex()}")

# Step 1: Combine block0 + block1 to get left subtree
left_subtree_computed = hashlib.sha256(block0_hash + block1_hash).digest()
print(f"Left subtree (block0 + block1): {left_subtree_computed.hex()}")

# Step 2: Combine left subtree + right subtree to get root
root_computed = hashlib.sha256(left_subtree_computed + right_subtree_hash).digest()
print(f"Root (left + right): {root_computed.hex()}")
print(f"Expected root:       {expected_root.hex()}")
print(f"Match: {root_computed == expected_root}")

print("\nCorrect proof for block0 should be:")
print(f"1. (block1_hash={block1_hash.hex()}, is_right=True)")
print(f"2. (right_subtree_hash={right_subtree_hash.hex()}, is_right=True)")

# Verify this proof
current = block0_hash
print(f"\nVerification:")
print(f"Start: {current.hex()}")

# Step 1: current + block1
current = hashlib.sha256(current + block1_hash).digest()
print(f"After step 1 (current + block1): {current.hex()}")

# Step 2: current + right_subtree
current = hashlib.sha256(current + right_subtree_hash).digest()
print(f"After step 2 (current + right): {current.hex()}")
print(f"Matches root: {current == expected_root}")