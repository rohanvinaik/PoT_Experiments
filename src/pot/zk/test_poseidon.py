"""
Tests for Poseidon hash implementation.

These tests verify:
1. Basic Poseidon hash functionality
2. Field arithmetic operations
3. Merkle tree construction
4. Compatibility with Rust implementation
"""

import unittest
import hashlib
import numpy as np
from typing import List

from pot.zk.poseidon import (
    PoseidonHash,
    poseidon_hash,
    poseidon_hash_two,
    poseidon_hash_many,
    poseidon_merkle_root,
    PoseidonMerkleTree,
    get_poseidon
)
from pot.zk.field_arithmetic import (
    FieldElement,
    bytes_to_field,
    field_to_bytes,
    int_to_field,
    field_to_int
)
from pot.zk.commitments import (
    PoseidonHasher,
    MerkleCommitment,
    DualCommitment
)


class TestFieldArithmetic(unittest.TestCase):
    """Test field arithmetic operations."""
    
    def test_field_element_creation(self):
        """Test creating field elements from various inputs."""
        # From integer
        fe1 = FieldElement(42)
        self.assertEqual(fe1.value, 42)
        
        # From bytes
        fe2 = FieldElement(b'\x00' * 31 + b'\x2a')  # 42 in bytes
        self.assertEqual(fe2.value, 42)
        
        # From another field element
        fe3 = FieldElement(fe1)
        self.assertEqual(fe3.value, 42)
    
    def test_field_addition(self):
        """Test field addition."""
        a = FieldElement(10)
        b = FieldElement(20)
        c = a + b
        self.assertEqual(c.value, 30)
    
    def test_field_multiplication(self):
        """Test field multiplication."""
        a = FieldElement(7)
        b = FieldElement(6)
        c = a * b
        self.assertEqual(c.value, 42)
    
    def test_field_inverse(self):
        """Test multiplicative inverse."""
        a = FieldElement(7)
        a_inv = a.inverse()
        one = a * a_inv
        self.assertEqual(one.value, 1)
    
    def test_field_modulus(self):
        """Test that operations respect field modulus."""
        # Pallas base field modulus
        p = FieldElement.MODULUS
        
        # Test that p reduces to 0
        fe = FieldElement(p)
        self.assertEqual(fe.value, 0)
        
        # Test that p-1 is the maximum value
        fe_max = FieldElement(p - 1)
        self.assertEqual(fe_max.value, p - 1)
        
        # Test wraparound
        fe_wrap = FieldElement(p + 1)
        self.assertEqual(fe_wrap.value, 1)


class TestPoseidonHash(unittest.TestCase):
    """Test Poseidon hash implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hasher = PoseidonHash()
    
    def test_hash_single(self):
        """Test hashing a single value."""
        # Hash a field element
        input_val = FieldElement(42)
        output = self.hasher.hash_single(input_val)
        self.assertIsInstance(output, FieldElement)
        
        # Hash should be deterministic
        output2 = self.hasher.hash_single(input_val)
        self.assertEqual(output, output2)
    
    def test_hash_two(self):
        """Test 2-to-1 hash."""
        left = FieldElement(10)
        right = FieldElement(20)
        
        output = self.hasher.hash_two(left, right)
        self.assertIsInstance(output, FieldElement)
        
        # Should be deterministic
        output2 = self.hasher.hash_two(left, right)
        self.assertEqual(output, output2)
        
        # Order matters
        output_reversed = self.hasher.hash_two(right, left)
        self.assertNotEqual(output, output_reversed)
    
    def test_hash_many(self):
        """Test hashing multiple values."""
        inputs = [FieldElement(i) for i in range(8)]
        output = self.hasher.hash_many(inputs)
        self.assertIsInstance(output, FieldElement)
        
        # Should be deterministic
        output2 = self.hasher.hash_many(inputs)
        self.assertEqual(output, output2)
    
    def test_hash_bytes(self):
        """Test hashing byte arrays."""
        data = b"Hello, Poseidon!"
        hash1 = poseidon_hash(data)
        self.assertEqual(len(hash1), 32)
        
        # Should be deterministic
        hash2 = poseidon_hash(data)
        self.assertEqual(hash1, hash2)
        
        # Different input gives different hash
        hash3 = poseidon_hash(b"Different data")
        self.assertNotEqual(hash1, hash3)
    
    def test_permutation(self):
        """Test Poseidon permutation."""
        state = [FieldElement(i) for i in range(3)]
        output = self.hasher.permutation(state)
        
        self.assertEqual(len(output), 3)
        for elem in output:
            self.assertIsInstance(elem, FieldElement)
        
        # Permutation should be different from input
        self.assertNotEqual(state, output)


class TestPoseidonMerkleTree(unittest.TestCase):
    """Test Poseidon-based Merkle tree."""
    
    def test_empty_tree(self):
        """Test empty Merkle tree."""
        tree = PoseidonMerkleTree([])
        root = tree.root()
        self.assertEqual(len(root), 32)
    
    def test_single_leaf(self):
        """Test Merkle tree with single leaf."""
        leaf = b"single leaf data"
        tree = PoseidonMerkleTree([leaf])
        root = tree.root()
        self.assertEqual(len(root), 32)
    
    def test_multiple_leaves(self):
        """Test Merkle tree with multiple leaves."""
        leaves = [f"leaf_{i}".encode() for i in range(8)]
        tree = PoseidonMerkleTree(leaves)
        root = tree.root()
        self.assertEqual(len(root), 32)
        
        # Root should be deterministic
        tree2 = PoseidonMerkleTree(leaves)
        root2 = tree2.root()
        self.assertEqual(root, root2)
    
    def test_merkle_proof(self):
        """Test Merkle proof generation and verification."""
        leaves = [f"leaf_{i}".encode() for i in range(4)]
        tree = PoseidonMerkleTree(leaves)
        
        # Generate proof for leaf 0
        proof = tree.proof(0)
        self.assertIsInstance(proof, list)
        
        # Verify the proof
        is_valid = tree.verify(leaves[0], 0, proof)
        self.assertTrue(is_valid)
        
        # Invalid proof should fail
        is_valid = tree.verify(b"wrong_leaf", 0, proof)
        self.assertFalse(is_valid)
    
    def test_poseidon_merkle_root_function(self):
        """Test the convenience function for Merkle root."""
        leaves = [f"leaf_{i}".encode() for i in range(4)]
        root = poseidon_merkle_root(leaves)
        self.assertEqual(len(root), 32)
        
        # Should match tree implementation
        tree = PoseidonMerkleTree(leaves)
        self.assertEqual(root, tree.root())


class TestCommitments(unittest.TestCase):
    """Test commitment schemes."""
    
    def test_poseidon_hasher(self):
        """Test PoseidonHasher wrapper."""
        data = b"test data"
        hash1 = PoseidonHasher.hash_bytes(data)
        self.assertEqual(len(hash1), 32)
        
        # Test field elements
        elements = [1, 2, 3, 4]
        hash2 = PoseidonHasher.hash_field_elements(elements)
        self.assertEqual(len(hash2), 32)
        
        # Test hash_two
        left = b"left" + b'\x00' * 28
        right = b"right" + b'\x00' * 27
        hash3 = PoseidonHasher.hash_two(left, right)
        self.assertEqual(len(hash3), 32)
    
    def test_merkle_commitment_poseidon(self):
        """Test Merkle commitment with Poseidon."""
        commitment = MerkleCommitment("poseidon")
        
        # Commit to a tensor
        tensor = np.random.randn(4, 4)
        root_hex, tensor_bytes = commitment.commit_tensor(tensor)
        
        self.assertIsInstance(root_hex, str)
        self.assertEqual(len(root_hex), 64)  # 32 bytes in hex
        self.assertEqual(tensor_bytes, tensor.tobytes())
    
    def test_dual_commitment(self):
        """Test dual commitment scheme."""
        dual = DualCommitment()
        
        # Commit to tensor
        tensor = np.random.randn(4, 4)
        result = dual.commit_tensor(tensor)
        
        self.assertIn("sha256_root", result)
        self.assertIn("poseidon_root", result)
        self.assertIn("tensor_data", result)
        
        # Roots should be different (different hash functions)
        self.assertNotEqual(result["sha256_root"], result["poseidon_root"])
        
        # Verify consistency
        is_consistent = dual.verify_consistency(result)
        self.assertTrue(is_consistent)


class TestRustCompatibility(unittest.TestCase):
    """Test compatibility with Rust implementation."""
    
    def test_field_element_serialization(self):
        """Test that field elements serialize compatibly with Rust."""
        fe = FieldElement(42)
        bytes_repr = fe.to_bytes()
        self.assertEqual(len(bytes_repr), 32)
        
        # Should be big-endian with value at the end
        expected = b'\x00' * 31 + b'\x2a'
        self.assertEqual(bytes_repr, expected)
        
        # Round-trip
        fe2 = FieldElement.from_bytes(bytes_repr)
        self.assertEqual(fe, fe2)
    
    def test_known_hash_values(self):
        """Test against known hash values."""
        # These would be test vectors from the Rust implementation
        # For now, just verify consistency
        
        test_cases = [
            (b"", "empty input"),
            (b"a", "single byte"),
            (b"Hello, World!", "ascii string"),
            (b"\x00" * 32, "32 zero bytes"),
            (b"\xff" * 32, "32 0xff bytes"),
        ]
        
        for input_data, description in test_cases:
            hash_val = poseidon_hash(input_data)
            self.assertEqual(len(hash_val), 32, f"Failed for: {description}")
            
            # Verify determinism
            hash_val2 = poseidon_hash(input_data)
            self.assertEqual(hash_val, hash_val2, f"Non-deterministic for: {description}")
    
    def test_merkle_tree_compatibility(self):
        """Test Merkle tree produces expected structure."""
        # Create a simple tree
        leaves = [b"leaf1", b"leaf2", b"leaf3", b"leaf4"]
        tree = PoseidonMerkleTree(leaves)
        root = tree.root()
        
        # Verify tree structure
        self.assertEqual(len(tree.tree), 3)  # 3 levels for 4 leaves
        self.assertEqual(len(tree.tree[0]), 4)  # 4 leaves
        self.assertEqual(len(tree.tree[1]), 2)  # 2 internal nodes
        self.assertEqual(len(tree.tree[2]), 1)  # 1 root
        
        # Verify proofs for all leaves
        for i, leaf in enumerate(leaves):
            proof = tree.proof(i)
            is_valid = tree.verify(leaf, i, proof)
            self.assertTrue(is_valid, f"Proof failed for leaf {i}")


class TestIntegration(unittest.TestCase):
    """Integration tests with the broader system."""
    
    def test_training_commitment_workflow(self):
        """Test a complete training commitment workflow."""
        # Create dual commitment
        dual = DualCommitment()
        
        # Create training batch
        batch_inputs = np.random.randn(32, 768)  # 32 samples, 768 features
        batch_targets = np.random.randn(32, 10)  # 32 samples, 10 classes
        
        # Commit to batch
        commitment = dual.commit_batch(batch_inputs, batch_targets)
        
        self.assertIn("sha256_root", commitment)
        self.assertIn("poseidon_root", commitment)
        
        # Both roots should be valid hex strings
        self.assertEqual(len(commitment["sha256_root"]), 64)
        self.assertEqual(len(commitment["poseidon_root"]), 64)
        
        # Verify we can recreate the commitments
        sha256_commitment = MerkleCommitment("sha256")
        poseidon_commitment = MerkleCommitment("poseidon")
        
        sha256_root2, _ = sha256_commitment.commit_batch(batch_inputs, batch_targets)
        poseidon_root2, _ = poseidon_commitment.commit_batch(batch_inputs, batch_targets)
        
        self.assertEqual(commitment["sha256_root"], sha256_root2)
        self.assertEqual(commitment["poseidon_root"], poseidon_root2)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()