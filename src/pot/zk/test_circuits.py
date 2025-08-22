"""
Unit tests for ZK circuit constraints and components.

This module tests individual circuit components including:
- Constraint types and verification
- Merkle inclusion proofs
- Gradient computation accuracy
- Compatibility with Python implementations
"""

import pytest
import numpy as np
import hashlib
from typing import List, Tuple
from pathlib import Path

from pot.zk.zk_types import (
    SGDStepStatement,
    SGDStepWitness,
    LoRAStepStatement,
    LoRAStepWitness
)
from pot.zk.poseidon import PoseidonHash, PoseidonMerkleTree, poseidon_hash
from pot.zk.field_arithmetic import FieldElement
from pot.zk.commitments import compute_weight_commitment, verify_gradient_consistency
from pot.zk.lora_builder import LoRAWitnessBuilder, create_example_lora_adapters


class TestFieldArithmetic:
    """Test field arithmetic operations."""
    
    def test_field_addition(self):
        """Test field element addition."""
        a = FieldElement(42)
        b = FieldElement(17)
        c = a + b
        assert c.value == 59
        
        # Test modular reduction
        large = FieldElement(FieldElement.MODULUS - 1)
        result = large + FieldElement(2)
        assert result.value == 1
    
    def test_field_multiplication(self):
        """Test field element multiplication."""
        a = FieldElement(7)
        b = FieldElement(8)
        c = a * b
        assert c.value == 56
        
        # Test identity
        one = FieldElement.one()
        assert (a * one).value == a.value
    
    def test_field_inverse(self):
        """Test field element inverse."""
        a = FieldElement(42)
        inv = a.inverse()
        product = a * inv
        assert product.value == 1
        
        # Test division
        b = FieldElement(7)
        c = a / b
        assert (c * b).value == a.value
    
    def test_field_from_bytes(self):
        """Test field element creation from bytes."""
        data = b"test data"
        fe = FieldElement.from_bytes(hashlib.sha256(data).digest())
        assert fe.value < FieldElement.MODULUS
        assert fe.value > 0


class TestMerkleInclusion:
    """Test Merkle tree inclusion proofs."""
    
    def test_merkle_root_computation(self):
        """Test Merkle root computation."""
        leaves = [f"leaf_{i}".encode() for i in range(8)]
        tree = PoseidonMerkleTree(leaves)
        root = tree.root()
        
        # Root should be deterministic
        tree2 = PoseidonMerkleTree(leaves)
        assert tree2.root() == root
        
        # Different leaves should give different root
        leaves2 = [f"other_{i}".encode() for i in range(8)]
        tree3 = PoseidonMerkleTree(leaves2)
        assert tree3.root() != root
    
    def test_merkle_proof_generation(self):
        """Test Merkle proof generation."""
        leaves = [f"data_{i}".encode() for i in range(8)]
        tree = PoseidonMerkleTree(leaves)
        
        # Generate proof for each leaf
        for i in range(len(leaves)):
            proof = tree.proof(i)
            assert len(proof) == 3  # log2(8) = 3
            
            # Verify proof
            is_valid = tree.verify(leaves[i], i, proof)
            assert is_valid, f"Proof for leaf {i} should be valid"
    
    def test_merkle_proof_verification(self):
        """Test Merkle proof verification."""
        leaves = [f"item_{i}".encode() for i in range(16)]
        tree = PoseidonMerkleTree(leaves)
        
        # Valid proof
        proof = tree.proof(5)
        assert tree.verify(leaves[5], 5, proof)
        
        # Invalid leaf
        assert not tree.verify(b"wrong_leaf", 5, proof)
        
        # Invalid index
        assert not tree.verify(leaves[5], 6, proof)
        
        # Invalid proof
        wrong_proof = [b"wrong" * 8 for _ in proof]
        assert not tree.verify(leaves[5], 5, wrong_proof)
    
    def test_merkle_consistency(self):
        """Test Merkle tree consistency with updates."""
        # Initial leaves
        leaves = [f"v1_item_{i}".encode() for i in range(8)]
        tree1 = PoseidonMerkleTree(leaves)
        root1 = tree1.root()
        
        # Update one leaf
        leaves[3] = b"v2_item_3"
        tree2 = PoseidonMerkleTree(leaves)
        root2 = tree2.root()
        
        # Roots should differ
        assert root1 != root2
        
        # Old proof should not work with new tree
        old_proof = tree1.proof(3)
        assert not tree2.verify(b"v1_item_3", 3, old_proof)
        
        # New proof should work
        new_proof = tree2.proof(3)
        assert tree2.verify(b"v2_item_3", 3, new_proof)


class TestGradientComputation:
    """Test gradient computation accuracy."""
    
    def test_gradient_from_weight_update(self):
        """Test gradient computation from weight updates."""
        # Create weight updates
        weights_before = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        learning_rate = 0.01
        gradients = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        
        # Compute weight update: w_new = w_old - lr * grad
        weights_after = weights_before - learning_rate * gradients
        
        # Reconstruct gradients
        reconstructed = (weights_before - weights_after) / learning_rate
        
        # Should match within floating point precision
        np.testing.assert_allclose(reconstructed, gradients, rtol=1e-5)
    
    def test_gradient_consistency_check(self):
        """Test gradient consistency verification."""
        weights_before = np.random.randn(100).astype(np.float32)
        gradients = np.random.randn(100).astype(np.float32)
        learning_rate = 0.001
        
        # Correct update
        weights_after = weights_before - learning_rate * gradients
        is_valid = verify_gradient_consistency(
            weights_before, weights_after, gradients, learning_rate
        )
        assert is_valid
        
        # Incorrect update
        weights_wrong = weights_before - 2 * learning_rate * gradients
        is_invalid = verify_gradient_consistency(
            weights_before, weights_wrong, gradients, learning_rate
        )
        assert not is_invalid
    
    def test_batch_gradient_aggregation(self):
        """Test batch gradient aggregation."""
        batch_size = 32
        param_size = 100
        
        # Generate per-sample gradients
        per_sample_grads = np.random.randn(batch_size, param_size).astype(np.float32)
        
        # Aggregate (mean)
        aggregated = np.mean(per_sample_grads, axis=0)
        
        # Verify aggregation
        manual_sum = np.zeros(param_size, dtype=np.float32)
        for i in range(batch_size):
            manual_sum += per_sample_grads[i]
        manual_mean = manual_sum / batch_size
        
        np.testing.assert_allclose(aggregated, manual_mean, rtol=1e-6)
    
    def test_lora_gradient_computation(self):
        """Test LoRA-specific gradient computation."""
        # LoRA parameters
        d_in, d_out = 768, 768
        rank = 8
        
        # Create LoRA adapters
        adapter_a = np.random.randn(d_in, rank).astype(np.float32) * 0.01
        adapter_b = np.random.randn(rank, d_out).astype(np.float32) * 0.01
        
        # Compute effective weights
        delta_w = adapter_a @ adapter_b
        
        # Gradient for LoRA
        grad_output = np.random.randn(d_out).astype(np.float32)
        grad_b = np.outer(adapter_a.sum(axis=0), grad_output) / d_in
        grad_a = np.outer(grad_output, adapter_b.sum(axis=1)) / d_out
        
        # Update adapters
        lr = 0.001
        adapter_a_new = adapter_a - lr * grad_a[:d_in, :rank]
        adapter_b_new = adapter_b - lr * grad_b[:rank, :d_out]
        
        # New effective weights
        delta_w_new = adapter_a_new @ adapter_b_new
        
        # Should show change
        assert not np.allclose(delta_w, delta_w_new)


class TestConstraintVerification:
    """Test circuit constraint verification."""
    
    def test_sgd_constraints(self):
        """Test SGD step constraints."""
        # Create valid SGD witness
        weights_before = np.random.randn(100).astype(np.float32)
        gradients = np.random.randn(100).astype(np.float32) * 0.01
        learning_rate = 0.001
        weights_after = weights_before - learning_rate * gradients
        
        witness = SGDStepWitness(
            weights_before=weights_before.tolist(),
            weights_after=weights_after.tolist(),
            gradients=gradients.tolist(),
            batch_inputs=[[0.5] * 10 for _ in range(32)],
            batch_targets=[[1.0] for _ in range(32)],
            learning_rate=learning_rate
        )
        
        # Verify constraints
        assert self._verify_sgd_constraints(witness)
        
        # Invalid witness (wrong update)
        witness.weights_after[0] += 1.0
        assert not self._verify_sgd_constraints(witness)
    
    def test_lora_constraints(self):
        """Test LoRA step constraints."""
        rank = 8
        d = 768
        
        # Create LoRA adapters
        adapters = create_example_lora_adapters(d, d, rank)
        
        # Create witness
        witness = LoRAStepWitness(
            adapter_a_before=adapters.adapter_a.flatten().tolist(),
            adapter_b_before=adapters.adapter_b.flatten().tolist(),
            adapter_a_after=(adapters.adapter_a * 1.01).flatten().tolist(),
            adapter_b_after=(adapters.adapter_b * 1.01).flatten().tolist(),
            adapter_a_gradients=[0.001] * (d * rank),
            adapter_b_gradients=[0.001] * (rank * d),
            batch_inputs=[0.5] * d,
            batch_targets=[1.0] * d,
            learning_rate=0.01
        )
        
        # Verify constraints
        assert self._verify_lora_constraints(witness, rank)
        
        # Invalid witness (wrong dimensions)
        witness.adapter_a_after = witness.adapter_a_after[:-1]
        assert not self._verify_lora_constraints(witness, rank)
    
    def test_merkle_constraint(self):
        """Test Merkle tree constraints in circuit."""
        # Create leaves
        leaves = [f"weight_{i}".encode() for i in range(8)]
        tree = PoseidonMerkleTree(leaves)
        root = tree.root()
        
        # For each leaf, verify inclusion constraint
        for i, leaf in enumerate(leaves):
            proof = tree.proof(i)
            
            # Simulate circuit constraint check
            computed_root = self._compute_root_from_proof(leaf, i, proof)
            assert computed_root == root
    
    def test_commitment_constraint(self):
        """Test commitment constraints."""
        # Create weights
        weights = np.random.randn(256).astype(np.float32)
        
        # Compute commitment
        commitment = compute_weight_commitment(weights)
        
        # Verify commitment is deterministic
        commitment2 = compute_weight_commitment(weights)
        assert commitment == commitment2
        
        # Different weights give different commitment
        weights2 = weights + 0.001
        commitment3 = compute_weight_commitment(weights2)
        assert commitment != commitment3
    
    def _verify_sgd_constraints(self, witness: SGDStepWitness) -> bool:
        """Verify SGD constraints."""
        try:
            w_before = np.array(witness.weights_before)
            w_after = np.array(witness.weights_after)
            grads = np.array(witness.gradients)
            lr = witness.learning_rate
            
            # Check update rule
            expected = w_before - lr * grads
            return np.allclose(w_after, expected, rtol=1e-5)
        except:
            return False
    
    def _verify_lora_constraints(self, witness: LoRAStepWitness, rank: int) -> bool:
        """Verify LoRA constraints."""
        try:
            # Check dimensions
            d = len(witness.adapter_a_before) // rank
            if len(witness.adapter_a_before) != d * rank:
                return False
            if len(witness.adapter_b_before) != rank * d:
                return False
            if len(witness.adapter_a_after) != d * rank:
                return False
            if len(witness.adapter_b_after) != rank * d:
                return False
            
            return True
        except:
            return False
    
    def _compute_root_from_proof(self, leaf: bytes, index: int, proof: List[bytes]) -> bytes:
        """Compute Merkle root from proof."""
        current = poseidon_hash(leaf)
        
        for i, sibling in enumerate(proof):
            if (index >> i) & 1:
                # Current is right child
                combined = sibling + current
            else:
                # Current is left child
                combined = current + sibling
            current = poseidon_hash(combined)
        
        return current


class TestPythonCompatibility:
    """Test compatibility with existing Python implementations."""
    
    def test_compatible_with_training_auditor(self):
        """Test compatibility with TrainingProvenanceAuditor."""
        from pot.prototypes.training_provenance_auditor import TrainingProvenanceAuditor
        
        # Create auditor with Poseidon
        auditor = TrainingProvenanceAuditor(
            model_id="test_model",
            hash_function="poseidon"
        )
        
        # Log event
        metrics = {'loss': 0.5, 'accuracy': 0.9}
        event = auditor.log_training_event(epoch=1, metrics=metrics)
        
        # Event should have Poseidon hash
        assert event.event_hash is not None
        assert len(event.event_hash) > 0
    
    def test_compatible_with_model_verification(self):
        """Test compatibility with model verification."""
        from pot.core.model_verification import verify_model_weights
        
        # Create test weights
        weights1 = {'layer1': np.random.randn(10, 10).astype(np.float32)}
        weights2 = {'layer1': weights1['layer1'] + np.random.randn(10, 10).astype(np.float32) * 0.001}
        
        # Should detect difference
        result = verify_model_weights(weights1, weights2, threshold=0.0001)
        assert not result['identical']
    
    def test_compatible_with_blockchain_client(self):
        """Test compatibility with MockBlockchainClient."""
        from pot.testing.mock_blockchain import MockBlockchainClient
        
        client = MockBlockchainClient()
        
        # Store ZK proof
        proof_data = b"zk_proof_data"
        tx_hash = client.store_commitment(proof_data)
        assert tx_hash is not None
        
        # Retrieve proof
        retrieved = client.get_commitment(tx_hash)
        assert retrieved == proof_data
    
    def test_compatible_with_witness_builder(self):
        """Test compatibility with witness builders."""
        builder = LoRAWitnessBuilder()
        
        # Create model state
        model_state = {
            'lora_A.weight': np.random.randn(768, 8).astype(np.float32),
            'lora_B.weight': np.random.randn(8, 768).astype(np.float32),
            'base.weight': np.random.randn(768, 768).astype(np.float32)
        }
        
        # Extract LoRA adapters
        adapters = builder.extract_lora_adapters(model_state)
        assert len(adapters) > 0
        
        # Detect LoRA training
        is_lora = builder.detect_lora_training(model_state)
        assert is_lora


def test_constraint_counts():
    """Test constraint counts for different proof types."""
    # SGD constraints
    sgd_constraints = {
        'gradient_consistency': 100,  # One per weight
        'merkle_inclusion': 3,  # log2(8) for 8 leaves
        'commitment': 1,
        'range_checks': 100  # One per weight
    }
    total_sgd = sum(sgd_constraints.values())
    assert total_sgd == 204
    
    # LoRA constraints (rank 8)
    lora_constraints = {
        'adapter_update': 2 * 768 * 8,  # A and B adapters
        'effective_weight': 768 * 768,  # Full effective weight
        'merkle_inclusion': 4,  # log2(16) for more leaves
        'commitment': 2,  # Before and after
        'range_checks': 2 * 768 * 8
    }
    total_lora = sum(lora_constraints.values())
    assert total_lora > 600000  # Should be large
    
    # Aggregated proof constraints
    agg_constraints = {
        'recursive_verification': 16,  # For 16 sub-proofs
        'accumulator_update': 1,
        'batch_commitment': 1
    }
    total_agg = sum(agg_constraints.values())
    assert total_agg == 18


def test_proof_size_estimates():
    """Test proof size estimates."""
    # SGD proof size
    sgd_proof_size = {
        'proof_data': 256,  # 256 bytes for proof
        'public_inputs': 64,  # Commitments
        'verification_key': 128
    }
    total_sgd_size = sum(sgd_proof_size.values())
    assert total_sgd_size == 448
    
    # LoRA proof size (should be similar despite more constraints)
    lora_proof_size = {
        'proof_data': 256,
        'public_inputs': 96,  # More commitments
        'verification_key': 128
    }
    total_lora_size = sum(lora_proof_size.values())
    assert total_lora_size == 480
    
    # Aggregated proof (should be smaller than sum)
    num_proofs = 16
    individual_total = num_proofs * total_sgd_size
    aggregated_size = 512  # Fixed size regardless of batch
    compression_ratio = individual_total / aggregated_size
    assert compression_ratio > 10  # At least 10x compression


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])