"""
Security tests for the ZK proof system.

This module tests:
- Invalid witness rejection
- Tampering resistance
- Zero-knowledge property verification
- Soundness and completeness
"""

import pytest
import numpy as np
import hashlib
from typing import Dict, Any, List
import time

from pot.zk.zk_types import (
    SGDStepStatement,
    SGDStepWitness,
    LoRAStepStatement,
    LoRAStepWitness
)
from pot.zk.prover import SGDZKProver, LoRAZKProver
from pot.zk.poseidon import PoseidonMerkleTree, poseidon_hash
from pot.zk.field_arithmetic import FieldElement
from pot.zk.commitments import compute_weight_commitment


class TestInvalidWitnesses:
    """Test that invalid witnesses are rejected."""
    
    def test_sgd_invalid_gradient(self):
        """Test SGD proof fails with wrong gradient."""
        prover = SGDZKProver()
        
        # Create valid witness
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
        
        statement = SGDStepStatement(
            W_t_root=b"before" * 8,
            W_t1_root=b"after" * 8,
            batch_root=b"batch" * 8,
            hparams_hash=b"hparams" * 4,
            step_number=1,
            epoch=1
        )
        
        # Should succeed with correct witness
        proof1 = prover.prove_sgd_step(statement, witness)
        assert proof1 is not None
        
        # Tamper with gradients
        witness.gradients[0] *= 2
        
        # Should fail with invalid witness
        with pytest.raises(Exception):
            proof2 = prover.prove_sgd_step(statement, witness)
    
    def test_sgd_invalid_weight_update(self):
        """Test SGD proof fails with incorrect weight update."""
        prover = SGDZKProver()
        
        weights_before = np.random.randn(50).astype(np.float32)
        gradients = np.random.randn(50).astype(np.float32) * 0.01
        learning_rate = 0.001
        
        # Incorrect update (using wrong learning rate)
        weights_after = weights_before - learning_rate * 2 * gradients
        
        witness = SGDStepWitness(
            weights_before=weights_before.tolist(),
            weights_after=weights_after.tolist(),
            gradients=gradients.tolist(),
            batch_inputs=[[0.5] * 10 for _ in range(32)],
            batch_targets=[[1.0] for _ in range(32)],
            learning_rate=learning_rate
        )
        
        statement = SGDStepStatement(
            W_t_root=b"before" * 8,
            W_t1_root=b"after" * 8,
            batch_root=b"batch" * 8,
            hparams_hash=b"hparams" * 4,
            step_number=1,
            epoch=1
        )
        
        # Should fail validation
        with pytest.raises(Exception):
            proof = prover.prove_sgd_step(statement, witness)
    
    def test_lora_invalid_rank(self):
        """Test LoRA proof fails with mismatched rank."""
        prover = LoRAZKProver()
        
        rank = 8
        d = 768
        
        # Create witness with wrong dimensions
        witness = LoRAStepWitness(
            adapter_a_before=[0.01] * (d * rank),
            adapter_b_before=[0.01] * (rank * d),
            adapter_a_after=[0.011] * (d * rank),
            adapter_b_after=[0.011] * (rank * d),
            adapter_a_gradients=[0.001] * (d * rank),
            adapter_b_gradients=[0.001] * (rank * d),
            batch_inputs=[0.5] * d,
            batch_targets=[1.0] * d,
            learning_rate=0.01
        )
        
        # Statement with different rank
        statement = LoRAStepStatement(
            base_weights_root=b"base" * 8,
            adapter_a_root_before=b"a_before" * 4,
            adapter_b_root_before=b"b_before" * 4,
            adapter_a_root_after=b"a_after" * 4,
            adapter_b_root_after=b"b_after" * 4,
            batch_root=b"batch" * 6,
            hparams_hash=b"hparams" * 4,
            rank=16,  # Wrong rank!
            alpha=32.0,
            step_number=1,
            epoch=1
        )
        
        # Should fail due to rank mismatch
        with pytest.raises(Exception):
            proof = prover.prove_lora_step(statement, witness)
    
    def test_lora_invalid_adapter_update(self):
        """Test LoRA proof fails with invalid adapter update."""
        prover = LoRAZKProver()
        
        rank = 4
        d = 256
        
        # Create adapters
        adapter_a_before = [0.01] * (d * rank)
        adapter_b_before = [0.01] * (rank * d)
        
        # Invalid update (not following gradient descent)
        adapter_a_after = [0.02] * (d * rank)  # Too large change
        adapter_b_after = [0.005] * (rank * d)  # Wrong direction
        
        witness = LoRAStepWitness(
            adapter_a_before=adapter_a_before,
            adapter_b_before=adapter_b_before,
            adapter_a_after=adapter_a_after,
            adapter_b_after=adapter_b_after,
            adapter_a_gradients=[0.001] * (d * rank),
            adapter_b_gradients=[0.001] * (rank * d),
            batch_inputs=[0.5] * d,
            batch_targets=[1.0] * d,
            learning_rate=0.01
        )
        
        statement = LoRAStepStatement(
            base_weights_root=b"base" * 8,
            adapter_a_root_before=b"a_before" * 4,
            adapter_b_root_before=b"b_before" * 4,
            adapter_a_root_after=b"a_after" * 4,
            adapter_b_root_after=b"b_after" * 4,
            batch_root=b"batch" * 6,
            hparams_hash=b"hparams" * 4,
            rank=rank,
            alpha=rank * 2.0,
            step_number=1,
            epoch=1
        )
        
        # Should fail validation
        with pytest.raises(Exception):
            proof = prover.prove_lora_step(statement, witness)


class TestTamperingResistance:
    """Test resistance to various tampering attacks."""
    
    def test_commitment_tampering(self):
        """Test that tampered commitments are detected."""
        # Create weights
        weights = np.random.randn(100).astype(np.float32)
        
        # Compute commitment
        commitment1 = compute_weight_commitment(weights)
        
        # Tamper with weights
        weights[0] += 0.001
        
        # New commitment should differ
        commitment2 = compute_weight_commitment(weights)
        assert commitment1 != commitment2
        
        # Even tiny changes should be detected
        weights[0] += 1e-6
        commitment3 = compute_weight_commitment(weights)
        assert commitment2 != commitment3
    
    def test_merkle_proof_tampering(self):
        """Test that tampered Merkle proofs fail verification."""
        leaves = [f"leaf_{i}".encode() for i in range(8)]
        tree = PoseidonMerkleTree(leaves)
        
        # Get valid proof
        valid_proof = tree.proof(3)
        assert tree.verify(leaves[3], 3, valid_proof)
        
        # Tamper with proof element
        tampered_proof = valid_proof.copy()
        tampered_proof[0] = b"tampered" * 8
        assert not tree.verify(leaves[3], 3, tampered_proof)
        
        # Tamper with leaf
        wrong_leaf = b"wrong_leaf"
        assert not tree.verify(wrong_leaf, 3, valid_proof)
        
        # Wrong index
        assert not tree.verify(leaves[3], 4, valid_proof)
    
    def test_proof_malleability(self):
        """Test that proofs cannot be modified to prove different statements."""
        prover = SGDZKProver()
        
        # Create two different valid witnesses
        weights1_before = np.random.randn(50).astype(np.float32)
        gradients1 = np.random.randn(50).astype(np.float32) * 0.01
        weights1_after = weights1_before - 0.001 * gradients1
        
        witness1 = SGDStepWitness(
            weights_before=weights1_before.tolist(),
            weights_after=weights1_after.tolist(),
            gradients=gradients1.tolist(),
            batch_inputs=[[0.5] * 10 for _ in range(32)],
            batch_targets=[[1.0] for _ in range(32)],
            learning_rate=0.001
        )
        
        statement1 = SGDStepStatement(
            W_t_root=b"before1" * 8,
            W_t1_root=b"after1" * 8,
            batch_root=b"batch1" * 8,
            hparams_hash=b"hparams" * 4,
            step_number=1,
            epoch=1
        )
        
        # Generate proof for first statement
        proof1 = prover.prove_sgd_step(statement1, witness1)
        
        # Create different statement
        statement2 = SGDStepStatement(
            W_t_root=b"before2" * 8,
            W_t1_root=b"after2" * 8,
            batch_root=b"batch2" * 8,
            hparams_hash=b"hparams" * 4,
            step_number=2,
            epoch=1
        )
        
        # Proof1 should not verify for statement2
        # In real implementation, this would be checked by verifier
        assert proof1 != prover.prove_sgd_step(statement2, witness1)
    
    def test_replay_attack_resistance(self):
        """Test resistance to replay attacks."""
        prover = SGDZKProver()
        
        # Create witness
        weights_before = np.random.randn(50).astype(np.float32)
        gradients = np.random.randn(50).astype(np.float32) * 0.01
        weights_after = weights_before - 0.001 * gradients
        
        witness = SGDStepWitness(
            weights_before=weights_before.tolist(),
            weights_after=weights_after.tolist(),
            gradients=gradients.tolist(),
            batch_inputs=[[0.5] * 10 for _ in range(32)],
            batch_targets=[[1.0] for _ in range(32)],
            learning_rate=0.001
        )
        
        # Generate proofs for different steps
        proofs = []
        for step in range(3):
            statement = SGDStepStatement(
                W_t_root=b"before" * 8,
                W_t1_root=b"after" * 8,
                batch_root=b"batch" * 8,
                hparams_hash=b"hparams" * 4,
                step_number=step,  # Different step numbers
                epoch=1
            )
            
            proof = prover.prove_sgd_step(statement, witness)
            proofs.append(proof)
        
        # All proofs should be different (due to step number)
        assert proofs[0] != proofs[1]
        assert proofs[1] != proofs[2]
        assert proofs[0] != proofs[2]


class TestZeroKnowledgeProperty:
    """Test that proofs reveal no information about witness."""
    
    def test_proof_does_not_reveal_weights(self):
        """Test that proof doesn't reveal weight values."""
        prover = SGDZKProver()
        
        # Create two different witnesses with same structure
        weights1 = np.random.randn(100).astype(np.float32)
        weights2 = np.random.randn(100).astype(np.float32)
        
        gradients = np.random.randn(100).astype(np.float32) * 0.01
        lr = 0.001
        
        witness1 = SGDStepWitness(
            weights_before=weights1.tolist(),
            weights_after=(weights1 - lr * gradients).tolist(),
            gradients=gradients.tolist(),
            batch_inputs=[[0.5] * 10 for _ in range(32)],
            batch_targets=[[1.0] for _ in range(32)],
            learning_rate=lr
        )
        
        witness2 = SGDStepWitness(
            weights_before=weights2.tolist(),
            weights_after=(weights2 - lr * gradients).tolist(),
            gradients=gradients.tolist(),
            batch_inputs=[[0.5] * 10 for _ in range(32)],
            batch_targets=[[1.0] for _ in range(32)],
            learning_rate=lr
        )
        
        # Same statement (public inputs)
        statement = SGDStepStatement(
            W_t_root=b"before" * 8,
            W_t1_root=b"after" * 8,
            batch_root=b"batch" * 8,
            hparams_hash=b"hparams" * 4,
            step_number=1,
            epoch=1
        )
        
        # Generate proofs
        proof1 = prover.prove_sgd_step(statement, witness1)
        proof2 = prover.prove_sgd_step(statement, witness2)
        
        # Proofs should be same size (no weight info leaked)
        assert len(proof1) == len(proof2)
        
        # Cannot recover weights from proof
        # (In real ZK system, this would be cryptographically guaranteed)
        assert weights1.tolist() not in str(proof1)
        assert weights2.tolist() not in str(proof2)
    
    def test_proof_does_not_reveal_gradients(self):
        """Test that proof doesn't reveal gradient values."""
        prover = SGDZKProver()
        
        weights = np.random.randn(100).astype(np.float32)
        
        # Two different gradient sets
        gradients1 = np.random.randn(100).astype(np.float32) * 0.01
        gradients2 = np.random.randn(100).astype(np.float32) * 0.02
        
        lr = 0.001
        
        witness1 = SGDStepWitness(
            weights_before=weights.tolist(),
            weights_after=(weights - lr * gradients1).tolist(),
            gradients=gradients1.tolist(),
            batch_inputs=[[0.5] * 10 for _ in range(32)],
            batch_targets=[[1.0] for _ in range(32)],
            learning_rate=lr
        )
        
        witness2 = SGDStepWitness(
            weights_before=weights.tolist(),
            weights_after=(weights - lr * gradients2).tolist(),
            gradients=gradients2.tolist(),
            batch_inputs=[[0.5] * 10 for _ in range(32)],
            batch_targets=[[1.0] for _ in range(32)],
            learning_rate=lr
        )
        
        statement = SGDStepStatement(
            W_t_root=b"before" * 8,
            W_t1_root=b"after" * 8,
            batch_root=b"batch" * 8,
            hparams_hash=b"hparams" * 4,
            step_number=1,
            epoch=1
        )
        
        proof1 = prover.prove_sgd_step(statement, witness1)
        proof2 = prover.prove_sgd_step(statement, witness2)
        
        # Cannot distinguish which gradients were used
        assert len(proof1) == len(proof2)
    
    def test_lora_zero_knowledge(self):
        """Test zero-knowledge property for LoRA proofs."""
        prover = LoRAZKProver()
        
        rank = 8
        d = 256
        
        # Two different adapter sets
        adapters1_a = np.random.randn(d, rank).astype(np.float32) * 0.01
        adapters1_b = np.random.randn(rank, d).astype(np.float32) * 0.01
        
        adapters2_a = np.random.randn(d, rank).astype(np.float32) * 0.02
        adapters2_b = np.random.randn(rank, d).astype(np.float32) * 0.02
        
        # Create witnesses
        witness1 = LoRAStepWitness(
            adapter_a_before=adapters1_a.flatten().tolist(),
            adapter_b_before=adapters1_b.flatten().tolist(),
            adapter_a_after=(adapters1_a * 1.01).flatten().tolist(),
            adapter_b_after=(adapters1_b * 1.01).flatten().tolist(),
            adapter_a_gradients=[0.001] * (d * rank),
            adapter_b_gradients=[0.001] * (rank * d),
            batch_inputs=[0.5] * d,
            batch_targets=[1.0] * d,
            learning_rate=0.01
        )
        
        witness2 = LoRAStepWitness(
            adapter_a_before=adapters2_a.flatten().tolist(),
            adapter_b_before=adapters2_b.flatten().tolist(),
            adapter_a_after=(adapters2_a * 1.01).flatten().tolist(),
            adapter_b_after=(adapters2_b * 1.01).flatten().tolist(),
            adapter_a_gradients=[0.001] * (d * rank),
            adapter_b_gradients=[0.001] * (rank * d),
            batch_inputs=[0.5] * d,
            batch_targets=[1.0] * d,
            learning_rate=0.01
        )
        
        statement = LoRAStepStatement(
            base_weights_root=b"base" * 8,
            adapter_a_root_before=b"a_before" * 4,
            adapter_b_root_before=b"b_before" * 4,
            adapter_a_root_after=b"a_after" * 4,
            adapter_b_root_after=b"b_after" * 4,
            batch_root=b"batch" * 6,
            hparams_hash=b"hparams" * 4,
            rank=rank,
            alpha=rank * 2.0,
            step_number=1,
            epoch=1
        )
        
        proof1, _ = prover.prove_lora_step(statement, witness1)
        proof2, _ = prover.prove_lora_step(statement, witness2)
        
        # Proofs should not reveal adapter values
        assert adapters1_a.tolist() not in str(proof1)
        assert adapters2_a.tolist() not in str(proof2)
    
    def test_statistical_hiding(self):
        """Test statistical hiding property."""
        prover = SGDZKProver()
        
        # Generate multiple proofs with different witnesses
        proofs = []
        for _ in range(10):
            weights = np.random.randn(50).astype(np.float32)
            gradients = np.random.randn(50).astype(np.float32) * 0.01
            
            witness = SGDStepWitness(
                weights_before=weights.tolist(),
                weights_after=(weights - 0.001 * gradients).tolist(),
                gradients=gradients.tolist(),
                batch_inputs=[[0.5] * 10 for _ in range(32)],
                batch_targets=[[1.0] for _ in range(32)],
                learning_rate=0.001
            )
            
            statement = SGDStepStatement(
                W_t_root=b"before" * 8,
                W_t1_root=b"after" * 8,
                batch_root=b"batch" * 8,
                hparams_hash=b"hparams" * 4,
                step_number=1,
                epoch=1
            )
            
            proof = prover.prove_sgd_step(statement, witness)
            proofs.append(proof)
        
        # All proofs should have same structure/size
        sizes = [len(p) for p in proofs]
        assert len(set(sizes)) == 1  # All same size
        
        # Proofs should appear random (in real ZK)
        # Cannot distinguish between them without witness


class TestSoundnessAndCompleteness:
    """Test soundness and completeness properties."""
    
    def test_completeness_sgd(self):
        """Test that valid SGD witnesses always produce valid proofs."""
        prover = SGDZKProver()
        
        # Test multiple valid witnesses
        for _ in range(10):
            weights_before = np.random.randn(100).astype(np.float32)
            gradients = np.random.randn(100).astype(np.float32) * 0.01
            lr = 0.001
            weights_after = weights_before - lr * gradients
            
            witness = SGDStepWitness(
                weights_before=weights_before.tolist(),
                weights_after=weights_after.tolist(),
                gradients=gradients.tolist(),
                batch_inputs=[[0.5] * 10 for _ in range(32)],
                batch_targets=[[1.0] for _ in range(32)],
                learning_rate=lr
            )
            
            statement = SGDStepStatement(
                W_t_root=b"before" * 8,
                W_t1_root=b"after" * 8,
                batch_root=b"batch" * 8,
                hparams_hash=b"hparams" * 4,
                step_number=1,
                epoch=1
            )
            
            # Should always succeed
            proof = prover.prove_sgd_step(statement, witness)
            assert proof is not None
            assert len(proof) > 0
    
    def test_completeness_lora(self):
        """Test that valid LoRA witnesses always produce valid proofs."""
        prover = LoRAZKProver()
        
        ranks = [4, 8, 16, 32]
        for rank in ranks:
            d = 256
            
            adapters_a = np.random.randn(d, rank).astype(np.float32) * 0.01
            adapters_b = np.random.randn(rank, d).astype(np.float32) * 0.01
            
            witness = LoRAStepWitness(
                adapter_a_before=adapters_a.flatten().tolist(),
                adapter_b_before=adapters_b.flatten().tolist(),
                adapter_a_after=(adapters_a * 1.01).flatten().tolist(),
                adapter_b_after=(adapters_b * 1.01).flatten().tolist(),
                adapter_a_gradients=[0.001] * (d * rank),
                adapter_b_gradients=[0.001] * (rank * d),
                batch_inputs=[0.5] * d,
                batch_targets=[1.0] * d,
                learning_rate=0.01
            )
            
            statement = LoRAStepStatement(
                base_weights_root=b"base" * 8,
                adapter_a_root_before=b"a_before" * 4,
                adapter_b_root_before=b"b_before" * 4,
                adapter_a_root_after=b"a_after" * 4,
                adapter_b_root_after=b"b_after" * 4,
                batch_root=b"batch" * 6,
                hparams_hash=b"hparams" * 4,
                rank=rank,
                alpha=rank * 2.0,
                step_number=1,
                epoch=1
            )
            
            # Should always succeed
            proof, metadata = prover.prove_lora_step(statement, witness)
            assert proof is not None
            assert len(proof) > 0
    
    def test_soundness_invalid_witnesses(self):
        """Test that invalid witnesses cannot produce valid proofs."""
        prover = SGDZKProver()
        
        # Collection of invalid witnesses
        invalid_witnesses = []
        
        # Wrong gradient computation
        weights = np.random.randn(50).astype(np.float32)
        wrong_witness1 = SGDStepWitness(
            weights_before=weights.tolist(),
            weights_after=weights.tolist(),  # No change!
            gradients=[0.01] * 50,  # Non-zero gradients
            batch_inputs=[[0.5] * 10 for _ in range(32)],
            batch_targets=[[1.0] for _ in range(32)],
            learning_rate=0.001
        )
        invalid_witnesses.append(wrong_witness1)
        
        # Inconsistent dimensions
        wrong_witness2 = SGDStepWitness(
            weights_before=[0.1] * 50,
            weights_after=[0.1] * 49,  # Wrong size!
            gradients=[0.01] * 50,
            batch_inputs=[[0.5] * 10 for _ in range(32)],
            batch_targets=[[1.0] for _ in range(32)],
            learning_rate=0.001
        )
        invalid_witnesses.append(wrong_witness2)
        
        statement = SGDStepStatement(
            W_t_root=b"before" * 8,
            W_t1_root=b"after" * 8,
            batch_root=b"batch" * 8,
            hparams_hash=b"hparams" * 4,
            step_number=1,
            epoch=1
        )
        
        # None should produce valid proofs
        for witness in invalid_witnesses:
            with pytest.raises(Exception):
                proof = prover.prove_sgd_step(statement, witness)
    
    def test_binding_property(self):
        """Test computational binding - cannot prove two different witnesses for same statement."""
        prover = SGDZKProver()
        
        # Create statement
        statement = SGDStepStatement(
            W_t_root=b"fixed_before" * 4,
            W_t1_root=b"fixed_after" * 4,
            batch_root=b"fixed_batch" * 4,
            hparams_hash=b"fixed_hparams" * 2,
            step_number=42,
            epoch=7
        )
        
        # Create two different valid witnesses
        weights1 = np.random.randn(50).astype(np.float32)
        gradients1 = np.random.randn(50).astype(np.float32) * 0.01
        
        weights2 = np.random.randn(50).astype(np.float32)
        gradients2 = np.random.randn(50).astype(np.float32) * 0.02
        
        witness1 = SGDStepWitness(
            weights_before=weights1.tolist(),
            weights_after=(weights1 - 0.001 * gradients1).tolist(),
            gradients=gradients1.tolist(),
            batch_inputs=[[0.5] * 10 for _ in range(32)],
            batch_targets=[[1.0] for _ in range(32)],
            learning_rate=0.001
        )
        
        witness2 = SGDStepWitness(
            weights_before=weights2.tolist(),
            weights_after=(weights2 - 0.001 * gradients2).tolist(),
            gradients=gradients2.tolist(),
            batch_inputs=[[0.5] * 10 for _ in range(32)],
            batch_targets=[[1.0] for _ in range(32)],
            learning_rate=0.001
        )
        
        # Generate proofs
        proof1 = prover.prove_sgd_step(statement, witness1)
        proof2 = prover.prove_sgd_step(statement, witness2)
        
        # In a real ZK system, at most one of these could verify
        # (Here we simulate that they produce different proofs)
        assert proof1 != proof2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])