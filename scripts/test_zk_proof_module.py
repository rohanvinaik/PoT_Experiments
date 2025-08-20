#!/usr/bin/env python3
"""
Test the ZK proof module integration with existing PoT infrastructure
"""

import sys
import os
import numpy as np
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.zk import (
    SGDStepStatement, SGDStepWitness,
    LoRAStepStatement, LoRAStepWitness,
    ZKProofType, CommitmentScheme,
    PoseidonHasher, MerkleCommitment, DualCommitment,
    extract_sgd_witness, extract_lora_witness, build_zk_statement
)
from pot.prototypes.training_provenance_auditor import (
    TrainingProvenanceAuditor, 
    MockBlockchainClient,
    EventType
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_zk_module_imports():
    """Test that all ZK module components can be imported"""
    
    logger.info("Testing ZK module imports...")
    
    # Test that all imports work
    assert SGDStepStatement is not None, "SGDStepStatement import failed"
    assert SGDStepWitness is not None, "SGDStepWitness import failed"
    assert LoRAStepStatement is not None, "LoRAStepStatement import failed"
    assert LoRAStepWitness is not None, "LoRAStepWitness import failed"
    assert ZKProofType is not None, "ZKProofType import failed"
    assert CommitmentScheme is not None, "CommitmentScheme import failed"
    
    # Test commitment functions
    assert PoseidonHasher is not None, "PoseidonHasher import failed"
    assert MerkleCommitment is not None, "MerkleCommitment import failed"
    assert DualCommitment is not None, "DualCommitment import failed"
    
    # Test witness functions
    assert extract_sgd_witness is not None, "extract_sgd_witness import failed"
    assert extract_lora_witness is not None, "extract_lora_witness import failed"
    assert build_zk_statement is not None, "build_zk_statement import failed"
    
    logger.info("✅ All ZK module imports successful")
    return True

def test_poseidon_hasher():
    """Test Poseidon hash function wrapper"""
    
    logger.info("Testing Poseidon hasher...")
    
    # Test basic hashing
    test_data = b"test data for poseidon hashing"
    hash1 = PoseidonHasher.hash_bytes(test_data)
    hash2 = PoseidonHasher.hash_bytes(test_data)
    
    assert hash1 == hash2, "Poseidon hash should be deterministic"
    assert len(hash1) == 32, f"Hash should be 32 bytes, got {len(hash1)}"
    assert isinstance(hash1, bytes), "Hash should be bytes"
    
    # Test different inputs produce different hashes
    different_data = b"different test data"
    hash3 = PoseidonHasher.hash_bytes(different_data)
    assert hash1 != hash3, "Different inputs should produce different hashes"
    
    # Test field element hashing
    elements = [1, 2, 3, 4, 5]
    hash4 = PoseidonHasher.hash_field_elements(elements)
    assert len(hash4) == 32, "Field element hash should be 32 bytes"
    
    # Test two-hash function
    hash5 = PoseidonHasher.hash_two(hash1, hash3)
    assert len(hash5) == 32, "Two-hash result should be 32 bytes"
    
    logger.info("✅ Poseidon hasher test passed")
    return True

def test_dual_commitment():
    """Test dual commitment scheme"""
    
    logger.info("Testing dual commitment scheme...")
    
    dual_committer = DualCommitment()
    
    # Test tensor commitment
    test_tensor = np.random.randn(10, 5).astype(np.float32)
    commitment_data = dual_committer.commit_tensor(test_tensor)
    
    assert "sha256_root" in commitment_data, "Should have SHA-256 commitment"
    assert "poseidon_root" in commitment_data, "Should have Poseidon commitment"
    assert "tensor_data" in commitment_data, "Should have tensor data"
    assert "tensor_shape" in commitment_data, "Should have tensor shape"
    
    # Verify commitment consistency
    is_consistent = dual_committer.verify_consistency(commitment_data)
    assert is_consistent, "Dual commitment should be consistent"
    
    # Test batch commitment
    batch_inputs = np.random.randn(32, 128).astype(np.float32)
    batch_targets = np.random.randint(0, 10, (32,)).astype(np.int64)
    
    batch_commitment = dual_committer.commit_batch(batch_inputs, batch_targets)
    
    assert "sha256_root" in batch_commitment, "Batch should have SHA-256 commitment"
    assert "poseidon_root" in batch_commitment, "Batch should have Poseidon commitment"
    assert "batch_data" in batch_commitment, "Should have batch data"
    
    logger.info(f"Tensor commitment SHA-256: {commitment_data['sha256_root'][:16]}...")
    logger.info(f"Tensor commitment Poseidon: {commitment_data['poseidon_root'][:16]}...")
    logger.info("✅ Dual commitment test passed")
    return True

def test_sgd_witness_extraction():
    """Test SGD witness extraction"""
    
    logger.info("Testing SGD witness extraction...")
    
    # Mock model weights
    weights_before = {
        "layer1": np.random.randn(128, 64).astype(np.float32),
        "layer2": np.random.randn(64, 32).astype(np.float32),
        "output": np.random.randn(32, 10).astype(np.float32)
    }
    
    # Simulate SGD step
    learning_rate = 0.01
    gradients = {
        name: np.random.randn(*weights.shape).astype(np.float32) * 0.1
        for name, weights in weights_before.items()
    }
    
    weights_after = {
        name: weights - learning_rate * gradients[name]
        for name, weights in weights_before.items()
    }
    
    # Mock batch data
    batch_inputs = np.random.randn(32, 128).astype(np.float32)
    batch_targets = np.random.randint(0, 10, (32,)).astype(np.int64)
    
    # Hyperparameters
    hyperparameters = {
        "learning_rate": learning_rate,
        "momentum": 0.9,
        "weight_decay": 0.0001
    }
    
    # Extract witness
    witness = extract_sgd_witness(
        model_weights_before=weights_before,
        model_weights_after=weights_after,
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        hyperparameters=hyperparameters,
        gradients=gradients,
        loss_value=0.5
    )
    
    # Verify witness structure
    assert hasattr(witness, 'weight_values'), "Witness should have weight values"
    assert hasattr(witness, 'weight_openings'), "Witness should have weight openings"
    assert hasattr(witness, 'batch_inputs'), "Witness should have batch inputs"
    assert hasattr(witness, 'gradients'), "Witness should have gradients"
    assert hasattr(witness, 'learning_rate'), "Witness should have learning rate"
    
    # Check weight consistency
    for layer_name in weights_before:
        assert layer_name in witness.weight_values, f"Layer {layer_name} missing from witness"
        assert np.array_equal(witness.weight_values[layer_name], weights_before[layer_name]), \
            f"Weight mismatch for layer {layer_name}"
    
    logger.info(f"SGD witness extracted with {len(witness.weight_values)} weight layers")
    logger.info(f"Batch size: {witness.batch_inputs.shape[0]}")
    logger.info("✅ SGD witness extraction test passed")
    return True

def test_lora_witness_extraction():
    """Test LoRA witness extraction"""
    
    logger.info("Testing LoRA witness extraction...")
    
    # Mock base model weights (frozen)
    base_weights = {
        "attention.query": np.random.randn(512, 512).astype(np.float32),
        "attention.key": np.random.randn(512, 512).astype(np.float32),
        "attention.value": np.random.randn(512, 512).astype(np.float32)
    }
    
    # LoRA matrices (low rank)
    rank = 8
    lora_A_before = {
        "attention.query": np.random.randn(512, rank).astype(np.float32),
        "attention.key": np.random.randn(512, rank).astype(np.float32),
        "attention.value": np.random.randn(512, rank).astype(np.float32)
    }
    
    lora_B_before = {
        "attention.query": np.random.randn(rank, 512).astype(np.float32),
        "attention.key": np.random.randn(rank, 512).astype(np.float32),
        "attention.value": np.random.randn(rank, 512).astype(np.float32)
    }
    
    # Simulate LoRA training step
    learning_rate = 0.001
    gradients_A = {
        name: np.random.randn(*matrix.shape).astype(np.float32) * 0.01
        for name, matrix in lora_A_before.items()
    }
    gradients_B = {
        name: np.random.randn(*matrix.shape).astype(np.float32) * 0.01
        for name, matrix in lora_B_before.items()
    }
    
    lora_A_after = {
        name: matrix - learning_rate * gradients_A[name]
        for name, matrix in lora_A_before.items()
    }
    lora_B_after = {
        name: matrix - learning_rate * gradients_B[name]
        for name, matrix in lora_B_before.items()
    }
    
    # Mock batch data
    batch_inputs = np.random.randn(16, 512).astype(np.float32)
    batch_targets = np.random.randn(16, 512).astype(np.float32)
    
    # LoRA hyperparameters
    lora_hyperparameters = {
        "rank": rank,
        "alpha": 16.0,
        "learning_rate": learning_rate,
        "dropout_rate": 0.1
    }
    
    # Extract LoRA witness
    witness = extract_lora_witness(
        base_weights=base_weights,
        lora_A_before=lora_A_before,
        lora_B_before=lora_B_before,
        lora_A_after=lora_A_after,
        lora_B_after=lora_B_after,
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        lora_hyperparameters=lora_hyperparameters,
        gradients_A=gradients_A,
        gradients_B=gradients_B,
        loss_value=0.3
    )
    
    # Verify witness structure
    assert hasattr(witness, 'base_weights'), "LoRA witness should have base weights"
    assert hasattr(witness, 'lora_A_matrices'), "LoRA witness should have A matrices"
    assert hasattr(witness, 'lora_B_matrices'), "LoRA witness should have B matrices"
    assert hasattr(witness, 'rank'), "LoRA witness should have rank"
    assert hasattr(witness, 'alpha'), "LoRA witness should have alpha"
    
    # Check LoRA parameters
    assert witness.rank == rank, f"Rank mismatch: expected {rank}, got {witness.rank}"
    assert witness.alpha == 16.0, f"Alpha mismatch: expected 16.0, got {witness.alpha}"
    
    logger.info(f"LoRA witness extracted with rank {witness.rank}")
    logger.info(f"LoRA modules: {list(witness.lora_A_matrices.keys())}")
    logger.info("✅ LoRA witness extraction test passed")
    return True

def test_zk_statement_building():
    """Test ZK statement building"""
    
    logger.info("Testing ZK statement building...")
    
    # Mock training step data
    weights_before = {
        "layer1": np.random.randn(64, 32).astype(np.float32),
        "layer2": np.random.randn(32, 10).astype(np.float32)
    }
    
    weights_after = {
        "layer1": weights_before["layer1"] - 0.01 * np.random.randn(64, 32).astype(np.float32),
        "layer2": weights_before["layer2"] - 0.01 * np.random.randn(32, 10).astype(np.float32)
    }
    
    batch_inputs = np.random.randn(16, 64).astype(np.float32)
    batch_targets = np.random.randint(0, 10, (16,)).astype(np.int64)
    
    hyperparameters = {"learning_rate": 0.01, "momentum": 0.9}
    step_info = {"step_number": 42, "epoch": 5, "nonce": 12345}
    
    # Build SGD statement
    sgd_statement = build_zk_statement(
        weights_before=weights_before,
        weights_after=weights_after,
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        hyperparameters=hyperparameters,
        step_info=step_info,
        proof_type=ZKProofType.SGD_STEP
    )
    
    # Verify statement structure
    assert hasattr(sgd_statement, 'W_t_root'), "Statement should have W_t_root"
    assert hasattr(sgd_statement, 'batch_root'), "Statement should have batch_root"
    assert hasattr(sgd_statement, 'hparams_hash'), "Statement should have hparams_hash"
    assert hasattr(sgd_statement, 'W_t1_root'), "Statement should have W_t1_root"
    assert hasattr(sgd_statement, 'step_nonce'), "Statement should have step_nonce"
    
    # Check values
    assert sgd_statement.step_number == 42, "Step number mismatch"
    assert sgd_statement.epoch == 5, "Epoch mismatch"
    assert sgd_statement.step_nonce == 12345, "Nonce mismatch"
    
    # Test serialization
    statement_dict = sgd_statement.to_dict()
    assert isinstance(statement_dict, dict), "Statement should be serializable to dict"
    assert "W_t_root" in statement_dict, "Serialized statement should have W_t_root"
    
    logger.info(f"SGD statement built for step {sgd_statement.step_number}")
    logger.info(f"W_t_root: {sgd_statement.W_t_root[:16]}...")
    logger.info("✅ ZK statement building test passed")
    return True

def test_training_provenance_auditor_integration():
    """Test integration with TrainingProvenanceAuditor"""
    
    logger.info("Testing TrainingProvenanceAuditor ZK integration...")
    
    # Create auditor with mock blockchain
    blockchain_client = MockBlockchainClient()
    auditor = TrainingProvenanceAuditor(
        model_id="test_zk_model",
        blockchain_client=blockchain_client
    )
    
    # Test dual commitment support
    auditor.add_dual_commitment_support()
    assert "zk_support" in auditor.metadata, "Auditor should have ZK support metadata"
    
    # Mock training step data
    weights_before = {
        "weights": {
            "layer1": np.random.randn(32, 16).astype(np.float32),
            "layer2": np.random.randn(16, 8).astype(np.float32)
        }
    }
    
    weights_after = {
        "weights": {
            "layer1": weights_before["weights"]["layer1"] - 0.01 * np.random.randn(32, 16).astype(np.float32),
            "layer2": weights_before["weights"]["layer2"] - 0.01 * np.random.randn(16, 8).astype(np.float32)
        }
    }
    
    batch_data = {
        "inputs": np.random.randn(8, 32).astype(np.float32),
        "targets": np.random.randint(0, 8, (8,)).astype(np.int64),
        "gradients": {
            "layer1": np.random.randn(32, 16).astype(np.float32) * 0.1,
            "layer2": np.random.randn(16, 8).astype(np.float32) * 0.1
        },
        "loss": 0.45
    }
    
    hyperparameters = {
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001
    }
    
    step_info = {
        "step_number": 100,
        "epoch": 10,
        "nonce": 54321
    }
    
    # Generate ZK witness
    zk_result = auditor.generate_zk_witness(
        weights_before=weights_before,
        weights_after=weights_after,
        batch_data=batch_data,
        hyperparameters=hyperparameters,
        step_info=step_info,
        proof_type="sgd_step"
    )
    
    # Verify result structure
    assert "witness" in zk_result, "ZK result should have witness"
    assert "statement" in zk_result, "ZK result should have statement"
    assert "witness_record" in zk_result, "ZK result should have witness record"
    assert "compatibility" in zk_result, "ZK result should have compatibility info"
    
    # Check compatibility flags
    compatibility = zk_result["compatibility"]
    assert compatibility["existing_merkle"], "Should be compatible with existing Merkle"
    assert compatibility["zk_friendly"], "Should be ZK friendly"
    assert compatibility["dual_commitment"], "Should support dual commitment"
    
    # Verify witness record
    witness_record = zk_result["witness_record"]
    assert witness_record["proof_type"] == "sgd_step", "Proof type should match"
    assert witness_record["witness_metadata"]["step_number"] == 100, "Step number should match"
    assert witness_record["witness_metadata"]["model_id"] == "test_zk_model", "Model ID should match"
    
    # Check that event was logged
    assert len(auditor.events) > 0, "ZK witness generation should be logged"
    last_event = auditor.events[-1]
    assert last_event.event_type == EventType.CUSTOM, "Should log custom event"
    assert last_event.metrics["witness_generated"], "Should mark witness as generated"
    
    logger.info(f"ZK witness generated: {witness_record['witness_id']}")
    logger.info(f"Logged {len(auditor.events)} training events")
    logger.info("✅ TrainingProvenanceAuditor integration test passed")
    return True

def test_compatibility_with_existing_infrastructure():
    """Test compatibility with existing PoT infrastructure"""
    
    logger.info("Testing compatibility with existing infrastructure...")
    
    # Test that existing Merkle functions still work
    from pot.prototypes.training_provenance_auditor import (
        MerkleNode, build_merkle_tree, compute_merkle_root
    )
    
    # Create test data
    test_blocks = [b"block1", b"block2", b"block3", b"block4"]
    
    # Build traditional Merkle tree
    merkle_tree = build_merkle_tree(test_blocks)
    merkle_root = compute_merkle_root(test_blocks)
    
    assert merkle_tree is not None, "Merkle tree should be built"
    assert merkle_root is not None, "Merkle root should be computed"
    
    # Test ZK compatibility wrapper
    from pot.zk.commitments import create_zk_compatible_commitment
    
    zk_commitment = create_zk_compatible_commitment(merkle_tree)
    
    assert "sha256_commitment" in zk_commitment, "Should have SHA-256 commitment"
    assert "poseidon_commitment" in zk_commitment, "Should have Poseidon commitment"
    assert zk_commitment["is_zk_compatible"], "Should be ZK compatible"
    
    # Test that ZK module doesn't break existing functionality
    auditor = TrainingProvenanceAuditor("compatibility_test")
    
    # Log some regular training events
    auditor.log_training_event(
        epoch=1,
        metrics={"loss": 0.5, "accuracy": 0.85},
        event_type=EventType.EPOCH_END
    )
    
    # Verify regular functionality still works
    assert len(auditor.events) == 1, "Regular event logging should work"
    
    # Test that auditor can handle both regular and ZK operations
    stats = auditor.get_statistics()
    assert "total_events" in stats, "Statistics should include event count"
    
    logger.info("✅ Compatibility with existing infrastructure test passed")
    return True

def main():
    """Run all ZK proof module tests"""
    logger.info("\n" + "="*70)
    logger.info("ZK PROOF MODULE INTEGRATION TESTS")
    logger.info("="*70)
    
    tests = [
        ("ZK Module Imports", test_zk_module_imports),
        ("Poseidon Hasher", test_poseidon_hasher),
        ("Dual Commitment", test_dual_commitment),
        ("SGD Witness Extraction", test_sgd_witness_extraction),
        ("LoRA Witness Extraction", test_lora_witness_extraction),
        ("ZK Statement Building", test_zk_statement_building),
        ("TrainingProvenanceAuditor Integration", test_training_provenance_auditor_integration),
        ("Compatibility with Existing Infrastructure", test_compatibility_with_existing_infrastructure)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            logger.info(f"\n--- Testing {name} ---")
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            logger.error(f"Test {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL ZK PROOF MODULE TESTS PASSED!")
        logger.info("Key achievements:")
        logger.info("  • ZK proof module successfully integrated with PoT infrastructure")
        logger.info("  • SGD and LoRA step proofs supported")
        logger.info("  • Dual commitment schemes (SHA-256 + Poseidon) working")
        logger.info("  • TrainingProvenanceAuditor enhanced with ZK witness generation")
        logger.info("  • Full compatibility with existing Merkle infrastructure")
        logger.info("  • Mock support ensures graceful degradation")
    else:
        logger.info("\n⚠️ Some tests failed - review output above")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())