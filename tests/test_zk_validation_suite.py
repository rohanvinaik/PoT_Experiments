#!/usr/bin/env python3
"""
Comprehensive validation suite for the ZK proof system.

This ensures all ZK components work correctly and integrate
properly with the existing PoT infrastructure.
"""

import pytest
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

sys.path.append(str(Path(__file__).parent.parent))

# Import all ZK modules to test
from pot.zk.prover import SGDZKProver, LoRAZKProver, auto_prove_training_step
from pot.zk.parallel_prover import OptimizedLoRAProver, ParallelProver, StreamingProver
from pot.zk.proof_aggregation import ProofAggregator, IncrementalAggregator, BatchVerifier
from pot.zk.cache import get_cache, clear_all_caches, get_all_stats
from pot.zk.metrics import get_monitor, record_proof_generation
from pot.zk.config_loader import set_mode, get_config
from pot.zk.poseidon import PoseidonHash, PoseidonMerkleTree
from pot.zk.field_arithmetic import FieldElement
from pot.zk.commitments import DualCommitment
from pot.zk.lora_builder import LoRAWitnessBuilder


class TestZKValidationSuite:
    """Comprehensive validation of ZK system."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        # Clear caches
        clear_all_caches()
        
        # Set development mode for faster tests
        set_mode('development')
        
        yield
        
        # Cleanup if needed
        pass
    
    def test_complete_sgd_workflow(self):
        """Test complete SGD proof workflow."""
        # 1. Create model update
        weights_before = np.random.randn(100).astype(np.float32)
        gradients = np.random.randn(100).astype(np.float32) * 0.01
        learning_rate = 0.001
        weights_after = weights_before - learning_rate * gradients
        
        # 2. Generate proof
        from pot.zk.zk_types import SGDStepStatement, SGDStepWitness
        
        witness = SGDStepWitness(
            weights_before=weights_before.tolist(),
            weights_after=weights_after.tolist(),
            batch_inputs=[0.5] * 320,  # Flattened batch inputs
            batch_targets=[1.0] * 32,  # Flattened batch targets
            learning_rate=learning_rate
        )
        
        statement = SGDStepStatement(
            W_t_root=b"before" * 8,
            W_t1_root=b"after" * 8,
            batch_root=b"batch" * 8,
            hparams_hash=b"hparams" * 4,
            step_nonce=0,
            step_number=1,
            epoch=1
        )
        
        prover = SGDZKProver()
        proof = prover.prove_sgd_step(statement, witness)
        
        # 3. Verify proof properties
        assert proof is not None
        assert len(proof) > 0
        assert isinstance(proof, bytes)
        
        # 4. Store on blockchain
        from pot.prototypes.training_provenance_auditor import MockBlockchainClient
        blockchain = MockBlockchainClient()
        tx_hash = blockchain.store_commitment(proof)
        
        # 5. Retrieve and verify
        retrieved = blockchain.get_commitment(tx_hash)
        assert retrieved == proof
    
    def test_complete_lora_workflow(self):
        """Test complete LoRA proof workflow."""
        # 1. Create LoRA adapters
        from pot.zk.lora_builder import create_example_lora_adapters
        
        d = 768
        rank = 8
        adapters = create_example_lora_adapters(d, d, rank)
        
        # 2. Create witness
        from pot.zk.zk_types import LoRAStepWitness, LoRAStepStatement
        
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
        
        # 3. Generate proof
        prover = LoRAZKProver()
        proof, metadata = prover.prove_lora_step(statement, witness)
        
        # 4. Verify properties
        assert proof is not None
        assert metadata['compression_ratio'] > 10
        
        # 5. Test optimized prover
        optimized = OptimizedLoRAProver()
        results = optimized.prove_lora_batch([(statement, witness)])
        assert results[0].success
    
    def test_parallel_proving(self):
        """Test parallel proof generation."""
        prover = ParallelProver(num_workers=2)
        
        # Create batch of tasks
        from pot.zk.parallel_prover import ProofTask
        from pot.zk.zk_types import SGDStepStatement, SGDStepWitness
        
        tasks = []
        for i in range(4):
            witness = SGDStepWitness(
                weights_before=[0.1] * 50,
                weights_after=[0.101] * 50,
                batch_inputs=[0.5] * 320,  # Flattened batch inputs
                batch_targets=[1.0] * 32,  # Flattened batch targets
                learning_rate=0.01
            )
            
            statement = SGDStepStatement(
                W_t_root=f"before_{i}".encode() * 4,
                W_t1_root=f"after_{i}".encode() * 4,
                batch_root=f"batch_{i}".encode() * 4,
                hparams_hash=b"hparams" * 4,
                step_nonce=0,
                step_number=i,
                epoch=1
            )
            
            task = ProofTask(
                task_id=f"task_{i}",
                statement=statement,
                witness=witness,
                proof_type="sgd"
            )
            tasks.append(task)
        
        # Generate in parallel
        results = prover.generate_batch(tasks)
        
        # Verify results
        assert len(results) == 4
        successful = sum(1 for r in results if r.success)
        assert successful == 4
    
    def test_proof_aggregation(self):
        """Test proof aggregation."""
        aggregator = ProofAggregator()
        
        # Create proofs
        from pot.zk.proof_aggregation import ProofBatch
        
        proofs = [f"proof_{i}".encode() * 32 for i in range(8)]
        batch = ProofBatch(
            proofs=proofs,
            statements=[f"stmt_{i}" for i in range(8)],
            proof_type="sgd"
        )
        
        # Aggregate
        aggregated = aggregator.aggregate_proofs(batch)
        
        # Verify properties
        assert aggregated.num_proofs == 8
        assert len(aggregated.proof_data) < sum(len(p) for p in proofs)
        
        # Verify aggregated proof
        is_valid = aggregator.verify_aggregated_proof(aggregated)
        assert is_valid
    
    def test_caching_system(self):
        """Test caching system."""
        # Test LRU cache
        lru = get_cache('lru')
        lru.put("key1", "value1")
        assert lru.get("key1") == "value1"
        
        # Test Merkle cache
        merkle = get_cache('merkle')
        leaves = [f"leaf_{i}".encode() for i in range(8)]
        root1 = merkle.get_root(leaves)
        root2 = merkle.get_root(leaves)  # Should hit cache
        assert root1 == root2
        
        # Test witness cache
        witness = get_cache('witness')
        witness.put_witness("witness1", {"data": "test"})
        retrieved = witness.get_witness("witness1")
        assert retrieved == {"data": "test"}
        
        # Check stats
        stats = get_all_stats()
        assert 'lru' in stats
        assert 'merkle' in stats
        assert 'witness' in stats
    
    def test_poseidon_integration(self):
        """Test Poseidon hash integration."""
        # Test basic hash
        hasher = PoseidonHash()
        data = b"test data"
        hash1 = hasher.hash(data)
        assert isinstance(hash1, bytes)
        
        # Test Merkle tree
        leaves = [f"leaf_{i}".encode() for i in range(16)]
        tree = PoseidonMerkleTree(leaves)
        root = tree.root()
        
        # Test proof generation and verification
        proof = tree.proof(5)
        is_valid = tree.verify(leaves[5], 5, proof)
        assert is_valid
        
        # Test field arithmetic
        a = FieldElement(42)
        b = FieldElement(17)
        c = a + b
        assert c.value == 59
    
    def test_dual_commitments(self):
        """Test dual commitment scheme."""
        dual = DualCommitment()
        
        # Test tensor commitment
        weights = np.random.randn(100, 100).astype(np.float32)
        commitment = dual.commit_tensor(weights)
        
        assert 'sha256_root' in commitment
        assert 'poseidon_root' in commitment
        
        # Verify consistency
        is_consistent = dual.verify_consistency(commitment)
        assert is_consistent
        
        # Test batch commitment
        inputs = np.random.randn(32, 100).astype(np.float32)
        targets = np.random.randn(32, 10).astype(np.float32)
        
        batch_commit = dual.commit_batch(inputs, targets)
        assert 'sha256_root' in batch_commit
        assert 'poseidon_root' in batch_commit
    
    def test_configuration_management(self):
        """Test configuration management."""
        # Test mode switching
        set_mode('development')
        config = get_config()
        assert config.mode == 'development'
        assert config.debug_mode == True
        
        set_mode('production')
        config = get_config()
        assert config.mode == 'production'
        assert config.debug_mode == False
        
        # Test configuration properties
        assert config.num_workers > 0
        assert config.memory_cache_size_mb > 0
        assert isinstance(config.features, dict)
    
    def test_monitoring_system(self):
        """Test monitoring system."""
        monitor = get_monitor()
        
        # Record some metrics
        for i in range(5):
            record_proof_generation(
                proof_type="sgd",
                generation_time_ms=100 + i * 10,
                proof_size=256,
                success=True
            )
        
        # Get dashboard
        dashboard = monitor.get_dashboard_data()
        
        assert 'summary' in dashboard
        assert dashboard['summary']['total_proofs'] >= 5
        assert 'monitoring' in dashboard
        assert dashboard['monitoring'] == True
    
    def test_auto_detection(self):
        """Test automatic proof type detection."""
        # Test with SGD model
        sgd_model_before = {'weights': np.random.randn(100).astype(np.float32)}
        sgd_model_after = {'weights': sgd_model_before['weights'] - 0.001}
        
        result = auto_prove_training_step(
            model_before=sgd_model_before,
            model_after=sgd_model_after,
            batch_data={'inputs': [[1.0]], 'targets': [[1.0]]},
            learning_rate=0.001
        )
        
        assert result['success']
        assert result['proof_type'] == 'sgd'
        
        # Test with LoRA model
        lora_model_before = {
            'lora_A.weight': np.random.randn(768, 8).astype(np.float32),
            'lora_B.weight': np.random.randn(8, 768).astype(np.float32),
            'base.weight': np.random.randn(768, 768).astype(np.float32)
        }
        
        lora_model_after = {
            'lora_A.weight': lora_model_before['lora_A.weight'] * 1.01,
            'lora_B.weight': lora_model_before['lora_B.weight'] * 1.01,
            'base.weight': lora_model_before['base.weight']
        }
        
        result = auto_prove_training_step(
            model_before=lora_model_before,
            model_after=lora_model_after,
            batch_data={'inputs': [[1.0]], 'targets': [[1.0]]},
            learning_rate=0.001
        )
        
        assert result['success']
        assert result['proof_type'] == 'lora'
    
    def test_streaming_prover(self):
        """Test streaming proof generation."""
        prover = StreamingProver(num_workers=2, queue_size=10)
        prover.start()
        
        # Submit tasks
        from pot.zk.parallel_prover import ProofTask
        from pot.zk.zk_types import SGDStepStatement, SGDStepWitness
        
        for i in range(5):
            witness = SGDStepWitness(
                weights_before=[0.1] * 50,
                weights_after=[0.101] * 50,
                batch_inputs=[0.5] * 320,  # Flattened batch inputs
                batch_targets=[1.0] * 32,  # Flattened batch targets
                learning_rate=0.01
            )
            
            statement = SGDStepStatement(
                W_t_root=f"before_{i}".encode() * 4,
                W_t1_root=f"after_{i}".encode() * 4,
                batch_root=f"batch_{i}".encode() * 4,
                hparams_hash=b"hparams" * 4,
                step_nonce=0,
                step_number=i,
                epoch=1
            )
            
            task = ProofTask(
                task_id=f"stream_{i}",
                statement=statement,
                witness=witness,
                proof_type="sgd"
            )
            
            submitted = prover.submit_task(task)
            assert submitted
        
        # Collect results
        results = []
        timeout = time.time() + 5
        while len(results) < 5 and time.time() < timeout:
            result = prover.get_result(timeout=0.5)
            if result:
                results.append(result)
        
        prover.stop()
        
        assert len(results) == 5
        assert all(r.success for r in results)
    
    def test_batch_verification(self):
        """Test batch verification."""
        verifier = BatchVerifier(batch_size=4)
        
        # Create proofs
        proofs = [f"proof_{i}".encode() * 32 for i in range(10)]
        statements = [f"statement_{i}" for i in range(10)]
        
        # Verify batch
        results = verifier.verify_batch(proofs, statements)
        
        assert len(results) == 10
        assert all(results)  # All should verify in mock
        
        # Check cache
        stats = verifier.get_stats()
        assert stats['total_verified'] == 10
    
    def test_incremental_aggregation(self):
        """Test incremental proof aggregation."""
        aggregator = IncrementalAggregator(window_size=3)
        
        # Add proofs incrementally
        aggregates = []
        for i in range(7):
            proof = f"proof_{i}".encode() * 32
            statement = f"statement_{i}"
            
            aggregate = aggregator.add_proof(proof, statement)
            if aggregate:
                aggregates.append(aggregate)
        
        # Should have 2 aggregates (6 proofs)
        assert len(aggregates) == 2
        
        # Flush remaining
        final = aggregator.flush()
        assert final is not None  # Should have 1 remaining
        
        # Check stats
        stats = aggregator.get_stats()
        assert stats['total_processed'] == 7


def run_validation_suite():
    """Run complete validation suite."""
    print("\n" + "="*60)
    print("RUNNING ZK VALIDATION SUITE")
    print("="*60)
    
    # Run pytest with verbose output
    result = pytest.main([__file__, "-v", "--tb=short"])
    
    if result == 0:
        print("\n✅ ALL VALIDATION TESTS PASSED")
    else:
        print("\n❌ SOME VALIDATION TESTS FAILED")
    
    return result


if __name__ == "__main__":
    sys.exit(run_validation_suite())