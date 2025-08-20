"""
End-to-end integration tests for the ZK proof system.

This module tests the complete integration of:
- ZK proofs with TrainingProvenanceAuditor
- Blockchain storage and retrieval
- Full training workflow with proofs
- Verification and auditing
"""

import pytest
import numpy as np
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any

from pot.prototypes.training_provenance_auditor import TrainingProvenanceAuditor
from pot.testing.mock_blockchain import MockBlockchainClient
from pot.zk.prover import auto_prove_training_step, SGDZKProver, LoRAZKProver
from pot.zk.lora_builder import LoRAWitnessBuilder
from pot.zk.proof_aggregation import ProofAggregator, IncrementalAggregator
from pot.zk.parallel_prover import OptimizedLoRAProver
from pot.zk.config_loader import set_mode, get_config
from pot.zk.metrics import get_monitor
from pot.core.model_verification import verify_model_weights


class TestEndToEndIntegration:
    """Test complete ZK proof integration."""
    
    @pytest.fixture
    def setup_environment(self):
        """Set up test environment."""
        # Set development mode for faster tests
        set_mode('development')
        
        # Create temporary directory for artifacts
        temp_dir = tempfile.mkdtemp()
        
        # Initialize components
        auditor = TrainingProvenanceAuditor(
            model_id="test_model_zk",
            hash_function="poseidon"
        )
        
        blockchain = MockBlockchainClient()
        
        prover = SGDZKProver()
        lora_prover = LoRAZKProver()
        
        yield {
            'auditor': auditor,
            'blockchain': blockchain,
            'prover': prover,
            'lora_prover': lora_prover,
            'temp_dir': Path(temp_dir)
        }
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_full_training_workflow_with_zk(self, setup_environment):
        """Test complete training workflow with ZK proofs."""
        env = setup_environment
        auditor = env['auditor']
        blockchain = env['blockchain']
        prover = env['prover']
        
        # Simulate training loop
        num_epochs = 3
        num_steps = 5
        proof_hashes = []
        
        for epoch in range(num_epochs):
            epoch_proofs = []
            
            for step in range(num_steps):
                # Create model states
                weights_before = np.random.randn(100).astype(np.float32)
                gradients = np.random.randn(100).astype(np.float32) * 0.01
                learning_rate = 0.001
                weights_after = weights_before - learning_rate * gradients
                
                # Create batch data
                batch_inputs = np.random.randn(32, 10).astype(np.float32)
                batch_targets = np.random.randn(32, 1).astype(np.float32)
                
                # Generate ZK proof
                proof_result = auto_prove_training_step(
                    model_before={'weights': weights_before},
                    model_after={'weights': weights_after},
                    batch_data={
                        'inputs': batch_inputs,
                        'targets': batch_targets
                    },
                    learning_rate=learning_rate,
                    step_number=epoch * num_steps + step,
                    epoch=epoch
                )
                
                assert proof_result['success']
                assert proof_result['proof'] is not None
                
                # Store proof on blockchain
                tx_hash = blockchain.store_commitment(proof_result['proof'])
                proof_hashes.append(tx_hash)
                epoch_proofs.append(proof_result['proof'])
                
                # Log training event
                metrics = {
                    'loss': np.random.random(),
                    'accuracy': np.random.random(),
                    'step': step,
                    'proof_hash': tx_hash
                }
                event = auditor.log_training_event(
                    epoch=epoch,
                    metrics=metrics,
                    metadata={'zk_proof': tx_hash}
                )
                
                assert event.event_hash is not None
            
            # Aggregate epoch proofs
            from pot.zk.proof_aggregation import ProofBatch
            batch = ProofBatch(
                proofs=epoch_proofs,
                statements=[f"step_{i}" for i in range(num_steps)],
                proof_type="sgd"
            )
            
            aggregator = ProofAggregator()
            aggregated = aggregator.aggregate_proofs(batch)
            
            # Store aggregated proof
            agg_tx = blockchain.store_commitment(aggregated.proof_data)
            
            # Create epoch checkpoint
            checkpoint = auditor.create_checkpoint(
                epoch=epoch,
                model_state={'aggregated_proof': agg_tx}
            )
            assert checkpoint is not None
        
        # Verify all proofs are stored
        assert len(proof_hashes) == num_epochs * num_steps
        
        # Retrieve and verify proofs
        for tx_hash in proof_hashes:
            proof_data = blockchain.get_commitment(tx_hash)
            assert proof_data is not None
            assert len(proof_data) > 0
        
        # Get training history
        history = auditor.get_training_history()
        assert len(history) == num_epochs * num_steps
        
        # Verify audit trail
        is_valid = auditor.verify_training_history()
        assert is_valid
    
    def test_lora_training_with_zk(self, setup_environment):
        """Test LoRA training with ZK proofs."""
        env = setup_environment
        lora_prover = env['lora_prover']
        blockchain = env['blockchain']
        
        # Create LoRA model
        d_in, d_out = 768, 768
        rank = 8
        
        # Initialize adapters
        adapter_a = np.random.randn(d_in, rank).astype(np.float32) * 0.01
        adapter_b = np.random.randn(rank, d_out).astype(np.float32) * 0.01
        base_weights = np.random.randn(d_in, d_out).astype(np.float32)
        
        # Training step
        learning_rate = 0.001
        adapter_a_grad = np.random.randn(d_in, rank).astype(np.float32) * 0.001
        adapter_b_grad = np.random.randn(rank, d_out).astype(np.float32) * 0.001
        
        adapter_a_new = adapter_a - learning_rate * adapter_a_grad
        adapter_b_new = adapter_b - learning_rate * adapter_b_grad
        
        # Build LoRA witness
        from pot.zk.lora_builder import LoRAWitnessBuilder
        builder = LoRAWitnessBuilder()
        
        model_before = {
            'lora_A': adapter_a,
            'lora_B': adapter_b,
            'base': base_weights
        }
        
        model_after = {
            'lora_A': adapter_a_new,
            'lora_B': adapter_b_new,
            'base': base_weights
        }
        
        witness = builder.build_lora_witness(
            model_before=model_before,
            model_after=model_after,
            batch_data={
                'inputs': np.random.randn(32, d_in).astype(np.float32),
                'targets': np.random.randn(32, d_out).astype(np.float32)
            },
            learning_rate=learning_rate,
            rank=rank,
            alpha=rank * 2.0
        )
        
        # Create statement
        from pot.zk.zk_types import LoRAStepStatement
        statement = LoRAStepStatement(
            base_weights_root=b"base_root" * 4,
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
        
        # Generate proof
        proof, metadata = lora_prover.prove_lora_step(statement, witness)
        assert proof is not None
        assert len(proof) > 0
        
        # Store on blockchain
        tx_hash = blockchain.store_commitment(proof)
        
        # Verify storage
        retrieved = blockchain.get_commitment(tx_hash)
        assert retrieved == proof
        
        # Verify metadata
        assert metadata['compression_ratio'] > 20  # LoRA should compress well
    
    def test_incremental_aggregation(self, setup_environment):
        """Test incremental proof aggregation."""
        env = setup_environment
        blockchain = env['blockchain']
        
        # Create incremental aggregator
        aggregator = IncrementalAggregator(window_size=5)
        
        # Generate proofs incrementally
        for i in range(12):
            # Create simple proof
            proof_data = f"proof_{i}".encode() * 32
            statement = f"statement_{i}"
            
            # Add to aggregator
            aggregate = aggregator.add_proof(proof_data, statement)
            
            # Every 5 proofs, should get aggregate
            if (i + 1) % 5 == 0:
                assert aggregate is not None
                
                # Store aggregate on blockchain
                tx_hash = blockchain.store_commitment(aggregate.proof_data)
                
                # Verify storage
                retrieved = blockchain.get_commitment(tx_hash)
                assert retrieved == aggregate.proof_data
        
        # Flush remaining
        final = aggregator.flush()
        assert final is not None  # Should have 2 remaining proofs
        
        # Check statistics
        stats = aggregator.get_stats()
        assert stats['total_processed'] == 12
    
    def test_proof_retrieval_and_verification(self, setup_environment):
        """Test proof retrieval and verification from blockchain."""
        env = setup_environment
        blockchain = env['blockchain']
        prover = env['prover']
        
        # Generate multiple proofs
        proof_data_list = []
        tx_hashes = []
        
        for i in range(5):
            # Create proof
            weights_before = np.random.randn(50).astype(np.float32)
            gradients = np.random.randn(50).astype(np.float32) * 0.01
            weights_after = weights_before - 0.001 * gradients
            
            from pot.zk.zk_types import SGDStepStatement, SGDStepWitness
            
            statement = SGDStepStatement(
                W_t_root=f"before_{i}".encode() * 4,
                W_t1_root=f"after_{i}".encode() * 4,
                batch_root=f"batch_{i}".encode() * 4,
                hparams_hash=b"hparams" * 4,
                step_number=i,
                epoch=1
            )
            
            witness = SGDStepWitness(
                weights_before=weights_before.tolist(),
                weights_after=weights_after.tolist(),
                gradients=gradients.tolist(),
                batch_inputs=[[0.5] * 10 for _ in range(32)],
                batch_targets=[[1.0] for _ in range(32)],
                learning_rate=0.001
            )
            
            proof = prover.prove_sgd_step(statement, witness)
            proof_data_list.append(proof)
            
            # Store on blockchain
            tx_hash = blockchain.store_commitment(proof)
            tx_hashes.append(tx_hash)
        
        # Retrieve all proofs
        retrieved_proofs = []
        for tx_hash in tx_hashes:
            proof = blockchain.get_commitment(tx_hash)
            assert proof is not None
            retrieved_proofs.append(proof)
        
        # Verify all match
        for original, retrieved in zip(proof_data_list, retrieved_proofs):
            assert original == retrieved
        
        # Test batch verification
        from pot.zk.proof_aggregation import BatchVerifier
        verifier = BatchVerifier()
        
        statements = [f"statement_{i}" for i in range(5)]
        results = verifier.verify_batch(retrieved_proofs, statements)
        
        # All should verify
        assert all(results)
        
        # Test with invalid proof
        invalid_proofs = retrieved_proofs + [b"invalid_proof"]
        invalid_statements = statements + ["invalid"]
        
        results = verifier.verify_batch(invalid_proofs, invalid_statements)
        assert results[-1] == False  # Last one should fail
    
    def test_optimized_lora_production(self, setup_environment):
        """Test optimized LoRA prover in production mode."""
        # Switch to production mode
        set_mode('production')
        config = get_config()
        
        # Create optimized prover
        prover = OptimizedLoRAProver()
        prover.optimize_for_hardware()
        
        # Create batch of LoRA updates
        from pot.zk.zk_types import LoRAStepStatement, LoRAStepWitness
        
        updates = []
        for i in range(4):
            rank = 8 * (i + 1)  # Test different ranks
            d = 768
            
            statement = LoRAStepStatement(
                base_weights_root=b"base" * 8,
                adapter_a_root_before=f"a_before_{i}".encode() * 2,
                adapter_b_root_before=f"b_before_{i}".encode() * 2,
                adapter_a_root_after=f"a_after_{i}".encode() * 2,
                adapter_b_root_after=f"b_after_{i}".encode() * 2,
                batch_root=b"batch" * 6,
                hparams_hash=b"hparams" * 4,
                rank=rank,
                alpha=rank * 2.0,
                step_number=i,
                epoch=1
            )
            
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
            
            updates.append((statement, witness))
        
        # Generate proofs with target time
        start = time.time()
        results = prover.prove_lora_batch(updates, target_time_ms=5000)
        elapsed = time.time() - start
        
        # Check results
        assert len(results) == 4
        successful = sum(1 for r in results if r.success)
        assert successful == 4
        
        # Should meet target (in mock implementation)
        assert elapsed < 10  # Give some buffer for mock
        
        # Check individual times
        for result in results:
            assert result.generation_time_ms < 5000
    
    def test_monitoring_integration(self, setup_environment):
        """Test monitoring integration with proof generation."""
        monitor = get_monitor()
        
        # Generate some proofs with monitoring
        from pot.zk.metrics import record_proof_generation, record_proof_verification
        
        for i in range(10):
            # Record proof generation
            record_proof_generation(
                proof_type="sgd",
                generation_time_ms=100 + i * 10,
                proof_size=256,
                success=True,
                step=i
            )
            
            # Record verification
            record_proof_verification(
                proof_type="sgd",
                verification_time_ms=10 + i,
                success=True
            )
        
        # Get dashboard data
        dashboard = monitor.get_dashboard_data()
        
        assert dashboard['summary']['total_proofs'] >= 10
        assert dashboard['summary']['total_verifications'] >= 10
        assert dashboard['summary']['proof_success_rate'] == 1.0
        
        # Check recent metrics
        recent_proofs = dashboard['recent_proofs']
        assert len(recent_proofs) > 0


class TestBlockchainIntegration:
    """Test blockchain-specific integration."""
    
    def test_proof_persistence(self):
        """Test proof persistence on blockchain."""
        blockchain = MockBlockchainClient()
        
        # Store multiple proofs
        proof_data = [
            b"proof_1" * 32,
            b"proof_2" * 32,
            b"proof_3" * 32
        ]
        
        tx_hashes = []
        for proof in proof_data:
            tx_hash = blockchain.store_commitment(proof)
            tx_hashes.append(tx_hash)
            
            # Should get confirmation
            confirmation = blockchain.get_transaction_status(tx_hash)
            assert confirmation == "confirmed"
        
        # Retrieve all
        for i, tx_hash in enumerate(tx_hashes):
            retrieved = blockchain.get_commitment(tx_hash)
            assert retrieved == proof_data[i]
        
        # Test history
        history = blockchain.get_commitment_history("test_address")
        assert len(history) >= 3
    
    def test_proof_verification_onchain(self):
        """Test on-chain proof verification simulation."""
        blockchain = MockBlockchainClient()
        
        # Create verification contract simulation
        class VerificationContract:
            def __init__(self, blockchain):
                self.blockchain = blockchain
                self.verified_proofs = {}
            
            def submit_proof(self, proof: bytes, statement: Any) -> str:
                """Submit proof for verification."""
                # Store proof
                tx_hash = self.blockchain.store_commitment(proof)
                
                # Simulate verification (always succeeds in mock)
                self.verified_proofs[tx_hash] = {
                    'statement': statement,
                    'verified': True,
                    'timestamp': time.time()
                }
                
                return tx_hash
            
            def is_verified(self, tx_hash: str) -> bool:
                """Check if proof is verified."""
                return self.verified_proofs.get(tx_hash, {}).get('verified', False)
        
        contract = VerificationContract(blockchain)
        
        # Submit proofs
        proofs = [
            (b"proof_a" * 32, "statement_a"),
            (b"proof_b" * 32, "statement_b")
        ]
        
        tx_hashes = []
        for proof, statement in proofs:
            tx_hash = contract.submit_proof(proof, statement)
            tx_hashes.append(tx_hash)
        
        # Verify all are verified
        for tx_hash in tx_hashes:
            assert contract.is_verified(tx_hash)
    
    def test_aggregated_proof_storage(self):
        """Test aggregated proof storage and retrieval."""
        blockchain = MockBlockchainClient()
        aggregator = ProofAggregator()
        
        # Create batch of proofs
        from pot.zk.proof_aggregation import ProofBatch
        
        individual_proofs = [f"proof_{i}".encode() * 32 for i in range(8)]
        batch = ProofBatch(
            proofs=individual_proofs,
            statements=[f"stmt_{i}" for i in range(8)],
            proof_type="sgd"
        )
        
        # Aggregate
        aggregated = aggregator.aggregate_proofs(batch)
        
        # Store individual proofs
        individual_hashes = []
        for proof in individual_proofs:
            tx = blockchain.store_commitment(proof)
            individual_hashes.append(tx)
        
        # Store aggregated proof with metadata
        metadata = {
            'type': 'aggregated',
            'num_proofs': aggregated.num_proofs,
            'individual_hashes': individual_hashes,
            'timestamp': aggregated.timestamp
        }
        
        agg_tx = blockchain.store_commitment(
            aggregated.proof_data,
            metadata=str(metadata)
        )
        
        # Retrieve and verify
        retrieved = blockchain.get_commitment(agg_tx)
        assert retrieved == aggregated.proof_data
        
        # Could retrieve individual proofs if needed
        for tx in individual_hashes:
            individual = blockchain.get_commitment(tx)
            assert individual is not None


class TestBackwardCompatibility:
    """Test backward compatibility with existing systems."""
    
    def test_compatible_with_existing_auditor(self):
        """Test compatibility with existing TrainingProvenanceAuditor."""
        # SHA-256 auditor (existing)
        auditor_sha = TrainingProvenanceAuditor(
            model_id="model_sha",
            hash_function="sha256"
        )
        
        # Poseidon auditor (new)
        auditor_pos = TrainingProvenanceAuditor(
            model_id="model_pos",
            hash_function="poseidon"
        )
        
        # Both should work
        metrics = {'loss': 0.5}
        
        event_sha = auditor_sha.log_training_event(1, metrics)
        event_pos = auditor_pos.log_training_event(1, metrics)
        
        assert event_sha.event_hash is not None
        assert event_pos.event_hash is not None
        
        # Hashes should be different (different algorithms)
        assert event_sha.event_hash != event_pos.event_hash
    
    def test_compatible_with_model_verification(self):
        """Test compatibility with existing model verification."""
        # Create models
        model1 = {'layer': np.random.randn(100).astype(np.float32)}
        model2 = {'layer': model1['layer'] + 0.001}
        
        # Existing verification should still work
        result = verify_model_weights(model1, model2, threshold=0.0001)
        assert not result['identical']
        
        # ZK proof should also work
        proof_result = auto_prove_training_step(
            model_before=model1,
            model_after=model2,
            batch_data={'inputs': [[1.0]], 'targets': [[1.0]]},
            learning_rate=0.001
        )
        assert proof_result['success']
    
    def test_migration_path(self):
        """Test migration path from non-ZK to ZK system."""
        # Start with non-ZK system
        auditor = TrainingProvenanceAuditor(
            model_id="migrating_model",
            hash_function="sha256"
        )
        
        # Train for some epochs without ZK
        for epoch in range(3):
            metrics = {'loss': 0.5 - epoch * 0.1}
            auditor.log_training_event(epoch, metrics)
        
        # Switch to ZK (would require new auditor in practice)
        auditor_zk = TrainingProvenanceAuditor(
            model_id="migrating_model",
            hash_function="poseidon"
        )
        
        # Continue training with ZK
        for epoch in range(3, 6):
            metrics = {'loss': 0.2 - (epoch - 3) * 0.05}
            event = auditor_zk.log_training_event(epoch, metrics)
            
            # Could also generate ZK proof here
            assert event.event_hash is not None
        
        # Both histories should be valid
        assert auditor.verify_training_history()
        assert auditor_zk.verify_training_history()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])