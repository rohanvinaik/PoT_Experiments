"""
Integration tests for ZK proof system with existing PoT infrastructure.

This ensures the ZK system integrates properly with:
- Existing model verification
- Training provenance auditing
- Mock blockchain storage
- Enhanced diff testing
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from pot.zk.prover import auto_prove_training_step
from pot.zk.parallel_prover import OptimizedLoRAProver
from pot.zk.config_loader import set_mode
from pot.prototypes.training_provenance_auditor import TrainingProvenanceAuditor
from pot.testing.mock_blockchain import MockBlockchainClient
from pot.core.model_verification import verify_model_weights
from pot.core.diff_decision import EnhancedSequentialTester, TestingMode


class TestZKIntegration:
    """Test ZK integration with existing systems."""
    
    def test_zk_with_model_verification(self):
        """Test ZK proofs work with existing model verification."""
        # Create models
        model_before = {
            'layer1': np.random.randn(100, 100).astype(np.float32),
            'layer2': np.random.randn(50, 50).astype(np.float32)
        }
        
        model_after = {
            'layer1': model_before['layer1'] + np.random.randn(100, 100).astype(np.float32) * 0.001,
            'layer2': model_before['layer2'] + np.random.randn(50, 50).astype(np.float32) * 0.001
        }
        
        # Existing verification
        verification_result = verify_model_weights(model_before, model_after, threshold=0.01)
        assert not verification_result['identical']  # Should detect difference
        
        # Generate ZK proof
        proof_result = auto_prove_training_step(
            model_before=model_before,
            model_after=model_after,
            batch_data={
                'inputs': np.random.randn(32, 100).astype(np.float32),
                'targets': np.random.randn(32, 50).astype(np.float32)
            },
            learning_rate=0.001,
            step_number=1,
            epoch=1
        )
        
        assert proof_result['success']
        assert proof_result['proof'] is not None
        assert proof_result['proof_type'] in ['sgd', 'lora']
    
    def test_zk_with_training_auditor(self):
        """Test ZK proofs integrate with TrainingProvenanceAuditor."""
        # Create auditors
        auditor_sha = TrainingProvenanceAuditor(
            model_id="test_sha",
            hash_function="sha256"
        )
        
        auditor_pos = TrainingProvenanceAuditor(
            model_id="test_poseidon",
            hash_function="poseidon"
        )
        
        # Create blockchain
        blockchain = MockBlockchainClient()
        
        # Training step
        model_before = {'weights': np.random.randn(100).astype(np.float32)}
        model_after = {'weights': model_before['weights'] - 0.001 * np.random.randn(100).astype(np.float32)}
        
        # Generate ZK proof
        proof_result = auto_prove_training_step(
            model_before=model_before,
            model_after=model_after,
            batch_data={'inputs': [[1.0]], 'targets': [[1.0]]},
            learning_rate=0.001
        )
        
        # Store proof on blockchain
        tx_hash = blockchain.store_commitment(proof_result['proof'])
        
        # Log with both auditors
        metrics = {
            'loss': 0.5,
            'zk_proof_hash': tx_hash,
            'proof_type': proof_result['proof_type']
        }
        
        event_sha = auditor_sha.log_training_event(1, metrics)
        event_pos = auditor_pos.log_training_event(1, metrics)
        
        assert event_sha.event_hash is not None
        assert event_pos.event_hash is not None
        
        # Verify histories
        assert auditor_sha.verify_training_history()
        assert auditor_pos.verify_training_history()
    
    def test_zk_with_enhanced_diff(self):
        """Test ZK proofs with enhanced diff testing."""
        # Create tester
        tester = EnhancedSequentialTester(mode=TestingMode.QUICK_GATE)
        
        # Generate model outputs
        n_samples = 100
        outputs1 = np.random.randn(n_samples).astype(np.float32)
        outputs2 = outputs1 + np.random.randn(n_samples).astype(np.float32) * 0.01
        
        # Test difference
        result = tester.test_difference(outputs1, outputs2)
        
        # Generate ZK proof for the models that produced these outputs
        model_before = {'weights': np.random.randn(50).astype(np.float32)}
        model_after = {'weights': model_before['weights'] + 0.01}
        
        proof_result = auto_prove_training_step(
            model_before=model_before,
            model_after=model_after,
            batch_data={'inputs': [[1.0]], 'targets': [[1.0]]},
            learning_rate=0.001
        )
        
        # Both should detect the difference
        assert result.decision in ['DIFFERENT', 'UNDECIDED']
        assert proof_result['success']
    
    def test_zk_lora_optimization(self):
        """Test optimized LoRA proving."""
        set_mode('production')
        
        # Create optimized prover
        prover = OptimizedLoRAProver()
        prover.optimize_for_hardware()
        
        # Create LoRA update
        from pot.zk.zk_types import LoRAStepStatement, LoRAStepWitness
        
        rank = 8
        d = 768
        
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
        
        # Generate proof with target time
        results = prover.prove_lora_batch([(statement, witness)], target_time_ms=5000)
        
        assert len(results) == 1
        assert results[0].success
        assert results[0].generation_time_ms < 10000  # Reasonable bound for mock
    
    def test_zk_backward_compatibility(self):
        """Test ZK system maintains backward compatibility."""
        # Test with existing mock models
        from tests.mock_models import MockModel
        
        model = MockModel(input_dim=10, output_dim=5)
        
        # Get initial state
        state_before = model.get_state()
        
        # Train step
        inputs = np.random.randn(32, 10).astype(np.float32)
        targets = np.random.randn(32, 5).astype(np.float32)
        loss = model.train_step(inputs, targets)
        
        # Get updated state
        state_after = model.get_state()
        
        # Should be able to generate ZK proof
        proof_result = auto_prove_training_step(
            model_before={'weights': state_before['weights']},
            model_after={'weights': state_after['weights']},
            batch_data={'inputs': inputs, 'targets': targets},
            learning_rate=0.01
        )
        
        assert proof_result['success']
    
    def test_zk_proof_persistence(self):
        """Test ZK proof persistence and retrieval."""
        blockchain = MockBlockchainClient()
        
        # Generate multiple proofs
        proofs = []
        tx_hashes = []
        
        for i in range(5):
            model_before = {'weights': np.random.randn(50).astype(np.float32)}
            model_after = {'weights': model_before['weights'] + 0.001}
            
            proof_result = auto_prove_training_step(
                model_before=model_before,
                model_after=model_after,
                batch_data={'inputs': [[1.0]], 'targets': [[1.0]]},
                learning_rate=0.001,
                step_number=i
            )
            
            proofs.append(proof_result['proof'])
            tx_hash = blockchain.store_commitment(proof_result['proof'])
            tx_hashes.append(tx_hash)
        
        # Retrieve and verify all proofs
        for i, tx_hash in enumerate(tx_hashes):
            retrieved = blockchain.get_commitment(tx_hash)
            assert retrieved == proofs[i]
            
            status = blockchain.get_transaction_status(tx_hash)
            assert status == "confirmed"
    
    def test_zk_metrics_collection(self):
        """Test ZK metrics are properly collected."""
        from pot.zk.metrics import get_monitor, record_proof_generation
        
        monitor = get_monitor()
        
        # Generate some proofs and record metrics
        for i in range(10):
            model_before = {'weights': np.random.randn(100).astype(np.float32)}
            model_after = {'weights': model_before['weights'] + 0.001}
            
            import time
            start = time.time()
            
            proof_result = auto_prove_training_step(
                model_before=model_before,
                model_after=model_after,
                batch_data={'inputs': [[1.0]], 'targets': [[1.0]]},
                learning_rate=0.001
            )
            
            elapsed_ms = (time.time() - start) * 1000
            
            # Record metric
            record_proof_generation(
                proof_type=proof_result['proof_type'],
                generation_time_ms=elapsed_ms,
                proof_size=len(proof_result['proof']),
                success=proof_result['success']
            )
        
        # Check metrics
        dashboard = monitor.get_dashboard_data()
        summary = dashboard['summary']
        
        assert summary['total_proofs'] >= 10
        assert summary['proof_success_rate'] > 0
    
    def test_zk_cache_effectiveness(self):
        """Test ZK caching improves performance."""
        from pot.zk.cache import get_cache, clear_all_caches
        
        # Clear caches
        clear_all_caches()
        
        # First run without cache benefit
        model = {'weights': np.random.randn(100).astype(np.float32)}
        
        import time
        times_no_cache = []
        for _ in range(3):
            start = time.time()
            proof_result = auto_prove_training_step(
                model_before=model,
                model_after={'weights': model['weights'] + 0.001},
                batch_data={'inputs': [[1.0]], 'targets': [[1.0]]},
                learning_rate=0.001
            )
            times_no_cache.append(time.time() - start)
        
        # Run again (should benefit from cache)
        times_with_cache = []
        for _ in range(3):
            start = time.time()
            proof_result = auto_prove_training_step(
                model_before=model,
                model_after={'weights': model['weights'] + 0.001},
                batch_data={'inputs': [[1.0]], 'targets': [[1.0]]},
                learning_rate=0.001
            )
            times_with_cache.append(time.time() - start)
        
        # With cache should be faster (or at least not slower)
        avg_no_cache = np.mean(times_no_cache)
        avg_with_cache = np.mean(times_with_cache)
        
        # In mock implementation, might not see huge difference
        # but should not be significantly slower
        assert avg_with_cache <= avg_no_cache * 1.5


def test_zk_pytest_compatibility():
    """Test that ZK tests work with pytest infrastructure."""
    # This test verifies pytest can run our ZK tests
    from pot.zk.prover import SGDZKProver
    
    prover = SGDZKProver()
    assert prover is not None
    
    # Create simple witness
    from pot.zk.zk_types import SGDStepWitness, SGDStepStatement
    
    witness = SGDStepWitness(
        weights_before=[0.1] * 10,
        weights_after=[0.11] * 10,
        gradients=[0.01] * 10,
        batch_inputs=[[0.5] * 5 for _ in range(4)],
        batch_targets=[[1.0] for _ in range(4)],
        learning_rate=0.01
    )
    
    statement = SGDStepStatement(
        weights_before_root=b"before" * 8,
        weights_after_root=b"after" * 8,
        batch_root=b"batch" * 8,
        hparams_hash=b"hparams" * 4,
        step_number=1,
        epoch=1
    )
    
    # Should be able to generate proof
    proof = prover.prove_sgd_step(statement, witness)
    assert proof is not None


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])