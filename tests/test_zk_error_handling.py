"""
Tests for ZK proof error handling and retry logic.
"""

import pytest
import sys
import tempfile
import time
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.append(str(Path(__file__).parent.parent))

from pot.zk.exceptions import (
    ProverNotFoundError, ProofGenerationError, WitnessBuilderError,
    InvalidModelStateError, RetryExhaustedError, ConfigurationError
)
from pot.zk.auto_prover import AutoProver
from pot.zk.auditor_integration import ZKAuditorIntegration, ZKAuditorConfig, ProofFailureAction


class TestZKExceptions:
    """Test ZK exception classes."""
    
    def test_prover_not_found_error(self):
        """Test ProverNotFoundError formatting."""
        error = ProverNotFoundError("prove_lora_stdin", ["/usr/bin", "/usr/local/bin"])
        
        assert "prove_lora_stdin" in str(error)
        assert "/usr/bin" in str(error)
        assert error.binary_name == "prove_lora_stdin"
        assert len(error.search_paths) == 2
    
    def test_proof_generation_error(self):
        """Test ProofGenerationError formatting."""
        error = ProofGenerationError(
            "Circuit constraint violation",
            binary_name="prove_sgd_stdin",
            exit_code=1,
            stderr="Error: Invalid witness data"
        )
        
        assert "Circuit constraint violation" in str(error)
        assert "prove_sgd_stdin" in str(error)
        assert "exit code: 1" in str(error)
        assert "Invalid witness data" in str(error)
    
    def test_retry_exhausted_error(self):
        """Test RetryExhaustedError with last error."""
        last_error = ProofGenerationError("Prover crashed")
        error = RetryExhaustedError("Proof generation", 3, last_error)
        
        assert "failed after 3 attempts" in str(error)
        assert "Prover crashed" in str(error)
        assert error.attempts == 3
        assert error.last_error == last_error


class TestAutoProver:
    """Test AutoProver error handling and retry logic."""
    
    def test_model_type_detection(self):
        """Test model type detection."""
        prover = AutoProver()
        
        # SGD model
        sgd_model = {'weights': np.random.randn(10, 10)}
        model_type = prover.detect_model_type(sgd_model, sgd_model)
        assert model_type == "sgd"
        
        # LoRA model
        lora_model = {
            'lora_A.weight': np.random.randn(768, 16),
            'lora_B.weight': np.random.randn(16, 768),
            'base.weight': np.random.randn(768, 768)
        }
        model_type = prover.detect_model_type(lora_model, lora_model)
        assert model_type == "lora"
    
    def test_invalid_model_state_detection(self):
        """Test detection of invalid model states."""
        prover = AutoProver()
        
        # Empty models
        with pytest.raises(InvalidModelStateError) as exc_info:
            prover.detect_model_type({}, {})
        assert "empty" in str(exc_info.value).lower()
        
        # Mismatched architectures
        model1 = {'layer1': np.random.randn(10, 10)}
        model2 = {'layer2': np.random.randn(10, 10)}  # Different key
        
        with pytest.raises(InvalidModelStateError) as exc_info:
            prover.detect_model_type(model1, model2)
        assert "mismatch" in str(exc_info.value).lower()
    
    def test_prover_binary_not_found(self):
        """Test handling when prover binary is not found."""
        prover = AutoProver()
        
        model_before = {'weights': np.random.randn(10, 10)}
        model_after = {'weights': model_before['weights'] + 0.01}
        batch_data = {'inputs': np.random.randn(32, 10), 'targets': np.random.randn(32, 10)}
        
        # Mock the binary check to fail
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(ProverNotFoundError) as exc_info:
                prover.prove_sgd_step(model_before, model_after, batch_data, 0.01)
            
            assert "prove_sgd_stdin" in str(exc_info.value)
    
    def test_witness_builder_error(self):
        """Test witness builder error handling."""
        prover = AutoProver()
        
        model_before = {'weights': np.random.randn(10, 10)}
        model_after = {'weights': model_before['weights'] + 0.01}
        batch_data = {'inputs': np.random.randn(32, 10), 'targets': np.random.randn(32, 10)}
        
        # Invalid learning rate
        with pytest.raises(WitnessBuilderError) as exc_info:
            prover.prove_sgd_step(model_before, model_after, batch_data, -0.01)  # Negative LR
        
        assert "learning rate" in str(exc_info.value).lower()
    
    def test_retry_logic_with_exponential_backoff(self):
        """Test retry logic with exponential backoff."""
        prover = AutoProver(max_retries=2, retry_delay=0.1, retry_exponential=True)
        
        # Mock function that fails twice then succeeds
        call_count = 0
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ProofGenerationError(f"Attempt {call_count} failed")
            return {"success": True}
        
        start_time = time.time()
        result = prover._retry_with_backoff(failing_function)
        elapsed = time.time() - start_time
        
        assert result["success"] is True
        assert call_count == 3
        # Should have delays: 0.1 + 0.2 = 0.3s minimum
        assert elapsed >= 0.25
    
    def test_retry_exhausted(self):
        """Test retry exhaustion."""
        prover = AutoProver(max_retries=2, retry_delay=0.01)
        
        def always_failing_function():
            raise ProofGenerationError("Always fails")
        
        with pytest.raises(RetryExhaustedError) as exc_info:
            prover._retry_with_backoff(always_failing_function)
        
        assert exc_info.value.attempts == 3  # max_retries + 1
        assert "Always fails" in str(exc_info.value)
    
    def test_fail_fast_mode(self):
        """Test fail-fast mode skips retries."""
        prover = AutoProver(fail_fast=True, max_retries=3)
        
        model_before = {'weights': np.random.randn(10, 10)}
        model_after = {'weights': model_before['weights'] + 0.01}
        batch_data = {'inputs': np.random.randn(32, 10), 'targets': np.random.randn(32, 10)}
        
        with patch('pathlib.Path.exists', return_value=False):
            start_time = time.time()
            with pytest.raises(ProverNotFoundError):
                prover.auto_prove_training_step(
                    model_before, model_after, batch_data, 0.01
                )
            elapsed = time.time() - start_time
            
            # Should fail immediately without retries
            assert elapsed < 0.1
    
    def test_auto_prove_with_error_handling(self):
        """Test auto_prove_training_step with various error scenarios."""
        prover = AutoProver(max_retries=1, retry_delay=0.01)
        
        # Test with valid but mock data
        model_before = {'weights': np.random.randn(10, 10)}
        model_after = {'weights': model_before['weights'] + 0.01}
        batch_data = {'inputs': np.random.randn(32, 10), 'targets': np.random.randn(32, 10)}
        
        # Mock successful binary existence and proof generation
        with patch('pathlib.Path.exists', return_value=True):
            result = prover.auto_prove_training_step(
                model_before, model_after, batch_data, 0.01, 1, 1
            )
            
            assert result['success'] is True
            assert result['proof_type'] == 'sgd'
            assert 'metadata' in result


class TestZKAuditorIntegration:
    """Test ZK auditor integration with different failure modes."""
    
    def test_disabled_zk_integration(self):
        """Test integration when ZK proofs are disabled."""
        config = ZKAuditorConfig(enabled=False)
        integration = ZKAuditorIntegration(config)
        
        result = integration.generate_training_proof(
            {}, {}, {}, 0.01, 1, 1
        )
        
        assert result is None
    
    def test_continue_on_failure(self):
        """Test CONTINUE failure action."""
        config = ZKAuditorConfig(
            enabled=True,
            failure_action=ProofFailureAction.CONTINUE,
            log_failures=True
        )
        integration = ZKAuditorIntegration(config)
        
        # Mock prover to always fail
        with patch.object(integration.prover, 'auto_prove_training_step',
                         side_effect=ProofGenerationError("Mock failure")):
            
            result = integration.generate_training_proof(
                {'weights': np.random.randn(10, 10)},
                {'weights': np.random.randn(10, 10)},
                {'inputs': [[1.0]], 'targets': [[1.0]]},
                0.01, 1, 1
            )
            
            # Should return None but not raise exception
            assert result is None
            assert integration.failure_count == 1
            assert integration.success_count == 0
    
    def test_fail_fast_on_failure(self):
        """Test FAIL_FAST failure action."""
        config = ZKAuditorConfig(
            enabled=True,
            failure_action=ProofFailureAction.FAIL_FAST
        )
        integration = ZKAuditorIntegration(config)
        
        # Mock prover to always fail
        with patch.object(integration.prover, 'auto_prove_training_step',
                         side_effect=ProofGenerationError("Mock failure")):
            
            with pytest.raises(ProofGenerationError) as exc_info:
                integration.generate_training_proof(
                    {'weights': np.random.randn(10, 10)},
                    {'weights': np.random.randn(10, 10)},
                    {'inputs': [[1.0]], 'targets': [[1.0]]},
                    0.01, 1, 1
                )
            
            assert "Mock failure" in str(exc_info.value)
    
    def test_successful_proof_generation(self):
        """Test successful proof generation."""
        config = ZKAuditorConfig(enabled=True)
        integration = ZKAuditorIntegration(config)
        
        # Mock successful proof generation
        mock_result = {
            'success': True,
            'proof': b'mock_proof_data',
            'proof_type': 'sgd',
            'metadata': {'step_number': 1}
        }
        
        with patch.object(integration.prover, 'auto_prove_training_step',
                         return_value=mock_result):
            
            result = integration.generate_training_proof(
                {'weights': np.random.randn(10, 10)},
                {'weights': np.random.randn(10, 10)},
                {'inputs': [[1.0]], 'targets': [[1.0]]},
                0.01, 1, 1
            )
            
            assert result == mock_result
            assert integration.success_count == 1
            assert integration.failure_count == 0
    
    def test_statistics(self):
        """Test statistics collection."""
        config = ZKAuditorConfig(enabled=True, failure_action=ProofFailureAction.CONTINUE)
        integration = ZKAuditorIntegration(config)
        
        # Simulate some successes and failures
        integration.success_count = 7
        integration.failure_count = 3
        
        stats = integration.get_stats()
        
        assert stats['enabled'] is True
        assert stats['total_attempts'] == 10
        assert stats['successful_proofs'] == 7
        assert stats['failed_proofs'] == 3
        assert stats['success_rate'] == 0.7
        assert stats['failure_action'] == 'continue'
    
    def test_batch_data_conversion(self):
        """Test batch data format conversion."""
        config = ZKAuditorConfig(enabled=True)
        integration = ZKAuditorIntegration(config)
        
        # Mock successful proof generation to check data conversion
        def check_batch_data(*args, **kwargs):
            # Verify that batch data was converted to numpy arrays
            batch_data = args[2]  # Third argument is batch_data
            assert isinstance(batch_data['inputs'], np.ndarray)
            assert isinstance(batch_data['targets'], np.ndarray)
            return {'success': True, 'proof': b'test', 'proof_type': 'sgd', 'metadata': {}}
        
        with patch.object(integration.prover, 'auto_prove_training_step',
                         side_effect=check_batch_data):
            
            result = integration.generate_training_proof(
                {'weights': np.random.randn(10, 10)},
                {'weights': np.random.randn(10, 10)},
                {'inputs': [[1.0], [2.0]], 'targets': [[0.5], [1.5]]},  # Lists
                0.01, 1, 1
            )
            
            assert result is not None


def test_integration_with_training_auditor():
    """Test integration pattern with training auditor."""
    from pot.zk.auditor_integration import create_zk_integration
    
    # Create integration with different configurations
    
    # Development mode: continue on failures, log everything
    dev_integration = create_zk_integration(
        enabled=True,
        failure_action="continue",
        max_retries=1
    )
    
    # Production mode: fail fast, no retries
    prod_integration = create_zk_integration(
        enabled=True,
        failure_action="fail_fast",
        max_retries=0
    )
    
    # Disabled mode: no proofs generated
    disabled_integration = create_zk_integration(enabled=False)
    
    assert dev_integration.config.enabled is True
    assert dev_integration.config.failure_action == ProofFailureAction.CONTINUE
    assert dev_integration.config.max_retries == 1
    
    assert prod_integration.config.failure_action == ProofFailureAction.FAIL_FAST
    
    assert disabled_integration.config.enabled is False


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])