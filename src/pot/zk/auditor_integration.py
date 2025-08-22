"""
Integration layer for ZK proofs with TrainingProvenanceAuditor.
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum

from .exceptions import (
    ProverNotFoundError, ProofGenerationError, 
    InvalidModelStateError, RetryExhaustedError
)
from .auto_prover import AutoProver, get_auto_prover


logger = logging.getLogger(__name__)


class ProofFailureAction(Enum):
    """Actions to take when proof generation fails."""
    FAIL_FAST = "fail_fast"      # Raise exception immediately
    CONTINUE = "continue"        # Log error and continue without proof
    RETRY = "retry"              # Retry with exponential backoff
    FALLBACK = "fallback"        # Try alternative proof method


class ZKAuditorConfig:
    """Configuration for ZK proof integration with auditor."""
    
    def __init__(self,
                 enabled: bool = True,
                 failure_action: ProofFailureAction = ProofFailureAction.CONTINUE,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 retry_exponential: bool = True,
                 log_failures: bool = True,
                 store_failure_info: bool = True):
        """
        Initialize ZK auditor configuration.
        
        Args:
            enabled: Whether to generate ZK proofs
            failure_action: What to do when proof generation fails
            max_retries: Maximum retry attempts
            retry_delay: Initial delay between retries (seconds)
            retry_exponential: Use exponential backoff for retries
            log_failures: Log proof generation failures
            store_failure_info: Store failure info in audit log
        """
        self.enabled = enabled
        self.failure_action = failure_action
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_exponential = retry_exponential
        self.log_failures = log_failures
        self.store_failure_info = store_failure_info


class ZKAuditorIntegration:
    """
    ZK proof integration for TrainingProvenanceAuditor.
    
    Handles proof generation with configurable error handling.
    """
    
    def __init__(self, config: ZKAuditorConfig = None):
        """
        Initialize ZK auditor integration.
        
        Args:
            config: Configuration for ZK proof handling
        """
        self.config = config or ZKAuditorConfig()
        self.prover = None
        self.failure_count = 0
        self.success_count = 0
        
        if self.config.enabled:
            try:
                self.prover = get_auto_prover(
                    max_retries=self.config.max_retries,
                    retry_delay=self.config.retry_delay,
                    retry_exponential=self.config.retry_exponential,
                    fail_fast=(self.config.failure_action == ProofFailureAction.FAIL_FAST)
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ZK prover: {e}")
                if self.config.failure_action == ProofFailureAction.FAIL_FAST:
                    raise
    
    def generate_training_proof(self,
                               model_before: Dict[str, Any],
                               model_after: Dict[str, Any], 
                               batch_data: Dict[str, Any],
                               learning_rate: float,
                               step_number: int = 0,
                               epoch: int = 0) -> Optional[Dict[str, Any]]:
        """
        Generate ZK proof for training step with error handling.
        
        Args:
            model_before: Model state before training
            model_after: Model state after training
            batch_data: Training batch data
            learning_rate: Learning rate used
            step_number: Training step number
            epoch: Training epoch
            
        Returns:
            Proof result dictionary or None if proof generation failed
            and failure_action is CONTINUE
        """
        if not self.config.enabled or self.prover is None:
            return None
            
        try:
            # Convert batch data format if needed
            import numpy as np
            if 'inputs' in batch_data and 'targets' in batch_data:
                batch_array = {
                    'inputs': np.array(batch_data['inputs'], dtype=np.float32),
                    'targets': np.array(batch_data['targets'], dtype=np.float32)
                }
            else:
                batch_array = batch_data
            
            # Generate proof
            result = self.prover.auto_prove_training_step(
                model_before, model_after, batch_array,
                learning_rate, step_number, epoch
            )
            
            self.success_count += 1
            logger.info(f"ZK proof generated successfully (type: {result.get('proof_type', 'unknown')})")
            
            return result
            
        except ProverNotFoundError as e:
            self._handle_proof_failure("Prover binary not found", e, step_number, epoch)
            
        except InvalidModelStateError as e:
            self._handle_proof_failure("Invalid model state", e, step_number, epoch)
            
        except ProofGenerationError as e:
            self._handle_proof_failure("Proof generation failed", e, step_number, epoch)
            
        except RetryExhaustedError as e:
            self._handle_proof_failure("All retry attempts failed", e, step_number, epoch)
            
        except Exception as e:
            self._handle_proof_failure("Unexpected error", e, step_number, epoch)
        
        return None
    
    def _handle_proof_failure(self, error_type: str, error: Exception, 
                             step_number: int, epoch: int) -> None:
        """Handle proof generation failure according to configuration."""
        self.failure_count += 1
        
        error_info = {
            'error_type': error_type,
            'error_message': str(error),
            'step_number': step_number,
            'epoch': epoch,
            'failure_count': self.failure_count
        }
        
        if self.config.log_failures:
            logger.error(f"ZK proof failure: {error_type} - {error} (step {step_number}, epoch {epoch})")
        
        if self.config.failure_action == ProofFailureAction.FAIL_FAST:
            raise error
        elif self.config.failure_action == ProofFailureAction.CONTINUE:
            logger.warning("Continuing training without ZK proof")
        # Additional actions like RETRY and FALLBACK would be implemented here
        
        # Store failure info for later analysis if configured
        if self.config.store_failure_info:
            self._store_failure_info(error_info)
    
    def _store_failure_info(self, error_info: Dict[str, Any]) -> None:
        """Store failure information for later analysis."""
        # This could write to a file, database, or in-memory store
        logger.debug(f"Storing failure info: {error_info}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get proof generation statistics."""
        total = self.success_count + self.failure_count
        success_rate = (self.success_count / total) if total > 0 else 0.0
        
        return {
            'enabled': self.config.enabled,
            'total_attempts': total,
            'successful_proofs': self.success_count,
            'failed_proofs': self.failure_count,
            'success_rate': success_rate,
            'failure_action': self.config.failure_action.value
        }


def create_zk_integration(enabled: bool = True,
                         failure_action: str = "continue",
                         max_retries: int = 3) -> ZKAuditorIntegration:
    """
    Convenience function to create ZK integration with common settings.
    
    Args:
        enabled: Whether to enable ZK proof generation
        failure_action: Action on failure ("fail_fast", "continue", "retry", "fallback")
        max_retries: Maximum retry attempts
        
    Returns:
        Configured ZKAuditorIntegration instance
    """
    try:
        action = ProofFailureAction(failure_action)
    except ValueError:
        logger.warning(f"Invalid failure action '{failure_action}', using 'continue'")
        action = ProofFailureAction.CONTINUE
    
    config = ZKAuditorConfig(
        enabled=enabled,
        failure_action=action,
        max_retries=max_retries
    )
    
    return ZKAuditorIntegration(config)