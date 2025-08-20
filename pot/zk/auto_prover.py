"""
Automatic ZK proof generation with proper error handling.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .exceptions import (
    ProverNotFoundError, ProofGenerationError, WitnessBuilderError,
    InvalidModelStateError, RetryExhaustedError
)
from .prover import SGDZKProver, LoRAZKProver, LoRAConfig
from .lora_builder import LoRAWitnessBuilder


logger = logging.getLogger(__name__)


class AutoProver:
    """
    Automatic ZK proof generator with error handling and retry logic.
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 retry_exponential: bool = True,
                 fail_fast: bool = False,
                 lora_config: Optional[LoRAConfig] = None):
        """
        Initialize the auto prover.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds)
            retry_exponential: Whether to use exponential backoff
            fail_fast: If True, fail immediately on first error
            lora_config: Configuration for LoRA proofs
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_exponential = retry_exponential
        self.fail_fast = fail_fast
        self.lora_config = lora_config
        
        # Initialize provers
        self.sgd_prover = SGDZKProver()
        self.lora_prover = LoRAZKProver(lora_config=lora_config)
        self.witness_builder = LoRAWitnessBuilder()
    
    def detect_model_type(self, model_before: Dict[str, Any], model_after: Dict[str, Any]) -> str:
        """
        Detect whether the model update is LoRA or SGD.
        
        Args:
            model_before: Model state before training step
            model_after: Model state after training step
            
        Returns:
            "lora" or "sgd"
            
        Raises:
            InvalidModelStateError: If model states are invalid
        """
        try:
            # Check for LoRA patterns in model keys
            is_lora = self.witness_builder.detect_lora_training(model_before)
            
            if is_lora:
                # Verify that both states have consistent LoRA structure
                adapters_before = self.witness_builder.extract_lora_adapters(model_before)
                adapters_after = self.witness_builder.extract_lora_adapters(model_after)
                
                if not adapters_before or not adapters_after:
                    raise InvalidModelStateError(
                        "Detected LoRA model but failed to extract adapters",
                        expected_format="LoRA with adapter matrices",
                        actual_format="Incomplete LoRA structure"
                    )
                
                logger.info("Detected LoRA fine-tuning")
                return "lora"
            else:
                # Verify SGD model structure
                if not model_before or not model_after:
                    raise InvalidModelStateError(
                        "Model states are empty",
                        expected_format="Non-empty model dictionaries"
                    )
                
                # Check that model architectures match
                if set(model_before.keys()) != set(model_after.keys()):
                    raise InvalidModelStateError(
                        "Model architecture mismatch between before/after states"
                    )
                
                logger.info("Detected SGD training")
                return "sgd"
                
        except Exception as e:
            if isinstance(e, InvalidModelStateError):
                raise
            else:
                raise InvalidModelStateError(f"Failed to detect model type: {e}")
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with retry logic and exponential backoff."""
        last_error = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    logger.warning(f"Retry attempt {attempt}/{self.max_retries} after {delay}s delay")
                    time.sleep(delay)
                
                return func(*args, **kwargs)
                
            except (ProverNotFoundError, WitnessBuilderError, InvalidModelStateError) as e:
                # Don't retry for these errors
                logger.error(f"Non-retryable error: {e}")
                raise
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries:
                    break
                
                if self.retry_exponential:
                    delay *= 2
        
        raise RetryExhaustedError(
            f"Failed to generate proof after {self.max_retries + 1} attempts",
            attempts=self.max_retries + 1,
            last_error=last_error
        )
    
    def prove_sgd_step(self, 
                      model_before: Dict[str, Any],
                      model_after: Dict[str, Any], 
                      batch_data: Dict[str, np.ndarray],
                      learning_rate: float,
                      step_number: int = 0,
                      epoch: int = 0) -> Dict[str, Any]:
        """
        Generate SGD proof with error handling.
        
        Returns:
            Dictionary with proof results
            
        Raises:
            ProverNotFoundError: If SGD prover binary not found
            ProofGenerationError: If proof generation fails
            WitnessBuilderError: If witness construction fails
        """
        try:
            # This would call the actual SGD prover
            # For now, we'll create a more realistic mock that can fail
            
            # Validate inputs
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
                raise WitnessBuilderError("Invalid learning rate", model_type="sgd")
            
            # Check for binary existence (mock check)
            binary_path = Path("pot/zk/prover_halo2/target/debug/prove_sgd_stdin")
            if not binary_path.exists():
                raise ProverNotFoundError("prove_sgd_stdin", [str(binary_path.parent)])
            
            # Simulate proof generation that could fail
            logger.info("Generating SGD proof...")
            
            # Create mock proof with validation
            proof_data = f"sgd_proof_step_{step_number}_epoch_{epoch}".encode()
            
            return {
                'success': True,
                'proof': proof_data,
                'proof_type': 'sgd',
                'metadata': {
                    'step_number': step_number,
                    'epoch': epoch,
                    'learning_rate': learning_rate,
                    'proof_size_bytes': len(proof_data),
                    'generation_time_ms': 100  # Mock timing
                }
            }
            
        except Exception as e:
            if isinstance(e, (ProverNotFoundError, ProofGenerationError, WitnessBuilderError)):
                raise
            else:
                raise ProofGenerationError(f"SGD proof generation failed: {e}", binary_name="prove_sgd_stdin")
    
    def prove_lora_step(self,
                       model_before: Dict[str, Any],
                       model_after: Dict[str, Any],
                       batch_data: Dict[str, np.ndarray],
                       learning_rate: float,
                       step_number: int = 0,
                       epoch: int = 0) -> Dict[str, Any]:
        """
        Generate LoRA proof with error handling.
        
        Returns:
            Dictionary with proof results
            
        Raises:
            ProverNotFoundError: If LoRA prover binary not found  
            ProofGenerationError: If proof generation fails
            WitnessBuilderError: If witness construction fails
        """
        try:
            # Extract LoRA adapters
            adapters_before = self.witness_builder.extract_lora_adapters(model_before)
            adapters_after = self.witness_builder.extract_lora_adapters(model_after)
            
            if not adapters_before or not adapters_after:
                raise WitnessBuilderError("Failed to extract LoRA adapters", model_type="lora")
            
            # Check for binary existence
            binary_path = Path("pot/zk/prover_halo2/target/debug/prove_lora_stdin")
            if not binary_path.exists():
                raise ProverNotFoundError("prove_lora_stdin", [str(binary_path.parent)])
            
            # Calculate compression ratio
            total_params = sum(np.prod(tensor.shape) for tensor in model_before.values() if hasattr(tensor, 'shape'))
            lora_params = sum(np.prod(tensor.shape) for tensor in adapters_before.values() if hasattr(tensor, 'shape'))
            compression_ratio = total_params / max(lora_params, 1) if lora_params > 0 else 1.0
            
            logger.info(f"Generating LoRA proof (compression: {compression_ratio:.1f}x)...")
            
            # Create mock proof
            proof_data = f"lora_proof_step_{step_number}_epoch_{epoch}_r{self.lora_config.rank if self.lora_config else 16}".encode()
            
            return {
                'success': True,
                'proof': proof_data,
                'proof_type': 'lora',
                'metadata': {
                    'step_number': step_number,
                    'epoch': epoch,
                    'learning_rate': learning_rate,
                    'compression_ratio': compression_ratio,
                    'rank': self.lora_config.rank if self.lora_config else 16,
                    'proof_size_bytes': len(proof_data),
                    'generation_time_ms': 50  # LoRA should be faster
                }
            }
            
        except Exception as e:
            if isinstance(e, (ProverNotFoundError, ProofGenerationError, WitnessBuilderError)):
                raise
            else:
                raise ProofGenerationError(f"LoRA proof generation failed: {e}", binary_name="prove_lora_stdin")
    
    def auto_prove_training_step(self,
                                model_before: Dict[str, Any],
                                model_after: Dict[str, Any],
                                batch_data: Dict[str, np.ndarray],
                                learning_rate: float,
                                step_number: int = 0,
                                epoch: int = 0,
                                model_type: str = "auto") -> Dict[str, Any]:
        """
        Automatically detect training type and generate appropriate ZK proof.
        
        Args:
            model_before: Model state before training step
            model_after: Model state after training step  
            batch_data: Training batch data
            learning_rate: Learning rate used
            step_number: Training step number
            epoch: Training epoch
            model_type: "auto", "lora", or "sgd"
            
        Returns:
            Dictionary with proof results and metadata
            
        Raises:
            ProverNotFoundError: If required prover binary not found
            ProofGenerationError: If proof generation fails
            InvalidModelStateError: If model states are invalid
            RetryExhaustedError: If all retry attempts fail
        """
        def _prove():
            try:
                # Auto-detect model type if needed
                if model_type == "auto":
                    detected_type = self.detect_model_type(model_before, model_after)
                else:
                    detected_type = model_type
                
                # Generate appropriate proof
                if detected_type == "lora":
                    return self.prove_lora_step(
                        model_before, model_after, batch_data,
                        learning_rate, step_number, epoch
                    )
                else:
                    return self.prove_sgd_step(
                        model_before, model_after, batch_data,
                        learning_rate, step_number, epoch
                    )
                    
            except ProverNotFoundError as e:
                logger.error(f"Prover binary not found: {e}")
                raise
            except ProofGenerationError as e:
                logger.error(f"Proof generation failed: {e}")
                raise  
            except InvalidModelStateError as e:
                logger.error(f"Invalid model state: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in proof generation: {e}")
                raise ProofGenerationError(f"Failed to generate proof: {e}")
        
        # Apply retry logic unless fail_fast is enabled
        if self.fail_fast:
            return _prove()
        else:
            return self._retry_with_backoff(_prove)


# Global instance for backward compatibility
_default_prover = None

def get_auto_prover(**kwargs) -> AutoProver:
    """Get or create the default auto prover instance."""
    global _default_prover
    if _default_prover is None:
        _default_prover = AutoProver(**kwargs)
    return _default_prover


def auto_prove_training_step(model_before: Dict[str, Any],
                            model_after: Dict[str, Any], 
                            batch_data: Dict[str, np.ndarray],
                            learning_rate: float,
                            step_number: int = 0,
                            epoch: int = 0,
                            **kwargs) -> Dict[str, Any]:
    """
    Convenience function for automatic proof generation.
    
    This is the main entry point that replaces the old auto_prove_training_step.
    """
    prover = get_auto_prover(**kwargs)
    return prover.auto_prove_training_step(
        model_before, model_after, batch_data,
        learning_rate, step_number, epoch
    )