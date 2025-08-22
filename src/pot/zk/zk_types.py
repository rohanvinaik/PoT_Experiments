"""
Simple types for ZK proof integration that match the Rust implementation.
"""

from dataclasses import dataclass
from typing import List, Union, Optional


@dataclass
class SGDStepStatement:
    """Public statement for SGD step verification."""
    W_t_root: Union[bytes, str]      # Merkle root of weights before
    batch_root: Union[bytes, str]    # Merkle root of batch data
    hparams_hash: Union[bytes, str]  # Hash of hyperparameters
    W_t1_root: Union[bytes, str]     # Merkle root of weights after
    step_nonce: int
    step_number: int
    epoch: int


@dataclass
class SGDStepWitness:
    """Private witness for SGD step verification."""
    weights_before: List[float]  # Flattened weight values before update
    weights_after: List[float]   # Flattened weight values after update
    batch_inputs: List[float]    # Flattened batch input values
    batch_targets: List[float]   # Flattened batch target values
    learning_rate: float
    loss_value: float = 0.5


@dataclass
class LoRAStepStatement:
    """Public statement for LoRA fine-tuning step.
    
    LoRA updates are much smaller since only adapter matrices change:
    - W_effective = W_base + α * (B × A)
    - A is d_in × r, B is r × d_out (r << d_in, d_out)
    """
    base_weights_root: Union[bytes, str]      # Root of frozen base model
    adapter_a_root_before: Union[bytes, str]  # Root of adapter A before
    adapter_b_root_before: Union[bytes, str]  # Root of adapter B before
    adapter_a_root_after: Union[bytes, str]   # Root of adapter A after
    adapter_b_root_after: Union[bytes, str]   # Root of adapter B after
    batch_root: Union[bytes, str]             # Root of training batch
    hparams_hash: Union[bytes, str]           # Hash of hyperparameters
    rank: int                                  # LoRA rank r
    alpha: float                               # LoRA scaling factor
    step_number: int
    epoch: int


@dataclass
class LoRAStepWitness:
    """Private witness for LoRA fine-tuning step.
    
    Much smaller than full SGD witness since only adapters are included.
    """
    # Adapter matrices before update (flattened)
    adapter_a_before: List[float]  # d_in × r matrix
    adapter_b_before: List[float]  # r × d_out matrix
    
    # Adapter matrices after update (flattened)
    adapter_a_after: List[float]
    adapter_b_after: List[float]
    
    # Gradients for adapters only
    adapter_a_gradients: List[float]
    adapter_b_gradients: List[float]
    
    # Training batch
    batch_inputs: List[float]
    batch_targets: List[float]
    
    # Training parameters
    learning_rate: float
    loss_value: float = 0.5
    
    # Optional Merkle proofs
    adapter_a_proof: Optional[List[bytes]] = None
    adapter_b_proof: Optional[List[bytes]] = None


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    rank: int = 16                          # LoRA rank (typically 4-64)
    alpha: float = 32.0                     # Scaling factor
    target_modules: List[str] = None        # Which modules to apply LoRA to
    dropout: float = 0.1                    # LoRA dropout
    bias: str = "none"                      # How to handle biases
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]  # Common for transformers
    
    @property
    def scaling(self) -> float:
        """Get the LoRA scaling factor."""
        return self.alpha / self.rank
    
    def compression_ratio(self, d_in: int, d_out: int) -> float:
        """Calculate compression ratio vs full fine-tuning."""
        full_params = d_in * d_out
        lora_params = self.rank * (d_in + d_out)
        return full_params / lora_params


@dataclass
class LoRAProofMetadata:
    """Metadata about LoRA proof efficiency."""
    full_model_params: int       # Total parameters in base model
    lora_params: int            # Total LoRA parameters
    compression_ratio: float    # full_model_params / lora_params
    proof_size_bytes: int       # Size of the ZK proof
    proof_generation_ms: int    # Time to generate proof
    circuit_constraints: int    # Number of circuit constraints
    memory_usage_mb: float     # Memory used during proof generation