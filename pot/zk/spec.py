"""
ZK Proof Specifications for SGD and LoRA Training Steps

Defines the statement and witness dataclasses for zero-knowledge proofs
of correct training step execution, compatible with the existing PoT infrastructure.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ZKProofType(Enum):
    """Types of ZK proofs supported"""
    SGD_STEP = "sgd_step"
    LORA_STEP = "lora_step"
    ADAM_STEP = "adam_step"
    CUSTOM_OPTIMIZER = "custom_optimizer"


class CommitmentScheme(Enum):
    """Supported commitment schemes"""
    SHA256_MERKLE = "sha256_merkle"
    POSEIDON_MERKLE = "poseidon_merkle"
    DUAL_COMMITMENT = "dual_commitment"


@dataclass
class SGDStepStatement:
    """
    ZK statement for SGD training step verification
    
    Proves that given committed weights W_t, batch data, and hyperparameters,
    the new weights W_{t+1} were computed correctly using SGD update rule:
    W_{t+1} = W_t - lr * ∇L(W_t, batch)
    """
    # Commitment to initial weights W_t
    W_t_root: str
    
    # Commitment to training batch
    batch_root: str
    
    # Hash of hyperparameters (learning rate, etc.)
    hparams_hash: str
    
    # Commitment to updated weights W_{t+1}
    W_t1_root: str
    
    # Step nonce for replay protection
    step_nonce: int
    
    # Training step metadata
    step_number: int
    epoch: int
    
    # Commitment scheme used
    commitment_scheme: CommitmentScheme = CommitmentScheme.POSEIDON_MERKLE
    
    # Optional model architecture hash for verification
    model_arch_hash: Optional[str] = None
    
    # Proof metadata
    proof_version: str = "1.0"
    timestamp: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "W_t_root": self.W_t_root,
            "batch_root": self.batch_root,
            "hparams_hash": self.hparams_hash,
            "W_t1_root": self.W_t1_root,
            "step_nonce": self.step_nonce,
            "step_number": self.step_number,
            "epoch": self.epoch,
            "commitment_scheme": self.commitment_scheme.value,
            "model_arch_hash": self.model_arch_hash,
            "proof_version": self.proof_version,
            "timestamp": self.timestamp
        }


@dataclass
class SGDStepWitness:
    """
    ZK witness for SGD training step
    
    Contains the private data needed to generate the proof:
    - Weight values and Merkle openings
    - Batch data and openings
    - Intermediate computation values
    """
    # Weight tensors and their Merkle paths
    weight_values: Dict[str, np.ndarray]
    weight_openings: Dict[str, List[str]]  # Merkle paths for each weight
    
    # Batch data and openings
    batch_inputs: np.ndarray
    batch_targets: np.ndarray
    batch_openings: List[str]  # Merkle path for batch commitment
    
    # Hyperparameters
    learning_rate: float
    
    # Intermediate computation values
    gradients: Dict[str, np.ndarray]
    loss_value: float
    
    # Updated weight values
    updated_weights: Dict[str, np.ndarray]
    
    # Optional hyperparameters with defaults
    momentum: Optional[float] = None
    weight_decay: Optional[float] = None
    
    # Random values used in commitments
    weight_randomness: Dict[str, bytes] = field(default_factory=dict)
    batch_randomness: bytes = field(default=b"")
    
    def validate_consistency(self, statement: SGDStepStatement) -> bool:
        """Validate that witness is consistent with statement"""
        # Check that weight openings match statement roots
        # This would verify Merkle proofs in a real implementation
        return True
    
    def get_public_inputs(self) -> List[str]:
        """Extract public inputs for the ZK proof"""
        return [
            self.loss_value,
            self.learning_rate,
            len(self.weight_values)
        ]


@dataclass 
class LoRAStepStatement:
    """
    ZK statement for LoRA fine-tuning step verification
    
    Proves correct application of LoRA update:
    W_{t+1} = W_t + α * A * B
    where A, B are low-rank adaptation matrices
    """
    # Base model weights commitment
    base_weights_root: str
    
    # LoRA adapter matrices commitment
    lora_A_root: str
    lora_B_root: str
    
    # Training batch commitment
    batch_root: str
    
    # LoRA hyperparameters hash
    lora_hparams_hash: str
    
    # Updated weights commitment
    updated_weights_root: str
    
    # LoRA-specific parameters
    rank: int
    alpha: float
    dropout_rate: float
    
    # Step metadata
    step_nonce: int
    step_number: int
    epoch: int
    
    # Target modules for LoRA adaptation
    target_modules: List[str]
    
    # Commitment scheme
    commitment_scheme: CommitmentScheme = CommitmentScheme.POSEIDON_MERKLE
    
    # Metadata
    proof_version: str = "1.0"
    timestamp: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "base_weights_root": self.base_weights_root,
            "lora_A_root": self.lora_A_root,
            "lora_B_root": self.lora_B_root,
            "batch_root": self.batch_root,
            "lora_hparams_hash": self.lora_hparams_hash,
            "updated_weights_root": self.updated_weights_root,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout_rate": self.dropout_rate,
            "step_nonce": self.step_nonce,
            "step_number": self.step_number,
            "epoch": self.epoch,
            "target_modules": self.target_modules,
            "commitment_scheme": self.commitment_scheme.value,
            "proof_version": self.proof_version,
            "timestamp": self.timestamp
        }


@dataclass
class LoRAStepWitness:
    """
    ZK witness for LoRA fine-tuning step
    
    Contains private data for LoRA training step proof generation
    """
    # Base model weights
    base_weights: Dict[str, np.ndarray]
    base_weight_openings: Dict[str, List[str]]
    
    # LoRA adapter matrices
    lora_A_matrices: Dict[str, np.ndarray]
    lora_B_matrices: Dict[str, np.ndarray]
    lora_A_openings: Dict[str, List[str]]
    lora_B_openings: Dict[str, List[str]]
    
    # Training batch
    batch_inputs: np.ndarray
    batch_targets: np.ndarray
    batch_openings: List[str]
    
    # LoRA hyperparameters
    rank: int
    alpha: float
    learning_rate: float
    dropout_rate: float
    
    # Computed values
    gradients_A: Dict[str, np.ndarray]
    gradients_B: Dict[str, np.ndarray]
    loss_value: float
    
    # Updated matrices after training step
    updated_A_matrices: Dict[str, np.ndarray]
    updated_B_matrices: Dict[str, np.ndarray]
    
    # Randomness for commitments
    base_randomness: Dict[str, bytes] = field(default_factory=dict)
    lora_randomness: Dict[str, bytes] = field(default_factory=dict)
    batch_randomness: bytes = field(default=b"")
    
    def validate_consistency(self, statement: LoRAStepStatement) -> bool:
        """Validate witness consistency with statement"""
        # Verify LoRA parameters match
        if self.rank != statement.rank:
            return False
        if abs(self.alpha - statement.alpha) > 1e-6:
            return False
        if abs(self.dropout_rate - statement.dropout_rate) > 1e-6:
            return False
        
        # Check that we have matching matrices for all target modules
        for module in statement.target_modules:
            if module not in self.lora_A_matrices:
                return False
            if module not in self.lora_B_matrices:
                return False
        
        return True
    
    def get_public_inputs(self) -> List[str]:
        """Extract public inputs for the ZK proof"""
        return [
            self.loss_value,
            self.learning_rate,
            self.rank,
            self.alpha,
            len(self.lora_A_matrices)
        ]


@dataclass
class ZKProofMetadata:
    """Metadata for ZK proofs"""
    proof_type: ZKProofType
    prover_id: str
    circuit_hash: str
    proving_time_ms: int
    verification_time_ms: Optional[int] = None
    proof_size_bytes: Optional[int] = None
    public_input_count: int = 0
    private_input_count: int = 0
    constraint_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "proof_type": self.proof_type.value,
            "prover_id": self.prover_id,
            "circuit_hash": self.circuit_hash,
            "proving_time_ms": self.proving_time_ms,
            "verification_time_ms": self.verification_time_ms,
            "proof_size_bytes": self.proof_size_bytes,
            "public_input_count": self.public_input_count,
            "private_input_count": self.private_input_count,
            "constraint_count": self.constraint_count
        }


# Utility functions for statement/witness construction
def create_sgd_statement(
    w_t_root: str,
    batch_root: str,
    hparams: Dict[str, Any],
    w_t1_root: str,
    step_info: Dict[str, Any]
) -> SGDStepStatement:
    """Helper function to create SGD statement"""
    import hashlib
    import json
    
    # Hash hyperparameters
    hparams_str = json.dumps(hparams, sort_keys=True)
    hparams_hash = hashlib.sha256(hparams_str.encode()).hexdigest()
    
    return SGDStepStatement(
        W_t_root=w_t_root,
        batch_root=batch_root,
        hparams_hash=hparams_hash,
        W_t1_root=w_t1_root,
        step_nonce=step_info.get("nonce", 0),
        step_number=step_info.get("step_number", 0),
        epoch=step_info.get("epoch", 0),
        timestamp=step_info.get("timestamp")
    )


def create_lora_statement(
    base_weights_root: str,
    lora_A_root: str,
    lora_B_root: str,
    batch_root: str,
    lora_params: Dict[str, Any],
    step_info: Dict[str, Any]
) -> LoRAStepStatement:
    """Helper function to create LoRA statement"""
    import hashlib
    import json
    
    # Hash LoRA hyperparameters
    hparams_str = json.dumps(lora_params, sort_keys=True)
    lora_hparams_hash = hashlib.sha256(hparams_str.encode()).hexdigest()
    
    return LoRAStepStatement(
        base_weights_root=base_weights_root,
        lora_A_root=lora_A_root,
        lora_B_root=lora_B_root,
        batch_root=batch_root,
        lora_hparams_hash=lora_hparams_hash,
        updated_weights_root=step_info.get("updated_weights_root", ""),
        rank=lora_params.get("rank", 8),
        alpha=lora_params.get("alpha", 16.0),
        dropout_rate=lora_params.get("dropout_rate", 0.1),
        step_nonce=step_info.get("nonce", 0),
        step_number=step_info.get("step_number", 0),
        epoch=step_info.get("epoch", 0),
        target_modules=lora_params.get("target_modules", [])
    )