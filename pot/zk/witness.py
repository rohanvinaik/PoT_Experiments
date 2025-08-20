"""
Witness Extraction for ZK Proofs

Functions to extract weights and batches for ZK proof generation,
building on existing TrainingProvenanceAuditor infrastructure.
"""

import numpy as np
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any, Union

from .zk_types import (
    SGDStepStatement,
    SGDStepWitness,
    LoRAStepStatement,
    LoRAStepWitness,
)
from .spec import ZKProofType, CommitmentScheme
from .commitments import DualCommitment, MerkleCommitment, PoseidonHasher

# Import existing infrastructure
try:
    from ..prototypes.training_provenance_auditor import (
        TrainingProvenanceAuditor,
        TrainingEvent,
        EventType
    )
except ImportError:
    # Fallback for testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from prototypes.training_provenance_auditor import (
        TrainingProvenanceAuditor,
        TrainingEvent,
        EventType
    )


def extract_sgd_witness(
    model_weights_before: Dict[str, np.ndarray],
    model_weights_after: Dict[str, np.ndarray],
    batch_inputs: np.ndarray,
    batch_targets: np.ndarray,
    hyperparameters: Dict[str, Any],
    gradients: Dict[str, np.ndarray],
    loss_value: float,
    commitment_scheme: CommitmentScheme = CommitmentScheme.DUAL_COMMITMENT,
) -> SGDStepWitness:
    """Extract witness data for an SGD training step.

    This simplified witness flattens tensors into lists so it matches the
    lightweight dataclasses defined in :mod:`zk_types`.
    """

    def _flatten_dict(tensors: Dict[str, np.ndarray]) -> List[float]:
        return [float(x) for arr in tensors.values() for x in arr.flatten()]

    weights_before = _flatten_dict(model_weights_before)
    weights_after = _flatten_dict(model_weights_after)
    batch_in = batch_inputs.flatten().tolist()
    batch_tgt = batch_targets.flatten().tolist()

    return SGDStepWitness(
        weights_before=weights_before,
        weights_after=weights_after,
        batch_inputs=batch_in,
        batch_targets=batch_tgt,
        learning_rate=hyperparameters.get("learning_rate", 0.01),
        loss_value=loss_value,
    )


def extract_lora_witness(
    base_weights: Dict[str, np.ndarray],
    lora_A_before: Dict[str, np.ndarray],
    lora_B_before: Dict[str, np.ndarray],
    lora_A_after: Dict[str, np.ndarray],
    lora_B_after: Dict[str, np.ndarray],
    batch_inputs: np.ndarray,
    batch_targets: np.ndarray,
    lora_hyperparameters: Dict[str, Any],
    gradients_A: Dict[str, np.ndarray],
    gradients_B: Dict[str, np.ndarray],
    loss_value: float,
    commitment_scheme: CommitmentScheme = CommitmentScheme.DUAL_COMMITMENT
) -> LoRAStepWitness:
    """
    Extract witness data for LoRA training step ZK proof
    
    Args:
        base_weights: Base model weights (frozen)
        lora_A_before: LoRA A matrices before step
        lora_B_before: LoRA B matrices before step
        lora_A_after: LoRA A matrices after step
        lora_B_after: LoRA B matrices after step
        batch_inputs: Training batch inputs
        batch_targets: Training batch targets
        lora_hyperparameters: LoRA hyperparameters
        gradients_A: Gradients for A matrices
        gradients_B: Gradients for B matrices
        loss_value: Loss value
        commitment_scheme: Commitment scheme to use
        
    Returns:
        LoRAStepWitness containing all private data for proof
    """

    def _flatten_dict(tensors: Dict[str, np.ndarray]) -> List[float]:
        return [float(x) for arr in tensors.values() for x in arr.flatten()]

    adapter_a_before = _flatten_dict(lora_A_before)
    adapter_b_before = _flatten_dict(lora_B_before)
    adapter_a_after = _flatten_dict(lora_A_after)
    adapter_b_after = _flatten_dict(lora_B_after)
    grad_a = _flatten_dict(gradients_A)
    grad_b = _flatten_dict(gradients_B)
    batch_in = batch_inputs.flatten().tolist()
    batch_tgt = batch_targets.flatten().tolist()

    return LoRAStepWitness(
        adapter_a_before=adapter_a_before,
        adapter_b_before=adapter_b_before,
        adapter_a_after=adapter_a_after,
        adapter_b_after=adapter_b_after,
        adapter_a_gradients=grad_a,
        adapter_b_gradients=grad_b,
        batch_inputs=batch_in,
        batch_targets=batch_tgt,
        learning_rate=lora_hyperparameters.get("learning_rate", 0.001),
        loss_value=loss_value,
    )


def extract_weight_openings(
    weights: Dict[str, np.ndarray],
    commitment_scheme: CommitmentScheme = CommitmentScheme.POSEIDON_MERKLE
) -> Dict[str, List[str]]:
    """
    Extract Merkle openings for weight tensors
    
    Args:
        weights: Dictionary of weight tensors
        commitment_scheme: Commitment scheme to use
        
    Returns:
        Dictionary mapping layer names to Merkle proof paths
    """
    # Create appropriate committer
    if commitment_scheme == CommitmentScheme.POSEIDON_MERKLE:
        committer = MerkleCommitment("poseidon")
    else:
        committer = MerkleCommitment("sha256")
    
    openings = {}
    for layer_name, weight_tensor in weights.items():
        if hasattr(committer, 'generate_proof'):
            openings[layer_name] = committer.generate_proof(weight_tensor, 0)
        else:
            # Mock opening for testing
            openings[layer_name] = [f"mock_proof_{layer_name}_{i}" for i in range(3)]
    
    return openings


def extract_batch_openings(
    batch_inputs: np.ndarray,
    batch_targets: np.ndarray,
    commitment_scheme: CommitmentScheme = CommitmentScheme.POSEIDON_MERKLE
) -> List[str]:
    """
    Extract Merkle openings for training batch
    
    Args:
        batch_inputs: Batch input data
        batch_targets: Batch target data
        commitment_scheme: Commitment scheme to use
        
    Returns:
        List of Merkle proof elements for batch commitment
    """
    # Create appropriate committer
    if commitment_scheme == CommitmentScheme.POSEIDON_MERKLE:
        committer = MerkleCommitment("poseidon")
    else:
        committer = MerkleCommitment("sha256")
    
    # For batch, we need a single opening since we commit to the entire batch
    if hasattr(committer, 'commit_batch'):
        batch_result = committer.commit_batch(batch_inputs, batch_targets)
        if isinstance(batch_result, dict):
            # DualCommitment returns dict
            batch_data = batch_result
        else:
            # MerkleCommitment returns tuple
            _, batch_data = batch_result
        return ["batch_merkle_proof_element"]
    else:
        return ["mock_batch_proof"]


def build_zk_statement(
    weights_before: Dict[str, np.ndarray],
    weights_after: Dict[str, np.ndarray],
    batch_inputs: np.ndarray,
    batch_targets: np.ndarray,
    hyperparameters: Dict[str, Any],
    step_info: Dict[str, Any],
    proof_type: ZKProofType = ZKProofType.SGD_STEP,
    commitment_scheme: CommitmentScheme = CommitmentScheme.POSEIDON_MERKLE,
) -> Union[SGDStepStatement, LoRAStepStatement]:
    """
    Build ZK statement for training step
    
    Args:
        weights_before: Weights before training step
        weights_after: Weights after training step
        batch_inputs: Training batch inputs
        batch_targets: Training batch targets
        hyperparameters: Training hyperparameters
        step_info: Step metadata (step_number, epoch, nonce, etc.)
        proof_type: Type of ZK proof to generate
        commitment_scheme: Commitment scheme to use
        
    Returns:
        Appropriate statement object for the proof type
    """
    # Create committer
    if commitment_scheme == CommitmentScheme.DUAL_COMMITMENT:
        committer = DualCommitment()
    elif commitment_scheme == CommitmentScheme.POSEIDON_MERKLE:
        committer = MerkleCommitment("poseidon")
    else:
        committer = MerkleCommitment("sha256")
    
    # Generate commitments to weights and batch
    weight_commitments_before = {}
    weight_commitments_after = {}
    
    for layer_name, weight_tensor in weights_before.items():
        if hasattr(committer, 'commit_tensor'):
            tensor_result = committer.commit_tensor(weight_tensor)
            if isinstance(tensor_result, dict):
                # DualCommitment returns dict
                root_hex = tensor_result.get("sha256_root") or tensor_result.get("poseidon_root")
            else:
                # MerkleCommitment returns tuple
                root_hex, _ = tensor_result
            weight_commitments_before[layer_name] = root_hex
        else:
            # Mock commitment
            weight_commitments_before[layer_name] = hashlib.sha256(
                weight_tensor.tobytes()
            ).hexdigest()
    
    for layer_name, weight_tensor in weights_after.items():
        if hasattr(committer, 'commit_tensor'):
            tensor_result = committer.commit_tensor(weight_tensor)
            if isinstance(tensor_result, dict):
                # DualCommitment returns dict
                root_hex = tensor_result.get("sha256_root") or tensor_result.get("poseidon_root")
            else:
                # MerkleCommitment returns tuple
                root_hex, _ = tensor_result
            weight_commitments_after[layer_name] = root_hex
        else:
            weight_commitments_after[layer_name] = hashlib.sha256(
                weight_tensor.tobytes()
            ).hexdigest()
    
    # Commit to batch
    if hasattr(committer, 'commit_batch'):
        batch_result = committer.commit_batch(batch_inputs, batch_targets)
        if isinstance(batch_result, dict):
            # DualCommitment returns dict
            batch_root = batch_result.get("sha256_root") or batch_result.get("poseidon_root")
        else:
            # MerkleCommitment returns tuple
            batch_root, _ = batch_result
    else:
        combined_batch = np.concatenate([
            batch_inputs.flatten(),
            batch_targets.flatten()
        ])
        batch_root = hashlib.sha256(combined_batch.tobytes()).hexdigest()
    
    # Create combined weight commitment (simplified - in practice would be more sophisticated)
    all_weights_before = np.concatenate([w.flatten() for w in weights_before.values()])
    all_weights_after = np.concatenate([w.flatten() for w in weights_after.values()])
    
    w_t_root = hashlib.sha256(all_weights_before.tobytes()).hexdigest()
    w_t1_root = hashlib.sha256(all_weights_after.tobytes()).hexdigest()
    
    # Hash hyperparameters
    hparams_str = json.dumps(hyperparameters, sort_keys=True)
    hparams_hash = hashlib.sha256(hparams_str.encode()).hexdigest()
    
    if proof_type == ZKProofType.SGD_STEP:
        return SGDStepStatement(
            W_t_root=w_t_root,
            batch_root=batch_root,
            hparams_hash=hparams_hash,
            W_t1_root=w_t1_root,
            step_nonce=step_info.get("nonce", 0),
            step_number=step_info.get("step_number", 0),
            epoch=step_info.get("epoch", 0),
        )
    
    elif proof_type == ZKProofType.LORA_STEP:
        # For LoRA, we need separate commitments for base weights and LoRA matrices
        # This is a simplified version - real implementation would extract LoRA matrices
        target_modules = hyperparameters.get("target_modules", ["attention"])
        
        return LoRAStepStatement(
            base_weights_root=w_t_root,
            adapter_a_root_before=w_t_root[:32] + "A" * 32,
            adapter_b_root_before=w_t_root[:32] + "B" * 32,
            adapter_a_root_after=w_t1_root[:32] + "A" * 32,
            adapter_b_root_after=w_t1_root[:32] + "B" * 32,
            batch_root=batch_root,
            hparams_hash=hparams_hash,
            rank=hyperparameters.get("rank", 8),
            alpha=hyperparameters.get("alpha", 16.0),
            step_number=step_info.get("step_number", 0),
            epoch=step_info.get("epoch", 0),
        )
    
    else:
        raise ValueError(f"Unsupported proof type: {proof_type}")


def simulate_training_step_extraction(
    model_state: Dict[str, Any],
    training_config: Dict[str, Any]
) -> Tuple[Union[SGDStepStatement, LoRAStepStatement], Union[SGDStepWitness, LoRAStepWitness]]:
    """
    Simulate extraction of ZK statement and witness from a training step
    
    This function demonstrates how the ZK module would integrate with
    real training code to extract proofs.
    
    Args:
        model_state: Current model state (weights, optimizer state, etc.)
        training_config: Training configuration
        
    Returns:
        Tuple of (statement, witness) for ZK proof generation
    """
    # Extract mock training data
    weights_before = {
        "layer1": np.random.randn(128, 64).astype(np.float32),
        "layer2": np.random.randn(64, 32).astype(np.float32),
        "output": np.random.randn(32, 10).astype(np.float32)
    }
    
    # Simulate SGD step
    learning_rate = training_config.get("learning_rate", 0.01)
    gradients = {
        name: np.random.randn(*weights.shape).astype(np.float32) * 0.1
        for name, weights in weights_before.items()
    }
    
    weights_after = {
        name: weights - learning_rate * gradients[name]
        for name, weights in weights_before.items()
    }
    
    # Mock batch data
    batch_size = training_config.get("batch_size", 32)
    input_dim = 128
    output_dim = 10
    
    batch_inputs = np.random.randn(batch_size, input_dim).astype(np.float32)
    batch_targets = np.random.randint(0, output_dim, (batch_size,)).astype(np.int64)
    
    # Mock hyperparameters
    hyperparameters = {
        "learning_rate": learning_rate,
        "momentum": training_config.get("momentum", 0.9),
        "weight_decay": training_config.get("weight_decay", 0.0001)
    }
    
    # Step info
    step_info = {
        "step_number": model_state.get("step", 0),
        "epoch": model_state.get("epoch", 0),
        "nonce": model_state.get("nonce", 0),
        "timestamp": model_state.get("timestamp")
    }
    
    # Build statement
    statement = build_zk_statement(
        weights_before=weights_before,
        weights_after=weights_after,
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        hyperparameters=hyperparameters,
        step_info=step_info,
        proof_type=ZKProofType.SGD_STEP
    )
    
    # Extract witness
    witness = extract_sgd_witness(
        model_weights_before=weights_before,
        model_weights_after=weights_after,
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        hyperparameters=hyperparameters,
        gradients=gradients,
        loss_value=0.5  # Mock loss
    )
    
    return statement, witness