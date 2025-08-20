"""
Witness Extraction for ZK Proofs

Functions to extract weights and batches for ZK proof generation,
building on existing TrainingProvenanceAuditor infrastructure.
"""

import numpy as np
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import asdict

from .spec import (
    SGDStepStatement, SGDStepWitness,
    LoRAStepStatement, LoRAStepWitness,
    ZKProofType, CommitmentScheme
)
from .commitments import DualCommitment, MerkleCommitment, PoseidonHasher

# Import from interfaces to avoid circular dependency
from ..core.interfaces import (
    EventType,
    TrainingEvent,
    IProvenanceAuditor,
    create_merkle_tree
)


def extract_sgd_witness(
    model_weights_before: Dict[str, np.ndarray],
    model_weights_after: Dict[str, np.ndarray],
    batch_inputs: np.ndarray,
    batch_targets: np.ndarray,
    hyperparameters: Dict[str, Any],
    gradients: Dict[str, np.ndarray],
    loss_value: float,
    commitment_scheme: CommitmentScheme = CommitmentScheme.DUAL_COMMITMENT
) -> SGDStepWitness:
    """
    Extract witness data for SGD training step ZK proof
    
    Args:
        model_weights_before: Weights before SGD step
        model_weights_after: Weights after SGD step
        batch_inputs: Training batch input data
        batch_targets: Training batch target data
        hyperparameters: SGD hyperparameters (lr, momentum, etc.)
        gradients: Computed gradients for each weight
        loss_value: Loss value on the batch
        commitment_scheme: Commitment scheme to use
        
    Returns:
        SGDStepWitness containing all private data for proof
    """
    # Create commitment scheme
    if commitment_scheme == CommitmentScheme.DUAL_COMMITMENT:
        committer = DualCommitment()
    elif commitment_scheme == CommitmentScheme.POSEIDON_MERKLE:
        committer = MerkleCommitment("poseidon")
    else:
        committer = MerkleCommitment("sha256")
    
    # Generate weight openings (Merkle proofs)
    weight_openings = {}
    weight_randomness = {}
    
    for layer_name, weight_tensor in model_weights_before.items():
        # Generate Merkle proof for this weight tensor
        openings = committer.generate_proof(weight_tensor, 0) if hasattr(committer, 'generate_proof') else []
        weight_openings[layer_name] = openings
        
        # Generate randomness for commitment (if needed)
        weight_randomness[layer_name] = hashlib.sha256(
            f"weight_randomness_{layer_name}".encode()
        ).digest()
    
    # Generate batch commitment openings
    if hasattr(committer, 'commit_batch'):
        batch_result = committer.commit_batch(batch_inputs, batch_targets)
        if isinstance(batch_result, dict):
            # DualCommitment returns dict
            batch_data = batch_result
        else:
            # MerkleCommitment returns tuple
            _, batch_data = batch_result
        batch_openings = ["batch_proof_element"] # Mock opening
    else:
        batch_openings = []
    
    batch_randomness = hashlib.sha256(b"batch_randomness").digest()
    
    return SGDStepWitness(
        weight_values=model_weights_before,
        weight_openings=weight_openings,
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        batch_openings=batch_openings,
        learning_rate=hyperparameters.get("learning_rate", 0.01),
        momentum=hyperparameters.get("momentum"),
        weight_decay=hyperparameters.get("weight_decay"),
        gradients=gradients,
        loss_value=loss_value,
        updated_weights=model_weights_after,
        weight_randomness=weight_randomness,
        batch_randomness=batch_randomness
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
    # Create commitment scheme
    if commitment_scheme == CommitmentScheme.DUAL_COMMITMENT:
        committer = DualCommitment()
    elif commitment_scheme == CommitmentScheme.POSEIDON_MERKLE:
        committer = MerkleCommitment("poseidon")
    else:
        committer = MerkleCommitment("sha256")
    
    # Generate openings for all weight matrices
    base_weight_openings = {}
    lora_A_openings = {}
    lora_B_openings = {}
    
    # Generate randomness
    base_randomness = {}
    lora_randomness = {}
    
    for layer_name, weight_tensor in base_weights.items():
        if hasattr(committer, 'generate_proof'):
            base_weight_openings[layer_name] = committer.generate_proof(weight_tensor, 0)
        else:
            base_weight_openings[layer_name] = []
        
        base_randomness[layer_name] = hashlib.sha256(
            f"base_randomness_{layer_name}".encode()
        ).digest()
    
    for layer_name, A_tensor in lora_A_before.items():
        if hasattr(committer, 'generate_proof'):
            lora_A_openings[layer_name] = committer.generate_proof(A_tensor, 0)
        else:
            lora_A_openings[layer_name] = []
    
    for layer_name, B_tensor in lora_B_before.items():
        if hasattr(committer, 'generate_proof'):
            lora_B_openings[layer_name] = committer.generate_proof(B_tensor, 0)
        else:
            lora_B_openings[layer_name] = []
        
        lora_randomness[layer_name] = hashlib.sha256(
            f"lora_randomness_{layer_name}".encode()
        ).digest()
    
    # Generate batch openings
    if hasattr(committer, 'commit_batch'):
        batch_result = committer.commit_batch(batch_inputs, batch_targets)
        if isinstance(batch_result, dict):
            # DualCommitment returns dict
            batch_data = batch_result
        else:
            # MerkleCommitment returns tuple
            _, batch_data = batch_result
        batch_openings = ["lora_batch_proof"]
    else:
        batch_openings = []
    
    batch_randomness = hashlib.sha256(b"lora_batch_randomness").digest()
    
    return LoRAStepWitness(
        base_weights=base_weights,
        base_weight_openings=base_weight_openings,
        lora_A_matrices=lora_A_before,
        lora_B_matrices=lora_B_before,
        lora_A_openings=lora_A_openings,
        lora_B_openings=lora_B_openings,
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        batch_openings=batch_openings,
        rank=lora_hyperparameters.get("rank", 8),
        alpha=lora_hyperparameters.get("alpha", 16.0),
        learning_rate=lora_hyperparameters.get("learning_rate", 0.001),
        dropout_rate=lora_hyperparameters.get("dropout_rate", 0.1),
        gradients_A=gradients_A,
        gradients_B=gradients_B,
        loss_value=loss_value,
        updated_A_matrices=lora_A_after,
        updated_B_matrices=lora_B_after,
        base_randomness=base_randomness,
        lora_randomness=lora_randomness,
        batch_randomness=batch_randomness
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
    commitment_scheme: CommitmentScheme = CommitmentScheme.POSEIDON_MERKLE
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
            commitment_scheme=commitment_scheme,
            timestamp=step_info.get("timestamp")
        )
    
    elif proof_type == ZKProofType.LORA_STEP:
        # For LoRA, we need separate commitments for base weights and LoRA matrices
        # This is a simplified version - real implementation would extract LoRA matrices
        target_modules = hyperparameters.get("target_modules", ["attention"])
        
        return LoRAStepStatement(
            base_weights_root=w_t_root,  # Base weights (frozen)
            lora_A_root=w_t_root[:32] + "A" * 32,  # Mock LoRA A commitment
            lora_B_root=w_t_root[:32] + "B" * 32,  # Mock LoRA B commitment
            batch_root=batch_root,
            lora_hparams_hash=hparams_hash,
            updated_weights_root=w_t1_root,
            rank=hyperparameters.get("rank", 8),
            alpha=hyperparameters.get("alpha", 16.0),
            dropout_rate=hyperparameters.get("dropout_rate", 0.1),
            step_nonce=step_info.get("nonce", 0),
            step_number=step_info.get("step_number", 0),
            epoch=step_info.get("epoch", 0),
            target_modules=target_modules,
            commitment_scheme=commitment_scheme,
            timestamp=step_info.get("timestamp")
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