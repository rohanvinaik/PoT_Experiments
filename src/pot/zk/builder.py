"""
Builder utilities for constructing ZK proof witnesses from model data.
"""

import hashlib
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Import from interfaces to avoid circular dependency
from ..core.interfaces import (
    MerkleNode,
    IMerkleTree,
    BasicMerkleTree,
    create_merkle_tree
)


@dataclass
class WeightExtraction:
    """Extracted weight values with Merkle paths."""
    values: np.ndarray
    merkle_paths: List[List[bytes]]
    root: bytes
    indices: List[int]


@dataclass
class BatchCommitment:
    """Batch data commitment with openings."""
    batch_root: bytes
    input_values: np.ndarray
    target_values: np.ndarray
    merkle_paths: List[List[bytes]]


class ZKWitnessBuilder:
    """Builder for constructing ZK proof witnesses from training data."""
    
    def __init__(self, hash_function=None):
        """
        Initialize the witness builder.
        
        Args:
            hash_function: Hash function to use (defaults to SHA256)
        """
        self.hash_function = hash_function or self._default_hash
    
    def extract_weights_for_zk(self, 
                               model_state: Dict[str, np.ndarray], 
                               indices: List[Tuple[str, int]]) -> WeightExtraction:
        """
        Extract weight values and Merkle paths from model state.
        
        Args:
            model_state: Dictionary of model parameters (layer_name -> weights)
            indices: List of (layer_name, flat_index) tuples specifying which weights to extract
            
        Returns:
            WeightExtraction with values and Merkle authentication paths
        """
        # Flatten all model weights into a single array
        flat_weights = []
        layer_offsets = {}
        current_offset = 0
        
        for layer_name in sorted(model_state.keys()):
            weights = model_state[layer_name]
            flat = weights.flatten()
            layer_offsets[layer_name] = (current_offset, current_offset + len(flat))
            flat_weights.extend(flat)
            current_offset += len(flat)
        
        # Build Merkle tree from all weights
        weight_leaves = [self._weight_to_bytes(w) for w in flat_weights]
        merkle_tree = self._build_simple_merkle_tree(weight_leaves)
        
        # Extract requested weights and their paths
        values = []
        merkle_paths = []
        global_indices = []
        
        for layer_name, local_idx in indices:
            if layer_name not in layer_offsets:
                raise ValueError(f"Layer {layer_name} not found in model state")
            
            start_offset, end_offset = layer_offsets[layer_name]
            global_idx = start_offset + local_idx
            
            if global_idx >= len(flat_weights):
                raise ValueError(f"Index {local_idx} out of bounds for layer {layer_name}")
            
            values.append(flat_weights[global_idx])
            
            # Get Merkle proof for this weight
            proof = merkle_tree.generate_proof(global_idx)
            merkle_paths.append(proof.path)
            global_indices.append(global_idx)
        
        return WeightExtraction(
            values=np.array(values),
            merkle_paths=merkle_paths,
            root=merkle_tree.root.hash,
            indices=global_indices
        )
    
    def build_batch_commitment(self, 
                              batch_data: Dict[str, np.ndarray]) -> BatchCommitment:
        """
        Build a Merkle commitment for batch data.
        
        Args:
            batch_data: Dictionary with 'inputs' and 'targets' arrays
            
        Returns:
            BatchCommitment with root and openings
        """
        inputs = batch_data.get('inputs', np.array([]))
        targets = batch_data.get('targets', np.array([]))
        
        # Flatten batch data
        flat_inputs = inputs.flatten()
        flat_targets = targets.flatten()
        
        # Create leaves from all batch elements
        batch_leaves = []
        
        # Add input values as leaves
        for val in flat_inputs:
            batch_leaves.append(self._value_to_bytes(val))
        
        # Add target values as leaves
        for val in flat_targets:
            batch_leaves.append(self._value_to_bytes(val))
        
        # Build Merkle tree
        if len(batch_leaves) == 0:
            # Empty batch
            return BatchCommitment(
                batch_root=b'\x00' * 32,
                input_values=np.array([]),
                target_values=np.array([]),
                merkle_paths=[]
            )
        
        merkle_tree = self._build_simple_merkle_tree(batch_leaves)
        
        # Generate paths for a subset of values (for efficiency)
        # In practice, we'd only open specific values needed for the proof
        num_openings = min(4, len(batch_leaves))  # Open first 4 values
        merkle_paths = []
        
        for i in range(num_openings):
            proof = merkle_tree.generate_proof(i)
            merkle_paths.append(proof.path)
        
        return BatchCommitment(
            batch_root=merkle_tree.root.hash,
            input_values=flat_inputs,
            target_values=flat_targets,
            merkle_paths=merkle_paths
        )
    
    def extract_sgd_update_witness(self,
                                   weights_before: Dict[str, np.ndarray],
                                   weights_after: Dict[str, np.ndarray],
                                   batch_data: Dict[str, np.ndarray],
                                   learning_rate: float,
                                   layer_indices: Optional[List[Tuple[str, int]]] = None) -> Dict[str, Any]:
        """
        Extract a complete witness for SGD update verification.
        
        Args:
            weights_before: Model weights before update
            weights_after: Model weights after update
            batch_data: Training batch with inputs and targets
            learning_rate: Learning rate used for update
            layer_indices: Optional specific indices to extract (defaults to first 64)
            
        Returns:
            Dictionary with witness data suitable for ZK proof generation
        """
        # Default to extracting first 64 weights (16x4 matrix)
        if layer_indices is None:
            # Get first layer name and extract first 64 weights
            first_layer = sorted(weights_before.keys())[0]
            layer_indices = [(first_layer, i) for i in range(min(64, 
                            weights_before[first_layer].size))]
        
        # Extract weights before update
        weights_before_extracted = self.extract_weights_for_zk(
            weights_before, layer_indices
        )
        
        # Extract weights after update
        weights_after_extracted = self.extract_weights_for_zk(
            weights_after, layer_indices
        )
        
        # Build batch commitment
        batch_commitment = self.build_batch_commitment(batch_data)
        
        # Compute gradients
        gradients = (weights_before_extracted.values - weights_after_extracted.values) / learning_rate
        
        # Compute simple loss value (MSE)
        if len(batch_commitment.input_values) > 0 and len(batch_commitment.target_values) > 0:
            # Simplified: just use first sample
            pred = batch_commitment.input_values[:16] @ weights_before_extracted.values[:64].reshape(16, 4)
            error = pred - batch_commitment.target_values[:4]
            loss_value = float(np.mean(error ** 2))
        else:
            loss_value = 0.0
        
        return {
            "weights_before": weights_before_extracted.values.tolist(),
            "weights_after": weights_after_extracted.values.tolist(),
            "weights_before_paths": weights_before_extracted.merkle_paths,
            "weights_after_paths": weights_after_extracted.merkle_paths,
            "w_t_root": weights_before_extracted.root,
            "w_t1_root": weights_after_extracted.root,
            "batch_inputs": batch_commitment.input_values.tolist(),
            "batch_targets": batch_commitment.target_values.tolist(),
            "batch_root": batch_commitment.batch_root,
            "batch_paths": batch_commitment.merkle_paths,
            "gradients": gradients.tolist(),
            "learning_rate": learning_rate,
            "loss_value": loss_value
        }
    
    def _default_hash(self, data: bytes) -> bytes:
        """Default hash function (SHA256)."""
        return hashlib.sha256(data).digest()
    
    def _weight_to_bytes(self, weight: float) -> bytes:
        """Convert a weight value to bytes."""
        # Convert to fixed-point representation (scale by 2^16)
        fixed_point = int(weight * 65536)
        # Convert to 8-byte representation
        return fixed_point.to_bytes(8, byteorder='little', signed=True)
    
    def _value_to_bytes(self, value: float) -> bytes:
        """Convert a general value to bytes."""
        # Similar to weight conversion
        fixed_point = int(value * 65536)
        return fixed_point.to_bytes(8, byteorder='little', signed=True)
    
    def _build_simple_merkle_tree(self, leaves: List[bytes]) -> 'SimpleMerkleTree':
        """Build a simple Merkle tree from leaf data."""
        return SimpleMerkleTree(leaves, self.hash_function)


class SimpleMerkleTree:
    """Simple Merkle tree implementation for witness building."""
    
    def __init__(self, leaves: List[bytes], hash_function=None):
        self.leaves = leaves
        self.hash_function = hash_function or (lambda x: hashlib.sha256(x).digest())
        self.tree = self._build_tree()
        self.root_hash = self.tree[0] if self.tree else b'\x00' * 32
        self.root = type('MerkleNode', (), {'hash': self.root_hash})()
    
    def compute_root(self) -> bytes:
        """Compute and return the Merkle root."""
        return self.root_hash
    
    def _build_tree(self) -> List[bytes]:
        """Build complete binary tree."""
        if not self.leaves:
            return []
        
        # Start with leaf hashes
        current_level = [self.hash_function(leaf) for leaf in self.leaves]
        tree = list(current_level)
        
        # Build tree bottom-up
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    combined = current_level[i] + current_level[i + 1]
                else:
                    combined = current_level[i] + current_level[i]  # Duplicate last node
                next_level.append(self.hash_function(combined))
            
            tree = next_level + tree  # Prepend to maintain level order
            current_level = next_level
        
        return tree
    
    def generate_proof(self, index: int) -> 'MerkleProof':
        """Generate Merkle proof for leaf at index."""
        if index >= len(self.leaves):
            return MerkleProof([], index, self.root_hash)
        
        path = []
        current_index = index
        level_start = len(self.tree) - len(self.leaves)
        level_size = len(self.leaves)
        
        while level_size > 1:
            # Find sibling
            if current_index % 2 == 0:
                sibling_index = current_index + 1
            else:
                sibling_index = current_index - 1
            
            if sibling_index < level_size:
                sibling_hash = self.tree[level_start + sibling_index]
            else:
                sibling_hash = self.tree[level_start + current_index]  # Use self as sibling
            
            path.append(sibling_hash)
            
            # Move to next level
            current_index //= 2
            level_size = (level_size + 1) // 2
            level_start -= level_size
        
        return MerkleProof(path, index, self.root_hash)


@dataclass
class MerkleProof:
    """Simple Merkle proof."""
    path: List[bytes]
    index: int
    root: bytes


# Convenience functions
def extract_weights_for_zk(model_state: Dict[str, np.ndarray], 
                          indices: List[Tuple[str, int]]) -> WeightExtraction:
    """Extract weight values and Merkle paths from model state."""
    builder = ZKWitnessBuilder()
    return builder.extract_weights_for_zk(model_state, indices)


def build_batch_commitment(batch_data: Dict[str, np.ndarray]) -> BatchCommitment:
    """Build a Merkle commitment for batch data."""
    builder = ZKWitnessBuilder()
    return builder.build_batch_commitment(batch_data)