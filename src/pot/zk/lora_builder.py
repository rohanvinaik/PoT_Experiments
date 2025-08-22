"""
Builder utilities for LoRA-specific ZK proof witnesses.

This module handles extraction and preparation of LoRA adapter weights
for zero-knowledge proof generation, which is much more efficient than
proving updates to full model weights.
"""

import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

from pot.zk.zk_types import (
    LoRAStepStatement, 
    LoRAStepWitness, 
    LoRAConfig,
    LoRAProofMetadata
)
from pot.zk.builder import SimpleMerkleTree


@dataclass
class LoRAAdapterWeights:
    """Container for LoRA adapter matrices."""
    adapter_a: np.ndarray  # Shape: (d_in, r)
    adapter_b: np.ndarray  # Shape: (r, d_out)
    layer_name: str
    rank: int
    alpha: float
    
    @property
    def d_in(self) -> int:
        return self.adapter_a.shape[0]
    
    @property
    def d_out(self) -> int:
        return self.adapter_b.shape[1]
    
    @property
    def num_params(self) -> int:
        """Total number of parameters in this adapter."""
        return self.adapter_a.size + self.adapter_b.size
    
    def get_effective_weight(self, base_weight: np.ndarray) -> np.ndarray:
        """Compute W_effective = W_base + α/r * (B @ A)."""
        scaling = self.alpha / self.rank
        delta = scaling * (self.adapter_b.T @ self.adapter_a.T)
        return base_weight + delta


class LoRAWitnessBuilder:
    """Builder for constructing LoRA-specific ZK proof witnesses."""
    
    def __init__(self, lora_config: Optional[LoRAConfig] = None):
        """
        Initialize the LoRA witness builder.
        
        Args:
            lora_config: LoRA configuration parameters
        """
        self.config = lora_config or LoRAConfig()
        self.hash_function = lambda x: hashlib.sha256(x).digest()
    
    def extract_lora_adapters(self, 
                             model_state: Dict[str, Any]) -> Dict[str, LoRAAdapterWeights]:
        """
        Extract LoRA adapter weights from model state.
        
        Args:
            model_state: Model state dictionary containing LoRA adapters
            
        Returns:
            Dictionary mapping layer names to LoRA adapter weights
        """
        adapters = {}
        
        for name, param in model_state.items():
            # Check if this is a LoRA adapter
            if 'lora_A' in name:
                base_name = name.replace('.lora_A', '')
                adapter_b_name = name.replace('.lora_A', '.lora_B')
                
                if adapter_b_name in model_state:
                    adapter_a = np.array(param)
                    adapter_b = np.array(model_state[adapter_b_name])
                    
                    # Ensure correct shapes
                    if len(adapter_a.shape) == 2 and len(adapter_b.shape) == 2:
                        adapters[base_name] = LoRAAdapterWeights(
                            adapter_a=adapter_a,
                            adapter_b=adapter_b,
                            layer_name=base_name,
                            rank=adapter_a.shape[1],
                            alpha=self.config.alpha
                        )
        
        # If no LoRA adapters found, check for alternative naming
        if not adapters:
            adapters = self._extract_alternative_format(model_state)
        
        return adapters
    
    def _extract_alternative_format(self, 
                                   model_state: Dict[str, Any]) -> Dict[str, LoRAAdapterWeights]:
        """Extract LoRA adapters with alternative naming conventions."""
        adapters = {}
        
        # Look for patterns like 'layer.weight_A' and 'layer.weight_B'
        for name, param in model_state.items():
            if name.endswith('_A'):
                base_name = name[:-2]
                b_name = base_name + '_B'
                
                if b_name in model_state:
                    adapter_a = np.array(param)
                    adapter_b = np.array(model_state[b_name])
                    
                    # For simplified examples, create synthetic LoRA adapters
                    if len(adapter_a.shape) == 2:
                        # Extract rank-r approximation
                        d_in, d_out = adapter_a.shape
                        rank = min(self.config.rank, min(d_in, d_out))
                        
                        # Use SVD to get low-rank approximation
                        U, S, Vt = np.linalg.svd(adapter_a, full_matrices=False)
                        
                        # Create LoRA adapters from SVD
                        lora_a = U[:, :rank] @ np.diag(np.sqrt(S[:rank]))
                        lora_b = np.diag(np.sqrt(S[:rank])) @ Vt[:rank, :]
                        
                        adapters[base_name] = LoRAAdapterWeights(
                            adapter_a=lora_a,
                            adapter_b=lora_b,
                            layer_name=base_name,
                            rank=rank,
                            alpha=self.config.alpha
                        )
        
        return adapters
    
    def build_lora_witness(self,
                          adapters_before: Dict[str, LoRAAdapterWeights],
                          adapters_after: Dict[str, LoRAAdapterWeights],
                          batch_data: Dict[str, np.ndarray],
                          learning_rate: float,
                          layer_name: Optional[str] = None) -> LoRAStepWitness:
        """
        Build witness for LoRA update verification.
        
        Args:
            adapters_before: LoRA adapters before update
            adapters_after: LoRA adapters after update
            batch_data: Training batch data
            learning_rate: Learning rate used
            layer_name: Specific layer to build witness for
            
        Returns:
            LoRA step witness for ZK proof
        """
        # Select the layer to prove
        if layer_name is None:
            layer_name = list(adapters_before.keys())[0]
        
        before = adapters_before[layer_name]
        after = adapters_after[layer_name]
        
        # Compute gradients
        grad_a = (before.adapter_a - after.adapter_a) / learning_rate
        grad_b = (before.adapter_b - after.adapter_b) / learning_rate
        
        # Flatten inputs
        batch_inputs = batch_data.get('inputs', batch_data.get('x', np.array([])))
        batch_targets = batch_data.get('targets', batch_data.get('y', np.array([])))
        
        # Compute loss (simplified)
        loss_value = np.mean((batch_inputs @ before.adapter_a @ before.adapter_b - batch_targets.flatten()) ** 2)
        
        return LoRAStepWitness(
            adapter_a_before=before.adapter_a.flatten().tolist(),
            adapter_b_before=before.adapter_b.flatten().tolist(),
            adapter_a_after=after.adapter_a.flatten().tolist(),
            adapter_b_after=after.adapter_b.flatten().tolist(),
            adapter_a_gradients=grad_a.flatten().tolist(),
            adapter_b_gradients=grad_b.flatten().tolist(),
            batch_inputs=batch_inputs.flatten().tolist(),
            batch_targets=batch_targets.flatten().tolist(),
            learning_rate=learning_rate,
            loss_value=float(loss_value)
        )
    
    def build_lora_statement(self,
                            adapters_before: Dict[str, LoRAAdapterWeights],
                            adapters_after: Dict[str, LoRAAdapterWeights],
                            batch_data: Dict[str, np.ndarray],
                            base_model_root: bytes,
                            step_number: int,
                            epoch: int,
                            layer_name: Optional[str] = None) -> LoRAStepStatement:
        """
        Build public statement for LoRA update.
        
        Args:
            adapters_before: LoRA adapters before update
            adapters_after: LoRA adapters after update
            batch_data: Training batch data
            base_model_root: Merkle root of frozen base model
            step_number: Training step number
            epoch: Training epoch
            layer_name: Specific layer to build statement for
            
        Returns:
            LoRA step statement for ZK proof
        """
        # Select the layer
        if layer_name is None:
            layer_name = list(adapters_before.keys())[0]
        
        before = adapters_before[layer_name]
        after = adapters_after[layer_name]
        
        # Build Merkle trees for adapters
        tree_a_before = SimpleMerkleTree(
            [before.adapter_a.tobytes()],
            self.hash_function
        )
        tree_b_before = SimpleMerkleTree(
            [before.adapter_b.tobytes()],
            self.hash_function
        )
        tree_a_after = SimpleMerkleTree(
            [after.adapter_a.tobytes()],
            self.hash_function
        )
        tree_b_after = SimpleMerkleTree(
            [after.adapter_b.tobytes()],
            self.hash_function
        )
        
        # Build batch tree
        batch_tree = SimpleMerkleTree(
            [batch_data['inputs'].tobytes(), batch_data['targets'].tobytes()],
            self.hash_function
        )
        
        # Create hyperparameters hash
        hparams = {
            'rank': before.rank,
            'alpha': before.alpha,
            'layer': layer_name,
            'epoch': epoch
        }
        hparams_hash = hashlib.sha256(str(hparams).encode()).digest()
        
        return LoRAStepStatement(
            base_weights_root=base_model_root,
            adapter_a_root_before=tree_a_before.compute_root(),
            adapter_b_root_before=tree_b_before.compute_root(),
            adapter_a_root_after=tree_a_after.compute_root(),
            adapter_b_root_after=tree_b_after.compute_root(),
            batch_root=batch_tree.compute_root(),
            hparams_hash=hparams_hash,
            rank=before.rank,
            alpha=before.alpha,
            step_number=step_number,
            epoch=epoch
        )
    
    def compute_proof_metadata(self,
                              adapters: Dict[str, LoRAAdapterWeights],
                              base_model_params: int,
                              proof_size: int,
                              proof_time_ms: int) -> LoRAProofMetadata:
        """
        Compute metadata about LoRA proof efficiency.
        
        Args:
            adapters: LoRA adapter weights
            base_model_params: Total parameters in base model
            proof_size: Size of generated proof in bytes
            proof_time_ms: Time to generate proof in milliseconds
            
        Returns:
            Metadata about proof efficiency
        """
        # Calculate total LoRA parameters
        lora_params = sum(a.num_params for a in adapters.values())
        
        # Compression ratio
        compression_ratio = base_model_params / lora_params if lora_params > 0 else 0
        
        # Estimate circuit constraints (simplified)
        # LoRA circuits are much smaller than full weight circuits
        circuit_constraints = lora_params * 10  # Each param needs ~10 constraints
        
        # Memory usage estimate (in MB)
        memory_usage_mb = (lora_params * 32) / (1024 * 1024)  # 32 bytes per field element
        
        return LoRAProofMetadata(
            full_model_params=base_model_params,
            lora_params=lora_params,
            compression_ratio=compression_ratio,
            proof_size_bytes=proof_size,
            proof_generation_ms=proof_time_ms,
            circuit_constraints=circuit_constraints,
            memory_usage_mb=memory_usage_mb
        )
    
    def detect_lora_training(self, model_state: Dict[str, Any]) -> bool:
        """
        Detect if model is using LoRA fine-tuning.
        
        Args:
            model_state: Model state dictionary
            
        Returns:
            True if LoRA adapters are detected
        """
        lora_patterns = ['lora_A', 'lora_B', '_A', '_B', 'adapter']
        return any(pattern in name for name in model_state.keys() 
                  for pattern in lora_patterns)
    
    def simulate_lora_from_full_weights(self,
                                       full_weights: np.ndarray,
                                       rank: Optional[int] = None) -> LoRAAdapterWeights:
        """
        Create LoRA adapters from full weight matrix using SVD.
        
        This is useful for converting existing full fine-tuning to LoRA format
        for more efficient proof generation.
        
        Args:
            full_weights: Full weight matrix
            rank: LoRA rank (default from config)
            
        Returns:
            LoRA adapter weights approximating the full weights
        """
        if rank is None:
            rank = self.config.rank
        
        # Ensure 2D
        if len(full_weights.shape) == 1:
            full_weights = full_weights.reshape(-1, 1)
        
        d_in, d_out = full_weights.shape
        actual_rank = min(rank, min(d_in, d_out))
        
        # Use SVD for low-rank approximation
        U, S, Vt = np.linalg.svd(full_weights, full_matrices=False)
        
        # Create LoRA adapters
        # A = U * sqrt(S), B = sqrt(S) * Vt
        adapter_a = U[:, :actual_rank] @ np.diag(np.sqrt(S[:actual_rank]))
        adapter_b = np.diag(np.sqrt(S[:actual_rank])) @ Vt[:actual_rank, :]
        
        return LoRAAdapterWeights(
            adapter_a=adapter_a,
            adapter_b=adapter_b,
            layer_name="simulated",
            rank=actual_rank,
            alpha=self.config.alpha
        )


def create_example_lora_adapters(d_in: int = 768, 
                                d_out: int = 768, 
                                rank: int = 16) -> LoRAAdapterWeights:
    """
    Create example LoRA adapters for testing.
    
    Args:
        d_in: Input dimension
        d_out: Output dimension
        rank: LoRA rank
        
    Returns:
        Example LoRA adapter weights
    """
    # Initialize with small random values
    adapter_a = np.random.randn(d_in, rank) * 0.01
    adapter_b = np.random.randn(rank, d_out) * 0.01
    
    return LoRAAdapterWeights(
        adapter_a=adapter_a,
        adapter_b=adapter_b,
        layer_name="example_layer",
        rank=rank,
        alpha=rank * 2.0  # Common choice: alpha = 2 * rank
    )


def compare_lora_vs_full_params(d_in: int, d_out: int, rank: int) -> Dict[str, Any]:
    """
    Compare parameter counts between LoRA and full fine-tuning.
    
    Args:
        d_in: Input dimension
        d_out: Output dimension
        rank: LoRA rank
        
    Returns:
        Comparison statistics
    """
    full_params = d_in * d_out
    lora_params = rank * (d_in + d_out)
    compression_ratio = full_params / lora_params
    
    # Estimate proof sizes (simplified)
    full_proof_size = full_params * 32  # 32 bytes per field element
    lora_proof_size = lora_params * 32
    
    # Estimate proving times (simplified)
    # Assuming linear scaling with parameter count
    full_proving_time = full_params * 0.001  # 1ms per 1000 params
    lora_proving_time = lora_params * 0.001
    
    return {
        'dimensions': {'d_in': d_in, 'd_out': d_out, 'rank': rank},
        'full_fine_tuning': {
            'params': full_params,
            'proof_size_bytes': full_proof_size,
            'proving_time_ms': full_proving_time
        },
        'lora_fine_tuning': {
            'params': lora_params,
            'proof_size_bytes': lora_proof_size,
            'proving_time_ms': lora_proving_time
        },
        'improvement': {
            'compression_ratio': compression_ratio,
            'proof_size_reduction': full_proof_size / lora_proof_size,
            'speedup': full_proving_time / lora_proving_time
        }
    }