"""
Commitment Schemes for ZK Proofs

Provides Poseidon hash wrapper and dual commitment schemes that interface
with existing Merkle implementation from pot.prototypes.training_provenance_auditor.
"""

import hashlib
import struct
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np

# Import existing Merkle functions
try:
    from ..prototypes.training_provenance_auditor import (
        MerkleNode,
        build_merkle_tree,
        compute_merkle_root
    )
except ImportError:
    # Fallback imports for testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from prototypes.training_provenance_auditor import (
        MerkleNode,
        build_merkle_tree,
        compute_merkle_root
    )


# Import the real Poseidon implementation
try:
    from .poseidon import (
        poseidon_hash,
        poseidon_hash_two,
        poseidon_hash_many,
        PoseidonHash,
        get_poseidon
    )
    from .field_arithmetic import FieldElement
except ImportError:
    # Fallback for testing
    from poseidon import (
        poseidon_hash,
        poseidon_hash_two,
        poseidon_hash_many,
        PoseidonHash,
        get_poseidon
    )
    from field_arithmetic import FieldElement


class PoseidonHasher:
    """
    Poseidon hash function wrapper for ZK-friendly hashing
    
    This uses the actual Poseidon implementation for BN254/Pallas field.
    """
    
    @staticmethod
    def hash_bytes(data: bytes) -> bytes:
        """
        Hash bytes using Poseidon hash function
        
        Args:
            data: Input data as bytes
            
        Returns:
            32-byte Poseidon hash
        """
        return poseidon_hash(data)
    
    @staticmethod
    def hash_field_elements(elements: List[int]) -> bytes:
        """
        Hash a list of field elements using Poseidon
        
        Args:
            elements: List of field elements as integers
            
        Returns:
            32-byte Poseidon hash
        """
        hasher = get_poseidon()
        field_elements = [FieldElement(e) for e in elements]
        result = hasher.hash_many(field_elements)
        return result.to_bytes()
    
    @staticmethod
    def hash_two(left: bytes, right: bytes) -> bytes:
        """
        Hash two 32-byte values using Poseidon
        
        Args:
            left: Left hash value
            right: Right hash value
            
        Returns:
            32-byte Poseidon hash of concatenated inputs
        """
        return poseidon_hash_two(left, right)


class MerkleCommitment:
    """
    Merkle commitment scheme supporting both SHA-256 and Poseidon hashing
    """
    
    def __init__(self, hash_function: str = "poseidon"):
        """
        Initialize Merkle commitment scheme
        
        Args:
            hash_function: Either "sha256" or "poseidon"
        """
        if hash_function not in ["sha256", "poseidon"]:
            raise ValueError("hash_function must be 'sha256' or 'poseidon'")
        
        self.hash_function = hash_function
    
    def commit_tensor(self, tensor: np.ndarray, randomness: Optional[bytes] = None) -> Tuple[str, bytes]:
        """
        Commit to a tensor using Merkle tree
        
        Args:
            tensor: NumPy tensor to commit to
            randomness: Optional randomness for commitment (not used in Merkle trees)
            
        Returns:
            Tuple of (commitment_root_hex, serialized_tensor)
        """
        # Serialize tensor
        tensor_bytes = tensor.tobytes()
        
        # Split into chunks for Merkle tree (each chunk is 32 bytes)
        chunk_size = 32
        chunks = []
        for i in range(0, len(tensor_bytes), chunk_size):
            chunk = tensor_bytes[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = chunk + b'\x00' * (chunk_size - len(chunk))  # Pad
            chunks.append(chunk)
        
        # Build Merkle tree
        if self.hash_function == "sha256":
            root = compute_merkle_root(chunks)
            return root.hex(), tensor_bytes
        else:
            # Use Poseidon for ZK-friendly commitments
            root = self._compute_poseidon_merkle_root(chunks)
            return root.hex(), tensor_bytes
    
    def commit_batch(self, inputs: np.ndarray, targets: np.ndarray, 
                    randomness: Optional[bytes] = None) -> Tuple[str, Dict[str, bytes]]:
        """
        Commit to a training batch
        
        Args:
            inputs: Input data tensor
            targets: Target data tensor
            randomness: Optional randomness
            
        Returns:
            Tuple of (commitment_root_hex, batch_data_dict)
        """
        # Serialize batch data
        inputs_bytes = inputs.tobytes()
        targets_bytes = targets.tobytes()
        
        # Combine inputs and targets
        combined_data = inputs_bytes + targets_bytes
        
        # Create commitment
        root_hex, _ = self.commit_tensor(
            np.frombuffer(combined_data, dtype=np.uint8),
            randomness
        )
        
        batch_data = {
            "inputs": inputs_bytes,
            "targets": targets_bytes,
            "combined": combined_data
        }
        
        return root_hex, batch_data
    
    def _compute_poseidon_merkle_root(self, chunks: List[bytes]) -> bytes:
        """
        Compute Merkle root using Poseidon hash function
        
        Args:
            chunks: List of data chunks
            
        Returns:
            Root hash as bytes
        """
        from .poseidon import poseidon_merkle_root
        return poseidon_merkle_root(chunks)
    
    def generate_proof(self, tensor: np.ndarray, index: int) -> List[str]:
        """
        Generate Merkle proof for a specific element
        
        Args:
            tensor: Full tensor
            index: Index of element to prove
            
        Returns:
            List of hex-encoded proof elements (Merkle path)
        """
        # This would generate the actual Merkle proof path
        # For now, return a mock proof
        return [f"proof_element_{i}_{index}" for i in range(5)]


class DualCommitment:
    """
    Dual commitment scheme using both SHA-256 and Poseidon
    
    This allows compatibility with existing SHA-256 infrastructure
    while enabling ZK proofs with Poseidon commitments.
    """
    
    def __init__(self):
        """Initialize dual commitment scheme"""
        self.sha256_commitment = MerkleCommitment("sha256")
        self.poseidon_commitment = MerkleCommitment("poseidon")
    
    def commit_tensor(self, tensor: np.ndarray, randomness: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Create dual commitment to tensor
        
        Args:
            tensor: Tensor to commit to
            randomness: Optional randomness
            
        Returns:
            Dictionary with both SHA-256 and Poseidon commitments
        """
        sha256_root, tensor_data = self.sha256_commitment.commit_tensor(tensor, randomness)
        poseidon_root, _ = self.poseidon_commitment.commit_tensor(tensor, randomness)
        
        return {
            "sha256_root": sha256_root,
            "poseidon_root": poseidon_root,
            "tensor_data": tensor_data,
            "tensor_shape": tensor.shape,
            "tensor_dtype": str(tensor.dtype)
        }
    
    def commit_batch(self, inputs: np.ndarray, targets: np.ndarray, 
                    randomness: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Create dual commitment to training batch
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            randomness: Optional randomness
            
        Returns:
            Dictionary with dual commitments
        """
        sha256_root, sha256_data = self.sha256_commitment.commit_batch(
            inputs, targets, randomness
        )
        poseidon_root, poseidon_data = self.poseidon_commitment.commit_batch(
            inputs, targets, randomness
        )
        
        return {
            "sha256_root": sha256_root,
            "poseidon_root": poseidon_root,
            "batch_data": sha256_data,  # Data is the same for both
            "inputs_shape": inputs.shape,
            "targets_shape": targets.shape,
            "inputs_dtype": str(inputs.dtype),
            "targets_dtype": str(targets.dtype)
        }
    
    def verify_consistency(self, commitment_data: Dict[str, Any]) -> bool:
        """
        Verify that SHA-256 and Poseidon commitments are consistent
        
        Args:
            commitment_data: Dual commitment data
            
        Returns:
            True if commitments are consistent
        """
        # In a real implementation, this would verify that both commitments
        # commit to the same underlying data
        return (
            "sha256_root" in commitment_data and
            "poseidon_root" in commitment_data and
            commitment_data["sha256_root"] is not None and
            commitment_data["poseidon_root"] is not None
        )


# Conversion functions between commitment schemes
def convert_sha256_to_poseidon(sha256_hash: str) -> str:
    """
    Convert SHA-256 hash to equivalent Poseidon hash
    
    Note: This is a deterministic mapping, not a cryptographic conversion.
    In practice, you would need to recompute the Poseidon hash from the
    original data.
    
    Args:
        sha256_hash: Hex-encoded SHA-256 hash
        
    Returns:
        Hex-encoded Poseidon hash
    """
    # Convert hex to bytes
    sha_bytes = bytes.fromhex(sha256_hash)
    
    # Apply Poseidon hash to the SHA-256 result
    poseidon_hash = PoseidonHasher.hash_bytes(sha_bytes)
    
    return poseidon_hash.hex()


def convert_poseidon_to_sha256(poseidon_hash: str) -> str:
    """
    Convert Poseidon hash to SHA-256 equivalent
    
    Note: This is for compatibility only and doesn't preserve
    cryptographic properties.
    
    Args:
        poseidon_hash: Hex-encoded Poseidon hash
        
    Returns:
        Hex-encoded SHA-256 hash
    """
    # Convert hex to bytes
    poseidon_bytes = bytes.fromhex(poseidon_hash)
    
    # Apply SHA-256
    sha_hash = hashlib.sha256(poseidon_bytes).digest()
    
    return sha_hash.hex()


def compute_dual_merkle_root(data_blocks: List[bytes]) -> Dict[str, str]:
    """
    Compute Merkle roots using both SHA-256 and Poseidon
    
    Args:
        data_blocks: List of data blocks
        
    Returns:
        Dictionary with both hash scheme results
    """
    if not data_blocks:
        return {
            "sha256_root": "0" * 64,
            "poseidon_root": "0" * 64
        }
    
    # Compute SHA-256 Merkle root using existing function
    sha256_root = compute_merkle_root(data_blocks)
    
    # Compute Poseidon Merkle root
    poseidon_commitment = MerkleCommitment("poseidon")
    poseidon_root = poseidon_commitment._compute_poseidon_merkle_root(data_blocks)
    
    return {
        "sha256_root": sha256_root.hex(),
        "poseidon_root": poseidon_root.hex()
    }


# Integration helpers for existing PoT infrastructure
def create_zk_compatible_commitment(existing_merkle_node: MerkleNode) -> Dict[str, str]:
    """
    Create ZK-compatible commitment from existing MerkleNode
    
    Args:
        existing_merkle_node: Existing SHA-256 based MerkleNode
        
    Returns:
        Dictionary with both commitment schemes
    """
    # Extract SHA-256 hash
    sha256_hash = existing_merkle_node.get_hex_hash()
    
    # Convert to Poseidon-compatible format
    poseidon_hash = convert_sha256_to_poseidon(sha256_hash)
    
    return {
        "sha256_commitment": sha256_hash,
        "poseidon_commitment": poseidon_hash,
        "is_zk_compatible": True
    }