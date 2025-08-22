"""
Core interfaces and abstract base classes to break circular dependencies.
This module provides common interfaces that different components can depend on
without creating circular imports.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib


class EventType(Enum):
    """Event types for provenance tracking"""
    TRAINING_START = "training_start"
    TRAINING_END = "training_end"
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    CHECKPOINT = "checkpoint"
    VALIDATION = "validation"
    MODEL_UPDATE = "model_update"
    HYPERPARAMETER_CHANGE = "hyperparameter_change"
    DATA_LOAD = "data_load"
    CUSTOM = "custom"


class ProofType(Enum):
    """Types of proofs that can be generated"""
    MERKLE_TREE = "merkle_tree"
    MERKLE = "merkle"  # Alias for backward compatibility
    ZK_PROOF = "zk_proof"
    ZERO_KNOWLEDGE = "zero_knowledge"  # Alias for backward compatibility
    SIGNATURE = "signature"
    TIMESTAMP = "timestamp"
    COMPOSITE = "composite"


@dataclass
class MerkleNode:
    """
    Basic Merkle tree node structure.
    Used by both training_provenance_auditor and ZK modules.
    """
    hash: str
    data: Optional[Any] = None
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf"""
        return self.left is None and self.right is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'hash': self.hash,
            'is_leaf': self.is_leaf(),
            'data': self.data if self.is_leaf() else None
        }


class IMerkleTree(ABC):
    """
    Abstract interface for Merkle tree implementations.
    Both SHA256 and Poseidon implementations should inherit from this.
    """
    
    @abstractmethod
    def add_leaf(self, data: Any) -> None:
        """Add a leaf to the tree"""
        pass
    
    @abstractmethod
    def get_root(self) -> Optional[str]:
        """Get the root hash of the tree"""
        pass
    
    @abstractmethod
    def get_proof(self, index: int) -> List[str]:
        """Get Merkle proof for a leaf at given index"""
        pass
    
    @abstractmethod
    def verify_proof(self, leaf_hash: str, proof: List[str], root: str) -> bool:
        """Verify a Merkle proof"""
        pass
    
    @abstractmethod
    def get_leaves(self) -> List[Any]:
        """Get all leaves in the tree"""
        pass


class ICommitment(ABC):
    """
    Abstract interface for commitment schemes.
    Used by both ZK and provenance modules.
    """
    
    @abstractmethod
    def commit(self, data: Any) -> str:
        """Create a commitment to data"""
        pass
    
    @abstractmethod
    def verify(self, data: Any, commitment: str) -> bool:
        """Verify a commitment"""
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        """Get the commitment type identifier"""
        pass


class IWitnessExtractor(ABC):
    """
    Abstract interface for witness extraction.
    Used by ZK modules to extract witness data.
    """
    
    @abstractmethod
    def extract_witness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract witness data from input"""
        pass
    
    @abstractmethod
    def validate_witness(self, witness: Dict[str, Any]) -> bool:
        """Validate witness data structure"""
        pass


class IProvenanceAuditor(ABC):
    """
    Abstract interface for provenance auditing.
    Core interface that both training_provenance_auditor and ZK modules can use.
    """
    
    @abstractmethod
    def log_event(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """Log a training event"""
        pass
    
    @abstractmethod
    def generate_proof(self, proof_type: ProofType) -> Dict[str, Any]:
        """Generate a proof of training"""
        pass
    
    @abstractmethod
    def verify_proof(self, proof: Dict[str, Any]) -> bool:
        """Verify a proof"""
        pass
    
    @abstractmethod
    def get_merkle_root(self) -> Optional[str]:
        """Get current Merkle root"""
        pass


class BasicMerkleTree(IMerkleTree):
    """
    Basic SHA256-based Merkle tree implementation.
    This is a simple implementation that can be used without circular imports.
    """
    
    def __init__(self):
        self.leaves = []
        self.nodes = []
        self.root = None
    
    def _hash(self, data: str) -> str:
        """Compute SHA256 hash"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _hash_pair(self, left: str, right: str) -> str:
        """Hash two nodes together"""
        combined = left + right
        return self._hash(combined)
    
    def add_leaf(self, data: Any) -> None:
        """Add a leaf to the tree"""
        leaf_hash = self._hash(str(data))
        self.leaves.append(leaf_hash)
        self._rebuild_tree()
    
    def _rebuild_tree(self) -> None:
        """Rebuild the tree from leaves"""
        if not self.leaves:
            self.root = None
            return
        
        if len(self.leaves) == 1:
            self.root = self.leaves[0]
            return
        
        # Build tree level by level
        current_level = self.leaves.copy()
        
        while len(current_level) > 1:
            next_level = []
            
            # Process pairs
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Hash pair
                    combined = self._hash_pair(current_level[i], current_level[i + 1])
                else:
                    # Odd number, carry forward
                    combined = current_level[i]
                next_level.append(combined)
            
            current_level = next_level
        
        self.root = current_level[0] if current_level else None
    
    def get_root(self) -> Optional[str]:
        """Get the root hash"""
        return self.root
    
    def get_proof(self, index: int) -> List[str]:
        """Get Merkle proof for a leaf"""
        if index >= len(self.leaves):
            return []
        
        proof = []
        current_index = index
        current_level = self.leaves.copy()
        
        while len(current_level) > 1:
            # Find sibling
            if current_index % 2 == 0:
                # Right sibling
                if current_index + 1 < len(current_level):
                    proof.append(current_level[current_index + 1])
            else:
                # Left sibling
                proof.append(current_level[current_index - 1])
            
            # Move to next level
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    combined = self._hash_pair(current_level[i], current_level[i + 1])
                else:
                    combined = current_level[i]
                next_level.append(combined)
            
            current_level = next_level
            current_index = current_index // 2
        
        return proof
    
    def verify_proof(self, leaf_hash: str, proof: List[str], root: str) -> bool:
        """Verify a Merkle proof"""
        current = leaf_hash
        
        for sibling in proof:
            # Determine order based on comparison
            if current < sibling:
                current = self._hash_pair(current, sibling)
            else:
                current = self._hash_pair(sibling, current)
        
        return current == root
    
    def get_leaves(self) -> List[Any]:
        """Get all leaves"""
        return self.leaves.copy()


class BasicCommitment(ICommitment):
    """
    Basic SHA256-based commitment implementation.
    """
    
    def __init__(self, salt: Optional[str] = None):
        self.salt = salt or "default_salt"
    
    def commit(self, data: Any) -> str:
        """Create a commitment"""
        data_str = str(data) + self.salt
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def verify(self, data: Any, commitment: str) -> bool:
        """Verify a commitment"""
        return self.commit(data) == commitment
    
    def get_type(self) -> str:
        """Get commitment type"""
        return "sha256"


# Factory functions to create implementations without circular imports

def create_merkle_tree(tree_type: str = "sha256") -> IMerkleTree:
    """
    Factory function to create Merkle tree instances.
    This allows modules to create trees without importing specific implementations.
    """
    if tree_type == "sha256":
        return BasicMerkleTree()
    elif tree_type == "poseidon":
        # Import here to avoid circular dependency
        try:
            from ..zk.poseidon import PoseidonMerkleTree
            return PoseidonMerkleTree()
        except ImportError:
            # Fallback to basic implementation
            return BasicMerkleTree()
    else:
        return BasicMerkleTree()


def create_commitment(commitment_type: str = "sha256", **kwargs) -> ICommitment:
    """
    Factory function to create commitment instances.
    """
    if commitment_type == "sha256":
        return BasicCommitment(**kwargs)
    elif commitment_type == "poseidon":
        # Import here to avoid circular dependency
        try:
            from ..zk.commitments import PoseidonCommitment
            return PoseidonCommitment(**kwargs)
        except ImportError:
            return BasicCommitment(**kwargs)
    else:
        return BasicCommitment(**kwargs)


# Export common data structures that don't cause circular imports

class TrainingEvent:
    """Data structure for training events"""
    
    def __init__(self, event_type: EventType, timestamp: float, data: Dict[str, Any]):
        self.event_type = event_type
        self.timestamp = timestamp
        self.data = data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'data': self.data
        }
    
    def get_hash(self) -> str:
        """Get hash of the event"""
        event_str = f"{self.event_type.value}:{self.timestamp}:{str(self.data)}"
        return hashlib.sha256(event_str.encode()).hexdigest()


class ProvenanceProof:
    """Data structure for provenance proofs"""
    
    def __init__(self, proof_type: ProofType, root: str, proof_data: Dict[str, Any]):
        self.proof_type = proof_type
        self.root = root
        self.proof_data = proof_data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'proof_type': self.proof_type.value,
            'root': self.root,
            'proof_data': self.proof_data
        }
    
    def verify(self, verifier: Optional[Any] = None) -> bool:
        """Verify the proof"""
        if self.proof_type == ProofType.MERKLE_TREE:
            # Basic verification
            return self.root is not None and len(self.root) == 64  # SHA256 hex length
        return False