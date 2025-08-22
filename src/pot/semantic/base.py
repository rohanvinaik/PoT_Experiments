"""
Base classes and interfaces for semantic module to avoid circular dependencies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
from enum import Enum


class EmbeddingType(Enum):
    """Types of embeddings supported"""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding operations"""
    dimension: int
    normalize: bool = True
    metric: str = "cosine"
    dtype: type = np.float32


class IConceptLibrary(ABC):
    """
    Abstract interface for concept libraries.
    Allows topography module to use concept library without circular import.
    """
    
    @abstractmethod
    def add_concept(self, name: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """Add a concept to the library"""
        pass
    
    @abstractmethod
    def get_concept(self, name: str) -> Optional[np.ndarray]:
        """Get a concept embedding by name"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar concepts"""
        pass
    
    @abstractmethod
    def get_all_embeddings(self) -> np.ndarray:
        """Get all embeddings as a matrix"""
        pass
    
    @abstractmethod
    def get_concept_names(self) -> List[str]:
        """Get all concept names"""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get number of concepts in library"""
        pass


class ITopographicalProjector(ABC):
    """
    Abstract interface for topographical projectors.
    Allows library module to use projector without circular import.
    """
    
    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> 'ITopographicalProjector':
        """Fit the projector to embeddings"""
        pass
    
    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to projection space"""
        pass
    
    @abstractmethod
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        pass
    
    @abstractmethod
    def get_projection_dim(self) -> int:
        """Get dimensionality of projection space"""
        pass


class IEmbeddingUtils(ABC):
    """
    Abstract interface for embedding utilities.
    Provides common utilities without circular dependencies.
    """
    
    @abstractmethod
    def normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings"""
        pass
    
    @abstractmethod
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray, metric: str = "cosine") -> float:
        """Compute similarity between embeddings"""
        pass
    
    @abstractmethod
    def compute_statistics(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Compute statistics of embeddings"""
        pass


# Concrete implementations of utilities that don't cause circular imports

def normalize_embeddings(embeddings: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize embeddings to unit length.
    
    Args:
        embeddings: Embedding matrix
        axis: Axis along which to normalize
        eps: Small value to avoid division by zero
        
    Returns:
        Normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return embeddings / norms


def compute_embedding_statistics(embeddings: np.ndarray) -> Dict[str, Any]:
    """
    Compute statistics of embedding matrix.
    
    Args:
        embeddings: Embedding matrix
        
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': np.mean(embeddings, axis=0),
        'std': np.std(embeddings, axis=0),
        'min': np.min(embeddings, axis=0),
        'max': np.max(embeddings, axis=0),
        'norm_mean': np.mean(np.linalg.norm(embeddings, axis=1)),
        'norm_std': np.std(np.linalg.norm(embeddings, axis=1)),
        'shape': embeddings.shape,
        'dtype': str(embeddings.dtype)
    }


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding
        emb2: Second embedding
        
    Returns:
        Cosine similarity score
    """
    emb1_norm = normalize_embeddings(emb1.reshape(1, -1))[0]
    emb2_norm = normalize_embeddings(emb2.reshape(1, -1))[0]
    return np.dot(emb1_norm, emb2_norm)


def compute_euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two embeddings.
    
    Args:
        emb1: First embedding
        emb2: Second embedding
        
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(emb1 - emb2)


class EmbeddingSpace:
    """
    Container for managing embedding spaces.
    Provides common functionality without circular dependencies.
    """
    
    def __init__(self, dimension: int, metric: str = "cosine"):
        """
        Initialize embedding space.
        
        Args:
            dimension: Embedding dimension
            metric: Distance metric to use
        """
        self.dimension = dimension
        self.metric = metric
        self.embeddings = []
        self.metadata = []
    
    def add(self, embedding: np.ndarray, metadata: Optional[Dict] = None) -> int:
        """
        Add an embedding to the space.
        
        Args:
            embedding: Embedding vector
            metadata: Optional metadata
            
        Returns:
            Index of added embedding
        """
        if embedding.shape[0] != self.dimension:
            raise ValueError(f"Embedding dimension {embedding.shape[0]} doesn't match space dimension {self.dimension}")
        
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})
        return len(self.embeddings) - 1
    
    def get_matrix(self) -> np.ndarray:
        """
        Get all embeddings as a matrix.
        
        Returns:
            Embedding matrix
        """
        if not self.embeddings:
            return np.empty((0, self.dimension))
        return np.vstack(self.embeddings)
    
    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for nearest neighbors.
        
        Args:
            query: Query embedding
            k: Number of neighbors
            
        Returns:
            List of (index, distance) tuples
        """
        if not self.embeddings:
            return []
        
        matrix = self.get_matrix()
        
        if self.metric == "cosine":
            # Normalize for cosine similarity
            matrix_norm = normalize_embeddings(matrix)
            query_norm = normalize_embeddings(query.reshape(1, -1))[0]
            similarities = np.dot(matrix_norm, query_norm)
            # Convert to distances (1 - similarity)
            distances = 1 - similarities
        elif self.metric == "euclidean":
            distances = np.linalg.norm(matrix - query, axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Get k nearest
        k = min(k, len(distances))
        indices = np.argsort(distances)[:k]
        
        return [(int(idx), float(distances[idx])) for idx in indices]
    
    def clear(self) -> None:
        """Clear all embeddings"""
        self.embeddings = []
        self.metadata = []
    
    def __len__(self) -> int:
        """Get number of embeddings"""
        return len(self.embeddings)