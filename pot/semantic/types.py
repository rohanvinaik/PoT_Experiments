"""
Type definitions for semantic verification in PoT experiments.
Defines core data structures and configuration classes for concept vectors and semantic matching.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib


class SemanticDistance(Enum):
    """Enumeration of supported semantic distance metrics."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    JACCARD = "jaccard"
    JENSEN_SHANNON = "jensen_shannon"


@dataclass
class ConceptVector:
    """
    Represents a concept vector with associated metadata.
    
    Attributes:
        vector: The numerical representation of the concept
        concept_id: Unique identifier for the concept
        label: Human-readable label for the concept
        source: Source of the concept (e.g., 'manual', 'extracted', 'synthetic')
        metadata: Additional metadata about the concept
        timestamp: Creation timestamp
        hash_value: SHA256 hash of the vector for integrity checking
    """
    vector: np.ndarray
    concept_id: str
    label: Optional[str] = None
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    hash_value: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization to compute hash and validate vector."""
        if self.hash_value is None:
            self.hash_value = self._compute_hash()
        self._validate_vector()
    
    def _compute_hash(self) -> str:
        """Compute SHA256 hash of the vector for integrity checking."""
        vector_bytes = self.vector.tobytes()
        return hashlib.sha256(vector_bytes).hexdigest()
    
    def _validate_vector(self) -> None:
        """Validate that the vector is properly formed."""
        if not isinstance(self.vector, np.ndarray):
            raise TypeError("vector must be a numpy array")
        if self.vector.ndim != 1:
            raise ValueError("vector must be 1-dimensional")
        if len(self.vector) == 0:
            raise ValueError("vector cannot be empty")
        if not np.isfinite(self.vector).all():
            raise ValueError("vector must contain only finite values")
    
    @property
    def dimension(self) -> int:
        """Return the dimensionality of the concept vector."""
        return len(self.vector)
    
    @property
    def norm(self) -> float:
        """Return the L2 norm of the concept vector."""
        return float(np.linalg.norm(self.vector))


@dataclass
class SemanticMatchResult:
    """
    Result of semantic matching between concept vectors.
    
    Attributes:
        query_concept_id: ID of the query concept
        matched_concept_id: ID of the matched concept
        distance: Semantic distance between concepts
        similarity_score: Similarity score (0-1, higher is more similar)
        distance_metric: Distance metric used for comparison
        confidence: Confidence level of the match
        metadata: Additional matching metadata
    """
    query_concept_id: str
    matched_concept_id: str
    distance: float
    similarity_score: float
    distance_metric: SemanticDistance
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not 0 <= self.similarity_score <= 1:
            raise ValueError("similarity_score must be between 0 and 1")
        if self.distance < 0:
            raise ValueError("distance must be non-negative")


@dataclass
class MatchingConfig:
    """
    Configuration for semantic matching operations.
    
    Attributes:
        distance_metric: Primary distance metric to use
        similarity_threshold: Minimum similarity score for positive matches
        max_candidates: Maximum number of candidate matches to consider
        normalize_vectors: Whether to normalize vectors before matching
        use_approximate_search: Whether to use approximate nearest neighbor search
        batch_size: Batch size for processing multiple queries
        cache_results: Whether to cache matching results
        fallback_metrics: Alternative metrics to try if primary fails
    """
    distance_metric: SemanticDistance = SemanticDistance.COSINE
    similarity_threshold: float = 0.8
    max_candidates: int = 10
    normalize_vectors: bool = True
    use_approximate_search: bool = False
    batch_size: int = 32
    cache_results: bool = True
    fallback_metrics: List[SemanticDistance] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass
class SemanticLibrary:
    """
    Container for a collection of concept vectors with metadata.
    
    Attributes:
        concepts: Dictionary mapping concept IDs to ConceptVector objects
        name: Name of the semantic library
        version: Version string for the library
        description: Description of the library contents
        created_timestamp: Creation timestamp
        modified_timestamp: Last modification timestamp
        metadata: Additional library metadata
    """
    concepts: Dict[str, ConceptVector] = field(default_factory=dict)
    name: str = "unnamed_library"
    version: str = "1.0.0"
    description: str = ""
    created_timestamp: Optional[float] = None
    modified_timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Return the number of concepts in the library."""
        return len(self.concepts)
    
    def __contains__(self, concept_id: str) -> bool:
        """Check if a concept ID exists in the library."""
        return concept_id in self.concepts
    
    def add_concept(self, concept: ConceptVector) -> None:
        """Add a concept to the library."""
        self.concepts[concept.concept_id] = concept
        import time
        self.modified_timestamp = time.time()
    
    def remove_concept(self, concept_id: str) -> bool:
        """Remove a concept from the library. Returns True if removed, False if not found."""
        if concept_id in self.concepts:
            del self.concepts[concept_id]
            import time
            self.modified_timestamp = time.time()
            return True
        return False
    
    def get_concept(self, concept_id: str) -> Optional[ConceptVector]:
        """Get a concept by ID, or None if not found."""
        return self.concepts.get(concept_id)
    
    def list_concept_ids(self) -> List[str]:
        """Return a list of all concept IDs in the library."""
        return list(self.concepts.keys())
    
    @property
    def dimension(self) -> Optional[int]:
        """Return the common dimension of all vectors, or None if empty/inconsistent."""
        if not self.concepts:
            return None
        
        dimensions = {concept.dimension for concept in self.concepts.values()}
        return dimensions.pop() if len(dimensions) == 1 else None
    
    def validate_consistency(self) -> Tuple[bool, List[str]]:
        """
        Validate that all concepts in the library are consistent.
        Returns (is_valid, list_of_errors).
        """
        errors = []
        
        if not self.concepts:
            return True, []
        
        # Check dimension consistency
        dimensions = {concept.dimension for concept in self.concepts.values()}
        if len(dimensions) > 1:
            errors.append(f"Inconsistent vector dimensions: {dimensions}")
        
        # Check for duplicate concept IDs (should not happen given dict structure)
        concept_ids = [concept.concept_id for concept in self.concepts.values()]
        if len(set(concept_ids)) != len(concept_ids):
            errors.append("Duplicate concept IDs found")
        
        # Validate individual concepts
        for concept_id, concept in self.concepts.items():
            if concept.concept_id != concept_id:
                errors.append(f"Concept ID mismatch: key='{concept_id}', concept.concept_id='{concept.concept_id}'")
        
        return len(errors) == 0, errors