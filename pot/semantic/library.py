"""
Concept vector library management for semantic verification.
Provides classes and functions for storing, retrieving, and managing concept vectors.
"""

import os
import json
import pickle
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
from pathlib import Path
import logging
import hashlib

from .types import ConceptVector, SemanticLibrary
from ..core.utils import set_deterministic

logger = logging.getLogger(__name__)


class ConceptVectorManager:
    """
    Manager for concept vectors with support for CRUD operations, persistence, and validation.
    """
    
    def __init__(self, library: Optional[SemanticLibrary] = None):
        """
        Initialize the concept vector manager.
        
        Args:
            library: Optional pre-existing semantic library to manage
        """
        self.library = library or SemanticLibrary()
        self._index_cache = {}
        self._dirty = False
    
    def add_concept(self, vector: np.ndarray, concept_id: str, 
                   label: Optional[str] = None, source: str = "manual",
                   metadata: Optional[Dict[str, Any]] = None) -> ConceptVector:
        """
        Add a new concept vector to the library.
        
        Args:
            vector: Numpy array representing the concept
            concept_id: Unique identifier for the concept
            label: Human-readable label for the concept
            source: Source of the concept
            metadata: Additional metadata
            
        Returns:
            The created ConceptVector object
            
        Raises:
            ValueError: If concept_id already exists or vector is invalid
        """
        if concept_id in self.library:
            raise ValueError(f"Concept ID '{concept_id}' already exists")
        
        concept = ConceptVector(
            vector=vector.copy(),
            concept_id=concept_id,
            label=label,
            source=source,
            metadata=metadata or {},
            timestamp=time.time()
        )
        
        self.library.add_concept(concept)
        self._dirty = True
        self._invalidate_cache()
        
        logger.info(f"Added concept '{concept_id}' to library")
        return concept
    
    def remove_concept(self, concept_id: str) -> bool:
        """
        Remove a concept from the library.
        
        Args:
            concept_id: ID of the concept to remove
            
        Returns:
            True if concept was removed, False if not found
        """
        removed = self.library.remove_concept(concept_id)
        if removed:
            self._dirty = True
            self._invalidate_cache()
            logger.info(f"Removed concept '{concept_id}' from library")
        return removed
    
    def get_concept(self, concept_id: str) -> Optional[ConceptVector]:
        """
        Retrieve a concept by ID.
        
        Args:
            concept_id: ID of the concept to retrieve
            
        Returns:
            ConceptVector if found, None otherwise
        """
        return self.library.get_concept(concept_id)
    
    def update_concept(self, concept_id: str, **updates) -> bool:
        """
        Update an existing concept's metadata or properties.
        
        Args:
            concept_id: ID of the concept to update
            **updates: Fields to update (label, metadata, etc.)
            
        Returns:
            True if concept was updated, False if not found
        """
        concept = self.library.get_concept(concept_id)
        if concept is None:
            return False
        
        # Update allowed fields
        if 'label' in updates:
            concept.label = updates['label']
        if 'metadata' in updates:
            concept.metadata.update(updates['metadata'])
        if 'source' in updates:
            concept.source = updates['source']
        
        self._dirty = True
        self.library.modified_timestamp = time.time()
        logger.info(f"Updated concept '{concept_id}'")
        return True
    
    def list_concepts(self, source_filter: Optional[str] = None,
                     label_pattern: Optional[str] = None) -> List[ConceptVector]:
        """
        List concepts with optional filtering.
        
        Args:
            source_filter: Filter by source type
            label_pattern: Filter by label pattern (simple substring match)
            
        Returns:
            List of matching ConceptVector objects
        """
        concepts = list(self.library.concepts.values())
        
        if source_filter:
            concepts = [c for c in concepts if c.source == source_filter]
        
        if label_pattern:
            concepts = [c for c in concepts if c.label and label_pattern in c.label]
        
        return concepts
    
    def get_concept_matrix(self, concept_ids: Optional[List[str]] = None) -> np.ndarray:
        """
        Get concept vectors as a matrix for batch operations.
        
        Args:
            concept_ids: Specific concept IDs to include, or None for all
            
        Returns:
            Matrix where each row is a concept vector
            
        Raises:
            ValueError: If library is empty or dimensions are inconsistent
        """
        if not self.library.concepts:
            raise ValueError("Library is empty")
        
        if concept_ids is None:
            concept_ids = self.library.list_concept_ids()
        
        concepts = [self.library.get_concept(cid) for cid in concept_ids]
        concepts = [c for c in concepts if c is not None]
        
        if not concepts:
            raise ValueError("No valid concepts found")
        
        # Check dimension consistency
        dimensions = {c.dimension for c in concepts}
        if len(dimensions) > 1:
            raise ValueError(f"Inconsistent vector dimensions: {dimensions}")
        
        return np.vstack([c.vector for c in concepts])
    
    def validate_library(self) -> Tuple[bool, List[str]]:
        """
        Validate the entire library for consistency and integrity.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        return self.library.validate_consistency()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the concept library.
        
        Returns:
            Dictionary with library statistics
        """
        if not self.library.concepts:
            return {
                'total_concepts': 0,
                'dimension': None,
                'sources': {},
                'has_labels': 0
            }
        
        concepts = list(self.library.concepts.values())
        sources = {}
        for concept in concepts:
            sources[concept.source] = sources.get(concept.source, 0) + 1
        
        return {
            'total_concepts': len(concepts),
            'dimension': self.library.dimension,
            'sources': sources,
            'has_labels': sum(1 for c in concepts if c.label),
            'created_timestamp': self.library.created_timestamp,
            'modified_timestamp': self.library.modified_timestamp
        }
    
    def _invalidate_cache(self) -> None:
        """Invalidate internal caches."""
        self._index_cache.clear()


class ConceptLibrary:
    """
    High-level interface for concept library operations with persistence support.
    """
    
    def __init__(self, library_path: Optional[Union[str, Path]] = None):
        """
        Initialize the concept library.
        
        Args:
            library_path: Optional path to load/save the library
        """
        self.library_path = Path(library_path) if library_path else None
        self.manager = ConceptVectorManager()
        
        if self.library_path and self.library_path.exists():
            self.load()
    
    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Load library from disk.
        
        Args:
            path: Path to load from, or use instance library_path
        """
        load_path = Path(path) if path else self.library_path
        if not load_path:
            raise ValueError("No load path specified")
        
        if not load_path.exists():
            raise FileNotFoundError(f"Library file not found: {load_path}")
        
        library = load_concept_library(load_path)
        self.manager.library = library
        self.manager._dirty = False
        
        logger.info(f"Loaded library from {load_path}")
    
    def save(self, path: Optional[Union[str, Path]] = None, 
             format: str = 'pickle') -> None:
        """
        Save library to disk.
        
        Args:
            path: Path to save to, or use instance library_path
            format: Save format ('pickle' or 'json')
        """
        save_path = Path(path) if path else self.library_path
        if not save_path:
            raise ValueError("No save path specified")
        
        save_concept_library(self.manager.library, save_path, format=format)
        self.manager._dirty = False
        
        logger.info(f"Saved library to {save_path}")
    
    def auto_save(self) -> None:
        """Automatically save if library has been modified and path is set."""
        if self.manager._dirty and self.library_path:
            self.save()
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the manager."""
        return getattr(self.manager, name)


def load_concept_library(path: Union[str, Path], 
                        format: Optional[str] = None) -> SemanticLibrary:
    """
    Load a concept library from disk.
    
    Args:
        path: Path to the library file
        format: File format ('pickle', 'json', or None for auto-detect)
        
    Returns:
        Loaded SemanticLibrary object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is unsupported or file is corrupted
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Library file not found: {path}")
    
    # Auto-detect format if not specified
    if format is None:
        if path.suffix in ['.pkl', '.pickle', '.pt']:
            format = 'pickle'
        elif path.suffix == '.json':
            format = 'json'
        else:
            # Try to detect by file content
            try:
                with open(path, 'r') as f:
                    json.load(f)
                format = 'json'
            except (json.JSONDecodeError, UnicodeDecodeError):
                format = 'pickle'
    
    try:
        if format == 'pickle':
            with open(path, 'rb') as f:
                library = pickle.load(f)
        elif format == 'json':
            with open(path, 'r') as f:
                data = json.load(f)
            library = _deserialize_library_from_json(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Validate loaded library
        is_valid, errors = library.validate_consistency()
        if not is_valid:
            raise ValueError(f"Loaded library is invalid: {errors}")
        
        return library
        
    except Exception as e:
        raise ValueError(f"Failed to load library from {path}: {e}")


def save_concept_library(library: SemanticLibrary, path: Union[str, Path],
                        format: Optional[str] = None) -> None:
    """
    Save a concept library to disk.
    
    Args:
        library: SemanticLibrary to save
        path: Path to save the library
        format: Save format ('pickle' or 'json', or None for auto-detect)
        
    Raises:
        ValueError: If format is unsupported or library is invalid
    """
    path = Path(path)
    
    # Auto-detect format if not specified
    if format is None:
        if path.suffix in ['.pkl', '.pickle', '.pt']:
            format = 'pickle'
        elif path.suffix == '.json':
            format = 'json'
        else:
            # Default to pickle if extension is unknown
            format = 'pickle'
    
    # Validate library before saving
    is_valid, errors = library.validate_consistency()
    if not is_valid:
        raise ValueError(f"Cannot save invalid library: {errors}")
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(library, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif format == 'json':
            data = _serialize_library_to_json(library)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    except Exception as e:
        raise ValueError(f"Failed to save library to {path}: {e}")


def _serialize_library_to_json(library: SemanticLibrary) -> Dict[str, Any]:
    """Serialize a SemanticLibrary to JSON-compatible format."""
    concepts_data = {}
    for concept_id, concept in library.concepts.items():
        concepts_data[concept_id] = {
            'vector': concept.vector.tolist(),
            'concept_id': concept.concept_id,
            'label': concept.label,
            'source': concept.source,
            'metadata': concept.metadata,
            'timestamp': concept.timestamp,
            'hash_value': concept.hash_value
        }
    
    return {
        'concepts': concepts_data,
        'name': library.name,
        'version': library.version,
        'description': library.description,
        'created_timestamp': library.created_timestamp,
        'modified_timestamp': library.modified_timestamp,
        'metadata': library.metadata
    }


def _deserialize_library_from_json(data: Dict[str, Any]) -> SemanticLibrary:
    """Deserialize a SemanticLibrary from JSON format."""
    concepts = {}
    for concept_id, concept_data in data['concepts'].items():
        vector = np.array(concept_data['vector'])
        concept = ConceptVector(
            vector=vector,
            concept_id=concept_data['concept_id'],
            label=concept_data.get('label'),
            source=concept_data.get('source', 'unknown'),
            metadata=concept_data.get('metadata', {}),
            timestamp=concept_data.get('timestamp'),
            hash_value=concept_data.get('hash_value')
        )
        concepts[concept_id] = concept
    
    return SemanticLibrary(
        concepts=concepts,
        name=data.get('name', 'unnamed_library'),
        version=data.get('version', '1.0.0'),
        description=data.get('description', ''),
        created_timestamp=data.get('created_timestamp'),
        modified_timestamp=data.get('modified_timestamp'),
        metadata=data.get('metadata', {})
    )


class ConceptLibrary:
    """
    Enhanced concept library for building and managing concept vectors from reference datasets.
    Supports both Gaussian statistical modeling and hypervector representations.
    
    This class builds concept vectors from embeddings extracted from teacher models or 
    reference datasets, storing statistical parameters (mean, covariance) for each concept.
    """
    
    def __init__(self, dim: int, method: str = 'gaussian'):
        """
        Initialize the concept library.
        
        Args:
            dim: Dimensionality of concept vectors
            method: Representation method ('gaussian' or 'hypervector')
            
        Raises:
            ValueError: If method is unsupported or dim is invalid
        """
        if dim <= 0:
            raise ValueError("Dimension must be positive")
        
        if method not in ['gaussian', 'hypervector']:
            raise ValueError("Method must be 'gaussian' or 'hypervector'")
        
        self.dim = dim
        self.method = method
        self.concepts = {}  # concept_name -> concept_data
        self.metadata = {
            'created_timestamp': time.time(),
            'modified_timestamp': time.time(),
            'version': '1.0.0',
            'total_embeddings_processed': 0
        }
        
        # Set deterministic behavior for reproducibility
        set_deterministic(42)
        
        logger.info(f"Initialized ConceptLibrary with dim={dim}, method={method}")
    
    def add_concept(self, name: str, embeddings: torch.Tensor) -> None:
        """
        Add a concept by computing statistics from embedding samples.
        
        Args:
            name: Concept name/identifier
            embeddings: Tensor of shape (n_samples, dim) containing embedding vectors
            
        Raises:
            ValueError: If embeddings have wrong shape or concept already exists
        """
        if name in self.concepts:
            raise ValueError(f"Concept '{name}' already exists")
        
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("embeddings must be a torch.Tensor")
        
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2-dimensional (n_samples, dim)")
        
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"embeddings dimension {embeddings.shape[1]} doesn't match library dimension {self.dim}")
        
        if embeddings.shape[0] == 0:
            raise ValueError("embeddings cannot be empty")
        
        # Compute statistics based on method
        if self.method == 'gaussian':
            mean, covariance = self.compute_statistics(embeddings)
            concept_vector = mean  # Use mean as the concept vector
        elif self.method == 'hypervector':
            concept_vector = self._build_hypervector(embeddings)
            mean, covariance = None, None
        
        # Store concept data
        concept_data = {
            'vector': concept_vector,
            'mean': mean,
            'covariance': covariance,
            'n_samples': embeddings.shape[0],
            'method': self.method,
            'timestamp': time.time(),
            'hash': self._compute_concept_hash(concept_vector)
        }
        
        self.concepts[name] = concept_data
        self.metadata['modified_timestamp'] = time.time()
        self.metadata['total_embeddings_processed'] += embeddings.shape[0]
        
        logger.info(f"Added concept '{name}' with {embeddings.shape[0]} samples")
    
    def compute_statistics(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and covariance statistics from embeddings.
        
        Args:
            embeddings: Tensor of shape (n_samples, dim)
            
        Returns:
            Tuple of (mean, covariance) tensors
            
        Raises:
            ValueError: If embeddings are invalid
        """
        if embeddings.shape[0] < 2:
            raise ValueError("Need at least 2 samples to compute covariance")
        
        # Compute mean
        mean = torch.mean(embeddings, dim=0)
        
        # Compute covariance matrix
        centered = embeddings - mean
        covariance = torch.mm(centered.t(), centered) / (embeddings.shape[0] - 1)
        
        # Add small regularization for numerical stability
        regularization = 1e-6 * torch.eye(self.dim, dtype=embeddings.dtype, device=embeddings.device)
        covariance = covariance + regularization
        
        logger.debug(f"Computed statistics: mean shape {mean.shape}, cov shape {covariance.shape}")
        return mean, covariance
    
    def _build_hypervector(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Build hypervector representation from embeddings.
        
        Uses majority voting to create binary/ternary hypervector.
        
        Args:
            embeddings: Tensor of shape (n_samples, dim)
            
        Returns:
            Hypervector tensor of shape (dim,)
        """
        # Normalize embeddings to [-1, 1] range
        embeddings_norm = torch.tanh(embeddings)
        
        # Compute element-wise majority vote
        mean_values = torch.mean(embeddings_norm, dim=0)
        
        # Create ternary hypervector {-1, 0, 1}
        hypervector = torch.zeros_like(mean_values)
        hypervector[mean_values > 0.1] = 1.0
        hypervector[mean_values < -0.1] = -1.0
        # Values in [-0.1, 0.1] remain 0
        
        logger.debug(f"Built hypervector with {torch.sum(hypervector == 1)} positive, "
                    f"{torch.sum(hypervector == -1)} negative, {torch.sum(hypervector == 0)} zero elements")
        
        return hypervector
    
    def get_concept_vector(self, name: str) -> torch.Tensor:
        """
        Get the concept vector for a given concept name.
        
        Args:
            name: Concept name
            
        Returns:
            Concept vector tensor
            
        Raises:
            KeyError: If concept doesn't exist
        """
        if name not in self.concepts:
            raise KeyError(f"Concept '{name}' not found")
        
        return self.concepts[name]['vector']
    
    def get_concept_statistics(self, name: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get mean and covariance statistics for a concept (Gaussian method only).
        
        Args:
            name: Concept name
            
        Returns:
            Tuple of (mean, covariance) tensors, or (None, None) for hypervector method
            
        Raises:
            KeyError: If concept doesn't exist
        """
        if name not in self.concepts:
            raise KeyError(f"Concept '{name}' not found")
        
        concept_data = self.concepts[name]
        return concept_data['mean'], concept_data['covariance']
    
    def list_concepts(self) -> List[str]:
        """Return list of all concept names."""
        return list(self.concepts.keys())
    
    def remove_concept(self, name: str) -> bool:
        """
        Remove a concept from the library.
        
        Args:
            name: Concept name to remove
            
        Returns:
            True if concept was removed, False if not found
        """
        if name in self.concepts:
            del self.concepts[name]
            self.metadata['modified_timestamp'] = time.time()
            logger.info(f"Removed concept '{name}'")
            return True
        return False
    
    def save(self, path: str) -> None:
        """
        Save library to disk using torch.save for efficient tensor storage.
        
        Args:
            path: File path to save to
            
        Raises:
            ValueError: If save fails
        """
        try:
            save_data = {
                'dim': self.dim,
                'method': self.method,
                'concepts': self.concepts,
                'metadata': self.metadata
            }
            
            # Create parent directory if needed
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(save_data, path)
            logger.info(f"Saved ConceptLibrary to {path}")
            
        except Exception as e:
            raise ValueError(f"Failed to save library: {e}")
    
    def load(self, path: str) -> None:
        """
        Load library from disk.
        
        Args:
            path: File path to load from
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If load fails or data is invalid
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Library file not found: {path}")
        
        try:
            data = torch.load(path, map_location='cpu')
            
            # Validate loaded data
            required_keys = ['dim', 'method', 'concepts', 'metadata']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key: {key}")
            
            self.dim = data['dim']
            self.method = data['method']
            self.concepts = data['concepts']
            self.metadata = data['metadata']
            
            # Validate dimensions
            for name, concept_data in self.concepts.items():
                vector_shape = concept_data['vector'].shape
                if len(vector_shape) != 1 or vector_shape[0] != self.dim:
                    raise ValueError(f"Invalid vector shape for concept '{name}': {vector_shape}")
            
            logger.info(f"Loaded ConceptLibrary from {path} with {len(self.concepts)} concepts")
            
        except Exception as e:
            raise ValueError(f"Failed to load library: {e}")
    
    def _compute_concept_hash(self, vector: torch.Tensor) -> str:
        """Compute SHA256 hash of concept vector for integrity checking."""
        vector_bytes = vector.detach().cpu().numpy().tobytes()
        return hashlib.sha256(vector_bytes).hexdigest()[:16]
    
    def validate_integrity(self) -> Tuple[bool, List[str]]:
        """
        Validate the integrity of all concepts in the library.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for name, concept_data in self.concepts.items():
            try:
                # Check vector shape
                vector = concept_data['vector']
                if vector.shape != (self.dim,):
                    errors.append(f"Concept '{name}': invalid vector shape {vector.shape}")
                
                # Check hash integrity
                expected_hash = self._compute_concept_hash(vector)
                if concept_data['hash'] != expected_hash:
                    errors.append(f"Concept '{name}': hash mismatch")
                
                # Check method-specific data
                if self.method == 'gaussian':
                    if concept_data['mean'] is None or concept_data['covariance'] is None:
                        errors.append(f"Concept '{name}': missing Gaussian statistics")
                    elif concept_data['covariance'].shape != (self.dim, self.dim):
                        errors.append(f"Concept '{name}': invalid covariance shape")
                
                # Check required fields
                required_fields = ['n_samples', 'method', 'timestamp']
                for field in required_fields:
                    if field not in concept_data:
                        errors.append(f"Concept '{name}': missing field '{field}'")
                        
            except Exception as e:
                errors.append(f"Concept '{name}': validation error - {e}")
        
        return len(errors) == 0, errors
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the library.
        
        Returns:
            Dictionary with library summary information
        """
        if not self.concepts:
            return {
                'n_concepts': 0,
                'total_samples': 0,
                'dimension': self.dim,
                'method': self.method
            }
        
        total_samples = sum(cd['n_samples'] for cd in self.concepts.values())
        concept_names = list(self.concepts.keys())
        
        return {
            'n_concepts': len(self.concepts),
            'total_samples': total_samples,
            'dimension': self.dim,
            'method': self.method,
            'concept_names': concept_names,
            'created_timestamp': self.metadata['created_timestamp'],
            'modified_timestamp': self.metadata['modified_timestamp'],
            'avg_samples_per_concept': total_samples / len(self.concepts) if self.concepts else 0
        }
    
    def export_vectors_matrix(self, concept_names: Optional[List[str]] = None) -> Tuple[torch.Tensor, List[str]]:
        """
        Export concept vectors as a matrix for batch operations.
        
        Args:
            concept_names: Specific concepts to export, or None for all
            
        Returns:
            Tuple of (matrix, concept_names) where matrix is (n_concepts, dim)
        """
        if not self.concepts:
            return torch.empty(0, self.dim), []
        
        if concept_names is None:
            concept_names = list(self.concepts.keys())
        
        # Filter to existing concepts
        valid_names = [name for name in concept_names if name in self.concepts]
        
        if not valid_names:
            return torch.empty(0, self.dim), []
        
        # Stack vectors into matrix
        vectors = torch.stack([self.concepts[name]['vector'] for name in valid_names])
        
        return vectors, valid_names