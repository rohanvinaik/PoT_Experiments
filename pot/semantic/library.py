"""
Concept vector library management for semantic verification.
Provides classes and functions for storing, retrieving, and managing concept vectors.
"""

import os
import json
import pickle
import time
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
from pathlib import Path
import logging

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
        if path.suffix == '.pkl':
            format = 'pickle'
        elif path.suffix == '.json':
            format = 'json'
        else:
            raise ValueError(f"Cannot auto-detect format for file: {path}")
    
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
                        format: str = 'pickle') -> None:
    """
    Save a concept library to disk.
    
    Args:
        library: SemanticLibrary to save
        path: Path to save the library
        format: Save format ('pickle' or 'json')
        
    Raises:
        ValueError: If format is unsupported or library is invalid
    """
    path = Path(path)
    
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