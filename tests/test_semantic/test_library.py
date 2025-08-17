"""
Test suite for concept library functionality.
Tests concept vector creation, storage, retrieval, and management.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

from pot.semantic.library import (
    ConceptLibrary,
    ConceptVectorManager,
    load_concept_library,
    save_concept_library
)
from pot.semantic.types import ConceptVector, SemanticLibrary


class TestConceptLibrary:
    """Test ConceptLibrary class functionality."""
    
    def test_initialization(self):
        """Test library initialization with different methods."""
        # Gaussian method
        lib_gaussian = ConceptLibrary(dim=128, method='gaussian')
        assert lib_gaussian.dim == 128
        assert lib_gaussian.method == 'gaussian'
        assert len(lib_gaussian.concepts) == 0
        
        # Hypervector method
        lib_hyper = ConceptLibrary(dim=256, method='hypervector')
        assert lib_hyper.dim == 256
        assert lib_hyper.method == 'hypervector'
        
        # Invalid dimension
        with pytest.raises(ValueError):
            ConceptLibrary(dim=-1, method='gaussian')
        
        # Invalid method
        with pytest.raises(ValueError):
            ConceptLibrary(dim=128, method='invalid')
    
    def test_add_concept_gaussian(self):
        """Test adding concepts with Gaussian method."""
        library = ConceptLibrary(dim=64, method='gaussian')
        
        # Create embeddings
        embeddings = torch.randn(20, 64)
        
        # Add concept
        library.add_concept('test_concept', embeddings)
        
        assert 'test_concept' in library.concepts
        assert library.concepts['test_concept']['n_samples'] == 20
        assert library.concepts['test_concept']['method'] == 'gaussian'
        
        # Check statistics were computed
        mean, cov = library.get_concept_statistics('test_concept')
        assert mean is not None
        assert mean.shape == (64,)
        assert cov is not None
        assert cov.shape == (64, 64)
        
        # Test duplicate concept
        with pytest.raises(ValueError):
            library.add_concept('test_concept', embeddings)
    
    def test_add_concept_hypervector(self):
        """Test adding concepts with hypervector method."""
        library = ConceptLibrary(dim=512, method='hypervector')
        
        # Create embeddings
        embeddings = torch.randn(30, 512)
        
        # Add concept
        library.add_concept('hyper_concept', embeddings)
        
        assert 'hyper_concept' in library.concepts
        vector = library.get_concept_vector('hyper_concept')
        
        # Check hypervector properties (ternary values)
        unique_values = torch.unique(vector)
        assert all(v in [-1.0, 0.0, 1.0] for v in unique_values)
    
    def test_compute_statistics(self):
        """Test statistical computation."""
        library = ConceptLibrary(dim=32, method='gaussian')
        
        # Create known distribution
        mean_true = torch.ones(32) * 2.0
        embeddings = torch.randn(100, 32) + mean_true
        
        mean, cov = library.compute_statistics(embeddings)
        
        # Check mean is close to true mean
        assert torch.allclose(mean, mean_true, atol=0.5)
        
        # Check covariance is positive definite
        eigenvalues = torch.linalg.eigvalsh(cov)
        assert torch.all(eigenvalues > 0)
        
        # Test with insufficient samples
        with pytest.raises(ValueError):
            library.compute_statistics(torch.randn(1, 32))
    
    def test_get_concept_vector(self):
        """Test retrieving concept vectors."""
        library = ConceptLibrary(dim=64, method='gaussian')
        
        embeddings = torch.randn(10, 64)
        library.add_concept('concept1', embeddings)
        
        # Retrieve existing concept
        vector = library.get_concept_vector('concept1')
        assert vector.shape == (64,)
        
        # Try to retrieve non-existent concept
        with pytest.raises(KeyError):
            library.get_concept_vector('nonexistent')
    
    def test_list_and_remove_concepts(self):
        """Test listing and removing concepts."""
        library = ConceptLibrary(dim=32, method='gaussian')
        
        # Add multiple concepts
        for i in range(3):
            embeddings = torch.randn(10, 32)
            library.add_concept(f'concept_{i}', embeddings)
        
        # List concepts
        concepts = library.list_concepts()
        assert len(concepts) == 3
        assert 'concept_0' in concepts
        
        # Remove concept
        removed = library.remove_concept('concept_1')
        assert removed
        assert len(library.list_concepts()) == 2
        
        # Remove non-existent
        removed = library.remove_concept('nonexistent')
        assert not removed
    
    def test_save_and_load(self):
        """Test saving and loading library."""
        library = ConceptLibrary(dim=64, method='gaussian')
        
        # Add concepts
        for i in range(2):
            embeddings = torch.randn(15, 64)
            library.add_concept(f'concept_{i}', embeddings)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save
            library.save(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Create new library and load
            library2 = ConceptLibrary(dim=32, method='hypervector')
            library2.load(tmp_path)
            
            # Check loaded correctly
            assert library2.dim == 64
            assert library2.method == 'gaussian'
            assert len(library2.concepts) == 2
            assert 'concept_0' in library2.concepts
            
        finally:
            os.unlink(tmp_path)
    
    def test_validate_integrity(self):
        """Test integrity validation."""
        library = ConceptLibrary(dim=64, method='gaussian')
        
        embeddings = torch.randn(10, 64)
        library.add_concept('test', embeddings)
        
        # Should be valid
        is_valid, errors = library.validate_integrity()
        assert is_valid
        assert len(errors) == 0
        
        # Corrupt the hash
        library.concepts['test']['hash'] = 'invalid_hash'
        is_valid, errors = library.validate_integrity()
        assert not is_valid
        assert len(errors) > 0
    
    def test_export_vectors_matrix(self):
        """Test exporting vectors as matrix."""
        library = ConceptLibrary(dim=32, method='gaussian')
        
        # Add concepts
        for i in range(3):
            embeddings = torch.randn(10, 32)
            library.add_concept(f'concept_{i}', embeddings)
        
        # Export all
        matrix, names = library.export_vectors_matrix()
        assert matrix.shape == (3, 32)
        assert len(names) == 3
        
        # Export specific
        matrix, names = library.export_vectors_matrix(['concept_0', 'concept_2'])
        assert matrix.shape == (2, 32)
        assert names == ['concept_0', 'concept_2']
        
        # Export non-existent
        matrix, names = library.export_vectors_matrix(['nonexistent'])
        assert matrix.shape == (0, 32)
        assert len(names) == 0


class TestConceptVectorManager:
    """Test ConceptVectorManager functionality."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = ConceptVectorManager()
        assert manager.library is not None
        assert not manager._dirty
        
        # With existing library
        library = SemanticLibrary()
        manager = ConceptVectorManager(library)
        assert manager.library == library
    
    def test_add_concept(self):
        """Test adding concepts through manager."""
        manager = ConceptVectorManager()
        
        vector = np.random.randn(128)
        concept = manager.add_concept(
            vector=vector,
            concept_id='test_id',
            label='Test Concept',
            source='test',
            metadata={'key': 'value'}
        )
        
        assert concept.concept_id == 'test_id'
        assert concept.label == 'Test Concept'
        assert concept.source == 'test'
        assert manager._dirty
        
        # Duplicate ID
        with pytest.raises(ValueError):
            manager.add_concept(vector, 'test_id')
    
    def test_remove_concept(self):
        """Test removing concepts."""
        manager = ConceptVectorManager()
        
        vector = np.random.randn(64)
        manager.add_concept(vector, 'test_id')
        
        # Remove existing
        removed = manager.remove_concept('test_id')
        assert removed
        assert 'test_id' not in manager.library
        
        # Remove non-existent
        removed = manager.remove_concept('nonexistent')
        assert not removed
    
    def test_update_concept(self):
        """Test updating concept metadata."""
        manager = ConceptVectorManager()
        
        vector = np.random.randn(64)
        manager.add_concept(vector, 'test_id', label='Original')
        
        # Update label
        updated = manager.update_concept('test_id', label='Updated')
        assert updated
        
        concept = manager.get_concept('test_id')
        assert concept.label == 'Updated'
        
        # Update non-existent
        updated = manager.update_concept('nonexistent', label='Test')
        assert not updated
    
    def test_list_concepts_with_filters(self):
        """Test listing concepts with filters."""
        manager = ConceptVectorManager()
        
        # Add concepts with different sources and labels
        for i in range(5):
            source = 'source_a' if i % 2 == 0 else 'source_b'
            label = f'concept_{i}' if i < 3 else f'special_{i}'
            manager.add_concept(
                np.random.randn(32),
                f'id_{i}',
                label=label,
                source=source
            )
        
        # Filter by source
        concepts = manager.list_concepts(source_filter='source_a')
        assert len(concepts) == 3
        
        # Filter by label pattern
        concepts = manager.list_concepts(label_pattern='special')
        assert len(concepts) == 2
        
        # Combined filters
        concepts = manager.list_concepts(
            source_filter='source_b',
            label_pattern='concept'
        )
        assert len(concepts) == 1
    
    def test_get_concept_matrix(self):
        """Test getting concept matrix."""
        manager = ConceptVectorManager()
        
        # Add concepts
        dim = 64
        for i in range(3):
            manager.add_concept(np.random.randn(dim), f'id_{i}')
        
        # Get all
        matrix = manager.get_concept_matrix()
        assert matrix.shape == (3, dim)
        
        # Get specific
        matrix = manager.get_concept_matrix(['id_0', 'id_2'])
        assert matrix.shape == (2, dim)
        
        # Empty library
        manager2 = ConceptVectorManager()
        with pytest.raises(ValueError):
            manager2.get_concept_matrix()
    
    def test_validate_library(self):
        """Test library validation."""
        manager = ConceptVectorManager()
        
        # Empty library should be valid
        is_valid, errors = manager.validate_library()
        assert is_valid
        
        # Add valid concepts
        for i in range(2):
            manager.add_concept(np.random.randn(32), f'id_{i}')
        
        is_valid, errors = manager.validate_library()
        assert is_valid
    
    def test_get_statistics(self):
        """Test getting library statistics."""
        manager = ConceptVectorManager()
        
        # Empty library
        stats = manager.get_statistics()
        assert stats['total_concepts'] == 0
        
        # Add concepts
        for i in range(3):
            manager.add_concept(
                np.random.randn(64),
                f'id_{i}',
                label=f'label_{i}' if i < 2 else None,
                source='test'
            )
        
        stats = manager.get_statistics()
        assert stats['total_concepts'] == 3
        assert stats['dimension'] == 64
        assert stats['sources']['test'] == 3
        assert stats['has_labels'] == 2


class TestSemanticLibraryPersistence:
    """Test saving and loading semantic libraries."""
    
    def test_save_load_pickle(self):
        """Test pickle format persistence."""
        # Create library
        library = SemanticLibrary()
        
        # Add concepts
        for i in range(2):
            vector = np.random.randn(128)
            concept = ConceptVector(
                vector=vector,
                concept_id=f'concept_{i}',
                label=f'Concept {i}',
                source='test'
            )
            library.add_concept(concept)
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            save_concept_library(library, tmp_path, format='pickle')
            
            # Load
            loaded = load_concept_library(tmp_path, format='pickle')
            
            assert len(loaded.concepts) == 2
            assert 'concept_0' in loaded.concepts
            assert loaded.concepts['concept_0'].label == 'Concept 0'
            
        finally:
            os.unlink(tmp_path)
    
    def test_save_load_json(self):
        """Test JSON format persistence."""
        library = SemanticLibrary()
        
        # Add concept
        vector = np.random.randn(64)
        concept = ConceptVector(
            vector=vector,
            concept_id='test_concept',
            label='Test',
            metadata={'key': 'value'}
        )
        library.add_concept(concept)
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            save_concept_library(library, tmp_path, format='json')
            
            # Load
            loaded = load_concept_library(tmp_path, format='json')
            
            assert 'test_concept' in loaded.concepts
            assert loaded.concepts['test_concept'].metadata['key'] == 'value'
            
        finally:
            os.unlink(tmp_path)
    
    def test_auto_format_detection(self):
        """Test automatic format detection."""
        library = SemanticLibrary()
        concept = ConceptVector(
            vector=np.random.randn(32),
            concept_id='test'
        )
        library.add_concept(concept)
        
        # Test pickle format
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            pkl_path = tmp.name
        
        # Test JSON format  
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            json_path = tmp.name
        
        try:
            # Save with auto-detect
            save_concept_library(library, pkl_path)
            save_concept_library(library, json_path)
            
            # Load with auto-detect
            loaded_pkl = load_concept_library(pkl_path)
            loaded_json = load_concept_library(json_path)
            
            assert len(loaded_pkl.concepts) == 1
            assert len(loaded_json.concepts) == 1
            
        finally:
            os.unlink(pkl_path)
            os.unlink(json_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])