"""
Test suite for semantic matching and scoring functionality.
Tests similarity computations, drift detection, and clustering.
"""

import pytest
import torch
import numpy as np
from typing import List

from pot.semantic.library import ConceptLibrary
from pot.semantic.match import (
    SemanticMatcher,
    compute_semantic_distance,
    semantic_similarity_score,
    batch_semantic_matching
)
from pot.semantic.types import (
    ConceptVector,
    SemanticLibrary,
    SemanticDistance,
    MatchingConfig
)


class TestSemanticMatcher:
    """Test SemanticMatcher class functionality."""
    
    @pytest.fixture
    def setup_library(self):
        """Create a test library with concepts."""
        library = ConceptLibrary(dim=128, method='gaussian')
        
        # Add several concepts with different distributions
        for i in range(3):
            # Create embeddings with different means
            mean = torch.ones(128) * i
            embeddings = torch.randn(20, 128) + mean
            library.add_concept(f'concept_{i}', embeddings)
        
        return library
    
    def test_initialization(self, setup_library):
        """Test matcher initialization."""
        matcher = SemanticMatcher(library=setup_library, threshold=0.8)
        
        assert matcher.library == setup_library
        assert matcher.threshold == 0.8
        assert len(matcher.similarity_methods) > 0
        
        # Test invalid library
        with pytest.raises(TypeError):
            SemanticMatcher(library="not_a_library")
    
    def test_compute_similarity_cosine(self, setup_library):
        """Test cosine similarity computation."""
        matcher = SemanticMatcher(library=setup_library)
        
        # Create test embedding
        embedding = torch.randn(128)
        
        # Compute similarity to each concept
        for i in range(3):
            similarity = matcher.compute_similarity(
                embedding, f'concept_{i}', method='cosine'
            )
            assert 0.0 <= similarity <= 1.0
        
        # Test with same vector
        concept_vec = setup_library.get_concept_vector('concept_0')
        similarity = matcher.compute_similarity(
            concept_vec, 'concept_0', method='cosine'
        )
        assert similarity > 0.9  # Should be very similar
        
        # Test non-existent concept
        with pytest.raises(KeyError):
            matcher.compute_similarity(embedding, 'nonexistent')
    
    def test_compute_similarity_euclidean(self, setup_library):
        """Test Euclidean similarity computation."""
        matcher = SemanticMatcher(library=setup_library)
        
        embedding = torch.randn(128)
        
        similarity = matcher.compute_similarity(
            embedding, 'concept_0', method='euclidean'
        )
        assert 0.0 <= similarity <= 1.0
        
        # Test with identical vector
        concept_vec = setup_library.get_concept_vector('concept_0')
        similarity = matcher.compute_similarity(
            concept_vec, 'concept_0', method='euclidean'
        )
        assert similarity > 0.8
    
    def test_compute_similarity_mahalanobis(self, setup_library):
        """Test Mahalanobis similarity computation."""
        matcher = SemanticMatcher(library=setup_library)
        
        embedding = torch.randn(128)
        
        # Should work with Gaussian library
        similarity = matcher.compute_similarity(
            embedding, 'concept_0', method='mahalanobis'
        )
        assert 0.0 <= similarity <= 1.0
    
    def test_compute_similarity_hamming(self):
        """Test Hamming similarity for hypervectors."""
        # Create hypervector library
        library = ConceptLibrary(dim=512, method='hypervector')
        
        # Add concept
        embeddings = torch.randn(10, 512)
        library.add_concept('hyper_concept', embeddings)
        
        matcher = SemanticMatcher(library=library)
        
        # Create test hypervector
        embedding = torch.sign(torch.randn(512))
        
        similarity = matcher.compute_similarity(
            embedding, 'hyper_concept', method='hamming'
        )
        assert 0.0 <= similarity <= 1.0
    
    def test_match_to_library(self, setup_library):
        """Test matching to all library concepts."""
        matcher = SemanticMatcher(library=setup_library)
        
        # Create embedding close to concept_1
        embedding = torch.ones(128) * 1.0 + torch.randn(128) * 0.1
        
        matches = matcher.match_to_library(embedding)
        
        assert len(matches) == 3
        assert all(0.0 <= v <= 1.0 for v in matches.values())
        
        # Should be sorted by similarity
        scores = list(matches.values())
        assert scores == sorted(scores, reverse=True)
        
        # concept_1 should have highest similarity
        assert list(matches.keys())[0] == 'concept_1'
    
    def test_compute_semantic_drift(self, setup_library):
        """Test semantic drift computation."""
        matcher = SemanticMatcher(library=setup_library)
        
        # Create embeddings that drift from concept_0
        reference_mean = torch.zeros(128)
        
        # No drift
        embeddings_no_drift = torch.randn(50, 128) + reference_mean
        drift_score = matcher.compute_semantic_drift(
            embeddings_no_drift, 'concept_0'
        )
        assert 0.0 <= drift_score <= 1.0
        assert drift_score < 0.5  # Low drift
        
        # High drift
        embeddings_drift = torch.randn(50, 128) + torch.ones(128) * 5
        drift_score = matcher.compute_semantic_drift(
            embeddings_drift, 'concept_0'
        )
        assert drift_score > 0.3  # Higher drift
        
        # Test with non-existent concept
        with pytest.raises(KeyError):
            matcher.compute_semantic_drift(embeddings_no_drift, 'nonexistent')
    
    def test_cluster_outputs(self, setup_library):
        """Test output clustering."""
        matcher = SemanticMatcher(library=setup_library)
        
        # Create embeddings with clear clusters
        embeddings = []
        for i in range(3):
            cluster_embeddings = torch.randn(10, 128) + torch.ones(128) * i * 3
            embeddings.extend(cluster_embeddings)
        
        # Cluster
        labels = matcher.cluster_outputs(embeddings, n_clusters=3)
        
        assert len(labels) == 30
        assert len(np.unique(labels)) == 3
        
        # Test auto cluster detection
        labels_auto = matcher.cluster_outputs(embeddings, n_clusters=None)
        assert len(labels_auto) == 30
        
        # Test with empty list
        labels_empty = matcher.cluster_outputs([])
        assert len(labels_empty) == 0
    
    def test_get_best_match(self, setup_library):
        """Test finding best matching concept."""
        matcher = SemanticMatcher(library=setup_library, threshold=0.7)
        
        # Embedding close to concept_2
        embedding = torch.ones(128) * 2.0 + torch.randn(128) * 0.1
        
        concept_name, score = matcher.get_best_match(embedding)
        
        assert concept_name == 'concept_2'
        assert score > 0.7
        
        # Embedding far from all concepts
        far_embedding = torch.ones(128) * 100
        concept_name, score = matcher.get_best_match(
            far_embedding, min_similarity=0.9
        )
        
        assert concept_name is None
        assert score < 0.9
    
    def test_batch_match(self, setup_library):
        """Test batch matching."""
        matcher = SemanticMatcher(library=setup_library)
        
        # Create batch of embeddings
        embeddings = torch.stack([
            torch.ones(128) * 0 + torch.randn(128) * 0.1,
            torch.ones(128) * 1 + torch.randn(128) * 0.1,
            torch.ones(128) * 2 + torch.randn(128) * 0.1,
        ])
        
        # Without scores
        matches = matcher.batch_match(embeddings, return_scores=False)
        assert len(matches) == 3
        assert matches[0] == 'concept_0'
        assert matches[1] == 'concept_1'
        assert matches[2] == 'concept_2'
        
        # With scores
        matches_with_scores = matcher.batch_match(embeddings, return_scores=True)
        assert len(matches_with_scores) == 3
        assert all(isinstance(m, tuple) for m in matches_with_scores)
        assert all(m[1] > 0.5 for m in matches_with_scores)
    
    def test_compute_coverage(self, setup_library):
        """Test coverage computation."""
        matcher = SemanticMatcher(library=setup_library, threshold=0.7)
        
        # Embeddings matching library concepts
        good_embeddings = torch.stack([
            torch.ones(128) * i + torch.randn(128) * 0.1
            for i in range(3)
        ])
        
        coverage = matcher.compute_coverage(good_embeddings)
        assert coverage == 1.0  # All match
        
        # Embeddings not matching library
        bad_embeddings = torch.randn(5, 128) * 10
        coverage = matcher.compute_coverage(bad_embeddings)
        assert coverage < 0.5  # Most don't match
    
    def test_similarity_caching(self, setup_library):
        """Test that similarity computations are cached."""
        matcher = SemanticMatcher(library=setup_library)
        
        embedding = torch.randn(128)
        
        # First computation
        _ = matcher.compute_similarity(embedding, 'concept_0')
        cache_size_1 = len(matcher._similarity_cache)
        
        # Same computation (should use cache)
        _ = matcher.compute_similarity(embedding, 'concept_0')
        cache_size_2 = len(matcher._similarity_cache)
        
        assert cache_size_2 == cache_size_1  # Cache size unchanged
        
        # Different computation
        _ = matcher.compute_similarity(embedding, 'concept_1')
        cache_size_3 = len(matcher._similarity_cache)
        
        assert cache_size_3 > cache_size_1  # Cache grew


class TestSemanticDistanceFunctions:
    """Test standalone semantic distance functions."""
    
    def test_compute_semantic_distance(self):
        """Test semantic distance computation."""
        vec1 = np.random.randn(64)
        vec2 = np.random.randn(64)
        
        concept1 = ConceptVector(vector=vec1, concept_id='c1')
        concept2 = ConceptVector(vector=vec2, concept_id='c2')
        
        # Cosine distance
        dist_cosine = compute_semantic_distance(
            concept1, concept2, 
            metric=SemanticDistance.COSINE
        )
        assert dist_cosine >= 0.0
        
        # Euclidean distance
        dist_euclidean = compute_semantic_distance(
            concept1, concept2,
            metric=SemanticDistance.EUCLIDEAN
        )
        assert dist_euclidean >= 0.0
        
        # Manhattan distance
        dist_manhattan = compute_semantic_distance(
            concept1, concept2,
            metric=SemanticDistance.MANHATTAN
        )
        assert dist_manhattan >= 0.0
        
        # Same vector should have 0 distance
        concept_same = ConceptVector(vector=vec1, concept_id='c1_copy')
        dist_same = compute_semantic_distance(
            concept1, concept_same,
            metric=SemanticDistance.COSINE
        )
        assert dist_same < 0.01
    
    def test_semantic_similarity_score(self):
        """Test similarity score computation."""
        vec1 = np.random.randn(64)
        vec2 = np.random.randn(64)
        
        concept1 = ConceptVector(vector=vec1, concept_id='c1')
        concept2 = ConceptVector(vector=vec2, concept_id='c2')
        
        # Similarity should be in [0, 1]
        similarity = semantic_similarity_score(
            concept1, concept2,
            metric=SemanticDistance.COSINE
        )
        assert 0.0 <= similarity <= 1.0
        
        # Same vector should have similarity ~1
        concept_same = ConceptVector(vector=vec1, concept_id='c1_copy')
        similarity_same = semantic_similarity_score(
            concept1, concept_same,
            metric=SemanticDistance.COSINE
        )
        assert similarity_same > 0.99
        
        # Test different metrics
        sim_euclidean = semantic_similarity_score(
            concept1, concept2,
            metric=SemanticDistance.EUCLIDEAN
        )
        assert 0.0 <= sim_euclidean <= 1.0
    
    def test_batch_semantic_matching(self):
        """Test batch matching functionality."""
        # Create library
        library = SemanticLibrary()
        for i in range(3):
            vec = np.random.randn(64) + np.ones(64) * i
            concept = ConceptVector(
                vector=vec,
                concept_id=f'lib_concept_{i}'
            )
            library.add_concept(concept)
        
        # Create query concepts
        query_concepts = []
        for i in range(2):
            vec = np.random.randn(64) + np.ones(64) * i
            concept = ConceptVector(
                vector=vec,
                concept_id=f'query_{i}'
            )
            query_concepts.append(concept)
        
        # Perform batch matching
        config = MatchingConfig(
            similarity_threshold=0.3,
            distance_metric=SemanticDistance.COSINE
        )
        
        results = batch_semantic_matching(
            query_concepts, library, config, top_k=2
        )
        
        assert len(results) == 2
        assert 'query_0' in results
        assert 'query_1' in results
        
        # Check results structure
        for query_id, matches in results.items():
            assert len(matches) <= 2  # top_k=2
            for match in matches:
                assert match.query_concept_id == query_id
                assert match.similarity_score >= 0.3  # threshold


class TestSemanticDrift:
    """Test semantic drift detection capabilities."""
    
    def test_drift_detection_gaussian(self):
        """Test drift detection with Gaussian concepts."""
        library = ConceptLibrary(dim=64, method='gaussian')
        
        # Add reference concept
        reference_embeddings = torch.randn(50, 64)
        library.add_concept('reference', reference_embeddings)
        
        matcher = SemanticMatcher(library=library)
        
        # Test no drift
        no_drift_embeddings = torch.randn(30, 64)
        drift_score = matcher.compute_semantic_drift(
            no_drift_embeddings, 'reference'
        )
        assert drift_score < 0.5
        
        # Test with drift
        drift_embeddings = torch.randn(30, 64) + torch.ones(64) * 3
        drift_score = matcher.compute_semantic_drift(
            drift_embeddings, 'reference'
        )
        assert drift_score > 0.3
    
    def test_drift_detection_hypervector(self):
        """Test drift detection with hypervector concepts."""
        library = ConceptLibrary(dim=512, method='hypervector')
        
        # Add reference concept
        reference_embeddings = torch.randn(20, 512)
        library.add_concept('reference', reference_embeddings)
        
        matcher = SemanticMatcher(library=library)
        
        # Test drift computation
        test_embeddings = torch.sign(torch.randn(15, 512))
        drift_score = matcher.compute_semantic_drift(
            test_embeddings, 'reference'
        )
        
        assert 0.0 <= drift_score <= 1.0
    
    def test_drift_with_single_sample(self):
        """Test drift computation with single sample."""
        library = ConceptLibrary(dim=32, method='gaussian')
        
        embeddings = torch.randn(10, 32)
        library.add_concept('reference', embeddings)
        
        matcher = SemanticMatcher(library=library)
        
        # Single sample
        single_embedding = torch.randn(32)
        drift_score = matcher.compute_semantic_drift(
            single_embedding, 'reference'
        )
        
        assert 0.0 <= drift_score <= 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_library(self):
        """Test with empty library."""
        library = ConceptLibrary(dim=64, method='gaussian')
        matcher = SemanticMatcher(library=library)
        
        embedding = torch.randn(64)
        matches = matcher.match_to_library(embedding)
        
        assert len(matches) == 0
    
    def test_dimension_mismatch(self):
        """Test dimension mismatch handling."""
        library = ConceptLibrary(dim=64, method='gaussian')
        embeddings = torch.randn(10, 64)
        library.add_concept('test', embeddings)
        
        matcher = SemanticMatcher(library=library)
        
        # Wrong dimension
        wrong_dim_embedding = torch.randn(32)
        
        with pytest.raises(ValueError):
            matcher.compute_similarity(wrong_dim_embedding, 'test')
    
    def test_zero_vectors(self):
        """Test handling of zero vectors."""
        library = ConceptLibrary(dim=64, method='gaussian')
        
        # Add concept with zero mean
        zero_embeddings = torch.zeros(10, 64)
        library.add_concept('zero_concept', zero_embeddings)
        
        matcher = SemanticMatcher(library=library)
        
        # Compare with zero vector
        zero_vec = torch.zeros(64)
        similarity = matcher.compute_similarity(
            zero_vec, 'zero_concept', method='cosine'
        )
        
        # Should handle gracefully
        assert 0.0 <= similarity <= 1.0
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        library = ConceptLibrary(dim=32, method='gaussian')
        embeddings = torch.randn(10, 32)
        library.add_concept('test', embeddings)
        
        matcher = SemanticMatcher(library=library)
        
        # Embedding with NaN
        nan_embedding = torch.randn(32)
        nan_embedding[0] = float('nan')
        
        # Should handle gracefully (might return 0 or raise)
        try:
            similarity = matcher.compute_similarity(nan_embedding, 'test')
            assert 0.0 <= similarity <= 1.0 or np.isnan(similarity)
        except (ValueError, RuntimeError):
            pass  # Also acceptable to raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])