"""
Semantic matching and scoring utilities for concept vector comparison.
Implements various distance metrics and similarity scoring functions with ConceptLibrary integration.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from scipy.spatial.distance import cosine, euclidean, cityblock
from scipy.stats import entropy
from sklearn.cluster import KMeans, AgglomerativeClustering
import logging
from functools import lru_cache
import time
import warnings

from .types import ConceptVector, SemanticMatchResult, MatchingConfig, SemanticDistance, SemanticLibrary
from .utils import normalize_concept_vector
from .library import ConceptLibrary

logger = logging.getLogger(__name__)


class SemanticMatcher:
    """
    Enhanced semantic matcher for scoring outputs against concept library.
    Supports multiple similarity measures and semantic drift analysis.
    """
    
    def __init__(self, library: ConceptLibrary, threshold: float = 0.8):
        """
        Initialize the semantic matcher with a concept library.
        
        Args:
            library: ConceptLibrary containing reference concepts
            threshold: Similarity threshold for positive matches
        """
        if not isinstance(library, ConceptLibrary):
            raise TypeError("library must be a ConceptLibrary instance")
        
        self.library = library
        self.threshold = threshold
        self._similarity_cache = {}
        
        # Setup similarity methods based on library type
        self.similarity_methods = {
            'cosine': self._cosine_similarity,
            'euclidean': self._euclidean_similarity,
            'mahalanobis': self._mahalanobis_similarity,
            'hamming': self._hamming_similarity
        }
        
        logger.info(f"Initialized SemanticMatcher with {len(library.list_concepts())} concepts, threshold={threshold}")
    
    def compute_similarity(self, embedding: torch.Tensor, concept: str, 
                         method: str = 'cosine') -> float:
        """
        Compute similarity between an embedding and a concept from the library.
        
        Args:
            embedding: Input embedding tensor
            concept: Concept name in the library
            method: Similarity method ('cosine', 'euclidean', 'mahalanobis', 'hamming')
            
        Returns:
            Similarity score in [0, 1], higher is more similar
            
        Raises:
            KeyError: If concept not found
            ValueError: If method unsupported or dimension mismatch
        """
        if concept not in self.library.list_concepts():
            raise KeyError(f"Concept '{concept}' not found in library")
        
        if method not in self.similarity_methods:
            raise ValueError(f"Unsupported similarity method: {method}")
        
        # Validate embedding dimensions
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)  # Add batch dimension
        
        if embedding.shape[-1] != self.library.dim:
            raise ValueError(f"Embedding dimension {embedding.shape[-1]} doesn't match library dimension {self.library.dim}")
        
        # Check cache
        cache_key = (embedding.data_ptr(), concept, method)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        # Get concept vector
        concept_vector = self.library.get_concept_vector(concept)
        
        # Compute similarity
        try:
            if method == 'mahalanobis' and self.library.method == 'gaussian':
                # Mahalanobis requires covariance matrix
                _, covariance = self.library.get_concept_statistics(concept)
                if covariance is None:
                    logger.warning(f"No covariance for concept '{concept}', falling back to cosine")
                    similarity = self._cosine_similarity(embedding.squeeze(), concept_vector)
                else:
                    similarity = self._mahalanobis_similarity(embedding.squeeze(), concept_vector, covariance)
            elif method == 'hamming' and self.library.method == 'hypervector':
                # Hamming distance for hypervectors
                similarity = self._hamming_similarity(embedding.squeeze(), concept_vector)
            else:
                # Standard similarity methods
                similarity_func = self.similarity_methods[method]
                if method == 'mahalanobis':
                    # For non-Gaussian libraries, fall back to cosine for Mahalanobis
                    logger.debug(f"Library method is '{self.library.method}', using cosine instead of mahalanobis")
                    similarity = self._cosine_similarity(embedding.squeeze(), concept_vector)
                else:
                    similarity = similarity_func(embedding.squeeze(), concept_vector)
            
            # Cache result
            self._similarity_cache[cache_key] = similarity
            
            # Clear cache if too large
            if len(self._similarity_cache) > 10000:
                self._similarity_cache.clear()
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error computing {method} similarity: {e}")
            raise
    
    def match_to_library(self, embedding: torch.Tensor) -> Dict[str, float]:
        """
        Match an embedding to all concepts in the library.
        
        Args:
            embedding: Input embedding tensor
            
        Returns:
            Dictionary mapping concept names to similarity scores
        """
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        
        if embedding.shape[-1] != self.library.dim:
            raise ValueError(f"Embedding dimension {embedding.shape[-1]} doesn't match library dimension {self.library.dim}")
        
        results = {}
        
        # Determine best method based on library type
        if self.library.method == 'gaussian':
            primary_method = 'mahalanobis'
            fallback_method = 'cosine'
        elif self.library.method == 'hypervector':
            primary_method = 'hamming'
            fallback_method = 'cosine'
        else:
            primary_method = 'cosine'
            fallback_method = 'euclidean'
        
        for concept_name in self.library.list_concepts():
            try:
                # Try primary method first
                similarity = self.compute_similarity(embedding, concept_name, primary_method)
            except Exception as e:
                logger.debug(f"Primary method {primary_method} failed for {concept_name}: {e}, trying {fallback_method}")
                try:
                    similarity = self.compute_similarity(embedding, concept_name, fallback_method)
                except Exception as e2:
                    logger.warning(f"Both methods failed for concept {concept_name}: {e2}")
                    similarity = 0.0
            
            results[concept_name] = similarity
        
        # Sort by similarity (descending)
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return results
    
    def compute_semantic_drift(self, embeddings: torch.Tensor, 
                              reference_concept: str) -> float:
        """
        Compute semantic drift between embeddings and a reference concept.
        
        Drift is measured as the average distance from the reference concept,
        normalized to [0, 1] where 0 means no drift and 1 means maximum drift.
        
        Args:
            embeddings: Batch of embeddings (n_samples, dim)
            reference_concept: Reference concept name
            
        Returns:
            Semantic drift score in [0, 1]
            
        Raises:
            KeyError: If reference concept not found
        """
        if reference_concept not in self.library.list_concepts():
            raise KeyError(f"Reference concept '{reference_concept}' not found")
        
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        
        if embeddings.shape[-1] != self.library.dim:
            raise ValueError(f"Embedding dimension {embeddings.shape[-1]} doesn't match library dimension {self.library.dim}")
        
        # Get reference concept vector
        reference_vector = self.library.get_concept_vector(reference_concept)
        
        # Compute distances for each embedding
        distances = []
        for i in range(embeddings.shape[0]):
            # Use cosine distance as default metric for drift
            similarity = self._cosine_similarity(embeddings[i], reference_vector)
            distance = 1.0 - similarity  # Convert similarity to distance
            distances.append(distance)
        
        # Calculate drift metrics
        mean_distance = np.mean(distances)
        
        # For Gaussian concepts, also consider variance change
        if self.library.method == 'gaussian':
            reference_mean, reference_cov = self.library.get_concept_statistics(reference_concept)
            
            if reference_mean is not None and reference_cov is not None:
                # Compute empirical mean and covariance of embeddings
                embeddings_mean = torch.mean(embeddings, dim=0)
                if embeddings.shape[0] > 1:
                    centered = embeddings - embeddings_mean
                    embeddings_cov = torch.mm(centered.t(), centered) / (embeddings.shape[0] - 1)
                    
                    # Measure drift in mean
                    mean_drift = torch.norm(embeddings_mean - reference_mean).item()
                    
                    # Measure drift in covariance (Frobenius norm)
                    cov_drift = torch.norm(embeddings_cov - reference_cov, p='fro').item()
                    
                    # Normalize drifts
                    mean_drift_norm = 1.0 - np.exp(-mean_drift / self.library.dim)
                    cov_drift_norm = 1.0 - np.exp(-cov_drift / (self.library.dim ** 2))
                    
                    # Combine distance and statistical drift
                    semantic_drift = 0.5 * mean_distance + 0.25 * mean_drift_norm + 0.25 * cov_drift_norm
                else:
                    # Only one sample, just use distance
                    semantic_drift = mean_distance
            else:
                semantic_drift = mean_distance
        else:
            # For non-Gaussian methods, use distance-based drift
            semantic_drift = mean_distance
        
        # Ensure drift is in [0, 1]
        semantic_drift = np.clip(semantic_drift, 0.0, 1.0)
        
        logger.debug(f"Semantic drift from '{reference_concept}': {semantic_drift:.3f}")
        return float(semantic_drift)
    
    def cluster_outputs(self, embeddings: List[torch.Tensor], 
                       n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Cluster output embeddings to identify semantic groups.
        
        Args:
            embeddings: List of embedding tensors
            n_clusters: Number of clusters (auto-determined if None)
            
        Returns:
            Cluster labels for each embedding
        """
        if not embeddings:
            return np.array([])
        
        # Convert to matrix
        if isinstance(embeddings[0], torch.Tensor):
            embeddings_matrix = torch.stack([e.squeeze() for e in embeddings])
            embeddings_matrix = embeddings_matrix.detach().cpu().numpy()
        else:
            embeddings_matrix = np.vstack([e.squeeze() for e in embeddings])
        
        n_samples = len(embeddings_matrix)
        
        # Auto-determine number of clusters if not specified
        if n_clusters is None:
            # Use elbow method heuristic: sqrt(n/2)
            n_clusters = max(2, min(int(np.sqrt(n_samples / 2)), 10))
            logger.debug(f"Auto-determined n_clusters={n_clusters} for {n_samples} samples")
        
        # Ensure valid number of clusters
        n_clusters = min(n_clusters, n_samples)
        
        if n_clusters < 2:
            # All in one cluster
            return np.zeros(n_samples, dtype=int)
        
        try:
            # Use KMeans for clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_matrix)
            
            # Compute cluster quality metrics
            inertia = kmeans.inertia_
            logger.info(f"Clustered {n_samples} embeddings into {n_clusters} clusters (inertia={inertia:.3f})")
            
            return labels
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            # Fall back to single cluster
            return np.zeros(n_samples, dtype=int)
    
    # Similarity computation methods
    
    def _cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Returns similarity in [0, 1] where 1 is identical.
        """
        # Handle edge cases
        if torch.allclose(vec1, torch.zeros_like(vec1)) or torch.allclose(vec2, torch.zeros_like(vec2)):
            logger.warning("Computing cosine similarity with zero vector")
            return 0.0
        
        # Normalize vectors
        vec1_norm = vec1 / (torch.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (torch.norm(vec2) + 1e-8)
        
        # Compute cosine similarity
        similarity = torch.dot(vec1_norm, vec2_norm).item()
        
        # Map from [-1, 1] to [0, 1]
        similarity = (similarity + 1.0) / 2.0
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def _euclidean_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute normalized Euclidean similarity between two vectors.
        
        Returns similarity in [0, 1] where 1 is identical.
        """
        # Compute Euclidean distance
        distance = torch.norm(vec1 - vec2).item()
        
        # Normalize by maximum possible distance (assuming unit vectors)
        max_distance = 2.0 * np.sqrt(self.library.dim)
        
        # Convert to similarity
        similarity = 1.0 - (distance / max_distance)
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def _mahalanobis_similarity(self, vec: torch.Tensor, mean: torch.Tensor, 
                               covariance: torch.Tensor) -> float:
        """
        Compute Mahalanobis similarity using stored covariance.
        
        Returns similarity in [0, 1] where 1 is at the mean.
        """
        try:
            # Center the vector
            diff = vec - mean
            
            # Add regularization for numerical stability
            cov_reg = covariance + 1e-6 * torch.eye(covariance.shape[0], 
                                                    dtype=covariance.dtype, 
                                                    device=covariance.device)
            
            # Compute inverse covariance
            try:
                cov_inv = torch.linalg.inv(cov_reg)
            except torch.linalg.LinAlgError:
                # If inversion fails, use pseudo-inverse
                logger.warning("Covariance matrix inversion failed, using pseudo-inverse")
                cov_inv = torch.linalg.pinv(cov_reg)
            
            # Compute Mahalanobis distance
            mahal_dist_sq = torch.dot(diff, torch.mv(cov_inv, diff)).item()
            
            # Ensure non-negative
            mahal_dist_sq = max(0.0, mahal_dist_sq)
            mahal_dist = np.sqrt(mahal_dist_sq)
            
            # Convert to similarity using exponential decay
            # This maps distance to [0, 1] where 0 distance = 1 similarity
            similarity = np.exp(-0.5 * mahal_dist)
            
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Mahalanobis computation failed: {e}")
            # Fall back to cosine similarity
            return self._cosine_similarity(vec, mean)
    
    def _hamming_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute Hamming similarity for hypervectors.
        
        Returns similarity in [0, 1] where 1 is identical.
        """
        # Convert to ternary/binary representation
        vec1_discrete = torch.sign(vec1)
        vec2_discrete = torch.sign(vec2)
        
        # Count matching elements
        matches = torch.sum(vec1_discrete == vec2_discrete).item()
        
        # Compute similarity
        similarity = matches / len(vec1)
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def get_best_match(self, embedding: torch.Tensor, 
                      min_similarity: Optional[float] = None) -> Tuple[Optional[str], float]:
        """
        Find the best matching concept for an embedding.
        
        Args:
            embedding: Input embedding
            min_similarity: Minimum similarity threshold (default: self.threshold)
            
        Returns:
            Tuple of (concept_name, similarity_score) or (None, 0.0) if no match
        """
        min_similarity = min_similarity or self.threshold
        
        matches = self.match_to_library(embedding)
        
        if not matches:
            return None, 0.0
        
        best_concept = next(iter(matches))
        best_score = matches[best_concept]
        
        if best_score >= min_similarity:
            return best_concept, best_score
        else:
            return None, best_score
    
    def batch_match(self, embeddings: torch.Tensor, 
                   return_scores: bool = False) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Batch matching of multiple embeddings to library.
        
        Args:
            embeddings: Batch of embeddings (n_samples, dim)
            return_scores: Whether to return similarity scores
            
        Returns:
            List of best matching concept names (and scores if requested)
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        
        results = []
        for i in range(embeddings.shape[0]):
            concept, score = self.get_best_match(embeddings[i])
            if return_scores:
                results.append((concept, score))
            else:
                results.append(concept)
        
        return results
    
    def compute_coverage(self, embeddings: torch.Tensor) -> float:
        """
        Compute how well the library covers a set of embeddings.
        
        Coverage is the fraction of embeddings that match at least one concept
        above the threshold.
        
        Args:
            embeddings: Batch of embeddings
            
        Returns:
            Coverage score in [0, 1]
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        
        matched_count = 0
        for i in range(embeddings.shape[0]):
            concept, score = self.get_best_match(embeddings[i])
            if concept is not None:
                matched_count += 1
        
        coverage = matched_count / embeddings.shape[0]
        return float(coverage)


# Standalone functions for backward compatibility

def compute_semantic_distance(concept1: ConceptVector, concept2: ConceptVector,
                            metric: SemanticDistance = SemanticDistance.COSINE,
                            normalize: bool = True) -> float:
    """
    Compute semantic distance between two concept vectors.
    
    Args:
        concept1: First concept vector
        concept2: Second concept vector
        metric: Distance metric to use
        normalize: Whether to normalize vectors before computing distance
        
    Returns:
        Distance value
    """
    vec1 = torch.tensor(concept1.vector) if isinstance(concept1.vector, np.ndarray) else concept1.vector
    vec2 = torch.tensor(concept2.vector) if isinstance(concept2.vector, np.ndarray) else concept2.vector
    
    if normalize:
        vec1 = vec1 / (torch.norm(vec1) + 1e-8)
        vec2 = vec2 / (torch.norm(vec2) + 1e-8)
    
    if metric == SemanticDistance.COSINE:
        distance = 1.0 - torch.dot(vec1, vec2).item()
    elif metric == SemanticDistance.EUCLIDEAN:
        distance = torch.norm(vec1 - vec2).item()
    elif metric == SemanticDistance.MANHATTAN:
        distance = torch.sum(torch.abs(vec1 - vec2)).item()
    else:
        # Default to cosine
        distance = 1.0 - torch.dot(vec1, vec2).item()
    
    return float(distance)


def semantic_similarity_score(concept1: ConceptVector, concept2: ConceptVector,
                            metric: SemanticDistance = SemanticDistance.COSINE,
                            normalize: bool = True) -> float:
    """
    Compute semantic similarity score between two concept vectors.
    
    Args:
        concept1: First concept vector
        concept2: Second concept vector
        metric: Distance metric to use
        normalize: Whether to normalize vectors before computing similarity
        
    Returns:
        Similarity score between 0 and 1 (higher is more similar)
    """
    distance = compute_semantic_distance(concept1, concept2, metric, normalize)
    
    # Convert distance to similarity
    if metric in [SemanticDistance.EUCLIDEAN, SemanticDistance.MANHATTAN]:
        # Use exponential decay
        similarity = np.exp(-distance)
    else:
        # For cosine and others, distance is already in [0, 2]
        similarity = 1.0 - (distance / 2.0)
    
    return float(np.clip(similarity, 0.0, 1.0))


def batch_semantic_matching(query_concepts: List[ConceptVector],
                          library: SemanticLibrary,
                          config: Optional[MatchingConfig] = None,
                          top_k: int = 5) -> Dict[str, List[SemanticMatchResult]]:
    """
    Perform batch semantic matching for multiple query concepts.
    
    Args:
        query_concepts: List of concepts to match
        library: Library to search against
        config: Matching configuration
        top_k: Number of top matches per query
        
    Returns:
        Dictionary mapping query concept IDs to their match results
    """
    from .types import MatchingConfig as MC
    config = config or MC()
    
    # Create a temporary matcher for batch processing
    class TempMatcher:
        def __init__(self):
            self.config = config
            
    matcher = TempMatcher()
    results = {}
    
    for query in query_concepts:
        query_results = []
        for lib_id, lib_concept in library.concepts.items():
            if lib_id == query.concept_id:
                continue
            
            similarity = semantic_similarity_score(
                query, lib_concept, 
                config.distance_metric, 
                config.normalize_vectors
            )
            
            if similarity >= config.similarity_threshold:
                result = SemanticMatchResult(
                    query_concept_id=query.concept_id,
                    matched_concept_id=lib_id,
                    distance=1.0 - similarity,
                    similarity_score=similarity,
                    distance_metric=config.distance_metric,
                    metadata={'timestamp': time.time()}
                )
                query_results.append(result)
        
        # Sort and take top_k
        query_results.sort(key=lambda x: x.similarity_score, reverse=True)
        results[query.concept_id] = query_results[:top_k]
    
    return results