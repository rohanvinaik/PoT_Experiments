"""
Semantic matching and scoring utilities for concept vector comparison.
Implements various distance metrics and similarity scoring functions with ConceptLibrary integration.
Performance optimized with caching, batching, and GPU acceleration.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from scipy.spatial.distance import cosine, euclidean, cityblock
from scipy.stats import entropy
from sklearn.cluster import KMeans, AgglomerativeClustering
import logging
from functools import lru_cache, wraps
import time
import warnings
from collections import OrderedDict
import hashlib

from .types import ConceptVector, SemanticMatchResult, MatchingConfig, SemanticDistance, SemanticLibrary
from .utils import normalize_concept_vector
from .library import ConceptLibrary

logger = logging.getLogger(__name__)

# Global device for GPU acceleration
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_device(device: Union[str, torch.device]):
    """Set the device for tensor operations."""
    global _device
    _device = torch.device(device) if isinstance(device, str) else device
    logger.info(f"Semantic matcher device set to: {_device}")

def get_device() -> torch.device:
    """Get the current device for tensor operations."""
    return _device


class LRUCache:
    """Simple LRU cache implementation for similarity results."""
    
    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: str) -> Optional[float]:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: float):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)
    
    def clear(self):
        self.cache.clear()


class SemanticMatcher:
    """
    Enhanced semantic matcher for scoring outputs against concept library.
    Supports multiple similarity measures and semantic drift analysis.
    Performance optimized with caching, batching, and GPU acceleration.
    """
    
    def __init__(self, library: ConceptLibrary, threshold: float = 0.8,
                 cache_size: int = 1000, use_gpu: bool = True,
                 batch_size: int = 32):
        """
        Initialize the semantic matcher with a concept library.
        
        Args:
            library: ConceptLibrary containing reference concepts
            threshold: Similarity threshold for positive matches
            cache_size: Size of LRU cache for similarity results
            use_gpu: Whether to use GPU acceleration if available
            batch_size: Batch size for batch operations
        """
        if not isinstance(library, ConceptLibrary):
            raise TypeError("library must be a ConceptLibrary instance")
        
        self.library = library
        self.threshold = threshold
        self.batch_size = batch_size
        
        # Setup device
        self.device = _device if use_gpu else torch.device('cpu')
        
        # Enhanced caching
        self._similarity_cache = LRUCache(capacity=cache_size)
        self._embedding_cache = {}  # Cache computed embeddings
        self._concept_tensor_cache = {}  # Cache concept tensors on GPU
        
        # Precompute concept tensors and move to device
        self._precompute_concept_tensors()
        
        # Setup similarity methods based on library type
        self.similarity_methods = {
            'cosine': self._cosine_similarity_optimized,
            'euclidean': self._euclidean_similarity_optimized,
            'mahalanobis': self._mahalanobis_similarity,
            'hamming': self._hamming_similarity_optimized
        }
        
        logger.info(f"Initialized SemanticMatcher with {len(library.list_concepts())} concepts, "
                   f"threshold={threshold}, device={self.device}, cache_size={cache_size}")
    
    def _precompute_concept_tensors(self):
        """Precompute and cache concept tensors on the appropriate device."""
        for concept_name in self.library.list_concepts():
            concept_vector = self.library.get_concept_vector(concept_name)
            if concept_vector is not None:
                if isinstance(concept_vector, torch.Tensor):
                    tensor = concept_vector.float().to(self.device)
                else:
                    tensor = torch.from_numpy(concept_vector).float().to(self.device)
                self._concept_tensor_cache[concept_name] = tensor
    
    def compute_similarity(self, embedding: torch.Tensor, concept: str, 
                         method: str = 'cosine') -> float:
        """
        Compute similarity between an embedding and a concept from the library.
        Performance optimized with caching and GPU acceleration.
        
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
        
        # Ensure embedding is on the right device
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        embedding = embedding.to(self.device)
        
        # Validate embedding dimensions
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)  # Add batch dimension
        
        if embedding.shape[-1] != self.library.dim:
            raise ValueError(f"Embedding dimension {embedding.shape[-1]} doesn't match library dimension {self.library.dim}")
        
        # Generate cache key
        embedding_hash = hashlib.md5(embedding.cpu().numpy().tobytes()).hexdigest()[:8]
        cache_key = f"{embedding_hash}_{concept}_{method}"
        
        # Check cache
        cached_result = self._similarity_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
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
            self._similarity_cache.put(cache_key, similarity)
            
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
    
    # Batch processing methods
    
    def compute_batch_similarities(self, embeddings: torch.Tensor, 
                                  concepts: Optional[List[str]] = None,
                                  method: str = 'cosine') -> torch.Tensor:
        """
        Compute similarities for a batch of embeddings efficiently.
        
        Args:
            embeddings: Batch of embeddings [batch_size, dim]
            concepts: List of concept names (None for all concepts)
            method: Similarity method to use
            
        Returns:
            Similarity matrix [batch_size, n_concepts]
        """
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        embeddings = embeddings.to(self.device)
        
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        
        if concepts is None:
            concepts = self.library.list_concepts()
        
        # Get concept tensors
        concept_tensors = []
        for concept in concepts:
            if concept in self._concept_tensor_cache:
                concept_tensors.append(self._concept_tensor_cache[concept])
            else:
                vec = self.library.get_concept_vector(concept)
                tensor = torch.from_numpy(vec).float().to(self.device)
                concept_tensors.append(tensor)
        
        concept_matrix = torch.stack(concept_tensors)  # [n_concepts, dim]
        
        # Compute similarities based on method
        if method == 'cosine':
            similarities = self._batch_cosine_similarity(embeddings, concept_matrix)
        elif method == 'euclidean':
            similarities = self._batch_euclidean_similarity(embeddings, concept_matrix)
        elif method == 'hamming':
            similarities = self._batch_hamming_similarity(embeddings, concept_matrix)
        else:
            # Fallback to individual computations
            similarities = torch.zeros(embeddings.shape[0], len(concepts))
            for i, embedding in enumerate(embeddings):
                for j, concept in enumerate(concepts):
                    similarities[i, j] = self.compute_similarity(embedding, concept, method)
        
        return similarities
    
    @staticmethod
    def _batch_cosine_similarity_jit(embeddings: torch.Tensor, 
                                     concepts: torch.Tensor) -> torch.Tensor:
        """Optimized batch cosine similarity for performance."""
        # Normalize embeddings and concepts
        embeddings_norm = F.normalize(embeddings, p=2.0, dim=1)
        concepts_norm = F.normalize(concepts, p=2.0, dim=1)
        
        # Compute cosine similarity matrix
        similarities = torch.mm(embeddings_norm, concepts_norm.t())
        
        # Map from [-1, 1] to [0, 1]
        similarities = (similarities + 1.0) / 2.0
        
        return similarities
    
    def _batch_cosine_similarity(self, embeddings: torch.Tensor, 
                                 concepts: torch.Tensor) -> torch.Tensor:
        """Batch cosine similarity computation."""
        return self._batch_cosine_similarity_jit(embeddings, concepts)
    
    def _batch_euclidean_similarity(self, embeddings: torch.Tensor, 
                                    concepts: torch.Tensor) -> torch.Tensor:
        """Batch Euclidean similarity computation."""
        # Compute pairwise distances
        distances = torch.cdist(embeddings, concepts, p=2)
        
        # Normalize by maximum possible distance
        max_distance = 2.0 * np.sqrt(self.library.dim)
        
        # Convert to similarity
        similarities = 1.0 - (distances / max_distance)
        
        return torch.clamp(similarities, 0.0, 1.0)
    
    def _batch_hamming_similarity(self, embeddings: torch.Tensor, 
                                  concepts: torch.Tensor) -> torch.Tensor:
        """Batch Hamming similarity computation."""
        # Binarize vectors
        embeddings_binary = (embeddings > 0).float()
        concepts_binary = (concepts > 0).float()
        
        # Compute Hamming distance
        distances = torch.cdist(embeddings_binary, concepts_binary, p=0)
        
        # Convert to similarity
        similarities = 1.0 - (distances / self.library.dim)
        
        return similarities
    
    # Optimized similarity computation methods
    
    def _cosine_similarity_optimized(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Optimized cosine similarity using PyTorch operations.
        
        Returns similarity in [0, 1] where 1 is identical.
        """
        # Use cached concept tensor if available
        if isinstance(vec2, np.ndarray):
            vec2 = torch.from_numpy(vec2).float().to(self.device)
        
        vec1 = vec1.to(self.device)
        vec2 = vec2.to(self.device)
        
        # Use PyTorch's F.cosine_similarity
        similarity = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=1).item()
        
        # Map from [-1, 1] to [0, 1]
        similarity = (similarity + 1.0) / 2.0
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def _cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Returns similarity in [0, 1] where 1 is identical.
        """
        return self._cosine_similarity_optimized(vec1, vec2)
    
    def _euclidean_similarity_optimized(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Optimized Euclidean similarity using PyTorch operations.
        
        Returns similarity in [0, 1] where 1 is identical.
        """
        # Use cached concept tensor if available
        if isinstance(vec2, np.ndarray):
            vec2 = torch.from_numpy(vec2).float().to(self.device)
        
        vec1 = vec1.to(self.device)
        vec2 = vec2.to(self.device)
        
        # Compute Euclidean distance using torch operations
        distance = torch.dist(vec1, vec2, p=2).item()
        
        # Normalize by maximum possible distance
        max_distance = 2.0 * np.sqrt(self.library.dim)
        
        # Convert to similarity
        similarity = 1.0 - (distance / max_distance)
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def _euclidean_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute normalized Euclidean similarity between two vectors.
        
        Returns similarity in [0, 1] where 1 is identical.
        """
        return self._euclidean_similarity_optimized(vec1, vec2)
    
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
    
    def _hamming_similarity_optimized(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Optimized Hamming similarity for hypervectors using GPU operations.
        
        Returns similarity in [0, 1] where 1 is identical.
        """
        # Use cached concept tensor if available
        if isinstance(vec2, np.ndarray):
            vec2 = torch.from_numpy(vec2).float().to(self.device)
        
        vec1 = vec1.to(self.device)
        vec2 = vec2.to(self.device)
        
        # Convert to ternary/binary representation
        vec1_discrete = torch.sign(vec1)
        vec2_discrete = torch.sign(vec2)
        
        # Count matching elements using GPU operations
        matches = torch.eq(vec1_discrete, vec2_discrete).sum().item()
        
        # Compute similarity
        similarity = matches / vec1.numel()
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def _hamming_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute Hamming similarity for hypervectors.
        
        Returns similarity in [0, 1] where 1 is identical.
        """
        return self._hamming_similarity_optimized(vec1, vec2)
    
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