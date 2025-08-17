"""
Semantic matching and scoring utilities for concept vector comparison.
Implements various distance metrics and similarity scoring functions.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from scipy.spatial.distance import cosine, euclidean, cityblock
from scipy.stats import entropy
import logging
from functools import lru_cache
import time

from .types import ConceptVector, SemanticMatchResult, MatchingConfig, SemanticDistance, SemanticLibrary
from .utils import normalize_concept_vector

logger = logging.getLogger(__name__)


class SemanticMatcher:
    """
    Main class for performing semantic matching between concept vectors.
    Supports multiple distance metrics and batch processing.
    """
    
    def __init__(self, config: Optional[MatchingConfig] = None):
        """
        Initialize the semantic matcher.
        
        Args:
            config: Configuration for matching operations
        """
        self.config = config or MatchingConfig()
        self._distance_functions = self._setup_distance_functions()
        self._cache = {} if self.config.cache_results else None
    
    def _setup_distance_functions(self) -> Dict[SemanticDistance, Callable]:
        """Setup distance function mappings."""
        return {
            SemanticDistance.COSINE: self._cosine_distance,
            SemanticDistance.EUCLIDEAN: self._euclidean_distance,
            SemanticDistance.MANHATTAN: self._manhattan_distance,
            SemanticDistance.JACCARD: self._jaccard_distance,
            SemanticDistance.JENSEN_SHANNON: self._jensen_shannon_distance
        }
    
    def match_concept(self, query_concept: ConceptVector, 
                     library: SemanticLibrary,
                     top_k: Optional[int] = None) -> List[SemanticMatchResult]:
        """
        Find the best matching concepts in a library for a query concept.
        
        Args:
            query_concept: The concept to match against
            library: Library of concepts to search
            top_k: Number of top matches to return (default: config.max_candidates)
            
        Returns:
            List of SemanticMatchResult objects sorted by similarity (best first)
        """
        if not library.concepts:
            return []
        
        top_k = top_k or self.config.max_candidates
        results = []
        
        query_vector = query_concept.vector
        if self.config.normalize_vectors:
            query_vector = normalize_concept_vector(query_vector)
        
        for candidate_id, candidate_concept in library.concepts.items():
            if candidate_id == query_concept.concept_id:
                continue  # Skip self-matching
            
            try:
                distance = self._compute_distance(
                    query_vector, 
                    candidate_concept.vector,
                    self.config.distance_metric
                )
                
                similarity_score = self._distance_to_similarity(
                    distance, 
                    self.config.distance_metric
                )
                
                if similarity_score >= self.config.similarity_threshold:
                    result = SemanticMatchResult(
                        query_concept_id=query_concept.concept_id,
                        matched_concept_id=candidate_id,
                        distance=distance,
                        similarity_score=similarity_score,
                        distance_metric=self.config.distance_metric,
                        metadata={
                            'query_label': query_concept.label,
                            'matched_label': candidate_concept.label,
                            'computation_time': time.time()
                        }
                    )
                    results.append(result)
                    
            except Exception as e:
                logger.warning(f"Failed to compute distance for concept {candidate_id}: {e}")
                continue
        
        # Sort by similarity score (descending) and take top_k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    def batch_match_concepts(self, query_concepts: List[ConceptVector],
                           library: SemanticLibrary,
                           top_k: Optional[int] = None) -> Dict[str, List[SemanticMatchResult]]:
        """
        Perform batch matching for multiple query concepts.
        
        Args:
            query_concepts: List of concepts to match
            library: Library to search against
            top_k: Number of top matches per query
            
        Returns:
            Dictionary mapping query concept IDs to their match results
        """
        results = {}
        
        for i in range(0, len(query_concepts), self.config.batch_size):
            batch = query_concepts[i:i + self.config.batch_size]
            
            for query_concept in batch:
                try:
                    matches = self.match_concept(query_concept, library, top_k)
                    results[query_concept.concept_id] = matches
                except Exception as e:
                    logger.error(f"Failed to match concept {query_concept.concept_id}: {e}")
                    results[query_concept.concept_id] = []
        
        return results
    
    def _compute_distance(self, vector1: np.ndarray, vector2: np.ndarray,
                         metric: SemanticDistance) -> float:
        """
        Compute distance between two vectors using the specified metric.
        
        Args:
            vector1: First vector
            vector2: Second vector
            metric: Distance metric to use
            
        Returns:
            Distance value
        """
        # Normalize vectors if configured
        if self.config.normalize_vectors:
            vector1 = normalize_concept_vector(vector1)
            vector2 = normalize_concept_vector(vector2)
        
        # Try primary metric
        try:
            distance_func = self._distance_functions[metric]
            return distance_func(vector1, vector2)
        except Exception as e:
            logger.warning(f"Primary metric {metric} failed: {e}")
            
            # Try fallback metrics
            for fallback_metric in self.config.fallback_metrics:
                try:
                    distance_func = self._distance_functions[fallback_metric]
                    return distance_func(vector1, vector2)
                except Exception:
                    continue
            
            # If all metrics fail, raise the original exception
            raise e
    
    def _distance_to_similarity(self, distance: float, 
                               metric: SemanticDistance) -> float:
        """
        Convert distance to similarity score (0-1, higher is more similar).
        
        Args:
            distance: Distance value
            metric: Distance metric used
            
        Returns:
            Similarity score between 0 and 1
        """
        if metric == SemanticDistance.COSINE:
            # Cosine distance is 1 - cosine_similarity, so similarity = 1 - distance
            return max(0.0, min(1.0, 1.0 - distance))
        elif metric in [SemanticDistance.EUCLIDEAN, SemanticDistance.MANHATTAN]:
            # For euclidean/manhattan, use exponential decay
            return np.exp(-distance)
        elif metric == SemanticDistance.JACCARD:
            # Jaccard distance is 1 - jaccard_similarity
            return max(0.0, min(1.0, 1.0 - distance))
        elif metric == SemanticDistance.JENSEN_SHANNON:
            # JS distance is sqrt(JS divergence), convert to similarity
            return max(0.0, min(1.0, 1.0 - distance))
        else:
            # Default: exponential decay
            return np.exp(-distance)
    
    # Distance function implementations
    def _cosine_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Compute cosine distance between vectors."""
        return float(cosine(vector1, vector2))
    
    def _euclidean_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Compute Euclidean distance between vectors."""
        return float(euclidean(vector1, vector2))
    
    def _manhattan_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Compute Manhattan (L1) distance between vectors."""
        return float(cityblock(vector1, vector2))
    
    def _jaccard_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute Jaccard distance between vectors.
        Treats vectors as sets by thresholding at mean value.
        """
        # Convert to binary vectors using mean as threshold
        threshold1 = np.mean(vector1)
        threshold2 = np.mean(vector2)
        
        binary1 = vector1 > threshold1
        binary2 = vector2 > threshold2
        
        intersection = np.sum(binary1 & binary2)
        union = np.sum(binary1 | binary2)
        
        if union == 0:
            return 1.0  # Maximum distance if both vectors are all zeros
        
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity
    
    def _jensen_shannon_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute Jensen-Shannon distance between vectors.
        Treats vectors as probability distributions.
        """
        # Ensure vectors are non-negative and sum to 1
        p = np.abs(vector1)
        q = np.abs(vector2)
        
        if np.sum(p) == 0 or np.sum(q) == 0:
            return 1.0  # Maximum distance if either vector is all zeros
        
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Compute JS divergence
        m = 0.5 * (p + q)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        m = m + epsilon
        
        js_divergence = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
        
        # JS distance is sqrt(JS divergence)
        return np.sqrt(js_divergence)


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
    config = MatchingConfig(
        distance_metric=metric,
        normalize_vectors=normalize,
        cache_results=False
    )
    
    matcher = SemanticMatcher(config)
    return matcher._compute_distance(concept1.vector, concept2.vector, metric)


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
    
    config = MatchingConfig(distance_metric=metric)
    matcher = SemanticMatcher(config)
    return matcher._distance_to_similarity(distance, metric)


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
    matcher = SemanticMatcher(config)
    return matcher.batch_match_concepts(query_concepts, library, top_k)


@lru_cache(maxsize=1000)
def _cached_distance_computation(vector1_hash: str, vector2_hash: str,
                               metric: SemanticDistance) -> float:
    """
    Cached distance computation for frequently accessed vector pairs.
    Note: This is a placeholder for actual caching implementation.
    """
    # This would be implemented with proper vector reconstruction from hashes
    # For now, it's just a cache key placeholder
    pass