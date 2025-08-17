"""
Utility functions for semantic verification operations.
Provides helper functions for vector normalization, clustering, validation, and other common tasks.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
import warnings

from .types import ConceptVector, SemanticLibrary, MatchingConfig, SemanticDistance

logger = logging.getLogger(__name__)


def normalize_concept_vector(vector: np.ndarray, method: str = 'l2') -> np.ndarray:
    """
    Normalize a concept vector using the specified method.
    
    Args:
        vector: Input vector to normalize
        method: Normalization method ('l2', 'l1', 'max', 'zscore')
        
    Returns:
        Normalized vector
        
    Raises:
        ValueError: If method is unsupported or vector is invalid
    """
    if not isinstance(vector, np.ndarray):
        raise TypeError("vector must be a numpy array")
    
    if vector.ndim != 1:
        raise ValueError("vector must be 1-dimensional")
    
    if len(vector) == 0:
        raise ValueError("vector cannot be empty")
    
    if not np.isfinite(vector).all():
        raise ValueError("vector must contain only finite values")
    
    if method == 'l2':
        norm = np.linalg.norm(vector, ord=2)
        if norm == 0:
            logger.warning("L2 norm is zero, returning original vector")
            return vector.copy()
        return vector / norm
    
    elif method == 'l1':
        norm = np.linalg.norm(vector, ord=1)
        if norm == 0:
            logger.warning("L1 norm is zero, returning original vector")
            return vector.copy()
        return vector / norm
    
    elif method == 'max':
        max_val = np.max(np.abs(vector))
        if max_val == 0:
            logger.warning("Max value is zero, returning original vector")
            return vector.copy()
        return vector / max_val
    
    elif method == 'zscore':
        mean_val = np.mean(vector)
        std_val = np.std(vector)
        if std_val == 0:
            logger.warning("Standard deviation is zero, returning centered vector")
            return vector - mean_val
        return (vector - mean_val) / std_val
    
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def compute_concept_centroid(concepts: List[ConceptVector], 
                           weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Compute the weighted centroid of a set of concept vectors.
    
    Args:
        concepts: List of concept vectors
        weights: Optional weights for each concept (default: equal weights)
        
    Returns:
        Centroid vector
        
    Raises:
        ValueError: If concepts list is empty or dimensions are inconsistent
    """
    if not concepts:
        raise ValueError("concepts list cannot be empty")
    
    # Check dimension consistency
    dimensions = {concept.dimension for concept in concepts}
    if len(dimensions) > 1:
        raise ValueError(f"Inconsistent vector dimensions: {dimensions}")
    
    vectors = np.vstack([concept.vector for concept in concepts])
    
    if weights is None:
        # Equal weights
        return np.mean(vectors, axis=0)
    else:
        if len(weights) != len(concepts):
            raise ValueError("Number of weights must match number of concepts")
        
        weights = np.array(weights)
        if not np.isfinite(weights).all():
            raise ValueError("Weights must be finite")
        
        if np.sum(weights) == 0:
            raise ValueError("Sum of weights cannot be zero")
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Compute weighted centroid
        return np.average(vectors, axis=0, weights=weights)


def cluster_concepts(concepts: List[ConceptVector], 
                    method: str = 'kmeans',
                    n_clusters: Optional[int] = None,
                    **kwargs) -> Tuple[List[int], Dict[str, Any]]:
    """
    Cluster concept vectors using the specified clustering method.
    
    Args:
        concepts: List of concept vectors to cluster
        method: Clustering method ('kmeans', 'dbscan')
        n_clusters: Number of clusters (required for kmeans)
        **kwargs: Additional arguments for the clustering algorithm
        
    Returns:
        Tuple of (cluster_labels, clustering_metadata)
        
    Raises:
        ValueError: If concepts list is empty or clustering fails
    """
    if not concepts:
        raise ValueError("concepts list cannot be empty")
    
    # Check dimension consistency
    dimensions = {concept.dimension for concept in concepts}
    if len(dimensions) > 1:
        raise ValueError(f"Inconsistent vector dimensions: {dimensions}")
    
    # Prepare data matrix
    vectors = np.vstack([concept.vector for concept in concepts])
    
    # Standardize vectors for clustering
    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(vectors)
    
    metadata = {
        'method': method,
        'n_concepts': len(concepts),
        'vector_dimension': concepts[0].dimension,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    }
    
    try:
        if method == 'kmeans':
            if n_clusters is None:
                raise ValueError("n_clusters must be specified for kmeans")
            
            # Set default parameters
            kmeans_kwargs = {
                'n_clusters': n_clusters,
                'random_state': 42,
                'n_init': 10
            }
            kmeans_kwargs.update(kwargs)
            
            clusterer = KMeans(**kmeans_kwargs)
            labels = clusterer.fit_predict(vectors_scaled)
            
            metadata.update({
                'n_clusters': n_clusters,
                'inertia': clusterer.inertia_,
                'cluster_centers': clusterer.cluster_centers_,
                'n_iter': clusterer.n_iter_
            })
        
        elif method == 'dbscan':
            # Set default parameters
            dbscan_kwargs = {
                'eps': 0.5,
                'min_samples': 5
            }
            dbscan_kwargs.update(kwargs)
            
            clusterer = DBSCAN(**dbscan_kwargs)
            labels = clusterer.fit_predict(vectors_scaled)
            
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            metadata.update({
                'eps': dbscan_kwargs['eps'],
                'min_samples': dbscan_kwargs['min_samples'],
                'n_clusters_found': n_clusters_found,
                'n_noise_points': n_noise,
                'core_sample_indices': clusterer.core_sample_indices_
            })
        
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        return labels.tolist(), metadata
    
    except Exception as e:
        raise ValueError(f"Clustering failed: {e}")


def reduce_concept_dimensions(concepts: List[ConceptVector], 
                            n_components: int = 2,
                            method: str = 'pca') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reduce dimensionality of concept vectors for visualization or analysis.
    
    Args:
        concepts: List of concept vectors
        n_components: Number of components to reduce to
        method: Dimensionality reduction method ('pca')
        
    Returns:
        Tuple of (reduced_vectors, reduction_metadata)
        
    Raises:
        ValueError: If concepts list is empty or reduction fails
    """
    if not concepts:
        raise ValueError("concepts list cannot be empty")
    
    # Check dimension consistency
    dimensions = {concept.dimension for concept in concepts}
    if len(dimensions) > 1:
        raise ValueError(f"Inconsistent vector dimensions: {dimensions}")
    
    original_dim = concepts[0].dimension
    if n_components >= original_dim:
        logger.warning(f"n_components ({n_components}) >= original dimension ({original_dim})")
    
    # Prepare data matrix
    vectors = np.vstack([concept.vector for concept in concepts])
    
    metadata = {
        'method': method,
        'n_concepts': len(concepts),
        'original_dimension': original_dim,
        'reduced_dimension': n_components
    }
    
    try:
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            reduced_vectors = reducer.fit_transform(vectors)
            
            metadata.update({
                'explained_variance_ratio': reducer.explained_variance_ratio_,
                'cumulative_variance_ratio': np.cumsum(reducer.explained_variance_ratio_),
                'total_variance_explained': np.sum(reducer.explained_variance_ratio_),
                'components': reducer.components_,
                'singular_values': reducer.singular_values_
            })
        
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        return reduced_vectors, metadata
    
    except Exception as e:
        raise ValueError(f"Dimensionality reduction failed: {e}")


def validate_semantic_config(config: MatchingConfig) -> Tuple[bool, List[str]]:
    """
    Validate a semantic matching configuration.
    
    Args:
        config: MatchingConfig to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check similarity threshold
    if not 0 <= config.similarity_threshold <= 1:
        errors.append("similarity_threshold must be between 0 and 1")
    
    # Check max_candidates
    if config.max_candidates <= 0:
        errors.append("max_candidates must be positive")
    
    # Check batch_size
    if config.batch_size <= 0:
        errors.append("batch_size must be positive")
    
    # Check distance metric
    if not isinstance(config.distance_metric, SemanticDistance):
        errors.append("distance_metric must be a SemanticDistance enum value")
    
    # Check fallback metrics
    for metric in config.fallback_metrics:
        if not isinstance(metric, SemanticDistance):
            errors.append(f"Invalid fallback metric: {metric}")
    
    # Check for circular fallbacks
    if config.distance_metric in config.fallback_metrics:
        errors.append("Primary distance metric should not be in fallback metrics")
    
    return len(errors) == 0, errors


def compute_concept_statistics(concepts: List[ConceptVector]) -> Dict[str, Any]:
    """
    Compute statistical properties of a collection of concept vectors.
    
    Args:
        concepts: List of concept vectors
        
    Returns:
        Dictionary with statistical information
    """
    if not concepts:
        return {
            'n_concepts': 0,
            'dimension': None,
            'mean_vector': None,
            'std_vector': None,
            'min_values': None,
            'max_values': None
        }
    
    # Check dimension consistency
    dimensions = {concept.dimension for concept in concepts}
    if len(dimensions) > 1:
        logger.warning(f"Inconsistent vector dimensions: {dimensions}")
        return {'error': 'Inconsistent vector dimensions'}
    
    vectors = np.vstack([concept.vector for concept in concepts])
    
    # Compute norms
    norms = [concept.norm for concept in concepts]
    
    # Compute sources distribution
    sources = {}
    for concept in concepts:
        sources[concept.source] = sources.get(concept.source, 0) + 1
    
    # Count concepts with labels
    labeled_count = sum(1 for concept in concepts if concept.label)
    
    return {
        'n_concepts': len(concepts),
        'dimension': concepts[0].dimension,
        'mean_vector': np.mean(vectors, axis=0),
        'std_vector': np.std(vectors, axis=0),
        'min_values': np.min(vectors, axis=0),
        'max_values': np.max(vectors, axis=0),
        'mean_norm': np.mean(norms),
        'std_norm': np.std(norms),
        'min_norm': np.min(norms),
        'max_norm': np.max(norms),
        'sources_distribution': sources,
        'labeled_concepts': labeled_count,
        'unlabeled_concepts': len(concepts) - labeled_count
    }


def find_outlier_concepts(concepts: List[ConceptVector], 
                         method: str = 'zscore',
                         threshold: float = 3.0) -> List[Tuple[int, ConceptVector, float]]:
    """
    Find outlier concepts based on their vector properties.
    
    Args:
        concepts: List of concept vectors
        method: Outlier detection method ('zscore', 'iqr')
        threshold: Threshold for outlier detection
        
    Returns:
        List of (index, concept, outlier_score) tuples
    """
    if not concepts:
        return []
    
    # Compute vector norms as outlier detection feature
    norms = np.array([concept.norm for concept in concepts])
    
    outliers = []
    
    if method == 'zscore':
        # Z-score based outlier detection
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        if std_norm == 0:
            logger.warning("Standard deviation is zero, no outliers detected")
            return []
        
        z_scores = np.abs((norms - mean_norm) / std_norm)
        
        for i, (concept, z_score) in enumerate(zip(concepts, z_scores)):
            if z_score > threshold:
                outliers.append((i, concept, float(z_score)))
    
    elif method == 'iqr':
        # Interquartile range based outlier detection
        q1 = np.percentile(norms, 25)
        q3 = np.percentile(norms, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        for i, (concept, norm) in enumerate(zip(concepts, norms)):
            if norm < lower_bound or norm > upper_bound:
                # Compute outlier score as distance from bounds
                if norm < lower_bound:
                    score = (lower_bound - norm) / iqr
                else:
                    score = (norm - upper_bound) / iqr
                outliers.append((i, concept, float(score)))
    
    else:
        raise ValueError(f"Unsupported outlier detection method: {method}")
    
    # Sort by outlier score (descending)
    outliers.sort(key=lambda x: x[2], reverse=True)
    
    return outliers


def concept_vector_integrity_check(concept: ConceptVector) -> Tuple[bool, List[str]]:
    """
    Check the integrity of a concept vector.
    
    Args:
        concept: ConceptVector to check
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        # Check vector properties
        if not isinstance(concept.vector, np.ndarray):
            errors.append("vector is not a numpy array")
        elif concept.vector.ndim != 1:
            errors.append("vector is not 1-dimensional")
        elif len(concept.vector) == 0:
            errors.append("vector is empty")
        elif not np.isfinite(concept.vector).all():
            errors.append("vector contains non-finite values")
        
        # Check concept ID
        if not concept.concept_id:
            errors.append("concept_id is empty")
        elif not isinstance(concept.concept_id, str):
            errors.append("concept_id is not a string")
        
        # Check hash integrity if provided
        if concept.hash_value:
            expected_hash = concept._compute_hash()
            if concept.hash_value != expected_hash:
                errors.append("hash_value does not match computed hash")
        
        # Check timestamp
        if concept.timestamp is not None and concept.timestamp <= 0:
            errors.append("timestamp must be positive")
        
    except Exception as e:
        errors.append(f"Exception during integrity check: {e}")
    
    return len(errors) == 0, errors