"""
Utility functions for semantic verification operations.
Provides helper functions for vector normalization, clustering, validation, and other common tasks.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.patches import Ellipse
import logging
import warnings

from .types import ConceptVector, SemanticLibrary, MatchingConfig, SemanticDistance
from .library import ConceptLibrary

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


# ============================================================================
# Embedding Extraction Helpers
# ============================================================================

def extract_embeddings_from_logits(logits: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
    """
    Extract embeddings from model logits or intermediate layers.
    
    Args:
        logits: Input tensor of logits or activations (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
        layer_idx: Layer index to extract from (-1 for last layer, -2 for second to last, etc.)
        
    Returns:
        Extracted embeddings tensor
        
    Raises:
        ValueError: If layer_idx is invalid or tensor shape is unexpected
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    
    if logits.dim() < 2:
        raise ValueError("logits must have at least 2 dimensions")
    
    # Handle different tensor shapes
    if logits.dim() == 2:
        # (batch_size, hidden_dim)
        embeddings = logits
    elif logits.dim() == 3:
        # (batch_size, seq_len, hidden_dim)
        # Extract from specified position
        if layer_idx < 0:
            # Negative indexing from the end
            if abs(layer_idx) > logits.shape[1]:
                raise ValueError(f"layer_idx {layer_idx} out of range for sequence length {logits.shape[1]}")
            embeddings = logits[:, layer_idx, :]
        else:
            # Positive indexing from the start
            if layer_idx >= logits.shape[1]:
                raise ValueError(f"layer_idx {layer_idx} out of range for sequence length {logits.shape[1]}")
            embeddings = logits[:, layer_idx, :]
    elif logits.dim() == 4:
        # (batch_size, num_layers, seq_len, hidden_dim) - for transformer outputs
        if abs(layer_idx) > logits.shape[1]:
            raise ValueError(f"layer_idx {layer_idx} out of range for {logits.shape[1]} layers")
        
        # Extract specified layer, then take mean over sequence
        layer_output = logits[:, layer_idx, :, :] if layer_idx >= 0 else logits[:, layer_idx, :, :]
        embeddings = torch.mean(layer_output, dim=1)  # Average over sequence length
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")
    
    # Ensure embeddings are 2D (batch_size, hidden_dim)
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    
    logger.debug(f"Extracted embeddings shape: {embeddings.shape} from logits shape: {logits.shape}")
    return embeddings


def normalize_embeddings(embeddings: torch.Tensor, method: str = 'l2', 
                        dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize embeddings using various methods.
    
    Args:
        embeddings: Input embeddings tensor
        method: Normalization method ('l2', 'l1', 'max', 'zscore', 'unit')
        dim: Dimension along which to normalize (default: -1)
        eps: Small epsilon for numerical stability
        
    Returns:
        Normalized embeddings tensor
    """
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("embeddings must be a torch.Tensor")
    
    if embeddings.numel() == 0:
        raise ValueError("embeddings cannot be empty")
    
    # Handle NaN/Inf values
    if not torch.isfinite(embeddings).all():
        logger.warning("embeddings contain non-finite values, replacing with zeros")
        embeddings = torch.where(torch.isfinite(embeddings), embeddings, torch.zeros_like(embeddings))
    
    if method == 'l2':
        # L2 normalization
        norm = torch.norm(embeddings, p=2, dim=dim, keepdim=True)
        normalized = embeddings / (norm + eps)
    
    elif method == 'l1':
        # L1 normalization
        norm = torch.norm(embeddings, p=1, dim=dim, keepdim=True)
        normalized = embeddings / (norm + eps)
    
    elif method == 'max':
        # Max normalization
        max_val = torch.max(torch.abs(embeddings), dim=dim, keepdim=True)[0]
        normalized = embeddings / (max_val + eps)
    
    elif method == 'zscore':
        # Z-score normalization
        mean = torch.mean(embeddings, dim=dim, keepdim=True)
        std = torch.std(embeddings, dim=dim, keepdim=True)
        normalized = (embeddings - mean) / (std + eps)
    
    elif method == 'unit':
        # Unit sphere normalization (project to unit hypersphere)
        norm = torch.norm(embeddings, p=2, dim=dim, keepdim=True)
        normalized = embeddings / (norm + eps)
        # Ensure unit norm
        normalized = normalized / torch.norm(normalized, p=2, dim=dim, keepdim=True)
    
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return normalized


def reduce_dimensionality(embeddings: torch.Tensor, target_dim: int, 
                         method: str = 'pca', random_state: int = 42) -> torch.Tensor:
    """
    Reduce dimensionality of embeddings using various methods.
    
    Args:
        embeddings: Input embeddings tensor (n_samples, n_features)
        target_dim: Target dimensionality
        method: Reduction method ('pca', 'svd', 'tsne', 'random')
        random_state: Random seed for reproducibility
        
    Returns:
        Reduced embeddings tensor (n_samples, target_dim)
        
    Raises:
        ValueError: If target_dim is invalid or method is unsupported
    """
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("embeddings must be a torch.Tensor")
    
    # Convert to numpy for sklearn
    embeddings_np = embeddings.detach().cpu().numpy()
    
    if embeddings_np.ndim == 1:
        embeddings_np = embeddings_np.reshape(1, -1)
    
    n_samples, n_features = embeddings_np.shape
    
    if target_dim <= 0:
        raise ValueError("target_dim must be positive")
    
    if target_dim > n_features:
        logger.warning(f"target_dim {target_dim} > n_features {n_features}, returning original")
        return embeddings
    
    if target_dim == n_features:
        return embeddings
    
    try:
        if method == 'pca':
            # Principal Component Analysis
            # PCA components cannot exceed min(n_samples, n_features)
            actual_components = min(target_dim, n_samples - 1, n_features)
            if actual_components < target_dim:
                logger.warning(f"PCA: reducing target_dim from {target_dim} to {actual_components} (limited by data)")
            reducer = PCA(n_components=actual_components, random_state=random_state)
            reduced = reducer.fit_transform(embeddings_np)
            
            # Pad with zeros if needed
            if actual_components < target_dim:
                padding = np.zeros((n_samples, target_dim - actual_components))
                reduced = np.hstack([reduced, padding])
            
            logger.debug(f"PCA explained variance ratio: {reducer.explained_variance_ratio_[:min(5, len(reducer.explained_variance_ratio_))]}")
        
        elif method == 'svd':
            # Truncated SVD (works with sparse matrices)
            # SVD components cannot exceed min(n_samples, n_features) - 1
            actual_components = min(target_dim, n_samples - 1, n_features - 1)
            if actual_components < target_dim:
                logger.warning(f"SVD: reducing target_dim from {target_dim} to {actual_components} (limited by data)")
            reducer = TruncatedSVD(n_components=actual_components, random_state=random_state)
            reduced = reducer.fit_transform(embeddings_np)
            
            # Pad with zeros if needed
            if actual_components < target_dim:
                padding = np.zeros((n_samples, target_dim - actual_components))
                reduced = np.hstack([reduced, padding])
            
            logger.debug(f"SVD explained variance ratio: {reducer.explained_variance_ratio_[:min(5, len(reducer.explained_variance_ratio_))]}")
        
        elif method == 'tsne':
            # t-SNE (typically for 2D/3D visualization)
            if target_dim > 3:
                logger.warning("t-SNE is typically used for 2D/3D visualization, consider using PCA for higher dims")
            
            # t-SNE requires n_components < n_samples
            if target_dim >= n_samples:
                raise ValueError(f"t-SNE requires target_dim ({target_dim}) < n_samples ({n_samples})")
            
            perplexity = min(30, n_samples - 1)  # Adjust perplexity for small datasets
            tsne = TSNE(n_components=target_dim, random_state=random_state, perplexity=perplexity)
            reduced = tsne.fit_transform(embeddings_np)
        
        elif method == 'random':
            # Random projection
            rng = np.random.RandomState(random_state)
            projection_matrix = rng.randn(n_features, target_dim)
            # Normalize columns
            projection_matrix = projection_matrix / np.linalg.norm(projection_matrix, axis=0)
            reduced = embeddings_np @ projection_matrix
        
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
        
        # Convert back to torch tensor
        reduced_tensor = torch.tensor(reduced, dtype=embeddings.dtype, device=embeddings.device)
        
        logger.info(f"Reduced embeddings from {n_features} to {target_dim} dimensions using {method}")
        return reduced_tensor
        
    except Exception as e:
        logger.error(f"Dimensionality reduction failed: {e}")
        raise


# ============================================================================
# Hypervector Operations
# ============================================================================

def create_random_hypervector(dim: int, sparsity: float = 0.5, 
                             ternary: bool = True, seed: Optional[int] = None) -> torch.Tensor:
    """
    Create a random hypervector with specified properties.
    
    Args:
        dim: Dimensionality of the hypervector
        sparsity: Fraction of zero elements (0 to 1)
        ternary: If True, create ternary vector {-1, 0, 1}, else binary {-1, 1}
        seed: Random seed for reproducibility
        
    Returns:
        Random hypervector tensor
        
    Raises:
        ValueError: If parameters are invalid
    """
    if dim <= 0:
        raise ValueError("dim must be positive")
    
    if not 0 <= sparsity <= 1:
        raise ValueError("sparsity must be between 0 and 1")
    
    if seed is not None:
        torch.manual_seed(seed)
    
    if ternary:
        # Create ternary hypervector {-1, 0, 1}
        # First determine positions of non-zero elements
        n_nonzero = int(dim * (1 - sparsity))
        
        # Initialize with zeros
        hypervector = torch.zeros(dim)
        
        # Randomly select positions for non-zero elements
        nonzero_positions = torch.randperm(dim)[:n_nonzero]
        
        # Randomly assign +1 or -1 to non-zero positions
        values = torch.randint(0, 2, (n_nonzero,)) * 2 - 1  # Maps {0,1} to {-1,1}
        hypervector[nonzero_positions] = values.float()
    
    else:
        # Create binary hypervector {-1, 1}
        if sparsity > 0:
            logger.warning("Binary hypervectors ignore sparsity parameter")
        
        # Random binary values
        hypervector = torch.randint(0, 2, (dim,)) * 2 - 1
        hypervector = hypervector.float()
    
    logger.debug(f"Created hypervector: dim={dim}, sparsity={sparsity:.2f}, "
                f"ternary={ternary}, nonzero={torch.count_nonzero(hypervector)}")
    
    return hypervector


def bind_hypervectors(hv1: torch.Tensor, hv2: torch.Tensor, 
                     method: str = 'xor') -> torch.Tensor:
    """
    Bind two hypervectors using various binding operations.
    
    Binding creates a new hypervector that is dissimilar to both inputs
    but can be used to retrieve one given the other.
    
    Args:
        hv1: First hypervector
        hv2: Second hypervector
        method: Binding method ('xor', 'multiply', 'circular')
        
    Returns:
        Bound hypervector
        
    Raises:
        ValueError: If dimensions don't match or method is unsupported
    """
    if hv1.shape != hv2.shape:
        raise ValueError(f"Hypervector dimensions must match: {hv1.shape} vs {hv2.shape}")
    
    if method == 'xor':
        # XOR binding (for binary/ternary vectors)
        # Map to binary first: positive -> 1, non-positive -> -1
        hv1_binary = torch.sign(hv1)
        hv2_binary = torch.sign(hv2)
        
        # XOR operation: same signs -> 1, different signs -> -1
        bound = hv1_binary * hv2_binary
    
    elif method == 'multiply':
        # Element-wise multiplication
        bound = hv1 * hv2
    
    elif method == 'circular':
        # Circular convolution (permutation-based binding)
        # Shift hv2 by positions determined by hv1
        dim = len(hv1)
        
        # Use magnitude of hv1 to determine shift amounts
        shifts = torch.abs(hv1) * dim
        shifts = shifts.long() % dim
        
        # Apply circular shifts
        bound = torch.zeros_like(hv1)
        for i in range(dim):
            shifted_idx = (i + shifts[i]) % dim
            bound[shifted_idx] = hv2[i]
    
    else:
        raise ValueError(f"Unsupported binding method: {method}")
    
    # Ensure hypervector properties (ternary/binary)
    if torch.allclose(torch.abs(hv1), torch.ones_like(hv1)):
        # Input was binary, keep output binary
        bound = torch.sign(bound)
    
    return bound


def bundle_hypervectors(hvs: List[torch.Tensor], 
                       weights: Optional[List[float]] = None,
                       threshold: float = 0.0) -> torch.Tensor:
    """
    Bundle multiple hypervectors into a single hypervector.
    
    Bundling creates a hypervector similar to all inputs (superposition).
    
    Args:
        hvs: List of hypervectors to bundle
        weights: Optional weights for weighted bundling
        threshold: Threshold for converting to ternary/binary
        
    Returns:
        Bundled hypervector
        
    Raises:
        ValueError: If list is empty or dimensions don't match
    """
    if not hvs:
        raise ValueError("Cannot bundle empty list of hypervectors")
    
    # Check dimension consistency
    dim = len(hvs[0])
    for i, hv in enumerate(hvs):
        if len(hv) != dim:
            raise ValueError(f"Hypervector {i} has dimension {len(hv)}, expected {dim}")
    
    # Stack hypervectors
    stacked = torch.stack(hvs)
    
    # Apply weights if provided
    if weights is not None:
        if len(weights) != len(hvs):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of hypervectors ({len(hvs)})")
        
        weights_tensor = torch.tensor(weights).reshape(-1, 1)
        weights_tensor = weights_tensor / weights_tensor.sum()  # Normalize
        stacked = stacked * weights_tensor
    
    # Bundle by averaging
    bundled = torch.mean(stacked, dim=0)
    
    # Threshold to create discrete hypervector
    if threshold == 0.0:
        # Ternary thresholding
        bundled = torch.sign(bundled)
    else:
        # Custom threshold
        bundled = torch.where(bundled > threshold, 1.0,
                             torch.where(bundled < -threshold, -1.0, 0.0))
    
    logger.debug(f"Bundled {len(hvs)} hypervectors, result sparsity: "
                f"{(bundled == 0).sum().item() / len(bundled):.2f}")
    
    return bundled


# ============================================================================
# Visualization Helpers
# ============================================================================

def plot_concept_space(library: ConceptLibrary, embeddings: Optional[torch.Tensor] = None,
                      method: str = 'pca', show_labels: bool = True,
                      figsize: Tuple[int, int] = (10, 8)) -> matplotlib.figure.Figure:
    """
    Visualize concept space in 2D using dimensionality reduction.
    
    Args:
        library: ConceptLibrary containing concepts to visualize
        embeddings: Optional additional embeddings to plot
        method: Dimensionality reduction method ('pca', 'tsne')
        show_labels: Whether to show concept labels
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    if len(library.list_concepts()) == 0:
        raise ValueError("Library contains no concepts to visualize")
    
    # Extract concept vectors
    concept_names = library.list_concepts()
    concept_vectors = []
    for name in concept_names:
        vector = library.get_concept_vector(name)
        concept_vectors.append(vector)
    
    # Stack concept vectors
    concepts_tensor = torch.stack(concept_vectors)
    
    # Combine with additional embeddings if provided
    if embeddings is not None:
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        
        # Ensure same dimensionality
        if embeddings.shape[-1] != concepts_tensor.shape[-1]:
            raise ValueError(f"Embedding dimension {embeddings.shape[-1]} doesn't match "
                           f"concept dimension {concepts_tensor.shape[-1]}")
        
        all_vectors = torch.cat([concepts_tensor, embeddings], dim=0)
    else:
        all_vectors = concepts_tensor
    
    # Reduce to 2D
    reduced = reduce_dimensionality(all_vectors, target_dim=2, method=method)
    reduced_np = reduced.detach().cpu().numpy()
    
    # Split back into concepts and embeddings
    concept_points = reduced_np[:len(concept_names)]
    embedding_points = reduced_np[len(concept_names):] if embeddings is not None else None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot concepts
    scatter = ax.scatter(concept_points[:, 0], concept_points[:, 1], 
                        c='blue', s=100, alpha=0.7, edgecolors='black',
                        label='Concepts')
    
    # Add labels
    if show_labels:
        for i, name in enumerate(concept_names):
            ax.annotate(name, (concept_points[i, 0], concept_points[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8)
    
    # Plot additional embeddings
    if embedding_points is not None and len(embedding_points) > 0:
        ax.scatter(embedding_points[:, 0], embedding_points[:, 1],
                  c='red', s=50, alpha=0.5, marker='^',
                  label='Test Embeddings')
    
    # Add Gaussian ellipses if using Gaussian method
    if library.method == 'gaussian' and method == 'pca':
        for i, name in enumerate(concept_names):
            mean, cov = library.get_concept_statistics(name)
            if mean is not None and cov is not None:
                # Project covariance to 2D space (simplified)
                try:
                    # Use PCA components to project covariance
                    cov_2d = np.array([[cov[0, 0].item(), cov[0, 1].item()],
                                      [cov[1, 0].item(), cov[1, 1].item()]])
                    
                    # Draw confidence ellipse
                    eigenvalues, eigenvectors = np.linalg.eig(cov_2d)
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                    
                    ellipse = Ellipse(concept_points[i], 
                                     2 * np.sqrt(eigenvalues[0]),
                                     2 * np.sqrt(eigenvalues[1]),
                                     angle=angle,
                                     facecolor='none',
                                     edgecolor='blue',
                                     alpha=0.3,
                                     linestyle='--')
                    ax.add_patch(ellipse)
                except Exception as e:
                    logger.debug(f"Could not draw ellipse for {name}: {e}")
    
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f'Concept Space Visualization ({library.method} method)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compute_embedding_statistics(embeddings: torch.Tensor) -> Dict[str, Any]:
    """
    Compute statistics for a batch of embeddings.
    
    Args:
        embeddings: Batch of embeddings (n_samples, dim)
        
    Returns:
        Dictionary with statistics
    """
    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(0)
    
    stats = {
        'mean': torch.mean(embeddings, dim=0),
        'std': torch.std(embeddings, dim=0),
        'min': torch.min(embeddings, dim=0)[0],
        'max': torch.max(embeddings, dim=0)[0],
        'norm_mean': torch.mean(torch.norm(embeddings, dim=1)),
        'n_samples': embeddings.shape[0],
        'dim': embeddings.shape[1]
    }
    
    return stats


def visualize_drift(drift_scores: List[float], timestamps: Optional[List[float]] = None,
                   threshold: float = 0.5, window_size: int = 10,
                   figsize: Tuple[int, int] = (12, 6)) -> matplotlib.figure.Figure:
    """
    Visualize semantic drift over time.
    
    Args:
        drift_scores: List of drift scores
        timestamps: Optional list of timestamps (uses indices if None)
        threshold: Drift threshold to highlight
        window_size: Window size for moving average
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    if not drift_scores:
        raise ValueError("drift_scores cannot be empty")
    
    # Use timestamps or indices
    if timestamps is None:
        timestamps = list(range(len(drift_scores)))
    elif len(timestamps) != len(drift_scores):
        raise ValueError(f"timestamps length ({len(timestamps)}) must match "
                        f"drift_scores length ({len(drift_scores)})")
    
    # Convert to numpy
    drift_array = np.array(drift_scores)
    time_array = np.array(timestamps)
    
    # Compute moving average
    if len(drift_scores) >= window_size:
        moving_avg = np.convolve(drift_array, np.ones(window_size)/window_size, mode='valid')
        ma_timestamps = time_array[window_size-1:]
    else:
        moving_avg = drift_array
        ma_timestamps = time_array
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    
    # Top plot: Drift scores over time
    ax1.plot(time_array, drift_array, 'b-', alpha=0.5, label='Drift Score')
    if len(drift_scores) >= window_size:
        ax1.plot(ma_timestamps, moving_avg, 'r-', linewidth=2, 
                label=f'Moving Avg (w={window_size})')
    
    # Add threshold line
    ax1.axhline(y=threshold, color='orange', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold})')
    
    # Highlight high drift regions
    high_drift = drift_array > threshold
    if np.any(high_drift):
        ax1.fill_between(time_array, 0, 1, where=high_drift,
                        color='red', alpha=0.2, transform=ax1.get_xaxis_transform(),
                        label='High Drift')
    
    ax1.set_ylabel('Drift Score')
    ax1.set_title('Semantic Drift Analysis')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(1.0, np.max(drift_array) * 1.1)])
    
    # Bottom plot: Drift histogram
    ax2.hist(drift_array, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=threshold, color='orange', linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(drift_array), color='green', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(drift_array):.3f}')
    ax2.set_xlabel('Drift Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Drift Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Statistics text
    stats_text = (f"Mean: {np.mean(drift_array):.3f}\n"
                 f"Std: {np.std(drift_array):.3f}\n"
                 f"Max: {np.max(drift_array):.3f}\n"
                 f"Above threshold: {np.sum(high_drift)}/{len(drift_scores)}")
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig