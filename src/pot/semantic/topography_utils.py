"""
Utility functions for topographical learning and analysis.
Provides helper functions for data preprocessing, metric computation,
and topological analysis.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr, kendalltau
import logging

logger = logging.getLogger(__name__)


def compute_distance_matrix(vectors: np.ndarray,
                           metric: str = "euclidean",
                           normalize: bool = False) -> np.ndarray:
    """
    Compute pairwise distance matrix.
    
    Args:
        vectors: Input vectors
        metric: Distance metric
        normalize: Whether to normalize distances
        
    Returns:
        Distance matrix
    """
    if metric == "precomputed":
        dist_matrix = vectors
    else:
        # Compute pairwise distances
        if metric == "cosine" and normalize:
            # Normalize for cosine similarity
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / (norms + 1e-10)
        
        dist_matrix = squareform(pdist(vectors, metric=metric))
    
    if normalize and metric != "precomputed":
        # Normalize to [0, 1]
        dist_max = dist_matrix.max()
        if dist_max > 0:
            dist_matrix = dist_matrix / dist_max
    
    return dist_matrix


def compute_neighborhood_preservation(X_high: np.ndarray,
                                     X_low: np.ndarray,
                                     k: int = 10) -> Dict[str, float]:
    """
    Compute neighborhood preservation metrics.
    
    Args:
        X_high: High-dimensional data
        X_low: Low-dimensional embedding
        k: Number of neighbors
        
    Returns:
        Dictionary of preservation metrics
    """
    n_samples = X_high.shape[0]
    k = min(k, n_samples - 1)
    
    # Compute distance matrices
    D_high = compute_distance_matrix(X_high)
    D_low = compute_distance_matrix(X_low)
    
    metrics = {}
    
    # Compute trustworthiness
    # Proportion of k-NN in low-d that were also k-NN in high-d
    trust_sum = 0
    for i in range(n_samples):
        # Get k nearest neighbors in both spaces
        nn_high = np.argsort(D_high[i])[1:k+1]
        nn_low = np.argsort(D_low[i])[1:k+1]
        
        # Find false neighbors (in low but not in high)
        false_nn = np.setdiff1d(nn_low, nn_high)
        
        for j in false_nn:
            # Rank of j in high-dimensional space
            rank_high = np.where(np.argsort(D_high[i]) == j)[0][0]
            trust_sum += max(0, rank_high - k)
    
    max_trust = n_samples * k * (n_samples - k - 1) / 2
    if max_trust > 0:
        metrics['trustworthiness'] = 1 - (2 * trust_sum / max_trust)
    else:
        metrics['trustworthiness'] = 1.0
    
    # Compute continuity
    # Proportion of k-NN in high-d that remain k-NN in low-d
    cont_sum = 0
    for i in range(n_samples):
        nn_high = np.argsort(D_high[i])[1:k+1]
        nn_low = np.argsort(D_low[i])[1:k+1]
        
        # Find missing neighbors (in high but not in low)
        missing_nn = np.setdiff1d(nn_high, nn_low)
        
        for j in missing_nn:
            # Rank of j in low-dimensional space
            rank_low = np.where(np.argsort(D_low[i]) == j)[0][0]
            cont_sum += max(0, rank_low - k)
    
    max_cont = n_samples * k * (n_samples - k - 1) / 2
    if max_cont > 0:
        metrics['continuity'] = 1 - (2 * cont_sum / max_cont)
    else:
        metrics['continuity'] = 1.0
    
    # Mean relative rank error
    rank_errors = []
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            rank_high = np.where(np.argsort(D_high[i]) == j)[0][0]
            rank_low = np.where(np.argsort(D_low[i]) == j)[0][0]
            error = abs(rank_high - rank_low) / (n_samples - 1)
            rank_errors.append(error)
    
    metrics['mean_relative_rank_error'] = np.mean(rank_errors)
    
    # Local continuity meta-criterion (LCMC)
    lcmc = 0
    for i in range(n_samples):
        nn_high = set(np.argsort(D_high[i])[1:k+1])
        nn_low = set(np.argsort(D_low[i])[1:k+1])
        lcmc += len(nn_high.intersection(nn_low))
    
    metrics['lcmc'] = lcmc / (n_samples * k)
    
    return metrics


def compute_stress_metrics(X_high: np.ndarray,
                          X_low: np.ndarray,
                          normalized: bool = True) -> Dict[str, float]:
    """
    Compute stress-based quality metrics.
    
    Args:
        X_high: High-dimensional data
        X_low: Low-dimensional embedding
        normalized: Whether to compute normalized stress
        
    Returns:
        Dictionary of stress metrics
    """
    # Compute distance matrices
    D_high = pdist(X_high)
    D_low = pdist(X_low)
    
    metrics = {}
    
    # Raw stress (Kruskal stress formula 1)
    stress_1 = np.sqrt(np.sum((D_high - D_low) ** 2) / np.sum(D_high ** 2))
    metrics['kruskal_stress_1'] = float(stress_1)
    
    # Sammon stress (weighted by original distances)
    sammon_numerator = np.sum(((D_high - D_low) ** 2) / (D_high + 1e-10))
    sammon_denominator = np.sum(D_high)
    if sammon_denominator > 0:
        metrics['sammon_stress'] = float(sammon_numerator / sammon_denominator)
    else:
        metrics['sammon_stress'] = 0.0
    
    # Normalized stress
    if normalized:
        mean_high = np.mean(D_high)
        mean_low = np.mean(D_low)
        std_high = np.std(D_high)
        std_low = np.std(D_low)
        
        # Standardize distances
        D_high_norm = (D_high - mean_high) / (std_high + 1e-10)
        D_low_norm = (D_low - mean_low) / (std_low + 1e-10)
        
        normalized_stress = np.sqrt(np.mean((D_high_norm - D_low_norm) ** 2))
        metrics['normalized_stress'] = float(normalized_stress)
    
    # Shepard correlation (monotonic relationship)
    spearman_corr, _ = spearmanr(D_high, D_low)
    metrics['shepard_correlation'] = float(spearman_corr)
    
    # Kendall's tau (rank correlation)
    kendall_corr, _ = kendalltau(D_high, D_low)
    metrics['kendall_tau'] = float(kendall_corr)
    
    return metrics


def estimate_intrinsic_dimension(X: np.ndarray,
                                method: str = "mle",
                                k: int = 10) -> float:
    """
    Estimate intrinsic dimensionality of data.
    
    Args:
        X: Input data
        method: Estimation method ('mle', 'correlation', 'pca')
        k: Number of neighbors for local methods
        
    Returns:
        Estimated intrinsic dimension
    """
    n_samples = X.shape[0]
    
    if method == "mle":
        # Maximum Likelihood Estimation (Levina-Bickel)
        from sklearn.neighbors import NearestNeighbors
        
        k = min(k, n_samples - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1)
        nbrs.fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        # Compute MLE for each point
        estimates = []
        for i in range(n_samples):
            dists = distances[i, 1:]  # Exclude self
            if dists[-1] > 0:
                # MLE formula
                dim = (k - 1) / np.sum(np.log(dists[-1] / dists[:-1]))
                estimates.append(dim)
        
        if estimates:
            return float(np.median(estimates))
        
    elif method == "correlation":
        # Correlation dimension
        dist_matrix = pdist(X)
        sorted_dists = np.sort(dist_matrix)
        
        # Count pairs within distance r
        log_r = []
        log_count = []
        
        for i in range(10, len(sorted_dists), len(sorted_dists) // 20):
            r = sorted_dists[i]
            if r > 0:
                count = i
                log_r.append(np.log(r))
                log_count.append(np.log(count))
        
        if len(log_r) > 1:
            # Fit line in log-log space
            coef = np.polyfit(log_r, log_count, 1)
            return float(coef[0])
    
    elif method == "pca":
        # PCA-based estimation
        from sklearn.decomposition import PCA
        
        pca = PCA()
        pca.fit(X)
        
        # Find elbow in explained variance
        explained_var = pca.explained_variance_ratio_
        cumsum = np.cumsum(explained_var)
        
        # Find dimension capturing 95% variance
        dim = np.argmax(cumsum >= 0.95) + 1
        return float(dim)
    
    return float(X.shape[1])  # Fallback to ambient dimension


def compute_geodesic_distances(X: np.ndarray,
                              n_neighbors: int = 10,
                              method: str = "isomap") -> np.ndarray:
    """
    Compute geodesic distances on manifold.
    
    Args:
        X: Input data points
        n_neighbors: Number of neighbors for graph construction
        method: Method for geodesic computation
        
    Returns:
        Geodesic distance matrix
    """
    n_samples = X.shape[0]
    n_neighbors = min(n_neighbors, n_samples - 1)
    
    if method == "isomap":
        from sklearn.manifold import Isomap
        
        iso = Isomap(n_neighbors=n_neighbors, n_components=2)
        iso.fit(X)
        
        if hasattr(iso, 'dist_matrix_'):
            return iso.dist_matrix_
    
    # Fallback: build graph and compute shortest paths
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path
    
    # Build k-NN graph
    graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance')
    
    # Compute shortest paths
    dist_matrix = shortest_path(graph, directed=False)
    
    # Handle infinite distances
    max_finite = np.max(dist_matrix[np.isfinite(dist_matrix)])
    dist_matrix[np.isinf(dist_matrix)] = max_finite * 2
    
    return dist_matrix


def compute_persistence_diagram(X: np.ndarray,
                               max_dimension: int = 1,
                               max_edge_length: float = np.inf) -> List[np.ndarray]:
    """
    Compute topological persistence diagram.
    
    Args:
        X: Input data points
        max_dimension: Maximum homology dimension
        max_edge_length: Maximum edge length for Rips complex
        
    Returns:
        List of persistence diagrams for each dimension
    """
    try:
        import gudhi
        
        # Build Rips complex
        rips = gudhi.RipsComplex(points=X, max_edge_length=max_edge_length)
        simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension + 1)
        
        # Compute persistence
        persistence = simplex_tree.persistence()
        
        # Organize by dimension
        diagrams = [[] for _ in range(max_dimension + 1)]
        for dim, (birth, death) in persistence:
            if dim <= max_dimension:
                if death == float('inf'):
                    death = max_edge_length
                diagrams[dim].append([birth, death])
        
        # Convert to numpy arrays
        diagrams = [np.array(diag) if diag else np.array([]).reshape(0, 2) 
                   for diag in diagrams]
        
        return diagrams
        
    except ImportError:
        logger.warning("GUDHI not installed. Cannot compute persistence diagram.")
        return [np.array([]).reshape(0, 2) for _ in range(max_dimension + 1)]


def compute_bottleneck_distance(diagram1: np.ndarray,
                               diagram2: np.ndarray) -> float:
    """
    Compute bottleneck distance between persistence diagrams.
    
    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram
        
    Returns:
        Bottleneck distance
    """
    try:
        import gudhi
        
        if len(diagram1) == 0 and len(diagram2) == 0:
            return 0.0
        
        distance = gudhi.bottleneck_distance(diagram1, diagram2)
        return float(distance)
        
    except ImportError:
        logger.warning("GUDHI not installed. Using simple approximation.")
        
        # Simple approximation: Hausdorff distance
        if len(diagram1) == 0 or len(diagram2) == 0:
            return np.inf
        
        dist_matrix = cdist(diagram1, diagram2)
        hausdorff1 = np.max(np.min(dist_matrix, axis=1))
        hausdorff2 = np.max(np.min(dist_matrix, axis=0))
        
        return float(max(hausdorff1, hausdorff2))


def compute_wasserstein_distance(diagram1: np.ndarray,
                                diagram2: np.ndarray,
                                p: int = 2) -> float:
    """
    Compute Wasserstein distance between persistence diagrams.
    
    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram
        p: Order of Wasserstein distance
        
    Returns:
        Wasserstein distance
    """
    try:
        import gudhi
        
        if len(diagram1) == 0 and len(diagram2) == 0:
            return 0.0
        
        distance = gudhi.wasserstein_distance(diagram1, diagram2, order=p)
        return float(distance)
        
    except ImportError:
        logger.warning("GUDHI not installed. Using scipy implementation.")
        
        try:
            from scipy.stats import wasserstein_distance
            
            # Project to persistence (death - birth)
            if len(diagram1) > 0:
                pers1 = diagram1[:, 1] - diagram1[:, 0]
            else:
                pers1 = np.array([0])
            
            if len(diagram2) > 0:
                pers2 = diagram2[:, 1] - diagram2[:, 0]
            else:
                pers2 = np.array([0])
            
            return float(wasserstein_distance(pers1, pers2))
            
        except:
            return np.inf


def subsample_data(X: np.ndarray,
                  n_samples: int,
                  method: str = "random",
                  preserve_structure: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample data while preserving structure.
    
    Args:
        X: Input data
        n_samples: Number of samples to select
        method: Subsampling method ('random', 'maxmin', 'density')
        preserve_structure: Whether to preserve topological structure
        
    Returns:
        Tuple of (subsampled data, selected indices)
    """
    n_total = X.shape[0]
    
    if n_samples >= n_total:
        return X, np.arange(n_total)
    
    if method == "random":
        indices = np.random.choice(n_total, n_samples, replace=False)
        
    elif method == "maxmin":
        # Maxmin sampling (furthest point sampling)
        indices = [np.random.randint(n_total)]
        dist_to_set = np.full(n_total, np.inf)
        
        for _ in range(n_samples - 1):
            last_idx = indices[-1]
            dists = np.linalg.norm(X - X[last_idx], axis=1)
            dist_to_set = np.minimum(dist_to_set, dists)
            next_idx = np.argmax(dist_to_set)
            indices.append(next_idx)
        
        indices = np.array(indices)
        
    elif method == "density":
        # Density-based sampling
        from sklearn.neighbors import KernelDensity
        
        kde = KernelDensity(bandwidth=1.0)
        kde.fit(X)
        log_density = kde.score_samples(X)
        
        # Sample proportional to inverse density (prefer outliers)
        if preserve_structure:
            weights = np.exp(-log_density)
        else:
            weights = np.exp(log_density)
        
        weights = weights / weights.sum()
        indices = np.random.choice(n_total, n_samples, replace=False, p=weights)
    
    else:
        indices = np.random.choice(n_total, n_samples, replace=False)
    
    return X[indices], indices


def align_embeddings(X_ref: np.ndarray,
                    X_test: np.ndarray,
                    method: str = "procrustes") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Align two embeddings for comparison.
    
    Args:
        X_ref: Reference embedding
        X_test: Test embedding to align
        method: Alignment method
        
    Returns:
        Tuple of (aligned test embedding, alignment info)
    """
    info = {'method': method}
    
    if method == "procrustes":
        from scipy.spatial import procrustes
        
        # Orthogonal Procrustes alignment
        _, X_aligned, disparity = procrustes(X_ref, X_test)
        info['disparity'] = float(disparity)
        
    elif method == "affine":
        # Affine transformation
        # Add bias term
        X_test_bias = np.c_[X_test, np.ones(X_test.shape[0])]
        
        # Solve for transformation matrix
        A, _, _, _ = np.linalg.lstsq(X_test_bias, X_ref, rcond=None)
        
        X_aligned = X_test_bias @ A
        info['transform_matrix'] = A
        
    elif method == "center_scale":
        # Simple centering and scaling
        X_test_centered = X_test - X_test.mean(axis=0)
        X_ref_centered = X_ref - X_ref.mean(axis=0)
        
        scale = np.std(X_ref_centered) / (np.std(X_test_centered) + 1e-10)
        X_aligned = X_test_centered * scale + X_ref.mean(axis=0)
        
        info['scale'] = float(scale)
    
    else:
        X_aligned = X_test
    
    return X_aligned, info


def compute_local_quality(X_high: np.ndarray,
                         X_low: np.ndarray,
                         k: int = 10) -> np.ndarray:
    """
    Compute per-point quality metrics.
    
    Args:
        X_high: High-dimensional data
        X_low: Low-dimensional embedding
        k: Number of neighbors
        
    Returns:
        Array of quality scores per point
    """
    n_samples = X_high.shape[0]
    k = min(k, n_samples - 1)
    
    quality_scores = np.zeros(n_samples)
    
    # Compute distance matrices
    D_high = squareform(pdist(X_high))
    D_low = squareform(pdist(X_low))
    
    for i in range(n_samples):
        # Get k nearest neighbors
        nn_high = set(np.argsort(D_high[i])[1:k+1])
        nn_low = set(np.argsort(D_low[i])[1:k+1])
        
        # Jaccard similarity of neighborhoods
        intersection = len(nn_high.intersection(nn_low))
        union = len(nn_high.union(nn_low))
        
        if union > 0:
            quality_scores[i] = intersection / union
        else:
            quality_scores[i] = 1.0
    
    return quality_scores


def detect_outliers_topological(X: np.ndarray,
                               contamination: float = 0.1,
                               method: str = "isolation") -> np.ndarray:
    """
    Detect outliers using topological methods.
    
    Args:
        X: Input data
        contamination: Expected proportion of outliers
        method: Detection method
        
    Returns:
        Binary array indicating outliers
    """
    n_samples = X.shape[0]
    
    if method == "isolation":
        try:
            from sklearn.ensemble import IsolationForest
            
            detector = IsolationForest(contamination=contamination, random_state=42)
            outliers = detector.fit_predict(X) == -1
            
        except ImportError:
            logger.warning("IsolationForest not available")
            outliers = np.zeros(n_samples, dtype=bool)
            
    elif method == "local_outlier":
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            detector = LocalOutlierFactor(contamination=contamination)
            outliers = detector.fit_predict(X) == -1
            
        except ImportError:
            logger.warning("LocalOutlierFactor not available")
            outliers = np.zeros(n_samples, dtype=bool)
            
    elif method == "distance":
        # Distance-based outlier detection
        dist_matrix = squareform(pdist(X))
        mean_distances = dist_matrix.mean(axis=1)
        
        threshold = np.percentile(mean_distances, (1 - contamination) * 100)
        outliers = mean_distances > threshold
        
    else:
        outliers = np.zeros(n_samples, dtype=bool)
    
    return outliers


def prepare_latents_for_projection(latents: torch.Tensor,
                                   normalize: bool = True,
                                   reduce_dims: Optional[int] = None) -> np.ndarray:
    """
    Prepare latent vectors for projection.
    
    Args:
        latents: Input latent vectors (torch.Tensor)
        normalize: Whether to normalize vectors
        reduce_dims: Optional dimensionality reduction before projection
        
    Returns:
        Prepared numpy array for projection
    """
    # Convert to numpy
    if isinstance(latents, torch.Tensor):
        data = latents.detach().cpu().numpy()
    else:
        data = np.array(latents)
    
    # Handle different shapes
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        # Flatten if needed
        data = data.reshape(data.shape[0], -1)
    
    # Normalize if requested
    if normalize:
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        data = data / (norms + 1e-10)
    
    # Initial dimensionality reduction if requested
    if reduce_dims is not None and reduce_dims < data.shape[1]:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=reduce_dims, random_state=42)
        data = pca.fit_transform(data)
        logger.info(f"Reduced dimensions from {latents.shape[-1]} to {reduce_dims}")
    
    return data


def compute_intrinsic_dimension(data: np.ndarray,
                                method: str = 'mle') -> float:
    """
    Estimate intrinsic dimensionality of data.
    
    Args:
        data: Input data
        method: Estimation method ('mle', 'correlation', 'pca')
        
    Returns:
        Estimated intrinsic dimension
    """
    return estimate_intrinsic_dimension(data, method=method)


def select_optimal_parameters(data: np.ndarray,
                              method: str) -> Dict[str, Any]:
    """
    Auto-select optimal parameters for projection method.
    
    Args:
        data: Input data for parameter selection
        method: Projection method name
        
    Returns:
        Dictionary of optimal parameters
    """
    n_samples = data.shape[0]
    n_features = data.shape[1]
    
    params = {}
    
    if method == 'umap':
        # UMAP parameter selection
        n_neighbors = min(15, max(5, int(np.sqrt(n_samples))))
        min_dist = 0.1 if n_samples < 1000 else 0.01
        
        params = {
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'n_components': 2 if n_features > 2 else n_features,
            'metric': 'euclidean',
            'random_state': 42
        }
        
    elif method == 'tsne':
        # t-SNE parameter selection
        perplexity = min(50, max(5, n_samples // 4))
        learning_rate = max(10.0, n_samples / 12.0)
        
        params = {
            'perplexity': perplexity,
            'learning_rate': learning_rate,
            'n_components': 2,
            'n_iter': 1000 if n_samples < 5000 else 500,
            'random_state': 42
        }
        
    elif method == 'som':
        # SOM parameter selection
        grid_size = int(np.sqrt(5 * np.sqrt(n_samples)))
        grid_size = max(5, min(grid_size, 50))
        
        params = {
            'x_size': grid_size,
            'y_size': grid_size,
            'input_len': n_features,
            'sigma': grid_size / 2.0,
            'learning_rate': 0.5,
            'num_iteration': max(500, n_samples * 10)
        }
        
    elif method == 'pca':
        # PCA parameter selection
        n_components = min(n_features, n_samples, 50)
        params = {
            'n_components': n_components,
            'whiten': True,
            'random_state': 42
        }
    
    logger.info(f"Auto-selected {method} parameters: {params}")
    return params


def compute_trustworthiness(X_high: np.ndarray,
                           X_low: np.ndarray,
                           n_neighbors: int = 5) -> float:
    """
    Measure how well local structure is preserved.
    
    Args:
        X_high: High-dimensional data
        X_low: Low-dimensional embedding
        n_neighbors: Number of neighbors to consider
        
    Returns:
        Trustworthiness score (0-1, higher is better)
    """
    metrics = compute_neighborhood_preservation(X_high, X_low, k=n_neighbors)
    return metrics.get('trustworthiness', 0.0)


def compute_continuity(X_high: np.ndarray,
                      X_low: np.ndarray,
                      n_neighbors: int = 5) -> float:
    """
    Measure projection continuity.
    
    Args:
        X_high: High-dimensional data
        X_low: Low-dimensional embedding
        n_neighbors: Number of neighbors to consider
        
    Returns:
        Continuity score (0-1, higher is better)
    """
    metrics = compute_neighborhood_preservation(X_high, X_low, k=n_neighbors)
    return metrics.get('continuity', 0.0)


def compute_shepard_correlation(distances_high: np.ndarray,
                                distances_low: np.ndarray) -> float:
    """
    Compute correlation between high/low dimensional distances.
    
    Args:
        distances_high: Pairwise distances in high-dimensional space
        distances_low: Pairwise distances in low-dimensional space
        
    Returns:
        Spearman correlation coefficient
    """
    # Flatten distance matrices if needed
    if distances_high.ndim == 2:
        distances_high = squareform(distances_high)
    if distances_low.ndim == 2:
        distances_low = squareform(distances_low)
    
    # Compute Spearman correlation
    corr, _ = spearmanr(distances_high, distances_low)
    return float(corr) if not np.isnan(corr) else 0.0


def identify_clusters_in_projection(projected_data: np.ndarray,
                                    method: str = 'dbscan',
                                    **kwargs) -> np.ndarray:
    """
    Identify clusters in projected space.
    
    Args:
        projected_data: 2D or 3D projected data
        method: Clustering method ('dbscan', 'kmeans', 'hierarchical')
        **kwargs: Additional parameters for clustering method
        
    Returns:
        Array of cluster labels
    """
    n_samples = projected_data.shape[0]
    
    if method == 'dbscan':
        try:
            from sklearn.cluster import DBSCAN
            
            # Auto-select eps if not provided
            if 'eps' not in kwargs:
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=min(5, n_samples))
                nbrs.fit(projected_data)
                distances, _ = nbrs.kneighbors(projected_data)
                distances = np.sort(distances[:, -1])
                # Find elbow in k-distance graph
                eps = np.percentile(distances, 90)
                kwargs['eps'] = eps
            
            clusterer = DBSCAN(min_samples=kwargs.get('min_samples', 5), 
                              eps=kwargs['eps'])
            labels = clusterer.fit_predict(projected_data)
            
        except ImportError:
            logger.warning("DBSCAN not available, falling back to KMeans")
            method = 'kmeans'
    
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        
        n_clusters = kwargs.get('n_clusters', min(10, n_samples // 10))
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(projected_data)
    
    elif method == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        
        n_clusters = kwargs.get('n_clusters', min(10, n_samples // 10))
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(projected_data)
    
    return labels


def track_cluster_evolution(snapshots: List[np.ndarray],
                           method: str = 'dbscan',
                           **kwargs) -> Dict[str, Any]:
    """
    Track how clusters change over time.
    
    Args:
        snapshots: List of projected data at different time points
        method: Clustering method to use
        **kwargs: Additional parameters for clustering
        
    Returns:
        Dictionary containing evolution metrics and cluster mappings
    """
    evolution = {
        'n_snapshots': len(snapshots),
        'cluster_labels': [],
        'n_clusters': [],
        'cluster_sizes': [],
        'transitions': [],
        'stability_scores': []
    }
    
    prev_labels = None
    
    for i, snapshot in enumerate(snapshots):
        # Identify clusters in current snapshot
        labels = identify_clusters_in_projection(snapshot, method=method, **kwargs)
        evolution['cluster_labels'].append(labels)
        
        # Count clusters (excluding noise label -1 for DBSCAN)
        unique_labels = np.unique(labels[labels >= 0])
        n_clusters = len(unique_labels)
        evolution['n_clusters'].append(n_clusters)
        
        # Compute cluster sizes
        sizes = {}
        for label in unique_labels:
            sizes[int(label)] = np.sum(labels == label)
        evolution['cluster_sizes'].append(sizes)
        
        # Track transitions between snapshots
        if prev_labels is not None:
            transitions = compute_cluster_transitions(prev_labels, labels)
            evolution['transitions'].append(transitions)
            
            # Compute stability score
            stability = compute_cluster_stability(prev_labels, labels)
            evolution['stability_scores'].append(stability)
        
        prev_labels = labels
    
    # Compute overall metrics
    evolution['mean_n_clusters'] = np.mean(evolution['n_clusters'])
    evolution['cluster_variance'] = np.var(evolution['n_clusters'])
    
    if evolution['stability_scores']:
        evolution['mean_stability'] = np.mean(evolution['stability_scores'])
    
    return evolution


def compute_cluster_transitions(labels_prev: np.ndarray,
                                labels_curr: np.ndarray) -> Dict[str, Any]:
    """
    Compute transition matrix between cluster assignments.
    
    Args:
        labels_prev: Previous cluster labels
        labels_curr: Current cluster labels
        
    Returns:
        Dictionary with transition information
    """
    unique_prev = np.unique(labels_prev[labels_prev >= 0])
    unique_curr = np.unique(labels_curr[labels_curr >= 0])
    
    # Build transition matrix
    n_prev = len(unique_prev)
    n_curr = len(unique_curr)
    transition_matrix = np.zeros((n_prev, n_curr))
    
    for i, label_prev in enumerate(unique_prev):
        mask_prev = labels_prev == label_prev
        for j, label_curr in enumerate(unique_curr):
            mask_curr = labels_curr == label_curr
            overlap = np.sum(mask_prev & mask_curr)
            transition_matrix[i, j] = overlap
    
    # Normalize by row (previous cluster sizes)
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_probs = transition_matrix / (row_sums + 1e-10)
    
    return {
        'transition_matrix': transition_matrix,
        'transition_probs': transition_probs,
        'n_splits': np.sum(transition_probs > 0.1, axis=1),  # Clusters that split
        'n_merges': np.sum(transition_probs > 0.1, axis=0),  # Clusters that merged
    }


def compute_cluster_stability(labels_prev: np.ndarray,
                              labels_curr: np.ndarray) -> float:
    """
    Compute stability score between two clusterings.
    
    Args:
        labels_prev: Previous cluster labels
        labels_curr: Current cluster labels
        
    Returns:
        Stability score (0-1, higher is more stable)
    """
    # Use Adjusted Rand Index as stability measure
    try:
        from sklearn.metrics import adjusted_rand_score
        
        # Filter out noise points
        mask = (labels_prev >= 0) & (labels_curr >= 0)
        if np.sum(mask) > 0:
            return adjusted_rand_score(labels_prev[mask], labels_curr[mask])
        
    except ImportError:
        pass
    
    # Fallback: simple agreement ratio
    agreement = np.sum(labels_prev == labels_curr) / len(labels_prev)
    return agreement


def interpolate_manifold(X: np.ndarray,
                        indices: List[int],
                        n_points: int = 10,
                        method: str = "geodesic") -> np.ndarray:
    """
    Interpolate points along manifold.
    
    Args:
        X: Embedded data points
        indices: Indices of points to interpolate between
        n_points: Number of interpolation points
        method: Interpolation method
        
    Returns:
        Array of interpolated points
    """
    if len(indices) < 2:
        raise ValueError("Need at least 2 points for interpolation")
    
    interpolated = []
    
    if method == "linear":
        # Simple linear interpolation
        for i in range(len(indices) - 1):
            start = X[indices[i]]
            end = X[indices[i + 1]]
            
            for alpha in np.linspace(0, 1, n_points, endpoint=False):
                point = (1 - alpha) * start + alpha * end
                interpolated.append(point)
        
        # Add last point
        interpolated.append(X[indices[-1]])
        
    elif method == "geodesic":
        # Approximate geodesic interpolation
        # Build local linear model
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=min(10, X.shape[0]))
        nbrs.fit(X)
        
        for i in range(len(indices) - 1):
            start_idx = indices[i]
            end_idx = indices[i + 1]
            
            # Find approximate geodesic path
            path_indices = [start_idx]
            current = start_idx
            
            while current != end_idx:
                # Find neighbors of current
                _, neighbors = nbrs.kneighbors([X[current]])
                neighbors = neighbors[0]
                
                # Find neighbor closest to target
                best_neighbor = None
                best_dist = np.inf
                
                for n in neighbors:
                    if n not in path_indices:
                        dist = np.linalg.norm(X[n] - X[end_idx])
                        if dist < best_dist:
                            best_dist = dist
                            best_neighbor = n
                
                if best_neighbor is None:
                    break
                    
                path_indices.append(best_neighbor)
                current = best_neighbor
                
                if len(path_indices) > 100:  # Prevent infinite loops
                    break
            
            # Interpolate along path
            path_points = X[path_indices]
            total_length = np.sum(np.linalg.norm(np.diff(path_points, axis=0), axis=1))
            
            if total_length > 0:
                for j in range(n_points):
                    target_length = j * total_length / n_points
                    accumulated = 0
                    
                    for k in range(len(path_points) - 1):
                        segment_length = np.linalg.norm(path_points[k + 1] - path_points[k])
                        
                        if accumulated + segment_length >= target_length:
                            alpha = (target_length - accumulated) / segment_length
                            point = (1 - alpha) * path_points[k] + alpha * path_points[k + 1]
                            interpolated.append(point)
                            break
                        
                        accumulated += segment_length
    
    elif method == "spline":
        # Spline interpolation
        try:
            from scipy.interpolate import interp1d
            
            points = X[indices]
            
            # Parameterize by cumulative distance
            distances = np.zeros(len(points))
            for i in range(1, len(points)):
                distances[i] = distances[i-1] + np.linalg.norm(points[i] - points[i-1])
            
            # Create spline for each dimension
            splines = []
            for dim in range(X.shape[1]):
                spline = interp1d(distances, points[:, dim], kind='cubic')
                splines.append(spline)
            
            # Sample along spline
            sample_dists = np.linspace(0, distances[-1], n_points * (len(indices) - 1) + 1)
            
            for dist in sample_dists:
                point = np.array([spline(dist) for spline in splines])
                interpolated.append(point)
                
        except:
            # Fallback to linear
            return interpolate_manifold(X, indices, n_points, method="linear")
    
    return np.array(interpolated)