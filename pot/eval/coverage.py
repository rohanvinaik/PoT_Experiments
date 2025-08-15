"""
Coverage metrics for challenge sets and their relationship to separation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
import warnings


def kcenter_radius(Z: np.ndarray, centers_idx: np.ndarray) -> float:
    """
    Compute k-center radius (maximum distance to nearest center)
    
    Args:
        Z: Data points (n_samples, n_features)
        centers_idx: Indices of center points
        
    Returns:
        Maximum distance to nearest center
    """
    if len(centers_idx) == 0:
        return float('inf')
    
    D = pairwise_distances(Z, Z[centers_idx])
    return float(D.min(axis=1).max())


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy with RBF kernel
    
    Args:
        X: First set of samples (n_samples_x, n_features)
        Y: Second set of samples (n_samples_y, n_features)
        gamma: RBF kernel parameter
        
    Returns:
        Biased MMD^2 estimate
    """
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    
    n, m = X.shape[0], Y.shape[0]
    
    # Biased MMD^2
    mmd2 = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    
    return float(mmd2)


def compute_coverage_metrics(embeddings: np.ndarray, 
                            centers_idx: Optional[np.ndarray] = None,
                            n_centers: int = 10) -> Dict[str, float]:
    """
    Compute various coverage metrics for a set of embeddings
    
    Args:
        embeddings: Challenge embeddings (n_samples, n_features)
        centers_idx: Pre-selected center indices (optional)
        n_centers: Number of centers to select if not provided
        
    Returns:
        Dict of coverage metrics
    """
    n_samples = embeddings.shape[0]
    
    # Select centers if not provided
    if centers_idx is None:
        # Simple k-means++ initialization for centers
        centers_idx = []
        for _ in range(min(n_centers, n_samples)):
            if len(centers_idx) == 0:
                # First center random
                centers_idx.append(np.random.randint(n_samples))
            else:
                # Next center: furthest from existing centers
                D = pairwise_distances(embeddings, embeddings[centers_idx])
                min_dists = D.min(axis=1)
                centers_idx.append(np.argmax(min_dists))
        centers_idx = np.array(centers_idx)
    
    # Compute k-center radius
    radius = kcenter_radius(embeddings, centers_idx)
    
    # Compute pairwise distances statistics
    D_all = pairwise_distances(embeddings)
    np.fill_diagonal(D_all, np.inf)  # Ignore self-distances
    
    metrics = {
        "kcenter_radius": radius,
        "n_centers": len(centers_idx),
        "mean_nn_dist": float(D_all.min(axis=1).mean()),
        "std_nn_dist": float(D_all.min(axis=1).std()),
        "mean_pairwise_dist": float(D_all[D_all != np.inf].mean()),
        "std_pairwise_dist": float(D_all[D_all != np.inf].std()),
        "effective_dimension": estimate_intrinsic_dimension(embeddings),
    }
    
    return metrics


def estimate_intrinsic_dimension(X: np.ndarray, k: int = 5) -> float:
    """
    Estimate intrinsic dimension using MLE on k-NN distances
    
    Args:
        X: Data points (n_samples, n_features)
        k: Number of neighbors to use
        
    Returns:
        Estimated intrinsic dimension
    """
    n = X.shape[0]
    if n <= k + 1:
        return float(X.shape[1])  # Return ambient dimension
    
    # Compute k-NN distances
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    
    # Sort distances and get k-th neighbor distance
    D_sorted = np.sort(D, axis=1)
    r_k = D_sorted[:, k-1]  # k-th nearest neighbor distance
    r_1 = D_sorted[:, 0]    # nearest neighbor distance
    
    # MLE estimate (Levina-Bickel estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_ratio = np.log(r_k / r_1)
        log_ratio = log_ratio[np.isfinite(log_ratio)]
        
        if len(log_ratio) > 0:
            d_est = (k - 1) / np.mean(log_ratio)
            return float(np.clip(d_est, 1, X.shape[1]))
        else:
            return float(X.shape[1])


def coverage_separation_analysis(challenge_embeddings: np.ndarray,
                                 genuine_distances: np.ndarray,
                                 impostor_distances: np.ndarray) -> Dict[str, Any]:
    """
    Analyze relationship between coverage and separation
    
    Args:
        challenge_embeddings: Embeddings of challenges
        genuine_distances: Distances for genuine pairs
        impostor_distances: Distances for impostor pairs
        
    Returns:
        Analysis results
    """
    from sklearn.metrics import roc_auc_score
    
    # Compute coverage metrics
    coverage = compute_coverage_metrics(challenge_embeddings)
    
    # Compute separation metrics
    labels = np.concatenate([
        np.ones(len(genuine_distances)),
        np.zeros(len(impostor_distances))
    ])
    scores = np.concatenate([genuine_distances, impostor_distances])
    
    auroc = roc_auc_score(labels, -scores)  # Negative because lower distance = genuine
    
    # T-statistic (normalized mean difference)
    mean_genuine = genuine_distances.mean()
    mean_impostor = impostor_distances.mean()
    std_genuine = genuine_distances.std()
    std_impostor = impostor_distances.std()
    
    pooled_std = np.sqrt((std_genuine**2 + std_impostor**2) / 2)
    t_statistic = abs(mean_impostor - mean_genuine) / (pooled_std + 1e-10)
    
    # D-prime (signal detection theory)
    d_prime = (mean_impostor - mean_genuine) / pooled_std if pooled_std > 0 else 0
    
    return {
        "coverage": coverage,
        "separation": {
            "auroc": float(auroc),
            "t_statistic": float(t_statistic),
            "d_prime": float(d_prime),
            "mean_genuine": float(mean_genuine),
            "mean_impostor": float(mean_impostor),
            "std_genuine": float(std_genuine),
            "std_impostor": float(std_impostor),
        },
        "correlation": {
            "radius_vs_auroc": (coverage["kcenter_radius"], auroc),
            "dimension_vs_auroc": (coverage["effective_dimension"], auroc),
        }
    }


def multi_scale_coverage(embeddings: np.ndarray, 
                         scales: List[int] = [5, 10, 20, 50]) -> Dict[str, Any]:
    """
    Compute coverage metrics at multiple scales
    
    Args:
        embeddings: Challenge embeddings
        scales: List of numbers of centers to use
        
    Returns:
        Multi-scale coverage analysis
    """
    results = {}
    
    for n_centers in scales:
        if n_centers > embeddings.shape[0]:
            continue
        
        metrics = compute_coverage_metrics(embeddings, n_centers=n_centers)
        results[f"scale_{n_centers}"] = metrics
    
    # Compute scale-invariant metrics
    radii = [r["kcenter_radius"] for r in results.values()]
    
    results["summary"] = {
        "min_radius": min(radii) if radii else 0,
        "max_radius": max(radii) if radii else 0,
        "mean_radius": np.mean(radii) if radii else 0,
        "radius_decay_rate": compute_decay_rate(scales, radii) if len(radii) > 1 else 0,
    }
    
    return results


def compute_decay_rate(scales: List[int], radii: List[float]) -> float:
    """
    Compute exponential decay rate of radius vs scale
    
    Args:
        scales: List of scales (n_centers)
        radii: Corresponding radii
        
    Returns:
        Decay rate (negative for decreasing radius)
    """
    if len(scales) < 2:
        return 0.0
    
    # Fit log(radius) ~ log(scale)
    log_scales = np.log(scales[:len(radii)])
    log_radii = np.log(np.maximum(radii, 1e-10))
    
    # Linear regression
    A = np.vstack([log_scales, np.ones(len(log_scales))]).T
    m, c = np.linalg.lstsq(A, log_radii, rcond=None)[0]
    
    return float(m)  # Slope of log-log plot