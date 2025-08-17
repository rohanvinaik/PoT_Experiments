"""
Distance Metrics for Vision Models
Implements specialized distance computations for vision model verification.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings

# Optional imports with fallbacks
try:
    from scipy.stats import wasserstein_distance
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import ot  # POT (Python Optimal Transport)
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False


class VisionDistanceMetrics:
    """Distance metrics for vision model outputs."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'epsilon': 1e-10,  # Small constant to avoid numerical issues
            'temperature': 1.0,  # Temperature for softmax
            'normalize_features': True,  # Whether to normalize features
            'batch_size': 1000,  # Batch size for large computations
        }
        
    def compute_logit_distance(self, 
                              logits1: torch.Tensor,
                              logits2: torch.Tensor,
                              metric: str = 'kl',
                              temperature: Optional[float] = None) -> float:
        """
        Compute distance between logit distributions.
        Args:
            logits1: First set of logits
            logits2: Second set of logits
            metric: 'kl', 'js', 'wasserstein', 'l2', 'cross_entropy'
            temperature: Temperature for softmax (uses config default if None)
        Returns:
            Distance value
        """
        temp = temperature or self.config.get('temperature', 1.0)
        
        # Apply temperature scaling
        scaled_logits1 = logits1 / temp
        scaled_logits2 = logits2 / temp
        
        # Convert to probabilities
        probs1 = torch.softmax(scaled_logits1, dim=-1)
        probs2 = torch.softmax(scaled_logits2, dim=-1)
        
        if metric == 'kl':
            return self._kl_divergence(probs1, probs2)
        elif metric == 'js':
            return self._js_divergence(probs1, probs2)
        elif metric == 'wasserstein':
            return self._wasserstein_distance(probs1, probs2)
        elif metric == 'l2':
            return torch.norm(logits1 - logits2, p=2).item()
        elif metric == 'l1':
            return torch.norm(logits1 - logits2, p=1).item()
        elif metric == 'cross_entropy':
            return self._cross_entropy_distance(probs1, probs2)
        elif metric == 'cosine':
            return self._cosine_distance(logits1, logits2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def compute_embedding_distance(self,
                                  emb1: torch.Tensor,
                                  emb2: torch.Tensor,
                                  metric: str = 'cosine') -> float:
        """
        Compute distance between embeddings.
        Args:
            emb1: First embedding tensor
            emb2: Second embedding tensor
            metric: 'cosine', 'euclidean', 'manhattan', 'chebyshev', 'mahalanobis'
        Returns:
            Distance value
        """
        # Ensure same shape
        if emb1.shape != emb2.shape:
            min_dim = min(emb1.shape[-1], emb2.shape[-1])
            emb1 = emb1[..., :min_dim]
            emb2 = emb2[..., :min_dim]
        
        # Normalize if configured
        if self.config.get('normalize_features', True):
            emb1 = nn.functional.normalize(emb1, p=2, dim=-1)
            emb2 = nn.functional.normalize(emb2, p=2, dim=-1)
        
        if metric == 'cosine':
            sim = nn.functional.cosine_similarity(emb1, emb2, dim=-1)
            return (1 - sim).mean().item()
        elif metric == 'euclidean':
            return torch.norm(emb1 - emb2, p=2, dim=-1).mean().item()
        elif metric == 'manhattan':
            return torch.norm(emb1 - emb2, p=1, dim=-1).mean().item()
        elif metric == 'chebyshev':
            return torch.max(torch.abs(emb1 - emb2), dim=-1)[0].mean().item()
        elif metric == 'mahalanobis':
            return self._mahalanobis_distance(emb1, emb2)
        elif metric == 'correlation':
            return self._correlation_distance(emb1, emb2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def compute_structural_distance(self,
                                   features1: Dict[str, torch.Tensor],
                                   features2: Dict[str, torch.Tensor],
                                   method: str = 'cka') -> float:
        """
        Compute structural similarity between feature hierarchies.
        Args:
            features1: First set of features
            features2: Second set of features
            method: 'cka', 'pwcca', 'linear_cka'
        Returns:
            Structural distance (0 = identical, 1 = completely different)
        """
        distances = []
        weights = self._get_layer_weights()
        
        for layer_name in features1.keys():
            if layer_name in features2:
                f1 = features1[layer_name]
                f2 = features2[layer_name]
                
                # Compute similarity based on method
                if method == 'cka':
                    similarity = self._compute_cka(f1, f2)
                elif method == 'linear_cka':
                    similarity = self._compute_linear_cka(f1, f2)
                elif method == 'pwcca':
                    similarity = self._compute_pwcca(f1, f2)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Weight by layer importance
                weight = weights.get(layer_name, 1.0)
                distances.append((1 - similarity) * weight)
        
        return np.mean(distances) if distances else 1.0
    
    def compute_multi_scale_distance(self,
                                    features1: Dict[str, torch.Tensor],
                                    features2: Dict[str, torch.Tensor],
                                    scales: List[str] = None) -> Dict[str, float]:
        """
        Compute distances at multiple scales.
        """
        scales = scales or ['early', 'mid', 'late', 'final']
        distances = {}
        
        for scale in scales:
            if scale in features1 and scale in features2:
                f1 = features1[scale]
                f2 = features2[scale]
                
                # Multiple distance metrics for robustness
                distances[scale] = {
                    'cka': self._compute_cka(f1, f2),
                    'cosine': 1 - nn.functional.cosine_similarity(
                        f1.flatten(), f2.flatten(), dim=0
                    ).item(),
                    'l2': torch.norm(f1 - f2, p=2).item(),
                }
        
        return distances
    
    def _kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """KL divergence KL(p||q)."""
        eps = self.config.get('epsilon', 1e-10)
        p = p + eps  # Avoid log(0)
        q = q + eps
        
        kl = (p * torch.log(p / q)).sum(dim=-1)
        return kl.mean().item()
    
    def _js_divergence(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """Jensen-Shannon divergence."""
        m = 0.5 * (p + q)
        return 0.5 * self._kl_divergence(p, m) + 0.5 * self._kl_divergence(q, m)
    
    def _wasserstein_distance(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """Wasserstein distance between distributions."""
        if not SCIPY_AVAILABLE:
            warnings.warn("SciPy not available, using L2 distance as fallback")
            return torch.norm(p - q, p=2).item()
        
        p_np = p.cpu().numpy()
        q_np = q.cpu().numpy()
        
        distances = []
        for i in range(p_np.shape[0]):
            try:
                dist = wasserstein_distance(
                    np.arange(len(p_np[i])), 
                    np.arange(len(q_np[i])),
                    p_np[i], 
                    q_np[i]
                )
                distances.append(dist)
            except:
                # Fallback to L2 distance for this sample
                distances.append(np.linalg.norm(p_np[i] - q_np[i]))
            
        return np.mean(distances)
    
    def _cross_entropy_distance(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """Cross-entropy based distance."""
        eps = self.config.get('epsilon', 1e-10)
        q = q + eps
        
        # Average of both directions
        ce1 = -(p * torch.log(q)).sum(dim=-1).mean()
        ce2 = -(q * torch.log(p + eps)).sum(dim=-1).mean()
        
        return 0.5 * (ce1 + ce2).item()
    
    def _cosine_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """Cosine distance between tensors."""
        cos_sim = nn.functional.cosine_similarity(x1.flatten(), x2.flatten(), dim=0)
        return (1 - cos_sim).item()
    
    def _mahalanobis_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """Mahalanobis distance between embeddings."""
        # Combine embeddings to estimate covariance
        combined = torch.cat([x1, x2], dim=0)
        
        # Compute covariance matrix
        combined_centered = combined - combined.mean(dim=0, keepdim=True)
        cov = torch.mm(combined_centered.t(), combined_centered) / (combined.shape[0] - 1)
        
        # Add regularization for numerical stability
        reg = 1e-6 * torch.eye(cov.shape[0], device=cov.device)
        cov_reg = cov + reg
        
        try:
            # Compute inverse
            cov_inv = torch.inverse(cov_reg)
            
            # Compute Mahalanobis distance
            diff = x1.mean(dim=0) - x2.mean(dim=0)
            mahal_dist = torch.sqrt(torch.mm(torch.mm(diff.unsqueeze(0), cov_inv), diff.unsqueeze(1)))
            
            return mahal_dist.item()
        except:
            # Fallback to Euclidean distance
            return torch.norm(x1.mean(dim=0) - x2.mean(dim=0), p=2).item()
    
    def _correlation_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """Correlation-based distance."""
        # Flatten if needed
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        if x2.dim() > 2:
            x2 = x2.reshape(x2.shape[0], -1)
        
        # Compute correlation coefficient
        x1_centered = x1 - x1.mean(dim=0, keepdim=True)
        x2_centered = x2 - x2.mean(dim=0, keepdim=True)
        
        numerator = (x1_centered * x2_centered).sum(dim=0)
        denominator = torch.sqrt((x1_centered**2).sum(dim=0) * (x2_centered**2).sum(dim=0))
        
        correlation = numerator / (denominator + self.config.get('epsilon', 1e-10))
        
        # Return distance (1 - correlation)
        return (1 - correlation.mean()).item()
    
    def _compute_cka(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Compute Centered Kernel Alignment (CKA).
        Measures similarity between representations.
        """
        # Flatten features if needed
        if X.dim() > 2:
            X = X.reshape(X.shape[0], -1)
        if Y.dim() > 2:
            Y = Y.reshape(Y.shape[0], -1)
        
        # Center the data
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        # Compute Gram matrices
        K_XX = torch.mm(X, X.t())
        K_YY = torch.mm(Y, Y.t())
        K_XY = torch.mm(X, Y.t())
        
        # Compute CKA
        numerator = torch.norm(K_XY, 'fro') ** 2
        denominator = torch.norm(K_XX, 'fro') * torch.norm(K_YY, 'fro')
        
        eps = self.config.get('epsilon', 1e-10)
        cka = numerator / (denominator + eps)
        
        return cka.item()
    
    def _compute_linear_cka(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Linear CKA (more efficient for large features)."""
        # Flatten features if needed
        if X.dim() > 2:
            X = X.reshape(X.shape[0], -1)
        if Y.dim() > 2:
            Y = Y.reshape(Y.shape[0], -1)
        
        # Center the data
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        # Linear CKA
        cov_XY = torch.mm(X.t(), Y)
        cov_XX = torch.mm(X.t(), X)
        cov_YY = torch.mm(Y.t(), Y)
        
        numerator = torch.norm(cov_XY, 'fro') ** 2
        denominator = torch.norm(cov_XX, 'fro') * torch.norm(cov_YY, 'fro')
        
        eps = self.config.get('epsilon', 1e-10)
        linear_cka = numerator / (denominator + eps)
        
        return linear_cka.item()
    
    def _compute_pwcca(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Projection Weighted Canonical Correlation Analysis."""
        # Flatten features if needed
        if X.dim() > 2:
            X = X.reshape(X.shape[0], -1)
        if Y.dim() > 2:
            Y = Y.reshape(Y.shape[0], -1)
        
        # Center the data
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        # Compute SVD for both matrices
        try:
            U_X, S_X, V_X = torch.svd(X)
            U_Y, S_Y, V_Y = torch.svd(Y)
            
            # Use top components (regularization)
            k = min(min(X.shape), min(Y.shape), 50)  # Limit to top 50 components
            
            U_X = U_X[:, :k]
            U_Y = U_Y[:, :k]
            
            # Compute cross-correlation
            cross_corr = torch.mm(U_X.t(), U_Y)
            U_corr, S_corr, V_corr = torch.svd(cross_corr)
            
            # PWCCA score
            pwcca = S_corr.sum().item() / k
            
            return pwcca
            
        except:
            # Fallback to linear CKA
            return self._compute_linear_cka(X, Y)
    
    def _get_layer_weights(self) -> Dict[str, float]:
        """Get importance weights for different layers."""
        return {
            'early': 0.1,
            'early_mid': 0.2,
            'mid': 0.3,
            'late': 0.4,
            'penultimate': 0.7,
            'final': 1.0,
            'patch_embed': 0.2,
            'conv': 0.3,
            'classifier': 1.0,
            'head': 1.0
        }


class AdvancedDistanceMetrics:
    """Advanced distance metrics for vision verification."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'ot_reg': 0.1,  # Regularization for optimal transport
            'mmd_kernel': 'rbf',  # Kernel for MMD
            'bandwidth': 'median',  # Bandwidth selection
        }
    
    def compute_ot_distance(self,
                           features1: torch.Tensor,
                           features2: torch.Tensor,
                           method: str = 'emd') -> float:
        """
        Compute Optimal Transport distance.
        Args:
            features1: First feature tensor
            features2: Second feature tensor  
            method: 'emd' (exact), 'sinkhorn' (regularized)
        Returns:
            Optimal transport distance
        """
        if not OT_AVAILABLE:
            warnings.warn("POT library not available, using Wasserstein fallback")
            return self._wasserstein_fallback(features1, features2)
        
        # Flatten features
        f1 = features1.cpu().numpy()
        f2 = features2.cpu().numpy()
        
        if f1.ndim > 2:
            f1 = f1.reshape(f1.shape[0], -1)
        if f2.ndim > 2:
            f2 = f2.reshape(f2.shape[0], -1)
        
        # Subsample if too large
        max_samples = 1000
        if f1.shape[0] > max_samples:
            idx1 = np.random.choice(f1.shape[0], max_samples, replace=False)
            f1 = f1[idx1]
        if f2.shape[0] > max_samples:
            idx2 = np.random.choice(f2.shape[0], max_samples, replace=False)
            f2 = f2[idx2]
        
        # Compute cost matrix
        M = ot.dist(f1, f2, metric='euclidean')
        
        # Uniform weights
        a = np.ones(f1.shape[0]) / f1.shape[0]
        b = np.ones(f2.shape[0]) / f2.shape[0]
        
        try:
            if method == 'emd':
                # Exact optimal transport
                distance = ot.emd2(a, b, M)
            elif method == 'sinkhorn':
                # Regularized optimal transport
                reg = self.config.get('ot_reg', 0.1)
                distance = ot.sinkhorn2(a, b, M, reg)
            else:
                raise ValueError(f"Unknown OT method: {method}")
            
            return float(distance)
            
        except Exception as e:
            warnings.warn(f"OT computation failed: {e}, using fallback")
            return self._wasserstein_fallback(features1, features2)
    
    def compute_mmd(self,
                   X: torch.Tensor,
                   Y: torch.Tensor,
                   kernel: str = 'rbf',
                   bandwidth: Union[float, str] = 'median') -> float:
        """
        Compute Maximum Mean Discrepancy (MMD).
        Args:
            X: First sample tensor
            Y: Second sample tensor
            kernel: 'rbf', 'linear', 'polynomial'
            bandwidth: Bandwidth for RBF kernel ('median' for heuristic)
        Returns:
            MMD distance
        """
        # Flatten if needed
        if X.dim() > 2:
            X = X.reshape(X.shape[0], -1)
        if Y.dim() > 2:
            Y = Y.reshape(Y.shape[0], -1)
        
        n_x, n_y = X.shape[0], Y.shape[0]
        
        # Subsample if too large
        max_samples = 1000
        if n_x > max_samples:
            idx = torch.randperm(n_x)[:max_samples]
            X = X[idx]
            n_x = max_samples
        if n_y > max_samples:
            idx = torch.randperm(n_y)[:max_samples]
            Y = Y[idx]
            n_y = max_samples
        
        if kernel == 'rbf':
            if bandwidth == 'median':
                gamma = 1.0 / (2 * self._median_heuristic(X, Y)**2)
            else:
                gamma = 1.0 / (2 * bandwidth**2)
            
            # RBF kernel matrices
            K_XX = self._rbf_kernel(X, X, gamma)
            K_YY = self._rbf_kernel(Y, Y, gamma)
            K_XY = self._rbf_kernel(X, Y, gamma)
            
        elif kernel == 'linear':
            # Linear kernel matrices
            K_XX = torch.mm(X, X.t())
            K_YY = torch.mm(Y, Y.t())
            K_XY = torch.mm(X, Y.t())
            
        elif kernel == 'polynomial':
            # Polynomial kernel (degree 2)
            K_XX = (torch.mm(X, X.t()) + 1) ** 2
            K_YY = (torch.mm(Y, Y.t()) + 1) ** 2
            K_XY = (torch.mm(X, Y.t()) + 1) ** 2
            
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
        
        # Compute MMD^2
        mmd_squared = (K_XX.sum() / (n_x * n_x) + 
                      K_YY.sum() / (n_y * n_y) - 
                      2 * K_XY.sum() / (n_x * n_y))
        
        # Take square root and ensure non-negative
        mmd = torch.sqrt(torch.clamp(mmd_squared, min=0))
        
        return mmd.item()
    
    def compute_fid_distance(self,
                            features1: torch.Tensor,
                            features2: torch.Tensor) -> float:
        """
        Compute Fréchet Inception Distance (FID).
        Assumes features are from a pretrained network like Inception.
        """
        # Flatten features
        if features1.dim() > 2:
            features1 = features1.reshape(features1.shape[0], -1)
        if features2.dim() > 2:
            features2 = features2.reshape(features2.shape[0], -1)
        
        # Compute means and covariances
        mu1 = features1.mean(dim=0)
        mu2 = features2.mean(dim=0)
        
        sigma1 = torch.cov(features1.t())
        sigma2 = torch.cov(features2.t())
        
        # Compute FID
        diff = mu1 - mu2
        
        # Compute matrix square root of sigma1 * sigma2
        try:
            # Use SVD for numerical stability
            U1, S1, V1 = torch.svd(sigma1)
            U2, S2, V2 = torch.svd(sigma2)
            
            sqrt_sigma1 = U1 @ torch.diag(torch.sqrt(S1)) @ V1.t()
            sqrt_sigma2 = U2 @ torch.diag(torch.sqrt(S2)) @ V2.t()
            
            product = sqrt_sigma1 @ sqrt_sigma2
            U_prod, S_prod, V_prod = torch.svd(product)
            sqrt_product = U_prod @ torch.diag(torch.sqrt(S_prod)) @ V_prod.t()
            
            fid = (diff @ diff + torch.trace(sigma1) + torch.trace(sigma2) - 
                   2 * torch.trace(sqrt_product))
            
            return fid.item()
            
        except Exception as e:
            warnings.warn(f"FID computation failed: {e}, using simplified version")
            # Simplified FID without matrix square root
            fid_simple = (diff @ diff + torch.trace(sigma1) + torch.trace(sigma2))
            return fid_simple.item()
    
    def compute_kid_distance(self,
                            features1: torch.Tensor,
                            features2: torch.Tensor,
                            subset_size: int = 100) -> float:
        """
        Compute Kernel Inception Distance (KID).
        More robust than FID for small sample sizes.
        """
        # Flatten features
        if features1.dim() > 2:
            features1 = features1.reshape(features1.shape[0], -1)
        if features2.dim() > 2:
            features2 = features2.reshape(features2.shape[0], -1)
        
        n1, n2 = features1.shape[0], features2.shape[0]
        
        # Subsample if needed
        if n1 > subset_size:
            idx1 = torch.randperm(n1)[:subset_size]
            features1 = features1[idx1]
        if n2 > subset_size:
            idx2 = torch.randperm(n2)[:subset_size]
            features2 = features2[idx2]
        
        # Compute polynomial kernel (degree 3)
        def poly_kernel(x, y, degree=3, gamma=1.0, coef0=1.0):
            return (gamma * torch.mm(x, y.t()) + coef0) ** degree
        
        # Kernel matrices
        K_XX = poly_kernel(features1, features1)
        K_YY = poly_kernel(features2, features2)
        K_XY = poly_kernel(features1, features2)
        
        n_x, n_y = K_XX.shape[0], K_YY.shape[0]
        
        # Remove diagonal elements for unbiased estimation
        K_XX.fill_diagonal_(0)
        K_YY.fill_diagonal_(0)
        
        # Compute KID
        kid = (K_XX.sum() / (n_x * (n_x - 1)) + 
               K_YY.sum() / (n_y * (n_y - 1)) - 
               2 * K_XY.sum() / (n_x * n_y))
        
        return kid.item()
    
    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor, gamma: float) -> torch.Tensor:
        """Compute RBF kernel matrix."""
        X_sqnorms = torch.sum(X**2, dim=1, keepdim=True)
        Y_sqnorms = torch.sum(Y**2, dim=1, keepdim=True)
        
        distances_squared = (X_sqnorms + Y_sqnorms.t() - 2 * torch.mm(X, Y.t()))
        
        return torch.exp(-gamma * distances_squared)
    
    def _median_heuristic(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Median heuristic for RBF kernel bandwidth."""
        with torch.no_grad():
            # Subsample for efficiency
            max_samples = 500
            if X.shape[0] > max_samples:
                idx = torch.randperm(X.shape[0])[:max_samples]
                X = X[idx]
            if Y.shape[0] > max_samples:
                idx = torch.randperm(Y.shape[0])[:max_samples]
                Y = Y[idx]
            
            Z = torch.cat([X, Y], dim=0)
            
            # Compute pairwise distances
            Z_sqnorms = torch.sum(Z**2, dim=1, keepdim=True)
            distances_squared = Z_sqnorms + Z_sqnorms.t() - 2 * torch.mm(Z, Z.t())
            distances = torch.sqrt(torch.clamp(distances_squared, min=0))
            
            # Get median (excluding diagonal)
            mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
            median = torch.median(distances[mask])
            
        return median.item()
    
    def _wasserstein_fallback(self, features1: torch.Tensor, features2: torch.Tensor) -> float:
        """Fallback Wasserstein distance computation."""
        # Simple 1D Wasserstein on feature means
        f1_mean = features1.mean(dim=0)
        f2_mean = features2.mean(dim=0)
        
        # Sort the means and compute L1 distance
        f1_sorted, _ = torch.sort(f1_mean)
        f2_sorted, _ = torch.sort(f2_mean)
        
        return torch.norm(f1_sorted - f2_sorted, p=1).item()


# Factory function for easy access
def create_distance_metrics(metric_type: str = 'standard', config: Optional[Dict] = None):
    """Factory function to create distance metrics."""
    if metric_type == 'standard':
        return VisionDistanceMetrics(config)
    elif metric_type == 'advanced':
        return AdvancedDistanceMetrics(config)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


# Utility functions
def batch_distance_computation(features_list1: List[torch.Tensor],
                              features_list2: List[torch.Tensor],
                              metric: str = 'cosine',
                              batch_size: int = 100) -> List[float]:
    """Compute distances in batches for memory efficiency."""
    metrics = VisionDistanceMetrics()
    distances = []
    
    for i in range(0, len(features_list1), batch_size):
        batch1 = features_list1[i:i+batch_size]
        batch2 = features_list2[i:i+batch_size]
        
        for f1, f2 in zip(batch1, batch2):
            if metric in ['cosine', 'euclidean', 'manhattan']:
                dist = metrics.compute_embedding_distance(f1, f2, metric)
            else:
                dist = metrics.compute_logit_distance(f1, f2, metric)
            distances.append(dist)
    
    return distances


def compute_distance_matrix(features_list: List[torch.Tensor],
                           metric: str = 'cosine') -> np.ndarray:
    """Compute pairwise distance matrix."""
    n = len(features_list)
    distance_matrix = np.zeros((n, n))
    
    metrics = VisionDistanceMetrics()
    
    for i in range(n):
        for j in range(i+1, n):
            if metric in ['cosine', 'euclidean', 'manhattan']:
                dist = metrics.compute_embedding_distance(
                    features_list[i], features_list[j], metric)
            else:
                dist = metrics.compute_logit_distance(
                    features_list[i], features_list[j], metric)
            
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric
    
    return distance_matrix


def analyze_distance_distribution(distances: List[float]) -> Dict[str, float]:
    """Analyze the distribution of distances."""
    distances_array = np.array(distances)
    
    return {
        'mean': float(np.mean(distances_array)),
        'std': float(np.std(distances_array)),
        'min': float(np.min(distances_array)),
        'max': float(np.max(distances_array)),
        'median': float(np.median(distances_array)),
        'q25': float(np.percentile(distances_array, 25)),
        'q75': float(np.percentile(distances_array, 75)),
    }