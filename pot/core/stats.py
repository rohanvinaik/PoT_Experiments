import numpy as np
from typing import Tuple
from math import log, sqrt

def t_statistic(distances: np.ndarray) -> float:
    """Test statistic T(f, f*, C) as defined in paper Section 2.2"""
    return float(np.mean(distances))

def empirical_bernstein_bound(distances: np.ndarray, delta: float, B: float = None) -> float:
    """
    Empirical Bernstein bound from Theorem 1 (Paper Section 2.3)
    
    Returns the confidence radius r such that:
    |T - E[T]| <= r with probability at least 1-delta
    
    Based on Maurer & Pontil (2009) empirical Bernstein inequality.
    """
    n = len(distances)
    if n <= 1: 
        return float("inf")
    
    # Sample variance with Bessel's correction
    sigma_hat_sq = np.var(distances, ddof=1)
    
    # Bound on the range (if not provided, estimate from data)
    if B is None:
        B = np.max(distances) - np.min(distances)
    
    # Empirical Bernstein bound from paper
    variance_term = sqrt(2 * sigma_hat_sq * log(2/delta) / n)
    bias_term = 7 * B * log(2/delta) / (3 * (n - 1))
    
    return variance_term + bias_term

def empirical_bernstein_radius(distances: np.ndarray, delta: float) -> float:
    """Legacy function - calls empirical_bernstein_bound"""
    return empirical_bernstein_bound(distances, delta)

def far_frr(distances_h0: np.ndarray, distances_h1: np.ndarray, tau: float) -> Tuple[float,float]:
    """
    Calculate False Accept Rate (FAR) and False Reject Rate (FRR)
    
    FAR: Probability of accepting H1 (different model) when H0 is true
    FRR: Probability of rejecting H0 (same model) when H0 is true
    """
    far = float(np.mean(distances_h1 <= tau))
    frr = float(np.mean(distances_h0 > tau))
    return far, frr

def confidence_interval_bernstein(mean: float, var: float, n: int, delta: float, B: float = 1.0) -> Tuple[float, float]:
    """
    Compute confidence interval using empirical Bernstein bound
    
    Returns (lower, upper) bounds for the true mean
    """
    radius = sqrt(2 * var * log(2/delta) / n) + 7 * B * log(2/delta) / (3 * (n - 1))
    return (mean - radius, mean + radius)