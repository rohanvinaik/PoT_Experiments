import numpy as np
from typing import Tuple

def t_statistic(distances: np.ndarray) -> float:
    return float(np.mean(distances))

def empirical_bernstein_radius(distances: np.ndarray, delta: float) -> float:
    # Maurer & Pontil style bound
    n = len(distances)
    if n <= 1: return float("inf")
    mean = np.mean(distances)
    var = np.var(distances, ddof=1)
    from math import log, sqrt
    r = np.sqrt(2*var*log(3/delta)/n) + 3*log(3/delta)/n
    return r

def far_frr(distances_h0: np.ndarray, distances_h1: np.ndarray, tau: float) -> Tuple[float,float]:
    far = float(np.mean(distances_h1 <= tau))
    frr = float(np.mean(distances_h0 > tau))
    return far, frr