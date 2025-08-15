import numpy as np
import hashlib
from typing import List, Dict, Any, Callable

def io_hash(responses: List[Any], precision:int=6) -> str:
    # serialize then hash
    ser = repr([_round_if_float(r, precision) for r in responses]).encode()
    return hashlib.sha256(ser).hexdigest()

def _round_if_float(x, p):
    if isinstance(x, float): return round(x, p)
    if isinstance(x, np.ndarray): return np.round(x, p).tolist()
    return x

def finite_diff_jacobian(f: Callable[[np.ndarray], np.ndarray],
                         x: np.ndarray, delta: float = 1e-3, max_dim: int = 256):
    # central differences on a projection of dims
    d = min(x.shape[-1], max_dim)
    idx = np.arange(d)
    J = []
    for i in idx:
        e = np.zeros_like(x)
        e[..., i] = delta
        J.append((f(x + e) - f(x - e)) / (2*delta))
    J = np.stack(J, axis=0)            # [d, out_dim]
    return J

def quantize(arr: np.ndarray, p:int=4):
    return np.round(arr, p)