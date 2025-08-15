"""
Utility functions for determinism, checksums, and other common operations
"""

import os
import hashlib
import random
import time
import numpy as np
from contextlib import contextmanager

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def set_reproducibility(seed: int = 42):
    """Legacy function name - redirects to set_deterministic"""
    set_deterministic(seed)


def set_deterministic(seed: int = 0):
    """
    Set all random seeds for deterministic behavior
    
    Args:
        seed: Random seed to use
    """
    # Python random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch if available
    if HAS_TORCH and torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Set deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Disable cudnn benchmarking for determinism
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


@contextmanager
def timer(name: str = "Operation"):
    start = time.time()
    yield
    print(f"{name} took {time.time() - start:.4f} seconds")


def sha256_bytes(b: bytes) -> str:
    """
    Compute SHA256 hash of bytes
    
    Args:
        b: Bytes to hash
    
    Returns:
        Hex string of hash
    """
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    """
    Compute SHA256 hash of file
    
    Args:
        path: Path to file
    
    Returns:
        Hex string of hash
    """
    h = hashlib.sha256()
    
    with open(path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    
    return h.hexdigest()


def sha256_model(model) -> str:
    """
    Compute SHA256 hash of model weights
    
    Args:
        model: PyTorch model or dict of parameters
    
    Returns:
        Hex string of hash
    """
    if HAS_TORCH and torch is not None:
        if hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
        else:
            state_dict = model
        
        # Sort keys for deterministic ordering
        sorted_keys = sorted(state_dict.keys())
        
        h = hashlib.sha256()
        for key in sorted_keys:
            param = state_dict[key]
            if isinstance(param, torch.Tensor):
                # Convert to numpy for consistent hashing
                param_bytes = param.detach().cpu().numpy().tobytes()
            else:
                param_bytes = str(param).encode('utf-8')
            
            h.update(key.encode('utf-8'))
            h.update(param_bytes)
        
        return h.hexdigest()
    else:
        # Fallback for non-torch models
        return sha256_bytes(str(model).encode('utf-8'))


def get_reproducibility_info(seed: int = None) -> dict:
    """
    Get information about reproducibility settings
    
    Args:
        seed: Random seed being used (if any)
    
    Returns:
        Dict with reproducibility information
    """
    info = {
        'seed': seed,
        'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
        'numpy_version': np.__version__,
    }
    
    if HAS_TORCH and torch is not None:
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        info['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled() if hasattr(torch, 'are_deterministic_algorithms_enabled') else None
        
        if hasattr(torch.backends, 'cudnn'):
            info['cudnn_deterministic'] = torch.backends.cudnn.deterministic
            info['cudnn_benchmark'] = torch.backends.cudnn.benchmark
    
    return info


def ensure_dir(path: str):
    """
    Ensure directory exists, creating if necessary
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def normalize_path(path: str) -> str:
    """
    Normalize a file path
    
    Args:
        path: Path to normalize
    
    Returns:
        Normalized absolute path
    """
    return os.path.abspath(os.path.expanduser(path))