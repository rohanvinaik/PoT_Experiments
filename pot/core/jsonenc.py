"""
JSON encoder for handling numpy and torch objects
"""

import json
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False
    import warnings
    warnings.warn("PyTorch not available; JSON encoding of tensors will be disabled.")


class NpTorchEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy and torch objects"""
    
    def default(self, o):
        # Handle numpy generic types
        if isinstance(o, (np.generic,)):
            return o.item()
        
        # Handle numpy arrays
        if isinstance(o, np.ndarray):
            return o.tolist()
        
        # Handle torch tensors if available
        if HAS_TORCH and torch is not None:
            if 'torch' in str(type(o)):
                try:
                    if hasattr(o, 'detach'):
                        return o.detach().cpu().numpy().tolist()
                    elif hasattr(o, 'tolist'):
                        return o.tolist()
                    else:
                        return str(o)
                except:
                    return str(o)
        
        # Handle other non-serializable objects
        try:
            return super().default(o)
        except TypeError:
            return str(o)


def safe_json_dump(obj, fp=None, **kwargs):
    """
    Safely dump object to JSON, handling numpy/torch objects
    
    Args:
        obj: Object to serialize
        fp: File pointer (if None, returns string)
        **kwargs: Additional arguments for json.dump/dumps
    
    Returns:
        JSON string if fp is None, otherwise None
    """
    kwargs['cls'] = NpTorchEncoder
    kwargs.setdefault('indent', 2)
    
    if fp is None:
        return json.dumps(obj, **kwargs)
    else:
        json.dump(obj, fp, **kwargs)


def safe_json_dumps(obj, **kwargs):
    """
    Safely dump object to JSON string
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
    
    Returns:
        JSON string
    """
    return safe_json_dump(obj, fp=None, **kwargs)