"""
JSON encoder for handling numpy and torch objects with atomic write support
"""

import json
import numpy as np
from pathlib import Path
from typing import Union, Any

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


def atomic_json_dump(obj, path: Union[str, Path], **kwargs):
    """
    Atomically dump object to JSON file to prevent corruption.
    
    Uses a temporary file and atomic rename to ensure the file is never 
    left in a partially written state, even if the process is interrupted.
    
    Args:
        obj: Object to serialize
        path: File path to write to
        **kwargs: Additional arguments for json.dumps
    
    Example:
        >>> data = {"key": "value", "array": [1, 2, 3]}
        >>> atomic_json_dump(data, "results.json")
    """
    path = Path(path)
    
    # Set defaults for robust JSON writing
    kwargs.setdefault('cls', NpTorchEncoder)
    kwargs.setdefault('indent', 2)
    kwargs.setdefault('ensure_ascii', False)
    
    # Write to temporary file first
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    
    try:
        # Create parent directories if needed
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file
        json_content = json.dumps(obj, **kwargs)
        tmp_path.write_text(json_content, encoding='utf-8')
        
        # Atomic rename - this is the critical operation
        tmp_path.replace(path)
        
    except Exception:
        # Clean up temporary file on any error
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def atomic_json_dumps(obj, **kwargs):
    """
    Safely dump object to JSON string with atomic guarantees.
    
    Args:
        obj: Object to serialize  
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string
    """
    kwargs.setdefault('cls', NpTorchEncoder)
    kwargs.setdefault('indent', 2)
    kwargs.setdefault('ensure_ascii', False)
    
    return json.dumps(obj, **kwargs)