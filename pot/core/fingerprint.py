import numpy as np
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Union
import torch
from .canonicalize import canonicalize_logits, canonicalize_text


@dataclass
class FingerprintResult:
    """Result of fingerprinting a model on a set of challenges.
    
    Attributes:
        io_hash: Stable hash of canonicalized input-output pairs
        jacobian_sketch: Optional compressed Jacobian representation (e.g., sign-pattern hash)
        raw_outputs: Canonicalized model outputs for further analysis
        timing_info: Execution time per challenge in seconds
        metadata: Additional metadata (model type, layers analyzed, etc.)
    """
    io_hash: str
    jacobian_sketch: Optional[str] = None
    raw_outputs: List[Any] = field(default_factory=list)
    timing_info: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


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


def jacobian_sign_hash(jacobian: np.ndarray, threshold: float = 1e-6) -> str:
    """Compute a hash of the Jacobian sign pattern.
    
    Args:
        jacobian: Jacobian matrix of shape [input_dim, output_dim]
        threshold: Values below this are considered zero
        
    Returns:
        Hexadecimal hash of the sign pattern
    """
    # Reference: §2.2 - compressed Jacobian representation
    sign_pattern = np.sign(jacobian)
    sign_pattern[np.abs(jacobian) < threshold] = 0
    
    # Convert to compact binary representation: -1 -> 00, 0 -> 01, 1 -> 10
    mapping = {-1: '00', 0: '01', 1: '10'}
    binary_str = ''.join(mapping[int(s)] for s in sign_pattern.flatten())
    
    # Hash the binary pattern
    return hashlib.sha256(binary_str.encode()).hexdigest()[:16]


def fingerprint_run(f: Callable, challenges: List[Any], cfg: Optional[Dict[str, Any]] = None) -> FingerprintResult:
    """Capture model behavior through input-output mappings and optional Jacobian analysis.
    
    Reference: §2.2 System Architecture - Behavioral Fingerprinter
    
    Args:
        f: Model function that accepts challenges and returns outputs.
           For vision models: f(image_tensor) -> logits/embeddings
           For language models: f(text_prompt) -> generated_text or logits
        challenges: List of challenge inputs (images, text prompts, tensors)
        cfg: Optional configuration dictionary with keys:
            - 'compute_jacobian': bool, whether to compute Jacobian (default: False)
            - 'jacobian_layers': List[str], specific layers for Jacobian computation
            - 'jacobian_delta': float, finite difference delta (default: 1e-3)
            - 'jacobian_max_dim': int, max dimensions for Jacobian (default: 256)
            - 'model_type': str, 'vision' or 'lm' (auto-detected if not specified)
            - 'canonicalize_precision': int, precision for numeric canonicalization (default: 6)
            - 'text_max_len': int, max length for text canonicalization (default: 512)
            
    Returns:
        FingerprintResult containing io_hash, optional jacobian_sketch, 
        canonicalized outputs, and timing information
    """
    cfg = cfg or {}
    
    # Extract configuration
    compute_jacobian = cfg.get('compute_jacobian', False)
    jacobian_layers = cfg.get('jacobian_layers', None)
    jacobian_delta = cfg.get('jacobian_delta', 1e-3)
    jacobian_max_dim = cfg.get('jacobian_max_dim', 256)
    model_type = cfg.get('model_type', None)
    precision = cfg.get('canonicalize_precision', 6)
    text_max_len = cfg.get('text_max_len', 512)
    
    raw_outputs = []
    canonicalized_outputs = []
    timing_info = []
    jacobian_sketches = []
    
    for challenge in challenges:
        start_time = time.perf_counter()
        
        try:
            # Execute model on challenge
            if isinstance(challenge, (np.ndarray, torch.Tensor)):
                # Handle tensor inputs (vision models)
                if isinstance(challenge, np.ndarray):
                    challenge_input = torch.from_numpy(challenge).float()
                else:
                    challenge_input = challenge
                    
                # Ensure proper dimensions (add batch dim if needed)
                if challenge_input.dim() == 3:  # [C, H, W]
                    challenge_input = challenge_input.unsqueeze(0)  # [1, C, H, W]
                    
                output = f(challenge_input)
                
                # Convert output to numpy if needed
                if isinstance(output, torch.Tensor):
                    output = output.detach().cpu().numpy()
                    
            elif isinstance(challenge, str):
                # Handle text inputs (language models)
                output = f(challenge)
                model_type = model_type or 'lm'
                
            elif isinstance(challenge, dict):
                # Handle structured challenges (e.g., with metadata)
                if 'input' in challenge:
                    output = f(challenge['input'])
                else:
                    output = f(challenge)
            else:
                # Generic case
                output = f(challenge)
                
        except Exception as e:
            # Record error but continue with other challenges
            output = f"ERROR: {str(e)}"
            timing_info.append(-1.0)
            raw_outputs.append(output)
            canonicalized_outputs.append(output)
            continue
            
        elapsed = time.perf_counter() - start_time
        timing_info.append(elapsed)
        raw_outputs.append(output)
        
        # Canonicalize output based on type
        if isinstance(output, str):
            # Text output from language models
            canonical = canonicalize_text(output, max_len=text_max_len)
            canonicalized_outputs.append(canonical)
            
        elif isinstance(output, (np.ndarray, torch.Tensor)):
            # Numeric output (logits, embeddings)
            if isinstance(output, torch.Tensor):
                output_np = output.detach().cpu().numpy()
            else:
                output_np = output
                
            # Apply canonicalization for numeric stability
            canonical = canonicalize_logits(output_np, p=precision)
            canonicalized_outputs.append(canonical)
            
        elif isinstance(output, (list, tuple)):
            # Handle structured outputs (e.g., multiple logits)
            canonical = []
            for item in output:
                if isinstance(item, str):
                    canonical.append(canonicalize_text(item, max_len=text_max_len))
                elif isinstance(item, (np.ndarray, torch.Tensor)):
                    if isinstance(item, torch.Tensor):
                        item = item.detach().cpu().numpy()
                    canonical.append(canonicalize_logits(item, p=precision))
                else:
                    canonical.append(item)
            canonicalized_outputs.append(canonical)
            
        else:
            # For other types, use as-is
            canonicalized_outputs.append(output)
            
        # Compute Jacobian if requested and applicable
        if compute_jacobian and isinstance(challenge, (np.ndarray, torch.Tensor)):
            try:
                # Create a wrapper for finite difference computation
                def model_wrapper(x):
                    if isinstance(x, np.ndarray):
                        x_tensor = torch.from_numpy(x).float()
                    else:
                        x_tensor = x
                    out = f(x_tensor)
                    if isinstance(out, torch.Tensor):
                        return out.detach().cpu().numpy()
                    return out
                    
                # Compute Jacobian using finite differences
                if isinstance(challenge, torch.Tensor):
                    challenge_np = challenge.detach().cpu().numpy()
                else:
                    challenge_np = challenge
                    
                if challenge_np.ndim == 4:  # batch dimension
                    challenge_np = challenge_np[0]  # Take first sample
                    
                # Flatten for Jacobian computation
                challenge_flat = challenge_np.flatten()
                jacobian = finite_diff_jacobian(
                    lambda x: model_wrapper(x.reshape(challenge_np.shape)).flatten(),
                    challenge_flat,
                    delta=jacobian_delta,
                    max_dim=jacobian_max_dim
                )
                
                # Compute sign-pattern hash of Jacobian
                j_hash = jacobian_sign_hash(jacobian)
                jacobian_sketches.append(j_hash)
                
            except Exception as e:
                # Jacobian computation failed, continue without it
                pass
    
    # Compute stable IO hash from canonicalized outputs
    io_hash_value = io_hash(canonicalized_outputs, precision=precision)
    
    # Combine Jacobian sketches if computed
    jacobian_sketch = None
    if jacobian_sketches:
        # Combine individual sketches into a single hash
        combined = ''.join(jacobian_sketches)
        jacobian_sketch = hashlib.sha256(combined.encode()).hexdigest()[:32]
    
    # Auto-detect model type if not specified
    if model_type is None:
        if all(isinstance(out, str) for out in raw_outputs if not isinstance(out, str) or not out.startswith("ERROR")):
            model_type = 'lm'
        else:
            model_type = 'vision'
    
    # Prepare metadata
    metadata = {
        'model_type': model_type,
        'num_challenges': len(challenges),
        'compute_jacobian': compute_jacobian,
        'avg_time_per_challenge': np.mean([t for t in timing_info if t > 0]) if timing_info else 0,
        'total_time': sum(t for t in timing_info if t > 0),
        'errors': sum(1 for t in timing_info if t < 0),
        'precision': precision
    }
    
    if jacobian_layers:
        metadata['jacobian_layers'] = jacobian_layers
        
    return FingerprintResult(
        io_hash=io_hash_value,
        jacobian_sketch=jacobian_sketch,
        raw_outputs=canonicalized_outputs,
        timing_info=timing_info,
        metadata=metadata
    )