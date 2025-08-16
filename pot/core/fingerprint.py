import numpy as np
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Union
import torch
from .canonicalize import canonicalize_logits, canonicalize_text


@dataclass
class CanonicalConfig:
    """Configuration for canonicalization parameters.
    
    Attributes:
        float_precision: Decimal places for floating point rounding (default: 6)
        float_eps: Threshold below which values are zeroed (default: 1e-6)
        text_lower: Convert text to lowercase (default: True)
        text_strip_punct: Remove punctuation from text (default: True)
        text_collapse_ws: Collapse whitespace in text (default: True)
        text_max_len: Maximum text length (default: 512)
        embedding_dims: Max embedding dimensions to preserve (default: None = all)
        handle_nan: How to handle NaN values ('zero', 'remove', 'raise') (default: 'zero')
        handle_inf: How to handle Inf values ('clip', 'remove', 'raise') (default: 'clip')
        deterministic_ordering: Ensure deterministic ordering for collections (default: True)
    """
    float_precision: int = 6
    float_eps: float = 1e-6
    text_lower: bool = True
    text_strip_punct: bool = True
    text_collapse_ws: bool = True
    text_max_len: Optional[int] = 512
    embedding_dims: Optional[int] = None
    handle_nan: str = 'zero'
    handle_inf: str = 'clip'
    deterministic_ordering: bool = True


def canonicalize_model_output(output: Any, 
                             output_type: str = 'auto',
                             config: Optional[CanonicalConfig] = None) -> Any:
    """Canonicalize model output for reproducible comparisons.
    
    Args:
        output: Model output to canonicalize
        output_type: Type of output ('auto', 'text', 'logits', 'embeddings', 'mixed')
        config: Canonicalization configuration
        
    Returns:
        Canonicalized output preserving behavioral information
    """
    if config is None:
        config = CanonicalConfig()
    
    # Auto-detect output type if needed
    if output_type == 'auto':
        if isinstance(output, str):
            output_type = 'text'
        elif isinstance(output, (list, tuple)) and all(isinstance(x, str) for x in output):
            output_type = 'text'
        elif isinstance(output, (np.ndarray, torch.Tensor)):
            # Heuristic: embeddings typically have more dimensions
            if isinstance(output, torch.Tensor):
                output_np = output.detach().cpu().numpy()
            else:
                output_np = output
            
            # Check shape to distinguish logits from embeddings
            if output_np.ndim == 1 or (output_np.ndim == 2 and output_np.shape[-1] < 100):
                output_type = 'logits'
            else:
                output_type = 'embeddings'
        elif isinstance(output, dict):
            output_type = 'mixed'
        elif isinstance(output, (list, tuple)):
            output_type = 'mixed'
        else:
            # Default to treating as generic
            output_type = 'mixed'
    
    # Handle text outputs
    if output_type == 'text':
        if isinstance(output, str):
            return canonicalize_text(
                output,
                lower=config.text_lower,
                strip_punct=config.text_strip_punct,
                collapse_ws=config.text_collapse_ws,
                max_len=config.text_max_len
            )
        elif isinstance(output, (list, tuple)):
            return [canonicalize_text(
                text,
                lower=config.text_lower,
                strip_punct=config.text_strip_punct,
                collapse_ws=config.text_collapse_ws,
                max_len=config.text_max_len
            ) for text in output]
    
    # Handle logits/embeddings
    elif output_type in ['logits', 'embeddings']:
        # Convert to numpy if needed
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        elif not isinstance(output, np.ndarray):
            output = np.array(output)
        
        # Handle NaN values
        if np.any(np.isnan(output)):
            if config.handle_nan == 'zero':
                output = np.nan_to_num(output, nan=0.0)
            elif config.handle_nan == 'remove':
                # For embeddings, can't remove individual values, so zero them
                output = np.nan_to_num(output, nan=0.0)
            elif config.handle_nan == 'raise':
                raise ValueError("Output contains NaN values")
        
        # Handle Inf values
        if np.any(np.isinf(output)):
            if config.handle_inf == 'clip':
                # Clip to large but finite values
                # Use a safe value that won't overflow
                if output.dtype == np.float64:
                    max_val = 1e308
                    min_val = -1e308
                else:
                    max_val = 1e38
                    min_val = -1e38
                output = np.where(np.isinf(output), 
                                np.where(output > 0, max_val, min_val), 
                                output)
            elif config.handle_inf == 'remove':
                # For embeddings, can't remove, so clip
                if output.dtype == np.float64:
                    max_val = 1e308
                    min_val = -1e308
                else:
                    max_val = 1e38
                    min_val = -1e38
                output = np.where(np.isinf(output), 
                                np.where(output > 0, max_val, min_val), 
                                output)
            elif config.handle_inf == 'raise':
                raise ValueError("Output contains Inf values")
        
        # Apply dimension reduction for embeddings if specified
        if output_type == 'embeddings' and config.embedding_dims is not None:
            if output.ndim == 1:
                output = output[:config.embedding_dims]
            elif output.ndim == 2:
                output = output[:, :config.embedding_dims]
            elif output.ndim == 3:
                output = output[:, :, :config.embedding_dims]
        
        # Apply canonicalization
        return canonicalize_logits(output, p=config.float_precision, eps=config.float_eps)
    
    # Handle mixed/structured outputs
    elif output_type == 'mixed':
        if isinstance(output, dict):
            # Canonicalize each value, maintaining deterministic order
            if config.deterministic_ordering:
                keys = sorted(output.keys())
                return {k: canonicalize_model_output(output[k], 'auto', config) for k in keys}
            else:
                return {k: canonicalize_model_output(v, 'auto', config) for k, v in output.items()}
        
        elif isinstance(output, (list, tuple)):
            # Check if it's homogeneous or mixed
            canonicalized = []
            for item in output:
                canonicalized.append(canonicalize_model_output(item, 'auto', config))
            return canonicalized if isinstance(output, list) else tuple(canonicalized)
        
        else:
            # For unknown types, preserve numeric types but convert others to string
            if isinstance(output, (int, float, bool, type(None))):
                return output  # Keep primitive types as-is
            else:
                try:
                    return canonicalize_text(str(output), max_len=config.text_max_len)
                except:
                    return output
    
    # Default: return as-is
    return output


def canonicalize_batch_outputs(outputs: List[Any],
                              output_types: Optional[List[str]] = None,
                              config: Optional[CanonicalConfig] = None) -> List[Any]:
    """Process multiple outputs consistently with deterministic ordering.
    
    Args:
        outputs: List of model outputs to canonicalize
        output_types: Optional list of output types for each output
        config: Canonicalization configuration
        
    Returns:
        List of canonicalized outputs with consistent processing
    """
    if config is None:
        config = CanonicalConfig()
    
    if output_types is None:
        output_types = ['auto'] * len(outputs)
    elif len(output_types) != len(outputs):
        raise ValueError(f"output_types length ({len(output_types)}) must match outputs length ({len(outputs)})")
    
    canonicalized = []
    
    for output, out_type in zip(outputs, output_types):
        try:
            canonical = canonicalize_model_output(output, out_type, config)
            canonicalized.append(canonical)
        except Exception as e:
            # Handle errors gracefully - preserve original on error
            if isinstance(e, (ValueError, TypeError)) and 'NaN' not in str(e) and 'Inf' not in str(e):
                # Re-raise structural errors
                raise
            # For numeric errors, try to recover
            canonicalized.append(output)
    
    # Ensure deterministic ordering for collections of outputs
    if config.deterministic_ordering and len(canonicalized) > 0:
        # Check if outputs are sortable (all same type and comparable)
        try:
            if all(isinstance(x, type(canonicalized[0])) for x in canonicalized):
                if isinstance(canonicalized[0], (str, int, float)):
                    # Sort simple types directly
                    indices = sorted(range(len(canonicalized)), key=lambda i: canonicalized[i])
                    return [canonicalized[i] for i in indices]
                elif isinstance(canonicalized[0], np.ndarray):
                    # Sort by hash of array for determinism
                    indices = sorted(range(len(canonicalized)), 
                                   key=lambda i: hashlib.sha256(canonicalized[i].tobytes()).hexdigest())
                    return [canonicalized[i] for i in indices]
        except:
            # If sorting fails, maintain original order
            pass
    
    return canonicalized


def get_default_config_for_model_type(model_type: str) -> CanonicalConfig:
    """Get recommended canonicalization config for a model type.
    
    Args:
        model_type: Type of model ('vision', 'lm', 'multimodal')
        
    Returns:
        CanonicalConfig with appropriate defaults
    """
    if model_type == 'vision':
        return CanonicalConfig(
            float_precision=6,
            float_eps=1e-6,
            embedding_dims=512,  # Common vision embedding size
            handle_nan='zero',
            handle_inf='clip'
        )
    elif model_type == 'lm':
        return CanonicalConfig(
            float_precision=5,
            float_eps=1e-5,
            text_lower=True,
            text_strip_punct=True,
            text_max_len=512,
            handle_nan='zero',
            handle_inf='clip'
        )
    elif model_type == 'multimodal':
        return CanonicalConfig(
            float_precision=6,
            float_eps=1e-6,
            text_max_len=512,
            embedding_dims=768,  # Common multimodal embedding size
            handle_nan='zero',
            handle_inf='clip'
        )
    else:
        # Default configuration
        return CanonicalConfig()


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


def jacobian_sign_hash(jacobian: np.ndarray, threshold: float = 1e-6) -> bytes:
    """Compute a hash of the Jacobian sign pattern.
    
    Args:
        jacobian: Jacobian matrix of shape [input_dim, output_dim]
        threshold: Values below this are considered zero
        
    Returns:
        Bytes representation of the sign pattern hash
    """
    # Reference: §2.2 - compressed Jacobian representation
    sign_pattern = np.sign(jacobian)
    sign_pattern[np.abs(jacobian) < threshold] = 0
    
    # Convert to compact binary representation: -1 -> 00, 0 -> 01, 1 -> 10
    # This creates a deterministic encoding of the sign pattern
    mapping = {-1: 0b00, 0: 0b01, 1: 0b10}
    
    # Pack into bytes more efficiently
    byte_list = []
    bits = []
    for val in sign_pattern.flatten():
        bits.extend([mapping[int(val)] >> 1, mapping[int(val)] & 1])
        if len(bits) >= 8:
            byte_val = 0
            for i in range(8):
                byte_val = (byte_val << 1) | bits[i]
            byte_list.append(byte_val)
            bits = bits[8:]
    
    # Handle remaining bits
    if bits:
        byte_val = 0
        for i, bit in enumerate(bits):
            byte_val = (byte_val << 1) | bit
        byte_val <<= (8 - len(bits))  # Pad with zeros
        byte_list.append(byte_val)
    
    # Hash the packed bytes for a fixed-size output
    raw_bytes = bytes(byte_list)
    return hashlib.sha256(raw_bytes).digest()[:16]  # Return 16 bytes


def jacobian_magnitude_sketch(jacobian_matrix: np.ndarray, num_bins: int = 8) -> bytes:
    """Quantize Jacobian magnitudes into bins and create a compact sketch.
    
    Args:
        jacobian_matrix: Jacobian matrix of any shape
        num_bins: Number of magnitude bins for quantization (default: 8)
        
    Returns:
        Bytes representation of the magnitude pattern
    """
    # Compute absolute values for magnitude
    magnitudes = np.abs(jacobian_matrix).flatten()
    
    # Handle edge case of all zeros
    if np.all(magnitudes == 0):
        return bytes(num_bins)  # Return zeros
    
    # Use logarithmic binning for better dynamic range coverage
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    log_mags = np.log10(magnitudes + eps)
    
    # Determine bin edges using percentiles for adaptive binning
    # This ensures each bin captures meaningful variation
    nonzero_log_mags = log_mags[magnitudes > eps]
    if len(nonzero_log_mags) > 0:
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(nonzero_log_mags, percentiles)
        bin_edges[0] = -np.inf  # Ensure we capture all small values
        bin_edges[-1] = np.inf   # Ensure we capture all large values
    else:
        # All near-zero, use uniform bins
        bin_edges = np.linspace(log_mags.min(), log_mags.max() + eps, num_bins + 1)
    
    # Quantize into bins
    bin_indices = np.digitize(log_mags, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    # Create histogram for global magnitude distribution
    histogram = np.bincount(bin_indices, minlength=num_bins)
    
    # Normalize histogram to make it scale-invariant
    histogram = histogram.astype(np.float32)
    histogram_sum = histogram.sum()
    if histogram_sum > 0:
        histogram = histogram / histogram_sum
    
    # Also encode spatial pattern (first N elements pattern)
    pattern_length = min(256, len(bin_indices))  # Limit pattern length
    spatial_pattern = bin_indices[:pattern_length]
    
    # Combine histogram and spatial pattern into bytes
    # Quantize histogram to bytes (0-255 range)
    hist_bytes = (histogram * 255).astype(np.uint8).tobytes()
    
    # Pack spatial pattern (each bin index in 4 bits if num_bins <= 16)
    if num_bins <= 16:
        pattern_bytes = []
        for i in range(0, len(spatial_pattern), 2):
            if i + 1 < len(spatial_pattern):
                byte_val = (spatial_pattern[i] << 4) | spatial_pattern[i + 1]
            else:
                byte_val = spatial_pattern[i] << 4
            pattern_bytes.append(byte_val)
        pattern_bytes = bytes(pattern_bytes)
    else:
        pattern_bytes = spatial_pattern.astype(np.uint8).tobytes()
    
    # Combine and hash for fixed-size output
    combined = hist_bytes + pattern_bytes
    return hashlib.sha256(combined).digest()[:16]  # Return 16 bytes


def compute_jacobian_sketch(model: Union[torch.nn.Module, Callable], 
                           input_data: Union[np.ndarray, torch.Tensor],
                           layer_name: Optional[str] = None,
                           epsilon: float = 1e-6,
                           method: str = 'sign',
                           **kwargs) -> bytes:
    """Compute a compact Jacobian sketch at a specified layer.
    
    Args:
        model: PyTorch model or callable function
        input_data: Input tensor or array
        layer_name: Name of layer to compute Jacobian at (None for output)
        epsilon: Threshold for determining zero values
        method: 'sign' for sign pattern, 'magnitude' for magnitude sketch
        **kwargs: Additional arguments for the sketch method
            - num_bins: For magnitude sketch (default: 8)
            - delta: For finite differences (default: 1e-3)
            - max_dim: Maximum input dimensions to consider (default: 256)
    
    Returns:
        Bytes representation of the Jacobian sketch
    """
    # Extract parameters
    delta = kwargs.get('delta', 1e-3)
    max_dim = kwargs.get('max_dim', 256)
    num_bins = kwargs.get('num_bins', 8)
    
    # Convert input to appropriate format
    if isinstance(input_data, np.ndarray):
        input_tensor = torch.from_numpy(input_data).float()
    else:
        input_tensor = input_data.float() if hasattr(input_data, 'float') else input_data
    
    # Ensure batch dimension
    if input_tensor.dim() == 3:  # [C, H, W]
        input_tensor = input_tensor.unsqueeze(0)
    
    # Setup hook if layer_name is specified
    activation = {}
    hook_handle = None
    
    if layer_name and isinstance(model, torch.nn.Module):
        def hook_fn(module, input, output):
            activation['output'] = output.detach()
        
        # Find the layer and attach hook
        for name, module in model.named_modules():
            if name == layer_name:
                hook_handle = module.register_forward_hook(hook_fn)
                break
        
        if not hook_handle:
            raise ValueError(f"Layer '{layer_name}' not found in model")
    
    # Define the function for Jacobian computation
    def model_fn(x):
        x_tensor = x
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
            
        # Reshape if needed - preserve original input shape
        original_shape = input_tensor.shape
        if x_tensor.dim() == 1:
            # For flattened input, reshape back to original
            if len(original_shape) == 4:
                # Batched input [B, C, H, W]
                x_tensor = x_tensor.reshape(original_shape)
            elif len(original_shape) == 3:
                # Single input [C, H, W]
                x_tensor = x_tensor.reshape(original_shape).unsqueeze(0)
            elif len(original_shape) == 2:
                # Already 2D [B, features]
                x_tensor = x_tensor.reshape(original_shape)
            elif len(original_shape) == 1:
                # 1D input [features]
                x_tensor = x_tensor.unsqueeze(0)  # Add batch dim
        elif x_tensor.dim() == 3:
            x_tensor = x_tensor.unsqueeze(0)
            
        # Forward pass
        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                output = model(x_tensor)
                
                # Use hooked activation if available
                if layer_name and 'output' in activation:
                    output = activation['output']
        else:
            # Callable function
            output = model(x_tensor)
        
        # Convert to numpy
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
            
        return output.flatten()
    
    # Compute Jacobian using finite differences
    input_np = input_tensor.detach().cpu().numpy()
    if input_np.ndim == 4:
        input_np = input_np[0]  # Take first sample from batch
    
    input_flat = input_np.flatten()
    
    # Limit dimensions for efficiency
    actual_dim = min(input_flat.shape[0], max_dim)
    if actual_dim < input_flat.shape[0]:
        # Sample dimensions uniformly
        dim_indices = np.linspace(0, input_flat.shape[0] - 1, actual_dim, dtype=int)
        
        # Create reduced input
        reduced_input = input_flat[dim_indices]
        
        # Wrapper to map reduced dimensions back
        def reduced_model_fn(x_reduced):
            x_full = input_flat.copy()
            x_full[dim_indices] = x_reduced
            return model_fn(x_full)
        
        jacobian = finite_diff_jacobian(reduced_model_fn, reduced_input, delta=delta, max_dim=actual_dim)
    else:
        jacobian = finite_diff_jacobian(model_fn, input_flat, delta=delta, max_dim=max_dim)
    
    # Clean up hook if used
    if hook_handle:
        hook_handle.remove()
    
    # Compute sketch based on method
    if method == 'sign':
        return jacobian_sign_hash(jacobian, threshold=epsilon)
    elif method == 'magnitude':
        return jacobian_magnitude_sketch(jacobian, num_bins=num_bins)
    else:
        raise ValueError(f"Unknown sketch method: {method}. Use 'sign' or 'magnitude'")


def compare_jacobian_sketches(sketch1: bytes, sketch2: bytes, method: str = 'hamming') -> float:
    """Compare two Jacobian sketches for similarity.
    
    Args:
        sketch1: First sketch bytes
        sketch2: Second sketch bytes  
        method: Comparison method ('hamming', 'jaccard', 'cosine')
        
    Returns:
        Similarity score between 0 and 1 (1 = identical)
    """
    if len(sketch1) != len(sketch2):
        raise ValueError(f"Sketch sizes don't match: {len(sketch1)} vs {len(sketch2)}")
    
    if method == 'hamming':
        # Hamming similarity (1 - normalized Hamming distance)
        diff_bits = sum(bin(b1 ^ b2).count('1') for b1, b2 in zip(sketch1, sketch2))
        total_bits = len(sketch1) * 8
        return 1.0 - (diff_bits / total_bits)
        
    elif method == 'jaccard':
        # Jaccard similarity based on set bits
        union_bits = sum(bin(b1 | b2).count('1') for b1, b2 in zip(sketch1, sketch2))
        if union_bits == 0:
            return 1.0  # Both are zero vectors
        intersect_bits = sum(bin(b1 & b2).count('1') for b1, b2 in zip(sketch1, sketch2))
        return intersect_bits / union_bits
        
    elif method == 'cosine':
        # Treat bytes as vectors and compute cosine similarity
        vec1 = np.frombuffer(sketch1, dtype=np.uint8).astype(np.float32)
        vec2 = np.frombuffer(sketch2, dtype=np.uint8).astype(np.float32)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0 if norm1 == norm2 else 0.0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
        
    else:
        raise ValueError(f"Unknown comparison method: {method}")


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
            - 'canonical_config': CanonicalConfig object or dict for canonicalization
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
    
    # Get canonicalization config
    canonical_config = cfg.get('canonical_config', None)
    if canonical_config is None:
        # Build from individual parameters for backward compatibility
        precision = cfg.get('canonicalize_precision', 6)
        text_max_len = cfg.get('text_max_len', 512)
        canonical_config = CanonicalConfig(
            float_precision=precision,
            text_max_len=text_max_len
        )
    elif isinstance(canonical_config, dict):
        # Convert dict to CanonicalConfig
        canonical_config = CanonicalConfig(**canonical_config)
    
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
        
        # Use the new canonicalization system
        # Auto-detect model type from first output if not specified
        if model_type is None and not isinstance(output, str):
            if isinstance(challenge, str):
                model_type = 'lm'
            elif isinstance(challenge, (np.ndarray, torch.Tensor)):
                model_type = 'vision'
        
        # Get appropriate config if model type is known
        if model_type and canonical_config == CanonicalConfig():
            canonical_config = get_default_config_for_model_type(model_type)
        
        # Canonicalize the output
        canonical = canonicalize_model_output(output, 'auto', canonical_config)
        canonicalized_outputs.append(canonical)
            
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
                # Convert bytes to hex string for storage
                jacobian_sketches.append(j_hash.hex())
                
            except Exception as e:
                # Jacobian computation failed, continue without it
                pass
    
    # Compute stable IO hash from canonicalized outputs
    io_hash_value = io_hash(canonicalized_outputs, precision=canonical_config.float_precision)
    
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
        'precision': canonical_config.float_precision,
        'canonical_config': {
            'float_precision': canonical_config.float_precision,
            'text_max_len': canonical_config.text_max_len,
            'handle_nan': canonical_config.handle_nan,
            'handle_inf': canonical_config.handle_inf
        }
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