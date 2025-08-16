"""PRF-based challenge derivation for cryptographically secure challenge generation."""

import hashlib
import hmac
import struct
from typing import Union


def prf_derive_key(master_key: bytes, label: str, nonce: bytes) -> bytes:
    """
    Derive a key from master key using HMAC-SHA256 as PRF.
    
    Args:
        master_key: Master key bytes
        label: String label for key separation
        nonce: Random nonce for this derivation
    
    Returns:
        32-byte derived key
    """
    # Combine label and nonce as info for key derivation
    info = label.encode('utf-8') + nonce
    
    # Use HMAC-SHA256 as PRF for key derivation
    derived_key = hmac.new(master_key, info, hashlib.sha256).digest()
    
    return derived_key


def prf_bytes(key: bytes, info: bytes, nbytes: int) -> bytes:
    """
    Generate deterministic pseudorandom bytes using HMAC-SHA256.
    
    Uses counter mode with HMAC-SHA256 to generate arbitrary length output.
    This follows NIST SP 800-108 counter mode construction.
    
    Args:
        key: PRF key (typically output from prf_derive_key)
        info: Context/info bytes for this generation
        nbytes: Number of bytes to generate
    
    Returns:
        Pseudorandom bytes of requested length
    """
    if nbytes <= 0:
        raise ValueError("nbytes must be positive")
    
    output = bytearray()
    counter = 1
    
    # Generate blocks until we have enough bytes
    while len(output) < nbytes:
        # Counter || info format (counter as 32-bit big-endian)
        input_data = struct.pack('>I', counter) + info
        
        # Generate next block using HMAC-SHA256
        block = hmac.new(key, input_data, hashlib.sha256).digest()
        output.extend(block)
        counter += 1
        
        # Prevent counter overflow (extremely unlikely in practice)
        if counter > 0xFFFFFFFF:
            raise ValueError("Too many bytes requested")
    
    # Return exactly nbytes
    return bytes(output[:nbytes])


def prf_derive_seed(master_key: bytes, family: str, params: dict, nonce: bytes) -> bytes:
    """
    Derive a deterministic seed for challenge generation.
    
    Args:
        master_key: Master key bytes
        family: Challenge family identifier
        params: Parameters dict (will be hashed)
        nonce: Session nonce
    
    Returns:
        32-byte seed for RNG initialization
    """
    # Create deterministic representation of params
    # Sort keys for determinism
    params_str = repr(sorted(params.items()))
    params_hash = hashlib.sha256(params_str.encode('utf-8')).digest()
    
    # Derive family-specific key
    family_key = prf_derive_key(master_key, f"challenge:{family}", nonce)
    
    # Mix in params hash for final seed
    seed = prf_bytes(family_key, params_hash, 32)
    
    return seed


def prf_expand(seed: bytes, length: int) -> bytes:
    """
    Expand a seed to arbitrary length using PRF.
    
    Args:
        seed: Seed bytes (typically 32 bytes)
        length: Desired output length
    
    Returns:
        Expanded pseudorandom bytes
    """
    return prf_bytes(seed, b"expand", length)


def prf_integers(key: bytes, info: bytes, count: int, max_value: int) -> list[int]:
    """
    Generate deterministic pseudorandom integers.
    
    Args:
        key: PRF key
        info: Context info
        count: Number of integers to generate
        max_value: Maximum value (exclusive)
    
    Returns:
        List of integers in range [0, max_value)
    """
    if max_value <= 0:
        raise ValueError("max_value must be positive")
    
    # Calculate bytes needed per integer (with some margin for rejection sampling)
    bytes_per_int = (max_value.bit_length() + 7) // 8 + 1
    
    integers = []
    offset = 0
    
    while len(integers) < count:
        # Generate batch of random bytes
        batch_size = (count - len(integers)) * bytes_per_int * 2  # 2x for rejection sampling
        random_bytes = prf_bytes(key, info + struct.pack('>I', offset), batch_size)
        
        # Convert to integers using rejection sampling for uniformity
        for i in range(0, len(random_bytes) - bytes_per_int + 1, bytes_per_int):
            if len(integers) >= count:
                break
            
            # Convert bytes to integer
            value = int.from_bytes(random_bytes[i:i+bytes_per_int], 'big')
            
            # Rejection sampling for uniform distribution
            # Find the largest multiple of max_value that fits in our range
            limit = ((2 ** (bytes_per_int * 8)) // max_value) * max_value
            
            if value < limit:
                integers.append(value % max_value)
        
        offset += 1
    
    return integers[:count]


def prf_floats(key: bytes, info: bytes, count: int, min_val: float = 0.0, max_val: float = 1.0) -> list[float]:
    """
    Generate deterministic pseudorandom floats in a range.
    
    Args:
        key: PRF key
        info: Context info
        count: Number of floats to generate
        min_val: Minimum value (inclusive)
        max_val: Maximum value (exclusive)
    
    Returns:
        List of floats in range [min_val, max_val)
    """
    if min_val >= max_val:
        raise ValueError("min_val must be less than max_val")
    
    # Generate random bytes (8 bytes per float for double precision)
    random_bytes = prf_bytes(key, info, count * 8)
    
    floats = []
    for i in range(count):
        # Convert 8 bytes to float in [0, 1) using fixed-point arithmetic
        # This ensures uniform distribution
        uint64 = int.from_bytes(random_bytes[i*8:(i+1)*8], 'big')
        # Divide by 2^64 to get value in [0, 1)
        unit_float = uint64 / (2 ** 64)
        # Scale to desired range
        scaled_float = min_val + unit_float * (max_val - min_val)
        floats.append(scaled_float)
    
    return floats


def prf_choice(key: bytes, info: bytes, choices: list, count: int) -> list:
    """
    Make deterministic random choices from a list.
    
    Args:
        key: PRF key
        info: Context info
        choices: List of items to choose from
        count: Number of choices to make
    
    Returns:
        List of chosen items
    """
    if not choices:
        raise ValueError("choices list cannot be empty")
    
    indices = prf_integers(key, info, count, len(choices))
    return [choices[i] for i in indices]


def prf_shuffle(key: bytes, info: bytes, items: list) -> list:
    """
    Deterministically shuffle a list using Fisher-Yates with PRF.
    
    Args:
        key: PRF key
        info: Context info
        items: List to shuffle
    
    Returns:
        New shuffled list (original is not modified)
    """
    items_copy = list(items)
    n = len(items_copy)
    
    if n <= 1:
        return items_copy
    
    # Generate random integers for Fisher-Yates shuffle
    # For position i, we need a random integer in range [i, n)
    for i in range(n - 1):
        # Generate random index in range [i, n)
        rand_ints = prf_integers(key, info + struct.pack('>I', i), 1, n - i)
        j = i + rand_ints[0]
        # Swap
        items_copy[i], items_copy[j] = items_copy[j], items_copy[i]
    
    return items_copy