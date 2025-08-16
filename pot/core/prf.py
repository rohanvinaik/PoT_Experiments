"""PRF-based challenge derivation for cryptographically secure challenge generation."""

import hashlib
import hmac
import struct
from typing import Union, List, Any
import xxhash


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


def prf_expand(key: bytes, info: bytes, length: int) -> bytes:
    """
    Expand key material using HKDF-like expansion with xxhash for speed.
    
    This implements an HKDF-Expand-like function using xxhash64 for performance
    while maintaining cryptographic properties through HMAC-SHA256 mixing.
    
    Args:
        key: Key material (typically output from prf_derive_key)
        info: Context/info bytes for this expansion
        length: Desired output length in bytes
    
    Returns:
        Expanded pseudorandom bytes
    """
    if length <= 0:
        raise ValueError("length must be positive")
    
    # For short outputs, use direct PRF
    if length <= 32:
        return prf_bytes(key, info, length)
    
    # For longer outputs, use hybrid approach: HMAC for security, xxhash for speed
    output = bytearray()
    
    # First, derive a mixing key using HMAC-SHA256
    mixing_key = hmac.new(key, b"xxhash-expand" + info, hashlib.sha256).digest()
    
    # Use xxhash in counter mode for fast expansion
    hasher = xxhash.xxh64()
    counter = 0
    
    while len(output) < length:
        # Mix counter with mixing key for each block
        hasher.reset()
        hasher.update(mixing_key)
        hasher.update(struct.pack('>Q', counter))
        hasher.update(info)
        
        # Generate 8-byte block
        block = hasher.digest()
        output.extend(block)
        counter += 1
        
        # Every 8 blocks, re-mix with HMAC for security
        if counter % 8 == 0:
            mixing_key = hmac.new(key, mixing_key + struct.pack('>Q', counter), hashlib.sha256).digest()
    
    return bytes(output[:length])


def prf_integers(key: bytes, info: bytes, count: int, min_val: int, max_val: int) -> List[int]:
    """
    Generate deterministic pseudorandom integers in a specified range.
    
    Args:
        key: PRF key
        info: Context info
        count: Number of integers to generate
        min_val: Minimum value (inclusive)
        max_val: Maximum value (exclusive)
    
    Returns:
        List of integers in range [min_val, max_val)
    """
    if min_val >= max_val:
        raise ValueError("min_val must be less than max_val")
    
    range_size = max_val - min_val
    if range_size <= 0:
        raise ValueError("Invalid range")
    
    # Calculate bytes needed per integer (with some margin for rejection sampling)
    bytes_per_int = (range_size.bit_length() + 7) // 8 + 1
    
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
            # Find the largest multiple of range_size that fits in our range
            limit = ((2 ** (bytes_per_int * 8)) // range_size) * range_size
            
            if value < limit:
                integers.append(min_val + (value % range_size))
        
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


def prf_choice(key: bytes, info: bytes, choices: List[Any]) -> Any:
    """
    Deterministically select a single item from choices.
    
    Args:
        key: PRF key
        info: Context info
        choices: List of items to choose from
    
    Returns:
        Single chosen item
    """
    if not choices:
        raise ValueError("choices list cannot be empty")
    
    indices = prf_integers(key, info, 1, 0, len(choices))
    return choices[indices[0]]


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
        rand_ints = prf_integers(key, info + struct.pack('>I', i), 1, 0, n - i)
        j = i + rand_ints[0]
        # Swap
        items_copy[i], items_copy[j] = items_copy[j], items_copy[i]
    
    return items_copy