"""
Caching system for ZK proof generation optimization.

This module provides caching for expensive computations like Merkle trees,
circuit components, and witness pre-computations to speed up proof generation.
"""

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np

from pot.zk.poseidon import PoseidonMerkleTree, poseidon_hash
from pot.zk.field_arithmetic import FieldElement


@dataclass
class CacheEntry:
    """Entry in the cache with metadata."""
    key: str
    value: Any
    size_bytes: int
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # Time to live in seconds


class LRUCache:
    """
    Least Recently Used cache with size limits.
    """
    
    def __init__(self, max_size_mb: float = 100, max_entries: int = 1000):
        """
        Initialize LRU cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            max_entries: Maximum number of entries
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_entries = max_entries
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.total_size = 0
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_access_time_ms': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        start = time.time()
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check TTL
            if entry.ttl and (time.time() - entry.creation_time) > entry.ttl:
                self.delete(key)
                self.stats['misses'] += 1
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_access = time.time()
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            
            self.stats['hits'] += 1
            self.stats['total_access_time_ms'] += (time.time() - start) * 1000
            return entry.value
        
        self.stats['misses'] += 1
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in cache."""
        # Calculate size
        size = self._estimate_size(value)
        
        # Check if we need to evict
        while (self.total_size + size > self.max_size_bytes or 
               len(self.cache) >= self.max_entries):
            if not self._evict_oldest():
                break
        
        # Add new entry
        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size,
            ttl=ttl
        )
        
        self.cache[key] = entry
        self.total_size += size
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.total_size -= entry.size_bytes
            del self.cache[key]
            return True
        return False
    
    def clear(self):
        """Clear entire cache."""
        self.cache.clear()
        self.total_size = 0
        self.stats['evictions'] += len(self.cache)
    
    def _evict_oldest(self) -> bool:
        """Evict oldest entry."""
        if not self.cache:
            return False
        
        # Get oldest (first) item
        key = next(iter(self.cache))
        self.delete(key)
        self.stats['evictions'] += 1
        return True
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        if isinstance(value, bytes):
            return len(value)
        elif isinstance(value, str):
            return len(value.encode())
        elif isinstance(value, np.ndarray):
            return value.nbytes
        else:
            # Use pickle for complex objects
            try:
                return len(pickle.dumps(value))
            except:
                return 1000  # Default estimate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.copy()
        stats['size_mb'] = self.total_size / (1024 * 1024)
        stats['entries'] = len(self.cache)
        
        total_requests = stats['hits'] + stats['misses']
        if total_requests > 0:
            stats['hit_rate'] = stats['hits'] / total_requests
            stats['avg_access_time_ms'] = stats['total_access_time_ms'] / total_requests
        
        return stats


class MerkleTreeCache:
    """
    Specialized cache for Poseidon Merkle trees.
    """
    
    def __init__(self, max_trees: int = 100):
        """
        Initialize Merkle tree cache.
        
        Args:
            max_trees: Maximum number of trees to cache
        """
        self.max_trees = max_trees
        self.trees: OrderedDict[str, PoseidonMerkleTree] = OrderedDict()
        self.tree_stats: Dict[str, Dict] = {}
    
    def get_tree(self, leaves: List[bytes]) -> PoseidonMerkleTree:
        """
        Get or create Merkle tree for leaves.
        
        Args:
            leaves: Leaf data
            
        Returns:
            Merkle tree
        """
        # Compute cache key
        key = self._compute_key(leaves)
        
        if key in self.trees:
            # Move to end (most recently used)
            self.trees.move_to_end(key)
            self.tree_stats[key]['hits'] += 1
            return self.trees[key]
        
        # Create new tree
        tree = PoseidonMerkleTree(leaves)
        
        # Add to cache
        if len(self.trees) >= self.max_trees:
            # Remove oldest
            oldest = next(iter(self.trees))
            del self.trees[oldest]
            del self.tree_stats[oldest]
        
        self.trees[key] = tree
        self.tree_stats[key] = {
            'hits': 0,
            'creation_time': time.time(),
            'num_leaves': len(leaves)
        }
        
        return tree
    
    def get_root(self, leaves: List[bytes]) -> bytes:
        """Get Merkle root for leaves."""
        tree = self.get_tree(leaves)
        return tree.root()
    
    def get_proof(self, leaves: List[bytes], index: int) -> List[bytes]:
        """Get Merkle proof for leaf at index."""
        tree = self.get_tree(leaves)
        return tree.proof(index)
    
    def _compute_key(self, leaves: List[bytes]) -> str:
        """Compute cache key for leaves."""
        # Hash all leaves together
        hasher = hashlib.blake2b()
        for leaf in leaves:
            hasher.update(leaf)
        return hasher.hexdigest()
    
    def clear(self):
        """Clear cache."""
        self.trees.clear()
        self.tree_stats.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(s['hits'] for s in self.tree_stats.values())
        total_trees = len(self.trees)
        
        return {
            'cached_trees': total_trees,
            'total_hits': total_hits,
            'avg_hits_per_tree': total_hits / total_trees if total_trees > 0 else 0,
            'max_trees': self.max_trees
        }


class WitnessCache:
    """
    Cache for pre-computed witness data.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize witness cache.
        
        Args:
            cache_dir: Directory for persistent cache
        """
        self.memory_cache = LRUCache(max_size_mb=50)
        self.cache_dir = cache_dir or Path(".zk_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_witness(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached witness data.
        
        Args:
            key: Witness identifier
            
        Returns:
            Witness data if cached
        """
        # Check memory cache first
        witness = self.memory_cache.get(key)
        if witness is not None:
            return witness
        
        # Check disk cache
        file_path = self.cache_dir / f"{key}.pkl"
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    witness = pickle.load(f)
                # Add to memory cache
                self.memory_cache.put(key, witness)
                return witness
            except:
                pass
        
        return None
    
    def put_witness(self, key: str, witness: Dict[str, Any], persist: bool = True):
        """
        Cache witness data.
        
        Args:
            key: Witness identifier
            witness: Witness data
            persist: Whether to persist to disk
        """
        # Add to memory cache
        self.memory_cache.put(key, witness)
        
        # Persist to disk if requested
        if persist:
            file_path = self.cache_dir / f"{key}.pkl"
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(witness, f)
            except:
                pass
    
    def precompute_gradients(self, weights_before: np.ndarray, 
                            weights_after: np.ndarray,
                            learning_rate: float) -> np.ndarray:
        """
        Pre-compute and cache gradients.
        
        Args:
            weights_before: Weights before update
            weights_after: Weights after update
            learning_rate: Learning rate used
            
        Returns:
            Computed gradients
        """
        # Create cache key
        key = self._gradient_key(weights_before, weights_after, learning_rate)
        
        # Check cache
        cached = self.memory_cache.get(f"grad_{key}")
        if cached is not None:
            return cached
        
        # Compute gradients
        gradients = (weights_before - weights_after) / learning_rate
        
        # Cache result
        self.memory_cache.put(f"grad_{key}", gradients, ttl=3600)  # 1 hour TTL
        
        return gradients
    
    def _gradient_key(self, w_before: np.ndarray, w_after: np.ndarray, lr: float) -> str:
        """Generate cache key for gradients."""
        hasher = hashlib.blake2b()
        hasher.update(w_before.tobytes())
        hasher.update(w_after.tobytes())
        hasher.update(str(lr).encode())
        return hasher.hexdigest()[:16]
    
    def clear_memory(self):
        """Clear memory cache."""
        self.memory_cache.clear()
    
    def clear_disk(self):
        """Clear disk cache."""
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        disk_files = list(self.cache_dir.glob("*.pkl"))
        disk_size = sum(f.stat().st_size for f in disk_files)
        
        return {
            'memory': self.memory_cache.get_stats(),
            'disk': {
                'files': len(disk_files),
                'size_mb': disk_size / (1024 * 1024)
            }
        }


class CircuitCache:
    """
    Cache for circuit components and setup parameters.
    """
    
    def __init__(self):
        """Initialize circuit cache."""
        self.setup_params: Dict[str, Any] = {}
        self.verification_keys: Dict[str, bytes] = {}
        self.proving_keys: Dict[str, bytes] = {}
        self.gadget_cache: Dict[str, Any] = {}
    
    def get_setup(self, circuit_type: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached setup parameters."""
        key = self._setup_key(circuit_type, params)
        return self.setup_params.get(key)
    
    def put_setup(self, circuit_type: str, params: Dict[str, Any], setup: Any):
        """Cache setup parameters."""
        key = self._setup_key(circuit_type, params)
        self.setup_params[key] = setup
    
    def get_verification_key(self, circuit_id: str) -> Optional[bytes]:
        """Get cached verification key."""
        return self.verification_keys.get(circuit_id)
    
    def put_verification_key(self, circuit_id: str, vk: bytes):
        """Cache verification key."""
        self.verification_keys[circuit_id] = vk
    
    def get_proving_key(self, circuit_id: str) -> Optional[bytes]:
        """Get cached proving key."""
        return self.proving_keys.get(circuit_id)
    
    def put_proving_key(self, circuit_id: str, pk: bytes):
        """Cache proving key."""
        self.proving_keys[circuit_id] = pk
    
    def cache_gadget(self, gadget_type: str, params: Dict[str, Any], gadget: Any):
        """Cache reusable gadget."""
        key = f"{gadget_type}:{hashlib.sha256(str(params).encode()).hexdigest()[:8]}"
        self.gadget_cache[key] = gadget
    
    def get_gadget(self, gadget_type: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached gadget."""
        key = f"{gadget_type}:{hashlib.sha256(str(params).encode()).hexdigest()[:8]}"
        return self.gadget_cache.get(key)
    
    def _setup_key(self, circuit_type: str, params: Dict[str, Any]) -> str:
        """Generate cache key for setup."""
        param_str = str(sorted(params.items()))
        return f"{circuit_type}:{hashlib.sha256(param_str.encode()).hexdigest()[:16]}"
    
    def clear(self):
        """Clear all caches."""
        self.setup_params.clear()
        self.verification_keys.clear()
        self.proving_keys.clear()
        self.gadget_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'setup_params': len(self.setup_params),
            'verification_keys': len(self.verification_keys),
            'proving_keys': len(self.proving_keys),
            'gadgets': len(self.gadget_cache)
        }


# Global cache instances
_global_caches = {
    'lru': None,
    'merkle': None,
    'witness': None,
    'circuit': None
}


def get_cache(cache_type: str) -> Any:
    """Get global cache instance."""
    if cache_type not in _global_caches:
        raise ValueError(f"Unknown cache type: {cache_type}")
    
    if _global_caches[cache_type] is None:
        if cache_type == 'lru':
            _global_caches[cache_type] = LRUCache()
        elif cache_type == 'merkle':
            _global_caches[cache_type] = MerkleTreeCache()
        elif cache_type == 'witness':
            _global_caches[cache_type] = WitnessCache()
        elif cache_type == 'circuit':
            _global_caches[cache_type] = CircuitCache()
    
    return _global_caches[cache_type]


def clear_all_caches():
    """Clear all global caches."""
    for cache in _global_caches.values():
        if cache is not None:
            cache.clear()


def get_all_stats() -> Dict[str, Any]:
    """Get statistics from all caches."""
    stats = {}
    for name, cache in _global_caches.items():
        if cache is not None:
            stats[name] = cache.get_stats()
    return stats