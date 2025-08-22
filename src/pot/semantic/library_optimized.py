"""
Optimized concept library with incremental updates, memory-mapped storage, and sparse representations.
Performance-critical implementation for large-scale semantic verification.
"""

import numpy as np
import torch
import torch.sparse as sparse
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import time
import logging
import hashlib
import mmap
import pickle
import json
from dataclasses import dataclass
from collections import deque
import scipy.sparse as sp

logger = logging.getLogger(__name__)


@dataclass
class IncrementalStats:
    """Incremental statistics tracker using Welford's algorithm."""
    n: int = 0
    mean: Optional[torch.Tensor] = None
    m2: Optional[torch.Tensor] = None  # Sum of squared deviations
    
    def update(self, x: torch.Tensor):
        """Update statistics with new sample(s)."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        for sample in x:
            self.n += 1
            if self.mean is None:
                self.mean = sample.clone()
                self.m2 = torch.zeros_like(sample)
            else:
                delta = sample - self.mean
                self.mean += delta / self.n
                delta2 = sample - self.mean
                self.m2 += delta * delta2
    
    @property
    def variance(self) -> Optional[torch.Tensor]:
        """Get variance."""
        if self.n < 2:
            return None
        return self.m2 / (self.n - 1)
    
    @property
    def std(self) -> Optional[torch.Tensor]:
        """Get standard deviation."""
        var = self.variance
        return torch.sqrt(var) if var is not None else None


class SparseHypervector:
    """Sparse representation for hypervectors to save memory."""
    
    def __init__(self, dim: int, sparsity: float = 0.99):
        """
        Initialize sparse hypervector.
        
        Args:
            dim: Vector dimension
            sparsity: Proportion of zero elements (0.99 = 99% zeros)
        """
        self.dim = dim
        self.sparsity = sparsity
        self.indices = []
        self.values = []
    
    @classmethod
    def from_dense(cls, dense_vector: torch.Tensor, sparsity: float = 0.99) -> 'SparseHypervector':
        """Create sparse hypervector from dense tensor."""
        obj = cls(dim=dense_vector.shape[0], sparsity=sparsity)
        
        # Find non-zero elements
        if sparsity > 0:
            # Keep only the top k% of values
            k = int(dense_vector.shape[0] * (1 - sparsity))
            topk_values, topk_indices = torch.topk(torch.abs(dense_vector), k)
            
            obj.indices = topk_indices.tolist()
            obj.values = dense_vector[topk_indices].tolist()
        else:
            # Keep all non-zero values
            nonzero_mask = dense_vector != 0
            obj.indices = torch.where(nonzero_mask)[0].tolist()
            obj.values = dense_vector[nonzero_mask].tolist()
        
        return obj
    
    def to_dense(self) -> torch.Tensor:
        """Convert to dense tensor."""
        dense = torch.zeros(self.dim)
        if self.indices:
            dense[self.indices] = torch.tensor(self.values)
        return dense
    
    def to_sparse_tensor(self) -> torch.sparse.FloatTensor:
        """Convert to PyTorch sparse tensor."""
        if not self.indices:
            return torch.sparse_coo_tensor(
                torch.zeros((1, 0), dtype=torch.long),
                torch.zeros(0),
                (self.dim,)
            )
        
        indices = torch.tensor(self.indices).unsqueeze(0)
        values = torch.tensor(self.values)
        return torch.sparse_coo_tensor(indices, values, (self.dim,))
    
    def hamming_distance(self, other: 'SparseHypervector') -> float:
        """Compute Hamming distance with another sparse hypervector."""
        # Convert to sets for efficient comparison
        self_set = set(zip(self.indices, [v > 0 for v in self.values]))
        other_set = set(zip(other.indices, [v > 0 for v in other.values]))
        
        # Count differences
        differences = len(self_set.symmetric_difference(other_set))
        
        # Account for positions that are zero in both (similar)
        explicit_positions = set(self.indices) | set(other.indices)
        zeros_in_both = self.dim - len(explicit_positions)
        
        return differences / self.dim


class MemoryMappedConceptLibrary:
    """
    Memory-efficient concept library using memory-mapped files for large-scale storage.
    Supports incremental updates and sparse representations.
    """
    
    def __init__(self, dim: int, method: str = 'gaussian',
                 storage_path: Optional[Path] = None,
                 max_memory_mb: int = 1000,
                 use_sparse: bool = True,
                 sparsity: float = 0.99):
        """
        Initialize memory-mapped concept library.
        
        Args:
            dim: Dimensionality of concept vectors
            method: Representation method ('gaussian' or 'hypervector')
            storage_path: Path for memory-mapped storage (None for in-memory)
            max_memory_mb: Maximum memory usage in MB
            use_sparse: Use sparse representations for hypervectors
            sparsity: Sparsity level for sparse hypervectors
        """
        self.dim = dim
        self.method = method
        self.storage_path = storage_path
        self.max_memory_mb = max_memory_mb
        self.use_sparse = use_sparse and method == 'hypervector'
        self.sparsity = sparsity
        
        # Concept storage
        self.concepts = {}  # concept_name -> concept_data
        self.incremental_stats = {}  # concept_name -> IncrementalStats
        
        # Memory-mapped arrays for large-scale storage
        self.mmap_arrays = {}
        if storage_path:
            self.storage_path = Path(storage_path)
            self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Memory tracking
        self.memory_usage = 0
        self.cache = deque(maxlen=100)  # LRU cache for recent concepts
        
        logger.info(f"Initialized MemoryMappedConceptLibrary: dim={dim}, method={method}, "
                   f"storage={storage_path}, sparse={use_sparse}")
    
    def add_concept_incremental(self, name: str, embedding: torch.Tensor,
                               batch_update: bool = False) -> None:
        """
        Add embedding to concept using incremental statistics update.
        Memory-efficient for streaming data.
        
        Args:
            name: Concept name
            embedding: Single embedding or batch
            batch_update: If True, delay covariance computation
        """
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        
        # Initialize incremental stats if needed
        if name not in self.incremental_stats:
            self.incremental_stats[name] = IncrementalStats()
            self.concepts[name] = {
                'method': self.method,
                'timestamp': time.time(),
                'incremental': True
            }
        
        # Update statistics incrementally
        stats = self.incremental_stats[name]
        stats.update(embedding)
        
        # Update concept vector
        if self.method == 'gaussian':
            self.concepts[name]['vector'] = stats.mean.numpy() if stats.mean is not None else None
            self.concepts[name]['n_samples'] = stats.n
            
            # Compute covariance only when needed (not during batch updates)
            if not batch_update and stats.n > 1:
                self.concepts[name]['variance'] = stats.variance.numpy()
        
        elif self.method == 'hypervector':
            # Build hypervector incrementally
            if stats.mean is not None:
                hv = self._build_hypervector_incremental(stats.mean)
                if self.use_sparse:
                    sparse_hv = SparseHypervector.from_dense(hv, self.sparsity)
                    self.concepts[name]['sparse_vector'] = sparse_hv
                    self.concepts[name]['vector'] = None  # Don't store dense
                else:
                    self.concepts[name]['vector'] = hv.numpy()
        
        # Update cache
        self.cache.append(name)
        
        # Check memory usage and offload if needed
        self._check_memory_usage()
    
    def _build_hypervector_incremental(self, mean: torch.Tensor) -> torch.Tensor:
        """Build hypervector from running mean."""
        # Ternary encoding based on percentiles
        percentile_33 = torch.quantile(mean, 0.33)
        percentile_67 = torch.quantile(mean, 0.67)
        
        hypervector = torch.zeros_like(mean)
        hypervector[mean < percentile_33] = -1
        hypervector[mean > percentile_67] = 1
        
        return hypervector
    
    def finalize_concept(self, name: str) -> None:
        """
        Finalize incremental concept by computing final statistics.
        Call after all embeddings have been added.
        """
        if name not in self.incremental_stats:
            return
        
        stats = self.incremental_stats[name]
        
        if self.method == 'gaussian' and stats.n > 1:
            # Compute final covariance
            self.concepts[name]['covariance'] = stats.variance.numpy()
            
            # Optionally save to memory-mapped file
            if self.storage_path and stats.n > 1000:
                self._save_to_mmap(name)
    
    def _save_to_mmap(self, name: str) -> None:
        """Save concept to memory-mapped file."""
        if not self.storage_path:
            return
        
        concept_data = self.concepts[name]
        mmap_path = self.storage_path / f"{name}.mmap"
        
        # Save numpy arrays as memory-mapped
        if 'vector' in concept_data and concept_data['vector'] is not None:
            np.save(mmap_path.with_suffix('.npy'), concept_data['vector'])
            
            # Replace with memory-mapped reference
            concept_data['vector_mmap'] = str(mmap_path.with_suffix('.npy'))
            concept_data['vector'] = None
        
        # Save metadata
        metadata_path = mmap_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'name': name,
                'method': concept_data['method'],
                'n_samples': concept_data.get('n_samples', 0),
                'timestamp': concept_data['timestamp']
            }, f)
    
    def load_concept_from_mmap(self, name: str) -> Optional[np.ndarray]:
        """Load concept vector from memory-mapped file."""
        if name not in self.concepts:
            return None
        
        concept_data = self.concepts[name]
        
        # Check if already loaded
        if 'vector' in concept_data and concept_data['vector'] is not None:
            return concept_data['vector']
        
        # Load from memory-mapped file
        if 'vector_mmap' in concept_data:
            vector = np.load(concept_data['vector_mmap'], mmap_mode='r')
            return vector
        
        # Load from sparse representation
        if 'sparse_vector' in concept_data:
            sparse_hv = concept_data['sparse_vector']
            return sparse_hv.to_dense().numpy()
        
        return None
    
    def get_concept_vector(self, name: str) -> Optional[np.ndarray]:
        """
        Get concept vector, loading from storage if needed.
        
        Args:
            name: Concept name
            
        Returns:
            Concept vector as numpy array
        """
        if name not in self.concepts:
            return None
        
        # Update cache
        self.cache.append(name)
        
        # Get vector (may load from mmap)
        return self.load_concept_from_mmap(name)
    
    def compute_similarity_sparse(self, embedding: torch.Tensor, 
                                 concept_name: str) -> float:
        """
        Compute similarity using sparse representations.
        
        Args:
            embedding: Query embedding
            concept_name: Concept to compare against
            
        Returns:
            Similarity score [0, 1]
        """
        if concept_name not in self.concepts:
            return 0.0
        
        concept_data = self.concepts[concept_name]
        
        if 'sparse_vector' in concept_data:
            # Sparse hypervector similarity
            sparse_hv = concept_data['sparse_vector']
            query_hv = SparseHypervector.from_dense(
                torch.sign(embedding), 
                self.sparsity
            )
            
            # Hamming similarity
            distance = sparse_hv.hamming_distance(query_hv)
            return 1.0 - distance
        
        else:
            # Fall back to dense computation
            concept_vector = self.get_concept_vector(concept_name)
            if concept_vector is None:
                return 0.0
            
            # Cosine similarity
            concept_tensor = torch.from_numpy(concept_vector)
            similarity = torch.nn.functional.cosine_similarity(
                embedding.unsqueeze(0),
                concept_tensor.unsqueeze(0)
            ).item()
            
            return (similarity + 1.0) / 2.0
    
    def _check_memory_usage(self) -> None:
        """Check memory usage and offload to disk if needed."""
        # Estimate memory usage
        estimated_mb = 0
        for name, data in self.concepts.items():
            if 'vector' in data and data['vector'] is not None:
                estimated_mb += data['vector'].nbytes / (1024 * 1024)
        
        self.memory_usage = estimated_mb
        
        # Offload oldest concepts if over limit
        if estimated_mb > self.max_memory_mb and self.storage_path:
            concepts_to_offload = []
            
            for name in self.concepts:
                if name not in self.cache:
                    concepts_to_offload.append(name)
                    if len(concepts_to_offload) >= 10:
                        break
            
            for name in concepts_to_offload:
                self._save_to_mmap(name)
                logger.debug(f"Offloaded concept '{name}' to disk")
    
    def save(self, path: Path) -> None:
        """Save library to disk."""
        path = Path(path)
        
        # Save metadata
        metadata = {
            'dim': self.dim,
            'method': self.method,
            'use_sparse': self.use_sparse,
            'sparsity': self.sparsity,
            'concepts': list(self.concepts.keys())
        }
        
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f)
        
        # Save concepts
        concepts_data = {}
        for name, data in self.concepts.items():
            # Convert sparse vectors to serializable format
            if 'sparse_vector' in data:
                sparse_hv = data['sparse_vector']
                concepts_data[name] = {
                    'indices': sparse_hv.indices,
                    'values': sparse_hv.values,
                    'dim': sparse_hv.dim,
                    'sparse': True
                }
            else:
                concepts_data[name] = data
        
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(concepts_data, f)
    
    @classmethod
    def load(cls, path: Path) -> 'MemoryMappedConceptLibrary':
        """Load library from disk."""
        path = Path(path)
        
        # Load metadata
        with open(path.with_suffix('.json'), 'r') as f:
            metadata = json.load(f)
        
        # Create library
        library = cls(
            dim=metadata['dim'],
            method=metadata['method'],
            use_sparse=metadata.get('use_sparse', False),
            sparsity=metadata.get('sparsity', 0.99)
        )
        
        # Load concepts
        with open(path.with_suffix('.pkl'), 'rb') as f:
            concepts_data = pickle.load(f)
        
        for name, data in concepts_data.items():
            if data.get('sparse', False):
                # Reconstruct sparse hypervector
                sparse_hv = SparseHypervector(dim=data['dim'])
                sparse_hv.indices = data['indices']
                sparse_hv.values = data['values']
                data['sparse_vector'] = sparse_hv
                del data['indices'], data['values'], data['dim'], data['sparse']
            
            library.concepts[name] = data
        
        return library


def create_optimized_library(dim: int, method: str = 'gaussian',
                            **kwargs) -> MemoryMappedConceptLibrary:
    """
    Factory function to create optimized concept library.
    
    Args:
        dim: Vector dimension
        method: 'gaussian' or 'hypervector'
        **kwargs: Additional arguments for MemoryMappedConceptLibrary
        
    Returns:
        Optimized concept library instance
    """
    return MemoryMappedConceptLibrary(dim, method, **kwargs)