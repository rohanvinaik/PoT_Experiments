"""
Proof aggregation for efficient batch verification.

This module implements proof aggregation techniques to combine multiple
SGD step proofs into a single proof, reducing verification costs and
on-chain storage requirements.
"""

import hashlib
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from pot.zk.zk_types import (
    SGDStepStatement,
    SGDStepWitness,
    LoRAStepStatement,
    LoRAStepWitness
)
from pot.zk.poseidon import poseidon_hash, poseidon_merkle_root


@dataclass
class AggregatedProof:
    """Container for aggregated proof data."""
    proof_data: bytes
    num_proofs: int
    statements: List[Any]  # List of statements being proven
    aggregate_root: bytes
    timestamp: float
    verification_key: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProofBatch:
    """Batch of proofs to be aggregated."""
    proofs: List[bytes]
    statements: List[Any]
    witnesses: Optional[List[Any]] = None
    proof_type: str = "sgd"  # "sgd" or "lora"


class ProofAggregator:
    """
    Aggregates multiple proofs into a single succinct proof.
    
    This uses recursive proof composition to create a tree of proofs,
    where each level proves the validity of the level below.
    """
    
    def __init__(self, max_batch_size: int = 16, use_parallel: bool = True):
        """
        Initialize proof aggregator.
        
        Args:
            max_batch_size: Maximum number of proofs to aggregate at once
            use_parallel: Whether to use parallel processing
        """
        self.max_batch_size = max_batch_size
        self.use_parallel = use_parallel
        self.aggregation_cache = {}
        self.stats = {
            'total_aggregated': 0,
            'total_time_ms': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def aggregate_proofs(self, batch: ProofBatch) -> AggregatedProof:
        """
        Aggregate a batch of proofs into a single proof.
        
        Args:
            batch: Batch of proofs to aggregate
            
        Returns:
            Aggregated proof
        """
        start_time = time.time()
        
        if len(batch.proofs) == 0:
            raise ValueError("Cannot aggregate empty batch")
        
        if len(batch.proofs) == 1:
            # Single proof doesn't need aggregation
            return AggregatedProof(
                proof_data=batch.proofs[0],
                num_proofs=1,
                statements=batch.statements,
                aggregate_root=poseidon_hash(batch.proofs[0]),
                timestamp=time.time(),
                metadata={'aggregation_level': 0}
            )
        
        # Check cache
        cache_key = self._compute_cache_key(batch)
        if cache_key in self.aggregation_cache:
            self.stats['cache_hits'] += 1
            return self.aggregation_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        # Split into smaller batches if needed
        if len(batch.proofs) > self.max_batch_size:
            aggregated = self._recursive_aggregate(batch)
        else:
            aggregated = self._aggregate_batch(batch)
        
        # Update cache and stats
        self.aggregation_cache[cache_key] = aggregated
        self.stats['total_aggregated'] += len(batch.proofs)
        self.stats['total_time_ms'] += int((time.time() - start_time) * 1000)
        
        return aggregated
    
    def _aggregate_batch(self, batch: ProofBatch) -> AggregatedProof:
        """
        Aggregate a small batch of proofs.
        
        This is where the actual proof aggregation logic would go.
        In production, this would call Rust code for Halo2 aggregation.
        """
        # Create aggregation tree
        proof_hashes = [poseidon_hash(p) for p in batch.proofs]
        aggregate_root = poseidon_merkle_root(proof_hashes)
        
        # For mock implementation, create aggregated proof
        # In production, this would use Halo2's accumulator
        aggregated_data = self._mock_aggregate_proofs(batch.proofs)
        
        metadata = {
            'aggregation_level': 1,
            'batch_size': len(batch.proofs),
            'proof_type': batch.proof_type,
            'compression_ratio': len(aggregated_data) / sum(len(p) for p in batch.proofs)
        }
        
        return AggregatedProof(
            proof_data=aggregated_data,
            num_proofs=len(batch.proofs),
            statements=batch.statements,
            aggregate_root=aggregate_root,
            timestamp=time.time(),
            metadata=metadata
        )
    
    def _recursive_aggregate(self, batch: ProofBatch) -> AggregatedProof:
        """
        Recursively aggregate large batches.
        
        Splits large batches into smaller chunks and aggregates them
        in a tree structure for efficiency.
        """
        # Split into chunks
        chunk_size = self.max_batch_size
        chunks = []
        
        for i in range(0, len(batch.proofs), chunk_size):
            chunk_proofs = batch.proofs[i:i + chunk_size]
            chunk_statements = batch.statements[i:i + chunk_size]
            chunk_witnesses = None
            if batch.witnesses:
                chunk_witnesses = batch.witnesses[i:i + chunk_size]
            
            chunks.append(ProofBatch(
                proofs=chunk_proofs,
                statements=chunk_statements,
                witnesses=chunk_witnesses,
                proof_type=batch.proof_type
            ))
        
        # Aggregate chunks in parallel if enabled
        if self.use_parallel and len(chunks) > 1:
            with ThreadPoolExecutor(max_workers=4) as executor:
                chunk_aggregates = list(executor.map(self._aggregate_batch, chunks))
        else:
            chunk_aggregates = [self._aggregate_batch(chunk) for chunk in chunks]
        
        # Create new batch from aggregated chunks
        aggregated_batch = ProofBatch(
            proofs=[agg.proof_data for agg in chunk_aggregates],
            statements=batch.statements,  # Keep original statements
            proof_type=batch.proof_type
        )
        
        # Recursively aggregate if needed
        if len(aggregated_batch.proofs) > 1:
            final_aggregate = self.aggregate_proofs(aggregated_batch)
            final_aggregate.metadata['aggregation_level'] = 2
            final_aggregate.num_proofs = len(batch.proofs)  # Original count
            return final_aggregate
        else:
            return chunk_aggregates[0]
    
    def _mock_aggregate_proofs(self, proofs: List[bytes]) -> bytes:
        """
        Mock proof aggregation for testing.
        
        In production, this would use actual Halo2 aggregation.
        """
        # Combine proofs with compression
        combined = b''.join(proofs)
        
        # Simulate compression/aggregation
        compressed = hashlib.blake2b(combined, digest_size=128).digest()
        
        # Add aggregation metadata
        metadata = f"AGG:{len(proofs)}:".encode()
        
        return metadata + compressed
    
    def _compute_cache_key(self, batch: ProofBatch) -> str:
        """Compute cache key for batch."""
        # Use hash of proof hashes as cache key
        proof_hashes = [hashlib.sha256(p).hexdigest() for p in batch.proofs]
        combined = ''.join(sorted(proof_hashes))
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def verify_aggregated_proof(self, proof: AggregatedProof) -> bool:
        """
        Verify an aggregated proof.
        
        Args:
            proof: Aggregated proof to verify
            
        Returns:
            True if proof is valid
        """
        # In production, this would call Rust verification
        # For now, check basic validity
        if not proof.proof_data:
            return False
        
        if proof.num_proofs <= 0:
            return False
        
        # Mock verification
        return proof.proof_data.startswith(b"AGG:")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        stats = self.stats.copy()
        if stats['total_aggregated'] > 0:
            stats['avg_time_ms'] = stats['total_time_ms'] / stats['total_aggregated']
        if stats['cache_hits'] + stats['cache_misses'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        return stats


class IncrementalAggregator:
    """
    Incremental proof aggregation for streaming scenarios.
    
    Maintains a running aggregated proof that can be updated
    with new proofs as they arrive.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize incremental aggregator.
        
        Args:
            window_size: Maximum number of proofs to keep in window
        """
        self.window_size = window_size
        self.current_window: List[bytes] = []
        self.current_statements: List[Any] = []
        self.aggregator = ProofAggregator()
        self.current_aggregate: Optional[AggregatedProof] = None
        self.total_processed = 0
    
    def add_proof(self, proof: bytes, statement: Any) -> Optional[AggregatedProof]:
        """
        Add a proof to the incremental aggregator.
        
        Args:
            proof: New proof to add
            statement: Statement for the proof
            
        Returns:
            Updated aggregate if window is full, None otherwise
        """
        self.current_window.append(proof)
        self.current_statements.append(statement)
        self.total_processed += 1
        
        # Check if window is full
        if len(self.current_window) >= self.window_size:
            return self.flush()
        
        return None
    
    def flush(self) -> Optional[AggregatedProof]:
        """
        Flush current window and create aggregate.
        
        Returns:
            Aggregated proof for current window
        """
        if not self.current_window:
            return None
        
        batch = ProofBatch(
            proofs=self.current_window,
            statements=self.current_statements
        )
        
        aggregate = self.aggregator.aggregate_proofs(batch)
        
        # Clear window
        self.current_window = []
        self.current_statements = []
        
        # Update running aggregate
        if self.current_aggregate is None:
            self.current_aggregate = aggregate
        else:
            # Merge with existing aggregate
            self.current_aggregate = self._merge_aggregates(
                self.current_aggregate, aggregate
            )
        
        return aggregate
    
    def _merge_aggregates(self, agg1: AggregatedProof, agg2: AggregatedProof) -> AggregatedProof:
        """Merge two aggregated proofs."""
        batch = ProofBatch(
            proofs=[agg1.proof_data, agg2.proof_data],
            statements=agg1.statements + agg2.statements
        )
        
        merged = self.aggregator.aggregate_proofs(batch)
        merged.num_proofs = agg1.num_proofs + agg2.num_proofs
        merged.metadata['merged'] = True
        
        return merged
    
    def get_current_aggregate(self) -> Optional[AggregatedProof]:
        """Get current running aggregate."""
        return self.current_aggregate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get incremental aggregation stats."""
        return {
            'total_processed': self.total_processed,
            'window_size': self.window_size,
            'current_window_size': len(self.current_window),
            'has_aggregate': self.current_aggregate is not None,
            'aggregator_stats': self.aggregator.get_stats()
        }


class BatchVerifier:
    """
    Batch verification of multiple proofs.
    
    More efficient than verifying proofs individually.
    """
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize batch verifier.
        
        Args:
            batch_size: Optimal batch size for verification
        """
        self.batch_size = batch_size
        self.verification_cache = {}
        self.stats = {
            'total_verified': 0,
            'total_batches': 0,
            'cache_hits': 0
        }
    
    def verify_batch(self, proofs: List[bytes], statements: List[Any]) -> List[bool]:
        """
        Verify a batch of proofs.
        
        Args:
            proofs: List of proofs to verify
            statements: Corresponding statements
            
        Returns:
            List of verification results
        """
        if len(proofs) != len(statements):
            raise ValueError("Number of proofs must match number of statements")
        
        results = []
        
        # Process in optimal batch sizes
        for i in range(0, len(proofs), self.batch_size):
            batch_proofs = proofs[i:i + self.batch_size]
            batch_statements = statements[i:i + self.batch_size]
            
            # Check cache
            cache_keys = [self._cache_key(p, s) for p, s in zip(batch_proofs, batch_statements)]
            batch_results = []
            
            for j, key in enumerate(cache_keys):
                if key in self.verification_cache:
                    batch_results.append(self.verification_cache[key])
                    self.stats['cache_hits'] += 1
                else:
                    # Verify proof
                    result = self._verify_single(batch_proofs[j], batch_statements[j])
                    batch_results.append(result)
                    self.verification_cache[key] = result
            
            results.extend(batch_results)
        
        self.stats['total_verified'] += len(proofs)
        self.stats['total_batches'] += 1
        
        return results
    
    def _verify_single(self, proof: bytes, statement: Any) -> bool:
        """Verify a single proof."""
        # Mock verification
        # In production, this would call Rust verification
        return len(proof) > 0 and statement is not None
    
    def _cache_key(self, proof: bytes, statement: Any) -> str:
        """Generate cache key for proof/statement pair."""
        proof_hash = hashlib.sha256(proof).hexdigest()
        statement_hash = hashlib.sha256(str(statement).encode()).hexdigest()
        return f"{proof_hash}:{statement_hash}"
    
    def clear_cache(self):
        """Clear verification cache."""
        self.verification_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        stats = self.stats.copy()
        if stats['total_batches'] > 0:
            stats['avg_batch_size'] = stats['total_verified'] / stats['total_batches']
        total_verifications = stats['total_verified']
        if total_verifications > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_verifications
        return stats


def optimize_proof_size(proof: bytes) -> bytes:
    """
    Optimize proof size for storage/transmission.
    
    Args:
        proof: Original proof
        
    Returns:
        Optimized proof
    """
    import zlib
    
    # Try compression
    compressed = zlib.compress(proof, level=9)
    
    # Only use compressed if smaller
    if len(compressed) < len(proof):
        return b'COMP:' + compressed
    
    return proof


def decompress_proof(proof: bytes) -> bytes:
    """
    Decompress optimized proof.
    
    Args:
        proof: Potentially compressed proof
        
    Returns:
        Original proof
    """
    import zlib
    
    if proof.startswith(b'COMP:'):
        return zlib.decompress(proof[5:])
    
    return proof