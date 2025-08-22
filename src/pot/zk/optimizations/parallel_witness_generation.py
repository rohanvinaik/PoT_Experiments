"""
Parallel Witness Generation for ZK Proofs

Multi-threaded witness computation for improved performance.
"""

import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import queue
import numpy as np


@dataclass
class WitnessChunk:
    """A chunk of witness to be computed"""
    chunk_id: int
    start_idx: int
    end_idx: int
    constraints: List[Any]
    inputs: Dict[str, Any]
    
    
@dataclass
class WitnessResult:
    """Result of witness computation"""
    chunk_id: int
    witness_values: List[Any]
    computation_time: float
    success: bool
    error: Optional[str] = None


class ParallelWitnessGenerator:
    """Generate witness in parallel for improved performance"""
    
    def __init__(self, num_workers: Optional[int] = None):
        """Initialize parallel witness generator"""
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.stats = {
            'total_witnesses': 0,
            'total_time': 0,
            'chunks_processed': 0,
            'parallel_speedup': 0
        }
    
    def generate_witness(
        self,
        circuit: Dict[str, Any],
        inputs: Dict[str, Any],
        use_multiprocessing: bool = False
    ) -> Tuple[List[Any], float]:
        """Generate witness using parallel computation"""
        
        start_time = time.perf_counter()
        
        # Determine chunk size based on circuit size
        num_constraints = circuit.get('constraints', 0)
        chunk_size = max(100, num_constraints // (self.num_workers * 4))
        
        # Create witness chunks
        chunks = self._create_witness_chunks(circuit, inputs, chunk_size)
        
        # Process chunks in parallel
        if use_multiprocessing:
            witness_values = self._process_chunks_multiprocess(chunks)
        else:
            witness_values = self._process_chunks_multithread(chunks)
        
        computation_time = time.perf_counter() - start_time
        
        # Update statistics
        self.stats['total_witnesses'] += 1
        self.stats['total_time'] += computation_time
        self.stats['chunks_processed'] += len(chunks)
        
        # Calculate speedup
        sequential_time = self._estimate_sequential_time(circuit)
        self.stats['parallel_speedup'] = sequential_time / computation_time if computation_time > 0 else 1
        
        return witness_values, computation_time
    
    def _create_witness_chunks(
        self,
        circuit: Dict[str, Any],
        inputs: Dict[str, Any],
        chunk_size: int
    ) -> List[WitnessChunk]:
        """Divide witness computation into chunks"""
        chunks = []
        num_constraints = circuit.get('constraints', 0)
        
        for i in range(0, num_constraints, chunk_size):
            chunk = WitnessChunk(
                chunk_id=len(chunks),
                start_idx=i,
                end_idx=min(i + chunk_size, num_constraints),
                constraints=circuit.get('gates', [])[i:i+chunk_size],
                inputs=inputs
            )
            chunks.append(chunk)
        
        return chunks
    
    def _process_chunks_multithread(self, chunks: List[WitnessChunk]) -> List[Any]:
        """Process chunks using threading"""
        witness_values = [None] * len(chunks)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_chunk = {
                executor.submit(self._compute_witness_chunk, chunk): chunk
                for chunk in chunks
            }
            
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    if result.success:
                        witness_values[result.chunk_id] = result.witness_values
                except Exception as e:
                    print(f"Chunk {chunk.chunk_id} failed: {e}")
        
        # Flatten witness values
        flattened = []
        for chunk_values in witness_values:
            if chunk_values:
                flattened.extend(chunk_values)
        
        return flattened
    
    def _process_chunks_multiprocess(self, chunks: List[WitnessChunk]) -> List[Any]:
        """Process chunks using multiprocessing"""
        witness_values = [None] * len(chunks)
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_chunk = {
                executor.submit(compute_witness_chunk_worker, chunk): chunk
                for chunk in chunks
            }
            
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    if result.success:
                        witness_values[result.chunk_id] = result.witness_values
                except Exception as e:
                    print(f"Chunk {chunk.chunk_id} failed: {e}")
        
        # Flatten witness values
        flattened = []
        for chunk_values in witness_values:
            if chunk_values:
                flattened.extend(chunk_values)
        
        return flattened
    
    def _compute_witness_chunk(self, chunk: WitnessChunk) -> WitnessResult:
        """Compute witness for a single chunk"""
        start_time = time.perf_counter()
        
        try:
            # Simulate witness computation
            num_values = chunk.end_idx - chunk.start_idx
            witness_values = list(range(chunk.start_idx, chunk.end_idx))
            
            # Simulate computation time based on chunk size
            time.sleep(0.0001 * num_values)
            
            computation_time = time.perf_counter() - start_time
            
            return WitnessResult(
                chunk_id=chunk.chunk_id,
                witness_values=witness_values,
                computation_time=computation_time,
                success=True
            )
        except Exception as e:
            return WitnessResult(
                chunk_id=chunk.chunk_id,
                witness_values=[],
                computation_time=time.perf_counter() - start_time,
                success=False,
                error=str(e)
            )
    
    def _estimate_sequential_time(self, circuit: Dict[str, Any]) -> float:
        """Estimate sequential computation time"""
        num_constraints = circuit.get('constraints', 0)
        return 0.0001 * num_constraints  # Simple linear estimate
    
    def optimize_worker_count(
        self,
        circuit: Dict[str, Any],
        test_sizes: List[int] = None
    ) -> int:
        """Find optimal number of workers for circuit"""
        if test_sizes is None:
            test_sizes = [1, 2, 4, 8, self.num_workers]
        
        best_time = float('inf')
        best_workers = 1
        
        dummy_inputs = {'test': True}
        
        for num_workers in test_sizes:
            self.num_workers = num_workers
            _, computation_time = self.generate_witness(circuit, dummy_inputs)
            
            if computation_time < best_time:
                best_time = computation_time
                best_workers = num_workers
        
        self.num_workers = best_workers
        return best_workers
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'total_witnesses': self.stats['total_witnesses'],
            'total_time': self.stats['total_time'],
            'avg_time': self.stats['total_time'] / self.stats['total_witnesses'] 
                        if self.stats['total_witnesses'] > 0 else 0,
            'chunks_processed': self.stats['chunks_processed'],
            'parallel_speedup': self.stats['parallel_speedup'],
            'num_workers': self.num_workers
        }


def compute_witness_chunk_worker(chunk: WitnessChunk) -> WitnessResult:
    """Worker function for multiprocessing"""
    start_time = time.perf_counter()
    
    try:
        # Simulate witness computation
        num_values = chunk.end_idx - chunk.start_idx
        witness_values = list(range(chunk.start_idx, chunk.end_idx))
        
        # Simulate computation time
        time.sleep(0.0001 * num_values)
        
        return WitnessResult(
            chunk_id=chunk.chunk_id,
            witness_values=witness_values,
            computation_time=time.perf_counter() - start_time,
            success=True
        )
    except Exception as e:
        return WitnessResult(
            chunk_id=chunk.chunk_id,
            witness_values=[],
            computation_time=time.perf_counter() - start_time,
            success=False,
            error=str(e)
        )