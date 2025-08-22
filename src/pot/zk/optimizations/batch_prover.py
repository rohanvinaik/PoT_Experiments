"""
Batch Prover for ZK Proof System

Efficient batch proof generation with optimizations.
"""

import time
import queue
import threading
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


@dataclass
class ProofRequest:
    """Request for proof generation"""
    request_id: str
    circuit: Dict[str, Any]
    witness: List[Any]
    public_inputs: List[Any]
    priority: int = 0
    
    def __lt__(self, other):
        return self.priority > other.priority  # Higher priority first


@dataclass
class ProofResult:
    """Result of proof generation"""
    request_id: str
    proof: bytes
    proof_size: int
    generation_time: float
    success: bool
    error: Optional[str] = None


class BatchProver:
    """Efficient batch proof generation"""
    
    def __init__(self, batch_size: int = 10, num_workers: int = 4):
        """Initialize batch prover"""
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.request_queue = queue.PriorityQueue()
        self.result_cache = {}
        self.stats = {
            'total_proofs': 0,
            'total_batches': 0,
            'total_time': 0,
            'cache_hits': 0,
            'batch_efficiency': 0
        }
    
    def generate_proof_batch(
        self,
        requests: List[ProofRequest],
        use_caching: bool = True
    ) -> List[ProofResult]:
        """Generate proofs for a batch of requests"""
        
        start_time = time.perf_counter()
        results = []
        
        # Check cache for existing proofs
        if use_caching:
            cached_results = []
            uncached_requests = []
            
            for request in requests:
                cache_key = self._get_cache_key(request)
                if cache_key in self.result_cache:
                    cached_results.append(self.result_cache[cache_key])
                    self.stats['cache_hits'] += 1
                else:
                    uncached_requests.append(request)
            
            results.extend(cached_results)
            requests = uncached_requests
        
        # Process uncached requests in parallel
        if requests:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_request = {
                    executor.submit(self._generate_single_proof, request): request
                    for request in requests
                }
                
                for future in as_completed(future_to_request):
                    request = future_to_request[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Cache result
                        if use_caching and result.success:
                            cache_key = self._get_cache_key(request)
                            self.result_cache[cache_key] = result
                    except Exception as e:
                        results.append(ProofResult(
                            request_id=request.request_id,
                            proof=b'',
                            proof_size=0,
                            generation_time=0,
                            success=False,
                            error=str(e)
                        ))
        
        batch_time = time.perf_counter() - start_time
        
        # Update statistics
        self.stats['total_proofs'] += len(results)
        self.stats['total_batches'] += 1
        self.stats['total_time'] += batch_time
        
        # Calculate batch efficiency
        sequential_time = sum(r.generation_time for r in results)
        self.stats['batch_efficiency'] = sequential_time / batch_time if batch_time > 0 else 1
        
        return results
    
    def generate_streaming_proofs(
        self,
        request_stream: queue.Queue,
        result_callback: callable,
        timeout: float = 60.0
    ):
        """Generate proofs from a streaming queue of requests"""
        
        batch = []
        last_batch_time = time.time()
        
        while True:
            try:
                # Collect requests for batch
                request = request_stream.get(timeout=1.0)
                batch.append(request)
                
                # Process batch if full or timeout reached
                if len(batch) >= self.batch_size or time.time() - last_batch_time > timeout:
                    results = self.generate_proof_batch(batch)
                    
                    # Send results via callback
                    for result in results:
                        result_callback(result)
                    
                    batch = []
                    last_batch_time = time.time()
                    
            except queue.Empty:
                # Process remaining batch
                if batch:
                    results = self.generate_proof_batch(batch)
                    for result in results:
                        result_callback(result)
                    batch = []
                
                # Check if we should continue
                if request_stream.empty():
                    break
    
    def _generate_single_proof(self, request: ProofRequest) -> ProofResult:
        """Generate a single proof"""
        start_time = time.perf_counter()
        
        try:
            # Simulate proof generation
            circuit_size = request.circuit.get('constraints', 0)
            time.sleep(0.001 * (circuit_size / 100))  # Simulate computation
            
            # Generate mock proof
            proof_data = f"{request.request_id}_{circuit_size}_{len(request.witness)}"
            proof = hashlib.sha256(proof_data.encode()).digest()
            
            generation_time = time.perf_counter() - start_time
            
            return ProofResult(
                request_id=request.request_id,
                proof=proof,
                proof_size=len(proof),
                generation_time=generation_time,
                success=True
            )
        except Exception as e:
            return ProofResult(
                request_id=request.request_id,
                proof=b'',
                proof_size=0,
                generation_time=time.perf_counter() - start_time,
                success=False,
                error=str(e)
            )
    
    def _get_cache_key(self, request: ProofRequest) -> str:
        """Generate cache key for a proof request"""
        # Create deterministic key from circuit and witness
        key_data = f"{request.circuit.get('constraints', 0)}_{len(request.witness)}_{len(request.public_inputs)}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def optimize_batch_size(
        self,
        test_requests: List[ProofRequest],
        test_sizes: List[int] = None
    ) -> int:
        """Find optimal batch size for given workload"""
        if test_sizes is None:
            test_sizes = [1, 5, 10, 20, 50]
        
        best_throughput = 0
        best_size = self.batch_size
        
        for batch_size in test_sizes:
            self.batch_size = batch_size
            
            start_time = time.perf_counter()
            results = self.generate_proof_batch(test_requests[:batch_size])
            batch_time = time.perf_counter() - start_time
            
            throughput = len(results) / batch_time if batch_time > 0 else 0
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_size = batch_size
        
        self.batch_size = best_size
        return best_size
    
    def clear_cache(self):
        """Clear the proof cache"""
        self.result_cache.clear()
        self.stats['cache_hits'] = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch prover statistics"""
        return {
            'total_proofs': self.stats['total_proofs'],
            'total_batches': self.stats['total_batches'],
            'avg_batch_size': self.stats['total_proofs'] / self.stats['total_batches']
                             if self.stats['total_batches'] > 0 else 0,
            'total_time': self.stats['total_time'],
            'avg_proof_time': self.stats['total_time'] / self.stats['total_proofs']
                             if self.stats['total_proofs'] > 0 else 0,
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': self.stats['cache_hits'] / self.stats['total_proofs']
                             if self.stats['total_proofs'] > 0 else 0,
            'batch_efficiency': self.stats['batch_efficiency'],
            'cache_size': len(self.result_cache)
        }