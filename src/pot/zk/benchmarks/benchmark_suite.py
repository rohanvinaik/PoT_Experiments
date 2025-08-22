"""
ZK Proof System Benchmarking Suite

Comprehensive benchmarking for proof generation, verification, and optimization.
"""

import time
import psutil
import json
import statistics
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
import tracemalloc
import gc
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class CircuitSize(Enum):
    """Standard circuit sizes for benchmarking"""
    TINY = "tiny"  # < 1K constraints
    SMALL = "small"  # 1K - 10K constraints
    MEDIUM = "medium"  # 10K - 100K constraints
    LARGE = "large"  # 100K - 1M constraints
    HUGE = "huge"  # > 1M constraints


class BenchmarkType(Enum):
    """Types of benchmarks"""
    PROOF_GENERATION = "proof_generation"
    VERIFICATION = "verification"
    WITNESS_GENERATION = "witness_generation"
    SETUP = "setup"
    MEMORY = "memory"
    THROUGHPUT = "throughput"


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    benchmark_type: BenchmarkType
    circuit_size: CircuitSize
    num_constraints: int
    num_variables: int
    
    # Timing metrics (in seconds)
    setup_time: float = 0.0
    witness_time: float = 0.0
    proof_time: float = 0.0
    verify_time: float = 0.0
    total_time: float = 0.0
    
    # Memory metrics (in MB)
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    memory_delta_mb: float = 0.0
    
    # Proof metrics
    proof_size_bytes: int = 0
    public_inputs_size: int = 0
    
    # Performance metrics
    constraints_per_second: float = 0.0
    throughput_proofs_per_hour: float = 0.0
    
    # Statistical metrics (from multiple runs)
    mean_time: float = 0.0
    std_dev_time: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    percentile_95: float = 0.0
    
    # System info
    cpu_count: int = field(default_factory=lambda: psutil.cpu_count())
    cpu_freq_mhz: float = field(default_factory=lambda: psutil.cpu_freq().current if psutil.cpu_freq() else 0)
    total_memory_gb: float = field(default_factory=lambda: psutil.virtual_memory().total / (1024**3))
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    notes: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['benchmark_type'] = self.benchmark_type.value
        result['circuit_size'] = self.circuit_size.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary"""
        data['benchmark_type'] = BenchmarkType(data['benchmark_type'])
        data['circuit_size'] = CircuitSize(data['circuit_size'])
        return cls(**data)


class MemoryTracker:
    """Track memory usage during benchmarks"""
    
    def __init__(self):
        self.baseline_memory = 0
        self.peak_memory = 0
        self.samples = []
        self.tracking = False
    
    def start(self):
        """Start memory tracking"""
        gc.collect()
        tracemalloc.start()
        self.baseline_memory = self._get_current_memory()
        self.peak_memory = self.baseline_memory
        self.samples = []
        self.tracking = True
    
    def sample(self):
        """Take a memory sample"""
        if not self.tracking:
            return
        current = self._get_current_memory()
        self.samples.append(current)
        self.peak_memory = max(self.peak_memory, current)
    
    def stop(self) -> Tuple[float, float, float]:
        """Stop tracking and return metrics (peak, avg, delta) in MB"""
        if not self.tracking:
            return 0, 0, 0
        
        tracemalloc.stop()
        self.tracking = False
        
        peak_mb = self.peak_memory / (1024 * 1024)
        avg_mb = statistics.mean(self.samples) / (1024 * 1024) if self.samples else 0
        delta_mb = (self.peak_memory - self.baseline_memory) / (1024 * 1024)
        
        return peak_mb, avg_mb, delta_mb
    
    def _get_current_memory(self) -> float:
        """Get current memory usage in bytes"""
        if tracemalloc.is_tracing():
            current, _ = tracemalloc.get_traced_memory()
            return current
        return psutil.Process().memory_info().rss


class ZKBenchmarkSuite:
    """Main benchmarking suite for ZK proof system"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize benchmark suite"""
        self.output_dir = output_dir or Path("benchmarks/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory_tracker = MemoryTracker()
        self.results = []
        
        # Circuit size configurations
        self.size_configs = {
            CircuitSize.TINY: {
                'constraints': 100,
                'variables': 50,
                'public_inputs': 10
            },
            CircuitSize.SMALL: {
                'constraints': 5000,
                'variables': 2500,
                'public_inputs': 50
            },
            CircuitSize.MEDIUM: {
                'constraints': 50000,
                'variables': 25000,
                'public_inputs': 100
            },
            CircuitSize.LARGE: {
                'constraints': 500000,
                'variables': 250000,
                'public_inputs': 500
            },
            CircuitSize.HUGE: {
                'constraints': 2000000,
                'variables': 1000000,
                'public_inputs': 1000
            }
        }
    
    def benchmark_proof_generation(
        self,
        circuit_size: CircuitSize,
        num_iterations: int = 10,
        warmup_iterations: int = 2
    ) -> BenchmarkResult:
        """Benchmark proof generation for a given circuit size"""
        config = self.size_configs[circuit_size]
        
        print(f"Benchmarking {circuit_size.value} circuit ({config['constraints']} constraints)...")
        
        # Create mock circuit
        circuit = self._create_mock_circuit(
            config['constraints'],
            config['variables'],
            config['public_inputs']
        )
        
        # Warmup runs
        for _ in range(warmup_iterations):
            self._generate_mock_proof(circuit)
        
        # Actual benchmark runs
        times = []
        memory_peaks = []
        proof_sizes = []
        
        for i in range(num_iterations):
            # Start memory tracking
            self.memory_tracker.start()
            
            # Time proof generation
            start_time = time.perf_counter()
            
            # Setup phase
            setup_start = time.perf_counter()
            setup_params = self._mock_setup(circuit)
            setup_time = time.perf_counter() - setup_start
            
            # Witness generation
            witness_start = time.perf_counter()
            witness = self._generate_mock_witness(circuit)
            witness_time = time.perf_counter() - witness_start
            
            # Proof generation
            proof_start = time.perf_counter()
            proof = self._generate_mock_proof_with_params(circuit, setup_params, witness)
            proof_time = time.perf_counter() - proof_start
            
            total_time = time.perf_counter() - start_time
            
            # Sample memory periodically
            self.memory_tracker.sample()
            
            # Stop memory tracking
            peak_mb, avg_mb, delta_mb = self.memory_tracker.stop()
            
            # Record metrics
            times.append(total_time)
            memory_peaks.append(peak_mb)
            proof_sizes.append(len(proof) if isinstance(proof, bytes) else 1024)  # Mock size
            
            print(f"  Iteration {i+1}/{num_iterations}: {total_time:.3f}s, Peak memory: {peak_mb:.1f}MB")
        
        # Calculate statistics
        result = BenchmarkResult(
            benchmark_type=BenchmarkType.PROOF_GENERATION,
            circuit_size=circuit_size,
            num_constraints=config['constraints'],
            num_variables=config['variables'],
            setup_time=setup_time,
            witness_time=witness_time,
            proof_time=proof_time,
            total_time=statistics.mean(times),
            peak_memory_mb=statistics.mean(memory_peaks),
            avg_memory_mb=avg_mb,
            memory_delta_mb=delta_mb,
            proof_size_bytes=statistics.mean(proof_sizes) if proof_sizes else 0,
            public_inputs_size=config['public_inputs'],
            constraints_per_second=config['constraints'] / statistics.mean(times) if times else 0,
            throughput_proofs_per_hour=3600 / statistics.mean(times) if times else 0,
            mean_time=statistics.mean(times),
            std_dev_time=statistics.stdev(times) if len(times) > 1 else 0,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            percentile_95=np.percentile(times, 95) if times else 0
        )
        
        self.results.append(result)
        return result
    
    def benchmark_verification(
        self,
        circuit_size: CircuitSize,
        num_iterations: int = 100
    ) -> BenchmarkResult:
        """Benchmark proof verification"""
        config = self.size_configs[circuit_size]
        
        print(f"Benchmarking verification for {circuit_size.value} circuit...")
        
        # Generate a proof to verify
        circuit = self._create_mock_circuit(
            config['constraints'],
            config['variables'],
            config['public_inputs']
        )
        proof = self._generate_mock_proof(circuit)
        
        # Benchmark verification
        times = []
        
        for i in range(num_iterations):
            start_time = time.perf_counter()
            valid = self._verify_mock_proof(proof, circuit)
            verify_time = time.perf_counter() - start_time
            times.append(verify_time)
            
            if i % 20 == 0:
                print(f"  Iteration {i+1}/{num_iterations}: {verify_time*1000:.3f}ms")
        
        result = BenchmarkResult(
            benchmark_type=BenchmarkType.VERIFICATION,
            circuit_size=circuit_size,
            num_constraints=config['constraints'],
            num_variables=config['variables'],
            verify_time=statistics.mean(times),
            total_time=statistics.mean(times),
            mean_time=statistics.mean(times),
            std_dev_time=statistics.stdev(times) if len(times) > 1 else 0,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            percentile_95=np.percentile(times, 95) if times else 0
        )
        
        self.results.append(result)
        return result
    
    def benchmark_memory_usage(
        self,
        circuit_size: CircuitSize
    ) -> BenchmarkResult:
        """Benchmark memory usage for different circuit sizes"""
        config = self.size_configs[circuit_size]
        
        print(f"Benchmarking memory usage for {circuit_size.value} circuit...")
        
        # Track memory throughout the process
        self.memory_tracker.start()
        
        # Create circuit
        circuit = self._create_mock_circuit(
            config['constraints'],
            config['variables'],
            config['public_inputs']
        )
        self.memory_tracker.sample()
        
        # Setup
        setup_params = self._mock_setup(circuit)
        self.memory_tracker.sample()
        
        # Witness generation
        witness = self._generate_mock_witness(circuit)
        self.memory_tracker.sample()
        
        # Proof generation
        proof = self._generate_mock_proof_with_params(circuit, setup_params, witness)
        self.memory_tracker.sample()
        
        # Get memory metrics
        peak_mb, avg_mb, delta_mb = self.memory_tracker.stop()
        
        result = BenchmarkResult(
            benchmark_type=BenchmarkType.MEMORY,
            circuit_size=circuit_size,
            num_constraints=config['constraints'],
            num_variables=config['variables'],
            peak_memory_mb=peak_mb,
            avg_memory_mb=avg_mb,
            memory_delta_mb=delta_mb,
            notes=f"Memory profile for {circuit_size.value} circuit"
        )
        
        self.results.append(result)
        return result
    
    def benchmark_throughput(
        self,
        circuit_size: CircuitSize,
        duration_seconds: int = 60
    ) -> BenchmarkResult:
        """Benchmark throughput (proofs per time unit)"""
        config = self.size_configs[circuit_size]
        
        print(f"Benchmarking throughput for {circuit_size.value} circuit ({duration_seconds}s test)...")
        
        circuit = self._create_mock_circuit(
            config['constraints'],
            config['variables'],
            config['public_inputs']
        )
        
        # Generate proofs for specified duration
        start_time = time.time()
        end_time = start_time + duration_seconds
        proof_count = 0
        proof_times = []
        
        while time.time() < end_time:
            proof_start = time.perf_counter()
            self._generate_mock_proof(circuit)
            proof_time = time.perf_counter() - proof_start
            proof_times.append(proof_time)
            proof_count += 1
            
            if proof_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = proof_count / elapsed
                print(f"  Generated {proof_count} proofs, Rate: {rate:.2f} proofs/sec")
        
        actual_duration = time.time() - start_time
        
        result = BenchmarkResult(
            benchmark_type=BenchmarkType.THROUGHPUT,
            circuit_size=circuit_size,
            num_constraints=config['constraints'],
            num_variables=config['variables'],
            total_time=actual_duration,
            throughput_proofs_per_hour=proof_count * 3600 / actual_duration,
            mean_time=statistics.mean(proof_times) if proof_times else 0,
            std_dev_time=statistics.stdev(proof_times) if len(proof_times) > 1 else 0,
            notes=f"Generated {proof_count} proofs in {actual_duration:.1f}s"
        )
        
        self.results.append(result)
        return result
    
    def run_comprehensive_benchmark(self) -> List[BenchmarkResult]:
        """Run comprehensive benchmark suite"""
        print("=" * 60)
        print("COMPREHENSIVE ZK BENCHMARK SUITE")
        print("=" * 60)
        
        all_results = []
        
        # Test different circuit sizes
        sizes_to_test = [
            CircuitSize.TINY,
            CircuitSize.SMALL,
            CircuitSize.MEDIUM,
            # CircuitSize.LARGE,  # Skip large for quick tests
        ]
        
        for size in sizes_to_test:
            print(f"\n Testing {size.value} circuits...")
            print("-" * 40)
            
            # Proof generation benchmark
            try:
                result = self.benchmark_proof_generation(size, num_iterations=5)
                all_results.append(result)
                print(f"  ✓ Proof generation: {result.mean_time:.3f}s")
            except Exception as e:
                print(f"  ✗ Proof generation failed: {e}")
            
            # Verification benchmark
            try:
                result = self.benchmark_verification(size, num_iterations=50)
                all_results.append(result)
                print(f"  ✓ Verification: {result.mean_time*1000:.3f}ms")
            except Exception as e:
                print(f"  ✗ Verification failed: {e}")
            
            # Memory benchmark
            try:
                result = self.benchmark_memory_usage(size)
                all_results.append(result)
                print(f"  ✓ Peak memory: {result.peak_memory_mb:.1f}MB")
            except Exception as e:
                print(f"  ✗ Memory benchmark failed: {e}")
            
            # Throughput benchmark (shorter for smaller circuits)
            if size in [CircuitSize.TINY, CircuitSize.SMALL]:
                try:
                    result = self.benchmark_throughput(size, duration_seconds=10)
                    all_results.append(result)
                    print(f"  ✓ Throughput: {result.throughput_proofs_per_hour:.0f} proofs/hour")
                except Exception as e:
                    print(f"  ✗ Throughput benchmark failed: {e}")
        
        # Save results
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results: Optional[List[BenchmarkResult]] = None):
        """Save benchmark results to file"""
        if results is None:
            results = self.results
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        data = {
            'timestamp': time.time(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'platform': sys.platform
            },
            'results': [r.to_dict() for r in results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        return output_file
    
    def generate_report(self, results: Optional[List[BenchmarkResult]] = None) -> str:
        """Generate human-readable benchmark report"""
        if results is None:
            results = self.results
        
        report = []
        report.append("=" * 60)
        report.append("ZK PROOF SYSTEM BENCHMARK REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Group by circuit size
        by_size = {}
        for r in results:
            if r.circuit_size not in by_size:
                by_size[r.circuit_size] = []
            by_size[r.circuit_size].append(r)
        
        for size in CircuitSize:
            if size not in by_size:
                continue
            
            report.append(f"\n{size.value.upper()} CIRCUITS")
            report.append("-" * 40)
            
            size_results = by_size[size]
            
            for r in size_results:
                if r.benchmark_type == BenchmarkType.PROOF_GENERATION:
                    report.append(f"Proof Generation:")
                    report.append(f"  Constraints: {r.num_constraints:,}")
                    report.append(f"  Mean time: {r.mean_time:.3f}s ± {r.std_dev_time:.3f}s")
                    report.append(f"  Throughput: {r.throughput_proofs_per_hour:.0f} proofs/hour")
                    report.append(f"  Peak memory: {r.peak_memory_mb:.1f} MB")
                    report.append(f"  Proof size: {r.proof_size_bytes:,} bytes")
                
                elif r.benchmark_type == BenchmarkType.VERIFICATION:
                    report.append(f"Verification:")
                    report.append(f"  Mean time: {r.mean_time*1000:.3f}ms ± {r.std_dev_time*1000:.3f}ms")
                    report.append(f"  95th percentile: {r.percentile_95*1000:.3f}ms")
                
                elif r.benchmark_type == BenchmarkType.MEMORY:
                    report.append(f"Memory Usage:")
                    report.append(f"  Peak: {r.peak_memory_mb:.1f} MB")
                    report.append(f"  Average: {r.avg_memory_mb:.1f} MB")
                    report.append(f"  Delta: {r.memory_delta_mb:.1f} MB")
                
                elif r.benchmark_type == BenchmarkType.THROUGHPUT:
                    report.append(f"Throughput:")
                    report.append(f"  Rate: {r.throughput_proofs_per_hour:.0f} proofs/hour")
                    report.append(f"  {r.notes}")
        
        report.append("\n" + "=" * 60)
        report.append("SYSTEM INFORMATION")
        report.append("-" * 40)
        if results:
            r = results[0]
            report.append(f"CPU Cores: {r.cpu_count}")
            report.append(f"CPU Frequency: {r.cpu_freq_mhz:.0f} MHz")
            report.append(f"Total Memory: {r.total_memory_gb:.1f} GB")
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = self.output_dir / f"benchmark_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"\nReport saved to: {report_file}")
        return report_text
    
    def compare_results(
        self,
        baseline_file: Path,
        current_results: Optional[List[BenchmarkResult]] = None
    ) -> Dict[str, Any]:
        """Compare current results with baseline"""
        if current_results is None:
            current_results = self.results
        
        # Load baseline
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        baseline_results = [
            BenchmarkResult.from_dict(r) for r in baseline_data['results']
        ]
        
        comparison = {}
        
        # Compare by circuit size and benchmark type
        for current in current_results:
            key = f"{current.circuit_size.value}_{current.benchmark_type.value}"
            
            # Find matching baseline
            baseline = next(
                (b for b in baseline_results 
                 if b.circuit_size == current.circuit_size 
                 and b.benchmark_type == current.benchmark_type),
                None
            )
            
            if baseline:
                comparison[key] = {
                    'circuit_size': current.circuit_size.value,
                    'benchmark_type': current.benchmark_type.value,
                    'current_time': current.mean_time,
                    'baseline_time': baseline.mean_time,
                    'speedup': baseline.mean_time / current.mean_time if current.mean_time > 0 else 0,
                    'regression': current.mean_time > baseline.mean_time * 1.1,  # 10% threshold
                    'memory_change': current.peak_memory_mb - baseline.peak_memory_mb
                }
        
        return comparison
    
    # Mock methods for testing (replace with actual ZK implementation)
    
    def _create_mock_circuit(self, constraints: int, variables: int, public_inputs: int) -> Dict[str, Any]:
        """Create a mock circuit for benchmarking"""
        return {
            'constraints': constraints,
            'variables': variables,
            'public_inputs': public_inputs,
            'gates': list(range(constraints))
        }
    
    def _mock_setup(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Mock setup phase"""
        # Simulate setup computation
        time.sleep(0.001 * (circuit['constraints'] / 1000))
        return {'params': 'mock_params'}
    
    def _generate_mock_witness(self, circuit: Dict[str, Any]) -> List[int]:
        """Generate mock witness"""
        # Simulate witness generation
        time.sleep(0.0001 * circuit['variables'])
        return list(range(circuit['variables']))
    
    def _generate_mock_proof(self, circuit: Dict[str, Any]) -> bytes:
        """Generate a mock proof"""
        # Simulate proof generation time based on circuit size
        time.sleep(0.001 * (circuit['constraints'] / 100))
        return b'mock_proof' * (circuit['constraints'] // 100)
    
    def _generate_mock_proof_with_params(
        self,
        circuit: Dict[str, Any],
        params: Dict[str, Any],
        witness: List[int]
    ) -> bytes:
        """Generate mock proof with parameters"""
        return self._generate_mock_proof(circuit)
    
    def _verify_mock_proof(self, proof: bytes, circuit: Dict[str, Any]) -> bool:
        """Verify a mock proof"""
        # Simulate verification time
        time.sleep(0.00001 * circuit['constraints'])
        return True


def run_quick_benchmark():
    """Run a quick benchmark for testing"""
    suite = ZKBenchmarkSuite()
    
    print("Running quick benchmark...")
    
    # Test tiny circuit
    result = suite.benchmark_proof_generation(CircuitSize.TINY, num_iterations=3)
    print(f"\nTiny circuit proof generation: {result.mean_time:.3f}s")
    
    # Test verification
    result = suite.benchmark_verification(CircuitSize.TINY, num_iterations=10)
    print(f"Tiny circuit verification: {result.mean_time*1000:.3f}ms")
    
    # Generate report
    report = suite.generate_report()
    print("\n" + report)
    
    return suite.results


if __name__ == "__main__":
    run_quick_benchmark()