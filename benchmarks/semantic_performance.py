#!/usr/bin/env python3
"""
Performance benchmarking for semantic verification module.
Measures latency, throughput, and memory usage under various configurations.
"""

import torch
import numpy as np
import time
import psutil
import gc
from pathlib import Path
import json
import argparse
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    def tabulate(data, headers, tablefmt=None):
        """Simple fallback for tabulate."""
        lines = [" | ".join(headers)]
        lines.append("-" * (len(lines[0]) + 10))
        for row in data:
            lines.append(" | ".join(str(x) for x in row))
        return "\n".join(lines)
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.semantic import ConceptLibrary, SemanticMatcher
from pot.semantic.library_optimized import MemoryMappedConceptLibrary, create_optimized_library
from pot.semantic.behavioral_fingerprint import BehavioralFingerprint, create_behavioral_monitor


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    operation: str
    config: Dict[str, Any]
    latency_ms: float
    throughput: float
    memory_mb: float
    gpu_memory_mb: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SemanticBenchmark:
    """Comprehensive benchmark suite for semantic verification."""
    
    def __init__(self, device: str = 'auto', verbose: bool = True):
        """
        Initialize benchmark suite.
        
        Args:
            device: Device to use ('cpu', 'cuda', or 'auto')
            verbose: Print progress messages
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.verbose = verbose
        self.results = []
        
        if self.verbose:
            print(f"Benchmarking on device: {self.device}")
            if self.device.type == 'cuda':
                print(f"GPU: {torch.cuda.get_device_name()}")
    
    def log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"[BENCH] {message}")
    
    def measure_memory(self) -> Tuple[float, float]:
        """Measure current memory usage (CPU and GPU)."""
        # CPU memory
        process = psutil.Process()
        cpu_mb = process.memory_info().rss / 1024 / 1024
        
        # GPU memory
        gpu_mb = 0.0
        if self.device.type == 'cuda':
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        return cpu_mb, gpu_mb
    
    def benchmark_concept_library(self, dims: List[int] = [128, 512, 768, 2048],
                                 n_concepts: List[int] = [10, 50, 100, 500],
                                 n_samples: int = 100) -> List[BenchmarkResult]:
        """
        Benchmark ConceptLibrary operations.
        
        Args:
            dims: Dimensions to test
            n_concepts: Number of concepts to test
            n_samples: Samples per concept
            
        Returns:
            List of benchmark results
        """
        self.log("Benchmarking ConceptLibrary...")
        results = []
        
        for dim in dims:
            for n_concept in n_concepts:
                config = {
                    'dim': dim,
                    'n_concepts': n_concept,
                    'n_samples': n_samples,
                    'method': 'gaussian'
                }
                
                # Create library
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                start_mem_cpu, start_mem_gpu = self.measure_memory()
                
                library = ConceptLibrary(dim=dim, method='gaussian')
                
                # Add concepts
                start_time = time.perf_counter()
                
                for i in range(n_concept):
                    embeddings = torch.randn(n_samples, dim)
                    library.add_concept(f'concept_{i}', embeddings)
                
                add_time = time.perf_counter() - start_time
                
                # Measure retrieval
                start_time = time.perf_counter()
                for i in range(n_concept):
                    _ = library.get_concept_vector(f'concept_{i}')
                retrieve_time = time.perf_counter() - start_time
                
                end_mem_cpu, end_mem_gpu = self.measure_memory()
                
                # Record results
                results.append(BenchmarkResult(
                    operation='library_add',
                    config=config,
                    latency_ms=(add_time / n_concept) * 1000,
                    throughput=n_concept / add_time,
                    memory_mb=end_mem_cpu - start_mem_cpu,
                    gpu_memory_mb=end_mem_gpu - start_mem_gpu
                ))
                
                results.append(BenchmarkResult(
                    operation='library_retrieve',
                    config=config,
                    latency_ms=(retrieve_time / n_concept) * 1000,
                    throughput=n_concept / retrieve_time,
                    memory_mb=0,
                    gpu_memory_mb=0
                ))
                
                self.log(f"  dim={dim}, n_concepts={n_concept}: "
                        f"add={add_time:.3f}s, retrieve={retrieve_time:.3f}s")
        
        return results
    
    def benchmark_semantic_matcher(self, dims: List[int] = [128, 512, 768],
                                  n_concepts: List[int] = [10, 50, 100],
                                  batch_sizes: List[int] = [1, 16, 32, 64]) -> List[BenchmarkResult]:
        """
        Benchmark SemanticMatcher operations.
        
        Args:
            dims: Dimensions to test
            n_concepts: Number of concepts in library
            batch_sizes: Batch sizes to test
            
        Returns:
            List of benchmark results
        """
        self.log("Benchmarking SemanticMatcher...")
        results = []
        
        for dim in dims:
            # Create library with concepts
            library = ConceptLibrary(dim=dim, method='gaussian')
            for i in range(max(n_concepts)):
                embeddings = torch.randn(50, dim)
                library.add_concept(f'concept_{i}', embeddings)
            
            for n_concept in n_concepts:
                for batch_size in batch_sizes:
                    config = {
                        'dim': dim,
                        'n_concepts': n_concept,
                        'batch_size': batch_size,
                        'cache_size': 1000,
                        'use_gpu': self.device.type == 'cuda'
                    }
                    
                    # Create matcher
                    gc.collect()
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    matcher = SemanticMatcher(
                        library=library,
                        cache_size=1000,
                        use_gpu=self.device.type == 'cuda',
                        batch_size=batch_size
                    )
                    
                    # Prepare batch
                    embeddings = torch.randn(batch_size, dim).to(self.device)
                    
                    # Warm up
                    for _ in range(3):
                        _ = matcher.compute_batch_similarities(embeddings, 
                                                              [f'concept_{i}' for i in range(n_concept)])
                    
                    # Measure batch similarity
                    start_time = time.perf_counter()
                    n_iterations = 100
                    
                    for _ in range(n_iterations):
                        similarities = matcher.compute_batch_similarities(
                            embeddings,
                            [f'concept_{i}' for i in range(n_concept)],
                            method='cosine'
                        )
                    
                    batch_time = time.perf_counter() - start_time
                    
                    # Measure single similarity (with caching)
                    single_embedding = embeddings[0]
                    
                    start_time = time.perf_counter()
                    for _ in range(n_iterations):
                        for i in range(min(10, n_concept)):
                            _ = matcher.compute_similarity(single_embedding, f'concept_{i}')
                    
                    single_time = time.perf_counter() - start_time
                    n_single_ops = n_iterations * min(10, n_concept)
                    
                    # Record results
                    results.append(BenchmarkResult(
                        operation='matcher_batch',
                        config=config,
                        latency_ms=(batch_time / n_iterations) * 1000,
                        throughput=(n_iterations * batch_size * n_concept) / batch_time,
                        memory_mb=0,
                        gpu_memory_mb=0
                    ))
                    
                    results.append(BenchmarkResult(
                        operation='matcher_single',
                        config=config,
                        latency_ms=(single_time / n_single_ops) * 1000,
                        throughput=n_single_ops / single_time,
                        memory_mb=0,
                        gpu_memory_mb=0
                    ))
                    
                    self.log(f"  dim={dim}, concepts={n_concept}, batch={batch_size}: "
                            f"batch={batch_time/n_iterations*1000:.2f}ms, "
                            f"single={single_time/n_single_ops*1000:.2f}ms")
        
        return results
    
    def benchmark_optimized_library(self, dims: List[int] = [512, 2048, 10000],
                                   n_concepts: List[int] = [100, 500, 1000]) -> List[BenchmarkResult]:
        """
        Benchmark optimized memory-mapped library.
        
        Args:
            dims: Dimensions to test
            n_concepts: Number of concepts to test
            
        Returns:
            List of benchmark results
        """
        self.log("Benchmarking Optimized Library...")
        results = []
        
        for dim in dims:
            for n_concept in n_concepts:
                for use_sparse in [False, True]:
                    if use_sparse and dim < 1000:
                        continue  # Sparse only makes sense for high dimensions
                    
                    config = {
                        'dim': dim,
                        'n_concepts': n_concept,
                        'use_sparse': use_sparse,
                        'method': 'hypervector' if use_sparse else 'gaussian'
                    }
                    
                    # Create optimized library
                    gc.collect()
                    start_mem_cpu, start_mem_gpu = self.measure_memory()
                    
                    library = create_optimized_library(
                        dim=dim,
                        method='hypervector' if use_sparse else 'gaussian',
                        use_sparse=use_sparse,
                        sparsity=0.99 if use_sparse else 0,
                        max_memory_mb=100
                    )
                    
                    # Add concepts incrementally
                    start_time = time.perf_counter()
                    
                    for i in range(n_concept):
                        for j in range(10):  # 10 samples per concept
                            embedding = torch.randn(dim)
                            library.add_concept_incremental(f'concept_{i}', embedding, batch_update=True)
                        library.finalize_concept(f'concept_{i}')
                    
                    add_time = time.perf_counter() - start_time
                    
                    # Measure retrieval
                    start_time = time.perf_counter()
                    for i in range(min(100, n_concept)):
                        _ = library.get_concept_vector(f'concept_{i}')
                    retrieve_time = time.perf_counter() - start_time
                    n_retrievals = min(100, n_concept)
                    
                    # Measure sparse similarity if applicable
                    if use_sparse:
                        embedding = torch.randn(dim)
                        start_time = time.perf_counter()
                        for i in range(min(100, n_concept)):
                            _ = library.compute_similarity_sparse(embedding, f'concept_{i}')
                        sparse_time = time.perf_counter() - start_time
                        
                        results.append(BenchmarkResult(
                            operation='sparse_similarity',
                            config=config,
                            latency_ms=(sparse_time / n_retrievals) * 1000,
                            throughput=n_retrievals / sparse_time,
                            memory_mb=0,
                            gpu_memory_mb=0
                        ))
                    
                    end_mem_cpu, end_mem_gpu = self.measure_memory()
                    
                    # Record results
                    results.append(BenchmarkResult(
                        operation='optimized_add',
                        config=config,
                        latency_ms=(add_time / n_concept) * 1000,
                        throughput=n_concept / add_time,
                        memory_mb=end_mem_cpu - start_mem_cpu,
                        gpu_memory_mb=end_mem_gpu - start_mem_gpu
                    ))
                    
                    results.append(BenchmarkResult(
                        operation='optimized_retrieve',
                        config=config,
                        latency_ms=(retrieve_time / n_retrievals) * 1000,
                        throughput=n_retrievals / retrieve_time,
                        memory_mb=0,
                        gpu_memory_mb=0
                    ))
                    
                    self.log(f"  dim={dim}, concepts={n_concept}, sparse={use_sparse}: "
                            f"add={add_time:.3f}s, mem={end_mem_cpu-start_mem_cpu:.1f}MB")
        
        return results
    
    def benchmark_behavioral_fingerprint(self, window_sizes: List[int] = [10, 50, 100],
                                        fingerprint_dims: List[int] = [32, 64, 128]) -> List[BenchmarkResult]:
        """
        Benchmark behavioral fingerprinting.
        
        Args:
            window_sizes: Window sizes to test
            fingerprint_dims: Fingerprint dimensions to test
            
        Returns:
            List of benchmark results
        """
        self.log("Benchmarking Behavioral Fingerprint...")
        results = []
        
        for window_size in window_sizes:
            for fp_dim in fingerprint_dims:
                config = {
                    'window_size': window_size,
                    'fingerprint_dim': fp_dim,
                    'decay_factor': 0.95
                }
                
                # Create fingerprint system
                fingerprint = BehavioralFingerprint(
                    window_size=window_size,
                    fingerprint_dim=fp_dim,
                    decay_factor=0.95
                )
                
                # Measure update time
                outputs = [torch.randn(128) for _ in range(window_size * 2)]
                
                start_time = time.perf_counter()
                for output in outputs:
                    fingerprint.update(output)
                update_time = time.perf_counter() - start_time
                
                # Measure fingerprint computation
                start_time = time.perf_counter()
                n_computations = 100
                for _ in range(n_computations):
                    fp = fingerprint.compute_fingerprint(normalize=True)
                compute_time = time.perf_counter() - start_time
                
                # Measure anomaly detection
                reference_fp = fingerprint.compute_fingerprint()
                fingerprint.set_reference(reference_fp, threshold=0.8)
                
                start_time = time.perf_counter()
                for _ in range(n_computations):
                    current_fp = fingerprint.compute_fingerprint()
                    is_anomaly, score = fingerprint.detect_anomaly(current_fp)
                anomaly_time = time.perf_counter() - start_time
                
                # Record results
                results.append(BenchmarkResult(
                    operation='fingerprint_update',
                    config=config,
                    latency_ms=(update_time / len(outputs)) * 1000,
                    throughput=len(outputs) / update_time,
                    memory_mb=0,
                    gpu_memory_mb=0
                ))
                
                results.append(BenchmarkResult(
                    operation='fingerprint_compute',
                    config=config,
                    latency_ms=(compute_time / n_computations) * 1000,
                    throughput=n_computations / compute_time,
                    memory_mb=0,
                    gpu_memory_mb=0
                ))
                
                results.append(BenchmarkResult(
                    operation='anomaly_detect',
                    config=config,
                    latency_ms=(anomaly_time / n_computations) * 1000,
                    throughput=n_computations / anomaly_time,
                    memory_mb=0,
                    gpu_memory_mb=0
                ))
                
                self.log(f"  window={window_size}, dim={fp_dim}: "
                        f"update={update_time/len(outputs)*1000:.2f}ms, "
                        f"compute={compute_time/n_computations*1000:.2f}ms")
        
        return results
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        all_results = []
        
        # Concept Library
        all_results.extend(self.benchmark_concept_library(
            dims=[128, 512, 768],
            n_concepts=[10, 50, 100],
            n_samples=50
        ))
        
        # Semantic Matcher
        all_results.extend(self.benchmark_semantic_matcher(
            dims=[128, 512],
            n_concepts=[10, 50],
            batch_sizes=[1, 16, 32]
        ))
        
        # Optimized Library
        all_results.extend(self.benchmark_optimized_library(
            dims=[512, 2048],
            n_concepts=[100, 500]
        ))
        
        # Behavioral Fingerprint
        all_results.extend(self.benchmark_behavioral_fingerprint(
            window_sizes=[10, 50, 100],
            fingerprint_dims=[32, 64]
        ))
        
        self.results = all_results
        return all_results
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate performance report.
        
        Args:
            output_path: Path to save report (optional)
            
        Returns:
            Report as string
        """
        if not self.results:
            return "No benchmark results available"
        
        # Group results by operation
        operations = {}
        for result in self.results:
            op = result.operation
            if op not in operations:
                operations[op] = []
            operations[op].append(result)
        
        # Create report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SEMANTIC VERIFICATION PERFORMANCE BENCHMARK REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Device: {self.device}")
        if self.device.type == 'cuda':
            report_lines.append(f"GPU: {torch.cuda.get_device_name()}")
        report_lines.append("")
        
        # Summary table for each operation
        for op_name, op_results in operations.items():
            report_lines.append(f"\n{op_name.upper()}")
            report_lines.append("-" * 40)
            
            # Create table
            table_data = []
            for result in op_results[:10]:  # Show top 10
                config_str = ', '.join([f"{k}={v}" for k, v in result.config.items() 
                                       if k in ['dim', 'n_concepts', 'batch_size', 'window_size']])
                table_data.append([
                    config_str,
                    f"{result.latency_ms:.2f}",
                    f"{result.throughput:.1f}",
                    f"{result.memory_mb:.1f}"
                ])
            
            headers = ["Configuration", "Latency (ms)", "Throughput (ops/s)", "Memory (MB)"]
            report_lines.append(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Performance insights
        report_lines.append("\n" + "=" * 80)
        report_lines.append("PERFORMANCE INSIGHTS")
        report_lines.append("=" * 80)
        
        # Find best configurations
        for op_name in ['matcher_batch', 'library_add', 'fingerprint_compute']:
            op_results = [r for r in self.results if r.operation == op_name]
            if op_results:
                best = min(op_results, key=lambda r: r.latency_ms)
                report_lines.append(f"\nBest {op_name}: {best.latency_ms:.2f}ms")
                report_lines.append(f"  Config: {best.config}")
        
        report = "\n".join(report_lines)
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save text report
            with open(output_path, 'w') as f:
                f.write(report)
            
            # Save JSON results
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump([r.to_dict() for r in self.results], f, indent=2)
            
            print(f"Report saved to {output_path}")
            print(f"JSON results saved to {json_path}")
        
        return report
    
    def plot_results(self, save_path: Optional[Path] = None):
        """
        Create visualization plots for benchmark results.
        
        Args:
            save_path: Path to save plots
        """
        if not self.results:
            print("No results to plot")
            return
        
        # Set style
        if HAS_SEABORN:
            sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Latency vs Dimension
        ax = axes[0, 0]
        for op in ['library_add', 'matcher_single', 'fingerprint_compute']:
            op_results = [r for r in self.results if r.operation == op]
            if op_results:
                dims = [r.config.get('dim', r.config.get('fingerprint_dim', 0)) for r in op_results]
                latencies = [r.latency_ms for r in op_results]
                if dims and latencies:
                    ax.scatter(dims, latencies, label=op, alpha=0.7)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency vs Dimension')
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Plot 2: Throughput vs Batch Size
        ax = axes[0, 1]
        batch_results = [r for r in self.results if r.operation == 'matcher_batch']
        if batch_results:
            batch_sizes = [r.config.get('batch_size', 1) for r in batch_results]
            throughputs = [r.throughput for r in batch_results]
            ax.scatter(batch_sizes, throughputs, alpha=0.7)
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (ops/s)')
            ax.set_title('Batch Processing Throughput')
        
        # Plot 3: Memory Usage
        ax = axes[1, 0]
        memory_ops = ['library_add', 'optimized_add']
        for op in memory_ops:
            op_results = [r for r in self.results if r.operation == op]
            if op_results:
                n_concepts = [r.config.get('n_concepts', 0) for r in op_results]
                memory = [r.memory_mb for r in op_results]
                if n_concepts and memory:
                    ax.scatter(n_concepts, memory, label=op, alpha=0.7)
        ax.set_xlabel('Number of Concepts')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage Scaling')
        ax.legend()
        
        # Plot 4: Operation Comparison
        ax = axes[1, 1]
        op_names = list(set(r.operation for r in self.results))
        avg_latencies = []
        for op in op_names:
            op_results = [r for r in self.results if r.operation == op]
            if op_results:
                avg_latencies.append(np.mean([r.latency_ms for r in op_results]))
            else:
                avg_latencies.append(0)
        
        ax.barh(op_names, avg_latencies)
        ax.set_xlabel('Average Latency (ms)')
        ax.set_title('Operation Performance Comparison')
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        
        plt.show()


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description='Semantic Verification Performance Benchmark')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use for benchmarking')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark with reduced parameters')
    parser.add_argument('--output', type=str, default='benchmark_report.txt',
                       help='Output path for report')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = SemanticBenchmark(device=args.device, verbose=True)
    
    # Run benchmarks
    if args.quick:
        print("\nRunning QUICK benchmark...")
        results = []
        results.extend(benchmark.benchmark_concept_library(
            dims=[128, 512],
            n_concepts=[10, 50],
            n_samples=20
        ))
        results.extend(benchmark.benchmark_semantic_matcher(
            dims=[128],
            n_concepts=[10],
            batch_sizes=[1, 16]
        ))
        results.extend(benchmark.benchmark_behavioral_fingerprint(
            window_sizes=[10, 50],
            fingerprint_dims=[32]
        ))
        benchmark.results = results
    else:
        print("\nRunning FULL benchmark...")
        benchmark.run_all_benchmarks()
    
    # Generate report
    report = benchmark.generate_report(output_path=args.output)
    print("\n" + report)
    
    # Generate plots
    if args.plot:
        plot_path = Path(args.output).with_suffix('.png')
        benchmark.plot_results(save_path=plot_path)


if __name__ == "__main__":
    main()