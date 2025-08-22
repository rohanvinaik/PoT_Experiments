#!/usr/bin/env python3
"""
ZK Proof System Benchmark Runner

Comprehensive benchmarking script for ZK proof generation and verification.
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pot.zk.benchmarks.benchmark_suite import (
    ZKBenchmarkSuite,
    CircuitSize,
    BenchmarkType
)
from src.pot.zk.optimizations.circuit_optimizer import CircuitOptimizer
from src.pot.zk.optimizations.parallel_witness_generation import ParallelWitnessGenerator
from src.pot.zk.optimizations.batch_prover import BatchProver, ProofRequest
from src.pot.zk.robustness.adversarial_tests import AdversarialTester
from src.pot.zk.artifacts.artifact_manager import ArtifactManager


def run_performance_benchmarks(args):
    """Run performance benchmarks"""
    print("=" * 60)
    print("ZK PERFORMANCE BENCHMARKS")
    print("=" * 60)
    print(f"Circuit sizes: {args.circuit_sizes}")
    print(f"Iterations: {args.iterations}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Create benchmark suite
    suite = ZKBenchmarkSuite(output_dir=Path(args.output_dir))
    
    # Map string sizes to enum
    size_map = {
        'tiny': CircuitSize.TINY,
        'small': CircuitSize.SMALL,
        'medium': CircuitSize.MEDIUM,
        'large': CircuitSize.LARGE,
        'huge': CircuitSize.HUGE
    }
    
    results = []
    
    for size_str in args.circuit_sizes:
        if size_str not in size_map:
            print(f"Warning: Unknown circuit size '{size_str}', skipping")
            continue
        
        circuit_size = size_map[size_str]
        
        print(f"\nBenchmarking {circuit_size.value} circuits...")
        print("-" * 40)
        
        # Proof generation benchmark
        if 'proof' in args.benchmark_types or 'all' in args.benchmark_types:
            try:
                result = suite.benchmark_proof_generation(
                    circuit_size,
                    num_iterations=args.iterations,
                    warmup_iterations=args.warmup
                )
                results.append(result)
                print(f"✓ Proof generation: {result.mean_time:.3f}s ± {result.std_dev_time:.3f}s")
            except Exception as e:
                print(f"✗ Proof generation failed: {e}")
        
        # Verification benchmark
        if 'verify' in args.benchmark_types or 'all' in args.benchmark_types:
            try:
                result = suite.benchmark_verification(
                    circuit_size,
                    num_iterations=args.iterations * 10  # More iterations for fast operation
                )
                results.append(result)
                print(f"✓ Verification: {result.mean_time*1000:.3f}ms ± {result.std_dev_time*1000:.3f}ms")
            except Exception as e:
                print(f"✗ Verification failed: {e}")
        
        # Memory benchmark
        if 'memory' in args.benchmark_types or 'all' in args.benchmark_types:
            try:
                result = suite.benchmark_memory_usage(circuit_size)
                results.append(result)
                print(f"✓ Peak memory: {result.peak_memory_mb:.1f}MB")
            except Exception as e:
                print(f"✗ Memory benchmark failed: {e}")
        
        # Throughput benchmark
        if 'throughput' in args.benchmark_types or 'all' in args.benchmark_types:
            if circuit_size in [CircuitSize.TINY, CircuitSize.SMALL]:
                try:
                    result = suite.benchmark_throughput(
                        circuit_size,
                        duration_seconds=args.throughput_duration
                    )
                    results.append(result)
                    print(f"✓ Throughput: {result.throughput_proofs_per_hour:.0f} proofs/hour")
                except Exception as e:
                    print(f"✗ Throughput benchmark failed: {e}")
    
    # Save results
    if results:
        output_file = suite.save_results(results)
        report = suite.generate_report(results)
        print("\n" + report)
        
        # Store as artifact if requested
        if args.store_artifact:
            artifact_manager = ArtifactManager(Path(args.output_dir) / "artifacts")
            artifact_id = artifact_manager.store_artifact(
                [r.to_dict() for r in results],
                tags=args.tags.split(',') if args.tags else None
            )
            print(f"\nArtifact stored: {artifact_id}")
    
    return results


def run_optimization_benchmarks(args):
    """Run circuit optimization benchmarks"""
    print("=" * 60)
    print("CIRCUIT OPTIMIZATION BENCHMARKS")
    print("=" * 60)
    
    optimizer = CircuitOptimizer()
    
    # Test different circuit sizes
    test_circuits = [
        {'constraints': 1000, 'variables': 500},
        {'constraints': 10000, 'variables': 5000},
        {'constraints': 100000, 'variables': 50000},
    ]
    
    for circuit in test_circuits:
        print(f"\nOptimizing circuit with {circuit['constraints']} constraints...")
        
        # Apply optimizations
        optimized_circuit, optimization_results = optimizer.optimize_circuit(circuit)
        
        for result in optimization_results:
            print(f"  {result}")
        
        # Analyze complexity
        analysis = optimizer.analyze_circuit_complexity(optimized_circuit)
        print(f"  Final complexity: {analysis['constraints']} constraints")
        
        # Get recommendations
        recommendations = optimizer.get_optimization_recommendations(optimized_circuit)
        if recommendations:
            print("  Recommendations:")
            for rec in recommendations:
                print(f"    - {rec}")
    
    print(f"\nOptimization statistics:")
    print(f"  Total optimizations: {optimizer.stats['total_optimizations']}")
    print(f"  Average reduction: {optimizer.stats['total_reduction'] / optimizer.stats['total_optimizations']:.1f}%")
    print(f"  Total time: {optimizer.stats['total_time']:.3f}s")


def run_parallel_benchmarks(args):
    """Run parallel processing benchmarks"""
    print("=" * 60)
    print("PARALLEL PROCESSING BENCHMARKS")
    print("=" * 60)
    
    # Test parallel witness generation
    print("\n1. Parallel Witness Generation")
    print("-" * 40)
    
    generator = ParallelWitnessGenerator()
    
    test_circuits = [
        {'constraints': 1000, 'variables': 500},
        {'constraints': 10000, 'variables': 5000},
        {'constraints': 100000, 'variables': 50000},
    ]
    
    for circuit in test_circuits:
        print(f"\nCircuit size: {circuit['constraints']} constraints")
        
        # Find optimal worker count
        if args.optimize_workers:
            optimal_workers = generator.optimize_worker_count(circuit)
            print(f"  Optimal workers: {optimal_workers}")
        
        # Generate witness
        witness, gen_time = generator.generate_witness(
            circuit,
            inputs={'test': True},
            use_multiprocessing=args.use_multiprocessing
        )
        
        stats = generator.get_statistics()
        print(f"  Generation time: {gen_time:.3f}s")
        print(f"  Parallel speedup: {stats['parallel_speedup']:.2f}x")
    
    # Test batch proving
    print("\n2. Batch Proving")
    print("-" * 40)
    
    prover = BatchProver(batch_size=args.batch_size)
    
    # Create test requests
    requests = []
    for i in range(args.batch_size * 2):
        requests.append(ProofRequest(
            request_id=f"proof_{i}",
            circuit={'constraints': 1000},
            witness=list(range(100)),
            public_inputs=list(range(10)),
            priority=i % 3
        ))
    
    # Benchmark batch proving
    start_time = time.perf_counter()
    results = prover.generate_proof_batch(requests)
    batch_time = time.perf_counter() - start_time
    
    successful = sum(1 for r in results if r.success)
    
    print(f"\nBatch results:")
    print(f"  Total proofs: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Batch time: {batch_time:.3f}s")
    print(f"  Throughput: {len(results)/batch_time:.1f} proofs/sec")
    
    stats = prover.get_statistics()
    print(f"  Cache hit rate: {stats['cache_hit_rate']*100:.1f}%")
    print(f"  Batch efficiency: {stats['batch_efficiency']:.2f}x")


def run_robustness_tests(args):
    """Run robustness and security tests"""
    print("=" * 60)
    print("ROBUSTNESS & SECURITY TESTS")
    print("=" * 60)
    
    tester = AdversarialTester()
    
    # Create test circuit
    circuit = {
        'constraints': 1000,
        'variables': 500,
        'public_inputs': 10
    }
    
    # Run adversarial tests
    results = tester.run_all_tests(circuit)
    
    # Generate report
    report = tester.generate_report()
    print("\n" + report)
    
    # Save results if requested
    if args.output_dir:
        output_file = Path(args.output_dir) / f"robustness_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")
    
    return tester.vulnerability_count == 0


def compare_artifacts(args):
    """Compare benchmark artifacts"""
    print("=" * 60)
    print("ARTIFACT COMPARISON")
    print("=" * 60)
    
    manager = ArtifactManager(Path(args.output_dir) / "artifacts")
    
    if args.list_artifacts:
        # List available artifacts
        artifacts = manager.list_artifacts(limit=20)
        print("\nAvailable artifacts:")
        for artifact in artifacts:
            print(f"  {artifact['artifact_id']} - {artifact['benchmark_type']} ({artifact['circuit_size']})")
            print(f"    Timestamp: {datetime.fromtimestamp(artifact['timestamp'])}")
            print(f"    Commit: {artifact.get('git_commit', 'unknown')}")
    
    elif args.artifact1 and args.artifact2:
        # Compare two artifacts
        print(f"\nComparing:")
        print(f"  Artifact 1: {args.artifact1}")
        print(f"  Artifact 2: {args.artifact2}")
        
        comparison = manager.compare_artifacts(
            args.artifact1,
            args.artifact2,
            save_plots=True
        )
        
        print("\nComparison Results:")
        
        if comparison['improvements']:
            print("\n✓ Improvements:")
            for key, data in comparison['improvements'].items():
                print(f"  {key}: {abs(data['time_change_pct']):.1f}% faster")
        
        if comparison['regressions']:
            print("\n✗ Regressions:")
            for key, data in comparison['regressions'].items():
                print(f"  {key}: {data['time_change_pct']:.1f}% slower")
    
    # Check for recent regressions
    if args.check_regressions:
        print("\nChecking for recent regressions...")
        regressions = manager.detect_regressions(threshold_pct=args.regression_threshold)
        
        if regressions:
            print(f"\n⚠️  Found {len(regressions)} regressions:")
            for reg in regressions:
                print(f"  {reg['benchmark_type']} ({reg['circuit_size']}): "
                      f"{reg['regression_pct']:.1f}% slower")
        else:
            print("✓ No regressions detected")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run ZK proof system benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Benchmark commands')
    
    # Performance benchmarks
    perf_parser = subparsers.add_parser('performance', help='Run performance benchmarks')
    perf_parser.add_argument(
        '--circuit-sizes',
        nargs='+',
        default=['tiny', 'small', 'medium'],
        choices=['tiny', 'small', 'medium', 'large', 'huge'],
        help='Circuit sizes to benchmark'
    )
    perf_parser.add_argument(
        '--benchmark-types',
        nargs='+',
        default=['all'],
        choices=['proof', 'verify', 'memory', 'throughput', 'all'],
        help='Types of benchmarks to run'
    )
    perf_parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of iterations per benchmark'
    )
    perf_parser.add_argument(
        '--warmup',
        type=int,
        default=2,
        help='Number of warmup iterations'
    )
    perf_parser.add_argument(
        '--throughput-duration',
        type=int,
        default=30,
        help='Duration for throughput tests (seconds)'
    )
    perf_parser.add_argument(
        '--store-artifact',
        action='store_true',
        help='Store results as artifact'
    )
    perf_parser.add_argument(
        '--tags',
        type=str,
        help='Comma-separated tags for artifact'
    )
    
    # Optimization benchmarks
    opt_parser = subparsers.add_parser('optimize', help='Run optimization benchmarks')
    
    # Parallel benchmarks
    parallel_parser = subparsers.add_parser('parallel', help='Run parallel processing benchmarks')
    parallel_parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for batch proving'
    )
    parallel_parser.add_argument(
        '--optimize-workers',
        action='store_true',
        help='Find optimal worker count'
    )
    parallel_parser.add_argument(
        '--use-multiprocessing',
        action='store_true',
        help='Use multiprocessing instead of threading'
    )
    
    # Robustness tests
    robust_parser = subparsers.add_parser('robustness', help='Run robustness tests')
    
    # Artifact comparison
    compare_parser = subparsers.add_parser('compare', help='Compare benchmark artifacts')
    compare_parser.add_argument(
        '--list-artifacts',
        action='store_true',
        help='List available artifacts'
    )
    compare_parser.add_argument(
        '--artifact1',
        type=str,
        help='First artifact ID for comparison'
    )
    compare_parser.add_argument(
        '--artifact2',
        type=str,
        help='Second artifact ID for comparison'
    )
    compare_parser.add_argument(
        '--check-regressions',
        action='store_true',
        help='Check for recent performance regressions'
    )
    compare_parser.add_argument(
        '--regression-threshold',
        type=float,
        default=10.0,
        help='Regression threshold percentage'
    )
    
    # Quick benchmark
    quick_parser = subparsers.add_parser('quick', help='Run quick benchmark suite')
    
    # Common arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmarks/results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run appropriate benchmark
    if args.command == 'performance':
        results = run_performance_benchmarks(args)
        sys.exit(0 if results else 1)
    
    elif args.command == 'optimize':
        run_optimization_benchmarks(args)
    
    elif args.command == 'parallel':
        run_parallel_benchmarks(args)
    
    elif args.command == 'robustness':
        success = run_robustness_tests(args)
        sys.exit(0 if success else 1)
    
    elif args.command == 'compare':
        compare_artifacts(args)
    
    elif args.command == 'quick':
        # Run quick benchmark suite
        print("Running quick benchmark suite...")
        suite = ZKBenchmarkSuite(output_dir=Path(args.output_dir))
        results = suite.run_comprehensive_benchmark()
        print(f"\n✅ Completed {len(results)} benchmarks")
        sys.exit(0)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()