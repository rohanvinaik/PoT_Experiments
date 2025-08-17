#!/usr/bin/env python3
"""
Example demonstrating vision verification benchmarking and robustness evaluation.
Shows how to use the comprehensive benchmarking suite for model evaluation.
"""

import torch
import torch.nn as nn
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append('/Users/rohanvinaik/PoT_Experiments')

def create_example_models():
    """Create example models with different architectures for comparison."""
    
    # Lightweight CNN
    lightweight = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(16 * 64, 10)
    )
    
    # Standard CNN
    standard = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64 * 16, 10)
    )
    
    # Heavy CNN
    heavy = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(256 * 4, 10)
    )
    
    return {
        'LightweightCNN': lightweight,
        'StandardCNN': standard,
        'HeavyCNN': heavy
    }

def demo_vision_benchmark():
    """Demonstrate comprehensive vision benchmarking."""
    print("=" * 80)
    print("VISION VERIFICATION BENCHMARKING DEMO")
    print("=" * 80)
    
    try:
        from pot.vision.benchmark import VisionBenchmark
        
        # Initialize benchmark suite
        benchmark = VisionBenchmark(device='cpu')
        print("✓ Vision benchmark suite initialized")
        
        # Create example models
        models = create_example_models()
        print(f"✓ Created {len(models)} example models")
        
        print("\nModel architectures:")
        for name, model in models.items():
            params = sum(p.numel() for p in model.parameters())
            print(f"  {name}: {params:,} parameters")
        
        # Demo 1: Single model comprehensive evaluation
        print("\n" + "=" * 60)
        print("DEMO 1: SINGLE MODEL EVALUATION")
        print("=" * 60)
        
        model = models['StandardCNN']
        
        print("Running comprehensive benchmark on StandardCNN...")
        results = benchmark.run_benchmark(
            model=model,
            model_name='StandardCNN_Demo',
            benchmark_level='intermediate',
            calibrate=True,
            num_calibration_samples=30,
            warmup_runs=2,
            measure_memory=True
        )
        
        print("\nSingle Model Results:")
        print(f"  Overall Success Rate: {results['success_rate'].mean():.3f}")
        print(f"  Average Confidence: {results['confidence'].mean():.3f}")
        print(f"  Average Throughput: {results['throughput'].mean():.1f} challenges/sec")
        if results['memory_usage'].notna().any():
            print(f"  Memory Usage: {results['memory_usage'].mean():.1f} MB")
        
        # Demo 2: Model comparison
        print("\n" + "=" * 60)
        print("DEMO 2: MODEL COMPARISON")
        print("=" * 60)
        
        print("Comparing all models on intermediate benchmark...")
        comparison_results = benchmark.compare_models(
            models=models,
            benchmark_level='intermediate',
            calibrate=True,
            num_calibration_samples=20,
            warmup_runs=1,
            measure_memory=True
        )
        
        print("\nModel Comparison Summary:")
        model_summary = comparison_results.groupby('model_name').agg({
            'success_rate': 'mean',
            'confidence': 'mean',
            'throughput': 'mean',
            'memory_usage': 'mean'
        }).round(3)
        
        print(model_summary.to_string())
        
        # Performance ranking
        model_scores = model_summary.copy()
        
        # Normalize metrics for ranking (higher is better)
        model_scores['norm_throughput'] = model_scores['throughput'] / model_scores['throughput'].max()
        model_scores['norm_memory'] = 1 - (model_scores['memory_usage'] / model_scores['memory_usage'].max())
        
        # Composite score
        model_scores['composite_score'] = (
            0.4 * model_scores['success_rate'] +
            0.3 * model_scores['confidence'] +
            0.2 * model_scores['norm_throughput'] +
            0.1 * model_scores['norm_memory']
        )
        
        model_scores = model_scores.sort_values('composite_score', ascending=False)
        
        print("\nPerformance Ranking:")
        for i, (model, row) in enumerate(model_scores.iterrows()):
            print(f"  {i+1}. {model}: {row['composite_score']:.3f}")
        
        # Demo 3: Report generation
        print("\n" + "=" * 60)
        print("DEMO 3: REPORT GENERATION")
        print("=" * 60)
        
        output_dir = Path('/tmp/vision_benchmark_demo')
        output_dir.mkdir(exist_ok=True)
        
        report_path = benchmark.generate_report(
            results=comparison_results,
            output_path=str(output_dir / 'model_comparison_report.html'),
            include_plots=True
        )
        
        print(f"✓ Comprehensive report generated: {report_path}")
        
        # Check generated files
        files = list(output_dir.glob('model_comparison_report.*'))
        print(f"✓ Generated {len(files)} report files:")
        for file in files:
            print(f"  - {file.name}")
        
        # Demo 4: Benchmark state management
        print("\n" + "=" * 60)
        print("DEMO 4: BENCHMARK STATE MANAGEMENT")
        print("=" * 60)
        
        state_file = output_dir / 'benchmark_state.json'
        benchmark.save_benchmark_state(str(state_file))
        print(f"✓ Benchmark state saved to {state_file}")
        
        # Load and analyze historical data
        new_benchmark = VisionBenchmark(device='cpu')
        new_benchmark.load_benchmark_state(str(state_file))
        
        print(f"✓ Loaded benchmark history:")
        print(f"  - Total results: {len(new_benchmark.results)}")
        print(f"  - History entries: {len(new_benchmark.benchmark_history)}")
        
        for i, entry in enumerate(new_benchmark.benchmark_history):
            print(f"  - Entry {i+1}: {entry['model_name']} ({len(entry['results'])} results)")
        
        return True
        
    except Exception as e:
        print(f"✗ Vision benchmark demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_robustness_evaluation():
    """Demonstrate robustness evaluation capabilities."""
    print("\n" + "=" * 80)
    print("ROBUSTNESS EVALUATION DEMO")
    print("=" * 80)
    
    try:
        from pot.vision.benchmark import VisionRobustnessEvaluator
        from pot.vision.verifier import EnhancedVisionVerifier
        
        # Select model for robustness testing
        models = create_example_models()
        test_model = models['StandardCNN']
        
        # Create verifier and robustness evaluator
        verifier_config = {
            'device': 'cpu',
            'verification_method': 'batch'
        }
        verifier = EnhancedVisionVerifier(test_model, verifier_config, device='cpu')
        evaluator = VisionRobustnessEvaluator(verifier, device='cpu')
        
        print(f"✓ Created robustness evaluator for StandardCNN")
        
        # Demo 1: Noise robustness
        print("\n" + "=" * 60)
        print("DEMO 1: NOISE ROBUSTNESS EVALUATION")
        print("=" * 60)
        
        print("Testing robustness to additive Gaussian noise...")
        noise_results = evaluator.evaluate_noise_robustness(
            noise_levels=[0.005, 0.01, 0.02, 0.05, 0.1],
            num_trials=10,
            challenge_types=['frequency', 'texture']
        )
        
        print("\nNoise Robustness Results:")
        for test_name, result in noise_results.items():
            print(f"  {test_name}:")
            print(f"    Success Rate: {result.success_rate:.3f} (±{result.std_dev:.3f})")
            print(f"    Robustness Score: {result.robustness_score:.3f}")
            print(f"    Baseline: {result.baseline_success:.3f}")
        
        # Demo 2: Transformation robustness
        print("\n" + "=" * 60)
        print("DEMO 2: TRANSFORMATION ROBUSTNESS")
        print("=" * 60)
        
        print("Testing robustness to image transformations...")
        transform_results = evaluator.evaluate_transformation_robustness(
            num_trials=10,
            challenge_types=['frequency', 'texture']
        )
        
        print("\nTransformation Robustness Results:")
        
        # Group by challenge type
        freq_results = {k: v for k, v in transform_results.items() if 'frequency' in k}
        texture_results = {k: v for k, v in transform_results.items() if 'texture' in k}
        
        print("\n  Frequency Challenges:")
        for test_name, result in sorted(freq_results.items(), key=lambda x: x[1].robustness_score, reverse=True):
            transform_name = test_name.replace('frequency_', '')
            print(f"    {transform_name}: {result.robustness_score:.3f}")
        
        print("\n  Texture Challenges:")
        for test_name, result in sorted(texture_results.items(), key=lambda x: x[1].robustness_score, reverse=True):
            transform_name = test_name.replace('texture_', '')
            print(f"    {transform_name}: {result.robustness_score:.3f}")
        
        # Demo 3: Adversarial robustness
        print("\n" + "=" * 60)
        print("DEMO 3: ADVERSARIAL ROBUSTNESS")
        print("=" * 60)
        
        print("Testing robustness to adversarial perturbations...")
        try:
            adv_results = evaluator.evaluate_adversarial_robustness(
                epsilon_values=[0.01, 0.03, 0.05],
                attack_steps=10,
                num_trials=5,
                challenge_types=['frequency']
            )
            
            print("\nAdversarial Robustness Results:")
            for test_name, result in adv_results.items():
                epsilon = test_name.split('_')[-1]
                print(f"  Epsilon {epsilon}:")
                print(f"    Success Rate: {result.success_rate:.3f}")
                print(f"    Robustness Score: {result.robustness_score:.3f}")
        
        except Exception as e:
            print(f"  ⚠ Adversarial testing failed (expected for simple models): {e}")
        
        # Demo 4: Comprehensive robustness report
        print("\n" + "=" * 60)
        print("DEMO 4: ROBUSTNESS REPORT GENERATION")
        print("=" * 60)
        
        output_dir = Path('/tmp/robustness_demo')
        output_dir.mkdir(exist_ok=True)
        
        # Combine all robustness results
        all_robustness_results = {**noise_results, **transform_results}
        
        report_path = evaluator.generate_robustness_report(
            results=all_robustness_results,
            output_path=str(output_dir / 'robustness_evaluation_report.html')
        )
        
        print(f"✓ Robustness report generated: {report_path}")
        
        # Summary statistics
        robustness_scores = [r.robustness_score for r in all_robustness_results.values()]
        
        print(f"\nRobustness Summary:")
        print(f"  Total tests: {len(all_robustness_results)}")
        print(f"  Average robustness score: {np.mean(robustness_scores):.3f}")
        print(f"  Best robustness score: {max(robustness_scores):.3f}")
        print(f"  Worst robustness score: {min(robustness_scores):.3f}")
        
        # Identify most vulnerable aspects
        worst_tests = sorted(all_robustness_results.items(), key=lambda x: x[1].robustness_score)[:3]
        print(f"\nMost vulnerable to:")
        for test_name, result in worst_tests:
            print(f"  - {test_name}: {result.robustness_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Robustness evaluation demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_performance_analysis():
    """Demonstrate performance analysis and optimization insights."""
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS DEMO")
    print("=" * 80)
    
    try:
        from pot.vision.benchmark import VisionBenchmark
        
        models = create_example_models()
        benchmark = VisionBenchmark(device='cpu')
        
        # Run stress test
        print("Running stress test benchmark...")
        stress_results = benchmark.run_benchmark(
            model=models['StandardCNN'],
            model_name='StressTest',
            benchmark_level='stress',
            calibrate=True,
            num_calibration_samples=50,
            warmup_runs=3,
            measure_memory=True
        )
        
        print("\nStress Test Results:")
        print(f"  Total challenges: {stress_results['num_challenges'].sum()}")
        print(f"  Total time: {stress_results['total_time'].sum():.2f} seconds")
        print(f"  Average throughput: {stress_results['throughput'].mean():.1f} challenges/sec")
        print(f"  Success rate: {stress_results['success_rate'].mean():.3f}")
        
        # Performance bottleneck analysis
        print("\nPerformance Analysis:")
        
        # Time per challenge type
        time_by_type = stress_results.groupby('challenge_type')['avg_inference_time'].mean()
        print("\n  Average inference time by challenge type:")
        for challenge_type, avg_time in time_by_type.items():
            print(f"    {challenge_type}: {avg_time*1000:.1f} ms")
        
        # Throughput analysis
        throughput_by_type = stress_results.groupby('challenge_type')['throughput'].mean()
        print("\n  Throughput by challenge type:")
        for challenge_type, throughput in throughput_by_type.items():
            print(f"    {challenge_type}: {throughput:.1f} challenges/sec")
        
        # Memory efficiency
        if stress_results['memory_usage'].notna().any():
            memory_by_type = stress_results.groupby('challenge_type')['memory_usage'].mean()
            print("\n  Memory usage by challenge type:")
            for challenge_type, memory in memory_by_type.items():
                print(f"    {challenge_type}: {memory:.1f} MB")
        
        # Recommendations
        print("\nOptimization Recommendations:")
        
        slowest_type = time_by_type.idxmax()
        fastest_type = time_by_type.idxmin()
        
        print(f"  - {slowest_type} challenges are slowest ({time_by_type[slowest_type]*1000:.1f} ms)")
        print(f"  - {fastest_type} challenges are fastest ({time_by_type[fastest_type]*1000:.1f} ms)")
        print(f"  - Consider optimizing {slowest_type} challenge processing")
        
        if stress_results['success_rate'].min() < 0.5:
            worst_performing = stress_results.loc[stress_results['success_rate'].idxmin()]
            print(f"  - {worst_performing['challenge_type']} has low success rate ({worst_performing['success_rate']:.3f})")
            print(f"  - Consider recalibrating or adjusting thresholds")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance analysis demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive benchmarking demonstration."""
    print("Vision Verification Benchmarking and Evaluation Demo")
    print("=" * 80)
    
    demos = [
        ("Vision Benchmark", demo_vision_benchmark),
        ("Robustness Evaluation", demo_robustness_evaluation),
        ("Performance Analysis", demo_performance_analysis),
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        print(f"\nStarting {demo_name} Demo...")
        try:
            result = demo_func()
            results.append((demo_name, result))
            if result:
                print(f"✓ {demo_name} demo completed successfully")
            else:
                print(f"✗ {demo_name} demo failed")
        except Exception as e:
            print(f"✗ {demo_name} demo crashed: {e}")
            results.append((demo_name, False))
    
    # Final summary
    print("\n" + "=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    
    passed = 0
    for demo_name, result in results:
        status = "✓ SUCCESS" if result else "✗ FAILED"
        print(f"  {demo_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nDemos completed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n✓ All demonstrations completed successfully!")
        print("\nGenerated files can be found in:")
        print("  - /tmp/vision_benchmark_demo/")
        print("  - /tmp/robustness_demo/")
        print("\nThese files include HTML reports, CSV data, and JSON summaries.")
    else:
        print("\n⚠ Some demonstrations failed")
    
    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    exit(main())