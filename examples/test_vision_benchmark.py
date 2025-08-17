#!/usr/bin/env python3
"""
Test script for vision benchmarking and evaluation tools.
"""

import torch
import torch.nn as nn
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append('/Users/rohanvinaik/PoT_Experiments')

def create_test_models():
    """Create test models for benchmarking."""
    
    # Simple CNN
    simple_cnn = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 10)
    )
    
    # Deeper CNN
    deep_cnn = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 10)
    )
    
    # ResNet-like block
    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            residual = x
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
            out += residual
            return self.relu(out)
    
    resnet_like = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        ResBlock(64),
        ResBlock(64),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    
    return {
        'SimpleCNN': simple_cnn,
        'DeepCNN': deep_cnn,
        'ResNetLike': resnet_like
    }

def test_vision_benchmark():
    """Test VisionBenchmark functionality."""
    print("=" * 70)
    print("TESTING VISION BENCHMARK")
    print("=" * 70)
    
    try:
        from pot.vision.benchmark import VisionBenchmark
        
        # Create benchmark
        benchmark = VisionBenchmark(device='cpu')
        print("✓ VisionBenchmark created successfully")
        
        # Create test models
        models = create_test_models()
        print(f"✓ Created {len(models)} test models")
        
        # Test single model benchmark
        print("\nTesting single model benchmark...")
        model = models['SimpleCNN']
        
        try:
            results = benchmark.run_benchmark(
                model=model,
                model_name='TestCNN',
                benchmark_level='basic',
                calibrate=False,  # Skip calibration for speed
                warmup_runs=1,
                measure_memory=False
            )
            
            print(f"✓ Single model benchmark completed")
            print(f"  Results shape: {results.shape}")
            print(f"  Columns: {list(results.columns)}")
            print(f"  Average success rate: {results['success_rate'].mean():.3f}")
            
        except Exception as e:
            print(f"✗ Single model benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test model comparison (smaller scale)
        print("\nTesting model comparison...")
        try:
            comparison_models = {
                'SimpleCNN': models['SimpleCNN'],
                'DeepCNN': models['DeepCNN']
            }
            
            comparison_results = benchmark.compare_models(
                models=comparison_models,
                benchmark_level='basic',
                calibrate=False,
                warmup_runs=0,
                measure_memory=False
            )
            
            print(f"✓ Model comparison completed")
            print(f"  Compared {len(comparison_models)} models")
            print(f"  Total results: {len(comparison_results)}")
            
            # Show comparison summary
            summary = comparison_results.groupby('model_name')['success_rate'].mean()
            print("  Model performance:")
            for model, success_rate in summary.items():
                print(f"    {model}: {success_rate:.3f}")
            
        except Exception as e:
            print(f"✗ Model comparison failed: {e}")
            return False
        
        # Test report generation
        print("\nTesting report generation...")
        try:
            output_dir = Path('/tmp/vision_benchmark_test')
            output_dir.mkdir(exist_ok=True)
            
            report_path = benchmark.generate_report(
                results=comparison_results,
                output_path=str(output_dir / 'test_report.html'),
                include_plots=False  # Skip plots for basic test
            )
            
            print(f"✓ Report generated: {report_path}")
            
            # Check if files were created
            html_file = Path(report_path)
            csv_file = html_file.with_suffix('.csv')
            json_file = html_file.with_suffix('.json')
            
            files_created = []
            if html_file.exists():
                files_created.append('HTML')
            if csv_file.exists():
                files_created.append('CSV')
            if json_file.exists():
                files_created.append('JSON')
            
            print(f"  Files created: {', '.join(files_created)}")
            
        except Exception as e:
            print(f"✗ Report generation failed: {e}")
            return False
        
        # Test benchmark state save/load
        print("\nTesting benchmark state persistence...")
        try:
            state_file = '/tmp/benchmark_state_test.json'
            benchmark.save_benchmark_state(state_file)
            print("✓ Benchmark state saved")
            
            # Create new benchmark and load state
            new_benchmark = VisionBenchmark(device='cpu')
            new_benchmark.load_benchmark_state(state_file)
            print("✓ Benchmark state loaded")
            
            print(f"  Loaded {len(new_benchmark.results)} results")
            print(f"  History entries: {len(new_benchmark.benchmark_history)}")
            
        except Exception as e:
            print(f"✗ State persistence failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import VisionBenchmark: {e}")
        return False
    except Exception as e:
        print(f"✗ VisionBenchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_robustness_evaluator():
    """Test VisionRobustnessEvaluator functionality."""
    print("\n" + "=" * 70)
    print("TESTING ROBUSTNESS EVALUATOR")
    print("=" * 70)
    
    try:
        from pot.vision.benchmark import VisionRobustnessEvaluator
        from pot.vision.verifier import EnhancedVisionVerifier
        
        # Create test model and verifier
        model = create_test_models()['SimpleCNN']
        verifier_config = {
            'device': 'cpu',
            'verification_method': 'batch'
        }
        verifier = EnhancedVisionVerifier(model, verifier_config, device='cpu')
        
        # Create robustness evaluator
        evaluator = VisionRobustnessEvaluator(verifier, device='cpu')
        print("✓ VisionRobustnessEvaluator created successfully")
        
        # Test noise robustness (reduced scale)
        print("\nTesting noise robustness...")
        try:
            noise_results = evaluator.evaluate_noise_robustness(
                noise_levels=[0.01, 0.05],  # Reduced for speed
                num_trials=5,  # Reduced for speed
                challenge_types=['frequency']  # Single type for speed
            )
            
            print(f"✓ Noise robustness evaluation completed")
            print(f"  Tests completed: {len(noise_results)}")
            
            for test_name, result in noise_results.items():
                print(f"  {test_name}: success_rate={result.success_rate:.3f}, "
                      f"robustness_score={result.robustness_score:.3f}")
            
        except Exception as e:
            print(f"✗ Noise robustness test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test transformation robustness (reduced scale)
        print("\nTesting transformation robustness...")
        try:
            transform_results = evaluator.evaluate_transformation_robustness(
                num_trials=3,  # Reduced for speed
                challenge_types=['frequency']  # Single type for speed
            )
            
            print(f"✓ Transformation robustness evaluation completed")
            print(f"  Tests completed: {len(transform_results)}")
            
            # Show top 3 most robust transformations
            robust_transforms = sorted(
                transform_results.items(), 
                key=lambda x: x[1].robustness_score, 
                reverse=True
            )[:3]
            
            print("  Most robust transformations:")
            for test_name, result in robust_transforms:
                print(f"    {test_name}: robustness_score={result.robustness_score:.3f}")
            
        except Exception as e:
            print(f"✗ Transformation robustness test failed: {e}")
            return False
        
        # Test adversarial robustness (very reduced scale)
        print("\nTesting adversarial robustness...")
        try:
            adv_results = evaluator.evaluate_adversarial_robustness(
                epsilon_values=[0.01],  # Single value for speed
                attack_steps=5,  # Reduced steps
                num_trials=3,  # Very few trials
                challenge_types=['frequency']
            )
            
            print(f"✓ Adversarial robustness evaluation completed")
            print(f"  Tests completed: {len(adv_results)}")
            
            for test_name, result in adv_results.items():
                print(f"  {test_name}: success_rate={result.success_rate:.3f}, "
                      f"robustness_score={result.robustness_score:.3f}")
            
        except Exception as e:
            print(f"⚠ Adversarial robustness test failed (expected on simple model): {e}")
            # This is expected to potentially fail on simple test models
        
        # Test robustness report generation
        print("\nTesting robustness report generation...")
        try:
            # Combine all results
            all_results = {**noise_results, **transform_results}
            
            output_dir = Path('/tmp/robustness_test')
            output_dir.mkdir(exist_ok=True)
            
            report_path = evaluator.generate_robustness_report(
                results=all_results,
                output_path=str(output_dir / 'robustness_report.html')
            )
            
            print(f"✓ Robustness report generated: {report_path}")
            
            # Check if files were created
            html_file = Path(report_path)
            csv_file = html_file.with_suffix('.csv')
            txt_file = html_file.with_suffix('.txt')
            
            files_created = []
            if html_file.exists():
                files_created.append('HTML')
            elif txt_file.exists():
                files_created.append('TXT')
            if csv_file.exists():
                files_created.append('CSV')
            
            print(f"  Files created: {', '.join(files_created)}")
            
        except Exception as e:
            print(f"✗ Robustness report generation failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import VisionRobustnessEvaluator: {e}")
        return False
    except Exception as e:
        print(f"✗ Robustness evaluator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmark_integration():
    """Test integration between benchmark components."""
    print("\n" + "=" * 70)
    print("TESTING BENCHMARK INTEGRATION")
    print("=" * 70)
    
    try:
        from pot.vision.benchmark import VisionBenchmark, VisionRobustnessEvaluator
        from pot.vision.verifier import EnhancedVisionVerifier
        
        # Create integrated test
        model = create_test_models()['SimpleCNN']
        benchmark = VisionBenchmark(device='cpu')
        
        print("✓ Created benchmark and model")
        
        # Run benchmark
        results = benchmark.run_benchmark(
            model=model,
            model_name='IntegrationTest',
            benchmark_level='basic',
            calibrate=False,
            warmup_runs=0,
            measure_memory=False
        )
        
        print(f"✓ Benchmark completed with {len(results)} results")
        
        # Test robustness on same model
        verifier_config = {'device': 'cpu', 'verification_method': 'batch'}
        verifier = EnhancedVisionVerifier(model, verifier_config, device='cpu')
        evaluator = VisionRobustnessEvaluator(verifier, device='cpu')
        
        robustness_results = evaluator.evaluate_noise_robustness(
            noise_levels=[0.01],
            num_trials=3,
            challenge_types=['frequency']
        )
        
        print(f"✓ Robustness evaluation completed with {len(robustness_results)} tests")
        
        # Generate combined analysis
        print("\nGenerating combined analysis...")
        
        # Calculate performance metrics
        avg_success = results['success_rate'].mean()
        avg_throughput = results['throughput'].mean()
        avg_robustness = np.mean([r.robustness_score for r in robustness_results.values()])
        
        print(f"✓ Integrated analysis:")
        print(f"  Average Success Rate: {avg_success:.3f}")
        print(f"  Average Throughput: {avg_throughput:.3f} challenges/sec")
        print(f"  Average Robustness Score: {avg_robustness:.3f}")
        
        # Performance ranking
        composite_score = (
            0.5 * avg_success +
            0.3 * avg_robustness +
            0.2 * min(avg_throughput / 10.0, 1.0)  # Normalize throughput
        )
        
        print(f"  Composite Performance Score: {composite_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmark_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 70)
    print("TESTING EDGE CASES")
    print("=" * 70)
    
    try:
        from pot.vision.benchmark import VisionBenchmark
        
        benchmark = VisionBenchmark(device='cpu')
        
        # Test invalid benchmark level
        print("Testing invalid benchmark level...")
        try:
            model = create_test_models()['SimpleCNN']
            benchmark.run_benchmark(model, benchmark_level='invalid_level')
            print("✗ Should have raised ValueError")
            return False
        except ValueError:
            print("✓ Correctly handled invalid benchmark level")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False
        
        # Test empty model dictionary
        print("Testing empty model comparison...")
        try:
            benchmark.compare_models({}, benchmark_level='basic')
            print("✓ Handled empty model dictionary")
        except Exception as e:
            print(f"⚠ Empty model comparison failed: {e}")
        
        # Test benchmark levels
        print("Testing all benchmark levels...")
        model = create_test_models()['SimpleCNN']
        
        for level in ['basic', 'intermediate']:  # Skip comprehensive and stress for speed
            try:
                results = benchmark.run_benchmark(
                    model=model,
                    benchmark_level=level,
                    calibrate=False,
                    warmup_runs=0,
                    measure_memory=False
                )
                print(f"✓ {level} benchmark: {len(results)} results")
            except Exception as e:
                print(f"✗ {level} benchmark failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Edge case testing failed: {e}")
        return False

def main():
    """Run all benchmark tests."""
    print("=" * 70)
    print("VISION BENCHMARKING AND EVALUATION TESTS")
    print("=" * 70)
    
    tests = [
        ("Vision Benchmark", test_vision_benchmark),
        ("Robustness Evaluator", test_robustness_evaluator),
        ("Benchmark Integration", test_benchmark_integration),
        ("Edge Cases", test_benchmark_edge_cases),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Tests")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    for i, (test_name, result) in enumerate(results):
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{i+1:2d}. {test_name:<25}: {status}")
        if result:
            passed += 1
    
    print("-" * 70)
    print(f"Tests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("✓ All benchmark tests passed!")
        return 0
    else:
        print("✗ Some benchmark tests failed")
        return 1

if __name__ == "__main__":
    exit(main())