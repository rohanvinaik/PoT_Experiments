#!/usr/bin/env python3
"""
Test script for ReproducibleExperimentRunner

This script tests the reproducible experiment runner with minimal configurations
to ensure all components work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from pot.experiments.reproducible_runner import ReproducibleExperimentRunner, ExperimentConfig

def test_basic_functionality():
    """Test basic functionality of the experiment runner."""
    print("Testing ReproducibleExperimentRunner...")
    
    # Create minimal config
    config = ExperimentConfig(
        experiment_name="test_basic",
        model_type="vision",
        model_architecture="basic_cnn", 
        challenge_families=["vision:freq"],
        n_challenges_per_family=3,
        max_challenges=5,
        verbose=True,
        output_dir="outputs/test_experiments"
    )
    
    # Initialize runner
    runner = ReproducibleExperimentRunner(config)
    print(f"✓ Created runner with trial_id: {config.trial_id}")
    
    # Test model setup
    try:
        models = runner.setup_models()
        print(f"✓ Set up {len(models)} models")
    except Exception as e:
        print(f"✗ Model setup failed: {e}")
        return False
    
    # Test challenge family execution
    try:
        results = runner.run_challenge_family("vision:freq")
        print(f"✓ Executed challenge family with {len(results)} results")
    except Exception as e:
        print(f"✗ Challenge execution failed: {e}")
        return False
    
    # Test metrics computation
    try:
        metrics = runner.compute_metrics()
        print(f"✓ Computed metrics: {list(metrics.keys())}")
    except Exception as e:
        print(f"✗ Metrics computation failed: {e}")
        return False
    
    # Test result saving
    try:
        csv_path, json_path = runner.save_results()
        print(f"✓ Saved results to {csv_path} and {json_path}")
    except Exception as e:
        print(f"✗ Result saving failed: {e}")
        return False
    
    print("✓ All basic tests passed!")
    return True

def test_full_experiment():
    """Test the full experiment pipeline."""
    print("\nTesting full experiment pipeline...")
    
    config = ExperimentConfig(
        experiment_name="test_full_pipeline",
        model_type="vision",
        challenge_families=["vision:freq"],
        n_challenges_per_family=2,
        max_challenges=3,
        verbose=False,  # Reduce output for testing
        output_dir="outputs/test_experiments"
    )
    
    runner = ReproducibleExperimentRunner(config)
    
    try:
        summary = runner.run_full_experiment()
        print(f"✓ Full experiment completed: {summary['total_results']} results")
        print(f"  - FAR: {summary['metrics'].get('far', 0):.4f}")
        print(f"  - FRR: {summary['metrics'].get('frr', 0):.4f}")
        print(f"  - Accuracy: {summary['metrics'].get('accuracy', 0):.4f}")
        return True
    except Exception as e:
        print(f"✗ Full experiment failed: {e}")
        return False

def test_reproducibility():
    """Test that experiments are reproducible with same seeds."""
    print("\nTesting reproducibility...")
    
    # Run same experiment twice with same seed
    configs = [
        ExperimentConfig(
            experiment_name=f"test_repro_{i}",
            global_seed=123,
            torch_seed=123,
            numpy_seed=123,
            model_type="vision",
            challenge_families=["vision:freq"],
            n_challenges_per_family=2,
            verbose=False,
            output_dir="outputs/test_experiments"
        ) for i in range(2)
    ]
    
    results = []
    for i, config in enumerate(configs):
        runner = ReproducibleExperimentRunner(config)
        try:
            summary = runner.run_full_experiment()
            results.append(summary)
            print(f"✓ Run {i+1} completed")
        except Exception as e:
            print(f"✗ Run {i+1} failed: {e}")
            return False
    
    # Compare results (should be identical due to seeding)
    if len(results) == 2:
        metrics1 = results[0]['metrics']
        metrics2 = results[1]['metrics']
        
        # Check if key metrics are close (allowing for small numerical differences)
        tolerance = 1e-6
        for key in ['avg_distance', 'avg_confidence']:
            if key in metrics1 and key in metrics2:
                diff = abs(metrics1[key] - metrics2[key])
                if diff > tolerance:
                    print(f"✗ Reproducibility test failed: {key} differs by {diff}")
                    return False
        
        print("✓ Reproducibility test passed!")
        return True
    
    return False

if __name__ == "__main__":
    print("Running ReproducibleExperimentRunner tests...\n")
    
    success = True
    
    # Run tests
    success &= test_basic_functionality()
    success &= test_full_experiment() 
    success &= test_reproducibility()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)