#!/usr/bin/env python3
"""
Example: Reproducible Experiment Runner for PoT Framework

This script demonstrates how to use the ReproducibleExperimentRunner for 
conducting comprehensive, deterministic verification experiments.

Usage:
    python example_reproducible_experiments.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from pot.experiments.reproducible_runner import ReproducibleExperimentRunner, ExperimentConfig

def main():
    print("🧪 PoT Reproducible Experiment Runner Demo")
    print("=" * 50)
    
    # Example 1: Basic Vision Experiment
    print("\n📸 Example 1: Basic Vision Model Verification")
    print("-" * 40)
    
    vision_config = ExperimentConfig(
        experiment_name="vision_basic_verification",
        model_type="vision",
        model_architecture="basic_cnn",
        challenge_families=["vision:freq", "vision:texture"],
        n_challenges_per_family=5,
        alpha=0.05,  # 5% Type I error
        beta=0.05,   # 5% Type II error
        tau_id=0.01, # Identity threshold
        global_seed=42,
        verbose=True,
        output_dir="outputs/demo_experiments"
    )
    
    runner = ReproducibleExperimentRunner(vision_config)
    summary = runner.run_full_experiment()
    
    print(f"✅ Vision experiment completed!")
    print(f"   - Trial ID: {summary['trial_id']}")
    print(f"   - Total Time: {summary['total_time']:.2f}s")
    print(f"   - Results: {summary['total_results']} challenges")
    print(f"   - FAR: {summary['metrics']['far']:.4f}")
    print(f"   - FRR: {summary['metrics']['frr']:.4f}")
    print(f"   - Accuracy: {summary['metrics']['accuracy']:.4f}")
    print(f"   - Output: {summary['output_paths']['output_dir']}")
    
    # Example 2: Language Model Experiment
    print("\n📝 Example 2: Language Model Verification")
    print("-" * 40)
    
    lm_config = ExperimentConfig(
        experiment_name="lm_basic_verification",
        model_type="language",
        model_architecture="mock",
        challenge_families=["lm:templates"],
        n_challenges_per_family=3,
        alpha=0.01,  # Stricter Type I error
        beta=0.01,   # Stricter Type II error
        tau_id=0.005, # Tighter identity threshold
        global_seed=123,
        verbose=True,
        output_dir="outputs/demo_experiments"
    )
    
    runner_lm = ReproducibleExperimentRunner(lm_config)
    summary_lm = runner_lm.run_full_experiment()
    
    print(f"✅ Language model experiment completed!")
    print(f"   - Trial ID: {summary_lm['trial_id']}")
    print(f"   - Total Time: {summary_lm['total_time']:.2f}s") 
    print(f"   - Results: {summary_lm['total_results']} challenges")
    print(f"   - FAR: {summary_lm['metrics']['far']:.4f}")
    print(f"   - FRR: {summary_lm['metrics']['frr']:.4f}")
    print(f"   - Accuracy: {summary_lm['metrics']['accuracy']:.4f}")
    print(f"   - Output: {summary_lm['output_paths']['output_dir']}")
    
    # Example 3: High-Security Verification
    print("\n🔒 Example 3: High-Security Verification with Jacobian")
    print("-" * 50)
    
    secure_config = ExperimentConfig(
        experiment_name="high_security_verification",
        model_type="vision",
        model_architecture="resnet18",
        challenge_families=["vision:freq"],
        n_challenges_per_family=3,
        alpha=0.001,  # Very strict error rates
        beta=0.001,
        tau_id=0.001,  # Very tight threshold
        use_jacobian=True,  # Enable Jacobian fingerprinting
        global_seed=999,
        verbose=True,
        save_models=True,  # Save models for audit
        output_dir="outputs/demo_experiments"
    )
    
    runner_secure = ReproducibleExperimentRunner(secure_config)
    summary_secure = runner_secure.run_full_experiment()
    
    print(f"✅ High-security experiment completed!")
    print(f"   - Trial ID: {summary_secure['trial_id']}")
    print(f"   - Total Time: {summary_secure['total_time']:.2f}s")
    print(f"   - Results: {summary_secure['total_results']} challenges")
    print(f"   - FAR: {summary_secure['metrics']['far']:.4f}")
    print(f"   - FRR: {summary_secure['metrics']['frr']:.4f}")
    print(f"   - Accuracy: {summary_secure['metrics']['accuracy']:.4f}")
    print(f"   - Avg Fingerprint Time: {summary_secure['metrics']['avg_fingerprint_time']:.4f}s")
    print(f"   - Output: {summary_secure['output_paths']['output_dir']}")
    
    # Example 4: Reproducibility Test
    print("\n🔄 Example 4: Reproducibility Verification")
    print("-" * 40)
    
    # Run same experiment twice with identical seeds
    repro_config_1 = ExperimentConfig(
        experiment_name="reproducibility_test_1",
        model_type="vision",
        challenge_families=["vision:freq"],
        n_challenges_per_family=2,
        global_seed=555,
        torch_seed=555, 
        numpy_seed=555,
        verbose=False,
        output_dir="outputs/demo_experiments"
    )
    
    repro_config_2 = ExperimentConfig(
        experiment_name="reproducibility_test_2", 
        model_type="vision",
        challenge_families=["vision:freq"],
        n_challenges_per_family=2,
        global_seed=555,  # Same seeds
        torch_seed=555,
        numpy_seed=555,
        verbose=False,
        output_dir="outputs/demo_experiments"
    )
    
    runner_1 = ReproducibleExperimentRunner(repro_config_1)
    summary_1 = runner_1.run_full_experiment()
    
    runner_2 = ReproducibleExperimentRunner(repro_config_2)
    summary_2 = runner_2.run_full_experiment()
    
    # Compare results
    distance_diff = abs(summary_1['metrics']['avg_distance'] - summary_2['metrics']['avg_distance'])
    confidence_diff = abs(summary_1['metrics']['avg_confidence'] - summary_2['metrics']['avg_confidence'])
    
    print(f"✅ Reproducibility test completed!")
    print(f"   - Run 1 avg distance: {summary_1['metrics']['avg_distance']:.6f}")
    print(f"   - Run 2 avg distance: {summary_2['metrics']['avg_distance']:.6f}")
    print(f"   - Distance difference: {distance_diff:.6f}")
    print(f"   - Confidence difference: {confidence_diff:.6f}")
    
    if distance_diff < 1e-6 and confidence_diff < 1e-6:
        print("   ✅ Results are identical - full reproducibility confirmed!")
    else:
        print("   ⚠️  Results differ slightly - check random seed management")
    
    print("\n🎉 Demo completed successfully!")
    print(f"📁 All results saved to: outputs/demo_experiments/")
    print("\nKey Features Demonstrated:")
    print("- ✅ Deterministic model setup and training")
    print("- ✅ Configurable challenge families and parameters")
    print("- ✅ Sequential decision making with statistical bounds")
    print("- ✅ Comprehensive metrics (FAR, FRR, confidence)")
    print("- ✅ Full result transparency with CSV/JSON export")
    print("- ✅ Progress tracking and structured logging")
    print("- ✅ Complete reproducibility with seed management")

if __name__ == "__main__":
    main()