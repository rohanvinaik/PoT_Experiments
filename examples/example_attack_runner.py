#!/usr/bin/env python3
"""
Demonstration of the Attack Runner and Evaluation Harness

This script shows how to use the comprehensive attack evaluation system
for systematic PoT defense testing.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from pot.eval.attacks_runner import (
    AttackRunner, AttackMetrics, AttackResultsLogger,
    BaselineMetrics, AttackImpactMetrics
)
from pot.core.attack_suites import get_benchmark_suite, create_custom_config


def create_test_model():
    """Create a simple test model."""
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
        nn.Softmax(dim=1)
    )


def create_test_dataloader(num_samples: int = 200, batch_size: int = 16):
    """Create a test dataloader."""
    # Generate synthetic data
    X = torch.randn(num_samples, 10)
    y = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class MockVerifier:
    """Mock PoT verifier for testing."""
    
    def __init__(self, detection_rate: float = 0.8):
        self.detection_rate = detection_rate
        self.call_count = 0
    
    def verify(self, output):
        """Mock verification with configurable detection rate."""
        self.call_count += 1
        
        # Simulate verification logic
        # Higher entropy outputs are more likely to be detected as attacks
        if torch.is_tensor(output):
            entropy = -torch.sum(output * torch.log(output + 1e-8), dim=-1).mean()
            detection_prob = min(0.95, self.detection_rate * (1 + entropy.item() / 10))
        else:
            detection_prob = self.detection_rate
        
        is_attack = np.random.random() < detection_prob
        confidence = np.random.uniform(0.6, 0.95) if is_attack else np.random.uniform(0.3, 0.7)
        
        return {
            'verified': not is_attack,
            'confidence': confidence,
            'is_attack': is_attack
        }
    
    def detect_attack(self, output):
        """Alternative interface for attack detection."""
        result = self.verify(output)
        return result['is_attack'], result['confidence']


def demonstrate_attack_metrics():
    """Demonstrate attack metrics computation."""
    print("=== Attack Metrics Demonstration ===")
    
    # Simulate verification data
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])  # 0=legit, 1=attack
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 1])  # predictions
    y_scores = np.array([0.2, 0.3, 0.8, 0.9, 0.7, 0.4, 0.1, 0.85, 0.25, 0.9])
    
    # Compute verification metrics
    metrics = AttackMetrics.compute_verification_metrics(y_true, y_pred, y_scores)
    
    print("Verification Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Compute FAR/FRR deltas
    baseline_far = 0.1
    baseline_frr = 0.05
    attack_far = 0.25
    attack_frr = 0.08
    
    deltas = AttackMetrics.compute_far_frr_delta(
        baseline_far, baseline_frr, attack_far, attack_frr
    )
    
    print("\\nFAR/FRR Delta Analysis:")
    for key, value in deltas.items():
        print(f"  {key}: {value:.4f}")
    
    # Simulate attack outputs
    attack_outputs = [torch.randn(1, 10) for _ in range(20)]
    verifier = MockVerifier(detection_rate=0.7)
    
    success_metrics = AttackMetrics.compute_attack_success_rate(
        attack_outputs, verifier
    )
    
    print("\\nAttack Success Metrics:")
    for key, value in success_metrics.items():
        if key != 'detection_scores':  # Skip the long list
            print(f"  {key}: {value}")
    
    # Compute robustness score
    mock_attack_results = {
        'distillation_weak': {
            'attack_type': 'distillation',
            'success_rate': 0.3,
            'detection_rate': 0.7,
            'far_degradation': 1.2,
            'frr_degradation': 1.1
        },
        'compression_strong': {
            'attack_type': 'compression',
            'success_rate': 0.6,
            'detection_rate': 0.4,
            'far_degradation': 1.8,
            'frr_degradation': 1.3
        }
    }
    
    robustness = AttackMetrics.compute_robustness_score(mock_attack_results)
    
    print("\\nRobustness Analysis:")
    for key, value in robustness.items():
        print(f"  {key}: {value}")


def demonstrate_attack_runner():
    """Demonstrate the main AttackRunner functionality."""
    print("\\n=== Attack Runner Demonstration ===")
    
    # Create test components
    model = create_test_model()
    verifier = MockVerifier(detection_rate=0.75)
    data_loader = create_test_dataloader(num_samples=100, batch_size=8)
    
    # Create output directory
    output_dir = "attack_evaluation_demo"
    
    # Initialize attack runner
    runner = AttackRunner(
        model=model,
        verifier=verifier,
        output_dir=output_dir,
        device='cpu'
    )
    
    print(f"Initialized AttackRunner with output directory: {output_dir}")
    
    # Create a small custom attack suite for testing
    test_configs = [
        create_custom_config(
            name="demo_weak_distillation",
            attack_type="distillation",
            budget={"queries": 50, "epochs": 2, "compute_time": 30.0},
            strength="weak",
            success_metrics={"fidelity": 0.6},
            parameters={"temperature": 3.0, "alpha": 0.7, "learning_rate": 0.01}
        ),
        create_custom_config(
            name="demo_light_pruning",
            attack_type="pruning",
            budget={"compression_ratio": 0.2, "compute_time": 20.0},
            strength="weak",
            success_metrics={"accuracy_drop": 0.05},
            parameters={"fine_tune_epochs": 1}
        )
    ]
    
    # Create mock attack suite
    class MockAttackSuite:
        def get_all_standard_configs(self):
            return test_configs
    
    suite = MockAttackSuite()
    
    print(f"\\nRunning attack suite with {len(test_configs)} configurations...")
    
    try:
        # Run attack suite (this will attempt to execute attacks)
        results = runner.run_attack_suite(
            suite=suite,
            data_loader=data_loader,
            compute_baseline=True
        )
        
        print("\\nAttack Suite Results:")
        print(f"  Suite: {results['suite_name']}")
        print(f"  Execution time: {results.get('execution_time', 0):.2f}s")
        
        # Print baseline metrics
        baseline = results.get('baseline_metrics')
        if baseline:
            print(f"\\nBaseline Metrics:")
            print(f"  FAR: {baseline.far:.4f}")
            print(f"  FRR: {baseline.frr:.4f}")
            print(f"  Accuracy: {baseline.accuracy:.4f}")
            print(f"  Samples: {baseline.sample_count}")
        
        # Print attack results summary
        attack_results = results.get('attack_results', {})
        print(f"\\nAttack Results ({len(attack_results)} attacks):")
        for attack_name, result in attack_results.items():
            if isinstance(result, dict):
                success = result.get('success', False)
                attack_type = result.get('attack_type', 'unknown')
                print(f"  {attack_name}: {attack_type} - {'SUCCESS' if success else 'FAILED'}")
                
                if 'error' in result:
                    print(f"    Error: {result['error']}")
        
        # Print effectiveness summary
        effectiveness = results.get('defense_effectiveness', {})
        if effectiveness:
            print(f"\\nDefense Effectiveness:")
            print(f"  Robustness Score: {effectiveness.get('robustness_score', 0):.3f}")
            print(f"  Robustness Class: {effectiveness.get('robustness_class', 'Unknown')}")
            print(f"  Assessment: {effectiveness.get('overall_assessment', 'Unknown')}")
        
        # Print summary
        summary = results.get('summary', {})
        if summary:
            print(f"\\nSummary:")
            print(f"  Total attacks: {summary.get('total_attacks', 0)}")
            print(f"  Successful attacks: {summary.get('successful_attacks', 0)}")
            print(f"  Success rate: {summary.get('overall_success_rate', 0):.1%}")
        
        print(f"\\nDetailed results saved to: {output_dir}/")
        
        return results
        
    except Exception as e:
        print(f"Error running attack suite: {e}")
        import traceback
        traceback.print_exc()
        return None


def demonstrate_results_logger():
    """Demonstrate the results logging and reporting."""
    print("\\n=== Results Logger Demonstration ===")
    
    output_dir = "logger_demo"
    logger = AttackResultsLogger(output_dir)
    
    # Create mock attack results
    mock_results = {
        'suite_name': 'demo_suite',
        'timestamp': 1234567890,
        'baseline_metrics': BaselineMetrics(
            far=0.05,
            frr=0.03,
            accuracy=0.92,
            precision=0.88,
            recall=0.85,
            f1_score=0.86,
            auc_roc=0.94,
            verification_time=0.02,
            sample_count=100,
            threshold=0.5
        ),
        'attack_results': {
            'weak_distillation': {
                'config': {'attack_type': 'distillation', 'strength': 'weak'},
                'success': True,
                'attack_type': 'distillation',
                'verification_impact': {
                    'impact_measured': True,
                    'post_attack_metrics': {
                        'far': 0.08,
                        'frr': 0.04,
                        'accuracy': 0.89,
                        'auc_roc': 0.91
                    },
                    'attack_success_metrics': {
                        'success_rate': 0.4,
                        'detection_rate': 0.6
                    }
                }
            },
            'strong_compression': {
                'config': {'attack_type': 'compression', 'strength': 'strong'},
                'success': True,
                'attack_type': 'compression',
                'verification_impact': {
                    'impact_measured': True,
                    'post_attack_metrics': {
                        'far': 0.12,
                        'frr': 0.06,
                        'accuracy': 0.85,
                        'auc_roc': 0.88
                    },
                    'attack_success_metrics': {
                        'success_rate': 0.7,
                        'detection_rate': 0.3
                    }
                }
            }
        },
        'defense_effectiveness': {
            'robustness_score': 0.75,
            'robustness_class': 'Good',
            'overall_assessment': 'Good - Resistant to most attacks',
            'attack_resistance': {
                'distillation': {'resistance_score': 0.8, 'detection_rate': 0.6, 'num_attacks': 1},
                'compression': {'resistance_score': 0.7, 'detection_rate': 0.3, 'num_attacks': 1}
            },
            'verification_degradation': {
                'avg_far_increase': 0.035,
                'avg_frr_increase': 0.015,
                'avg_accuracy_drop': 0.04
            }
        },
        'summary': {
            'total_attacks': 2,
            'successful_attacks': 2,
            'overall_success_rate': 1.0,
            'robustness_score': 0.75,
            'robustness_class': 'Good'
        }
    }
    
    # Log individual results
    for attack_name, result in mock_results['attack_results'].items():
        logger.log_attack_result(attack_name, result)
    
    # Generate comprehensive report
    logger.generate_report(mock_results)
    
    print(f"Generated comprehensive report in: {output_dir}/")
    print("Report includes:")
    print("  - Individual attack result files")
    print("  - Comprehensive HTML report")
    print("  - Summary JSON")
    print("  - Visualization plots (if matplotlib available)")


def demonstrate_baseline_computation():
    """Demonstrate baseline metrics computation."""
    print("\\n=== Baseline Metrics Demonstration ===")
    
    # Create test setup
    model = create_test_model()
    verifier = MockVerifier(detection_rate=0.8)
    data_loader = create_test_dataloader(50, 8)
    
    runner = AttackRunner(model, verifier, "baseline_demo")
    
    # Compute baseline metrics
    print("Computing baseline metrics...")
    baseline = runner._compute_baseline_metrics(data_loader)
    
    print(f"Baseline Results:")
    print(f"  False Acceptance Rate: {baseline.far:.4f}")
    print(f"  False Rejection Rate: {baseline.frr:.4f}")
    print(f"  Accuracy: {baseline.accuracy:.4f}")
    print(f"  Precision: {baseline.precision:.4f}")
    print(f"  Recall: {baseline.recall:.4f}")
    print(f"  F1 Score: {baseline.f1_score:.4f}")
    print(f"  AUC-ROC: {baseline.auc_roc:.4f}")
    print(f"  Verification Time: {baseline.verification_time:.4f}s")
    print(f"  Sample Count: {baseline.sample_count}")
    print(f"  Threshold: {baseline.threshold}")
    
    return baseline


def main():
    """Run comprehensive attack runner demonstration."""
    print("Attack Runner and Evaluation Harness Demonstration")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Demonstrate individual components
    demonstrate_attack_metrics()
    baseline = demonstrate_baseline_computation()
    results = demonstrate_attack_runner()
    demonstrate_results_logger()
    
    print("\\n" + "=" * 60)
    print("Attack Runner System Summary:")
    print("✓ AttackMetrics - Comprehensive metric computation")
    print("✓ AttackRunner - Systematic attack execution and evaluation")
    print("✓ AttackResultsLogger - Professional reporting and visualization")
    print("✓ BaselineMetrics - Thorough baseline characterization")
    print("✓ Robustness Analysis - Multi-dimensional defense assessment")
    print("✓ FAR/FRR Analysis - Detailed verification performance tracking")
    print("✓ HTML Reports - Professional attack evaluation documentation")
    print("\\nThe attack evaluation harness is ready for production PoT testing!")
    
    # Cleanup recommendation
    print("\\nNote: Demo created several output directories that can be cleaned up:")
    print("  - attack_evaluation_demo/")
    print("  - logger_demo/")
    print("  - baseline_demo/")


if __name__ == "__main__":
    main()