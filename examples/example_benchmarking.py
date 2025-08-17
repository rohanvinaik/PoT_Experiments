#!/usr/bin/env python3
"""
Demonstration of attack benchmarking and metrics dashboard.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add path
sys.path.insert(0, str(Path(__file__).parent))

from pot.eval.attack_benchmarks import (
    AttackBenchmark,
    AttackMetricsDashboard,
    run_standard_benchmark,
    create_comparison_dashboard
)
from pot.core.defenses import MockBaseVerifier


class SimpleConvNet(nn.Module):
    """Simple CNN for benchmarking."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


class SimpleMLP(nn.Module):
    """Simple MLP for benchmarking."""
    def __init__(self, input_dim=3072, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


def create_synthetic_data(num_samples=100, image_size=32):
    """Create synthetic dataset for benchmarking."""
    # Create random image data
    images = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(images, labels)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    return data_loader


def run_single_model_benchmark():
    """Run benchmark on a single model."""
    print("="*60)
    print("SINGLE MODEL BENCHMARK")
    print("="*60)
    
    # Create model and data
    model = SimpleConvNet()
    verifier = MockBaseVerifier()
    data_loader = create_synthetic_data(num_samples=50)
    
    # Initialize benchmark
    benchmark = AttackBenchmark(
        device='cpu',
        verbose=True,
        save_results=True,
        results_dir="benchmark_results"
    )
    
    # Run subset of attacks for demonstration
    quick_attacks = [
        "distillation_weak",
        "distillation_strong",
        "pruning_30",
        "pruning_50",
        "quantization_8bit",
        "quantization_4bit",
        "wrapper_naive"
    ]
    
    print("\nRunning benchmark on SimpleConvNet...")
    results = benchmark.run_benchmark(
        model=model,
        verifier=verifier,
        data_loader=data_loader,
        attack_names=quick_attacks,
        include_defenses=True
    )
    
    # Compute robustness score
    robustness = benchmark.compute_robustness_score(results)
    print(f"\nRobustness Score: {robustness:.1f}/100")
    
    # Generate report
    report = benchmark.generate_report(
        results,
        save_path="benchmark_results/convnet_report.json"
    )
    
    # Print summary statistics
    print("\nAttack Success Rates by Type:")
    for attack_type, metrics in report['by_attack_type'].items():
        print(f"  {attack_type}: {metrics['success_rate']:.1%} ({metrics['count']} attacks)")
    
    print("\nPerformance Impact:")
    print(f"  Average FAR increase: {report['performance_impact']['avg_far_increase']:.3f}")
    print(f"  Average FRR increase: {report['performance_impact']['avg_frr_increase']:.3f}")
    print(f"  Average accuracy drop: {report['performance_impact']['avg_accuracy_drop']:.3f}")
    
    print("\nDefense Effectiveness:")
    print(f"  Detection rate: {report['defense_effectiveness']['detection_rate']:.1%}")
    print(f"  Average confidence: {report['defense_effectiveness']['avg_confidence']:.3f}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    return results


def run_model_comparison():
    """Compare multiple models using benchmarking."""
    print("\n" + "="*60)
    print("MODEL COMPARISON BENCHMARK")
    print("="*60)
    
    # Create models
    models = {
        'SimpleConvNet': SimpleConvNet(),
        'SimpleMLP': SimpleMLP()
    }
    
    verifier = MockBaseVerifier()
    data_loader = create_synthetic_data(num_samples=50)
    
    # Initialize benchmark
    benchmark = AttackBenchmark(
        device='cpu',
        verbose=False,  # Less verbose for multiple models
        save_results=True,
        results_dir="benchmark_results"
    )
    
    # Quick attack subset for demonstration
    quick_attacks = [
        "distillation_weak",
        "pruning_50",
        "quantization_4bit",
        "wrapper_naive"
    ]
    
    # Run benchmarks for each model
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\nBenchmarking {model_name}...")
        results = benchmark.run_benchmark(
            model=model,
            verifier=verifier,
            data_loader=data_loader,
            attack_names=quick_attacks,
            include_defenses=True
        )
        
        all_results[model_name] = results
        robustness = benchmark.compute_robustness_score(results)
        print(f"  Robustness Score: {robustness:.1f}/100")
        print(f"  Success Rate: {results['success'].mean():.1%}")
    
    # Generate leaderboard
    print("\nGenerating leaderboard...")
    leaderboard = benchmark.generate_leaderboard(
        all_results,
        save_path="benchmark_results/leaderboard.csv"
    )
    
    print("\n" + "="*60)
    print("MODEL LEADERBOARD")
    print("="*60)
    print(leaderboard[['rank', 'model', 'robustness_score', 'attack_success_rate', 
                       'defense_detection_rate']].to_string(index=False))
    
    return all_results, leaderboard


def create_metrics_dashboard():
    """Create interactive metrics dashboard."""
    print("\n" + "="*60)
    print("CREATING METRICS DASHBOARD")
    print("="*60)
    
    # Check if results exist
    results_dir = Path("benchmark_results")
    if not results_dir.exists() or not list(results_dir.glob("*.csv")):
        print("No benchmark results found. Running quick benchmark first...")
        run_single_model_benchmark()
    
    # Create dashboard
    try:
        dashboard = AttackMetricsDashboard("benchmark_results")
        
        # Generate summary statistics
        print("\nSummary Statistics:")
        summary = dashboard.generate_summary_statistics()
        print(summary.to_string())
        
        # Create visualizations if plotly is available
        try:
            # Create main dashboard
            dashboard.create_dashboard("benchmark_results/dashboard.html")
            print("\n✓ Interactive dashboard created: benchmark_results/dashboard.html")
            
            # Create individual plots
            fig = dashboard.plot_attack_success_rates("benchmark_results/success_rates.html")
            if fig:
                print("✓ Success rates plot created: benchmark_results/success_rates.html")
            
            fig = dashboard.plot_far_frr_tradeoffs("benchmark_results/far_frr.html")
            if fig:
                print("✓ FAR/FRR plot created: benchmark_results/far_frr.html")
            
            fig = dashboard.plot_defense_adaptation("benchmark_results/defense_adaptation.html")
            if fig:
                print("✓ Defense adaptation plot created: benchmark_results/defense_adaptation.html")
                
        except ImportError:
            print("\n⚠ Plotly not installed. Install with: pip install plotly")
            print("  Skipping interactive visualizations.")
            
    except Exception as e:
        print(f"Error creating dashboard: {e}")


def run_quick_evaluation():
    """Run a quick evaluation using the standard benchmark function."""
    print("\n" + "="*60)
    print("QUICK STANDARD EVALUATION")
    print("="*60)
    
    # Create model and data
    model = SimpleConvNet()
    verifier = MockBaseVerifier()
    data_loader = create_synthetic_data(num_samples=30)
    
    # Run standard benchmark
    print("\nRunning standard benchmark suite...")
    results, report = run_standard_benchmark(
        model=model,
        verifier=verifier,
        data_loader=data_loader,
        device='cpu',
        save_results=True
    )
    
    # Display results table
    print("\nResults Summary:")
    print(results[['attack_name', 'success', 'confidence', 'defense_detected', 
                  'execution_time']].head(10).to_string(index=False))
    
    return results, report


def main():
    """Run all benchmark demonstrations."""
    print("="*60)
    print("ATTACK BENCHMARKING DEMONSTRATION")
    print("="*60)
    print("\nThis demo will:")
    print("1. Run single model benchmark")
    print("2. Compare multiple models")
    print("3. Create metrics dashboard")
    print("4. Run quick standard evaluation")
    
    # Create results directory
    Path("benchmark_results").mkdir(exist_ok=True)
    
    # 1. Single model benchmark
    print("\n" + "-"*60)
    input("Press Enter to run single model benchmark...")
    single_results = run_single_model_benchmark()
    
    # 2. Model comparison
    print("\n" + "-"*60)
    input("Press Enter to run model comparison...")
    comparison_results, leaderboard = run_model_comparison()
    
    # 3. Metrics dashboard
    print("\n" + "-"*60)
    input("Press Enter to create metrics dashboard...")
    create_metrics_dashboard()
    
    # 4. Quick evaluation
    print("\n" + "-"*60)
    input("Press Enter to run quick evaluation...")
    quick_results, quick_report = run_quick_evaluation()
    
    # Final summary
    print("\n" + "="*60)
    print("BENCHMARKING COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  • benchmark_results/*.csv - Raw benchmark data")
    print("  • benchmark_results/*.json - Detailed reports")
    print("  • benchmark_results/leaderboard.csv - Model comparison")
    print("  • benchmark_results/dashboard.html - Interactive dashboard")
    print("  • benchmark_results/*_rates.html - Individual visualizations")
    print("\nUse AttackBenchmark class for custom benchmarking")
    print("Use AttackMetricsDashboard class for custom visualizations")


if __name__ == "__main__":
    main()