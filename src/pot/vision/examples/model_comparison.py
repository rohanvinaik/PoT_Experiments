#!/usr/bin/env python3
"""
Model Comparison Example

This script demonstrates how to compare multiple vision models
using the PoT benchmarking framework.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from pot.vision.benchmark import VisionBenchmark


def create_test_models():
    """Create different models for comparison."""
    
    # Lightweight model
    lightweight = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(8 * 64, 10)
    )
    
    # Standard model
    standard = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64 * 16, 10)
    )
    
    # Deep model
    deep = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(256 * 4, 10)
    )
    
    # ResNet-like model with skip connections
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
        'Lightweight': lightweight,
        'Standard': standard,
        'Deep': deep,
        'ResNetLike': resnet_like
    }


def analyze_model_properties(models):
    """Analyze basic properties of the models."""
    
    print("Model Architecture Analysis:")
    print("-" * 50)
    
    model_info = []
    
    for name, model in models.items():
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count layers
        num_layers = len(list(model.modules())) - 1  # Exclude the top-level Sequential
        
        # Model size (approximate)
        model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        
        model_info.append({
            'Model': name,
            'Parameters': f"{total_params:,}",
            'Trainable': f"{trainable_params:,}",
            'Layers': num_layers,
            'Size (MB)': f"{model_size_mb:.2f}"
        })
        
        print(f"{name}:")
        print(f"  Parameters: {total_params:,}")
        print(f"  Layers: {num_layers}")
        print(f"  Size: {model_size_mb:.2f} MB")
        print()
    
    return pd.DataFrame(model_info)


def run_benchmark_comparison(models):
    """Run benchmark comparison on all models."""
    
    print("Running Benchmark Comparison:")
    print("-" * 50)
    
    # Create benchmark suite
    benchmark = VisionBenchmark(device='cpu')
    
    # Run comparison benchmark
    print("Starting model comparison (this may take a few minutes)...")
    
    results = benchmark.compare_models(
        models=models,
        benchmark_level='intermediate',  # Balanced speed/thoroughness
        calibrate=True,  # Enable calibration for accuracy
        num_calibration_samples=30,  # Moderate calibration for demo
        warmup_runs=1,
        measure_memory=False  # Disable for CPU compatibility
    )
    
    print(f"✓ Benchmark completed with {len(results)} total tests")
    
    return results


def analyze_benchmark_results(results):
    """Analyze and display benchmark results."""
    
    print("\nBenchmark Results Analysis:")
    print("=" * 60)
    
    # Overall summary
    model_summary = results.groupby('model_name').agg({
        'success_rate': ['mean', 'std'],
        'confidence': ['mean', 'std'],
        'throughput': ['mean', 'std'],
        'total_time': 'sum',
        'verified': 'mean'
    }).round(3)
    
    # Flatten column names
    model_summary.columns = ['_'.join(col).strip() for col in model_summary.columns]
    
    print("Overall Performance Summary:")
    print(model_summary.to_string())
    
    # Performance ranking
    print(f"\nPerformance Ranking:")
    print("-" * 30)
    
    # Create composite score
    model_scores = results.groupby('model_name').agg({
        'success_rate': 'mean',
        'confidence': 'mean',
        'throughput': 'mean',
        'verified': 'mean'
    })
    
    # Normalize metrics for fair comparison
    model_scores['norm_throughput'] = model_scores['throughput'] / model_scores['throughput'].max()
    
    # Weighted composite score
    model_scores['composite_score'] = (
        0.35 * model_scores['success_rate'] +
        0.25 * model_scores['confidence'] +
        0.25 * model_scores['verified'] +
        0.15 * model_scores['norm_throughput']
    )
    
    # Sort by composite score
    ranking = model_scores.sort_values('composite_score', ascending=False)
    
    for i, (model, row) in enumerate(ranking.iterrows()):
        print(f"  {i+1}. {model}")
        print(f"     Composite Score: {row['composite_score']:.3f}")
        print(f"     Success Rate: {row['success_rate']:.2%}")
        print(f"     Confidence: {row['confidence']:.2%}")
        print(f"     Throughput: {row['throughput']:.1f} challenges/sec")
        print()
    
    return ranking


def challenge_type_analysis(results):
    """Analyze performance by challenge type."""
    
    print("Challenge Type Analysis:")
    print("-" * 40)
    
    # Performance by challenge type and model
    challenge_analysis = results.pivot_table(
        index='challenge_type',
        columns='model_name',
        values='success_rate',
        aggfunc='mean'
    ).round(3)
    
    print("Success Rate by Challenge Type:")
    print(challenge_analysis.to_string())
    
    # Find best and worst challenge types
    challenge_means = results.groupby('challenge_type')['success_rate'].mean()
    
    print(f"\nChallenge Type Summary:")
    print(f"  Easiest: {challenge_means.idxmax()} ({challenge_means.max():.2%} avg success)")
    print(f"  Hardest: {challenge_means.idxmin()} ({challenge_means.min():.2%} avg success)")
    
    return challenge_analysis


def generate_recommendations(ranking, model_info, challenge_analysis):
    """Generate recommendations based on analysis."""
    
    print("\nRecommendations:")
    print("=" * 40)
    
    best_model = ranking.index[0]
    worst_model = ranking.index[-1]
    
    print(f"🏆 Best Overall Model: {best_model}")
    print(f"   • Highest composite performance score")
    print(f"   • Recommended for production deployment")
    
    print(f"\n⚠️ Lowest Performing Model: {worst_model}")
    print(f"   • May need architecture improvements")
    print(f"   • Consider additional training or optimization")
    
    # Speed vs accuracy tradeoffs
    speed_ranking = ranking.sort_values('throughput', ascending=False)
    accuracy_ranking = ranking.sort_values('success_rate', ascending=False)
    
    fastest_model = speed_ranking.index[0]
    most_accurate = accuracy_ranking.index[0]
    
    if fastest_model != most_accurate:
        print(f"\n⚡ Speed-Accuracy Tradeoff:")
        print(f"   • Fastest: {fastest_model} ({speed_ranking.loc[fastest_model, 'throughput']:.1f} challenges/sec)")
        print(f"   • Most Accurate: {most_accurate} ({accuracy_ranking.loc[most_accurate, 'success_rate']:.2%} success)")
    
    # Challenge-specific insights
    if hasattr(challenge_analysis, 'mean'):
        challenge_difficulty = challenge_analysis.mean(axis=1).sort_values()
        hardest_challenge = challenge_difficulty.index[0]
        easiest_challenge = challenge_difficulty.index[-1]
        
        print(f"\n🎯 Challenge Insights:")
        print(f"   • Focus training on: {hardest_challenge} challenges")
        print(f"   • Strong performance on: {easiest_challenge} challenges")
    
    # Resource considerations
    print(f"\n💾 Resource Considerations:")
    model_sizes = {name: sum(p.numel() for p in models[name].parameters()) 
                   for name in ranking.index}
    
    # Find models with good performance/size ratio
    efficiency_scores = []
    for model in ranking.index:
        score = ranking.loc[model, 'composite_score']
        size = model_sizes[model]
        efficiency = score / (size / 1e6)  # Score per million parameters
        efficiency_scores.append((model, efficiency))
    
    efficiency_scores.sort(key=lambda x: x[1], reverse=True)
    most_efficient = efficiency_scores[0][0]
    
    print(f"   • Most efficient (score/param): {most_efficient}")
    print(f"   • Consider for resource-constrained deployments")


def main():
    """Run model comparison example."""
    
    print("Vision Model Comparison Example")
    print("=" * 60)
    
    # Step 1: Create models
    print("\n1. Creating test models...")
    models = create_test_models()
    print(f"   Created {len(models)} models for comparison")
    
    # Step 2: Analyze model properties
    print(f"\n2. Analyzing model architectures...")
    model_info = analyze_model_properties(models)
    
    # Step 3: Run benchmark comparison
    print(f"\n3. Running benchmark comparison...")
    results = run_benchmark_comparison(models)
    
    # Step 4: Analyze results
    print(f"\n4. Analyzing results...")
    ranking = analyze_benchmark_results(results)
    
    # Step 5: Challenge type analysis
    print(f"\n5. Challenge type analysis...")
    challenge_analysis = challenge_type_analysis(results)
    
    # Step 6: Generate recommendations
    print(f"\n6. Generating recommendations...")
    generate_recommendations(ranking, model_info, challenge_analysis)
    
    # Step 7: Save results
    output_dir = Path('/tmp/model_comparison_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    results.to_csv(output_dir / 'detailed_results.csv', index=False)
    ranking.to_csv(output_dir / 'model_ranking.csv')
    model_info.to_csv(output_dir / 'model_info.csv', index=False)
    
    print(f"\n7. Results saved:")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   📄 detailed_results.csv - Full benchmark data")
    print(f"   📄 model_ranking.csv - Performance ranking")
    print(f"   📄 model_info.csv - Model architecture info")
    
    print(f"\n{'=' * 60}")
    print("Model comparison completed successfully!")
    
    # Return best model name for potential use
    best_model = ranking.index[0]
    print(f"🏆 Winner: {best_model}")
    
    return 0


if __name__ == "__main__":
    exit(main())