#!/usr/bin/env python3
"""
Robustness Analysis Example

This script demonstrates comprehensive robustness evaluation
of vision models against various perturbations and attacks.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from pot.vision.verifier import EnhancedVisionVerifier
from pot.vision.benchmark import VisionRobustnessEvaluator


def create_test_model():
    """Create a test model for robustness evaluation."""
    return nn.Sequential(
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


def setup_verifier_and_evaluator():
    """Set up the verifier and robustness evaluator."""
    
    print("Setting up verifier and evaluator...")
    
    # Create model
    model = create_test_model()
    model.eval()
    
    # Create verifier
    config = {
        'device': 'cpu',
        'verification_method': 'batch',
        'temperature': 1.0
    }
    verifier = EnhancedVisionVerifier(model, config)
    
    # Create robustness evaluator
    evaluator = VisionRobustnessEvaluator(verifier, device='cpu')
    
    print(f"✓ Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✓ Verifier created with device: {verifier.device}")
    print(f"✓ Robustness evaluator ready")
    
    return verifier, evaluator


def evaluate_noise_robustness(evaluator):
    """Evaluate robustness against additive noise."""
    
    print("\n" + "=" * 50)
    print("NOISE ROBUSTNESS EVALUATION")
    print("=" * 50)
    
    print("Testing robustness to additive Gaussian noise...")
    
    # Test multiple noise levels
    noise_levels = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    
    noise_results = evaluator.evaluate_noise_robustness(
        noise_levels=noise_levels,
        num_trials=15,  # Good balance of accuracy and speed
        challenge_types=['frequency', 'texture']
    )
    
    print(f"✓ Completed {len(noise_results)} noise robustness tests")
    
    # Analyze results
    print(f"\nNoise Robustness Results:")
    print("-" * 40)
    
    noise_data = []
    for test_name, result in noise_results.items():
        parts = test_name.split('_')
        challenge_type = parts[0]
        noise_level = float(parts[-1])
        
        noise_data.append({
            'Challenge Type': challenge_type,
            'Noise Level': noise_level,
            'Success Rate': result.success_rate,
            'Std Dev': result.std_dev,
            'Robustness Score': result.robustness_score,
            'Baseline': result.baseline_success
        })
    
    noise_df = pd.DataFrame(noise_data)
    
    # Display summary by challenge type
    for challenge_type in noise_df['Challenge Type'].unique():
        type_data = noise_df[noise_df['Challenge Type'] == challenge_type]
        print(f"\n{challenge_type.upper()} Challenges:")
        
        for _, row in type_data.iterrows():
            print(f"  Noise {row['Noise Level']:5.3f}: "
                  f"{row['Success Rate']:5.2%} success "
                  f"({row['Robustness Score']:5.3f} robustness)")
    
    # Find degradation threshold
    print(f"\nRobustness Analysis:")
    
    for challenge_type in noise_df['Challenge Type'].unique():
        type_data = noise_df[noise_df['Challenge Type'] == challenge_type].sort_values('Noise Level')
        
        # Find first point where robustness drops below 50%
        threshold_idx = (type_data['Robustness Score'] < 0.5).idxmax()
        if threshold_idx and type_data.loc[threshold_idx, 'Robustness Score'] < 0.5:
            threshold = type_data.loc[threshold_idx, 'Noise Level']
            print(f"  {challenge_type}: Degrades significantly at noise level {threshold:.3f}")
        else:
            print(f"  {challenge_type}: Maintains >50% robustness across all tested levels")
    
    return noise_results, noise_df


def evaluate_transformation_robustness(evaluator):
    """Evaluate robustness against image transformations."""
    
    print("\n" + "=" * 50)
    print("TRANSFORMATION ROBUSTNESS EVALUATION")
    print("=" * 50)
    
    print("Testing robustness to image transformations...")
    
    transform_results = evaluator.evaluate_transformation_robustness(
        num_trials=15,
        challenge_types=['frequency', 'texture']
    )
    
    print(f"✓ Completed {len(transform_results)} transformation tests")
    
    # Organize results by transformation type
    print(f"\nTransformation Robustness Results:")
    print("-" * 45)
    
    transform_data = []
    for test_name, result in transform_results.items():
        parts = test_name.split('_', 1)
        challenge_type = parts[0]
        transform_name = parts[1] if len(parts) > 1 else 'unknown'
        
        transform_data.append({
            'Challenge Type': challenge_type,
            'Transformation': transform_name,
            'Success Rate': result.success_rate,
            'Robustness Score': result.robustness_score,
            'Baseline': result.baseline_success
        })
    
    transform_df = pd.DataFrame(transform_data)
    
    # Group by transformation type
    transform_summary = transform_df.groupby('Transformation').agg({
        'Success Rate': 'mean',
        'Robustness Score': 'mean'
    }).sort_values('Robustness Score', ascending=False)
    
    print(f"Average Robustness by Transformation:")
    for transform, row in transform_summary.iterrows():
        print(f"  {transform:20s}: {row['Robustness Score']:5.3f} robustness "
              f"({row['Success Rate']:5.2%} success)")
    
    # Identify vulnerabilities
    print(f"\nVulnerability Analysis:")
    worst_transforms = transform_summary.head(3)  # Most vulnerable
    best_transforms = transform_summary.tail(3)   # Most robust
    
    print(f"  Most vulnerable to:")
    for transform, row in worst_transforms.iterrows():
        print(f"    - {transform} (robustness: {row['Robustness Score']:.3f})")
    
    print(f"  Most robust against:")
    for transform, row in best_transforms.iterrows():
        print(f"    - {transform} (robustness: {row['Robustness Score']:.3f})")
    
    return transform_results, transform_df


def evaluate_adversarial_robustness(evaluator):
    """Evaluate robustness against adversarial attacks."""
    
    print("\n" + "=" * 50)
    print("ADVERSARIAL ROBUSTNESS EVALUATION")
    print("=" * 50)
    
    print("Testing robustness to adversarial perturbations...")
    
    try:
        adv_results = evaluator.evaluate_adversarial_robustness(
            epsilon_values=[0.01, 0.03, 0.05, 0.1],
            attack_steps=10,
            num_trials=10,
            challenge_types=['frequency']  # Focus on one type for speed
        )
        
        print(f"✓ Completed {len(adv_results)} adversarial tests")
        
        # Analyze adversarial results
        print(f"\nAdversarial Robustness Results:")
        print("-" * 40)
        
        adv_data = []
        for test_name, result in adv_results.items():
            epsilon = float(test_name.split('_')[-1])
            
            adv_data.append({
                'Epsilon': epsilon,
                'Success Rate': result.success_rate,
                'Robustness Score': result.robustness_score,
                'Baseline': result.baseline_success
            })
            
            print(f"  ε = {epsilon:5.3f}: {result.success_rate:5.2%} success "
                  f"({result.robustness_score:5.3f} robustness)")
        
        adv_df = pd.DataFrame(adv_data)
        
        # Find attack threshold
        if len(adv_df) > 0:
            critical_epsilon = None
            for _, row in adv_df.iterrows():
                if row['Robustness Score'] < 0.1:  # Less than 10% robustness
                    critical_epsilon = row['Epsilon']
                    break
            
            if critical_epsilon:
                print(f"\n  ⚠️ Critical vulnerability at ε = {critical_epsilon:.3f}")
            else:
                print(f"\n  ✓ Maintains reasonable robustness across tested range")
        
        return adv_results, adv_df
        
    except Exception as e:
        print(f"⚠️ Adversarial evaluation failed: {e}")
        print("This is common with simple test models")
        return {}, pd.DataFrame()


def create_robustness_visualization(noise_df, transform_df, adv_df, output_dir):
    """Create visualizations of robustness results."""
    
    print("\nCreating robustness visualizations...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Vision Model Robustness Analysis', fontsize=16, fontweight='bold')
        
        # 1. Noise robustness plot
        if not noise_df.empty:
            for challenge_type in noise_df['Challenge Type'].unique():
                type_data = noise_df[noise_df['Challenge Type'] == challenge_type]
                axes[0, 0].plot(type_data['Noise Level'], type_data['Robustness Score'], 
                               'o-', label=challenge_type, linewidth=2, markersize=6)
            
            axes[0, 0].set_xlabel('Noise Level')
            axes[0, 0].set_ylabel('Robustness Score')
            axes[0, 0].set_title('Noise Robustness')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim([0, 1])
        
        # 2. Transformation robustness
        if not transform_df.empty:
            transform_summary = transform_df.groupby('Transformation')['Robustness Score'].mean().sort_values()
            
            # Show top 10 transformations
            top_transforms = transform_summary.tail(10)
            
            bars = axes[0, 1].barh(range(len(top_transforms)), top_transforms.values, 
                                  color='steelblue', alpha=0.7)
            axes[0, 1].set_yticks(range(len(top_transforms)))
            axes[0, 1].set_yticklabels([t.replace('_', ' ').title() for t in top_transforms.index])
            axes[0, 1].set_xlabel('Robustness Score')
            axes[0, 1].set_title('Transformation Robustness (Top 10)')
            axes[0, 1].set_xlim([0, 1])
            
            # Add value labels
            for i, (bar, score) in enumerate(zip(bars, top_transforms.values)):
                axes[0, 1].text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                               f'{score:.2f}', va='center', fontsize=8)
        
        # 3. Adversarial robustness (if available)
        if not adv_df.empty:
            axes[1, 0].plot(adv_df['Epsilon'], adv_df['Robustness Score'], 
                           'ro-', linewidth=2, markersize=8)
            axes[1, 0].set_xlabel('Epsilon (Perturbation Magnitude)')
            axes[1, 0].set_ylabel('Robustness Score')
            axes[1, 0].set_title('Adversarial Robustness')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0, 1])
        else:
            axes[1, 0].text(0.5, 0.5, 'Adversarial evaluation\nnot available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontsize=12, style='italic')
            axes[1, 0].set_title('Adversarial Robustness')
        
        # 4. Overall robustness summary
        all_scores = []
        all_labels = []
        
        if not noise_df.empty:
            all_scores.extend(noise_df['Robustness Score'].tolist())
            all_labels.extend(['Noise'] * len(noise_df))
        
        if not transform_df.empty:
            all_scores.extend(transform_df['Robustness Score'].tolist())
            all_labels.extend(['Transform'] * len(transform_df))
        
        if not adv_df.empty:
            all_scores.extend(adv_df['Robustness Score'].tolist())
            all_labels.extend(['Adversarial'] * len(adv_df))
        
        if all_scores:
            # Box plot of all robustness scores
            box_data = []
            box_labels = []
            for label in ['Noise', 'Transform', 'Adversarial']:
                if label in all_labels:
                    scores = [score for score, lbl in zip(all_scores, all_labels) if lbl == label]
                    if scores:
                        box_data.append(scores)
                        box_labels.append(label)
            
            if box_data:
                axes[1, 1].boxplot(box_data, labels=box_labels)
                axes[1, 1].set_ylabel('Robustness Score')
                axes[1, 1].set_title('Robustness Distribution by Test Type')
                axes[1, 1].set_ylim([0, 1])
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'robustness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to {output_dir / 'robustness_analysis.png'}")
        
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")


def generate_robustness_report(noise_results, transform_results, adv_results, 
                             noise_df, transform_df, adv_df, output_dir):
    """Generate comprehensive robustness report."""
    
    print("\nGenerating comprehensive robustness report...")
    
    # Calculate overall statistics
    all_robustness_scores = []
    if not noise_df.empty:
        all_robustness_scores.extend(noise_df['Robustness Score'].tolist())
    if not transform_df.empty:
        all_robustness_scores.extend(transform_df['Robustness Score'].tolist())
    if not adv_df.empty:
        all_robustness_scores.extend(adv_df['Robustness Score'].tolist())
    
    # Generate summary statistics
    summary = {
        'total_tests': len(all_robustness_scores),
        'average_robustness': float(np.mean(all_robustness_scores)) if all_robustness_scores else 0,
        'min_robustness': float(np.min(all_robustness_scores)) if all_robustness_scores else 0,
        'max_robustness': float(np.max(all_robustness_scores)) if all_robustness_scores else 0,
        'std_robustness': float(np.std(all_robustness_scores)) if all_robustness_scores else 0,
        'tests_completed': {
            'noise': len(noise_results),
            'transformation': len(transform_results),
            'adversarial': len(adv_results)
        }
    }
    
    # Identify vulnerabilities
    vulnerabilities = []
    if not transform_df.empty:
        worst_transforms = transform_df.nsmallest(3, 'Robustness Score')
        for _, row in worst_transforms.iterrows():
            vulnerabilities.append({
                'type': 'transformation',
                'name': row['Transformation'],
                'robustness_score': row['Robustness Score']
            })
    
    if not noise_df.empty:
        critical_noise = noise_df[noise_df['Robustness Score'] < 0.3]
        if not critical_noise.empty:
            vulnerabilities.append({
                'type': 'noise',
                'description': f"Vulnerable to noise levels above {critical_noise['Noise Level'].min():.3f}",
                'robustness_score': critical_noise['Robustness Score'].min()
            })
    
    summary['vulnerabilities'] = vulnerabilities
    
    # Save detailed report
    report = {
        'summary': summary,
        'noise_results': {k: {
            'success_rate': v.success_rate,
            'robustness_score': v.robustness_score,
            'std_dev': v.std_dev,
            'baseline_success': v.baseline_success
        } for k, v in noise_results.items()},
        'transform_results': {k: {
            'success_rate': v.success_rate,
            'robustness_score': v.robustness_score,
            'std_dev': v.std_dev,
            'baseline_success': v.baseline_success
        } for k, v in transform_results.items()},
        'adversarial_results': {k: {
            'success_rate': v.success_rate,
            'robustness_score': v.robustness_score,
            'std_dev': v.std_dev,
            'baseline_success': v.baseline_success
        } for k, v in adv_results.items()} if adv_results else {}
    }
    
    # Save to JSON
    with open(output_dir / 'robustness_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save DataFrames to CSV
    if not noise_df.empty:
        noise_df.to_csv(output_dir / 'noise_robustness.csv', index=False)
    if not transform_df.empty:
        transform_df.to_csv(output_dir / 'transformation_robustness.csv', index=False)
    if not adv_df.empty:
        adv_df.to_csv(output_dir / 'adversarial_robustness.csv', index=False)
    
    print(f"✓ Comprehensive report saved to {output_dir}")
    
    return summary


def main():
    """Run comprehensive robustness analysis."""
    
    print("Vision Model Robustness Analysis")
    print("=" * 60)
    
    # Setup
    print("\n1. Setting up analysis environment...")
    verifier, evaluator = setup_verifier_and_evaluator()
    
    # Create output directory
    output_dir = Path('/tmp/robustness_analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # Run evaluations
    print("\n2. Running robustness evaluations...")
    
    # Noise robustness
    noise_results, noise_df = evaluate_noise_robustness(evaluator)
    
    # Transformation robustness
    transform_results, transform_df = evaluate_transformation_robustness(evaluator)
    
    # Adversarial robustness
    adv_results, adv_df = evaluate_adversarial_robustness(evaluator)
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    create_robustness_visualization(noise_df, transform_df, adv_df, output_dir)
    
    # Generate comprehensive report
    print("\n4. Generating final report...")
    summary = generate_robustness_report(
        noise_results, transform_results, adv_results,
        noise_df, transform_df, adv_df, output_dir
    )
    
    # Display final summary
    print("\n" + "=" * 60)
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"✓ Total tests completed: {summary['total_tests']}")
    print(f"✓ Average robustness score: {summary['average_robustness']:.3f}")
    print(f"✓ Robustness range: {summary['min_robustness']:.3f} - {summary['max_robustness']:.3f}")
    print(f"✓ Standard deviation: {summary['std_robustness']:.3f}")
    
    # Robustness assessment
    avg_robustness = summary['average_robustness']
    if avg_robustness > 0.8:
        assessment = "EXCELLENT - Model shows high robustness"
        icon = "🛡️"
    elif avg_robustness > 0.6:
        assessment = "GOOD - Model shows moderate robustness"
        icon = "✅"
    elif avg_robustness > 0.4:
        assessment = "FAIR - Model shows limited robustness"
        icon = "⚠️"
    else:
        assessment = "POOR - Model shows low robustness"
        icon = "❌"
    
    print(f"\n{icon} Overall Assessment: {assessment}")
    
    # Key vulnerabilities
    if summary['vulnerabilities']:
        print(f"\n⚠️ Key Vulnerabilities:")
        for vuln in summary['vulnerabilities'][:3]:  # Show top 3
            if vuln['type'] == 'transformation':
                print(f"   • {vuln['name']}: {vuln['robustness_score']:.3f} robustness")
            else:
                print(f"   • {vuln['description']}")
    
    # Recommendations
    print(f"\n📋 Recommendations:")
    if avg_robustness < 0.6:
        print("   • Consider adversarial training to improve robustness")
        print("   • Implement input preprocessing and filtering")
        print("   • Add robustness regularization during training")
    
    if summary['vulnerabilities']:
        print("   • Focus robustness improvements on identified vulnerabilities")
        print("   • Consider ensemble methods for critical applications")
    
    print("   • Monitor model performance in production environments")
    print("   • Implement continuous robustness testing")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"   📄 robustness_report.json - Comprehensive analysis")
    print(f"   📊 robustness_analysis.png - Visual summary")
    print(f"   📋 *_robustness.csv - Detailed data tables")
    
    print(f"\n{'=' * 60}")
    print("Robustness analysis completed!")
    
    return 0


if __name__ == "__main__":
    exit(main())