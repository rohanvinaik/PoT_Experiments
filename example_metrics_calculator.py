#!/usr/bin/env python3
"""
Metrics Calculator Demo for PoT Framework

This script demonstrates the comprehensive metrics calculation system with:
- FAR/FRR calculation with bootstrap confidence intervals
- Statistical significance testing and hypothesis validation
- Paper claims comparison and discrepancy analysis
- Academic-style table formatting and reporting
- Cross-validation aggregation and performance analysis
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import json
from pathlib import Path

from pot.experiments.metrics_calculator import (
    MetricsCalculator, PaperClaims, MetricType, AggregationMethod, StatisticalTest,
    POT_PAPER_CLAIMS, create_metrics_calculator, calculate_all_metrics
)

def demo_basic_calculations():
    """Demonstrate basic metric calculations."""
    print("🧮 Basic Metrics Calculation Demo")
    print("=" * 50)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    # Simulate PoT verification results
    print("📊 Simulating PoT verification results...")
    
    np.random.seed(42)
    n_models = 1000
    
    # 70% legitimate models, 30% illegitimate
    legitimate_ratio = 0.7
    legitimate_models = int(n_models * legitimate_ratio)
    
    predictions = []
    labels = []
    
    # Legitimate models (high pass rate in challenges)
    for _ in range(legitimate_models):
        labels.append(1)  # Legitimate
        # 95% chance of being correctly identified
        predictions.append(1 if np.random.random() > 0.05 else 0)
    
    # Illegitimate models (low pass rate in challenges)  
    for _ in range(n_models - legitimate_models):
        labels.append(0)  # Illegitimate
        # 90% chance of being correctly identified
        predictions.append(0 if np.random.random() > 0.10 else 1)
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    print(f"  Total models evaluated: {n_models}")
    print(f"  Legitimate models: {legitimate_models} ({legitimate_ratio:.1%})")
    print(f"  Predicted legitimate: {np.sum(predictions)} ({np.mean(predictions):.1%})")
    
    # Calculate basic metrics
    far_result = calculator.calculate_far(predictions, labels)
    frr_result = calculator.calculate_frr(predictions, labels)
    accuracy_result = calculator.calculate_accuracy(predictions, labels)
    
    print(f"\n📈 Calculated Metrics:")
    print(f"  {far_result}")
    print(f"  {frr_result}")
    print(f"  {accuracy_result}")
    
    return {"far": far_result, "frr": frr_result, "accuracy": accuracy_result}

def demo_efficiency_analysis():
    """Demonstrate efficiency and query analysis."""
    print("\n⚡ Efficiency Analysis Demo")
    print("=" * 50)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    # Simulate adaptive vs fixed testing
    print("🔄 Comparing adaptive vs fixed-sample testing...")
    
    np.random.seed(42)
    n_experiments = 500
    fixed_sample_size = 50
    
    # Simulate stopping times for different model types
    legitimate_stopping_times = []
    illegitimate_stopping_times = []
    
    # Legitimate models stop faster (clear evidence)
    for _ in range(int(n_experiments * 0.7)):
        stopping_time = max(1, np.random.poisson(8))  # Average 8 queries
        legitimate_stopping_times.append(stopping_time)
    
    # Illegitimate models take longer (need more evidence)
    for _ in range(int(n_experiments * 0.3)):
        stopping_time = max(1, np.random.poisson(15))  # Average 15 queries
        illegitimate_stopping_times.append(stopping_time)
    
    all_stopping_times = legitimate_stopping_times + illegitimate_stopping_times
    
    print(f"  Fixed sample size: {fixed_sample_size} queries")
    print(f"  Legitimate models average: {np.mean(legitimate_stopping_times):.1f} queries")
    print(f"  Illegitimate models average: {np.mean(illegitimate_stopping_times):.1f} queries")
    print(f"  Overall adaptive average: {np.mean(all_stopping_times):.1f} queries")
    
    # Calculate efficiency metrics
    avg_queries_result = calculator.calculate_average_queries(all_stopping_times)
    efficiency_result = calculator.calculate_efficiency_gain(all_stopping_times, fixed_sample_size)
    
    print(f"\n📊 Efficiency Results:")
    print(f"  {avg_queries_result}")
    print(f"  {efficiency_result}")
    print(f"  Time savings: {efficiency_result.value * 100:.1f}% fewer queries needed")
    
    return {"average_queries": avg_queries_result, "efficiency_gain": efficiency_result}

def demo_statistical_testing():
    """Demonstrate statistical significance testing."""
    print("\n📈 Statistical Testing Demo")
    print("=" * 50)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    print("🧪 Testing statistical significance...")
    
    # Compare two different experimental setups
    np.random.seed(42)
    
    # Experiment A: Standard PoT
    accuracy_a = np.random.beta(95, 5, 100)  # ~95% accuracy
    
    # Experiment B: Enhanced PoT  
    accuracy_b = np.random.beta(98, 2, 100)  # ~98% accuracy
    
    print(f"  Setup A mean accuracy: {np.mean(accuracy_a):.4f}")
    print(f"  Setup B mean accuracy: {np.mean(accuracy_b):.4f}")
    
    # Perform various statistical tests
    tests = [
        (StatisticalTest.T_TEST, "Student's t-test"),
        (StatisticalTest.WILCOXON, "Wilcoxon rank-sum"),
        (StatisticalTest.BOOTSTRAP_TEST, "Bootstrap test"),
        (StatisticalTest.PERMUTATION_TEST, "Permutation test")
    ]
    
    print(f"\n🔬 Statistical Test Results:")
    for test_type, test_name in tests:
        result = calculator.perform_statistical_test(accuracy_a, accuracy_b, test_type)
        significance = "✅ Significant" if result.is_significant else "❌ Not significant"
        print(f"  {test_name}: p={result.p_value:.4f} ({significance})")
        
        if result.effect_size is not None:
            print(f"    Effect size (Cohen's d): {result.effect_size:.3f}")
    
    return tests

def demo_paper_comparison():
    """Demonstrate comparison with paper claims."""
    print("\n📋 Paper Claims Comparison Demo")
    print("=" * 50)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    # Create realistic experimental results
    print("🔬 Generating experimental results...")
    
    np.random.seed(42)
    n_samples = 2000
    
    # Simulate realistic PoT results with some deviation from paper claims
    labels = np.random.binomial(1, 0.6, n_samples)
    predictions = labels.copy()
    
    # Add realistic error patterns
    # False accepts: 1.2% (slightly higher than claimed 1%)
    fa_rate = 0.012
    negatives = np.where(labels == 0)[0]
    n_false_accepts = int(len(negatives) * fa_rate)
    fa_indices = np.random.choice(negatives, size=n_false_accepts, replace=False)
    predictions[fa_indices] = 1
    
    # False rejects: 0.8% (slightly lower than claimed 1%)
    fr_rate = 0.008
    positives = np.where(labels == 1)[0]
    n_false_rejects = int(len(positives) * fr_rate)
    fr_indices = np.random.choice(positives, size=n_false_rejects, replace=False)
    predictions[fr_indices] = 0
    
    # Generate stopping times with efficiency slightly better than claimed
    stopping_times = np.random.poisson(9, n_samples).tolist()  # Slightly better than claimed 10
    
    # Calculate comprehensive metrics
    metrics = calculate_all_metrics(predictions, labels, stopping_times, 100, calculator)
    
    print(f"📊 Experimental Results:")
    for name, result in metrics.items():
        print(f"  {name.upper()}: {result.value:.4f} [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
    
    # Compare to paper claims
    print(f"\n📋 Paper Claims:")
    print(f"  FAR: {POT_PAPER_CLAIMS.far:.4f}")
    print(f"  FRR: {POT_PAPER_CLAIMS.frr:.4f}")
    print(f"  Accuracy: {POT_PAPER_CLAIMS.accuracy:.4f}")
    print(f"  Average queries: {POT_PAPER_CLAIMS.average_queries:.1f}")
    print(f"  Efficiency gain: {POT_PAPER_CLAIMS.efficiency_gain:.4f}")
    
    # Generate discrepancy report
    discrepancy_report = calculator.compare_to_paper_claims(metrics, POT_PAPER_CLAIMS)
    
    print(f"\n🔍 Discrepancy Analysis:")
    print(f"  Overall assessment: {discrepancy_report.overall_assessment}")
    print(f"  Significant discrepancies: {len(discrepancy_report.significant_discrepancies)}")
    print(f"  Suspicious patterns: {len(discrepancy_report.suspicious_patterns)}")
    
    if discrepancy_report.significant_discrepancies:
        print(f"  📋 Discrepancies found:")
        for disc in discrepancy_report.significant_discrepancies[:3]:  # Show first 3
            print(f"    • {disc}")
    
    if discrepancy_report.suspicious_patterns:
        print(f"  ⚠️  Suspicious patterns:")
        for pattern in discrepancy_report.suspicious_patterns[:3]:  # Show first 3
            print(f"    • {pattern}")
    
    return metrics, discrepancy_report

def demo_cross_validation():
    """Demonstrate cross-validation style analysis."""
    print("\n🔄 Cross-Validation Analysis Demo")
    print("=" * 50)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    print("📊 Simulating 5-fold cross-validation...")
    
    np.random.seed(42)
    k_folds = 5
    n_per_fold = 400
    
    fold_metrics = []
    
    for fold in range(k_folds):
        print(f"\n  Fold {fold + 1}:")
        
        # Generate data for this fold
        labels = np.random.binomial(1, 0.65, n_per_fold)
        predictions = labels.copy()
        
        # Add fold-specific variation in error rates
        base_error_rate = 0.02
        fold_variation = 0.005 * (fold - 2)  # -0.01 to +0.01 variation
        error_rate = max(0.001, base_error_rate + fold_variation)
        
        n_errors = int(n_per_fold * error_rate)
        error_indices = np.random.choice(n_per_fold, size=n_errors, replace=False)
        predictions[error_indices] = 1 - predictions[error_indices]
        
        # Calculate fold metrics
        fold_result = calculate_all_metrics(predictions, labels, calculator=calculator)
        fold_metrics.append(fold_result)
        
        print(f"    FAR: {fold_result['far'].value:.4f}")
        print(f"    FRR: {fold_result['frr'].value:.4f}")
        print(f"    Accuracy: {fold_result['accuracy'].value:.4f}")
    
    # Aggregate cross-validation results
    far_values = np.array([fold['far'].value for fold in fold_metrics])
    frr_values = np.array([fold['frr'].value for fold in fold_metrics])
    accuracy_values = np.array([fold['accuracy'].value for fold in fold_metrics])
    
    # Calculate aggregated confidence intervals
    far_ci = calculator.calculate_confidence_intervals(far_values)
    frr_ci = calculator.calculate_confidence_intervals(frr_values)
    accuracy_ci = calculator.calculate_confidence_intervals(accuracy_values)
    
    print(f"\n📊 Cross-Validation Summary:")
    print(f"  FAR: {np.mean(far_values):.4f} ± {np.std(far_values):.4f} | 95% CI: [{far_ci[0]:.4f}, {far_ci[1]:.4f}]")
    print(f"  FRR: {np.mean(frr_values):.4f} ± {np.std(frr_values):.4f} | 95% CI: [{frr_ci[0]:.4f}, {frr_ci[1]:.4f}]")
    print(f"  Accuracy: {np.mean(accuracy_values):.4f} ± {np.std(accuracy_values):.4f} | 95% CI: [{accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f}]")
    
    # Statistical consistency check
    print(f"\n🔬 Consistency Analysis:")
    far_consistency = calculator.perform_statistical_test(
        far_values, np.array([POT_PAPER_CLAIMS.far]), StatisticalTest.T_TEST
    )
    print(f"  FAR vs paper claim: p={far_consistency.p_value:.4f} ({'Consistent' if not far_consistency.is_significant else 'Inconsistent'})")
    
    return fold_metrics, (far_values, frr_values, accuracy_values)

def demo_table_formatting():
    """Demonstrate academic table formatting."""
    print("\n📋 Academic Table Formatting Demo")
    print("=" * 50)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    # Create comprehensive metrics
    np.random.seed(42)
    n_samples = 1500
    
    labels = np.random.binomial(1, 0.6, n_samples)
    predictions = labels.copy()
    
    # Add realistic errors
    n_errors = int(n_samples * 0.015)  # 1.5% total error rate
    error_indices = np.random.choice(n_samples, size=n_errors, replace=False)
    predictions[error_indices] = 1 - predictions[error_indices]
    
    stopping_times = np.random.poisson(9.5, n_samples).tolist()
    
    metrics = calculate_all_metrics(predictions, labels, stopping_times, 50, calculator)
    
    print("🎨 Different Table Styles:")
    
    # Academic style (most detailed)
    print("\n📊 ACADEMIC STYLE:")
    academic_table = calculator.format_results_table(metrics, POT_PAPER_CLAIMS, "academic")
    print(academic_table)
    
    # Simple style
    print("\n📋 SIMPLE STYLE:")
    simple_table = calculator.format_results_table(metrics, POT_PAPER_CLAIMS, "simple")
    print(simple_table)
    
    # Markdown style
    print("\n📝 MARKDOWN STYLE:")
    markdown_table = calculator.format_results_table(metrics, POT_PAPER_CLAIMS, "markdown")
    print(markdown_table)
    
    return metrics

def demo_validation_and_quality_control():
    """Demonstrate result validation and quality control."""
    print("\n🔍 Validation & Quality Control Demo")
    print("=" * 50)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    print("🧪 Testing various data quality scenarios...")
    
    # Scenario 1: High-quality data
    print("\n  Scenario 1: High-Quality Data")
    np.random.seed(42)
    good_labels = np.random.binomial(1, 0.5, 1000)
    good_predictions = good_labels.copy()
    n_errors = int(1000 * 0.02)  # 2% error rate
    error_indices = np.random.choice(1000, size=n_errors, replace=False)
    good_predictions[error_indices] = 1 - good_predictions[error_indices]
    
    good_metrics = calculate_all_metrics(good_predictions, good_labels, calculator=calculator)
    good_warnings = calculator.validate_results(good_metrics)
    
    print(f"    Validation warnings: {len(good_warnings)}")
    if good_warnings:
        for warning in good_warnings[:2]:
            print(f"      • {warning}")
    
    # Scenario 2: Suspicious data (too perfect)
    print("\n  Scenario 2: Suspicious Data (Too Perfect)")
    perfect_labels = np.array([1] * 500 + [0] * 500)
    perfect_predictions = perfect_labels.copy()  # Exactly perfect
    
    perfect_metrics = calculate_all_metrics(perfect_predictions, perfect_labels, calculator=calculator)
    perfect_warnings = calculator.validate_results(perfect_metrics)
    
    print(f"    Validation warnings: {len(perfect_warnings)}")
    if perfect_warnings:
        for warning in perfect_warnings:
            print(f"      • {warning}")
    
    # Scenario 3: Inconsistent data
    print("\n  Scenario 3: Inconsistent Data")
    # Create metrics that don't match internally
    from pot.experiments.metrics_calculator import MetricResult
    
    inconsistent_metrics = {
        "far": MetricResult(
            metric_type=MetricType.FAR,
            value=0.1,  # High FAR
            confidence_interval=(0.08, 0.12),
            sample_size=1000,
            standard_error=0.01,
            method=AggregationMethod.MEAN
        ),
        "frr": MetricResult(
            metric_type=MetricType.FRR,
            value=0.1,  # High FRR
            confidence_interval=(0.08, 0.12),
            sample_size=1000,
            standard_error=0.01,
            method=AggregationMethod.MEAN
        ),
        "accuracy": MetricResult(
            metric_type=MetricType.ACCURACY,
            value=0.95,  # High accuracy (inconsistent with high error rates)
            confidence_interval=(0.93, 0.97),
            sample_size=1000,
            standard_error=0.01,
            method=AggregationMethod.MEAN
        )
    }
    
    inconsistent_warnings = calculator.validate_results(inconsistent_metrics)
    
    print(f"    Validation warnings: {len(inconsistent_warnings)}")
    if inconsistent_warnings:
        for warning in inconsistent_warnings:
            print(f"      • {warning}")
    
    return good_warnings, perfect_warnings, inconsistent_warnings

def main():
    """Run comprehensive metrics calculator demonstration."""
    print("🧮 PoT Metrics Calculator Demo")
    print("=" * 70)
    print("Comprehensive demonstration of metrics calculation with")
    print("statistical analysis, paper comparison, and validation.\n")
    
    try:
        # Run all demonstrations
        basic_metrics = demo_basic_calculations()
        efficiency_metrics = demo_efficiency_analysis()
        statistical_results = demo_statistical_testing()
        experimental_metrics, discrepancy_report = demo_paper_comparison()
        cv_results, cv_aggregated = demo_cross_validation()
        formatted_metrics = demo_table_formatting()
        validation_results = demo_validation_and_quality_control()
        
        # Save comprehensive results
        print("\n💾 Saving Comprehensive Results")
        print("=" * 50)
        
        # Combine all metrics
        all_metrics = {**basic_metrics, **efficiency_metrics, **experimental_metrics}
        
        # Create output directory
        output_dir = Path("metrics_demo_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Save metrics
        metrics_file = output_dir / "calculated_metrics.json"
        metrics_data = {
            name: {
                "value": result.value,
                "confidence_interval": result.confidence_interval,
                "sample_size": result.sample_size,
                "standard_error": result.standard_error,
                "metric_type": result.metric_type.value
            }
            for name, result in all_metrics.items()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"✅ Metrics saved to: {metrics_file}")
        
        # Save discrepancy report
        calculator = create_metrics_calculator()
        report_file = output_dir / "discrepancy_report.md"
        report_text = calculator.generate_discrepancy_report(discrepancy_report, report_file)
        
        print(f"✅ Discrepancy report saved to: {report_file}")
        
        # Save formatted tables
        tables_file = output_dir / "formatted_tables.md"
        with open(tables_file, 'w') as f:
            f.write("# PoT Metrics Results\n\n")
            f.write("## Academic Style Table\n\n")
            f.write(calculator.format_results_table(formatted_metrics, POT_PAPER_CLAIMS, "academic"))
            f.write("\n\n## Markdown Style Table\n\n")
            f.write(calculator.format_results_table(formatted_metrics, POT_PAPER_CLAIMS, "markdown"))
        
        print(f"✅ Formatted tables saved to: {tables_file}")
        
        # Final summary
        print(f"\n🎉 Demo completed successfully!")
        print(f"\n✅ Key Features Demonstrated:")
        print(f"   • FAR/FRR calculation with bootstrap confidence intervals")
        print(f"   • Statistical significance testing (t-test, Wilcoxon, bootstrap, permutation)")
        print(f"   • Efficiency analysis and query optimization metrics")
        print(f"   • Paper claims comparison and discrepancy detection")
        print(f"   • Cross-validation aggregation and consistency analysis")
        print(f"   • Academic-style table formatting for publications")
        print(f"   • Result validation and quality control checks")
        print(f"   • Comprehensive reporting and documentation")
        
        # Performance summary
        print(f"\n📊 Results Summary:")
        far_val = basic_metrics['far'].value
        frr_val = basic_metrics['frr'].value
        acc_val = basic_metrics['accuracy'].value
        eff_val = efficiency_metrics['efficiency_gain'].value
        
        print(f"   • False Accept Rate: {far_val:.1%} (vs {POT_PAPER_CLAIMS.far:.1%} claimed)")
        print(f"   • False Reject Rate: {frr_val:.1%} (vs {POT_PAPER_CLAIMS.frr:.1%} claimed)")
        print(f"   • Accuracy: {acc_val:.1%} (vs {POT_PAPER_CLAIMS.accuracy:.1%} claimed)")
        print(f"   • Efficiency Gain: {eff_val:.1%} (vs {POT_PAPER_CLAIMS.efficiency_gain:.1%} claimed)")
        print(f"   • Overall Assessment: {discrepancy_report.overall_assessment}")
        
        print(f"\n📁 Output Files:")
        print(f"   • {metrics_file}")
        print(f"   • {report_file}")
        print(f"   • {tables_file}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)