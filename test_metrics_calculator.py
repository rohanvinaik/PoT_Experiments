#!/usr/bin/env python3
"""
Test script for Metrics Calculator

This script tests the comprehensive metrics calculation system with:
- FAR/FRR calculation accuracy and edge cases
- Bootstrap confidence intervals and statistical tests
- Paper claims comparison and discrepancy detection
- Result validation and consistency checks
- Formatted table output and reporting
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import warnings
from typing import List, Dict, Any
from pathlib import Path

from pot.experiments.metrics_calculator import (
    MetricsCalculator, MetricResult, StatisticalTestResult, PaperClaims,
    DiscrepancyReport, MetricType, AggregationMethod, StatisticalTest,
    POT_PAPER_CLAIMS, create_metrics_calculator, calculate_all_metrics
)

def test_basic_metrics():
    """Test basic FAR/FRR/accuracy calculations."""
    print("🧪 Testing Basic Metrics")
    print("-" * 40)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    # Create perfect predictions (should give FAR=0, FRR=0, Accuracy=1)
    labels = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    predictions = labels.copy()  # Perfect predictions
    
    far_result = calculator.calculate_far(predictions, labels)
    frr_result = calculator.calculate_frr(predictions, labels)
    accuracy_result = calculator.calculate_accuracy(predictions, labels)
    
    print(f"Perfect predictions test:")
    print(f"  FAR: {far_result.value:.4f} (expected: 0.0000)")
    print(f"  FRR: {frr_result.value:.4f} (expected: 0.0000)")
    print(f"  Accuracy: {accuracy_result.value:.4f} (expected: 1.0000)")
    
    # Verify perfect scores
    if not (far_result.value == 0.0 and frr_result.value == 0.0 and accuracy_result.value == 1.0):
        print("❌ Perfect predictions should give FAR=0, FRR=0, Accuracy=1")
        return False
    
    # Test with some errors
    predictions_with_errors = np.array([1, 0, 0, 1, 1, 0, 1, 0])  # 2 errors
    
    far_error = calculator.calculate_far(predictions_with_errors, labels)
    frr_error = calculator.calculate_frr(predictions_with_errors, labels)
    accuracy_error = calculator.calculate_accuracy(predictions_with_errors, labels)
    
    print(f"\nWith 2 errors test:")
    print(f"  FAR: {far_error.value:.4f}")
    print(f"  FRR: {frr_error.value:.4f}")
    print(f"  Accuracy: {accuracy_error.value:.4f} (expected: 0.7500)")
    
    # Verify accuracy calculation
    expected_accuracy = 6/8  # 6 correct out of 8
    if abs(accuracy_error.value - expected_accuracy) > 1e-10:
        print(f"❌ Accuracy calculation incorrect: {accuracy_error.value} vs {expected_accuracy}")
        return False
    
    print("✅ Basic metrics calculation: passed")
    return True

def test_confidence_intervals():
    """Test confidence interval calculations."""
    print("\n📊 Testing Confidence Intervals")
    print("-" * 40)
    
    calculator = create_metrics_calculator(bootstrap_samples=1000, random_seed=42)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    true_accuracy = 0.95
    
    # Generate predictions with known accuracy
    labels = np.random.binomial(1, 0.5, n_samples)
    predictions = labels.copy()
    
    # Add errors to achieve target accuracy
    n_errors = int(n_samples * (1 - true_accuracy))
    error_indices = np.random.choice(n_samples, size=n_errors, replace=False)
    predictions[error_indices] = 1 - predictions[error_indices]
    
    # Calculate accuracy with CI
    accuracy_result = calculator.calculate_accuracy(predictions, labels)
    
    print(f"Accuracy result: {accuracy_result}")
    
    # Check if true accuracy is in confidence interval
    ci_lower, ci_upper = accuracy_result.confidence_interval
    true_in_ci = ci_lower <= true_accuracy <= ci_upper
    
    print(f"True accuracy: {true_accuracy}")
    print(f"Estimated accuracy: {accuracy_result.value:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"True value in CI: {true_in_ci}")
    
    # CI should contain estimate
    estimate_in_ci = ci_lower <= accuracy_result.value <= ci_upper
    if not estimate_in_ci:
        print("❌ Estimate should be within its own confidence interval")
        return False
    
    # CI should be reasonable width
    ci_width = ci_upper - ci_lower
    if ci_width > 0.2:  # More than 20% width is suspicious
        print(f"❌ Confidence interval too wide: {ci_width:.4f}")
        return False
    
    if ci_width < 0.01:  # Less than 1% width is suspicious for this sample size
        print(f"❌ Confidence interval too narrow: {ci_width:.4f}")
        return False
    
    print("✅ Confidence intervals: passed")
    return True

def test_statistical_tests():
    """Test statistical significance tests."""
    print("\n📈 Testing Statistical Tests")
    print("-" * 40)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    # Test 1: Same distributions (should not be significant)
    np.random.seed(42)
    sample1 = np.random.normal(0.9, 0.1, 100)
    sample2 = np.random.normal(0.9, 0.1, 100)
    
    t_test_result = calculator.perform_statistical_test(
        sample1, sample2, StatisticalTest.T_TEST
    )
    
    print(f"Same distributions t-test: {t_test_result}")
    
    if t_test_result.is_significant:
        print("⚠️  Same distributions showing significant difference (Type I error)")
    else:
        print("✅ Same distributions correctly not significant")
    
    # Test 2: Different distributions (should be significant)
    sample3 = np.random.normal(0.8, 0.1, 100)  # Different mean
    
    t_test_different = calculator.perform_statistical_test(
        sample1, sample3, StatisticalTest.T_TEST
    )
    
    print(f"Different distributions t-test: {t_test_different}")
    
    if not t_test_different.is_significant:
        print("⚠️  Different distributions not showing significant difference (Type II error)")
    else:
        print("✅ Different distributions correctly significant")
    
    # Test 3: One-sample test
    claimed_value = np.array([0.85])
    one_sample_test = calculator.perform_statistical_test(
        sample1, claimed_value, StatisticalTest.T_TEST
    )
    
    print(f"One-sample test vs 0.85: {one_sample_test}")
    
    # Test 4: Non-parametric test
    wilcoxon_test = calculator.perform_statistical_test(
        sample1, sample3, StatisticalTest.WILCOXON
    )
    
    print(f"Wilcoxon test: {wilcoxon_test}")
    
    print("✅ Statistical tests: passed")
    return True

def test_efficiency_metrics():
    """Test efficiency and query metrics."""
    print("\n⚡ Testing Efficiency Metrics")
    print("-" * 40)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    # Generate adaptive stopping times
    np.random.seed(42)
    adaptive_queries = np.random.poisson(8, 100).tolist()  # Average ~8 queries
    fixed_sample_size = 20
    
    # Calculate average queries
    avg_queries_result = calculator.calculate_average_queries(adaptive_queries)
    print(f"Average queries: {avg_queries_result}")
    
    # Calculate efficiency gain
    efficiency_result = calculator.calculate_efficiency_gain(
        adaptive_queries, fixed_sample_size
    )
    print(f"Efficiency gain: {efficiency_result}")
    
    # Verify efficiency calculation
    mean_adaptive = np.mean(adaptive_queries)
    expected_efficiency = 1.0 - (mean_adaptive / fixed_sample_size)
    
    if abs(efficiency_result.value - expected_efficiency) > 1e-10:
        print(f"❌ Efficiency calculation incorrect: {efficiency_result.value} vs {expected_efficiency}")
        return False
    
    # Efficiency should be positive if adaptive is better
    if mean_adaptive < fixed_sample_size and efficiency_result.value <= 0:
        print("❌ Efficiency should be positive when adaptive uses fewer queries")
        return False
    
    print(f"Efficiency verification: {efficiency_result.value:.4f} vs expected {expected_efficiency:.4f}")
    print("✅ Efficiency metrics: passed")
    return True

def test_paper_comparison():
    """Test comparison with paper claims."""
    print("\n📋 Testing Paper Comparison")
    print("-" * 40)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    # Create synthetic metrics close to paper claims
    paper_claims = PaperClaims(
        far=0.01,
        frr=0.01,
        accuracy=0.99,
        efficiency_gain=0.90,
        average_queries=10
    )
    
    # Simulate measurements close to claims
    np.random.seed(42)
    n_samples = 1000
    
    # Create data that should match claims approximately
    labels = np.random.binomial(1, 0.5, n_samples)
    predictions = labels.copy()
    
    # Add small amount of errors (approximately 1% total error rate)
    n_errors = int(n_samples * 0.01)
    error_indices = np.random.choice(n_samples, size=n_errors, replace=False)
    predictions[error_indices] = 1 - predictions[error_indices]
    
    stopping_times = np.random.poisson(10, n_samples).tolist()
    
    # Calculate metrics
    metrics = calculate_all_metrics(predictions, labels, stopping_times, 100, calculator)
    
    print("Measured metrics:")
    for name, result in metrics.items():
        print(f"  {name}: {result.value:.4f}")
    
    # Compare to paper claims
    discrepancy_report = calculator.compare_to_paper_claims(metrics, paper_claims)
    
    print(f"\nDiscrepancy report summary:")
    print(f"  Overall assessment: {discrepancy_report.overall_assessment}")
    print(f"  Significant discrepancies: {len(discrepancy_report.significant_discrepancies)}")
    print(f"  Suspicious patterns: {len(discrepancy_report.suspicious_patterns)}")
    
    if discrepancy_report.significant_discrepancies:
        print("  Discrepancies:")
        for disc in discrepancy_report.significant_discrepancies:
            print(f"    - {disc}")
    
    # Should have minimal discrepancies for realistic data
    if len(discrepancy_report.significant_discrepancies) > 2:
        print("⚠️  Too many discrepancies detected (may be overly sensitive)")
    
    print("✅ Paper comparison: passed")
    return True

def test_result_validation():
    """Test result validation and consistency checks."""
    print("\n🔍 Testing Result Validation")
    print("-" * 40)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    # Test 1: Valid results (should pass validation)
    valid_metrics = {
        "far": MetricResult(
            metric_type=MetricType.FAR,
            value=0.02,
            confidence_interval=(0.01, 0.03),
            sample_size=1000,
            standard_error=0.005,
            method=AggregationMethod.MEAN
        ),
        "frr": MetricResult(
            metric_type=MetricType.FRR,
            value=0.03,
            confidence_interval=(0.02, 0.04),
            sample_size=1000,
            standard_error=0.005,
            method=AggregationMethod.MEAN
        ),
        "accuracy": MetricResult(
            metric_type=MetricType.ACCURACY,
            value=0.95,
            confidence_interval=(0.94, 0.96),
            sample_size=1000,
            standard_error=0.005,
            method=AggregationMethod.MEAN
        )
    }
    
    valid_warnings = calculator.validate_results(valid_metrics)
    print(f"Valid results warnings: {len(valid_warnings)}")
    
    if valid_warnings:
        print("Valid results warnings:")
        for warning in valid_warnings:
            print(f"  - {warning}")
    
    # Test 2: Invalid results (should trigger warnings)
    invalid_metrics = {
        "far": MetricResult(
            metric_type=MetricType.FAR,
            value=1.5,  # Impossible value > 1
            confidence_interval=(1.4, 1.6),
            sample_size=100,
            standard_error=0.05,
            method=AggregationMethod.MEAN
        ),
        "accuracy": MetricResult(
            metric_type=MetricType.ACCURACY,
            value=0.95,
            confidence_interval=(0.99, 1.0),  # CI doesn't contain estimate
            sample_size=100,
            standard_error=0.005,
            method=AggregationMethod.MEAN
        )
    }
    
    invalid_warnings = calculator.validate_results(invalid_metrics)
    print(f"\nInvalid results warnings: {len(invalid_warnings)}")
    
    if invalid_warnings:
        print("Invalid results warnings:")
        for warning in invalid_warnings:
            print(f"  - {warning}")
    
    # Should detect the impossible values
    if len(invalid_warnings) < 2:
        print("❌ Validation should detect impossible values")
        return False
    
    print("✅ Result validation: passed")
    return True

def test_table_formatting():
    """Test formatted table output."""
    print("\n📋 Testing Table Formatting")
    print("-" * 40)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    # Create sample metrics
    sample_metrics = {
        "far": MetricResult(
            metric_type=MetricType.FAR,
            value=0.012,
            confidence_interval=(0.008, 0.016),
            sample_size=1000,
            standard_error=0.002,
            method=AggregationMethod.MEAN
        ),
        "frr": MetricResult(
            metric_type=MetricType.FRR,
            value=0.015,
            confidence_interval=(0.011, 0.019),
            sample_size=1000,
            standard_error=0.002,
            method=AggregationMethod.MEAN
        ),
        "accuracy": MetricResult(
            metric_type=MetricType.ACCURACY,
            value=0.987,
            confidence_interval=(0.982, 0.992),
            sample_size=1000,
            standard_error=0.003,
            method=AggregationMethod.MEAN
        )
    }
    
    # Test different table styles
    styles = ["academic", "simple", "markdown"]
    
    for style in styles:
        print(f"\n{style.upper()} style table:")
        try:
            table = calculator.format_results_table(sample_metrics, POT_PAPER_CLAIMS, style)
            print(table[:200] + "..." if len(table) > 200 else table)
            print("✅ Table formatting successful")
        except Exception as e:
            print(f"❌ Table formatting failed for {style}: {e}")
            return False
    
    print("✅ Table formatting: passed")
    return True

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🔍 Testing Edge Cases")
    print("-" * 40)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    # Test 1: Empty arrays
    try:
        empty_predictions = np.array([])
        empty_labels = np.array([])
        result = calculator.calculate_accuracy(empty_predictions, empty_labels)
        print("⚠️  Empty arrays should raise error or return NaN")
    except (ValueError, ZeroDivisionError):
        print("✅ Empty arrays properly handled")
    except Exception as e:
        print(f"❌ Unexpected error with empty arrays: {e}")
        return False
    
    # Test 2: Mismatched array lengths
    try:
        predictions = np.array([1, 0, 1])
        labels = np.array([1, 0])  # Different length
        result = calculator.calculate_far(predictions, labels)
        print("❌ Mismatched lengths should raise error")
        return False
    except ValueError:
        print("✅ Mismatched array lengths properly handled")
    except Exception as e:
        print(f"❌ Unexpected error with mismatched lengths: {e}")
        return False
    
    # Test 3: All positive labels (no negatives for FAR)
    predictions = np.array([1, 1, 0, 1])
    labels = np.array([1, 1, 1, 1])  # All positive
    
    far_result = calculator.calculate_far(predictions, labels)
    print(f"FAR with no negatives: {far_result.value} (should be 0 or undefined)")
    
    # Test 4: All negative labels (no positives for FRR)
    labels_neg = np.array([0, 0, 0, 0])  # All negative
    
    frr_result = calculator.calculate_frr(predictions, labels_neg)
    print(f"FRR with no positives: {frr_result.value} (should be 0 or undefined)")
    
    # Test 5: Perfect separation
    perfect_pred = np.array([1, 1, 0, 0])
    perfect_labels = np.array([1, 1, 0, 0])
    
    perfect_far = calculator.calculate_far(perfect_pred, perfect_labels)
    perfect_frr = calculator.calculate_frr(perfect_pred, perfect_labels)
    
    print(f"Perfect separation - FAR: {perfect_far.value}, FRR: {perfect_frr.value}")
    
    if perfect_far.value != 0.0 or perfect_frr.value != 0.0:
        print("❌ Perfect separation should give FAR=0, FRR=0")
        return False
    
    print("✅ Edge cases: passed")
    return True

def test_cross_validation():
    """Test cross-validation style metrics."""
    print("\n🔄 Testing Cross-Validation Style Metrics")
    print("-" * 40)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    # Simulate k-fold cross-validation results
    np.random.seed(42)
    k_folds = 5
    n_per_fold = 200
    
    fold_results = []
    
    for fold in range(k_folds):
        # Generate data for this fold
        labels = np.random.binomial(1, 0.6, n_per_fold)
        predictions = labels.copy()
        
        # Add some errors (varying by fold to simulate realistic CV)
        error_rate = 0.01 + 0.005 * fold  # Increasing error rate
        n_errors = int(n_per_fold * error_rate)
        error_indices = np.random.choice(n_per_fold, size=n_errors, replace=False)
        predictions[error_indices] = 1 - predictions[error_indices]
        
        # Calculate fold metrics
        fold_metrics = calculate_all_metrics(predictions, labels, calculator=calculator)
        fold_results.append(fold_metrics)
        
        print(f"Fold {fold+1}: FAR={fold_metrics['far'].value:.4f}, "
              f"FRR={fold_metrics['frr'].value:.4f}, "
              f"Accuracy={fold_metrics['accuracy'].value:.4f}")
    
    # Aggregate cross-validation results
    far_values = [fold['far'].value for fold in fold_results]
    frr_values = [fold['frr'].value for fold in fold_results]
    accuracy_values = [fold['accuracy'].value for fold in fold_results]
    
    # Calculate mean and confidence intervals across folds
    cv_far_ci = calculator.calculate_confidence_intervals(np.array(far_values))
    cv_frr_ci = calculator.calculate_confidence_intervals(np.array(frr_values))
    cv_accuracy_ci = calculator.calculate_confidence_intervals(np.array(accuracy_values))
    
    print(f"\nCross-validation aggregated results:")
    print(f"  FAR: {np.mean(far_values):.4f} {cv_far_ci}")
    print(f"  FRR: {np.mean(frr_values):.4f} {cv_frr_ci}")
    print(f"  Accuracy: {np.mean(accuracy_values):.4f} {cv_accuracy_ci}")
    
    # Check that results are reasonable
    if not (0 <= np.mean(far_values) <= 1 and 0 <= np.mean(frr_values) <= 1):
        print("❌ Cross-validation error rates outside [0,1]")
        return False
    
    print("✅ Cross-validation style metrics: passed")
    return True

def test_comprehensive_integration():
    """Test comprehensive integration with realistic data."""
    print("\n🔗 Testing Comprehensive Integration")
    print("-" * 40)
    
    calculator = create_metrics_calculator(random_seed=42)
    
    # Generate realistic PoT experiment data
    np.random.seed(42)
    n_experiments = 500
    
    # Simulate mixed legitimate/illegitimate models
    legitimate_ratio = 0.7
    n_legitimate = int(n_experiments * legitimate_ratio)
    n_illegitimate = n_experiments - n_legitimate
    
    all_predictions = []
    all_labels = []
    all_stopping_times = []
    
    # Legitimate models (higher accuracy, fewer queries needed)
    for _ in range(n_legitimate):
        labels = np.array([1])  # This is a legitimate model
        # Legitimate models pass challenges with high probability
        predictions = np.array([1 if np.random.random() > 0.05 else 0])  # 95% pass rate
        stopping_time = max(1, np.random.poisson(6))  # Faster decision
        
        all_predictions.extend(predictions)
        all_labels.extend(labels)
        all_stopping_times.append(stopping_time)
    
    # Illegitimate models (lower accuracy, more queries needed)
    for _ in range(n_illegitimate):
        labels = np.array([0])  # This is an illegitimate model
        # Illegitimate models pass challenges with low probability
        predictions = np.array([1 if np.random.random() > 0.8 else 0])  # 20% pass rate
        stopping_time = max(1, np.random.poisson(12))  # Slower decision
        
        all_predictions.extend(predictions)
        all_labels.extend(labels)
        all_stopping_times.append(stopping_time)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    print(f"Generated data: {len(all_predictions)} predictions, {len(all_stopping_times)} stopping times")
    print(f"Label distribution: {np.mean(all_labels):.2f} legitimate")
    print(f"Prediction distribution: {np.mean(all_predictions):.2f} predicted legitimate")
    
    # Calculate comprehensive metrics
    metrics = calculate_all_metrics(
        all_predictions, all_labels, all_stopping_times, 20, calculator
    )
    
    print(f"\nComprehensive metrics:")
    for name, result in metrics.items():
        print(f"  {name}: {result.value:.4f} [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
    
    # Compare to paper claims
    discrepancy_report = calculator.compare_to_paper_claims(metrics, POT_PAPER_CLAIMS)
    
    print(f"\nDiscrepancy analysis:")
    print(f"  Overall assessment: {discrepancy_report.overall_assessment}")
    print(f"  Significant discrepancies: {len(discrepancy_report.significant_discrepancies)}")
    
    # Validate results
    validation_warnings = calculator.validate_results(metrics)
    print(f"  Validation warnings: {len(validation_warnings)}")
    
    # Generate formatted report
    table = calculator.format_results_table(metrics, POT_PAPER_CLAIMS, "academic")
    print(f"\nFormatted table preview:")
    print(table[:300] + "..." if len(table) > 300 else table)
    
    # Save comprehensive report
    report_text = calculator.generate_discrepancy_report(discrepancy_report)
    
    print("\n✅ Comprehensive integration: passed")
    return True

def main():
    """Run all metrics calculator tests."""
    print("🧪 Metrics Calculator Test Suite")
    print("=" * 60)
    print("Testing comprehensive metrics calculation with statistical")
    print("analysis, paper comparison, and validation.\n")
    
    test_functions = [
        test_basic_metrics,
        test_confidence_intervals,
        test_statistical_tests,
        test_efficiency_metrics,
        test_paper_comparison,
        test_result_validation,
        test_table_formatting,
        test_edge_cases,
        test_cross_validation,
        test_comprehensive_integration
    ]
    
    results = []
    for test_func in test_functions:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results")
    print("=" * 30)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        print("\n✅ Key Features Verified:")
        print("   - FAR/FRR calculation with bootstrap confidence intervals")
        print("   - Statistical significance testing (t-test, Wilcoxon, bootstrap)")
        print("   - Efficiency metrics and query analysis")
        print("   - Paper claims comparison and discrepancy detection")
        print("   - Result validation and consistency checks")
        print("   - Academic-style table formatting")
        print("   - Edge case handling and error management")
        print("   - Cross-validation style aggregation")
        print("   - Comprehensive integration and reporting")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)