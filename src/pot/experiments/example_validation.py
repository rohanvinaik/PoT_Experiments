#!/usr/bin/env python3
"""
Example usage of the ResultValidator for PoT experiments.

This demonstrates various validation scenarios and how to use
the validator in different contexts.
"""

import json
import tempfile
import os
from pathlib import Path
from pot.experiments.result_validator import (
    ResultValidator, ClaimedMetrics, ValidationStatus, 
    ValidationSeverity, validate_experiment_results
)


def example_basic_validation():
    """Basic validation of experimental results."""
    print("=== Basic Result Validation ===\n")
    
    # Create sample results
    results = [
        {
            "experiment_id": "exp_001",
            "far": 0.009,
            "frr": 0.011,
            "accuracy": 0.99,
            "queries": 9,
            "processing_time": 0.45,
            "ground_truth": True,
            "verified": True,
            "p_value": 0.03
        },
        {
            "experiment_id": "exp_002",
            "far": 0.008,
            "frr": 0.012,
            "accuracy": 0.99,
            "queries": 11,
            "processing_time": 0.52,
            "ground_truth": True,
            "verified": True,
            "p_value": 0.04
        }
    ]
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(results, f)
        temp_file = f.name
    
    try:
        # Create validator and validate file
        validator = ResultValidator()
        report = validator.validate_single_file(temp_file)
        
        print(f"Validation Status: {report.status.value}")
        print(f"Valid: {report.is_valid}")
        print(f"Confidence Score: {report.confidence_score:.1%}")
        
        if report.issues:
            print(f"\nIssues Found ({len(report.issues)}):")
            for issue in report.issues[:5]:  # Show first 5 issues
                print(f"  - [{issue.severity.value}] {issue.message}")
        else:
            print("\n✓ No issues found!")
        
    finally:
        os.unlink(temp_file)


def example_integrity_check():
    """Example of data integrity validation."""
    print("\n=== Data Integrity Check ===\n")
    
    # Create data with various integrity issues
    problematic_data = [
        {
            "experiment_id": "exp_problem_1",
            "far": 1.5,  # Error: > 1
            "frr": -0.01,  # Error: negative
            "accuracy": 0.95,
            "queries": 0,  # Error: should be positive
            "verified": True
            # Warning: missing ground_truth
        },
        {
            "experiment_id": "exp_problem_2",
            "far": 0.6,
            "frr": 0.5,  # Warning: FAR + FRR > 1
            "accuracy": 0.4,  # Inconsistent with FAR/FRR
            "queries": -5  # Error: negative
        }
    ]
    
    validator = ResultValidator()
    report = validator.check_data_integrity(problematic_data)
    
    print(f"Integrity Check Status: {report.status.value}")
    print(f"Valid: {report.is_valid}")
    
    # Group issues by severity
    by_severity = {}
    for issue in report.issues:
        severity = issue.severity.value
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append(issue)
    
    for severity in ['critical', 'error', 'warning', 'info']:
        if severity in by_severity:
            print(f"\n{severity.upper()} Issues ({len(by_severity[severity])}):")
            for issue in by_severity[severity][:3]:
                print(f"  • {issue.message}")
                if issue.suggestion:
                    print(f"    → Suggestion: {issue.suggestion}")


def example_comparison_with_claims():
    """Example of comparing actual results with claimed metrics."""
    print("\n=== Comparison with Paper Claims ===\n")
    
    # Define expected metrics from paper
    claimed = ClaimedMetrics(
        far=0.01,
        frr=0.01,
        accuracy=0.99,
        avg_queries=10.0,
        confidence_level=0.95
    )
    
    # Actual experimental results
    actual_results = [
        {"far": 0.015, "frr": 0.018, "accuracy": 0.983, "queries": 12},
        {"far": 0.012, "frr": 0.016, "accuracy": 0.986, "queries": 11},
        {"far": 0.014, "frr": 0.017, "accuracy": 0.984, "queries": 13}
    ]
    
    validator = ResultValidator(tolerance=0.1)  # 10% tolerance
    report = validator.compare_claimed_vs_actual(claimed, actual_results)
    
    print(f"Comparison Status: {report.status.value}")
    print(f"Valid: {report.is_valid}")
    
    # Show metric comparisons
    print("\nMetric Comparisons:")
    for issue in report.issues:
        if issue.category == "Metric Comparison":
            status_symbol = "✓" if issue.severity == ValidationSeverity.INFO else "✗"
            print(f"  {status_symbol} {issue.field}: "
                  f"Expected={issue.expected:.3f}, Actual={issue.actual:.3f}")
    
    # Show reconciliation suggestions
    if report.reconciliation_suggestions:
        print("\nReconciliation Suggestions:")
        for i, suggestion in enumerate(report.reconciliation_suggestions, 1):
            print(f"{i}. {suggestion}")


def example_strict_mode():
    """Example of strict mode validation for CI/CD."""
    print("\n=== Strict Mode Validation (CI/CD) ===\n")
    
    # Results with minor deviations
    results = [
        {"far": 0.012, "frr": 0.013, "accuracy": 0.987, "queries": 11}
    ]
    
    claimed = {"far": 0.01, "frr": 0.01, "accuracy": 0.99, "avg_queries": 10}
    
    # Normal mode
    normal_validator = ResultValidator(strict_mode=False)
    normal_report = normal_validator.compare_claimed_vs_actual(claimed, results)
    
    # Strict mode
    strict_validator = ResultValidator(strict_mode=True)
    strict_report = strict_validator.compare_claimed_vs_actual(claimed, results)
    
    print(f"Normal Mode: {normal_report.status.value} (Valid: {normal_report.is_valid})")
    print(f"Strict Mode: {strict_report.status.value} (Valid: {strict_report.is_valid})")
    
    if not strict_report.is_valid:
        print("\n⚠️ Strict mode would fail CI/CD pipeline")
        print("Consider adjusting tolerance or addressing deviations")


def example_directory_validation():
    """Example of validating all results in a directory."""
    print("\n=== Directory Validation ===\n")
    
    # Create temporary directory with multiple result files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create different result files
        good_results = [
            {"experiment_id": "good_1", "far": 0.01, "frr": 0.01, "accuracy": 0.99}
        ]
        
        bad_results = [
            {"experiment_id": "bad_1", "far": 2.0, "frr": -1.0, "accuracy": 0.5}
        ]
        
        mixed_results = [
            {"experiment_id": "mixed_1", "far": 0.01, "frr": 0.01, "accuracy": 0.99},
            {"experiment_id": "mixed_2", "far": 0.5, "frr": 0.6, "accuracy": 0.4}
        ]
        
        # Save files
        with open(os.path.join(temp_dir, "good.json"), 'w') as f:
            json.dump(good_results, f)
        
        with open(os.path.join(temp_dir, "bad.json"), 'w') as f:
            json.dump(bad_results, f)
        
        with open(os.path.join(temp_dir, "mixed.json"), 'w') as f:
            json.dump(mixed_results, f)
        
        # Validate directory
        validator = ResultValidator()
        report = validator.validate_json_files(temp_dir)
        
        print(f"Directory: {temp_dir}")
        print(f"Total Files: {report.total_files}")
        print(f"Valid Files: {report.valid_files}")
        print(f"Invalid Files: {report.invalid_files}")
        print(f"Overall Status: {report.status.value}")
        
        # Show summary
        print("\n" + report.summary)
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)


def example_validation_report():
    """Example of generating comprehensive validation report."""
    print("\n=== Validation Report Generation ===\n")
    
    # Create sample data with various issues
    data = [
        {
            "experiment_id": "exp_001",
            "far": 0.008,
            "frr": 0.012,
            "accuracy": 0.99,
            "queries": 8,
            "ground_truth": True,
            "verified": True
        },
        {
            "experiment_id": "exp_002",
            "far": 0.02,  # Higher than expected
            "frr": 0.025,  # Higher than expected
            "accuracy": 0.977,  # Lower than expected
            "queries": 15,  # More queries than expected
            "ground_truth": True,
            "verified": False
        }
    ]
    
    # Run full validation pipeline
    validator = ResultValidator()
    
    # Check integrity
    validator.check_data_integrity(data)
    
    # Compare with claims
    claimed = {"far": 0.01, "frr": 0.01, "accuracy": 0.99, "avg_queries": 10}
    validator.compare_claimed_vs_actual(claimed, data)
    
    # Generate text report
    report_text = validator.generate_validation_report()
    
    # Show first part of report
    lines = report_text.split('\n')
    for line in lines[:30]:  # Show first 30 lines
        print(line)
    
    if len(lines) > 30:
        print("\n... (report continues)")
    
    # Save full report
    report_file = "validation_report_example.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    print(f"\nFull report saved to: {report_file}")
    
    # Also save JSON version
    json_file = "validation_report_example.json"
    validator.report.save(json_file)
    print(f"JSON report saved to: {json_file}")


def example_convenience_function():
    """Example using the convenience validation function."""
    print("\n=== Convenience Function ===\n")
    
    # Create temporary directory with results
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create result file
        results = [
            {"far": 0.009, "frr": 0.011, "accuracy": 0.99, "queries": 10},
            {"far": 0.011, "frr": 0.009, "accuracy": 0.99, "queries": 9}
        ]
        
        with open(os.path.join(temp_dir, "results.json"), 'w') as f:
            json.dump(results, f)
        
        # Use convenience function
        claimed = {"far": 0.01, "frr": 0.01, "accuracy": 0.99, "avg_queries": 10}
        
        is_valid, report = validate_experiment_results(
            temp_dir,
            claimed_metrics=claimed,
            strict_mode=False
        )
        
        print(f"Valid: {is_valid}")
        print(f"Status: {report.status.value}")
        print(f"Confidence: {report.confidence_score:.1%}")
        
        # Return code for CI/CD
        exit_code = 0 if is_valid else 1
        print(f"\nExit code for CI/CD: {exit_code}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run all examples
    example_basic_validation()
    example_integrity_check()
    example_comparison_with_claims()
    example_strict_mode()
    example_directory_validation()
    example_validation_report()
    example_convenience_function()
    
    print("\n" + "="*60)
    print("All validation examples completed!")
    print("="*60)
    
    # Clean up example files
    for file in ["validation_report_example.txt", "validation_report_example.json"]:
        if os.path.exists(file):
            os.remove(file)