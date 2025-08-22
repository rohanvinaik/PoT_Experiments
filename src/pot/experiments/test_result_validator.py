#!/usr/bin/env python3
"""
Test suite for ResultValidator class.
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
import numpy as np

# Add parent directory to path for pot imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pot.experiments.result_validator import (
    ResultValidator, ValidationReport, ValidationStatus, 
    ValidationSeverity, ClaimedMetrics, validate_experiment_results
)


class TestResultValidator(unittest.TestCase):
    """Test cases for ResultValidator functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = ResultValidator(strict_mode=False)
        
        # Create valid sample data
        self.valid_data = [
            {
                "experiment_id": "exp_001",
                "far": 0.01,
                "frr": 0.01,
                "accuracy": 0.99,
                "queries": 10,
                "processing_time": 0.5,
                "ground_truth": True,
                "verified": True,
                "p_value": 0.05,
                "threshold": 0.5
            },
            {
                "experiment_id": "exp_002",
                "far": 0.008,
                "frr": 0.012,
                "accuracy": 0.99,
                "queries": 8,
                "processing_time": 0.4,
                "ground_truth": True,
                "verified": True,
                "p_value": 0.03,
                "threshold": 0.45
            }
        ]
        
        # Create invalid sample data
        self.invalid_data = [
            {
                "experiment_id": "exp_invalid_001",
                "far": 1.5,  # Invalid: > 1
                "frr": -0.01,  # Invalid: < 0
                "accuracy": 0.95,
                "queries": 0,  # Invalid: should be > 0
                "processing_time": -1.0,  # Invalid: negative
                "p_value": 2.0,  # Invalid: > 1
                "threshold": 1.5  # Invalid: > 1
            },
            {
                "experiment_id": "exp_invalid_002",
                "far": np.nan,  # Invalid: NaN
                "frr": np.inf,  # Invalid: infinite
                "accuracy": "invalid",  # Invalid: wrong type
                "queries": "ten",  # Invalid: wrong type
                "verified": True
                # Missing ground_truth
            }
        ]
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test validator initialization."""
        # Default initialization
        validator = ResultValidator()
        self.assertFalse(validator.strict_mode)
        self.assertEqual(validator.tolerance, 0.1)
        
        # Strict mode initialization
        strict_validator = ResultValidator(strict_mode=True, tolerance=0.05)
        self.assertTrue(strict_validator.strict_mode)
        self.assertEqual(strict_validator.tolerance, 0.05)
    
    def test_validate_single_valid_file(self):
        """Test validation of a single valid file."""
        # Save valid data to file
        file_path = os.path.join(self.temp_dir, "valid_results.json")
        with open(file_path, 'w') as f:
            json.dump(self.valid_data, f)
        
        # Validate
        report = self.validator.validate_single_file(file_path)
        
        self.assertEqual(report.status, ValidationStatus.PASSED)
        self.assertEqual(report.valid_files, 1)
        self.assertEqual(report.invalid_files, 0)
        self.assertTrue(report.is_valid)
    
    def test_validate_single_invalid_file(self):
        """Test validation of a single invalid file."""
        # Save invalid data to file
        file_path = os.path.join(self.temp_dir, "invalid_results.json")
        with open(file_path, 'w') as f:
            json.dump(self.invalid_data, f, default=str)
        
        # Validate
        report = self.validator.validate_single_file(file_path)
        
        self.assertIn(report.status, [ValidationStatus.FAILED, ValidationStatus.CRITICAL_FAILURE])
        self.assertFalse(report.is_valid)
        self.assertGreater(len(report.issues), 0)
    
    def test_validate_json_files_directory(self):
        """Test validation of multiple JSON files in a directory."""
        # Create multiple files
        valid_file = os.path.join(self.temp_dir, "valid.json")
        invalid_file = os.path.join(self.temp_dir, "invalid.json")
        
        with open(valid_file, 'w') as f:
            json.dump(self.valid_data, f)
        
        with open(invalid_file, 'w') as f:
            json.dump(self.invalid_data, f, default=str)
        
        # Validate directory
        report = self.validator.validate_json_files(self.temp_dir)
        
        self.assertEqual(report.total_files, 2)
        self.assertGreater(len(report.issues), 0)
    
    def test_check_data_integrity(self):
        """Test data integrity checks."""
        # Check valid data
        report = self.validator.check_data_integrity(self.valid_data)
        self.assertTrue(report.is_valid)
        
        # Check invalid data (new validator instance for clean report)
        invalid_validator = ResultValidator()
        report = invalid_validator.check_data_integrity(self.invalid_data)
        self.assertFalse(report.is_valid)
        
        # Check for specific issues
        issue_messages = [issue.message for issue in report.issues]
        self.assertTrue(any("must be in [0, 1]" in msg for msg in issue_messages))
        self.assertTrue(any("must be positive" in msg for msg in issue_messages))
        self.assertTrue(any("NaN value" in msg for msg in issue_messages))
        self.assertTrue(any("Infinite value" in msg for msg in issue_messages))
    
    def test_far_frr_consistency_check(self):
        """Test FAR + FRR consistency validation."""
        data_with_inconsistency = [
            {
                "experiment_id": "exp_test",
                "far": 0.6,
                "frr": 0.5,  # FAR + FRR > 1
                "accuracy": 0.95
            }
        ]
        
        validator = ResultValidator()
        report = validator.check_data_integrity(data_with_inconsistency)
        
        # Should have warning about FAR + FRR > 1
        consistency_issues = [i for i in report.issues if "FAR + FRR" in i.message]
        self.assertGreater(len(consistency_issues), 0)
    
    def test_compare_claimed_vs_actual(self):
        """Test comparison of claimed vs actual metrics."""
        # Define claimed metrics
        claimed = ClaimedMetrics(far=0.01, frr=0.01, accuracy=0.99, avg_queries=10.0)
        
        # Compare with valid data
        report = self.validator.compare_claimed_vs_actual(claimed, self.valid_data)
        
        # Should pass or have minor deviations
        self.assertTrue(report.is_valid)
        
        # Test with large deviations
        deviated_data = [
            {
                "far": 0.05,  # 5x higher than claimed
                "frr": 0.05,  # 5x higher than claimed
                "accuracy": 0.90,  # Lower than claimed
                "queries": 50  # 5x higher than claimed
            }
        ]
        
        deviated_validator = ResultValidator(strict_mode=False)
        report = deviated_validator.compare_claimed_vs_actual(claimed, deviated_data)
        
        # Should have issues about deviations
        self.assertGreater(len(report.issues), 0)
        deviation_issues = [i for i in report.issues if "deviation" in i.message.lower()]
        self.assertGreater(len(deviation_issues), 0)
    
    def test_strict_mode(self):
        """Test strict mode validation."""
        # Create validator in strict mode
        strict_validator = ResultValidator(strict_mode=True)
        
        # Data with minor deviation
        slightly_off_data = [
            {
                "far": 0.015,  # 50% higher than claimed
                "frr": 0.015,
                "accuracy": 0.985,
                "queries": 12
            }
        ]
        
        claimed = ClaimedMetrics(far=0.01, frr=0.01, accuracy=0.99, avg_queries=10.0)
        report = strict_validator.compare_claimed_vs_actual(claimed, slightly_off_data)
        
        # Should fail in strict mode
        self.assertFalse(report.is_valid)
        self.assertEqual(report.status, ValidationStatus.FAILED)
    
    def test_reconciliation_suggestions(self):
        """Test reconciliation suggestion generation."""
        # Data with systematic issues
        problematic_data = [
            {"far": 0.02, "frr": 0.02, "accuracy": 0.96, "queries": 15},
            {"far": 0.025, "frr": 0.025, "accuracy": 0.95, "queries": 20},
            {"far": 0.03, "frr": 0.03, "accuracy": 0.94, "queries": 25}
        ]
        
        claimed = ClaimedMetrics(far=0.01, frr=0.01, accuracy=0.99, avg_queries=10.0)
        
        validator = ResultValidator()
        report = validator.compare_claimed_vs_actual(claimed, problematic_data)
        
        # Should generate reconciliation suggestions
        suggestions = validator.reconcile_discrepancies()
        self.assertGreater(len(suggestions), 0)
        self.assertEqual(report.reconciliation_suggestions, suggestions)
    
    def test_validation_report_generation(self):
        """Test generation of validation report."""
        # Run validation
        file_path = os.path.join(self.temp_dir, "test_results.json")
        with open(file_path, 'w') as f:
            json.dump(self.valid_data, f)
        
        report = self.validator.validate_single_file(file_path)
        
        # Generate text report
        report_text = self.validator.generate_validation_report()
        
        self.assertIn("POT EXPERIMENT VALIDATION REPORT", report_text)
        self.assertIn("Status:", report_text)
        self.assertIn("SUMMARY", report_text)
        
        # Save report
        report_path = os.path.join(self.temp_dir, "validation_report.txt")
        saved_text = self.validator.generate_validation_report(report_path)
        
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, 'r') as f:
            saved_content = f.read()
        self.assertEqual(saved_content, saved_text)
    
    def test_validation_report_json_export(self):
        """Test JSON export of validation report."""
        # Run validation
        report = self.validator.check_data_integrity(self.valid_data)
        
        # Convert to JSON
        json_str = report.to_json()
        json_data = json.loads(json_str)
        
        self.assertIn("status", json_data)
        self.assertIn("issues", json_data)
        self.assertIn("confidence_score", json_data)
        
        # Save to file
        json_path = os.path.join(self.temp_dir, "report.json")
        report.save(json_path)
        
        self.assertTrue(os.path.exists(json_path))
    
    def test_missing_ground_truth_warning(self):
        """Test warning for missing ground truth labels."""
        data_without_ground_truth = [
            {
                "experiment_id": "exp_test",
                "verified": True,
                "far": 0.01,
                "frr": 0.01
                # Missing ground_truth
            }
        ]
        
        validator = ResultValidator()
        report = validator.check_data_integrity(data_without_ground_truth)
        
        # Should have warning about missing ground truth
        ground_truth_issues = [i for i in report.issues 
                              if "ground truth" in i.message.lower()]
        self.assertGreater(len(ground_truth_issues), 0)
    
    def test_validate_experiment_results_function(self):
        """Test the convenience validation function."""
        # Create test directory with files
        os.makedirs(os.path.join(self.temp_dir, "experiment_results"))
        file_path = os.path.join(self.temp_dir, "experiment_results", "results.json")
        
        with open(file_path, 'w') as f:
            json.dump(self.valid_data, f)
        
        # Run validation
        claimed = {"far": 0.01, "frr": 0.01, "accuracy": 0.99}
        is_valid, report = validate_experiment_results(
            os.path.join(self.temp_dir, "experiment_results"),
            claimed_metrics=claimed,
            strict_mode=False
        )
        
        self.assertTrue(is_valid)
        self.assertIsInstance(report, ValidationReport)
    
    def test_duplicate_id_detection(self):
        """Test detection of duplicate experiment IDs."""
        data_with_duplicates = [
            {"experiment_id": "exp_001", "far": 0.01},
            {"experiment_id": "exp_001", "far": 0.02},  # Duplicate ID
            {"experiment_id": "exp_002", "far": 0.01}
        ]
        
        # Save to file
        file_path = os.path.join(self.temp_dir, "duplicates.json")
        with open(file_path, 'w') as f:
            json.dump(data_with_duplicates, f)
        
        validator = ResultValidator()
        report = validator.validate_single_file(file_path)
        
        # Should have warning about duplicate IDs
        duplicate_issues = [i for i in report.issues if "Duplicate" in i.message]
        self.assertGreater(len(duplicate_issues), 0)
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation based on issues."""
        # No issues - should have high confidence
        validator1 = ResultValidator()
        report1 = validator1.check_data_integrity(self.valid_data)
        self.assertGreater(report1.confidence_score, 0.9)
        
        # Many issues - should have low confidence
        validator2 = ResultValidator()
        report2 = validator2.check_data_integrity(self.invalid_data)
        self.assertLess(report2.confidence_score, 0.5)


if __name__ == '__main__':
    unittest.main()