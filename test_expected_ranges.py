#!/usr/bin/env python3
"""
Comprehensive test suite for Expected Ranges Verification System

Tests the ExpectedRanges, ValidationReport, RangeCalibrator, and integration
with the ProofOfTraining system for behavioral validation.
"""

import sys
import os
import time
import numpy as np
import unittest
from unittest.mock import Mock, patch
from typing import Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pot.security.proof_of_training import (
    ExpectedRanges, ValidationReport, RangeCalibrator,
    ProofOfTraining, VerificationResult, ModelType
)


class MockModel:
    """Mock model for testing"""
    
    def __init__(self, output_value=None, latency=0.001):
        self.output_value = output_value or np.random.randn(10)
        self.latency = latency
        self.call_count = 0
    
    def forward(self, x):
        """Mock forward pass"""
        self.call_count += 1
        time.sleep(self.latency)  # Simulate processing time
        
        # Return deterministic output based on input
        if isinstance(x, np.ndarray):
            return self.output_value + np.sum(x) * 0.01
        else:
            return self.output_value + hash(str(x)) % 100 * 0.001
    
    def state_dict(self):
        return {'layer1': 'weights'}


class TestExpectedRanges(unittest.TestCase):
    """Test ExpectedRanges validation logic"""
    
    def setUp(self):
        self.ranges = ExpectedRanges(
            accuracy_range=(0.7, 0.9),
            latency_range=(1.0, 10.0),
            fingerprint_similarity=(0.85, 0.99),
            jacobian_norm_range=(0.5, 2.0)
        )
        
        # Create mock verification result
        self.mock_result = VerificationResult(
            verified=True,
            confidence=0.85,
            verification_type="fuzzy",
            model_id="test_model",
            challenges_passed=5,
            challenges_total=5,
            fuzzy_similarity=0.85,
            statistical_score=None,
            provenance_verified=None,
            details={},
            timestamp=None,
            duration_seconds=1.0
        )
    
    def test_ranges_validation_within_bounds(self):
        """Test validation when all metrics are within bounds"""
        self.mock_result.accuracy = 0.8
        self.mock_result.latency_ms = 5.0
        self.mock_result.fingerprint_similarity = 0.92
        self.mock_result.jacobian_norm = 1.2
        
        report = self.ranges.validate(self.mock_result)
        
        self.assertTrue(report.passed)
        self.assertEqual(len(report.violations), 0)
        self.assertGreater(report.confidence, 0.8)
        self.assertEqual(len(report.range_scores), 4)
    
    def test_ranges_validation_outside_bounds(self):
        """Test validation when metrics are outside bounds"""
        self.mock_result.accuracy = 0.5  # Below minimum
        self.mock_result.latency_ms = 15.0  # Above maximum
        self.mock_result.fingerprint_similarity = 0.75  # Below minimum
        self.mock_result.jacobian_norm = 3.0  # Above maximum
        
        report = self.ranges.validate(self.mock_result)
        
        self.assertFalse(report.passed)
        self.assertEqual(len(report.violations), 4)
        self.assertLess(report.confidence, 0.5)
        
        # Check violation messages
        violations_text = " ".join(report.violations)
        self.assertIn("Accuracy 0.500", violations_text)
        self.assertIn("Latency 15.0ms", violations_text)
        self.assertIn("Fingerprint similarity 0.750", violations_text)
        self.assertIn("Jacobian norm 3.000000", violations_text)
    
    def test_ranges_validation_partial_metrics(self):
        """Test validation when only some metrics are available"""
        self.mock_result.accuracy = 0.8
        self.mock_result.latency_ms = 5.0
        # fingerprint_similarity and jacobian_norm are None
        
        report = self.ranges.validate(self.mock_result)
        
        self.assertTrue(report.passed)
        self.assertEqual(len(report.range_scores), 2)
        self.assertIn("accuracy", report.range_scores)
        self.assertIn("latency", report.range_scores)
    
    def test_ranges_validation_tolerance_factor(self):
        """Test that tolerance factor allows slight excursions"""
        # Set tolerance factor to 1.2 (20% tolerance)
        self.ranges.tolerance_factor = 1.2
        
        # Test value slightly outside original range but within tolerance
        self.mock_result.accuracy = 0.68  # Below 0.7 but within tolerance
        self.mock_result.latency_ms = 5.0  # Within original range
        # Don't set other metrics to avoid violations
        
        report = self.ranges.validate(self.mock_result)
        
        self.assertTrue(report.passed)
        self.assertEqual(len(report.violations), 0)
    
    def test_range_score_computation(self):
        """Test range score computation logic"""
        # Test center of range (should give high score)
        center_score = self.ranges._compute_range_score(0.8, (0.7, 0.9), "test")
        self.assertAlmostEqual(center_score, 1.0, places=2)
        
        # Test edge of range (should give lower score)
        edge_score = self.ranges._compute_range_score(0.7, (0.7, 0.9), "test")
        self.assertGreater(edge_score, 0.0)
        self.assertLess(edge_score, 1.0)
        
        # Test outside range (should give penalty)
        outside_score = self.ranges._compute_range_score(0.6, (0.7, 0.9), "test")
        self.assertEqual(outside_score, 0.0)
    
    def test_statistical_significance(self):
        """Test statistical significance computation"""
        # Create range scores that should be significantly different from 0.5
        range_scores = {"metric1": 0.9, "metric2": 0.85, "metric3": 0.95}
        
        # Test with scipy not available
        p_value_no_scipy = self.ranges._compute_statistical_significance(range_scores)
        
        # Should return None when scipy is not available
        if p_value_no_scipy is None:
            self.assertIsNone(p_value_no_scipy)
        else:
            # If scipy is available, should return a reasonable p-value
            self.assertIsInstance(p_value_no_scipy, float)
            self.assertGreaterEqual(p_value_no_scipy, 0.0)
            self.assertLessEqual(p_value_no_scipy, 1.0)


class TestRangeCalibrator(unittest.TestCase):
    """Test RangeCalibrator functionality"""
    
    def setUp(self):
        self.calibrator = RangeCalibrator(
            confidence_level=0.95,
            percentile_margin=0.1
        )
        self.mock_model = MockModel()
        
        # Create simple test suite
        self.test_suite = [
            np.random.randn(5) for _ in range(10)
        ]
    
    def test_calibrate_basic(self):
        """Test basic calibration functionality"""
        ranges = self.calibrator.calibrate(
            reference_model=self.mock_model,
            test_suite=self.test_suite,
            model_type=ModelType.GENERIC,
            num_runs=3
        )
        
        self.assertIsInstance(ranges, ExpectedRanges)
        
        # Check that ranges are reasonable
        self.assertGreater(ranges.accuracy_range[1], ranges.accuracy_range[0])
        self.assertGreater(ranges.latency_range[1], ranges.latency_range[0])
        self.assertGreater(ranges.fingerprint_similarity[1], ranges.fingerprint_similarity[0])
        self.assertGreater(ranges.jacobian_norm_range[1], ranges.jacobian_norm_range[0])
        
        # Check that latency ranges are reasonable (should be > 0)
        self.assertGreater(ranges.latency_range[0], 0)
        self.assertLess(ranges.latency_range[1], 1000)  # Should be less than 1 second
    
    def test_calibrate_vision_model(self):
        """Test calibration with vision model type"""
        ranges = self.calibrator.calibrate(
            reference_model=self.mock_model,
            test_suite=self.test_suite,
            model_type=ModelType.VISION,
            num_runs=2
        )
        
        self.assertIsInstance(ranges, ExpectedRanges)
        # Vision models should have accuracy measurements
        self.assertGreaterEqual(ranges.accuracy_range[0], 0.0)
        self.assertLessEqual(ranges.accuracy_range[1], 1.0)
    
    def test_calibrate_language_model(self):
        """Test calibration with language model type"""
        # Language models expect string inputs
        text_suite = ["test input", "another test", "final test"]
        
        ranges = self.calibrator.calibrate(
            reference_model=self.mock_model,
            test_suite=text_suite,
            model_type=ModelType.LANGUAGE,
            num_runs=2
        )
        
        self.assertIsInstance(ranges, ExpectedRanges)
    
    def test_measure_accuracy(self):
        """Test accuracy measurement"""
        test_input = np.random.randn(5)
        
        accuracy = self.calibrator._measure_accuracy(
            self.mock_model, test_input, ModelType.GENERIC
        )
        
        self.assertIsNotNone(accuracy)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_measure_latency(self):
        """Test latency measurement"""
        # Set a known latency
        self.mock_model.latency = 0.005  # 5ms
        test_input = np.random.randn(5)
        
        latency = self.calibrator._measure_latency(self.mock_model, test_input)
        
        self.assertIsNotNone(latency)
        self.assertGreater(latency, 0.0)
        self.assertLess(latency, 100.0)  # Should be reasonable
    
    def test_measure_fingerprint_similarity(self):
        """Test fingerprint similarity measurement"""
        test_input = np.random.randn(5)
        
        similarity = self.calibrator._measure_fingerprint_similarity(
            self.mock_model, test_input
        )
        
        self.assertIsNotNone(similarity)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_measure_jacobian_norm(self):
        """Test Jacobian norm measurement"""
        test_input = np.random.randn(5)
        
        jacobian_norm = self.calibrator._measure_jacobian_norm(
            self.mock_model, test_input
        )
        
        self.assertIsNotNone(jacobian_norm)
        self.assertGreater(jacobian_norm, 0.0)
    
    def test_compute_range_from_samples(self):
        """Test range computation from samples"""
        # Create samples with known distribution
        samples = [0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.75, 0.65, 0.55]
        
        range_bounds = self.calibrator._compute_range_from_samples(samples, "test")
        
        # Range should be reasonable but not necessarily extend beyond data
        # (depends on percentile margin)
        self.assertLessEqual(range_bounds[0], np.percentile(samples, 20))
        self.assertGreaterEqual(range_bounds[1], np.percentile(samples, 80))
        self.assertGreater(range_bounds[1], range_bounds[0])
    
    def test_compute_range_empty_samples(self):
        """Test range computation with empty samples"""
        range_bounds = self.calibrator._compute_range_from_samples([], "test")
        
        # Should return default range
        self.assertEqual(range_bounds, (0.0, 1.0))


class TestProofOfTrainingIntegration(unittest.TestCase):
    """Test integration of expected ranges with ProofOfTraining"""
    
    def setUp(self):
        config = {
            'verification_type': 'fuzzy',
            'model_type': 'generic',
            'security_level': 'medium'
        }
        
        self.pot_system = ProofOfTraining(config)
        self.mock_model = MockModel()
    
    def test_calibrate_expected_ranges(self):
        """Test expected ranges calibration integration"""
        # Register model first
        model_id = self.pot_system.register_model(
            self.mock_model,
            architecture="test_architecture",
            parameter_count=1000
        )
        
        # Calibrate expected ranges
        ranges = self.pot_system.calibrate_expected_ranges(
            self.mock_model, model_id, num_calibration_runs=2
        )
        
        self.assertIsInstance(ranges, ExpectedRanges)
        self.assertIn(model_id, self.pot_system.expected_ranges)
        self.assertEqual(self.pot_system.expected_ranges[model_id], ranges)
    
    def test_set_get_expected_ranges(self):
        """Test manual setting and getting of expected ranges"""
        # Register model first
        model_id = self.pot_system.register_model(
            self.mock_model,
            architecture="test_architecture",
            parameter_count=1000
        )
        
        # Set manual ranges
        manual_ranges = ExpectedRanges(
            accuracy_range=(0.5, 0.8),
            latency_range=(1.0, 20.0),
            fingerprint_similarity=(0.7, 0.95),
            jacobian_norm_range=(0.1, 5.0)
        )
        
        self.pot_system.set_expected_ranges(model_id, manual_ranges)
        
        # Get ranges back
        retrieved_ranges = self.pot_system.get_expected_ranges(model_id)
        
        self.assertEqual(retrieved_ranges, manual_ranges)
    
    def test_verification_with_expected_ranges(self):
        """Test verification with expected ranges validation"""
        # Register model
        model_id = self.pot_system.register_model(
            self.mock_model,
            architecture="test_architecture",
            parameter_count=1000
        )
        
        # Set lenient expected ranges (should pass)
        lenient_ranges = ExpectedRanges(
            accuracy_range=(0.0, 1.0),
            latency_range=(0.1, 1000.0),
            fingerprint_similarity=(0.0, 1.0),
            jacobian_norm_range=(0.001, 100.0)
        )
        self.pot_system.set_expected_ranges(model_id, lenient_ranges)
        
        # Perform comprehensive verification
        result = self.pot_system.perform_verification(
            self.mock_model, model_id, 'comprehensive'
        )
        
        # Check that range validation was performed
        self.assertIsNotNone(result.range_validation)
        self.assertTrue(result.range_validation.passed)
        self.assertEqual(len(result.range_validation.violations), 0)
        
        # Check that measurements were collected
        self.assertIsNotNone(result.accuracy)
        self.assertIsNotNone(result.latency_ms)
        self.assertIsNotNone(result.fingerprint_similarity)
        self.assertIsNotNone(result.jacobian_norm)
    
    def test_verification_range_violations(self):
        """Test verification with range violations"""
        # Register model
        model_id = self.pot_system.register_model(
            self.mock_model,
            architecture="test_architecture",
            parameter_count=1000
        )
        
        # Set very strict ranges (should fail)
        strict_ranges = ExpectedRanges(
            accuracy_range=(0.99, 1.0),  # Very high accuracy required
            latency_range=(0.001, 0.002),  # Very low latency required
            fingerprint_similarity=(0.999, 1.0),  # Perfect similarity required
            jacobian_norm_range=(0.0001, 0.0002)  # Very small norm required
        )
        self.pot_system.set_expected_ranges(model_id, strict_ranges)
        
        # Perform comprehensive verification
        result = self.pot_system.perform_verification(
            self.mock_model, model_id, 'comprehensive'
        )
        
        # Check that range validation failed
        self.assertIsNotNone(result.range_validation)
        self.assertFalse(result.range_validation.passed)
        self.assertGreater(len(result.range_validation.violations), 0)
        
        # Overall verification should be affected
        # (depending on how strict the integration is)
    
    def test_statistics_with_expected_ranges(self):
        """Test that statistics include expected ranges information"""
        # Register model and set ranges
        model_id = self.pot_system.register_model(
            self.mock_model,
            architecture="test_architecture",
            parameter_count=1000
        )
        
        ranges = ExpectedRanges(
            accuracy_range=(0.5, 0.9),
            latency_range=(1.0, 50.0),
            fingerprint_similarity=(0.8, 0.99),
            jacobian_norm_range=(0.1, 10.0)
        )
        self.pot_system.set_expected_ranges(model_id, ranges)
        
        # Get statistics
        stats = self.pot_system.get_statistics()
        
        self.assertEqual(stats['models_with_expected_ranges'], 1)
        self.assertTrue(stats['components']['expected_ranges'])
        self.assertTrue(stats['components']['range_calibrator'])


def run_all_tests():
    """Run all expected ranges tests"""
    print("=" * 70)
    print("COMPREHENSIVE EXPECTED RANGES TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestExpectedRanges,
        TestRangeCalibrator,
        TestProofOfTrainingIntegration
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("\nExpected ranges verification system ready for production!")
    else:
        print(f"❌ {len(result.failures + result.errors)} tests failed")
    
    return success


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)