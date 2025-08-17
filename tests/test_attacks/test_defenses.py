"""
Comprehensive tests for defense mechanisms.
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import tempfile
import json
import os
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Import defense components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pot.core.defenses import (
    DefenseConfig,
    AdaptiveVerifier,
    InputFilter,
    RandomizedDefense,
    IntegratedDefenseSystem,
    MockBaseVerifier
)
from pot.core.attack_suites import AttackConfig, StandardAttackSuite


class TestModel(nn.Module):
    """Simple test model for defense testing."""
    
    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class TestDefenseEffectiveness(unittest.TestCase):
    """Test defense effectiveness against attack suite."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        self.device = 'cpu'
        
        # Create test data
        self.clean_data = torch.randn(100, 1, 28, 28)
        self.labels = torch.randint(0, 10, (100,))
        
        # Create adversarial data
        self.adv_data = self.clean_data + torch.randn_like(self.clean_data) * 0.1
        self.adv_data = torch.clamp(self.adv_data, -1, 1)
        
        # Create data loaders
        clean_dataset = TensorDataset(self.clean_data, self.labels)
        self.clean_loader = DataLoader(clean_dataset, batch_size=10)
        
        adv_dataset = TensorDataset(self.adv_data, self.labels)
        self.adv_loader = DataLoader(adv_dataset, batch_size=10)
        
        # Initialize defense config
        self.defense_config = DefenseConfig(
            adaptive_threshold=0.05,
            input_filter_strength=0.5,
            randomization_scale=0.1,
            ensemble_size=3
        )
    
    def test_defense_against_distillation(self):
        """Test defense effectiveness against distillation attacks."""
        # Create defense system
        defense = IntegratedDefenseSystem(self.defense_config)
        
        # Simulate distillation attack
        attack_config = AttackConfig(
            name="distillation_test",
            attack_type="distillation",
            parameters={"temperature": 5.0, "epochs": 10}
        )
        
        # Test defense detection
        with patch.object(defense.adaptive_verifier, 'verify') as mock_verify:
            mock_verify.return_value = {
                'verified': False,
                'confidence': 0.3,
                'attack_detected': True,
                'attack_type': 'distillation'
            }
            
            result = defense.comprehensive_defense(
                self.clean_data[:10],
                self.model,
                threat_level=0.5
            )
            
            self.assertIn('adaptive_result', result)
            self.assertTrue(result['adaptive_result']['attack_detected'])
    
    def test_defense_against_compression(self):
        """Test defense against compression attacks."""
        defense = IntegratedDefenseSystem(self.defense_config)
        
        # Simulate compressed model behavior
        compressed_outputs = torch.randn(10, 10) * 0.5  # Lower variance
        
        with patch.object(self.model, 'forward') as mock_forward:
            mock_forward.return_value = compressed_outputs
            
            result = defense.comprehensive_defense(
                self.clean_data[:10],
                self.model,
                threat_level=0.6
            )
            
            # Defense should detect anomalous outputs
            self.assertIsNotNone(result)
            self.assertIn('threat_assessment', result)
    
    def test_defense_against_adversarial_examples(self):
        """Test defense against adversarial examples."""
        defense = IntegratedDefenseSystem(self.defense_config)
        
        # Test with adversarial data
        result = defense.comprehensive_defense(
            self.adv_data[:10],
            self.model,
            threat_level=0.7
        )
        
        # Check if filtering was applied
        self.assertIn('filter_result', result)
        if result['filter_result']['anomalies_detected']:
            self.assertIsNotNone(result['filter_result']['filtered_input'])
            
            # Verify filtered input is different from original
            filtered = result['filter_result']['filtered_input']
            self.assertFalse(torch.allclose(filtered, self.adv_data[:10]))
    
    def test_ensemble_defense_voting(self):
        """Test ensemble defense voting mechanism."""
        defense = IntegratedDefenseSystem(self.defense_config)
        
        # Create multiple verification results
        verification_results = [
            {'verified': True, 'confidence': 0.8},
            {'verified': False, 'confidence': 0.6},
            {'verified': True, 'confidence': 0.7}
        ]
        
        # Test voting
        consensus = defense._ensemble_vote(verification_results)
        
        self.assertTrue(consensus['verified'])  # Majority vote
        self.assertAlmostEqual(consensus['confidence'], 0.7, places=1)
    
    def test_defense_performance_metrics(self):
        """Test defense performance measurement."""
        defense = IntegratedDefenseSystem(self.defense_config)
        
        # Measure clean data performance
        clean_result = defense.comprehensive_defense(
            self.clean_data[:10],
            self.model,
            threat_level=0.3
        )
        
        # Measure adversarial data performance
        adv_result = defense.comprehensive_defense(
            self.adv_data[:10],
            self.model,
            threat_level=0.7
        )
        
        # Compare results
        self.assertIsNotNone(clean_result)
        self.assertIsNotNone(adv_result)
        
        # Defense should be more suspicious of adversarial data
        if 'threat_assessment' in clean_result and 'threat_assessment' in adv_result:
            self.assertLess(
                clean_result['threat_assessment']['overall_threat'],
                adv_result['threat_assessment']['overall_threat']
            )


class TestAdaptiveThreshold(unittest.TestCase):
    """Test adaptive threshold mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_verifier = MockBaseVerifier()
        self.adaptive_verifier = AdaptiveVerifier(
            base_verifier=self.base_verifier,
            initial_threshold=0.05,
            adaptation_rate=0.1
        )
        
        self.model = TestModel()
        self.test_input = torch.randn(10, 1, 28, 28)
    
    def test_adaptive_threshold_update(self):
        """Test dynamic threshold adaptation."""
        initial_threshold = self.adaptive_verifier.threshold
        
        # Simulate successful attacks
        for _ in range(5):
            self.adaptive_verifier.update_from_attack({
                'success': True,
                'confidence': 0.8,
                'attack_type': 'distillation'
            })
        
        # Threshold should become more strict
        self.assertLess(self.adaptive_verifier.threshold, initial_threshold)
        
        # Simulate failed attacks
        for _ in range(10):
            self.adaptive_verifier.update_from_attack({
                'success': False,
                'confidence': 0.2,
                'attack_type': 'compression'
            })
        
        # Threshold might relax slightly
        current_threshold = self.adaptive_verifier.threshold
        self.assertLess(current_threshold, initial_threshold)  # Still stricter than initial
    
    def test_attack_pattern_learning(self):
        """Test learning from attack patterns."""
        # Feed various attack patterns
        attacks = [
            {'type': 'distillation', 'params': {'temp': 5}, 'success': True},
            {'type': 'distillation', 'params': {'temp': 5}, 'success': True},
            {'type': 'compression', 'params': {'rate': 0.5}, 'success': False},
            {'type': 'distillation', 'params': {'temp': 5}, 'success': True}
        ]
        
        for attack in attacks:
            self.adaptive_verifier.learn_attack_pattern(attack)
        
        # Check if patterns were learned
        patterns = self.adaptive_verifier.get_attack_patterns()
        self.assertIn('distillation', patterns)
        self.assertGreater(patterns['distillation']['success_rate'], 0.9)
    
    def test_adaptive_verification_with_history(self):
        """Test verification using attack history."""
        # Build attack history
        for i in range(10):
            self.adaptive_verifier.attack_history.append({
                'timestamp': time.time() - (10 - i),
                'type': 'distillation' if i % 2 == 0 else 'compression',
                'success': i % 3 == 0
            })
        
        # Perform verification
        result = self.adaptive_verifier.verify(
            self.model,
            self.test_input,
            reference_output=torch.randn(10, 10)
        )
        
        self.assertIn('verified', result)
        self.assertIn('confidence', result)
        self.assertIn('adapted_threshold', result)
        self.assertIn('attack_risk', result)
    
    def test_threshold_bounds(self):
        """Test threshold stays within bounds."""
        # Try to push threshold to extremes
        for _ in range(100):
            self.adaptive_verifier.update_from_attack({
                'success': True,
                'confidence': 0.99
            })
        
        # Should not go below minimum
        self.assertGreaterEqual(self.adaptive_verifier.threshold, 0.001)
        
        # Reset and test upper bound
        self.adaptive_verifier.threshold = 0.05
        for _ in range(100):
            self.adaptive_verifier.update_from_attack({
                'success': False,
                'confidence': 0.01
            })
        
        # Should not exceed maximum
        self.assertLessEqual(self.adaptive_verifier.threshold, 0.2)


class TestInputSanitization(unittest.TestCase):
    """Test input filtering and sanitization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_filter = InputFilter(
            filter_strength=0.5,
            detection_threshold=0.1
        )
        
        # Create test inputs
        self.clean_input = torch.randn(10, 3, 32, 32)
        
        # Create adversarial input with noise
        self.adv_input = self.clean_input.clone()
        noise = torch.randn_like(self.clean_input) * 0.2
        self.adv_input += noise
        
        # Create input with outliers
        self.outlier_input = self.clean_input.clone()
        self.outlier_input[0, 0, :5, :5] = 10.0  # Add outlier patch
    
    def test_adversarial_detection(self):
        """Test adversarial input detection."""
        # Test clean input
        clean_detected = self.input_filter.detect_adversarial(
            self.clean_input,
            reference_distribution={'mean': 0, 'std': 1}
        )
        self.assertFalse(clean_detected['is_adversarial'])
        
        # Test adversarial input
        adv_detected = self.input_filter.detect_adversarial(
            self.adv_input,
            reference_distribution={'mean': 0, 'std': 1}
        )
        # May or may not detect depending on noise level
        self.assertIn('is_adversarial', adv_detected)
        self.assertIn('confidence', adv_detected)
    
    def test_input_sanitization_filters(self):
        """Test various sanitization filters."""
        # Test Gaussian blur
        blurred = self.input_filter.apply_gaussian_blur(
            self.adv_input,
            kernel_size=3
        )
        self.assertEqual(blurred.shape, self.adv_input.shape)
        
        # Test median filter
        filtered = self.input_filter.apply_median_filter(
            self.adv_input,
            kernel_size=3
        )
        self.assertEqual(filtered.shape, self.adv_input.shape)
        
        # Test quantization
        quantized = self.input_filter.apply_quantization(
            self.adv_input,
            levels=32
        )
        self.assertEqual(quantized.shape, self.adv_input.shape)
        unique_values = torch.unique(quantized).numel()
        self.assertLessEqual(unique_values, 32 * 3)  # 32 levels per channel
    
    def test_outlier_removal(self):
        """Test outlier detection and removal."""
        # Detect outliers
        outliers = self.input_filter.detect_outliers(
            self.outlier_input,
            threshold=3.0
        )
        
        self.assertGreater(outliers['num_outliers'], 0)
        self.assertTrue(outliers['outlier_mask'][0, 0, 0, 0])
        
        # Remove outliers
        cleaned = self.input_filter.remove_outliers(
            self.outlier_input,
            outliers['outlier_mask']
        )
        
        # Check outliers were replaced
        self.assertLess(cleaned[0, 0, 0, 0].item(), 10.0)
    
    def test_combined_sanitization(self):
        """Test combined sanitization pipeline."""
        result = self.input_filter.sanitize(
            self.adv_input,
            apply_blur=True,
            apply_median=True,
            apply_quantization=True,
            remove_outliers=True
        )
        
        self.assertIn('sanitized_input', result)
        self.assertIn('filters_applied', result)
        self.assertIn('anomaly_score', result)
        
        sanitized = result['sanitized_input']
        self.assertEqual(sanitized.shape, self.adv_input.shape)
        
        # Should be different from original
        self.assertFalse(torch.allclose(sanitized, self.adv_input))
    
    def test_adaptive_filtering(self):
        """Test adaptive filter strength."""
        # Low threat - mild filtering
        mild_result = self.input_filter.sanitize(
            self.adv_input,
            threat_level=0.2
        )
        
        # High threat - strong filtering
        strong_result = self.input_filter.sanitize(
            self.adv_input,
            threat_level=0.9
        )
        
        # Strong filtering should modify input more
        mild_diff = (mild_result['sanitized_input'] - self.adv_input).abs().mean()
        strong_diff = (strong_result['sanitized_input'] - self.adv_input).abs().mean()
        
        self.assertGreater(strong_diff, mild_diff)


class TestRandomizedDefense(unittest.TestCase):
    """Test randomized defense mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        self.randomized_defense = RandomizedDefense(
            randomization_scale=0.1,
            num_samples=10,
            aggregation_method='mean'
        )
        
        self.test_input = torch.randn(5, 1, 28, 28)
    
    def test_random_smoothing(self):
        """Test randomized smoothing defense."""
        smoothed_result = self.randomized_defense.random_smoothing(
            self.model,
            self.test_input,
            num_samples=20,
            noise_scale=0.1
        )
        
        self.assertIn('smoothed_output', smoothed_result)
        self.assertIn('variance', smoothed_result)
        self.assertIn('confidence', smoothed_result)
        
        # Check output shape
        output = smoothed_result['smoothed_output']
        self.assertEqual(output.shape[0], self.test_input.shape[0])
        self.assertEqual(output.shape[1], 10)  # num_classes
    
    def test_stochastic_verification(self):
        """Test stochastic verification process."""
        reference_output = torch.randn(5, 10)
        
        result = self.randomized_defense.stochastic_verify(
            self.model,
            self.test_input,
            reference_output,
            num_trials=15
        )
        
        self.assertIn('verified', result)
        self.assertIn('confidence', result)
        self.assertIn('agreement_rate', result)
        self.assertIn('trial_results', result)
        
        self.assertEqual(len(result['trial_results']), 15)
    
    def test_noise_injection_types(self):
        """Test different noise injection strategies."""
        # Gaussian noise
        gaussian_noisy = self.randomized_defense.add_noise(
            self.test_input,
            noise_type='gaussian',
            scale=0.1
        )
        self.assertEqual(gaussian_noisy.shape, self.test_input.shape)
        
        # Uniform noise
        uniform_noisy = self.randomized_defense.add_noise(
            self.test_input,
            noise_type='uniform',
            scale=0.1
        )
        self.assertEqual(uniform_noisy.shape, self.test_input.shape)
        
        # Laplace noise
        laplace_noisy = self.randomized_defense.add_noise(
            self.test_input,
            noise_type='laplace',
            scale=0.1
        )
        self.assertEqual(laplace_noisy.shape, self.test_input.shape)
        
        # Verify noise was added
        self.assertFalse(torch.allclose(gaussian_noisy, self.test_input))
        self.assertFalse(torch.allclose(uniform_noisy, self.test_input))
        self.assertFalse(torch.allclose(laplace_noisy, self.test_input))
    
    def test_aggregation_methods(self):
        """Test different output aggregation methods."""
        outputs = [torch.randn(5, 10) for _ in range(10)]
        
        # Mean aggregation
        mean_agg = self.randomized_defense.aggregate_outputs(
            outputs,
            method='mean'
        )
        self.assertEqual(mean_agg.shape, (5, 10))
        
        # Median aggregation
        median_agg = self.randomized_defense.aggregate_outputs(
            outputs,
            method='median'
        )
        self.assertEqual(median_agg.shape, (5, 10))
        
        # Voting aggregation
        vote_agg = self.randomized_defense.aggregate_outputs(
            outputs,
            method='vote'
        )
        self.assertEqual(vote_agg.shape, (5, 10))
    
    def test_certified_radius(self):
        """Test certified radius computation."""
        # Create predictions with known properties
        base_prediction = torch.tensor([0, 1, 0, 1, 0])
        smoothed_predictions = [base_prediction.clone() for _ in range(100)]
        
        # Add some variations
        for i in range(20):
            smoothed_predictions[i][0] = 1 - smoothed_predictions[i][0]
        
        radius = self.randomized_defense.compute_certified_radius(
            base_prediction,
            smoothed_predictions,
            noise_scale=0.1,
            confidence=0.95
        )
        
        self.assertIsInstance(radius, float)
        self.assertGreaterEqual(radius, 0.0)


class TestIntegratedDefense(unittest.TestCase):
    """Test integrated defense system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DefenseConfig(
            adaptive_threshold=0.05,
            input_filter_strength=0.5,
            randomization_scale=0.1,
            ensemble_size=3
        )
        
        self.defense_system = IntegratedDefenseSystem(self.config)
        self.model = TestModel()
        self.test_input = torch.randn(10, 1, 28, 28)
    
    def test_end_to_end_defense_pipeline(self):
        """Test full defense pipeline: detection → filtering → verification."""
        result = self.defense_system.comprehensive_defense(
            self.test_input,
            self.model,
            threat_level=0.5
        )
        
        # Check all components ran
        self.assertIn('filter_result', result)
        self.assertIn('adaptive_result', result)
        self.assertIn('randomized_result', result)
        self.assertIn('final_decision', result)
        self.assertIn('threat_assessment', result)
        
        # Verify final decision
        decision = result['final_decision']
        self.assertIn('verified', decision)
        self.assertIn('confidence', decision)
        self.assertIn('defense_actions', decision)
    
    def test_threat_level_adaptation(self):
        """Test defense adaptation to threat levels."""
        # Low threat
        low_threat_result = self.defense_system.comprehensive_defense(
            self.test_input,
            self.model,
            threat_level=0.1
        )
        
        # High threat
        high_threat_result = self.defense_system.comprehensive_defense(
            self.test_input,
            self.model,
            threat_level=0.9
        )
        
        # High threat should trigger more defenses
        low_actions = len(low_threat_result['final_decision']['defense_actions'])
        high_actions = len(high_threat_result['final_decision']['defense_actions'])
        
        self.assertGreaterEqual(high_actions, low_actions)
    
    def test_defense_coordination(self):
        """Test coordination between defense components."""
        # Mock individual defense results
        with patch.object(self.defense_system.input_filter, 'sanitize') as mock_filter:
            mock_filter.return_value = {
                'sanitized_input': self.test_input,
                'anomalies_detected': True,
                'anomaly_score': 0.7
            }
            
            with patch.object(self.defense_system.adaptive_verifier, 'verify') as mock_adaptive:
                mock_adaptive.return_value = {
                    'verified': False,
                    'confidence': 0.3,
                    'attack_detected': True
                }
                
                result = self.defense_system.comprehensive_defense(
                    self.test_input,
                    self.model,
                    threat_level=0.6
                )
                
                # System should coordinate responses
                self.assertTrue(result['threat_assessment']['high_risk'])
                self.assertIn('coordinated_response', result['final_decision'])
    
    def test_performance_monitoring(self):
        """Test defense performance monitoring."""
        # Run defense multiple times
        results = []
        for _ in range(5):
            result = self.defense_system.comprehensive_defense(
                self.test_input + torch.randn_like(self.test_input) * 0.01,
                self.model,
                threat_level=0.5
            )
            results.append(result)
        
        # Get performance metrics
        metrics = self.defense_system.get_performance_metrics()
        
        self.assertIn('total_defenses', metrics)
        self.assertIn('average_latency', metrics)
        self.assertIn('detection_rate', metrics)
        self.assertEqual(metrics['total_defenses'], 5)
    
    def test_defense_fallback_mechanisms(self):
        """Test fallback when primary defenses fail."""
        # Simulate primary defense failure
        with patch.object(self.defense_system.adaptive_verifier, 'verify') as mock_verify:
            mock_verify.side_effect = Exception("Verification failed")
            
            result = self.defense_system.comprehensive_defense(
                self.test_input,
                self.model,
                threat_level=0.5
            )
            
            # Should fall back to other defenses
            self.assertIsNotNone(result)
            self.assertIn('fallback_used', result)
            self.assertTrue(result['fallback_used'])


class TestDefenseMetrics(unittest.TestCase):
    """Test defense evaluation metrics."""
    
    def test_false_positive_rate(self):
        """Test false positive rate calculation."""
        defense = IntegratedDefenseSystem(DefenseConfig())
        
        # Simulate clean data being flagged
        clean_results = [
            {'clean': True, 'flagged': False},
            {'clean': True, 'flagged': False},
            {'clean': True, 'flagged': True},  # False positive
            {'clean': True, 'flagged': False},
            {'clean': True, 'flagged': True},  # False positive
        ]
        
        fpr = defense.calculate_false_positive_rate(clean_results)
        self.assertAlmostEqual(fpr, 0.4, places=2)
    
    def test_detection_accuracy(self):
        """Test attack detection accuracy."""
        defense = IntegratedDefenseSystem(DefenseConfig())
        
        # Simulate detection results
        detection_results = [
            {'is_attack': True, 'detected': True},   # True positive
            {'is_attack': True, 'detected': True},   # True positive
            {'is_attack': False, 'detected': False}, # True negative
            {'is_attack': True, 'detected': False},  # False negative
            {'is_attack': False, 'detected': True},  # False positive
        ]
        
        accuracy = defense.calculate_detection_accuracy(detection_results)
        self.assertAlmostEqual(accuracy, 0.6, places=2)
    
    def test_defense_overhead(self):
        """Test measurement of defense computational overhead."""
        defense = IntegratedDefenseSystem(DefenseConfig())
        model = TestModel()
        test_input = torch.randn(10, 1, 28, 28)
        
        # Measure baseline
        start = time.time()
        _ = model(test_input)
        baseline_time = time.time() - start
        
        # Measure with defense
        start = time.time()
        _ = defense.comprehensive_defense(test_input, model, threat_level=0.5)
        defense_time = time.time() - start
        
        overhead = (defense_time - baseline_time) / baseline_time
        
        # Defense should add overhead but not excessive
        self.assertGreater(overhead, 0)  # Some overhead expected
        self.assertLess(overhead, 100)   # Less than 100x slower


def run_tests():
    """Run all defense tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()