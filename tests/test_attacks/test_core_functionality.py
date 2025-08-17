"""
Core functionality tests for attack and defense systems.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pot.core.attack_suites import AttackConfig, StandardAttackSuite, AttackRunner
from pot.core.defenses import DefenseConfig, IntegratedDefenseSystem, MockBaseVerifier


class SimpleCNN(nn.Module):
    """Simple CNN for testing."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TestCoreAttackDefense(unittest.TestCase):
    """Test core attack-defense functionality."""
    
    def test_attack_config_creation(self):
        """Test attack configuration creation."""
        config = AttackConfig(
            name="test_attack",
            attack_type="distillation",
            parameters={"temperature": 5.0}
        )
        
        self.assertEqual(config.name, "test_attack")
        self.assertEqual(config.attack_type, "distillation")
        self.assertEqual(config.parameters["temperature"], 5.0)
    
    def test_standard_suite_initialization(self):
        """Test standard attack suite initialization."""
        suite = StandardAttackSuite()
        
        # Check suite has required methods
        self.assertTrue(hasattr(suite, 'get_distillation_configs'))
        self.assertTrue(hasattr(suite, 'get_compression_configs'))
        self.assertTrue(hasattr(suite, 'get_vision_specific_configs'))
        
        # Get configs
        dist_configs = suite.get_distillation_configs()
        self.assertGreater(len(dist_configs), 0)
        self.assertIsInstance(dist_configs[0], AttackConfig)
    
    def test_defense_config_creation(self):
        """Test defense configuration."""
        config = DefenseConfig(
            adaptive_threshold=0.05,
            input_filter_strength=0.5,
            randomization_scale=0.1
        )
        
        self.assertEqual(config.adaptive_threshold, 0.05)
        self.assertEqual(config.input_filter_strength, 0.5)
        self.assertEqual(config.randomization_scale, 0.1)
    
    def test_integrated_defense_initialization(self):
        """Test integrated defense system initialization."""
        config = DefenseConfig()
        defense = IntegratedDefenseSystem(config)
        
        # Check components are initialized
        self.assertIsNotNone(defense.adaptive_verifier)
        self.assertIsNotNone(defense.input_filter)
        self.assertIsNotNone(defense.randomized_defense)
        
        # Check methods exist
        self.assertTrue(hasattr(defense, 'comprehensive_defense'))
        self.assertTrue(hasattr(defense, 'get_performance_metrics'))
    
    def test_attack_runner_basic(self):
        """Test basic attack runner functionality."""
        runner = AttackRunner(device='cpu')
        
        # Check methods
        self.assertTrue(hasattr(runner, 'run_single_attack'))
        self.assertTrue(hasattr(runner, 'run_attack_suite'))
        self.assertTrue(hasattr(runner, 'calculate_metrics'))
        self.assertTrue(hasattr(runner, 'calculate_robustness_score'))
        
        # Test metric calculation
        results = [
            {'success': True, 'confidence': 0.8},
            {'success': False, 'confidence': 0.3},
            {'success': True, 'confidence': 0.6}
        ]
        
        metrics = runner.calculate_metrics(results)
        self.assertEqual(metrics['total_attacks'], 3)
        self.assertEqual(metrics['successful_attacks'], 2)
        self.assertAlmostEqual(metrics['success_rate'], 2/3, places=2)
    
    def test_mock_verifier(self):
        """Test mock base verifier for testing."""
        verifier = MockBaseVerifier()
        
        model = SimpleCNN()
        input_data = torch.randn(5, 3, 32, 32)
        reference = torch.randn(5, 10)
        
        result = verifier.verify(model, input_data, reference)
        
        self.assertIn('verified', result)
        self.assertIn('confidence', result)
        self.assertIsInstance(result['verified'], bool)
        self.assertIsInstance(result['confidence'], float)
    
    def test_wrapper_detector_initialization(self):
        """Test wrapper detector initialization."""
        from pot.core.attack_suites import WrapperDetector
        
        baseline_times = np.random.normal(10, 1, 100)
        baseline_responses = np.random.normal(0, 1, (100, 10))
        
        detector = WrapperDetector(baseline_times, baseline_responses)
        
        self.assertIsNotNone(detector.baseline_times)
        self.assertIsNotNone(detector.baseline_responses)
        self.assertTrue(hasattr(detector, 'detect_wrapper'))
    
    def test_vision_attack_imports(self):
        """Test vision attack imports."""
        from pot.vision.attacks import (
            AdversarialPatchAttack,
            UniversalPerturbationAttack,
            VisionModelExtraction,
            BackdoorAttack,
            execute_vision_attack
        )
        
        # Check classes can be instantiated
        patch_attack = AdversarialPatchAttack(patch_size=8, epsilon=0.03)
        self.assertIsNotNone(patch_attack)
        
        universal_attack = UniversalPerturbationAttack(epsilon=0.05)
        self.assertIsNotNone(universal_attack)
        
        extraction_attack = VisionModelExtraction(extraction_method='jacobian')
        self.assertIsNotNone(extraction_attack)
        
        backdoor_attack = BackdoorAttack(trigger_pattern='checkerboard')
        self.assertIsNotNone(backdoor_attack)
    
    def test_defense_component_imports(self):
        """Test defense component imports."""
        from pot.core.defenses import (
            AdaptiveVerifier,
            InputFilter,
            RandomizedDefense
        )
        
        # Check components can be instantiated
        base_verifier = MockBaseVerifier()
        adaptive = AdaptiveVerifier(base_verifier, initial_threshold=0.05)
        self.assertIsNotNone(adaptive)
        
        filter_defense = InputFilter(filter_strength=0.5)
        self.assertIsNotNone(filter_defense)
        
        randomized = RandomizedDefense(randomization_scale=0.1)
        self.assertIsNotNone(randomized)
    
    def test_comprehensive_suite_categories(self):
        """Test comprehensive suite attack categories."""
        from pot.core.attack_suites import ComprehensiveAttackSuite
        
        suite = ComprehensiveAttackSuite()
        categories = suite.get_attacks_by_category()
        
        # Check expected categories exist
        expected_categories = ['distillation', 'compression', 'fine_tuning']
        for category in expected_categories:
            self.assertIn(category, categories)
            self.assertGreater(len(categories[category]), 0)


def run_core_tests():
    """Run core functionality tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_core_tests()