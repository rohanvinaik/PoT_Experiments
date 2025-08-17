"""
Comprehensive tests for attack suite functionality.
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

# Import attack components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pot.core.attack_suites import (
    AttackConfig,
    StandardAttackSuite,
    AdaptiveAttackSuite,
    ComprehensiveAttackSuite,
    BenchmarkSuite,
    AttackRunner,
    WrapperDetector
)
from pot.vision.attacks import (
    AdversarialPatchAttack,
    UniversalPerturbationAttack,
    VisionModelExtraction,
    BackdoorAttack,
    execute_vision_attack
)


class TestModel(nn.Module):
    """Simple test model for attack testing."""
    
    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class TestDistillationAttack(unittest.TestCase):
    """Test distillation-based attacks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        self.device = 'cpu'
        
        # Create test data
        self.test_data = torch.randn(100, 1, 28, 28)
        self.test_labels = torch.randint(0, 10, (100,))
        self.dataset = TensorDataset(self.test_data, self.test_labels)
        self.data_loader = DataLoader(self.dataset, batch_size=10)
        
    def test_distillation_attack_weak(self):
        """Test weak distillation attack."""
        # Get weak distillation config
        suite = StandardAttackSuite()
        configs = suite.get_distillation_configs()
        weak_config = configs[0]  # First config is weak
        
        self.assertEqual(weak_config.name, "distillation_weak")
        self.assertEqual(weak_config.parameters["temperature"], 3.0)
        
        # Create mock student model
        student = TestModel()
        
        # Run distillation attack
        with patch('pot.core.attack_suites.StandardAttackSuite._execute_attack') as mock_attack:
            mock_attack.return_value = {
                'success': True,
                'confidence': 0.65,
                'student_model': student,
                'agreement_rate': 0.82
            }
            
            result = suite._execute_attack(weak_config, self.model, self.data_loader)
            
            self.assertTrue(result['success'])
            self.assertGreater(result['agreement_rate'], 0.8)
            self.assertLess(result['confidence'], 0.7)
    
    def test_distillation_attack_strong(self):
        """Test strong distillation attack."""
        suite = StandardAttackSuite()
        configs = suite.get_distillation_configs()
        strong_config = configs[2]  # Third config is strong
        
        self.assertEqual(strong_config.name, "distillation_strong")
        self.assertEqual(strong_config.parameters["temperature"], 10.0)
        self.assertEqual(strong_config.parameters["epochs"], 50)
        
        # Test with higher expected performance
        with patch('pot.core.attack_suites.StandardAttackSuite._execute_attack') as mock_attack:
            mock_attack.return_value = {
                'success': True,
                'confidence': 0.88,
                'agreement_rate': 0.94,
                'loss': 0.08
            }
            
            result = suite._execute_attack(strong_config, self.model, self.data_loader)
            
            self.assertTrue(result['success'])
            self.assertGreater(result['agreement_rate'], 0.9)
            self.assertGreater(result['confidence'], 0.85)
    
    def test_distillation_with_data_augmentation(self):
        """Test distillation with data augmentation."""
        suite = StandardAttackSuite()
        config = AttackConfig(
            name="distillation_augmented",
            attack_type="distillation",
            parameters={
                "temperature": 5.0,
                "epochs": 30,
                "learning_rate": 0.001,
                "augmentation": True,
                "augmentation_factor": 5
            }
        )
        
        # Mock augmented training
        with patch('pot.core.attack_suites.StandardAttackSuite._execute_attack') as mock_attack:
            # Simulate improved performance with augmentation
            mock_attack.return_value = {
                'success': True,
                'confidence': 0.92,
                'agreement_rate': 0.96,
                'augmented_samples': 500
            }
            
            result = suite._execute_attack(config, self.model, self.data_loader)
            
            self.assertTrue(result['success'])
            self.assertIn('augmented_samples', result)
            self.assertGreater(result['agreement_rate'], 0.95)


class TestCompressionAttack(unittest.TestCase):
    """Test compression-based attacks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        self.device = 'cpu'
        
        # Create test data
        self.test_data = torch.randn(50, 1, 28, 28)
        self.test_labels = torch.randint(0, 10, (50,))
        self.dataset = TensorDataset(self.test_data, self.test_labels)
        self.data_loader = DataLoader(self.dataset, batch_size=10)
    
    def test_compression_attack_weak(self):
        """Test weak compression attack (minimal pruning)."""
        suite = StandardAttackSuite()
        configs = suite.get_compression_configs()
        weak_config = configs[0]
        
        self.assertEqual(weak_config.parameters["pruning_rate"], 0.3)
        self.assertEqual(weak_config.parameters["quantization_bits"], 8)
        
        # Mock compression results
        with patch('pot.core.attack_suites.StandardAttackSuite._execute_attack') as mock_attack:
            mock_attack.return_value = {
                'success': True,
                'confidence': 0.72,
                'model_size_reduction': 0.35,
                'accuracy_retention': 0.92,
                'pruned_params': 30000
            }
            
            result = suite._execute_attack(weak_config, self.model, self.data_loader)
            
            self.assertTrue(result['success'])
            self.assertGreater(result['accuracy_retention'], 0.9)
            self.assertLess(result['model_size_reduction'], 0.4)
    
    def test_compression_attack_strong(self):
        """Test strong compression attack (aggressive pruning/quantization)."""
        suite = StandardAttackSuite()
        configs = suite.get_compression_configs()
        strong_config = configs[2]
        
        self.assertEqual(strong_config.parameters["pruning_rate"], 0.8)
        self.assertEqual(strong_config.parameters["quantization_bits"], 2)
        
        # Mock aggressive compression
        with patch('pot.core.attack_suites.StandardAttackSuite._execute_attack') as mock_attack:
            mock_attack.return_value = {
                'success': True,
                'confidence': 0.85,
                'model_size_reduction': 0.82,
                'accuracy_retention': 0.68,
                'pruned_params': 80000,
                'quantization_error': 0.15
            }
            
            result = suite._execute_attack(strong_config, self.model, self.data_loader)
            
            self.assertTrue(result['success'])
            self.assertGreater(result['model_size_reduction'], 0.8)
            self.assertLess(result['accuracy_retention'], 0.7)
    
    def test_structured_pruning(self):
        """Test structured pruning attack."""
        suite = StandardAttackSuite()
        config = AttackConfig(
            name="structured_pruning",
            attack_type="compression",
            parameters={
                "pruning_type": "structured",
                "pruning_rate": 0.5,
                "target_layers": ["fc1", "fc2"],
                "preserve_accuracy": False
            }
        )
        
        with patch('pot.core.attack_suites.StandardAttackSuite._execute_attack') as mock_attack:
            mock_attack.return_value = {
                'success': True,
                'pruned_channels': 64,
                'pruned_layers': 2,
                'speedup': 2.1
            }
            
            result = suite._execute_attack(config, self.model, self.data_loader)
            
            self.assertTrue(result['success'])
            self.assertIn('pruned_channels', result)
            self.assertGreater(result['speedup'], 2.0)


class TestWrapperDetection(unittest.TestCase):
    """Test wrapper detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        
        # Create baseline statistics
        self.baseline_times = np.random.normal(10, 1, 100)  # ms
        self.baseline_responses = np.random.normal(0, 1, (100, 10))
        
        self.detector = WrapperDetector(
            baseline_inference_times=self.baseline_times,
            baseline_responses=self.baseline_responses
        )
    
    def test_wrapper_detection_accuracy(self):
        """Test wrapper detection on known wrapped models."""
        # Test legitimate model (similar to baseline)
        legit_times = np.random.normal(10.5, 1.2, 50)
        legit_responses = np.random.normal(0, 1.1, (50, 10))
        
        result = self.detector.detect_wrapper(legit_times, legit_responses)
        self.assertFalse(result['is_wrapped'])
        self.assertLess(result['confidence'], 0.3)
        
        # Test wrapped model (different characteristics)
        wrapped_times = np.random.normal(25, 3, 50)  # Much slower
        wrapped_responses = np.random.normal(0.5, 2, (50, 10))  # Different distribution
        
        result = self.detector.detect_wrapper(wrapped_times, wrapped_responses)
        self.assertTrue(result['is_wrapped'])
        self.assertGreater(result['confidence'], 0.7)
        self.assertIn('timing_anomaly', result)
        self.assertIn('response_anomaly', result)
    
    def test_statistical_detection_methods(self):
        """Test different statistical detection methods."""
        # Test with significant timing differences
        slow_times = np.random.normal(50, 5, 100)
        normal_responses = np.random.normal(0, 1, (100, 10))
        
        result = self.detector.detect_wrapper(slow_times, normal_responses)
        
        self.assertTrue(result['timing_anomaly'])
        self.assertGreater(result['timing_metrics']['ks_statistic'], 0.5)
        self.assertLess(result['timing_metrics']['p_value'], 0.05)
    
    def test_adaptive_wrapper_detection(self):
        """Test detection of adaptive wrappers."""
        # Simulate adaptive wrapper that changes behavior
        adaptive_times = []
        for i in range(100):
            if i < 30:
                # Initially slow
                adaptive_times.append(np.random.normal(30, 2))
            elif i < 60:
                # Adapts to be faster
                adaptive_times.append(np.random.normal(20, 1.5))
            else:
                # Converges to normal
                adaptive_times.append(np.random.normal(12, 1))
        
        adaptive_times = np.array(adaptive_times)
        adaptive_responses = np.random.normal(0, 1, (100, 10))
        
        result = self.detector.detect_wrapper(adaptive_times, adaptive_responses)
        
        # Should detect variance anomaly
        self.assertIn('variance_anomaly', result)
        self.assertTrue(result['is_wrapped'] or result['variance_anomaly'])


class TestAdaptiveAttacks(unittest.TestCase):
    """Test adaptive attack generation and evolution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        self.device = 'cpu'
        
        # Create test data
        self.test_data = torch.randn(100, 1, 28, 28)
        self.test_labels = torch.randint(0, 10, (100,))
        self.dataset = TensorDataset(self.test_data, self.test_labels)
        self.data_loader = DataLoader(self.dataset, batch_size=10)
        
        self.adaptive_suite = AdaptiveAttackSuite(population_size=10)
    
    def test_adaptive_attack_evolution(self):
        """Test adaptive attack generation and evolution."""
        # Initial population
        initial_pop = self.adaptive_suite._initialize_population()
        self.assertEqual(len(initial_pop), 10)
        
        # Test evolution
        with patch.object(self.adaptive_suite, '_evaluate_fitness') as mock_fitness:
            # Mock fitness scores
            mock_fitness.side_effect = lambda x, y, z: np.random.random()
            
            # Evolve population
            evolved_pop = self.adaptive_suite._evolve_population(
                initial_pop, self.model, self.data_loader
            )
            
            self.assertEqual(len(evolved_pop), 10)
            
            # Verify mutation occurred
            mutated = False
            for i, config in enumerate(evolved_pop):
                if i < len(initial_pop):
                    if config.parameters != initial_pop[i].parameters:
                        mutated = True
                        break
            
            self.assertTrue(mutated, "Population should have mutated")
    
    def test_defense_observation(self):
        """Test observation and adaptation to defenses."""
        # Mock defense responses
        defense_responses = [
            {'detected': True, 'confidence': 0.8, 'defense_type': 'input_filter'},
            {'detected': False, 'confidence': 0.3, 'defense_type': 'none'},
            {'detected': True, 'confidence': 0.9, 'defense_type': 'adaptive_threshold'}
        ]
        
        # Learn from defenses
        self.adaptive_suite.observe_defense(defense_responses[0])
        self.adaptive_suite.observe_defense(defense_responses[1])
        self.adaptive_suite.observe_defense(defense_responses[2])
        
        self.assertEqual(len(self.adaptive_suite.defense_history), 3)
        
        # Generate adapted attack
        adapted_config = self.adaptive_suite._generate_evolved_attack(
            base_config=AttackConfig("base", "distillation", {"temperature": 5.0}),
            fitness_scores={'base': 0.5}
        )
        
        self.assertIsNotNone(adapted_config)
        self.assertIn('evasion', adapted_config.name.lower())
    
    def test_fitness_evaluation(self):
        """Test fitness evaluation for attacks."""
        config = AttackConfig(
            name="test_attack",
            attack_type="distillation",
            parameters={"temperature": 5.0, "epochs": 10}
        )
        
        with patch.object(self.adaptive_suite, '_execute_attack') as mock_attack:
            mock_attack.return_value = {
                'success': True,
                'confidence': 0.75,
                'evasion_rate': 0.6
            }
            
            fitness = self.adaptive_suite._evaluate_fitness(
                config, self.model, self.data_loader
            )
            
            # Fitness should combine success and evasion
            self.assertGreater(fitness, 0.5)
            self.assertLess(fitness, 1.0)


class TestVisionAttacks(unittest.TestCase):
    """Test vision-specific attacks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel(input_dim=3*32*32, num_classes=10)
        self.device = 'cpu'
        
        # Create vision test data (3x32x32 images)
        self.test_data = torch.randn(50, 3, 32, 32)
        self.test_labels = torch.randint(0, 10, (50,))
        self.dataset = TensorDataset(self.test_data, self.test_labels)
        self.data_loader = DataLoader(self.dataset, batch_size=5)
    
    def test_adversarial_patch_attack(self):
        """Test adversarial patch generation and application."""
        attack = AdversarialPatchAttack(
            patch_size=8,
            epsilon=0.03,
            optimization_steps=10
        )
        
        # Generate patch
        patch = attack.generate_patch(
            self.model,
            self.data_loader,
            target_class=0,
            device=self.device
        )
        
        self.assertEqual(patch.shape, (3, 8, 8))
        self.assertLessEqual(patch.abs().max().item(), 0.03)
        
        # Apply patch
        patched_images = attack.apply_patch(
            self.test_data[:10],
            patch,
            location=(12, 12)
        )
        
        self.assertEqual(patched_images.shape, self.test_data[:10].shape)
        
        # Verify patch was applied
        patch_region = patched_images[0, :, 12:20, 12:20]
        self.assertTrue(torch.allclose(patch_region, patch, atol=1e-5))
    
    def test_universal_perturbation_attack(self):
        """Test universal perturbation generation."""
        attack = UniversalPerturbationAttack(
            epsilon=0.05,
            max_iterations=5,
            overshoot=0.02
        )
        
        # Generate universal perturbation
        perturbation = attack.generate_universal_perturbation(
            self.model,
            self.data_loader,
            device=self.device
        )
        
        self.assertEqual(perturbation.shape, (3, 32, 32))
        self.assertLessEqual(perturbation.abs().max().item(), 0.05)
        
        # Test transferability
        transfer_rate = attack.evaluate_transferability(
            perturbation,
            self.model,  # Using same model for testing
            self.data_loader,
            device=self.device
        )
        
        self.assertIsInstance(transfer_rate, float)
        self.assertGreaterEqual(transfer_rate, 0.0)
        self.assertLessEqual(transfer_rate, 1.0)
    
    def test_model_extraction_attack(self):
        """Test model extraction attacks."""
        attack = VisionModelExtraction(
            extraction_method='jacobian',
            query_budget=100
        )
        
        # Test Jacobian-based extraction
        jacobian_info = attack.extract_via_jacobian(
            self.model,
            self.test_data[:10],
            device=self.device
        )
        
        self.assertIn('jacobians', jacobian_info)
        self.assertIn('layer_info', jacobian_info)
        self.assertEqual(len(jacobian_info['jacobians']), 10)
        
        # Test prediction-based extraction
        attack.extraction_method = 'prediction'
        predictions = attack.extract_via_predictions(
            self.model,
            self.data_loader,
            num_queries=50,
            device=self.device
        )
        
        self.assertIn('queries', predictions)
        self.assertIn('responses', predictions)
        self.assertEqual(len(predictions['queries']), 50)
    
    def test_backdoor_attack(self):
        """Test backdoor injection and detection."""
        attack = BackdoorAttack(
            trigger_pattern='checkerboard',
            trigger_size=4,
            poison_rate=0.1
        )
        
        # Inject backdoor
        backdoor_result = attack.inject_backdoor(
            self.model,
            self.data_loader,
            target_class=7,
            epochs=2,
            device=self.device
        )
        
        self.assertIn('backdoored_model', backdoor_result)
        self.assertIn('trigger', backdoor_result)
        self.assertIn('success_rate', backdoor_result)
        
        # Detect backdoor
        detection_result = attack.detect_backdoor(
            self.model,
            self.data_loader,
            method='activation_clustering',
            device=self.device
        )
        
        self.assertIn('is_backdoored', detection_result)
        self.assertIn('confidence', detection_result)
        self.assertIn('suspicious_neurons', detection_result)


class TestAttackIntegration(unittest.TestCase):
    """Test end-to-end attack integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        self.device = 'cpu'
        
        # Create test data
        self.test_data = torch.randn(100, 1, 28, 28)
        self.test_labels = torch.randint(0, 10, (100,))
        self.dataset = TensorDataset(self.test_data, self.test_labels)
        self.data_loader = DataLoader(self.dataset, batch_size=10)
        
        self.runner = AttackRunner(device=self.device)
    
    def test_attack_runner_pipeline(self):
        """Test complete attack evaluation harness."""
        # Configure attacks
        attack_configs = [
            AttackConfig("dist_test", "distillation", {"temperature": 5.0}),
            AttackConfig("comp_test", "compression", {"pruning_rate": 0.5})
        ]
        
        # Run attack suite
        with patch.object(self.runner, 'run_single_attack') as mock_run:
            mock_run.side_effect = [
                {'success': True, 'confidence': 0.7},
                {'success': False, 'confidence': 0.3}
            ]
            
            results = self.runner.run_attack_suite(
                self.model,
                attack_configs,
                self.data_loader
            )
            
            self.assertEqual(len(results), 2)
            self.assertTrue(results[0]['success'])
            self.assertFalse(results[1]['success'])
    
    def test_comprehensive_suite_execution(self):
        """Test comprehensive attack suite execution."""
        suite = ComprehensiveAttackSuite()
        
        # Get all attacks
        all_attacks = suite.get_all_attacks()
        self.assertGreater(len(all_attacks), 10)
        
        # Test categorization
        categories = suite.get_attacks_by_category()
        self.assertIn('distillation', categories)
        self.assertIn('compression', categories)
        self.assertIn('fine_tuning', categories)
        self.assertIn('vision', categories)
    
    def test_benchmark_suite(self):
        """Test benchmark suite configurations."""
        benchmark = BenchmarkSuite()
        
        # Test baseline config
        baseline = benchmark.get_baseline_config()
        self.assertEqual(baseline['name'], 'baseline')
        self.assertIn('attacks', baseline)
        
        # Test adversarial config
        adversarial = benchmark.get_adversarial_config()
        self.assertEqual(adversarial['name'], 'adversarial')
        self.assertIn('vision', adversarial['focus'])
        
        # Test comprehensive config
        comprehensive = benchmark.get_comprehensive_config()
        self.assertEqual(comprehensive['name'], 'comprehensive')
        self.assertGreater(len(comprehensive['attacks']), 15)
    
    def test_attack_success_evaluation(self):
        """Test attack success evaluation metrics."""
        # Create mock verification results
        baseline_result = {'verified': True, 'confidence': 0.95}
        attack_result = {'verified': False, 'confidence': 0.45}
        
        # Evaluate success
        success = self.runner.evaluate_attack_success(
            baseline_result,
            attack_result,
            threshold=0.5
        )
        
        self.assertTrue(success)
        
        # Test with different threshold
        success_strict = self.runner.evaluate_attack_success(
            baseline_result,
            {'verified': True, 'confidence': 0.7},
            threshold=0.3
        )
        
        self.assertFalse(success_strict)


class TestAttackMetrics(unittest.TestCase):
    """Test attack evaluation metrics."""
    
    def test_attack_success_rate(self):
        """Test calculation of attack success rates."""
        results = [
            {'attack': 'dist1', 'success': True, 'confidence': 0.8},
            {'attack': 'dist2', 'success': True, 'confidence': 0.6},
            {'attack': 'comp1', 'success': False, 'confidence': 0.3},
            {'attack': 'comp2', 'success': True, 'confidence': 0.7}
        ]
        
        runner = AttackRunner()
        metrics = runner.calculate_metrics(results)
        
        self.assertEqual(metrics['total_attacks'], 4)
        self.assertEqual(metrics['successful_attacks'], 3)
        self.assertEqual(metrics['success_rate'], 0.75)
        self.assertAlmostEqual(metrics['average_confidence'], 0.6, places=2)
    
    def test_robustness_scoring(self):
        """Test model robustness scoring."""
        runner = AttackRunner()
        
        # High robustness (few successful attacks)
        results_robust = [
            {'success': False, 'confidence': 0.2},
            {'success': False, 'confidence': 0.3},
            {'success': True, 'confidence': 0.55},
            {'success': False, 'confidence': 0.1}
        ]
        
        score_robust = runner.calculate_robustness_score(results_robust)
        self.assertGreater(score_robust, 0.7)
        
        # Low robustness (many successful attacks)
        results_weak = [
            {'success': True, 'confidence': 0.9},
            {'success': True, 'confidence': 0.85},
            {'success': True, 'confidence': 0.8},
            {'success': False, 'confidence': 0.4}
        ]
        
        score_weak = runner.calculate_robustness_score(results_weak)
        self.assertLess(score_weak, 0.3)
    
    def test_performance_degradation_metrics(self):
        """Test measurement of performance degradation."""
        baseline_perf = {
            'accuracy': 0.95,
            'latency': 10.0,
            'memory': 100.0
        }
        
        attacked_perf = {
            'accuracy': 0.78,
            'latency': 25.0,
            'memory': 150.0
        }
        
        runner = AttackRunner()
        degradation = runner.calculate_degradation(baseline_perf, attacked_perf)
        
        self.assertAlmostEqual(degradation['accuracy_drop'], 0.17, places=2)
        self.assertAlmostEqual(degradation['latency_increase'], 1.5, places=1)
        self.assertAlmostEqual(degradation['memory_increase'], 0.5, places=1)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()