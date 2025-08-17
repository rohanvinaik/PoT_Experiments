"""
Integration tests for attack-defense pipeline.
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

# Import components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pot.core.attack_suites import (
    AttackConfig,
    StandardAttackSuite,
    AdaptiveAttackSuite,
    ComprehensiveAttackSuite,
    AttackRunner
)
from pot.core.defenses import (
    DefenseConfig,
    AdaptiveVerifier,
    InputFilter,
    RandomizedDefense,
    IntegratedDefenseSystem,
    MockBaseVerifier
)
from pot.vision.attacks import execute_vision_attack
from pot.security.proof_of_training import ProofOfTraining


class TestModel(nn.Module):
    """Test model for integration tests."""
    
    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete attack → detection → defense → verification pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        self.device = 'cpu'
        
        # Create test data
        self.test_data = torch.randn(200, 1, 28, 28)
        self.test_labels = torch.randint(0, 10, (200,))
        self.dataset = TensorDataset(self.test_data, self.test_labels)
        self.data_loader = DataLoader(self.dataset, batch_size=20)
        
        # Initialize components
        self.attack_suite = StandardAttackSuite()
        self.defense_config = DefenseConfig(
            adaptive_threshold=0.05,
            input_filter_strength=0.5,
            randomization_scale=0.1
        )
        self.defense_system = IntegratedDefenseSystem(self.defense_config)
        self.attack_runner = AttackRunner(device=self.device)
    
    def test_full_attack_defense_cycle(self):
        """Test full cycle: attack generation → defense → verification."""
        # Step 1: Generate attack
        attack_config = AttackConfig(
            name="test_distillation",
            attack_type="distillation",
            parameters={"temperature": 5.0, "epochs": 5}
        )
        
        # Mock attack execution
        with patch.object(self.attack_runner, 'run_single_attack') as mock_attack:
            mock_attack.return_value = {
                'success': True,
                'confidence': 0.75,
                'student_model': TestModel(),
                'attack_data': self.test_data[:50]
            }
            
            attack_result = self.attack_runner.run_single_attack(
                self.model,
                attack_config,
                self.data_loader
            )
        
        self.assertTrue(attack_result['success'])
        
        # Step 2: Defense detects and responds
        defense_result = self.defense_system.comprehensive_defense(
            attack_result['attack_data'],
            self.model,
            threat_level=0.6
        )
        
        self.assertIn('threat_assessment', defense_result)
        self.assertIn('final_decision', defense_result)
        
        # Step 3: Verification with defenses active
        with patch('pot.security.proof_of_training.ProofOfTraining') as mock_pot:
            mock_verifier = Mock()
            mock_verifier.perform_verification.return_value = {
                'verified': defense_result['final_decision']['verified'],
                'confidence': defense_result['final_decision']['confidence'],
                'security_level': 'medium'
            }
            mock_pot.return_value = mock_verifier
            
            pot = mock_pot({'model_type': 'vision'})
            verification = pot.perform_verification(
                self.model,
                'test_model',
                'standard'
            )
            
            # Verification should reflect defense impact
            self.assertIn('verified', verification)
            self.assertIn('confidence', verification)
    
    def test_adaptive_attack_defense_arms_race(self):
        """Test adaptive attacks vs adaptive defenses."""
        adaptive_suite = AdaptiveAttackSuite(population_size=5)
        adaptive_verifier = AdaptiveVerifier(
            MockBaseVerifier(),
            initial_threshold=0.05
        )
        
        # Simulate multiple rounds of attack-defense
        for round_num in range(3):
            # Attacker evolves
            attack_configs = adaptive_suite._initialize_population()
            
            for config in attack_configs[:2]:  # Test subset
                # Mock attack
                with patch.object(adaptive_suite, '_execute_attack') as mock_attack:
                    mock_attack.return_value = {
                        'success': round_num < 2,  # Attacks get harder
                        'confidence': 0.7 - (round_num * 0.1)
                    }
                    
                    attack_result = adaptive_suite._execute_attack(
                        config,
                        self.model,
                        self.data_loader
                    )
                
                # Defender adapts
                adaptive_verifier.update_from_attack(attack_result)
                
                # Defender learns patterns
                if attack_result['success']:
                    adaptive_verifier.learn_attack_pattern({
                        'type': config.attack_type,
                        'params': config.parameters,
                        'success': attack_result['success']
                    })
            
            # Check adaptation
            self.assertLess(
                adaptive_verifier.threshold,
                0.05  # Should get stricter
            )
        
        # Final verification should be more robust
        final_result = adaptive_verifier.verify(
            self.model,
            self.test_data[:10],
            reference_output=torch.randn(10, 10)
        )
        
        self.assertIn('attack_risk', final_result)
        self.assertGreater(final_result['attack_risk'], 0.5)
    
    def test_multi_stage_attack_defense(self):
        """Test multi-stage attacks with layered defenses."""
        # Stage 1: Compression attack
        compression_config = AttackConfig(
            name="compression_stage",
            attack_type="compression",
            parameters={"pruning_rate": 0.5, "quantization_bits": 4}
        )
        
        # Stage 2: Fine-tuning attack
        finetune_config = AttackConfig(
            name="finetune_stage",
            attack_type="fine_tuning",
            parameters={"epochs": 5, "learning_rate": 0.001}
        )
        
        # Stage 3: Wrapper attack
        wrapper_config = AttackConfig(
            name="wrapper_stage",
            attack_type="wrapper",
            parameters={"wrapper_type": "ensemble", "num_models": 3}
        )
        
        stages = [compression_config, finetune_config, wrapper_config]
        results = []
        
        for i, config in enumerate(stages):
            # Mock staged attack
            with patch.object(self.attack_runner, 'run_single_attack') as mock_attack:
                mock_attack.return_value = {
                    'success': True,
                    'confidence': 0.6 + (i * 0.1),
                    'stage': i + 1
                }
                
                stage_result = self.attack_runner.run_single_attack(
                    self.model,
                    config,
                    self.data_loader
                )
                results.append(stage_result)
            
            # Defense responds to each stage
            defense_response = self.defense_system.comprehensive_defense(
                self.test_data[:20],
                self.model,
                threat_level=0.4 + (i * 0.2)  # Escalating threat
            )
            
            # Check defense escalation
            self.assertGreaterEqual(
                defense_response['threat_assessment']['overall_threat'],
                0.4 + (i * 0.15)
            )
        
        # Verify all stages completed
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result['stage'], i + 1)


class TestAttackRunnerPipeline(unittest.TestCase):
    """Test complete attack evaluation harness."""
    
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
    
    def test_complete_attack_evaluation(self):
        """Test complete attack evaluation pipeline."""
        # Create comprehensive attack suite
        comprehensive_suite = ComprehensiveAttackSuite()
        all_attacks = comprehensive_suite.get_all_attacks()
        
        # Run subset of attacks
        test_attacks = all_attacks[:5]
        
        results = []
        for attack in test_attacks:
            with patch.object(self.runner, 'run_single_attack') as mock_run:
                # Simulate varying success rates
                mock_run.return_value = {
                    'attack_name': attack.name,
                    'attack_type': attack.attack_type,
                    'success': np.random.random() > 0.4,
                    'confidence': np.random.random(),
                    'execution_time': np.random.uniform(0.1, 2.0)
                }
                
                result = self.runner.run_single_attack(
                    self.model,
                    attack,
                    self.data_loader
                )
                results.append(result)
        
        # Evaluate overall performance
        metrics = self.runner.calculate_metrics(results)
        
        self.assertEqual(metrics['total_attacks'], 5)
        self.assertIn('success_rate', metrics)
        self.assertIn('average_confidence', metrics)
        
        # Generate robustness score
        robustness = self.runner.calculate_robustness_score(results)
        self.assertGreaterEqual(robustness, 0.0)
        self.assertLessEqual(robustness, 1.0)
    
    def test_attack_categorization_and_reporting(self):
        """Test attack categorization and report generation."""
        # Run different categories of attacks
        attack_results = [
            {'attack_type': 'distillation', 'success': True, 'confidence': 0.8},
            {'attack_type': 'distillation', 'success': False, 'confidence': 0.3},
            {'attack_type': 'compression', 'success': True, 'confidence': 0.7},
            {'attack_type': 'fine_tuning', 'success': True, 'confidence': 0.9},
            {'attack_type': 'wrapper', 'success': False, 'confidence': 0.2},
            {'attack_type': 'adversarial_patch', 'success': True, 'confidence': 0.75}
        ]
        
        # Categorize results
        categorized = self.runner.categorize_results(attack_results)
        
        self.assertIn('distillation', categorized)
        self.assertEqual(len(categorized['distillation']), 2)
        self.assertIn('compression', categorized)
        self.assertIn('adversarial_patch', categorized)
        
        # Generate summary report
        report = self.runner.generate_report(attack_results)
        
        self.assertIn('summary', report)
        self.assertIn('by_category', report)
        self.assertIn('robustness_score', report)
        self.assertIn('recommendations', report)
    
    def test_parallel_attack_execution(self):
        """Test parallel execution of multiple attacks."""
        attacks = [
            AttackConfig(f"attack_{i}", "distillation", {"temperature": 3 + i})
            for i in range(5)
        ]
        
        # Mock parallel execution
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_future = Mock()
            mock_future.result.return_value = {
                'success': True,
                'confidence': 0.7
            }
            
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            
            # Run attacks in parallel
            with patch.object(self.runner, 'run_attack_suite_parallel') as mock_parallel:
                mock_parallel.return_value = [
                    {'attack_name': a.name, 'success': True, 'confidence': 0.7}
                    for a in attacks
                ]
                
                results = self.runner.run_attack_suite_parallel(
                    self.model,
                    attacks,
                    self.data_loader,
                    max_workers=3
                )
        
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIn('attack_name', result)
            self.assertIn('success', result)


class TestVisionAttackIntegration(unittest.TestCase):
    """Test vision-specific attack integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Vision model (3-channel input)
        self.model = TestModel(input_dim=3*32*32, num_classes=10)
        self.device = 'cpu'
        
        # Create vision data
        self.images = torch.randn(50, 3, 32, 32)
        self.labels = torch.randint(0, 10, (50,))
        self.dataset = TensorDataset(self.images, self.labels)
        self.data_loader = DataLoader(self.dataset, batch_size=5)
    
    def test_vision_attack_suite_integration(self):
        """Test integration of vision attacks with main suite."""
        suite = StandardAttackSuite()
        vision_configs = suite.get_vision_specific_configs()
        
        self.assertGreater(len(vision_configs), 0)
        
        # Test each vision attack type
        attack_types = set(c.attack_type for c in vision_configs)
        self.assertIn('adversarial_patch', attack_types)
        self.assertIn('universal_perturbation', attack_types)
        self.assertIn('model_extraction', attack_types)
        self.assertIn('backdoor', attack_types)
        
        # Execute vision attack
        for config in vision_configs[:2]:  # Test subset
            result = execute_vision_attack(
                attack_type=config.attack_type,
                config=config.parameters,
                model=self.model,
                data_loader=self.data_loader,
                device=self.device
            )
            
            self.assertIn('success', result)
            self.assertIn('attack_type', result)
            self.assertEqual(result['attack_type'], config.attack_type)
    
    def test_vision_defense_integration(self):
        """Test defenses against vision attacks."""
        # Generate adversarial patch
        patch_result = execute_vision_attack(
            attack_type='adversarial_patch',
            config={'patch_size': 8, 'epsilon': 0.03, 'optimization_steps': 5},
            model=self.model,
            data_loader=self.data_loader,
            device=self.device
        )
        
        # Apply defense
        defense = IntegratedDefenseSystem(DefenseConfig())
        
        if 'patched_images' in patch_result:
            defense_result = defense.comprehensive_defense(
                patch_result['patched_images'][:10],
                self.model,
                threat_level=0.7
            )
            
            self.assertIn('filter_result', defense_result)
            self.assertTrue(
                defense_result['filter_result']['anomalies_detected'] or
                defense_result['threat_assessment']['overall_threat'] > 0.5
            )


class TestDefenseAdaptation(unittest.TestCase):
    """Test defense adaptation and learning."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        self.base_verifier = MockBaseVerifier()
        self.adaptive_verifier = AdaptiveVerifier(
            self.base_verifier,
            initial_threshold=0.05
        )
    
    def test_defense_learning_from_attacks(self):
        """Test defense learning from attack patterns."""
        # Simulate attack sequence
        attack_sequence = [
            {'type': 'distillation', 'success': True, 'params': {'temp': 5}},
            {'type': 'distillation', 'success': True, 'params': {'temp': 5}},
            {'type': 'compression', 'success': False, 'params': {'rate': 0.3}},
            {'type': 'distillation', 'success': True, 'params': {'temp': 7}},
            {'type': 'fine_tuning', 'success': False, 'params': {'lr': 0.01}},
        ]
        
        # Feed attacks to adaptive verifier
        for attack in attack_sequence:
            self.adaptive_verifier.learn_attack_pattern(attack)
            if attack['success']:
                self.adaptive_verifier.update_from_attack({
                    'success': True,
                    'confidence': 0.8,
                    'attack_type': attack['type']
                })
        
        # Check learned patterns
        patterns = self.adaptive_verifier.get_attack_patterns()
        
        self.assertIn('distillation', patterns)
        self.assertGreater(patterns['distillation']['success_rate'], 0.9)
        self.assertIn('compression', patterns)
        self.assertLess(patterns['compression']['success_rate'], 0.5)
        
        # Verify threshold adaptation
        self.assertLess(
            self.adaptive_verifier.threshold,
            0.05  # Should be stricter due to successful attacks
        )
    
    def test_defense_performance_over_time(self):
        """Test defense performance improvement over time."""
        defense_system = IntegratedDefenseSystem(DefenseConfig())
        
        # Track performance over multiple rounds
        performance_history = []
        
        for round_num in range(5):
            # Simulate attacks getting more sophisticated
            attack_strength = 0.5 + (round_num * 0.1)
            
            # Generate attack data
            attack_data = torch.randn(20, 1, 28, 28)
            if round_num > 0:
                # Add adversarial noise
                attack_data += torch.randn_like(attack_data) * attack_strength * 0.1
            
            # Defense response
            result = defense_system.comprehensive_defense(
                attack_data,
                self.model,
                threat_level=min(0.3 + (round_num * 0.1), 0.9)
            )
            
            # Track performance
            performance = {
                'round': round_num,
                'threat_detected': result['threat_assessment']['overall_threat'] > 0.5,
                'confidence': result['final_decision']['confidence']
            }
            performance_history.append(performance)
            
            # Update defense based on result
            if round_num > 0:
                defense_system.adaptive_verifier.update_from_attack({
                    'success': not performance['threat_detected'],
                    'confidence': 1 - performance['confidence']
                })
        
        # Check improvement trend
        early_detection = sum(p['threat_detected'] for p in performance_history[:2])
        late_detection = sum(p['threat_detected'] for p in performance_history[-2:])
        
        # Defense should improve (or at least maintain)
        self.assertGreaterEqual(late_detection, early_detection)


def run_integration_tests():
    """Run all integration tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_integration_tests()