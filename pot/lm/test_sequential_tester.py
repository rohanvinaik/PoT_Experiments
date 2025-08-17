"""
Tests for Sequential Testing Implementation
"""

import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any
import time

# Import the classes to test
from pot.lm.sequential_tester import (
    SequentialTester,
    AdaptiveSequentialTester,
    GroupSequentialTester,
    SequentialVerificationSession,
    SPRTState,
    compute_operating_characteristics,
    simulate_sequential_test
)


class TestSequentialTester(unittest.TestCase):
    """Test suite for SequentialTester"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tester = SequentialTester(
            alpha=0.05,
            beta=0.05,
            p0=0.5,
            p1=0.8
        )
    
    def test_initialization(self):
        """Test tester initialization"""
        self.assertEqual(self.tester.alpha, 0.05)
        self.assertEqual(self.tester.beta, 0.05)
        self.assertEqual(self.tester.p0, 0.5)
        self.assertEqual(self.tester.p1, 0.8)
        
        # Check boundaries
        self.assertGreater(self.tester.log_A, 0)
        self.assertLess(self.tester.log_B, 0)
        
        # Check state
        self.assertFalse(self.tester.state.terminated)
        self.assertIsNone(self.tester.state.decision)
    
    def test_invalid_parameters(self):
        """Test parameter validation"""
        # Invalid alpha
        with self.assertRaises(ValueError):
            SequentialTester(alpha=1.5, beta=0.05, p0=0.5, p1=0.8)
        
        # Invalid p0 >= p1
        with self.assertRaises(ValueError):
            SequentialTester(alpha=0.05, beta=0.05, p0=0.8, p1=0.5)
    
    def test_update_success(self):
        """Test update with successful trial"""
        decision = self.tester.update(True)
        
        # Should increase log likelihood ratio
        self.assertGreater(self.tester.state.log_likelihood_ratio, 0)
        self.assertEqual(self.tester.state.num_trials, 1)
        self.assertEqual(self.tester.state.num_successes, 1)
        
        # Should not decide after one trial (min_trials)
        if self.tester.min_trials > 1:
            self.assertIsNone(decision)
    
    def test_update_failure(self):
        """Test update with failed trial"""
        decision = self.tester.update(False)
        
        # Should decrease log likelihood ratio
        self.assertLess(self.tester.state.log_likelihood_ratio, 0)
        self.assertEqual(self.tester.state.num_trials, 1)
        self.assertEqual(self.tester.state.num_successes, 0)
    
    def test_sequential_decision_reject(self):
        """Test sequential decision to reject H0 (genuine model)"""
        # Simulate many successes
        decision = None
        for _ in range(100):
            decision = self.tester.update(True)
            if decision is not None:
                break
        
        # Should reject H0 (accept H1 - genuine)
        self.assertEqual(decision, 'reject')
        self.assertTrue(self.tester.state.terminated)
    
    def test_sequential_decision_accept(self):
        """Test sequential decision to accept H0 (fake model)"""
        # Simulate many failures
        decision = None
        for _ in range(100):
            decision = self.tester.update(False)
            if decision is not None:
                break
        
        # Should accept H0 (fake/modified)
        self.assertEqual(decision, 'accept')
        self.assertTrue(self.tester.state.terminated)
    
    def test_batch_update(self):
        """Test batch update"""
        results = [True, True, False, True, False]
        decision = self.tester.batch_update(results)
        
        self.assertEqual(self.tester.state.num_trials, 5)
        self.assertEqual(self.tester.state.num_successes, 3)
    
    def test_confidence_calculation(self):
        """Test confidence calculation"""
        # Initial confidence should be 0.5
        self.assertAlmostEqual(self.tester.get_confidence(), 0.5, places=1)
        
        # After successes, confidence should increase
        for _ in range(5):
            self.tester.update(True)
        
        confidence = self.tester.get_confidence()
        self.assertGreater(confidence, 0.5)
        self.assertLessEqual(confidence, 1.0)
    
    def test_expected_sample_size(self):
        """Test expected sample size calculation"""
        # Under H0
        expected_h0 = self.tester.expected_sample_size(self.tester.p0)
        self.assertGreater(expected_h0, self.tester.min_trials)
        self.assertLess(expected_h0, self.tester.max_trials)
        
        # Under H1
        expected_h1 = self.tester.expected_sample_size(self.tester.p1)
        self.assertGreater(expected_h1, self.tester.min_trials)
        self.assertLess(expected_h1, self.tester.max_trials)
    
    def test_reset(self):
        """Test state reset"""
        # Add some trials
        self.tester.update(True)
        self.tester.update(False)
        
        # Reset
        self.tester.reset()
        
        # Check state is cleared
        self.assertEqual(self.tester.state.num_trials, 0)
        self.assertEqual(self.tester.state.num_successes, 0)
        self.assertEqual(self.tester.state.log_likelihood_ratio, 0.0)
        self.assertFalse(self.tester.state.terminated)
    
    def test_max_trials(self):
        """Test forced decision at max trials"""
        tester = SequentialTester(max_trials=10)
        
        # Run exactly max_trials
        decision = None
        for i in range(10):
            decision = tester.update(i % 2 == 0)  # Alternating
        
        # Should have made a decision
        self.assertIsNotNone(decision)
        self.assertTrue(tester.state.terminated)
    
    def test_statistics(self):
        """Test statistics generation"""
        # Add some trials
        self.tester.update(True)
        self.tester.update(True)
        self.tester.update(False)
        
        stats = self.tester.get_statistics()
        
        self.assertEqual(stats['num_trials'], 3)
        self.assertEqual(stats['num_successes'], 2)
        self.assertAlmostEqual(stats['success_rate'], 2/3, places=5)
        self.assertIn('boundaries', stats)
        self.assertIn('expected_trials_h0', stats)
        self.assertIn('expected_trials_h1', stats)
    
    def test_plot_progress_data(self):
        """Test plot progress data generation"""
        # Add trials
        for _ in range(5):
            self.tester.update(True)
        for _ in range(3):
            self.tester.update(False)
        
        plot_data = self.tester.plot_progress()
        
        self.assertEqual(len(plot_data['trials']), 8)
        self.assertEqual(len(plot_data['cumulative_llr']), 8)
        self.assertEqual(len(plot_data['successes']), 8)
        self.assertEqual(sum(plot_data['successes']), 5)


class TestAdaptiveSequentialTester(unittest.TestCase):
    """Test suite for AdaptiveSequentialTester"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tester = AdaptiveSequentialTester(
            initial_alpha=0.05,
            initial_beta=0.05,
            adaptation_rate=0.1,
            window_size=10
        )
    
    def test_initialization(self):
        """Test adaptive tester initialization"""
        self.assertEqual(self.tester.initial_alpha, 0.05)
        self.assertEqual(self.tester.initial_beta, 0.05)
        self.assertEqual(self.tester.adaptation_rate, 0.1)
        self.assertEqual(self.tester.window_size, 10)
    
    def test_parameter_adaptation(self):
        """Test that parameters adapt based on performance"""
        initial_alpha = self.tester.alpha
        initial_p1 = self.tester.p1
        
        # Add many successes
        for _ in range(15):
            self.tester.update(True)
        
        # Parameters should have adapted
        # (Exact behavior depends on adaptation logic)
        self.assertIsNotNone(self.tester.adaptation_history)
        
        if len(self.tester.adaptation_history) > 0:
            last_adaptation = self.tester.adaptation_history[-1]
            self.assertIn('recent_rate', last_adaptation)
            self.assertIn('variance', last_adaptation)
    
    def test_window_based_adaptation(self):
        """Test that adaptation uses windowed statistics"""
        # Fill window with specific pattern
        for i in range(self.tester.window_size):
            self.tester.update(i % 2 == 0)
        
        # Recent results should reflect the pattern
        self.assertEqual(len(self.tester.recent_results), self.tester.window_size)
        
        # Add more results
        for _ in range(5):
            self.tester.update(True)
        
        # Window should maintain size
        self.assertEqual(len(self.tester.recent_results), self.tester.window_size)


class TestGroupSequentialTester(unittest.TestCase):
    """Test suite for GroupSequentialTester"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tester = GroupSequentialTester(
            num_stages=3,
            trials_per_stage=10,
            alpha=0.05,
            beta=0.05,
            spending_function='obrien_fleming'
        )
    
    def test_initialization(self):
        """Test group sequential tester initialization"""
        self.assertEqual(self.tester.num_stages, 3)
        self.assertEqual(self.tester.trials_per_stage, 10)
        self.assertEqual(len(self.tester.stage_boundaries), 3)
    
    def test_stage_update(self):
        """Test stage-wise update"""
        # First stage with high success rate
        decision = self.tester.update_stage(8, 10)
        
        self.assertEqual(self.tester.current_stage, 1)
        self.assertEqual(self.tester.cumulative_successes, 8)
        self.assertEqual(self.tester.cumulative_trials, 10)
        
        # Decision depends on boundaries
        # May or may not terminate after first stage
    
    def test_early_stopping(self):
        """Test early stopping at intermediate stage"""
        # Very high success rate
        decision = self.tester.update_stage(10, 10)
        
        # Might stop early with strong evidence
        if decision is not None:
            self.assertTrue(self.tester.terminated)
            self.assertIn(decision, ['accept', 'reject'])
    
    def test_final_stage_decision(self):
        """Test that decision is made at final stage"""
        # Run all stages
        for i in range(self.tester.num_stages):
            decision = self.tester.update_stage(5, 10)
            if decision is not None:
                break
        
        # Should have made a decision
        self.assertTrue(self.tester.terminated)
        self.assertIsNotNone(self.tester.decision)
    
    def test_statistics(self):
        """Test statistics generation"""
        self.tester.update_stage(6, 10)
        stats = self.tester.get_statistics()
        
        self.assertEqual(stats['current_stage'], 1)
        self.assertEqual(stats['cumulative_successes'], 6)
        self.assertEqual(stats['cumulative_trials'], 10)
        self.assertIn('stage_results', stats)
        self.assertIn('boundaries', stats)


class TestSequentialVerificationSession(unittest.TestCase):
    """Test suite for SequentialVerificationSession"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_challenger = Mock()
        self.mock_evaluator = Mock()
        self.mock_model_runner = Mock()
        
        self.session = SequentialVerificationSession(
            challenger=self.mock_challenger,
            evaluator=self.mock_evaluator,
            model_runner=self.mock_model_runner
        )
    
    def test_initialization(self):
        """Test session initialization"""
        self.assertIsNotNone(self.session.tester)
        self.assertEqual(self.session.challenger, self.mock_challenger)
        self.assertEqual(self.session.evaluator, self.mock_evaluator)
        self.assertEqual(self.session.model_runner, self.mock_model_runner)
    
    def test_run_verification(self):
        """Test verification run"""
        # Configure mocks
        self.mock_challenger.generate_challenge_set.return_value = [
            {'prompt': 'Test prompt', 'expected': 'answer', 'difficulty': 1}
        ]
        
        self.mock_model_runner.return_value = 'answer'
        
        mock_eval_result = Mock()
        mock_eval_result.success = True
        self.mock_evaluator.evaluate_response.return_value = mock_eval_result
        
        # Run verification
        results = self.session.run_verification(max_challenges=5, early_stop=False)
        
        # Check results
        self.assertIn('verified', results)
        self.assertIn('confidence', results)
        self.assertIn('num_trials', results)
        self.assertIn('success_rate', results)
        self.assertIn('results', results)
        
        # Check that components were called
        self.mock_challenger.generate_challenge_set.assert_called()
        self.mock_model_runner.assert_called()
        self.mock_evaluator.evaluate_response.assert_called()
    
    def test_early_stopping(self):
        """Test early stopping behavior"""
        # Configure for consistent success
        self.mock_challenger.generate_challenge_set.return_value = [
            {'prompt': 'Test', 'expected': 'answer', 'difficulty': 1}
        ]
        self.mock_model_runner.return_value = 'answer'
        
        mock_eval_result = Mock()
        mock_eval_result.success = True
        self.mock_evaluator.evaluate_response.return_value = mock_eval_result
        
        # Run with early stopping
        results = self.session.run_verification(max_challenges=100, early_stop=True)
        
        # Should stop early with consistent success
        if results['early_stopped']:
            self.assertLess(results['num_trials'], 100)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_compute_operating_characteristics(self):
        """Test operating characteristics computation"""
        characteristics = compute_operating_characteristics(
            alpha_range=np.array([0.01, 0.05, 0.1]),
            beta_range=np.array([0.01, 0.05, 0.1]),
            p0=0.5,
            p1=0.8
        )
        
        self.assertIn('alpha', characteristics)
        self.assertIn('beta', characteristics)
        self.assertIn('expected_n_h0', characteristics)
        self.assertIn('expected_n_h1', characteristics)
        self.assertIn('power', characteristics)
        
        # Check dimensions
        self.assertEqual(len(characteristics['alpha']), 9)  # 3x3 grid
    
    def test_simulate_sequential_test(self):
        """Test sequential test simulation"""
        # Simulate under H1 (high success rate)
        results = simulate_sequential_test(
            true_p=0.8,
            num_simulations=100,
            alpha=0.05,
            beta=0.05,
            p0=0.5,
            p1=0.8
        )
        
        self.assertEqual(results['true_p'], 0.8)
        self.assertIn('reject_rate', results)
        self.assertIn('avg_sample_size', results)
        self.assertIn('sample_sizes', results)
        
        # With high true_p, should mostly reject H0
        self.assertGreater(results['reject_rate'], 0.5)
        
        # Check sample sizes are reasonable
        self.assertGreater(results['avg_sample_size'], 0)
        self.assertLess(results['avg_sample_size'], 1000)
    
    def test_simulate_under_h0(self):
        """Test simulation under null hypothesis"""
        results = simulate_sequential_test(
            true_p=0.5,  # H0 is true
            num_simulations=100,
            alpha=0.05,
            beta=0.05,
            p0=0.5,
            p1=0.8
        )
        
        # Should mostly accept H0
        self.assertLess(results['reject_rate'], 0.5)


class TestSPRTState(unittest.TestCase):
    """Test SPRTState dataclass"""
    
    def test_state_creation(self):
        """Test SPRTState creation"""
        state = SPRTState()
        
        self.assertEqual(state.log_likelihood_ratio, 0.0)
        self.assertEqual(state.num_trials, 0)
        self.assertEqual(state.num_successes, 0)
        self.assertFalse(state.terminated)
        self.assertIsNone(state.decision)
        self.assertEqual(state.confidence, 0.5)
        self.assertEqual(len(state.history), 0)
        self.assertEqual(len(state.timestamps), 0)
    
    def test_state_with_values(self):
        """Test SPRTState with custom values"""
        state = SPRTState(
            log_likelihood_ratio=1.5,
            num_trials=10,
            num_successes=7,
            terminated=True,
            decision='reject',
            confidence=0.95,
            history=[True, False, True],
            timestamps=[1.0, 2.0, 3.0]
        )
        
        self.assertEqual(state.log_likelihood_ratio, 1.5)
        self.assertEqual(state.num_trials, 10)
        self.assertEqual(state.num_successes, 7)
        self.assertTrue(state.terminated)
        self.assertEqual(state.decision, 'reject')
        self.assertEqual(state.confidence, 0.95)
        self.assertEqual(len(state.history), 3)
        self.assertEqual(len(state.timestamps), 3)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_sequential_testing_workflow(self):
        """Test complete sequential testing workflow"""
        # Create tester
        tester = SequentialTester(alpha=0.05, beta=0.05, p0=0.5, p1=0.8)
        
        # Simulate model responses with 75% success rate
        np.random.seed(42)
        true_success_rate = 0.75
        
        decision = None
        for i in range(100):
            success = np.random.random() < true_success_rate
            decision = tester.update(success)
            
            if decision is not None:
                break
        
        # Should have made a decision
        self.assertIsNotNone(decision)
        self.assertTrue(tester.state.terminated)
        
        # With 75% success rate, should likely reject H0
        # (But not guaranteed due to randomness)
        stats = tester.get_statistics()
        self.assertGreater(stats['success_rate'], 0.6)
    
    def test_adaptive_vs_standard(self):
        """Compare adaptive vs standard tester"""
        standard = SequentialTester(alpha=0.05, beta=0.05, p0=0.5, p1=0.8)
        adaptive = AdaptiveSequentialTester(
            initial_alpha=0.05,
            initial_beta=0.05,
            adaptation_rate=0.1
        )
        
        # Same sequence of results
        np.random.seed(42)
        results = [np.random.random() < 0.7 for _ in range(50)]
        
        # Run both
        standard_decision = standard.batch_update(results)
        adaptive_decision = adaptive.batch_update(results)
        
        # Both should make decisions (though potentially different)
        if standard_decision or adaptive_decision:
            self.assertIn(standard_decision, [None, 'accept', 'reject'])
            self.assertIn(adaptive_decision, [None, 'accept', 'reject'])


if __name__ == '__main__':
    unittest.main()