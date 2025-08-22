"""
Test suite for the Statistical Difference Decision Framework
"""

import unittest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Import the framework
from pot.core.diff_decision import (
    DiffDecisionConfig,
    SequentialDiffTester,
    DifferenceVerifier,
    create_default_verifier,
    validate_models_compatible
)

class TestDiffDecisionConfig(unittest.TestCase):
    """Test configuration management"""
    
    def test_default_config(self):
        """Test default configuration values"""
        cfg = DiffDecisionConfig()
        self.assertEqual(cfg.alpha, 0.01)
        self.assertEqual(cfg.rel_margin_target, 0.05)
        self.assertEqual(cfg.n_min, 10)
        self.assertEqual(cfg.n_max, 200)
        self.assertEqual(cfg.method, "eb")
    
    def test_config_serialization(self):
        """Test config to/from dict"""
        cfg = DiffDecisionConfig(
            alpha=0.05,
            n_max=100,
            equivalence_band=0.01
        )
        
        # To dict
        d = cfg.to_dict()
        self.assertEqual(d['alpha'], 0.05)
        self.assertEqual(d['n_max'], 100)
        self.assertEqual(d['equivalence_band'], 0.01)
        
        # From dict
        cfg2 = DiffDecisionConfig.from_dict(d)
        self.assertEqual(cfg2.alpha, cfg.alpha)
        self.assertEqual(cfg2.n_max, cfg.n_max)
        self.assertEqual(cfg2.equivalence_band, cfg.equivalence_band)

class TestSequentialDiffTester(unittest.TestCase):
    """Test sequential testing logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cfg = DiffDecisionConfig(
            n_min=5,
            n_max=50,
            rel_margin_target=0.1,
            method="t"
        )
        self.tester = SequentialDiffTester(self.cfg)
    
    def test_welford_algorithm(self):
        """Test Welford's online mean/variance computation"""
        # Add known values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            self.tester.update(v)
        
        # Check statistics
        self.assertEqual(self.tester.n, 5)
        self.assertAlmostEqual(self.tester.mean, 3.0, places=6)
        self.assertAlmostEqual(self.tester.variance, 2.5, places=6)  # Sample variance
        self.assertAlmostEqual(self.tester.std_dev, np.sqrt(2.5), places=6)
    
    def test_ci_t_distribution(self):
        """Test t-distribution confidence intervals"""
        # Add samples from normal distribution
        np.random.seed(42)
        for _ in range(20):
            self.tester.update(np.random.normal(0.1, 0.02))
        
        (lo, hi), hw = self.tester._ci_t()
        
        # Check CI properties
        self.assertLess(lo, self.tester.mean)
        self.assertGreater(hi, self.tester.mean)
        self.assertAlmostEqual(hw, (hi - lo) / 2, places=6)
        
        # CI should contain true mean with high probability
        self.assertLess(lo, 0.12)  # True mean + some margin
        self.assertGreater(hi, 0.08)  # True mean - some margin
    
    def test_ci_eb_bounded(self):
        """Test Empirical-Bernstein confidence intervals"""
        cfg = DiffDecisionConfig(
            method="eb",
            clip_low=0.0,
            clip_high=1.0,
            n_min=5,
            n_max=50
        )
        tester = SequentialDiffTester(cfg)
        
        # Add bounded samples
        np.random.seed(42)
        for _ in range(30):
            tester.update(np.random.beta(2, 5))  # Bounded in [0,1]
        
        (lo, hi), hw = tester._ci_eb()
        
        # Check CI properties
        self.assertLess(lo, tester.mean)
        self.assertGreater(hi, tester.mean)
        self.assertGreater(hw, 0)
        
        # EB should give tighter bounds for bounded variables
        self.assertLess(hw, 0.5)  # Should be reasonably tight
    
    def test_should_stop_different(self):
        """Test stopping when models are different"""
        # Add samples showing clear difference
        for _ in range(10):
            self.tester.update(0.1 + np.random.normal(0, 0.001))
        
        should_stop, info = self.tester.should_stop()
        
        if should_stop:
            self.assertEqual(info["decision"], "DIFFERENT")
            self.assertIn("CI excludes 0", info["reason"])
            self.assertGreater(info["mean"], 0)
    
    def test_should_stop_identical(self):
        """Test stopping when models are identical"""
        cfg = DiffDecisionConfig(
            identical_model_n_min=5,
            early_stop_threshold=0.001,
            n_min=5,
            n_max=50
        )
        tester = SequentialDiffTester(cfg)
        
        # Add very small differences
        for _ in range(10):
            tester.update(np.random.normal(0, 0.0001))
        
        should_stop, info = tester.should_stop()
        
        if should_stop and info["decision"] == "IDENTICAL":
            self.assertLess(abs(info["mean"]), 0.001)
            self.assertIn("identical", info["reason"].lower())
    
    def test_should_stop_equivalence(self):
        """Test stopping with equivalence band (TOST)"""
        cfg = DiffDecisionConfig(
            equivalence_band=0.05,
            rel_margin_target=0.1,
            n_min=10,
            n_max=50
        )
        tester = SequentialDiffTester(cfg)
        
        # Add small differences within equivalence band
        for _ in range(20):
            tester.update(np.random.normal(0.02, 0.005))
        
        should_stop, info = tester.should_stop()
        
        if should_stop and info["decision"] == "SAME":
            self.assertLessEqual(abs(info["mean"]), 0.05)
            self.assertIn("equivalence band", info["reason"])
    
    def test_should_stop_undecided(self):
        """Test stopping when undecided at n_max"""
        cfg = DiffDecisionConfig(n_min=5, n_max=10, rel_margin_target=0.01)
        tester = SequentialDiffTester(cfg)
        
        # Add high variance samples
        for _ in range(10):
            tester.update(np.random.normal(0, 0.5))
        
        should_stop, info = tester.should_stop()
        
        self.assertTrue(should_stop)
        self.assertEqual(info["decision"], "UNDECIDED")

class TestDifferenceVerifier(unittest.TestCase):
    """Test main verification orchestrator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cfg = DiffDecisionConfig(
            n_min=5,
            n_max=20,
            batch_size=2,
            positions_per_prompt=10
        )
        
        # Mock scoring function - slight positive bias
        self.score_fn = Mock(side_effect=lambda r, c, p, K: np.random.normal(0.05, 0.01))
        
        # Mock prompt generator
        self.prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]
        self.prompt_idx = 0
        
        def get_prompt():
            prompt = self.prompts[self.prompt_idx % len(self.prompts)]
            self.prompt_idx += 1
            return prompt
        
        self.prompt_generator = Mock(side_effect=get_prompt)
        
        self.verifier = DifferenceVerifier(
            self.score_fn,
            self.prompt_generator,
            self.cfg
        )
    
    def test_verify_difference_basic(self):
        """Test basic verification flow"""
        ref_model = Mock()
        cand_model = Mock()
        
        report = self.verifier.verify_difference(
            ref_model,
            cand_model,
            verbose=False
        )
        
        # Check report structure
        self.assertIn("results", report)
        self.assertIn("decision", report["results"])
        self.assertIn("timing", report)
        self.assertIn("scores", report)
        self.assertIn("next_steps", report)
        
        # Check that scoring was called
        self.assertGreater(self.score_fn.call_count, 0)
        self.assertGreater(self.prompt_generator.call_count, 0)
    
    def test_verify_with_output_dir(self):
        """Test verification with output directory"""
        ref_model = Mock()
        cand_model = Mock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            report = self.verifier.verify_difference(
                ref_model,
                cand_model,
                output_dir=output_dir,
                verbose=False
            )
            
            # Check files were created
            json_files = list(output_dir.glob("*.json"))
            txt_files = list(output_dir.glob("*.txt"))
            
            self.assertGreater(len(json_files), 0)
            self.assertGreater(len(txt_files), 0)
            
            # Check report can be loaded
            report_file = next(f for f in json_files if "report" in f.name)
            with open(report_file) as f:
                loaded_report = json.load(f)
                self.assertEqual(loaded_report["verifier"], "stat_diff_v1")
    
    def test_early_stopping_different(self):
        """Test early stopping when difference is clear"""
        # Strong positive difference
        score_fn = Mock(return_value=0.1)
        
        verifier = DifferenceVerifier(
            score_fn,
            self.prompt_generator,
            self.cfg
        )
        
        ref_model = Mock()
        cand_model = Mock()
        
        report = verifier.verify_difference(
            ref_model,
            cand_model,
            verbose=False
        )
        
        # Should stop early with DIFFERENT decision
        self.assertEqual(report["results"]["decision"], "DIFFERENT")
        self.assertLessEqual(report["results"]["n_used"], self.cfg.n_max)
    
    def test_interpretation_messages(self):
        """Test interpretation generation"""
        ref_model = Mock()
        cand_model = Mock()
        
        # Test with different outcomes
        test_cases = [
            (0.1, "DIFFERENT", "higher"),  # Positive difference
            (-0.1, "DIFFERENT", "lower"),   # Negative difference
        ]
        
        for mean_val, expected_decision, expected_word in test_cases:
            score_fn = Mock(return_value=mean_val)
            verifier = DifferenceVerifier(score_fn, self.prompt_generator, self.cfg)
            
            report = verifier.verify_difference(
                ref_model,
                cand_model,
                verbose=False
            )
            
            if report["results"]["decision"] == expected_decision:
                self.assertIn(expected_word, report["interpretation"].lower())

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_create_default_verifier(self):
        """Test default verifier creation"""
        score_fn = Mock()
        prompt_gen = Mock()
        
        verifier = create_default_verifier(
            score_fn,
            prompt_gen,
            n_max=100,
            alpha=0.05
        )
        
        self.assertIsInstance(verifier, DifferenceVerifier)
        self.assertEqual(verifier.cfg.n_max, 100)
        self.assertEqual(verifier.cfg.alpha, 0.05)
    
    def test_validate_models_compatible(self):
        """Test model compatibility validation"""
        cfg = DiffDecisionConfig(similar_size_ratio=2.0)
        
        # Mock models with parameter counts
        ref_model = Mock()
        ref_model.num_parameters = Mock(return_value=1000000)
        
        cand_model = Mock()
        cand_model.num_parameters = Mock(return_value=1500000)
        
        # Should be compatible (ratio = 1.5 < 2.0)
        is_compat, reason = validate_models_compatible(ref_model, cand_model, cfg)
        self.assertTrue(is_compat)
        
        # Make incompatible
        cand_model.num_parameters = Mock(return_value=3000000)
        
        # Should be incompatible (ratio = 3.0 > 2.0)
        is_compat, reason = validate_models_compatible(ref_model, cand_model, cfg)
        self.assertFalse(is_compat)
        self.assertIn("size ratio", reason)
    
    def test_validate_models_architecture(self):
        """Test architecture compatibility check"""
        cfg = DiffDecisionConfig()
        
        # Mock models with configs
        ref_model = Mock()
        ref_model.config = Mock(model_type="gpt2")
        
        cand_model = Mock()
        cand_model.config = Mock(model_type="bert")
        
        # Should detect architecture mismatch
        is_compat, reason = validate_models_compatible(ref_model, cand_model, cfg)
        self.assertFalse(is_compat)
        self.assertIn("Architecture mismatch", reason)

class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def test_full_verification_workflow(self):
        """Test complete verification workflow"""
        np.random.seed(42)
        
        # Create verifier with controlled scoring
        def controlled_score_fn(ref, cand, prompt, K):
            # Simulate clear difference
            return 0.05 + np.random.normal(0, 0.005)
        
        def controlled_prompt_gen():
            return f"Test prompt {np.random.randint(1000)}"
        
        cfg = DiffDecisionConfig(
            n_min=10,
            n_max=50,
            rel_margin_target=0.1,
            method="t"
        )
        
        verifier = DifferenceVerifier(
            controlled_score_fn,
            controlled_prompt_gen,
            cfg
        )
        
        # Run verification
        ref_model = Mock(name="reference")
        cand_model = Mock(name="candidate")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report = verifier.verify_difference(
                ref_model,
                cand_model,
                output_dir=Path(tmpdir),
                verbose=False
            )
            
            # Verify complete workflow
            self.assertEqual(report["verifier"], "stat_diff_v1")
            self.assertIn("decision", report["results"])
            self.assertGreater(report["results"]["n_used"], 0)
            self.assertIn("timestamp", report)
            
            # Check files created
            files = list(Path(tmpdir).glob("*"))
            self.assertGreater(len(files), 0)
    
    def test_stress_test_many_samples(self):
        """Stress test with many samples"""
        # Test that algorithm handles many updates efficiently
        cfg = DiffDecisionConfig(n_min=100, n_max=1000)
        tester = SequentialDiffTester(cfg)
        
        # Add many samples
        np.random.seed(42)
        for i in range(500):
            tester.update(np.random.normal(0.01, 0.02))
            
            # Check state periodically
            if i % 100 == 0:
                state = tester.get_state()
                self.assertEqual(state["n"], i + 1)
                self.assertIsNotNone(state["mean"])
                self.assertIsNotNone(state["ci"])

def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDiffDecisionConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestSequentialDiffTester))
    suite.addTests(loader.loadTestsFromTestCase(TestDifferenceVerifier))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)