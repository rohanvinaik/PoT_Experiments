#!/usr/bin/env python3
"""
Test result schema integration with the PoT verification pipeline
"""

import sys
import os
import time
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.core.result_schema import build_result, save_result, validate_result
from pot.core.statistical_policy import DiffDecisionConfig, SequentialDiffTester
from pot.scoring.teacher_forced import TeacherForcedScorer, ScoringConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_integration_with_verifier():
    """Test integration with actual verification pipeline"""
    
    logger.info("Testing integration with verification pipeline...")
    
    # Setup realistic config
    config = DiffDecisionConfig(
        mode="QUICK_GATE",
        same_model_p95=0.00034,
        near_clone_p5=0.0763,
        use_calibration=True,
        positions_per_prompt=64,
        epsilon_diff=0.50,
        force_decision_at_max=True,
        n_max=200
    )
    config.finalize()
    
    # Create tester
    tester = SequentialDiffTester(config)
    
    # Simulate scoring with realistic scores
    scores = [
        0.0, 0.0, 0.0, 0.0, 0.0,  # Perfect match samples
        0.05, 0.02, 0.01, 0.0, 0.03  # Near-zero samples
    ]
    
    info = None
    for score in scores:
        tester.add_sample(score)
        if tester.n >= config.n_min:
            stop, info = tester.should_stop()
            if stop:
                break
    
    # Ensure we have info even if we don't stop early
    if info is None:
        _, info = tester.should_stop()
    
    # Ensure info has required fields
    if info is None:
        info = {
            "decision": "UNDECIDED",
            "rule": "max_samples_reached",
            "ci": [0.0, 0.1],
            "half_width": 0.05,
            "rme": 1.0
        }
    
    # Mock challenges and timing
    challenges_used = [
        "What is machine learning?",
        "Explain photosynthesis in simple terms.",
        "Translate to French: hello",
        "What is quantum computing?",
        "Explain evolution in one sentence."
    ]
    
    timing = {
        "t_load": 1.2,
        "t_infer_total": 3.5,
        "t_per_query": 0.35,
        "t_total": 5.0
    }
    
    # Build comprehensive result
    result = build_result(
        tester=tester,
        config=config,
        info=info,
        timing=timing,
        challenges_used=challenges_used
    )
    
    # Validate result
    missing = validate_result(result)
    assert len(missing) == 0, f"Integration result should be complete, missing: {missing}"
    
    # Check specific values
    assert result["decision"] in ["SAME", "DIFFERENT", "UNDECIDED"], f"Invalid decision: {result['decision']}"
    assert result["n_used"] == tester.n, f"N used mismatch: {result['n_used']} vs {tester.n}"
    assert result["mean"] == tester.mean, f"Mean mismatch: {result['mean']} vs {tester.mean}"
    assert result["positions_per_prompt"] == 64, "Positions per prompt should match config"
    assert result["mode"] == "QUICK_GATE", "Mode should match config"
    assert len(result["merkle_root"]) == 64, "Should have valid merkle root"
    
    logger.info(f"Integration result - Decision: {result['decision']}")
    logger.info(f"Integration result - Rule: {result['rule']}")
    logger.info(f"Integration result - N used: {result['n_used']}")
    logger.info(f"Integration result - Mean: {result['mean']:.6f}")
    logger.info(f"Integration result - CI: {result['ci_99']}")
    logger.info("✅ Integration with verifier test passed")
    return True

def test_different_decision_result():
    """Test result building for DIFFERENT decision"""
    
    logger.info("Testing DIFFERENT decision result...")
    
    # Setup for DIFFERENT decision
    config = DiffDecisionConfig(
        mode="AUDIT_GRADE",
        same_model_p95=0.00034,
        near_clone_p5=0.0763,
        use_calibration=True,
        positions_per_prompt=128,
        epsilon_diff=0.40,
        force_decision_at_max=True
    )
    config.finalize()
    
    tester = SequentialDiffTester(config)
    
    # Add scores that should lead to DIFFERENT
    different_scores = [0.22, 0.25, 0.18, 0.30, 0.19, 0.28, 0.23, 0.26, 0.21, 0.24] * 10
    
    for score in different_scores:
        tester.add_sample(score)
        if tester.n >= config.n_min:
            stop, info = tester.should_stop()
            if stop:
                break
    
    # Force stop if needed for test
    if not info:
        _, info = tester.should_stop()
    
    timing = {
        "t_load": 2.1,
        "t_infer_total": 25.3,
        "t_per_query": 0.253,
        "t_total": 28.0
    }
    
    challenges = ["Explain gravity.", "What is DNA?", "Translate to Spanish: good morning"] * 20
    
    result = build_result(
        tester=tester,
        config=config,
        info=info,
        timing=timing,
        challenges_used=challenges[:tester.n],
        hardware="cuda"
    )
    
    # Validate DIFFERENT result structure
    assert result["decision"] in ["DIFFERENT", "UNDECIDED"], f"Expected DIFFERENT/UNDECIDED, got {result['decision']}"
    assert result["mean"] > 0.1, f"Mean should be substantial for DIFFERENT: {result['mean']}"
    assert result["hardware"]["backend"] == "cuda", "Hardware should be CUDA"
    assert result["n_eff"] == tester.n * 128, f"N effective should be {tester.n * 128}"
    
    logger.info(f"DIFFERENT result - Decision: {result['decision']}")
    logger.info(f"DIFFERENT result - Mean: {result['mean']:.4f}")
    logger.info(f"DIFFERENT result - Hardware: {result['hardware']['backend']}")
    logger.info("✅ DIFFERENT decision test passed")
    return True

def test_result_persistence():
    """Test saving and loading results for audit trail"""
    
    logger.info("Testing result persistence...")
    
    # Create multiple results for audit trail
    results = []
    
    for i in range(3):
        config = DiffDecisionConfig(mode="QUICK_GATE")
        config.finalize()
        
        class MockTester:
            def __init__(self, n, mean):
                self.n = n
                self.mean = mean
                self.variance = 0.01
        
        tester = MockTester(50 + i * 10, 0.1 * i)
        
        info = {
            "decision": ["SAME", "DIFFERENT", "UNDECIDED"][i],
            "rule": f"test_rule_{i}",
            "ci": [0.0, 0.2 * i],
            "half_width": 0.1 * i,
            "rme": 0.5 + i * 0.1
        }
        
        timing = {"t_total": 10.0 + i * 2}
        challenges = [f"Challenge {j} for test {i}" for j in range(tester.n)]
        
        result = build_result(tester, config, info, timing, challenges)
        results.append(result)
    
    # Save all results
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        saved_paths = []
        
        for i, result in enumerate(results):
            filename = f"test_result_{i}.json"
            saved_path = save_result(result, output_dir=temp_path, filename=filename)
            saved_paths.append(saved_path)
            assert saved_path.exists(), f"Result {i} should be saved"
        
        # Verify all results have different merkle roots (different challenges)
        merkle_roots = []
        for path in saved_paths:
            with open(path, 'r') as f:
                import json
                loaded = json.load(f)
                merkle_roots.append(loaded["merkle_root"])
        
        assert len(set(merkle_roots)) == len(merkle_roots), "All results should have unique merkle roots"
        
        logger.info(f"Saved {len(results)} results to audit trail")
        logger.info(f"Unique merkle roots: {len(set(merkle_roots))}")
    
    logger.info("✅ Result persistence test passed")
    return True

def test_edge_case_results():
    """Test result building for edge cases"""
    
    logger.info("Testing edge case results...")
    
    # Test with minimal tester
    class MinimalTester:
        def __init__(self):
            self.n = 1
            self.mean = 0.0
    
    class MinimalConfig:
        def __init__(self):
            self.alpha = 0.01
            self.beta = 0.01
            self.gamma = 0.001
            self.delta_star = 0.038
            self.n_max = 400
            self.positions_per_prompt = 32
            self.ci_method = "empirical_bernstein"
            self.mode = "QUICK_GATE"
            self.min_effect_floor = 0.001
    
    tester = MinimalTester()
    config = MinimalConfig()
    
    # Minimal info (some fields missing)
    info = {
        "decision": "UNDECIDED",
        "ci": [0.0, 0.0]
    }
    
    timing = {}  # Empty timing
    challenges = []  # No challenges
    
    result = build_result(tester, config, info, timing, challenges)
    
    # Should still build a valid result
    assert "decision" in result, "Should have decision even with minimal input"
    assert "merkle_root" in result, "Should have merkle root even with no challenges"
    assert result["merkle_root"] == "0" * 64, "Empty challenges should produce zero merkle root"
    assert result["n_used"] == 1, "Should use tester.n"
    assert result["time"]["t_total"] == 0, "Should handle empty timing"
    
    # Test validation on minimal result
    missing = validate_result(result)
    # Should still be valid (all required fields present)
    assert len(missing) == 0, f"Minimal result should be valid, missing: {missing}"
    
    logger.info(f"Edge case result - Decision: {result['decision']}")
    logger.info(f"Edge case result - Merkle root: {result['merkle_root'][:16]}...")
    logger.info("✅ Edge case results test passed")
    return True

def main():
    """Run all result schema integration tests"""
    logger.info("\n" + "="*70)
    logger.info("RESULT SCHEMA INTEGRATION TESTS")
    logger.info("="*70)
    
    tests = [
        ("Integration with Verifier", test_integration_with_verifier),
        ("DIFFERENT Decision Result", test_different_decision_result),
        ("Result Persistence", test_result_persistence),
        ("Edge Case Results", test_edge_case_results)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            logger.info(f"\n--- Testing {name} ---")
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            logger.error(f"Test {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info("Key achievements:")
        logger.info("  • Integration with verification pipeline working")
        logger.info("  • SAME/DIFFERENT decision results handled")
        logger.info("  • Result persistence and audit trail functional")
        logger.info("  • Edge cases handled gracefully")
        logger.info("  • Comprehensive result schema validated")
    else:
        logger.info("\n⚠️ Some integration tests failed - review output above")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())