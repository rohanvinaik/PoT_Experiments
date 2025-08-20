#!/usr/bin/env python3
"""
Test the comprehensive result schema implementation
"""

import sys
import os
import tempfile
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.core.result_schema import generate_merkle_root, build_result, save_result, validate_result
from pot.core.statistical_policy import DiffDecisionConfig, SequentialDiffTester
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_merkle_root_generation():
    """Test Merkle root generation for challenges"""
    
    logger.info("Testing Merkle root generation...")
    
    # Test empty challenges
    empty_root = generate_merkle_root([])
    assert empty_root == "0" * 64, f"Empty root should be all zeros, got: {empty_root}"
    
    # Test single challenge
    single_challenge = ["What is machine learning?"]
    single_root = generate_merkle_root(single_challenge)
    assert len(single_root) == 64, f"Root should be 64 chars, got {len(single_root)}"
    assert all(c in "0123456789abcdef" for c in single_root), "Root should be hex"
    
    # Test multiple challenges
    challenges = [
        "What is machine learning?",
        "Explain photosynthesis.",
        "What is gravity?"
    ]
    multi_root = generate_merkle_root(challenges)
    assert len(multi_root) == 64, f"Multi-challenge root should be 64 chars, got {len(multi_root)}"
    
    # Test determinism
    multi_root2 = generate_merkle_root(challenges)
    assert multi_root == multi_root2, "Merkle root generation should be deterministic"
    
    # Test different challenges produce different roots
    different_challenges = [
        "What is quantum computing?",
        "Explain evolution.",
        "What is democracy?"
    ]
    different_root = generate_merkle_root(different_challenges)
    assert different_root != multi_root, "Different challenges should produce different roots"
    
    logger.info(f"Single challenge root: {single_root[:16]}...")
    logger.info(f"Multi challenge root: {multi_root[:16]}...")
    logger.info("✅ Merkle root generation test passed")
    return True

def test_result_building():
    """Test comprehensive result building"""
    
    logger.info("Testing result building...")
    
    # Create mock tester and config
    config = DiffDecisionConfig(
        mode="AUDIT_GRADE",
        same_model_p95=0.001,
        near_clone_p5=0.08,
        use_calibration=True,
        positions_per_prompt=128
    )
    config.finalize()
    
    # Mock tester
    class MockTester:
        def __init__(self):
            self.n = 150
            self.mean = 0.235
            self.variance = 0.045
    
    tester = MockTester()
    
    # Mock info from decision
    info = {
        "decision": "DIFFERENT",
        "rule": "min_effect_forced",
        "ci": [0.1234, 0.3456],
        "half_width": 0.1111,
        "rme": 0.472,
        "n_used": 150
    }
    
    # Mock timing
    timing = {
        "t_load": 2.5,
        "t_infer_total": 12.3,
        "t_per_query": 0.082,
        "t_total": 15.2
    }
    
    # Mock challenges
    challenges_used = [
        "What is machine learning?",
        "Explain photosynthesis in simple terms.",
        "Translate to French: hello"
    ]
    
    # Build result
    result = build_result(
        tester=tester,
        config=config,
        info=info,
        timing=timing,
        challenges_used=challenges_used,
        hardware="cpu"
    )
    
    # Validate structure
    assert "decision" in result, "Result missing 'decision'"
    assert "rule" in result, "Result missing 'rule'"
    assert "alpha" in result, "Result missing 'alpha'"
    assert "beta" in result, "Result missing 'beta'"
    assert "gamma" in result, "Result missing 'gamma'"
    assert "delta_star" in result, "Result missing 'delta_star'"
    assert "n_used" in result, "Result missing 'n_used'"
    assert "n_max" in result, "Result missing 'n_max'"
    assert "n_eff" in result, "Result missing 'n_eff'"
    assert "mean" in result, "Result missing 'mean'"
    assert "variance" in result, "Result missing 'variance'"
    assert "ci_99" in result, "Result missing 'ci_99'"
    assert "half_width" in result, "Result missing 'half_width'"
    assert "rme" in result, "Result missing 'rme'"
    assert "positions_per_prompt" in result, "Result missing 'positions_per_prompt'"
    assert "ci_method" in result, "Result missing 'ci_method'"
    assert "mode" in result, "Result missing 'mode'"
    assert "time" in result, "Result missing 'time'"
    assert "hardware" in result, "Result missing 'hardware'"
    assert "challenge_namespace" in result, "Result missing 'challenge_namespace'"
    assert "merkle_root" in result, "Result missing 'merkle_root'"
    assert "timestamp" in result, "Result missing 'timestamp'"
    assert "version" in result, "Result missing 'version'"
    
    # Validate values
    assert result["decision"] == "DIFFERENT", f"Expected DIFFERENT, got {result['decision']}"
    assert result["rule"] == "min_effect_forced", f"Expected min_effect_forced, got {result['rule']}"
    assert result["n_used"] == 150, f"Expected 150, got {result['n_used']}"
    assert result["n_eff"] == 150 * 128, f"Expected {150 * 128}, got {result['n_eff']}"
    assert abs(result["mean"] - 0.235) < 0.001, f"Expected ~0.235, got {result['mean']}"
    assert result["ci_99"] == [0.1234, 0.3456], f"Expected [0.1234, 0.3456], got {result['ci_99']}"
    assert result["hardware"]["backend"] == "cpu", f"Expected cpu, got {result['hardware']['backend']}"
    assert len(result["merkle_root"]) == 64, f"Expected 64-char merkle root, got {len(result['merkle_root'])}"
    
    logger.info(f"Built result with decision: {result['decision']}")
    logger.info(f"N effective: {result['n_eff']}")
    logger.info(f"Mean: {result['mean']:.4f}")
    logger.info(f"Merkle root: {result['merkle_root'][:16]}...")
    logger.info("✅ Result building test passed")
    return True

def test_result_validation():
    """Test result validation"""
    
    logger.info("Testing result validation...")
    
    # Complete valid result
    valid_result = {
        "decision": "SAME",
        "rule": "identical_early_stop",
        "alpha": 0.01,
        "beta": 0.01,
        "n_used": 50,
        "mean": 0.0,
        "ci_99": [0.0, 0.0],
        "half_width": 0.0,
        "rme": 0.0,
        "positions_per_prompt": 64,
        "time": {"t_load": 1.0, "t_infer_total": 5.0},
        "hardware": {"backend": "cpu"},
        "merkle_root": "a" * 64
    }
    
    missing = validate_result(valid_result)
    assert len(missing) == 0, f"Valid result should have no missing fields, got: {missing}"
    
    # Incomplete result
    incomplete_result = {
        "decision": "DIFFERENT",
        "n_used": 100,
        "mean": 0.5
        # Missing many required fields
    }
    
    missing = validate_result(incomplete_result)
    assert len(missing) > 0, "Incomplete result should have missing fields"
    
    expected_missing = ["rule", "alpha", "beta", "ci_99", "half_width", "rme", 
                       "positions_per_prompt", "time", "hardware", "merkle_root"]
    for field in expected_missing:
        assert field in missing, f"Expected missing field '{field}' not found in: {missing}"
    
    logger.info(f"Valid result: {len(missing)} missing fields")
    logger.info(f"Incomplete result: {len(missing)} missing fields")
    logger.info("✅ Result validation test passed")
    return True

def test_result_saving():
    """Test result saving and loading"""
    
    logger.info("Testing result saving...")
    
    # Create a test result
    test_result = {
        "decision": "DIFFERENT",
        "rule": "min_effect_achieved",
        "alpha": 0.01,
        "beta": 0.01,
        "gamma": 0.001,
        "delta_star": 0.038,
        "n_used": 75,
        "n_max": 400,
        "n_eff": 9600,
        "mean": 0.123,
        "variance": 0.045,
        "ci_99": [0.05, 0.195],
        "half_width": 0.0725,
        "rme": 0.589,
        "positions_per_prompt": 128,
        "ci_method": "empirical_bernstein",
        "mode": "AUDIT_GRADE",
        "time": {
            "t_load": 3.2,
            "t_infer_total": 18.7,
            "t_per_query": 0.249,
            "t_total": 22.1
        },
        "hardware": {
            "backend": "mps",
            "device": "mps"
        },
        "challenge_namespace": "test_schema",
        "merkle_root": "abc123" + "0" * 58,
        "timestamp": "20250820_123456",
        "version": "1.0.0"
    }
    
    # Test saving to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save with auto-generated filename
        saved_path = save_result(test_result, output_dir=temp_path)
        assert saved_path.exists(), f"Saved file should exist: {saved_path}"
        assert saved_path.suffix == ".json", f"Should be JSON file: {saved_path}"
        
        # Load and verify
        with open(saved_path, 'r') as f:
            loaded_result = json.load(f)
        
        assert loaded_result["decision"] == test_result["decision"], "Decision should match"
        assert loaded_result["n_used"] == test_result["n_used"], "N used should match"
        assert abs(loaded_result["mean"] - test_result["mean"]) < 0.001, "Mean should match"
        assert loaded_result["merkle_root"] == test_result["merkle_root"], "Merkle root should match"
        
        # Save with custom filename
        custom_filename = "custom_test_result.json"
        custom_path = save_result(test_result, output_dir=temp_path, filename=custom_filename)
        assert custom_path.name == custom_filename, f"Should use custom filename: {custom_path.name}"
        assert custom_path.exists(), f"Custom file should exist: {custom_path}"
        
        logger.info(f"Saved to: {saved_path}")
        logger.info(f"Custom saved to: {custom_path}")
    
    logger.info("✅ Result saving test passed")
    return True

def test_hardware_detection():
    """Test hardware detection in result building"""
    
    logger.info("Testing hardware detection...")
    
    # Simple mock setup for testing
    class MockConfig:
        def __init__(self):
            self.alpha = 0.01
            self.beta = 0.01
            self.gamma = 0.001
            self.delta_star = 0.038
            self.n_max = 400
            self.positions_per_prompt = 64
            self.ci_method = "empirical_bernstein"
            self.mode = "QUICK_GATE"
            self.min_effect_floor = 0.001
            self.namespace = "hw_test"
    
    class MockTester:
        def __init__(self):
            self.n = 100
            self.mean = 0.15
    
    config = MockConfig()
    tester = MockTester()
    info = {"decision": "DIFFERENT", "ci": [0.1, 0.2], "half_width": 0.05}
    timing = {"t_total": 10.0}
    challenges = ["test challenge"]
    
    # Test explicit hardware
    result_cpu = build_result(tester, config, info, timing, challenges, hardware="cpu")
    assert result_cpu["hardware"]["backend"] == "cpu", "Should use explicit CPU"
    
    result_cuda = build_result(tester, config, info, timing, challenges, hardware="cuda")  
    assert result_cuda["hardware"]["backend"] == "cuda", "Should use explicit CUDA"
    
    # Test auto-detection (will detect actual hardware)
    result_auto = build_result(tester, config, info, timing, challenges)
    assert "hardware" in result_auto, "Should have hardware field"
    assert "backend" in result_auto["hardware"], "Should have backend field"
    
    detected_hw = result_auto["hardware"]["backend"]
    logger.info(f"Auto-detected hardware: {detected_hw}")
    assert detected_hw in ["cpu", "cuda", "mps"], f"Should detect valid hardware, got: {detected_hw}"
    
    logger.info("✅ Hardware detection test passed")
    return True

def main():
    """Run all result schema tests"""
    logger.info("\n" + "="*70)
    logger.info("COMPREHENSIVE RESULT SCHEMA TESTS")
    logger.info("="*70)
    
    tests = [
        ("Merkle Root Generation", test_merkle_root_generation),
        ("Result Building", test_result_building),
        ("Result Validation", test_result_validation),
        ("Result Saving", test_result_saving),
        ("Hardware Detection", test_hardware_detection)
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
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL RESULT SCHEMA TESTS PASSED!")
        logger.info("Key achievements:")
        logger.info("  • Comprehensive result structure implemented")
        logger.info("  • Merkle root generation for audit trail")
        logger.info("  • Result validation and saving functional")
        logger.info("  • Hardware detection working")
        logger.info("  • All required fields included")
    else:
        logger.info("\n⚠️ Some tests failed - review output above")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())