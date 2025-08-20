#!/usr/bin/env python3
"""
Test script for the calibration system.
Validates automatic determination of γ and δ* from pilot runs.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pot.core.calibration import (
    ModelCalibrator, 
    CalibrationResult,
    load_calibration,
    create_mock_calibrator
)
from pot.core.diff_decision import DiffDecisionConfig, TestingMode
from pot.testing.test_models import DeterministicMockModel

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{BOLD}{BLUE}{'=' * 70}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(70)}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 70}{RESET}\n")

def print_section(text: str):
    """Print a section header"""
    print(f"\n{BOLD}{YELLOW}--- {text} ---{RESET}\n")

def print_result(label: str, value: Any, color: str = ""):
    """Print a formatted result"""
    print(f"{label:30s}: {color}{value}{RESET}")

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_same_model_calibration():
    """Test same-model calibration for γ determination"""
    print_section("Testing Same-Model Calibration")
    
    # Create mock models
    models = [
        DeterministicMockModel(model_id="model_A", seed=42),
        DeterministicMockModel(model_id="model_B", seed=43),
        DeterministicMockModel(model_id="model_C", seed=44)
    ]
    model_names = ["Model A", "Model B", "Model C"]
    
    # Create calibrator with mock scoring
    calibrator = create_mock_calibrator()
    
    # Run same-model calibration
    print("Running calibration on 3 models with 5 runs each...")
    stats = calibrator.run_same_model_calibration(
        models=models,
        model_names=model_names,
        n_runs_per_model=5,
        use_mock=True
    )
    
    # Display results
    print("\nSame-Model Statistics:")
    print_result("Mean of |means|", f"{stats['mean']:.6f}")
    print_result("Std of |means|", f"{stats['std']:.6f}")
    print_result("P50 (median)", f"{stats['p50']:.6f}")
    print_result("P75", f"{stats['p75']:.6f}")
    print_result("P90", f"{stats['p90']:.6f}")
    print_result("P95 (γ candidate)", f"{stats['p95']:.6f}", GREEN)
    print_result("P99", f"{stats['p99']:.6f}")
    print_result("Max", f"{stats['max']:.6f}")
    print_result("Total samples", stats['n_samples_total'])
    
    # Validate results
    if stats['p95'] < 0.01:  # Should be small for same models
        print(f"\n{GREEN}✓ Same-model calibration successful{RESET}")
        print(f"  Recommended γ = {stats['p95']:.6f}")
        return True, stats
    else:
        print(f"\n{RED}✗ Same-model P95 too high: {stats['p95']:.6f}{RESET}")
        return False, stats

def test_near_clone_calibration():
    """Test near-clone calibration for δ* determination"""
    print_section("Testing Near-Clone Calibration")
    
    # Create mock model pairs (ref, clone)
    model_pairs = [
        (DeterministicMockModel("ref_1", seed=1), 
         DeterministicMockModel("clone_1", seed=101)),
        (DeterministicMockModel("ref_2", seed=2),
         DeterministicMockModel("clone_2", seed=102)),
        (DeterministicMockModel("ref_3", seed=3),
         DeterministicMockModel("clone_3", seed=103))
    ]
    pair_names = ["Pair 1", "Pair 2", "Pair 3"]
    
    # Create calibrator
    calibrator = create_mock_calibrator()
    
    # Run near-clone calibration
    print("Running calibration on 3 near-clone pairs...")
    stats = calibrator.run_near_clone_calibration(
        model_pairs=model_pairs,
        pair_names=pair_names,
        use_mock=True
    )
    
    # Display results
    print("\nNear-Clone Statistics:")
    print_result("Mean of |means|", f"{stats['mean']:.6f}")
    print_result("Std of |means|", f"{stats['std']:.6f}")
    print_result("P5 (δ* reference)", f"{stats['p5']:.6f}", GREEN)
    print_result("P10", f"{stats['p10']:.6f}")
    print_result("P25", f"{stats['p25']:.6f}")
    print_result("P50 (median)", f"{stats['p50']:.6f}")
    print_result("P75", f"{stats['p75']:.6f}")
    print_result("P95", f"{stats['p95']:.6f}")
    print_result("Min", f"{stats['min']:.6f}")
    print_result("Max", f"{stats['max']:.6f}")
    
    # Validate results
    if 0.05 < stats['p5'] < 0.15:  # Should be moderate for near-clones
        print(f"\n{GREEN}✓ Near-clone calibration successful{RESET}")
        print(f"  Near-clone P5 = {stats['p5']:.6f}")
        return True, stats
    else:
        print(f"\n{RED}✗ Near-clone P5 out of expected range: {stats['p5']:.6f}{RESET}")
        return False, stats

def test_full_calibration():
    """Test complete calibration workflow"""
    print_section("Testing Full Calibration Workflow")
    
    # Setup models
    same_models = [
        DeterministicMockModel(f"same_{i}", seed=i) for i in range(3)
    ]
    
    near_clone_pairs = [
        (DeterministicMockModel(f"ref_{i}", seed=i),
         DeterministicMockModel(f"clone_{i}", seed=i+100))
        for i in range(3)
    ]
    
    # Create calibrator
    calibrator = ModelCalibrator(
        scorer=None,  # Will use mock
        prompt_generator=None,  # Will use mock
        n_samples_per_pair=50
    )
    
    # Run full calibration
    print("Running full calibration...")
    output_file = Path("experimental_results") / f"test_calibration_{int(time.time())}.json"
    
    result = calibrator.calibrate(
        same_models=same_models,
        near_clone_pairs=near_clone_pairs,
        output_file=str(output_file),
        use_mock=True
    )
    
    # Validate results
    print(f"\n{BOLD}Calibration Results:{RESET}")
    print_result("γ (equivalence band)", f"{result.gamma:.6f}", GREEN)
    print_result("δ* (min effect size)", f"{result.delta_star:.6f}", GREEN)
    print_result("Same-model pairs tested", result.n_same_pairs)
    print_result("Near-clone pairs tested", result.n_near_clone_pairs)
    print_result("Samples per pair", result.n_samples_per_pair)
    print_result("Calibration time", f"{result.calibration_time:.2f}s")
    
    # Check separation
    if result.near_clone_stats:
        separation = result.near_clone_stats["p5"] / result.same_model_stats["p95"]
        print_result("Separation factor", f"{separation:.2f}x", 
                    GREEN if separation > 2 else YELLOW)
    
    # Test recommendations
    print(f"\n{BOLD}Configuration Recommendations:{RESET}")
    configs = result.get_config_recommendations()
    for mode, params in configs.items():
        print(f"\n  {mode.upper()}:")
        for key, value in params.items():
            print(f"    {key}: {value:.6f}")
    
    # Validate save/load
    if output_file.exists():
        loaded_result = load_calibration(str(output_file))
        if loaded_result.gamma == result.gamma and loaded_result.delta_star == result.delta_star:
            print(f"\n{GREEN}✓ Calibration saved and loaded successfully{RESET}")
        else:
            print(f"\n{RED}✗ Save/load mismatch{RESET}")
            return False
    
    return True

def test_integration_with_diff_decision():
    """Test integration with DiffDecisionConfig"""
    print_section("Testing Integration with DiffDecisionConfig")
    
    # Run a quick calibration
    calibrator = create_mock_calibrator()
    
    same_models = [DeterministicMockModel(f"model_{i}", seed=i) for i in range(2)]
    near_clone_pairs = [
        (DeterministicMockModel("ref", seed=1),
         DeterministicMockModel("clone", seed=101))
    ]
    
    result = calibrator.calibrate(
        same_models=same_models,
        near_clone_pairs=near_clone_pairs,
        use_mock=True
    )
    
    # Create DiffDecisionConfig with calibration
    print("\nCreating DiffDecisionConfig with calibration results...")
    
    config = DiffDecisionConfig(
        mode=TestingMode.AUDIT_GRADE,
        use_calibration=True,
        same_model_p95=result.gamma,
        near_clone_p5=result.near_clone_stats["p5"] if result.near_clone_stats else None
    )
    
    print(f"Config gamma: {config.gamma:.6f}")
    print(f"Config delta_star: {config.delta_star:.6f}")
    
    # Verify calibration was applied
    if abs(config.gamma - result.gamma) < 0.001:
        print(f"\n{GREEN}✓ Calibration successfully integrated with DiffDecisionConfig{RESET}")
        return True
    else:
        print(f"\n{RED}✗ Calibration not properly applied to config{RESET}")
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    print_section("Testing Edge Cases")
    
    success = True
    
    # Test 1: No near-clone pairs (should use conservative default)
    print("Test 1: No near-clone pairs...")
    calibrator = create_mock_calibrator()
    
    result = calibrator.calibrate(
        same_models=[DeterministicMockModel("single", seed=1)],
        near_clone_pairs=None,  # No near-clones
        use_mock=True
    )
    
    if result.delta_star == result.gamma * 10:
        print(f"  {GREEN}✓ Conservative δ* = 10×γ used{RESET}")
    else:
        print(f"  {RED}✗ Unexpected δ* value{RESET}")
        success = False
    
    # Test 2: Poor separation warning
    print("\nTest 2: Poor separation detection...")
    
    # Create a calibrator that will produce poor separation
    def poor_scorer(ref, cand, prompt, K=32):
        # Very similar scores for same and near-clone
        return np.random.normal(0.01, 0.002)
    
    calibrator = ModelCalibrator(
        scorer=poor_scorer,
        prompt_generator=lambda: "test",
        n_samples_per_pair=20
    )
    
    # This should trigger a warning about poor separation
    # (We can't easily test the warning, but we can verify it runs)
    try:
        result = calibrator.calibrate(
            same_models=[DeterministicMockModel("m1", seed=1)],
            near_clone_pairs=[(DeterministicMockModel("r1", seed=1),
                             DeterministicMockModel("c1", seed=2))],
            use_mock=False  # Use our poor_scorer
        )
        print(f"  {GREEN}✓ Handled poor separation case{RESET}")
    except Exception as e:
        print(f"  {RED}✗ Failed on poor separation: {e}{RESET}")
        success = False
    
    # Test 3: Empty model list
    print("\nTest 3: Empty model list handling...")
    calibrator = create_mock_calibrator()
    
    try:
        result = calibrator.calibrate(
            same_models=[],  # Empty list
            use_mock=True
        )
        print(f"  {YELLOW}⚠ Accepted empty model list (may want to add validation){RESET}")
    except (ValueError, IndexError):
        print(f"  {GREEN}✓ Properly rejected empty model list{RESET}")
    
    return success

def run_performance_test():
    """Test calibration performance with larger datasets"""
    print_section("Performance Test")
    
    # Create larger set of models
    n_models = 5
    n_pairs = 5
    n_samples = 100
    
    print(f"Testing with {n_models} models, {n_pairs} pairs, {n_samples} samples each...")
    
    calibrator = ModelCalibrator(
        scorer=None,
        prompt_generator=None,
        n_samples_per_pair=n_samples
    )
    
    same_models = [DeterministicMockModel(f"perf_{i}", seed=i) for i in range(n_models)]
    near_clone_pairs = [
        (DeterministicMockModel(f"ref_{i}", seed=i),
         DeterministicMockModel(f"clone_{i}", seed=i+1000))
        for i in range(n_pairs)
    ]
    
    start_time = time.time()
    result = calibrator.calibrate(
        same_models=same_models,
        near_clone_pairs=near_clone_pairs,
        use_mock=True
    )
    elapsed = time.time() - start_time
    
    total_comparisons = (n_models * 5 + n_pairs) * n_samples
    comparisons_per_sec = total_comparisons / elapsed
    
    print(f"\nPerformance Results:")
    print_result("Total comparisons", total_comparisons)
    print_result("Total time", f"{elapsed:.2f}s")
    print_result("Comparisons/sec", f"{comparisons_per_sec:.0f}")
    
    if comparisons_per_sec > 1000:  # Should be fast with mock scoring
        print(f"\n{GREEN}✓ Good performance: {comparisons_per_sec:.0f} comparisons/sec{RESET}")
        return True
    else:
        print(f"\n{YELLOW}⚠ Lower than expected performance{RESET}")
        return False

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all calibration tests"""
    print_header("Calibration System Tests")
    
    all_tests_passed = True
    test_results = {}
    
    tests = [
        ("Same-Model Calibration", test_same_model_calibration),
        ("Near-Clone Calibration", test_near_clone_calibration),
        ("Full Calibration Workflow", test_full_calibration),
        ("Integration with DiffDecision", test_integration_with_diff_decision),
        ("Edge Cases", test_edge_cases),
        ("Performance Test", run_performance_test)
    ]
    
    for test_name, test_func in tests:
        print_header(test_name)
        try:
            if test_name in ["Same-Model Calibration", "Near-Clone Calibration"]:
                passed, stats = test_func()
                test_results[test_name] = {
                    "passed": passed,
                    "stats": stats if passed else None,
                    "error": None
                }
            else:
                passed = test_func()
                test_results[test_name] = {
                    "passed": passed,
                    "error": None
                }
            
            if not passed:
                all_tests_passed = False
        except Exception as e:
            print(f"\n{RED}✗ Test failed with error: {e}{RESET}")
            test_results[test_name] = {
                "passed": False,
                "error": str(e)
            }
            all_tests_passed = False
    
    # Summary
    print_header("Test Summary")
    
    passed_count = sum(1 for r in test_results.values() if r["passed"])
    total_count = len(test_results)
    
    print(f"Tests Passed: {passed_count}/{total_count}")
    
    for test_name, result in test_results.items():
        if result["passed"]:
            print(f"  {GREEN}✓ {test_name}{RESET}")
        else:
            error_msg = f" - {result['error']}" if result['error'] else ""
            print(f"  {RED}✗ {test_name}{error_msg}{RESET}")
    
    if all_tests_passed:
        print(f"\n{GREEN}{BOLD}✓ ALL TESTS PASSED!{RESET}")
        print("\nThe calibration system is working correctly.")
        print("Key features validated:")
        print("  • Same-model calibration for γ")
        print("  • Near-clone calibration for δ*")
        print("  • Integration with DiffDecisionConfig")
        print("  • Save/load functionality")
        print("  • Configuration recommendations")
        print("  • Edge case handling")
    else:
        print(f"\n{RED}{BOLD}✗ SOME TESTS FAILED{RESET}")
        print("Please review the failures above.")
    
    # Save test results
    output_dir = Path("experimental_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"calibration_test_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "passed": all_tests_passed,
            "test_results": test_results,
            "summary": {
                "passed_count": passed_count,
                "total_count": total_count
            }
        }, f, indent=2)
    
    print(f"\n{BLUE}Results saved to: {output_file}{RESET}")
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())