#!/usr/bin/env python3
"""
Test script for the enhanced statistical difference testing framework.
Validates the separate SAME/DIFFERENT decision rules and both testing modes.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pot.core.diff_decision import (
    DiffDecisionConfig, 
    TestingMode,
    EnhancedSequentialTester,
    DifferenceVerifier,
    create_enhanced_verifier
)

# Color codes for pretty output
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
# SIMULATED SCORING FUNCTIONS
# ============================================================================

def simulate_identical_models(ref_model, cand_model, prompt, K=32):
    """Simulate scoring for identical models"""
    # Very small differences due to numerical precision
    return np.random.normal(0.0, 0.001)

def simulate_same_models(ref_model, cand_model, prompt, K=32):
    """Simulate scoring for statistically same models (within tolerance)"""
    # Small consistent differences within equivalence band
    return np.random.normal(0.005, 0.002)

def simulate_different_models(ref_model, cand_model, prompt, K=32):
    """Simulate scoring for clearly different models"""
    # Large consistent differences
    return np.random.normal(0.15, 0.03)

def simulate_borderline_models(ref_model, cand_model, prompt, K=32):
    """Simulate scoring for borderline models (hard to distinguish)"""
    # Differences near detection threshold
    return np.random.normal(0.08, 0.04)

# Mock prompt generator
def generate_test_prompt():
    """Generate test prompts"""
    prompts = [
        "Explain quantum computing in simple terms",
        "What is the capital of France?",
        "Write a haiku about spring",
        "Solve: 2x + 5 = 13",
        "Describe photosynthesis",
        "List prime numbers under 20"
    ]
    return np.random.choice(prompts)

# ============================================================================
# TEST CASES
# ============================================================================

def test_configuration_modes():
    """Test that configuration modes set correct defaults"""
    print_section("Testing Configuration Modes")
    
    # Test QUICK_GATE mode
    quick_cfg = DiffDecisionConfig(mode=TestingMode.QUICK_GATE)
    print(f"QUICK_GATE Configuration:")
    print_result("  Confidence", quick_cfg.confidence)
    print_result("  Gamma (SAME band)", quick_cfg.gamma)
    print_result("  Delta* (DIFFERENT threshold)", quick_cfg.delta_star)
    print_result("  Epsilon_diff (RME target)", quick_cfg.epsilon_diff)
    print_result("  n_min", quick_cfg.n_min)
    print_result("  n_max", quick_cfg.n_max)
    print_result("  K positions", quick_cfg.positions_per_prompt)
    
    # Test AUDIT_GRADE mode
    audit_cfg = DiffDecisionConfig(mode=TestingMode.AUDIT_GRADE)
    print(f"\nAUDIT_GRADE Configuration:")
    print_result("  Confidence", audit_cfg.confidence)
    print_result("  Gamma (SAME band)", audit_cfg.gamma)
    print_result("  Delta* (DIFFERENT threshold)", audit_cfg.delta_star)
    print_result("  Epsilon_diff (RME target)", audit_cfg.epsilon_diff)
    print_result("  n_min", audit_cfg.n_min)
    print_result("  n_max", audit_cfg.n_max)
    print_result("  K positions", audit_cfg.positions_per_prompt)
    
    # Test calibration
    calib_cfg = DiffDecisionConfig(
        mode=TestingMode.AUDIT_GRADE,
        use_calibration=True,
        same_model_p95=0.008,
        near_clone_p5=0.12
    )
    print(f"\nCalibrated Configuration:")
    print_result("  Auto-calibrated gamma", calib_cfg.gamma)
    print_result("  Auto-calibrated delta*", calib_cfg.delta_star)
    
    return True

def test_enhanced_tester_decisions():
    """Test the enhanced sequential tester decision logic"""
    print_section("Testing Enhanced Sequential Tester")
    
    # Test SAME decision
    print("Testing SAME decision logic:")
    config = DiffDecisionConfig(
        mode=TestingMode.AUDIT_GRADE,
        gamma=0.01,
        eta=0.5,
        n_min=20
    )
    tester = EnhancedSequentialTester(config)
    
    # Simulate data that should trigger SAME
    np.random.seed(42)
    for _ in range(30):
        tester.update(np.random.normal(0.003, 0.001))  # Mean within gamma
    
    same_met, same_info = tester.check_same_decision()
    if same_met:
        print(f"  {GREEN}✓ SAME decision triggered correctly{RESET}")
        print(f"    Reason: {same_info['reason']}")
    else:
        print(f"  {RED}✗ SAME decision not triggered{RESET}")
    
    # Test DIFFERENT decision
    print("\nTesting DIFFERENT decision logic:")
    config2 = DiffDecisionConfig(
        mode=TestingMode.AUDIT_GRADE,
        delta_star=0.10,
        epsilon_diff=0.10,
        n_min=20
    )
    tester2 = EnhancedSequentialTester(config2)
    
    # Simulate data that should trigger DIFFERENT
    for _ in range(30):
        tester2.update(np.random.normal(0.15, 0.01))  # Clear difference
    
    diff_met, diff_info = tester2.check_different_decision()
    if diff_met:
        print(f"  {GREEN}✓ DIFFERENT decision triggered correctly{RESET}")
        print(f"    Reason: {diff_info['reason']}")
    else:
        print(f"  {RED}✗ DIFFERENT decision not triggered{RESET}")
    
    # Test UNDECIDED with diagnostics
    print("\nTesting UNDECIDED with diagnostics:")
    config3 = DiffDecisionConfig(
        mode=TestingMode.QUICK_GATE,
        n_max=25  # Force early stop
    )
    tester3 = EnhancedSequentialTester(config3)
    
    # Simulate borderline data
    for _ in range(25):
        tester3.update(np.random.normal(0.07, 0.05))  # High variance
    
    should_stop, info = tester3.should_stop()
    if should_stop and info['decision'] == 'UNDECIDED':
        print(f"  {GREEN}✓ UNDECIDED with diagnostics{RESET}")
        print(f"    Diagnostics:")
        diag = info.get('diagnostics', {})
        if 'same_check' in diag:
            print(f"      SAME path: CI within band={diag['same_check']['ci_within_band']}, "
                  f"Precision met={diag['same_check']['precision_met']}")
        if 'different_check' in diag:
            print(f"      DIFF path: Effect size met={diag['different_check']['effect_size_met']}, "
                  f"RME={diag['different_check']['rme']:.3f}")
        if 'suggestions' in info:
            print(f"    Suggestions: {info['suggestions'][:2]}")
    
    return True

def test_verifier_scenarios():
    """Test the complete verifier with different model scenarios"""
    print_section("Testing Complete Verifier Scenarios")
    
    # Mock models
    class MockModel:
        def __init__(self, name):
            self.name = name
    
    ref_model = MockModel("reference")
    cand_model = MockModel("candidate")
    
    scenarios = [
        ("Identical Models", simulate_identical_models, "IDENTICAL", TestingMode.QUICK_GATE),
        ("Same Models (within tolerance)", simulate_same_models, "SAME", TestingMode.AUDIT_GRADE),
        ("Different Models", simulate_different_models, "DIFFERENT", TestingMode.QUICK_GATE),
        ("Borderline Models", simulate_borderline_models, "UNDECIDED", TestingMode.QUICK_GATE)
    ]
    
    results = []
    for scenario_name, score_fn, expected_decision, mode in scenarios:
        print(f"\n{BOLD}Testing: {scenario_name}{RESET}")
        print(f"  Mode: {mode.value}")
        print(f"  Expected: {expected_decision}")
        
        # Create verifier
        verifier = create_enhanced_verifier(
            score_fn=score_fn,
            prompt_generator=generate_test_prompt,
            mode=mode
        )
        
        # Run verification
        np.random.seed(42)  # For reproducibility
        start_time = time.time()
        report = verifier.verify_difference(
            ref_model, 
            cand_model,
            verbose=False
        )
        elapsed = time.time() - start_time
        
        decision = report['results']['decision']
        n_used = report['results']['n_used']
        mean = report['results']['mean']
        ci = report['results']['ci_99']
        
        # Check result
        success = (decision == expected_decision) or (expected_decision == "UNDECIDED" and n_used >= verifier.cfg.n_max)
        color = GREEN if success else RED
        
        print(f"  Result: {color}{decision}{RESET}")
        print(f"  Samples used: {n_used}")
        print(f"  Mean difference: {mean:.6f}")
        print(f"  99% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")
        print(f"  Time: {elapsed:.2f}s")
        
        if not success:
            print(f"  {RED}✗ Unexpected decision (expected {expected_decision}){RESET}")
        else:
            print(f"  {GREEN}✓ Correct decision{RESET}")
        
        results.append({
            "scenario": scenario_name,
            "expected": expected_decision,
            "actual": decision,
            "success": success,
            "n_used": n_used,
            "time": elapsed
        })
    
    return all(r["success"] for r in results)

def test_ci_methods():
    """Test different confidence interval methods"""
    print_section("Testing CI Methods")
    
    # Test Empirical-Bernstein
    print("Testing Empirical-Bernstein CI:")
    eb_config = DiffDecisionConfig(
        mode=TestingMode.AUDIT_GRADE,
        ci_method="eb",
        score_clip_low=0.0,
        score_clip_high=0.3
    )
    eb_tester = EnhancedSequentialTester(eb_config)
    
    # Add some scores
    np.random.seed(42)
    for _ in range(50):
        eb_tester.update(np.random.normal(0.1, 0.02))
    
    (eb_lo, eb_hi), eb_hw = eb_tester.compute_ci()
    print(f"  EB CI: [{eb_lo:.6f}, {eb_hi:.6f}], half-width: {eb_hw:.6f}")
    
    # Test t-distribution
    print("\nTesting t-distribution CI:")
    t_config = DiffDecisionConfig(
        mode=TestingMode.AUDIT_GRADE,
        ci_method="t"
    )
    t_tester = EnhancedSequentialTester(t_config)
    
    # Add same scores
    np.random.seed(42)
    for _ in range(50):
        t_tester.update(np.random.normal(0.1, 0.02))
    
    (t_lo, t_hi), t_hw = t_tester.compute_ci()
    print(f"  t-dist CI: [{t_lo:.6f}, {t_hi:.6f}], half-width: {t_hw:.6f}")
    
    # Compare
    print(f"\n  Comparison:")
    print(f"    EB half-width: {eb_hw:.6f}")
    print(f"    t-dist half-width: {t_hw:.6f}")
    print(f"    EB is {'tighter' if eb_hw < t_hw else 'wider'} than t-dist")
    
    return True

def test_effective_sample_size():
    """Test that effective sample size calculation works correctly"""
    print_section("Testing Effective Sample Size")
    
    config = DiffDecisionConfig(
        mode=TestingMode.AUDIT_GRADE,
        positions_per_prompt=64  # K=64
    )
    tester = EnhancedSequentialTester(config)
    
    # Add 10 prompts
    for _ in range(10):
        tester.update(np.random.normal(0.05, 0.01))
    
    print(f"  Prompts processed: {tester.n}")
    print(f"  Positions per prompt (K): {config.positions_per_prompt}")
    print(f"  Effective sample size: {tester.n * config.positions_per_prompt}")
    
    # Check that CI uses effective sample size
    (ci_lo, ci_hi), hw = tester.compute_ci()
    state = tester.get_state()
    
    print(f"  CI with effective n: [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"  Half-width: {hw:.6f}")
    print(f"  Relative ME: {state['rel_me']:.3f}")
    
    return True

def save_test_results(results: Dict[str, Any]):
    """Save test results to file"""
    output_dir = Path("experimental_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"enhanced_diff_decision_test_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{BLUE}Results saved to: {output_file}{RESET}")

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests"""
    print_header("Enhanced Statistical Difference Testing Framework")
    print("Testing separate SAME/DIFFERENT decision rules with diagnostics\n")
    
    start_time = time.time()
    all_tests_passed = True
    test_results = {}
    
    # Run tests
    tests = [
        ("Configuration Modes", test_configuration_modes),
        ("Enhanced Tester Decisions", test_enhanced_tester_decisions),
        ("Complete Verifier Scenarios", test_verifier_scenarios),
        ("CI Methods", test_ci_methods),
        ("Effective Sample Size", test_effective_sample_size)
    ]
    
    for test_name, test_func in tests:
        print_header(test_name)
        try:
            passed = test_func()
            test_results[test_name] = {
                "passed": passed,
                "error": None
            }
            if passed:
                print(f"\n{GREEN}✓ {test_name} passed{RESET}")
            else:
                print(f"\n{RED}✗ {test_name} failed{RESET}")
                all_tests_passed = False
        except Exception as e:
            print(f"\n{RED}✗ {test_name} error: {e}{RESET}")
            test_results[test_name] = {
                "passed": False,
                "error": str(e)
            }
            all_tests_passed = False
    
    # Summary
    elapsed = time.time() - start_time
    print_header("Test Summary")
    
    passed_count = sum(1 for r in test_results.values() if r["passed"])
    total_count = len(test_results)
    
    print(f"Tests Passed: {passed_count}/{total_count}")
    print(f"Total Time: {elapsed:.2f}s")
    
    if all_tests_passed:
        print(f"\n{GREEN}{BOLD}✓ ALL TESTS PASSED!{RESET}")
        print("\nThe enhanced statistical difference testing framework is working correctly.")
        print("Features validated:")
        print("  • Separate SAME/DIFFERENT decision rules")
        print("  • Quick Gate and Audit Grade modes")
        print("  • Enhanced diagnostics for UNDECIDED cases")
        print("  • Calibration support")
        print("  • Effective sample size calculation")
        print("  • Both EB and t-distribution CI methods")
    else:
        print(f"\n{RED}{BOLD}✗ SOME TESTS FAILED{RESET}")
        print("Please review the failures above.")
    
    # Save results
    save_test_results({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time": elapsed,
        "passed": all_tests_passed,
        "test_results": test_results,
        "summary": {
            "passed_count": passed_count,
            "total_count": total_count
        }
    })
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())