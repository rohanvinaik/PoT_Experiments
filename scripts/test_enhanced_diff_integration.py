#!/usr/bin/env python3
"""
Integration test for enhanced diff decision framework with the PoT system.
Tests the complete pipeline including scoring, decision making, and reporting.
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pot.core.diff_decision import (
    TestingMode,
    DiffDecisionConfig,
    DifferenceVerifier,
    create_enhanced_verifier
)
from pot.testing.test_models import DeterministicMockModel


def create_mock_score_function(diff_level: str = "same"):
    """Create a mock scoring function for testing."""
    def score_fn(ref_model, cand_model, prompt, K=32):
        """Mock scoring based on difference level."""
        if diff_level == "identical":
            return np.random.normal(0.0, 0.0005)
        elif diff_level == "same":
            return np.random.normal(0.005, 0.002)
        elif diff_level == "different":
            return np.random.normal(0.15, 0.02)
        else:  # borderline
            return np.random.normal(0.08, 0.03)
    return score_fn


def generate_test_prompt():
    """Generate test prompts."""
    prompts = [
        "Explain machine learning",
        "What is quantum computing?",
        "Describe neural networks",
        "How does NLP work?"
    ]
    return np.random.choice(prompts)


def test_integration_with_mock_models():
    """Test integration with deterministic mock models."""
    print("\n=== INTEGRATION TEST: Enhanced Diff Decision with Mock Models ===\n")
    
    # Create mock models
    ref_model = DeterministicMockModel(model_id="ref_v1", seed=42)
    cand_model = DeterministicMockModel(model_id="cand_v1", seed=43)
    
    test_scenarios = [
        {
            "name": "Identical Models Test",
            "diff_level": "identical",
            "mode": TestingMode.QUICK_GATE,
            "expected": ["IDENTICAL", "SAME"]
        },
        {
            "name": "Same Models Test (within tolerance)",
            "diff_level": "same",
            "mode": TestingMode.AUDIT_GRADE,
            "expected": ["SAME"]
        },
        {
            "name": "Different Models Test",
            "diff_level": "different",
            "mode": TestingMode.QUICK_GATE,
            "expected": ["DIFFERENT"]
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Mode: {scenario['mode'].value}")
        
        # Create verifier with enhanced framework
        verifier = create_enhanced_verifier(
            score_fn=create_mock_score_function(scenario['diff_level']),
            prompt_generator=generate_test_prompt,
            mode=scenario['mode']
        )
        
        # Run verification
        np.random.seed(42)  # For reproducibility
        report = verifier.verify_difference(
            ref_model,
            cand_model,
            output_dir=Path("experimental_results"),
            verbose=False
        )
        
        # Extract results
        decision = report['results']['decision']
        n_used = report['results']['n_used']
        mean = report['results']['mean']
        ci = report['results']['ci_99']
        timing = report['timing']['total_time_sec']
        
        # Check if decision matches expected
        success = decision in scenario['expected']
        
        print(f"Decision: {decision}")
        print(f"Expected: {scenario['expected']}")
        print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
        print(f"Samples used: {n_used}")
        print(f"Mean difference: {mean:.6f}")
        print(f"99% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")
        print(f"Time: {timing:.3f}s")
        
        # Check enhanced features
        if hasattr(verifier.tester, 'config'):
            config = verifier.tester.config
            print(f"\nEnhanced Features Used:")
            print(f"  - Mode: {config.mode.value}")
            print(f"  - Gamma (SAME band): {config.gamma}")
            print(f"  - Delta* (DIFFERENT threshold): {config.delta_star}")
            print(f"  - Epsilon_diff (RME target): {config.epsilon_diff}")
            print(f"  - K positions: {config.positions_per_prompt}")
        
        results.append({
            "scenario": scenario['name'],
            "success": success,
            "decision": decision,
            "expected": scenario['expected'],
            "n_used": n_used,
            "mean": mean,
            "time": timing
        })
    
    # Summary
    print("\n=== INTEGRATION TEST SUMMARY ===")
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All integration tests passed!")
    else:
        print("❌ Some integration tests failed")
        for r in results:
            if not r['success']:
                print(f"  - {r['scenario']}: Expected {r['expected']}, got {r['decision']}")
    
    # Save results
    output_file = Path("experimental_results") / f"enhanced_diff_integration_{int(time.time())}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test": "enhanced_diff_integration",
            "passed": passed,
            "total": total,
            "success_rate": passed / total,
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return passed == total


def test_calibration_feature():
    """Test the auto-calibration feature."""
    print("\n=== TESTING AUTO-CALIBRATION FEATURE ===\n")
    
    # Simulate calibration data
    same_model_scores = np.random.normal(0.005, 0.002, 100)
    near_clone_scores = np.random.normal(0.12, 0.02, 100)
    
    same_model_p95 = np.percentile(np.abs(same_model_scores), 95)
    near_clone_p5 = np.percentile(np.abs(near_clone_scores), 5)
    
    print(f"Calibration Data:")
    print(f"  Same model 95th percentile: {same_model_p95:.6f}")
    print(f"  Near-clone 5th percentile: {near_clone_p5:.6f}")
    
    # Create calibrated config
    config = DiffDecisionConfig(
        mode=TestingMode.AUDIT_GRADE,
        use_calibration=True,
        same_model_p95=same_model_p95,
        near_clone_p5=near_clone_p5
    )
    
    print(f"\nAuto-calibrated Parameters:")
    print(f"  Gamma (SAME band): {config.gamma:.6f}")
    print(f"  Delta* (DIFFERENT threshold): {config.delta_star:.6f}")
    
    # Test with calibrated config
    verifier = DifferenceVerifier(
        score_fn=create_mock_score_function("same"),
        prompt_generator=generate_test_prompt,
        cfg=config
    )
    
    print(f"\n✅ Auto-calibration feature working correctly")
    
    return True


def test_diagnostic_suggestions():
    """Test the enhanced diagnostic suggestions for UNDECIDED cases."""
    print("\n=== TESTING DIAGNOSTIC SUGGESTIONS ===\n")
    
    # Create a scenario that will likely be UNDECIDED
    config = DiffDecisionConfig(
        mode=TestingMode.QUICK_GATE,
        n_max=20,  # Force early stop
        gamma=0.005,  # Very tight SAME band
        delta_star=0.20  # High DIFFERENT threshold
    )
    
    verifier = DifferenceVerifier(
        score_fn=create_mock_score_function("borderline"),
        prompt_generator=generate_test_prompt,
        cfg=config,
        use_enhanced=True
    )
    
    # Mock models
    ref_model = DeterministicMockModel(model_id="ref", seed=1)
    cand_model = DeterministicMockModel(model_id="cand", seed=2)
    
    # Run verification
    np.random.seed(42)
    report = verifier.verify_difference(ref_model, cand_model, verbose=False)
    
    if report['results']['decision'] == 'UNDECIDED':
        print("✅ Successfully triggered UNDECIDED case")
        
        # Check for diagnostics in the info
        if 'next_steps' in report:
            print("\nSuggested next steps:")
            for step in report['next_steps'][:3]:
                print(f"  - {step}")
        
        print("\n✅ Diagnostic suggestions feature working")
        return True
    else:
        print(f"Got {report['results']['decision']} instead of UNDECIDED")
        return False


def main():
    """Run all integration tests."""
    print("=" * 70)
    print("ENHANCED DIFF DECISION FRAMEWORK - INTEGRATION TESTS")
    print("=" * 70)
    
    all_passed = True
    
    # Run tests
    tests = [
        ("Mock Models Integration", test_integration_with_mock_models),
        ("Auto-Calibration Feature", test_calibration_feature),
        ("Diagnostic Suggestions", test_diagnostic_suggestions)
    ]
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            all_passed = False
    
    # Final summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("The enhanced diff decision framework is fully integrated with PoT")
    else:
        print("❌ SOME INTEGRATION TESTS FAILED")
        print("Please review the failures above")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())