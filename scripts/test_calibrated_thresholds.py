#!/usr/bin/env python3
"""
Test calibrated thresholds with real model pairs
Verifies that calibration resolves UNDECIDED outcomes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_with_calibration(config=None):
    """Test decision framework with calibrated thresholds"""
    
    print("\n🧪 TESTING CALIBRATED THRESHOLDS")
    print("=" * 60)
    
    from pot.core.diff_decision import DiffDecisionConfig, EnhancedSequentialTester, TestingMode
    from pot.scoring.optimized_scorer import FastScorer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load latest calibration if not provided
    if config is None:
        calib_dir = Path("experimental_results/calibration")
        if calib_dir.exists():
            config_files = list(calib_dir.glob("recommended_config_*.json"))
            if config_files:
                latest_config = max(config_files, key=lambda p: p.stat().st_mtime)
                with open(latest_config, 'r') as f:
                    config = json.load(f)
                print(f"Loaded calibration from: {latest_config}")
            else:
                print("No calibration found. Run calibrate_thresholds.py first.")
                return
        else:
            print("No calibration directory found. Run calibrate_thresholds.py first.")
            return
    
    # Test cases
    test_cases = [
        {
            "name": "Self-consistency (GPT-2 vs GPT-2)",
            "model_a": "gpt2",
            "model_b": "gpt2",
            "expected": "SAME",
            "mode": "quick_gate"
        },
        {
            "name": "Different models (GPT-2 vs DistilGPT-2)",
            "model_a": "gpt2",
            "model_b": "distilgpt2", 
            "expected": "DIFFERENT",
            "mode": "audit_grade"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📊 TEST: {test_case['name']}")
        print("-" * 40)
        
        # Get calibrated config
        mode_config = config.get(test_case['mode'], {})
        
        # Create DiffDecisionConfig with calibrated thresholds
        if test_case['mode'] == 'quick_gate':
            test_mode = TestingMode.QUICK_GATE
        else:
            test_mode = TestingMode.AUDIT_GRADE
            
        diff_config = DiffDecisionConfig(mode=test_mode)
        
        # Apply calibrated thresholds
        diff_config.gamma = mode_config.get('gamma', diff_config.gamma)
        diff_config.delta_star = mode_config.get('delta_star', diff_config.delta_star)
        diff_config.eta = mode_config.get('eta', diff_config.eta)
        diff_config.epsilon_diff = mode_config.get('epsilon_diff', diff_config.epsilon_diff)
        diff_config.n_min = mode_config.get('n_min', diff_config.n_min)
        diff_config.n_max = mode_config.get('n_max', diff_config.n_max)
        
        print(f"Using calibrated thresholds:")
        print(f"   γ = {diff_config.gamma:.6f}")
        print(f"   δ* = {diff_config.delta_star:.6f}")
        print(f"   ε_diff = {diff_config.epsilon_diff:.3f}")
        print(f"   n_min = {diff_config.n_min}, n_max = {diff_config.n_max}")
        
        # Load models
        print(f"\nLoading models: {test_case['model_a']} vs {test_case['model_b']}")
        
        tokenizer = AutoTokenizer.from_pretrained(test_case['model_a'])
        tokenizer.pad_token = tokenizer.eos_token
        
        import torch
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        model_a = AutoModelForCausalLM.from_pretrained(test_case['model_a'])
        model_a = model_a.to(device)
        
        model_b = AutoModelForCausalLM.from_pretrained(test_case['model_b'])
        model_b = model_b.to(device)
        
        # Initialize scorer and tester
        scorer = FastScorer(k=32, top_k=100)
        tester = EnhancedSequentialTester(diff_config)
        
        # Generate test prompts
        prompts = [
            "The capital of France is",
            "To make a sandwich, you need",
            "The sky is blue because",
            "Water freezes at",
            "The largest planet is",
            "If all birds can fly and penguins are birds, then",
            "The opposite of hot is",
            "Shakespeare wrote",
            "Machine learning is",
            "The future of technology"
        ]
        
        print(f"\nRunning test with {len(prompts)} initial prompts...")
        
        # Collect scores
        t_start = time.time()
        n_samples = 0
        
        for i in range(diff_config.n_max):
            prompt = prompts[i % len(prompts)]
            score = scorer.score(model_a, model_b, prompt, tokenizer)
            tester.update(score)
            n_samples += 1
            
            # Check for early stopping every 10 samples
            if n_samples >= diff_config.n_min and n_samples % 10 == 0:
                # Check if we have a decision
                scores_array = np.array(tester.clipped_scores)
                mean = np.mean(scores_array)
                std = np.std(scores_array, ddof=1) if len(scores_array) > 1 else 0
                margin = 2.576 * std / np.sqrt(len(scores_array)) if len(scores_array) > 0 else float('inf')
                ci_low, ci_high = mean - margin, mean + margin
                
                # Check SAME condition
                same_ci = (ci_low >= -diff_config.gamma) and (ci_high <= diff_config.gamma)
                same_precision = margin <= (diff_config.eta * diff_config.gamma)
                
                # Check DIFFERENT condition
                effect_size = abs(mean)
                relative_me = margin / effect_size if effect_size > 0 else float('inf')
                diff_effect = effect_size >= diff_config.delta_star
                diff_precision = relative_me <= diff_config.epsilon_diff
                
                if (same_ci and same_precision) or (diff_effect and diff_precision):
                    break
        
        t_elapsed = time.time() - t_start
        
        # Final decision
        scores_array = np.array(tester.clipped_scores)
        mean = np.mean(scores_array)
        std = np.std(scores_array, ddof=1) if len(scores_array) > 1 else 0
        margin = 2.576 * std / np.sqrt(len(scores_array))
        ci_low, ci_high = mean - margin, mean + margin
        half_width = margin
        
        # Make decision
        same_ci = (ci_low >= -diff_config.gamma) and (ci_high <= diff_config.gamma)
        same_precision = half_width <= (diff_config.eta * diff_config.gamma)
        
        effect_size = abs(mean)
        relative_me = half_width / effect_size if effect_size > 0 else float('inf')
        diff_effect = effect_size >= diff_config.delta_star
        diff_precision = relative_me <= diff_config.epsilon_diff
        
        if same_ci and same_precision:
            decision = "SAME"
            rule = f"CI ⊂ [-{diff_config.gamma:.4f}, +{diff_config.gamma:.4f}] and precision met"
        elif diff_effect and diff_precision:
            decision = "DIFFERENT"
            rule = f"Effect size {effect_size:.4f} ≥ {diff_config.delta_star:.4f} and RME {relative_me:.3f} ≤ {diff_config.epsilon_diff:.3f}"
        else:
            decision = "UNDECIDED"
            rule = f"Neither criteria met (effect={effect_size:.4f}, RME={relative_me:.3f})"
        
        # Display results
        print(f"\n📈 Results:")
        print(f"   Decision: {decision} (expected: {test_case['expected']})")
        print(f"   Rule: {rule}")
        print(f"   Samples used: {n_samples}/{diff_config.n_max}")
        print(f"   Mean difference: {mean:.6f}")
        print(f"   99% CI: [{ci_low:.6f}, {ci_high:.6f}]")
        print(f"   Half-width: {half_width:.6f}")
        print(f"   Time: {t_elapsed:.2f}s ({t_elapsed/n_samples*1000:.0f}ms per sample)")
        
        # Check success
        success = (decision == test_case['expected'])
        if success:
            print(f"   ✅ TEST PASSED!")
        else:
            print(f"   ❌ TEST FAILED (got {decision}, expected {test_case['expected']})")
        
        results.append({
            "test": test_case['name'],
            "expected": test_case['expected'],
            "decision": decision,
            "n_samples": n_samples,
            "mean": float(mean),
            "ci": [float(ci_low), float(ci_high)],
            "success": success,
            "calibrated_thresholds": {
                "gamma": diff_config.gamma,
                "delta_star": diff_config.delta_star,
                "epsilon_diff": diff_config.epsilon_diff
            }
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 CALIBRATION TEST SUMMARY")
    print("=" * 60)
    
    n_success = sum(r['success'] for r in results)
    n_total = len(results)
    
    print(f"\nTests passed: {n_success}/{n_total}")
    
    if n_success == n_total:
        print("✅ ALL TESTS PASSED! Calibration successfully resolves UNDECIDED outcomes.")
    else:
        print("⚠️  Some tests failed. May need further calibration tuning.")
    
    # Check for UNDECIDED
    n_undecided = sum(1 for r in results if r['decision'] == 'UNDECIDED')
    if n_undecided == 0:
        print("✅ NO UNDECIDED OUTCOMES! Calibration is effective.")
    else:
        print(f"⚠️  Still have {n_undecided} UNDECIDED outcomes. Need more aggressive calibration.")
    
    # Save results
    output_file = f"experimental_results/calibration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config_used": config,
            "results": results,
            "summary": {
                "tests_passed": n_success,
                "tests_total": n_total,
                "undecided_count": n_undecided,
                "success_rate": n_success / n_total if n_total > 0 else 0
            }
        }, f, indent=2)
    
    print(f"\n💾 Test results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    # Run calibration first if needed
    calib_dir = Path("experimental_results/calibration")
    if not calib_dir.exists() or not list(calib_dir.glob("recommended_config_*.json")):
        print("No calibration found. Please run calibration first:")
        print("  python scripts/calibrate_thresholds.py")
        print("\nExiting...")
        import sys
        sys.exit(1)
    else:
        config = None
    
    # Test with calibrated thresholds
    results = test_with_calibration(config)