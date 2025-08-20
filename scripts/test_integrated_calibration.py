#!/usr/bin/env python3
"""
Integration test: CorrectedDifferenceScorer + CalibratedConfig
Tests the complete pipeline with proper score orientation and calibrated thresholds
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pot.config.calibrated_thresholds import CalibratedConfig, ValidationMode, get_calibrated_config
from pot.scoring.diff_scorer import CorrectedDifferenceScorer
from pot.core.diff_decision import EnhancedSequentialTester, DiffDecisionConfig, TestingMode
import json
from datetime import datetime
from pathlib import Path
import time

def run_integrated_test():
    """Test the integrated system with calibrated thresholds and corrected scorer"""
    
    print("\n" + "="*70)
    print("INTEGRATED CALIBRATION TEST")
    print("CorrectedDifferenceScorer + CalibratedConfig + EnhancedSequentialTester")
    print("="*70)
    
    # Load models
    print("\n📦 Loading models...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model_gpt2 = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    model_distil = AutoModelForCausalLM.from_pretrained('distilgpt2').to(device)
    
    print(f"✅ Models loaded on {device}")
    
    # Initialize corrected scorer
    scorer = CorrectedDifferenceScorer()
    
    # Test prompts
    prompts = [
        "The capital of France is",
        "Machine learning is a field that",
        "The sky appears blue because",
        "Water freezes at a temperature of",
        "The largest planet in our solar system is",
        "Artificial intelligence will",
        "Climate change is caused by",
        "The human brain contains",
        "In the future, technology will",
        "The speed of light is",
        "Quantum mechanics suggests that",
        "The theory of evolution explains",
        "Democracy is a system where",
        "The internet has changed",
        "Renewable energy sources include"
    ]
    
    results = []
    
    # Test both validation modes
    for mode_name in ["quick", "audit"]:
        print(f"\n{'='*50}")
        print(f"Testing {mode_name.upper()} Mode")
        print(f"{'='*50}")
        
        # Get calibrated configuration
        config = get_calibrated_config(mode_name)
        
        print(f"\nCalibrated thresholds:")
        print(f"  γ (SAME): {config.gamma:.6f}")
        print(f"  δ* (DIFFERENT): {config.delta_star:.6f}")
        print(f"  ε_diff: {config.epsilon_diff:.3f}")
        print(f"  k_positions: {config.k_positions}")
        print(f"  n_max: {config.n_max}")
        
        # Test 1: Same model (GPT-2 vs GPT-2)
        print(f"\n1️⃣ Same Model Test (GPT-2 vs GPT-2)")
        print("-" * 40)
        
        # Create tester with calibrated config
        diff_config = DiffDecisionConfig(
            mode=TestingMode.QUICK_GATE if mode_name == "quick" else TestingMode.AUDIT_GRADE
        )
        diff_config.gamma = config.gamma
        diff_config.delta_star = config.delta_star
        diff_config.epsilon_diff = config.epsilon_diff
        diff_config.eta = config.eta
        diff_config.n_min = config.n_min
        diff_config.n_max = config.n_max
        
        tester = EnhancedSequentialTester(diff_config)
        
        # Collect scores
        scores_same = []
        t_start = time.time()
        
        for i in range(min(config.n_max, 40)):  # Limit for speed
            prompt = prompts[i % len(prompts)]
            score = scorer.score_batch(
                model_gpt2, model_gpt2,
                [prompt],
                tokenizer,
                k=config.k_positions,
                method="delta_ce_abs"
            )[0]
            
            scores_same.append(score)
            tester.update(score)
            
            if (i + 1) % 10 == 0:
                mean = np.mean(scores_same)
                print(f"  {i+1} samples: mean={mean:.6f}")
        
        t_elapsed = time.time() - t_start
        
        # Get decision
        should_stop, decision_info = tester.should_stop()
        if not decision_info:
            # Force a decision based on current state
            state = tester.get_state()
            if state['ci'][1] <= config.gamma and state['half_width'] <= config.eta * config.gamma:
                decision_info = {"decision": "SAME", "mean": state['mean'], 
                                "ci_low": state['ci'][0], "ci_high": state['ci'][1], "n": state['n']}
            else:
                decision_info = {"decision": "UNDECIDED", "mean": state['mean'],
                                "ci_low": state['ci'][0], "ci_high": state['ci'][1], "n": state['n']}
        else:
            decision_info['ci_low'] = decision_info.get('ci', (0, 0))[0]
            decision_info['ci_high'] = decision_info.get('ci', (0, 0))[1]
            decision_info['n'] = tester.n
        
        print(f"\n  Decision: {decision_info['decision']}")
        print(f"  Mean: {decision_info['mean']:.6f}")
        print(f"  99% CI: [{decision_info['ci_low']:.6f}, {decision_info['ci_high']:.6f}]")
        print(f"  Samples: {decision_info['n']}")
        print(f"  Time: {t_elapsed:.2f}s")
        
        success_same = decision_info['decision'] in ["SAME", "IDENTICAL"]
        print(f"  {'✅ PASSED' if success_same else '❌ FAILED'} (expected SAME/IDENTICAL)")
        
        results.append({
            "mode": mode_name,
            "test": "GPT-2 vs GPT-2",
            "expected": "SAME/IDENTICAL",
            "decision": decision_info['decision'],
            "mean": float(decision_info['mean']),
            "ci": [float(decision_info['ci_low']), float(decision_info['ci_high'])],
            "n_samples": decision_info['n'],
            "success": success_same,
            "time": t_elapsed
        })
        
        # Test 2: Different models (GPT-2 vs DistilGPT-2)
        print(f"\n2️⃣ Different Model Test (GPT-2 vs DistilGPT-2)")
        print("-" * 40)
        
        # Reset tester
        tester = EnhancedSequentialTester(diff_config)
        
        # Collect scores
        scores_diff = []
        t_start = time.time()
        
        for i in range(min(config.n_max, 40)):  # Limit for speed
            prompt = prompts[i % len(prompts)]
            score = scorer.score_batch(
                model_gpt2, model_distil,
                [prompt],
                tokenizer,
                k=config.k_positions,
                method="delta_ce_abs"
            )[0]
            
            scores_diff.append(score)
            tester.update(score)
            
            if (i + 1) % 10 == 0:
                mean = np.mean(scores_diff)
                print(f"  {i+1} samples: mean={mean:.6f}")
        
        t_elapsed = time.time() - t_start
        
        # Get decision
        should_stop, decision_info = tester.should_stop()
        if not decision_info:
            # Force a decision based on current state
            state = tester.get_state()
            if state['ci'][1] <= config.gamma and state['half_width'] <= config.eta * config.gamma:
                decision_info = {"decision": "SAME", "mean": state['mean'], 
                                "ci_low": state['ci'][0], "ci_high": state['ci'][1], "n": state['n']}
            else:
                decision_info = {"decision": "UNDECIDED", "mean": state['mean'],
                                "ci_low": state['ci'][0], "ci_high": state['ci'][1], "n": state['n']}
        else:
            decision_info['ci_low'] = decision_info.get('ci', (0, 0))[0]
            decision_info['ci_high'] = decision_info.get('ci', (0, 0))[1]
            decision_info['n'] = tester.n
        
        print(f"\n  Decision: {decision_info['decision']}")
        print(f"  Mean: {decision_info['mean']:.6f}")
        print(f"  99% CI: [{decision_info['ci_low']:.6f}, {decision_info['ci_high']:.6f}]")
        print(f"  Samples: {decision_info['n']}")
        print(f"  Time: {t_elapsed:.2f}s")
        
        success_diff = decision_info['decision'] == "DIFFERENT"
        print(f"  {'✅ PASSED' if success_diff else '❌ FAILED'} (expected DIFFERENT)")
        
        results.append({
            "mode": mode_name,
            "test": "GPT-2 vs DistilGPT-2",
            "expected": "DIFFERENT",
            "decision": decision_info['decision'],
            "mean": float(decision_info['mean']),
            "ci": [float(decision_info['ci_low']), float(decision_info['ci_high'])],
            "n_samples": decision_info['n'],
            "success": success_diff,
            "time": t_elapsed
        })
    
    # Summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    
    # Analyze results
    all_success = all(r['success'] for r in results)
    n_undecided = sum(1 for r in results if r['decision'] == 'UNDECIDED')
    
    print("\nResults by mode:")
    for mode in ["quick", "audit"]:
        mode_results = [r for r in results if r['mode'] == mode]
        print(f"\n{mode.upper()} Mode:")
        for r in mode_results:
            status = "✅" if r['success'] else "❌"
            print(f"  {status} {r['test']}: {r['decision']} (mean={r['mean']:.6f})")
    
    print("\nScore Analysis:")
    same_scores = [r['mean'] for r in results if r['test'] == "GPT-2 vs GPT-2"]
    diff_scores = [r['mean'] for r in results if r['test'] == "GPT-2 vs DistilGPT-2"]
    
    if same_scores and diff_scores:
        print(f"  Same model scores: {min(same_scores):.6f} - {max(same_scores):.6f}")
        print(f"  Different model scores: {min(diff_scores):.6f} - {max(diff_scores):.6f}")
        print(f"  Separation: {min(diff_scores) - max(same_scores):.6f}")
    
    print("\nCalibration Performance:")
    print(f"  All tests passed: {'✅ YES' if all_success else '❌ NO'}")
    print(f"  UNDECIDED outcomes: {n_undecided}")
    
    if all_success and n_undecided == 0:
        print("\n🎉 PERFECT CALIBRATION!")
        print("  • CorrectedDifferenceScorer produces proper scores")
        print("  • Calibrated thresholds correctly separate SAME/DIFFERENT")
        print("  • No UNDECIDED outcomes")
    elif all_success:
        print("\n✅ Good calibration with correct decisions")
    else:
        print("\n⚠️ Calibration needs adjustment")
    
    # Save results
    output_dir = Path("experimental_results/integrated_calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"integrated_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "all_passed": all_success,
                "undecided_count": n_undecided,
                "score_ranges": {
                    "same_model": [float(min(same_scores)), float(max(same_scores))] if same_scores else None,
                    "different_models": [float(min(diff_scores)), float(max(diff_scores))] if diff_scores else None
                }
            },
            "configuration": {
                "scorer": "CorrectedDifferenceScorer",
                "config": "CalibratedConfig",
                "tester": "EnhancedSequentialTester"
            },
            "device": str(device)
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return all_success

if __name__ == "__main__":
    success = run_integrated_test()
    sys.exit(0 if success else 1)