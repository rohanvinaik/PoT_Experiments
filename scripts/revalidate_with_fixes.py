#!/usr/bin/env python3
"""
Revalidation with all fixes applied:
1. Calibrated thresholds (γ=0.00102, δ*=0.0383)
2. Corrected score orientation (always non-negative)
3. Fixed EnhancedSequentialTester with .differences field
4. Proper K=64/128 for variance control
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import json
import time
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Any, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_validation_test(ref_model_name: str, 
                       cand_model_name: str,
                       config: Any,
                       expected: str) -> Dict[str, Any]:
    """Run a single validation test with proper configuration"""
    
    logger.info(f"\nTesting {ref_model_name} vs {cand_model_name}")
    logger.info(f"Config: γ={config.gamma:.6f}, δ*={config.delta_star:.4f}, K={config.k_positions}")
    
    # Load models
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load models
    logger.info("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name).to(device)
    ref_model.eval()
    
    if ref_model_name == cand_model_name:
        cand_model = ref_model  # Same model test
    else:
        cand_model = AutoModelForCausalLM.from_pretrained(cand_model_name).to(device)
        cand_model.eval()
    
    # Setup components
    from pot.scoring.diff_scorer import CorrectedDifferenceScorer
    from pot.core.diff_decision import EnhancedSequentialTester, DiffDecisionConfig, TestingMode
    
    # Create scorer with corrected orientation
    scorer = CorrectedDifferenceScorer()
    
    # Create test prompts
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
        "Renewable energy sources include",
        "The Great Wall of China was built",
        "Shakespeare wrote plays about",
        "The moon affects Earth's",
        "DNA contains instructions for",
        "Computer programming involves"
    ]
    
    # Create enhanced tester with calibrated config
    diff_config = DiffDecisionConfig(mode=TestingMode.AUDIT_GRADE)
    diff_config.gamma = config.gamma
    diff_config.delta_star = config.delta_star
    diff_config.epsilon_diff = config.epsilon_diff
    diff_config.eta = config.eta
    diff_config.n_min = config.n_min
    diff_config.n_max = config.n_max
    diff_config.positions_per_prompt = config.k_positions
    
    tester = EnhancedSequentialTester(diff_config)
    
    # Run test
    t_start = time.perf_counter()
    t_infer_total = 0
    
    for i in range(config.n_max):
        prompt = prompts[i % len(prompts)]
        
        t_score_start = time.perf_counter()
        # Use score_batch for consistency
        scores = scorer.score_batch(
            ref_model, cand_model, 
            [prompt], 
            tokenizer, 
            k=config.k_positions,
            method="delta_ce_abs"
        )
        score = scores[0]
        t_score_end = time.perf_counter()
        
        t_infer_total += (t_score_end - t_score_start)
        tester.update(score)
        
        # Check stopping
        if i >= config.n_min - 1:
            should_stop, info = tester.should_stop()
            if should_stop:
                break
        
        # Progress logging
        if (i + 1) % 10 == 0:
            (ci_lo, ci_hi), hw = tester.compute_ci()
            logger.info(f"  n={i+1}: mean={tester.mean:.6f}, CI=[{ci_lo:.6f}, {ci_hi:.6f}]")
    
    t_total = time.perf_counter() - t_start
    
    # Get final decision
    _, final_info = tester.should_stop()
    if not final_info:
        # Force a decision
        (ci_lo, ci_hi), hw = tester.compute_ci()
        final_info = {
            "decision": "UNDECIDED",
            "ci": (ci_lo, ci_hi),
            "half_width": hw,
            "mean": tester.mean,
            "reason": "Test ended without decision"
        }
    
    # Adjust expected for identical models
    if expected == "SAME" and final_info["decision"] == "IDENTICAL":
        final_info["decision"] = "SAME"  # Treat IDENTICAL as SAME for comparison
    
    # Build result
    result = {
        "test_case": f"{ref_model_name}_vs_{cand_model_name}",
        "expected": expected,
        "decision": final_info["decision"],
        "passed": final_info["decision"] == expected,
        "n_used": tester.n,
        "n_eff": tester.n * config.k_positions,
        "mean": float(tester.mean),
        "ci": [float(final_info["ci"][0]), float(final_info["ci"][1])],
        "half_width": float(final_info["half_width"]),
        "rme": float(final_info.get("rme", 0)) if final_info.get("rme") else None,
        "timing": {
            "t_total": t_total,
            "t_infer_total": t_infer_total,
            "t_per_query": t_infer_total / tester.n if tester.n > 0 else 0
        },
        "config": {
            "gamma": config.gamma,
            "delta_star": config.delta_star,
            "epsilon_diff": config.epsilon_diff,
            "k_positions": config.k_positions,
            "n_min": config.n_min,
            "n_max": config.n_max,
            "confidence": config.confidence
        },
        "reason": final_info.get("reason", "")
    }
    
    return result

def main():
    logger.info("="*70)
    logger.info("REVALIDATION WITH CALIBRATED THRESHOLDS")
    logger.info("="*70)
    
    # Import configurations
    from pot.config.calibrated_thresholds import get_calibrated_config
    
    # Test cases
    test_cases = [
        ("gpt2", "gpt2", "SAME"),
        ("gpt2", "distilgpt2", "DIFFERENT")
    ]
    
    results = {}
    
    # Run with AUDIT_GRADE settings
    logger.info("\n🎯 Using AUDIT GRADE configuration")
    config = get_calibrated_config("audit")
    
    logger.info(f"Calibrated thresholds:")
    logger.info(f"  γ (SAME band): {config.gamma:.6f}")
    logger.info(f"  δ* (DIFFERENT threshold): {config.delta_star:.4f}")
    logger.info(f"  K (positions): {config.k_positions}")
    logger.info(f"  Confidence: {config.confidence*100:.1f}%")
    
    for ref_model, cand_model, expected in test_cases:
        result = run_validation_test(ref_model, cand_model, config, expected)
        results[f"{ref_model}_{cand_model}"] = result
        
        # Display result
        logger.info(f"\n{'✅' if result['passed'] else '❌'} {result['test_case']}")
        logger.info(f"  Expected: {expected}, Got: {result['decision']}")
        logger.info(f"  Mean: {result['mean']:.6f}, CI: [{result['ci'][0]:.6f}, {result['ci'][1]:.6f}]")
        logger.info(f"  Samples: {result['n_used']}, Time: {result['timing']['t_total']:.2f}s")
    
    # Generate summary report
    summary = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "fixes_applied": [
            "Calibrated thresholds from actual model runs (γ=0.00102, δ*=0.0383)",
            "Corrected score orientation (absolute CE difference, always ≥ 0)",
            "Fixed EnhancedSequentialTester with .differences field",
            "Proper K=128 for audit grade variance control",
            "Empirical-Bernstein CI with clipped scores"
        ],
        "calibration_source": "Based on CorrectedDifferenceScorer measurements",
        "test_results": results,
        "overall_passed": all(r["passed"] for r in results.values()),
        "summary_table": {
            "test": ["GPT2 vs GPT2", "GPT2 vs DistilGPT2"],
            "expected": ["SAME", "DIFFERENT"],
            "got": [results["gpt2_gpt2"]["decision"], results["gpt2_distilgpt2"]["decision"]],
            "passed": [results["gpt2_gpt2"]["passed"], results["gpt2_distilgpt2"]["passed"]],
            "mean": [results["gpt2_gpt2"]["mean"], results["gpt2_distilgpt2"]["mean"]],
            "n_used": [results["gpt2_gpt2"]["n_used"], results["gpt2_distilgpt2"]["n_used"]]
        }
    }
    
    # Save report
    output_dir = Path("experimental_results/revalidation_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / f"revalidation_fixed_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Display final summary
    logger.info("\n" + "="*70)
    logger.info("REVALIDATION SUMMARY")
    logger.info("="*70)
    
    if summary["overall_passed"]:
        logger.info("✅ ALL TESTS PASSED WITH CALIBRATED THRESHOLDS")
    else:
        logger.info("❌ SOME TESTS FAILED")
    
    logger.info("\nCalibrated Thresholds Used:")
    logger.info(f"  γ = {config.gamma:.6f} (3 × same-model P95)")
    logger.info(f"  δ* = {config.delta_star:.4f} (midpoint between same P95 and near-clone P5)")
    logger.info(f"  K = {config.k_positions} positions per prompt")
    logger.info(f"  n_max = {config.n_max} samples")
    
    logger.info("\nResults Summary:")
    for test, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        logger.info(f"  {test}: {result['decision']} ({status})")
        logger.info(f"    Mean: {result['mean']:.6f}, n_eff: {result['n_eff']}")
    
    # Check for UNDECIDED outcomes
    n_undecided = sum(1 for r in results.values() if r["decision"] == "UNDECIDED")
    if n_undecided == 0:
        logger.info("\n🎉 NO UNDECIDED OUTCOMES - Perfect calibration achieved!")
    else:
        logger.info(f"\n⚠️ {n_undecided} UNDECIDED outcomes remain")
    
    logger.info(f"\nReport saved to: {report_file}")
    logger.info("="*70)
    
    return 0 if summary["overall_passed"] else 1

if __name__ == "__main__":
    sys.exit(main())