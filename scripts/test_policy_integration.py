#!/usr/bin/env python3
"""
Integration test: Statistical Policy + CorrectedDifferenceScorer + Real Models
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pot.core.statistical_policy import DiffDecisionConfig, SequentialDiffTester
from pot.scoring.diff_scorer import CorrectedDifferenceScorer
import logging
import json
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_model_test(model_a_name: str, model_b_name: str, expected: str) -> dict:
    """Run a test with real models using the statistical policy"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_a_name} vs {model_b_name}")
    logger.info(f"Expected: {expected}")
    logger.info(f"{'='*60}")
    
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_a_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model_a = AutoModelForCausalLM.from_pretrained(model_a_name).to(device)
    model_a.eval()
    
    if model_a_name == model_b_name:
        model_b = model_a
    else:
        model_b = AutoModelForCausalLM.from_pretrained(model_b_name).to(device)
        model_b.eval()
    
    # Configure with calibrated values
    config = DiffDecisionConfig(
        mode="AUDIT_GRADE",
        same_model_p95=3.39e-4,   # From actual calibration
        near_clone_p5=0.0763,      # From actual calibration
        use_calibration=True,
        confidence=0.99,
        n_min=30,
        n_max=400,  # Full audit grade
        positions_per_prompt=128,
        ci_method="eb",
        score_clip_low=0.0,
        score_clip_high=1.0,  # Allow full range for CorrectedDifferenceScorer
        epsilon_diff=0.15  # Slightly relaxed for real model variance
    )
    config.finalize()
    
    logger.info(f"Configuration:")
    logger.info(f"  γ = {config.gamma:.6f}")
    logger.info(f"  δ* = {config.delta_star:.6f}")
    logger.info(f"  ε_diff = {config.epsilon_diff}")
    logger.info(f"  K = {config.positions_per_prompt}")
    
    # Initialize scorer and tester
    scorer = CorrectedDifferenceScorer()
    tester = SequentialDiffTester(config)
    
    # Test prompts - more variety for better testing
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
        "Python is a programming language that",
        "The Great Wall of China was built",
        "Shakespeare wrote many plays including",
        "The human genome contains approximately",
        "Quantum computing differs from classical computing",
        "The process of photosynthesis converts",
        "Black holes are regions of space where",
        "The theory of evolution explains how",
        "Neural networks are inspired by",
        "The internet was first developed in"
    ]
    
    # Run sequential testing
    logger.info("\nRunning sequential test...")
    decision_info = None
    prompt_idx = 0
    
    for i in range(config.n_max):
        # Use different prompt each time for variety
        prompt = prompts[prompt_idx % len(prompts)]
        prompt_idx += 1
        
        # Set random seed for position sampling variety
        torch.manual_seed(i * 42 + prompt_idx)
        
        # Get score using CorrectedDifferenceScorer
        scores = scorer.score_batch(
            model_a, model_b, 
            [prompt], 
            tokenizer, 
            k=config.positions_per_prompt,
            method="delta_ce_abs"
        )
        score = scores[0]
        
        # Add to tester
        tester.add_sample(score)
        
        # Check stopping condition
        if i >= config.n_min - 1:
            should_stop, info = tester.should_stop()
            if should_stop:
                decision_info = info
                logger.info(f"\nStopped at n={i+1}")
                break
        
        # Progress update
        if (i + 1) % 10 == 0:
            (lo, hi), h = tester.compute_ci()
            logger.info(f"  n={i+1}: mean={tester.mean:.6f}, CI=[{lo:.6f}, {hi:.6f}]")
    
    # Get final decision if not stopped early
    if decision_info is None:
        _, decision_info = tester.should_stop()
    
    # Display results
    logger.info(f"\n📊 Results:")
    logger.info(f"  Decision: {decision_info['decision']}")
    logger.info(f"  Mean: {decision_info.get('mean', 0):.6f}")
    logger.info(f"  CI: [{decision_info['ci'][0]:.6f}, {decision_info['ci'][1]:.6f}]")
    logger.info(f"  Half-width: {decision_info['half_width']:.6f}")
    if 'rme' in decision_info:
        logger.info(f"  RME: {decision_info['rme']:.4f}")
    logger.info(f"  Rule: {decision_info.get('rule', 'N/A')}")
    logger.info(f"  n_used: {decision_info['n_used']}")
    
    # Check if passed
    passed = decision_info['decision'] == expected or \
             (expected == "SAME" and decision_info['decision'] == "IDENTICAL")
    
    if passed:
        logger.info(f"  ✅ TEST PASSED")
    else:
        logger.info(f"  ❌ TEST FAILED (expected {expected})")
    
    return {
        "test": f"{model_a_name} vs {model_b_name}",
        "expected": expected,
        "decision": decision_info['decision'],
        "passed": passed,
        "mean": decision_info.get('mean', 0),
        "ci": decision_info['ci'],
        "half_width": decision_info['half_width'],
        "rme": decision_info.get('rme'),
        "rule": decision_info.get('rule'),
        "n_used": decision_info['n_used'],
        "config": {
            "gamma": config.gamma,
            "delta_star": config.delta_star,
            "epsilon_diff": config.epsilon_diff,
            "k": config.positions_per_prompt
        }
    }

def main():
    """Run integration tests with real models"""
    
    logger.info("\n" + "="*70)
    logger.info("STATISTICAL POLICY + REAL MODELS INTEGRATION TEST")
    logger.info("="*70)
    
    # Test cases
    test_cases = [
        ("gpt2", "gpt2", "SAME"),
        ("gpt2", "distilgpt2", "DIFFERENT")
    ]
    
    results = []
    
    for model_a, model_b, expected in test_cases:
        result = run_model_test(model_a, model_b, expected)
        results.append(result)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("="*70)
    
    all_passed = all(r['passed'] for r in results)
    
    for result in results:
        status = "✅" if result['passed'] else "❌"
        logger.info(f"{status} {result['test']}: {result['decision']} (mean={result['mean']:.6f}, n={result['n_used']})")
    
    if all_passed:
        logger.info("\n🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info("Statistical policy working correctly with:")
        logger.info("  • CorrectedDifferenceScorer (proper orientation)")
        logger.info("  • Calibrated thresholds (γ=0.001017, δ*=0.038320)")
        logger.info("  • Real models (GPT-2, DistilGPT-2)")
        logger.info("  • Empirical-Bernstein CI with clipping")
    else:
        logger.info("\n⚠️ Some tests failed")
    
    # Save results
    output_dir = Path("experimental_results/policy_integration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "all_passed": all_passed,
            "summary": {
                "same_model_mean": results[0]['mean'] if results else None,
                "different_model_mean": results[1]['mean'] if len(results) > 1 else None,
                "separation": results[1]['mean'] - results[0]['mean'] if len(results) > 1 else None
            }
        }, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {output_file}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())