#!/usr/bin/env python3
"""
Test comparing GPT-2 vs DistilGPT-2 using open models
No authentication tokens required - both models are publicly available
"""

import os
import sys
import time
import json
import pathlib
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PoT modules
from pot.lm.verifier import LMVerifier
from pot.lm.lm_config import LMVerifierConfig
from pot.core.diff_decision import DiffDecisionConfig, DifferenceVerifier


def main():
    """Run test comparing GPT-2 and DistilGPT-2"""
    print("\n🔬 Sequential Test: GPT-2 vs DistilGPT-2 (Open Models)\n")
    
    # Use open models that don't require authentication
    reference_model = "gpt2"  # Base GPT-2 (124M params)
    candidate_same = "gpt2"   # Same model for baseline
    candidate_diff = "distilgpt2"  # DistilGPT-2 (82M params)
    
    # Check for transformers
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("❌ transformers not installed. Run: pip install transformers")
        return 1
    
    print(f"Reference model: {reference_model}")
    print(f"Candidate (same): {candidate_same}")
    print(f"Candidate (different): {candidate_diff}")
    print()
    
    # Configuration for difference testing
    config = DiffDecisionConfig(
        alpha=0.01,
        rel_margin_target=0.05,
        n_min=10,
        n_max=50,
        batch_size=4,
        positions_per_prompt=32,
        method="eb",
        clip_low=0.0,
        clip_high=0.2,
        equivalence_band=None,
        similar_size_ratio=2.0
    )
    
    # Test prompts
    test_prompts = [
        "The weather today is",
        "In the future, artificial intelligence will",
        "The most important thing about",
        "Scientists have discovered that",
        "The economic impact of"
    ]
    
    # Create verifier
    verifier = DifferenceVerifier(
        config=config,
        reference_model=reference_model,
        model_type="causal_lm",
        device="cpu",
        namespace="gpt2_vs_distilgpt2",
        output_dir="outputs/gpt2_vs_distilgpt2"
    )
    
    print("=" * 60)
    print("TEST 1: GPT-2 vs GPT-2 (Same Model)")
    print("-" * 60)
    
    try:
        # Test same model
        report_same = verifier.verify_model_pair(
            candidate_model=candidate_same,
            prompts=test_prompts[:3],  # Use fewer prompts for speed
            load_time=0.5
        )
        
        print(f"Decision: {report_same['decision']}")
        print(f"Mean score: {report_same['mean_score']:.6f}")
        print(f"99% CI: [{report_same['ci_99'][0]:.6f}, {report_same['ci_99'][1]:.6f}]")
        print(f"Samples used: {report_same['n_used']}")
        
        if report_same['decision'] in ['SAME', 'IDENTICAL']:
            print("✅ PASS: Same model correctly identified")
            test1_pass = True
        else:
            print("❌ FAIL: Same model should be identified as SAME")
            test1_pass = False
            
    except Exception as e:
        print(f"❌ Error testing same model: {e}")
        test1_pass = False
    
    print("\n" + "=" * 60)
    print("TEST 2: GPT-2 vs DistilGPT-2 (Different Models)")
    print("-" * 60)
    
    try:
        # Test different model
        report_diff = verifier.verify_model_pair(
            candidate_model=candidate_diff,
            prompts=test_prompts,
            load_time=0.5
        )
        
        print(f"Decision: {report_diff['decision']}")
        print(f"Mean score: {report_diff['mean_score']:.6f}")
        print(f"99% CI: [{report_diff['ci_99'][0]:.6f}, {report_diff['ci_99'][1]:.6f}]")
        print(f"Samples used: {report_diff['n_used']}")
        
        if report_diff['decision'] == 'DIFFERENT':
            print("✅ PASS: Different models correctly distinguished")
            test2_pass = True
        else:
            print("❌ FAIL: Different models should be identified as DIFFERENT")
            test2_pass = False
            
    except Exception as e:
        print(f"❌ Error testing different model: {e}")
        test2_pass = False
    
    # Save results
    results = {
        "reference": reference_model,
        "test1": {
            "candidate": candidate_same,
            "passed": test1_pass,
            "report": report_same if 'report_same' in locals() else None
        },
        "test2": {
            "candidate": candidate_diff,
            "passed": test2_pass,
            "report": report_diff if 'report_diff' in locals() else None
        }
    }
    
    output_file = "experimental_results/gpt2_vs_distilgpt2_results.json"
    os.makedirs("experimental_results", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Overall result
    print("\n" + "=" * 60)
    if test1_pass and test2_pass:
        print("🎉 ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())