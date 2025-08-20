#!/usr/bin/env python3
"""
Test the progressive testing strategy with real models
Demonstrates efficiency gains from staged approach
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from pathlib import Path
from datetime import datetime

def test_progressive_vs_standard():
    """Compare progressive testing to standard approach"""
    
    print("\n🔬 PROGRESSIVE TESTING STRATEGY EVALUATION")
    print("=" * 60)
    
    from pot.core.progressive_testing import ProgressiveTestRunner
    from pot.core.diff_decision import DiffDecisionConfig, EnhancedSequentialTester, TestingMode
    from pot.scoring.optimized_scorer import FastScorer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import numpy as np
    
    # Test cases
    test_cases = [
        {
            "name": "Self-consistency (should be quick)",
            "model_a": "gpt2",
            "model_b": "gpt2",
            "expected": "SAME"
        },
        {
            "name": "Different models (may need deeper analysis)",
            "model_a": "gpt2",
            "model_b": "distilgpt2",
            "expected": "DIFFERENT"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📊 TEST: {test_case['name']}")
        print("-" * 60)
        
        # 1. Run with progressive strategy
        print("\n🚀 Progressive Testing:")
        prog_start = time.time()
        
        prog_result = ProgressiveTestRunner.run(
            test_case["model_a"],
            test_case["model_b"],
            n_prompts=10,
            save_results=False
        )
        
        prog_time = time.time() - prog_start
        prog_samples = prog_result["progression"]["total_samples"]
        prog_stages = prog_result["progression"]["stages_used"]
        
        print(f"\n   Progressive Results:")
        print(f"   - Decision: {prog_result['decision']}")
        print(f"   - Stages used: {prog_stages}")
        print(f"   - Total samples: {prog_samples}")
        print(f"   - Total time: {prog_time:.1f}s")
        print(f"   - Time per sample: {prog_time/prog_samples*1000:.0f}ms")
        
        # 2. Run with standard approach (for comparison)
        print("\n📈 Standard Testing (fixed parameters):")
        std_start = time.time()
        
        # Load models
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(test_case["model_a"])
        tokenizer.pad_token = tokenizer.eos_token
        
        model_a = AutoModelForCausalLM.from_pretrained(test_case["model_a"]).to(device)
        model_b = AutoModelForCausalLM.from_pretrained(test_case["model_b"]).to(device)
        
        # Standard config (like AUDIT_GRADE)
        config = DiffDecisionConfig(mode=TestingMode.AUDIT_GRADE)
        config.n_max = 200  # Fixed high sample count
        
        # Run standard test
        scorer = FastScorer(k=32, top_k=100)
        tester = EnhancedSequentialTester(config)
        
        prompts = [
            "The capital of France is",
            "To make a sandwich, you need",
            "The sky is blue because",
            "Water freezes at",
            "The largest planet is"
        ]
        
        scores = []
        for i in range(config.n_max):
            prompt = prompts[i % len(prompts)]
            score = scorer.score(model_a, model_b, prompt, tokenizer)
            scores.append(score)
            tester.update(score)
            
            # No early stopping in standard approach
        
        std_time = time.time() - std_start
        
        # Compute decision
        scores_array = np.array(scores)
        mean = np.mean(scores_array)
        std = np.std(scores_array, ddof=1)
        margin = 2.576 * std / np.sqrt(len(scores_array))
        ci_low, ci_high = mean - margin, mean + margin
        
        # Decision logic
        if ci_low >= -config.gamma and ci_high <= config.gamma and margin <= config.eta * config.gamma:
            std_decision = "SAME"
        elif abs(mean) >= config.delta_star and margin/abs(mean) <= config.epsilon_diff:
            std_decision = "DIFFERENT"
        else:
            std_decision = "UNDECIDED"
        
        print(f"\n   Standard Results:")
        print(f"   - Decision: {std_decision}")
        print(f"   - Samples used: {config.n_max}")
        print(f"   - Total time: {std_time:.1f}s")
        print(f"   - Time per sample: {std_time/config.n_max*1000:.0f}ms")
        
        # Compare efficiency
        speedup = std_time / prog_time
        sample_reduction = (config.n_max - prog_samples) / config.n_max * 100
        
        print(f"\n⚡ EFFICIENCY GAINS:")
        print(f"   - Speedup: {speedup:.1f}x faster")
        print(f"   - Sample reduction: {sample_reduction:.0f}% fewer samples")
        print(f"   - Same decision: {prog_result['decision'] == std_decision}")
        
        results.append({
            "test": test_case["name"],
            "expected": test_case["expected"],
            "progressive": {
                "decision": prog_result["decision"],
                "stages": prog_stages,
                "samples": prog_samples,
                "time": prog_time
            },
            "standard": {
                "decision": std_decision,
                "samples": config.n_max,
                "time": std_time
            },
            "efficiency": {
                "speedup": speedup,
                "sample_reduction": sample_reduction,
                "decisions_match": prog_result["decision"] == std_decision
            }
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PROGRESSIVE TESTING SUMMARY")
    print("=" * 60)
    
    total_prog_time = sum(r["progressive"]["time"] for r in results)
    total_std_time = sum(r["standard"]["time"] for r in results)
    total_prog_samples = sum(r["progressive"]["samples"] for r in results)
    total_std_samples = sum(r["standard"]["samples"] for r in results)
    
    print(f"\nOverall Performance:")
    print(f"  Progressive: {total_prog_time:.1f}s total, {total_prog_samples} samples")
    print(f"  Standard: {total_std_time:.1f}s total, {total_std_samples} samples")
    print(f"  Overall speedup: {total_std_time/total_prog_time:.1f}x")
    print(f"  Overall sample reduction: {(total_std_samples-total_prog_samples)/total_std_samples*100:.0f}%")
    
    # Check accuracy
    all_match = all(r["efficiency"]["decisions_match"] for r in results)
    if all_match:
        print("\n✅ All decisions match between progressive and standard!")
    else:
        print("\n⚠️ Some decisions differ - may need threshold tuning")
    
    # Save results
    output_dir = Path("experimental_results/progressive")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "total_speedup": total_std_time/total_prog_time,
                "total_sample_reduction": (total_std_samples-total_prog_samples)/total_std_samples,
                "all_decisions_match": all_match
            }
        }, f, indent=2)
    
    print(f"\n💾 Comparison saved to: {output_file}")
    
    return results


def demonstrate_stage_progression():
    """Show how stages progress for different scenarios"""
    
    print("\n🎭 STAGE PROGRESSION DEMONSTRATION")
    print("=" * 60)
    
    from pot.core.progressive_testing import ProgressiveVerifier, ProgressiveTestRunner
    
    # Test identical models (should stop early)
    print("\n1️⃣ Identical Models (GPT-2 vs GPT-2):")
    print("   Expected: Should stop at Stage 1 or 2")
    
    result = ProgressiveTestRunner.run("gpt2", "gpt2", n_prompts=5, save_results=False)
    
    print(f"\n   Stage Progression:")
    for i, stage in enumerate(result["progression"]["history"]):
        print(f"   Stage {i+1}: {stage['stage']} - {stage['decision']} "
              f"(n={stage['n_used']}, mean={stage['mean']:.6f})")
    
    # Test different models (may need more stages)
    print("\n2️⃣ Different Models (GPT-2 vs DistilGPT-2):")
    print("   Expected: May progress through multiple stages")
    
    result = ProgressiveTestRunner.run("gpt2", "distilgpt2", n_prompts=5, save_results=False)
    
    print(f"\n   Stage Progression:")
    for i, stage in enumerate(result["progression"]["history"]):
        print(f"   Stage {i+1}: {stage['stage']} - {stage['decision']} "
              f"(n={stage['n_used']}, mean={stage['mean']:.6f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test progressive testing strategy")
    parser.add_argument("--compare", action="store_true", 
                       help="Compare progressive vs standard testing")
    parser.add_argument("--demo", action="store_true",
                       help="Demonstrate stage progression")
    parser.add_argument("--both", action="store_true",
                       help="Run both tests")
    
    args = parser.parse_args()
    
    if args.both or (not args.compare and not args.demo):
        # Run both by default
        test_progressive_vs_standard()
        print("\n" + "=" * 60)
        demonstrate_stage_progression()
    elif args.compare:
        test_progressive_vs_standard()
    elif args.demo:
        demonstrate_stage_progression()