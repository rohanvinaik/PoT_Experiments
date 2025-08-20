#!/usr/bin/env python3
"""
Performance test for optimized scoring
Compares original vs optimized inference speed
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_optimized_scorer():
    """Test the optimized scorer performance"""
    
    from pot.scoring.optimized_scorer import (
        OptimizedTeacherForcedScorer,
        OptimizedScoringConfig,
        FastScorer
    )
    
    print("🚀 OPTIMIZED SCORER PERFORMANCE TEST")
    print("=" * 60)
    
    # Load models
    print("\n📦 Loading models...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model_a = AutoModelForCausalLM.from_pretrained("gpt2")
    model_b = AutoModelForCausalLM.from_pretrained("gpt2")  # Same model for testing
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "To make a sandwich, you need",
        "The sky is blue because",
        "Water freezes at",
        "The largest planet is",
        "If all birds can fly and penguins are birds, then",
        "The opposite of hot is",
        "Shakespeare wrote"
    ]
    
    print(f"✅ Models loaded, testing with {len(test_prompts)} prompts")
    
    # Test different configurations
    configs = {
        "fastest": OptimizedScoringConfig(
            use_top_k_only=True,
            top_k=50,
            batch_size=8,
            positions_per_prompt=16,
            max_length=128
        ),
        "balanced": OptimizedScoringConfig(
            use_top_k_only=True,
            top_k=100,
            batch_size=4,
            positions_per_prompt=32,
            max_length=256
        ),
        "accurate": OptimizedScoringConfig(
            use_top_k_only=False,
            batch_size=2,
            positions_per_prompt=64,
            max_length=512
        )
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n📊 Testing '{config_name}' configuration:")
        print(f"   - Top-k only: {config.use_top_k_only}")
        print(f"   - Top-k: {config.top_k}")
        print(f"   - Batch size: {config.batch_size}")
        print(f"   - Positions: {config.positions_per_prompt}")
        
        scorer = OptimizedTeacherForcedScorer(config)
        
        # Warmup
        _ = scorer.score_batch(model_a, model_b, test_prompts[:2], tokenizer)
        
        # Time the scoring
        start = time.time()
        scores = scorer.score_batch(model_a, model_b, test_prompts, tokenizer)
        elapsed = time.time() - start
        
        per_prompt = elapsed / len(test_prompts)
        
        results[config_name] = {
            "total_time": elapsed,
            "per_prompt": per_prompt,
            "scores": scores
        }
        
        print(f"   ⏱️  Total time: {elapsed:.3f}s")
        print(f"   ⏱️  Per prompt: {per_prompt:.3f}s ({per_prompt*1000:.0f}ms)")
        print(f"   📈 Mean score: {np.mean(scores):.6f}")
        
        scorer.clear_cache()
    
    # Test FastScorer
    print(f"\n📊 Testing FastScorer (simplified):")
    fast_scorer = FastScorer(k=32, top_k=50)
    
    start = time.time()
    fast_scores = []
    for prompt in test_prompts:
        score = fast_scorer.score(model_a, model_b, prompt, tokenizer)
        fast_scores.append(score)
    elapsed = time.time() - start
    
    per_prompt = elapsed / len(test_prompts)
    print(f"   ⏱️  Total time: {elapsed:.3f}s")
    print(f"   ⏱️  Per prompt: {per_prompt:.3f}s ({per_prompt*1000:.0f}ms)")
    print(f"   📈 Mean score: {np.mean(fast_scores):.6f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY:")
    print("-" * 60)
    
    baseline_time = 1.0  # Original ~1s per query
    
    for config_name, result in results.items():
        per_prompt = result["per_prompt"]
        speedup = baseline_time / per_prompt
        print(f"{config_name:10} : {per_prompt*1000:6.0f}ms per prompt ({speedup:5.1f}x speedup)")
    
    # Fast scorer
    fast_per_prompt = elapsed / len(test_prompts)
    fast_speedup = baseline_time / fast_per_prompt
    print(f"{'fast':10} : {fast_per_prompt*1000:6.0f}ms per prompt ({fast_speedup:5.1f}x speedup)")
    
    # Check if we met the target
    print("\n🎯 TARGET PERFORMANCE (<300ms per query):")
    best_time = min(r["per_prompt"] for r in results.values())
    if best_time < 0.3:
        print(f"✅ ACHIEVED: Best time {best_time*1000:.0f}ms per query")
    else:
        print(f"⚠️  NOT MET: Best time {best_time*1000:.0f}ms per query")
    
    return results


def test_batch_sizes():
    """Test impact of different batch sizes"""
    
    from pot.scoring.optimized_scorer import (
        OptimizedTeacherForcedScorer,
        OptimizedScoringConfig
    )
    
    print("\n" + "=" * 60)
    print("📊 BATCH SIZE IMPACT TEST")
    print("=" * 60)
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Generate more test prompts
    test_prompts = [f"Test prompt number {i}: The answer is" for i in range(32)]
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        config = OptimizedScoringConfig(
            use_top_k_only=True,
            top_k=50,
            batch_size=batch_size,
            positions_per_prompt=16,
            max_length=128
        )
        
        scorer = OptimizedTeacherForcedScorer(config)
        
        start = time.time()
        scores = scorer.score_batch(model, model, test_prompts, tokenizer)
        elapsed = time.time() - start
        
        per_prompt = elapsed / len(test_prompts)
        print(f"Batch size {batch_size:2}: {elapsed:.3f}s total, {per_prompt*1000:.0f}ms per prompt")
        
        scorer.clear_cache()


if __name__ == "__main__":
    # Run performance tests
    results = test_optimized_scorer()
    
    # Test batch size impact
    test_batch_sizes()
    
    print("\n✅ Performance testing complete!")