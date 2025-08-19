#!/usr/bin/env python3
"""Demo script showing teacher-forced scoring for POT verification"""

import sys
import os
import torch
import numpy as np
from unittest.mock import Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.scoring.teacher_forced import (
    TeacherForcedScorer,
    FastTeacherForcedScorer,
    ScoringConfig,
    create_teacher_forced_challenges
)

def simulate_model_outputs(vocab_size: int = 1000, seq_len: int = 20):
    """Create mock model outputs for demonstration"""
    class MockModel:
        def __init__(self, seed: int = 42):
            torch.manual_seed(seed)
            self.base_logits = torch.randn(1, seq_len, vocab_size)
        
        def __call__(self, **kwargs):
            # Handle batch inputs
            if "input_ids" in kwargs:
                batch_size = kwargs["input_ids"].shape[0]
                seq_len = kwargs["input_ids"].shape[1]
            else:
                batch_size = 1
                seq_len = 20
            
            # Create logits for the batch
            logits = self.base_logits.repeat(batch_size, 1, 1)[:, :seq_len, :]
            logits = logits + torch.randn_like(logits) * 0.1
            return Mock(logits=logits)
    
    return MockModel

def mock_tokenizer():
    """Create a mock tokenizer for demonstration"""
    def tokenizer(text, **kwargs):
        if isinstance(text, str):
            text = [text]
        
        batch_size = len(text)
        seq_len = max(len(t.split()) + 5 for t in text)  # Approximate token count
        
        return {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len)
        }
    
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    return tokenizer

def main():
    print("🎯 POT Teacher-Forced Scoring Demo\n")
    print("=" * 60)
    
    # Create mock models
    print("\n1️⃣ Setting up Models")
    print("-" * 40)
    reference_model = simulate_model_outputs()(42)
    candidate_model_similar = simulate_model_outputs()(43)  # Similar
    candidate_model_different = simulate_model_outputs()(100)  # Different
    
    tokenizer = mock_tokenizer()
    print("   ✅ Created reference and candidate models")
    print("   ✅ Created mock tokenizer")
    
    # Demo different scoring methods
    print("\n2️⃣ Scoring Methods Comparison")
    print("-" * 40)
    
    prompt = "What is artificial intelligence?"
    prompt_type = "factual"
    
    methods = ["delta_ce", "symmetric_kl", "js_divergence"]
    
    for method in methods:
        config = ScoringConfig(method=method, num_positions=5)
        scorer = TeacherForcedScorer(config)
        
        # Score similar model
        result_similar = scorer.score_models(
            reference_model, candidate_model_similar,
            prompt, prompt_type, tokenizer
        )
        
        # Score different model
        result_different = scorer.score_models(
            reference_model, candidate_model_different,
            prompt, prompt_type, tokenizer
        )
        
        print(f"\n   Method: {method}")
        print(f"      Similar model score: {result_similar.score:.4f}")
        print(f"      Different model score: {result_different.score:.4f}")
        print(f"      Per-position scores (similar): {[f'{s:.3f}' for s in result_similar.per_position_scores[:3]]}")
    
    # Demo canonical suffixes
    print("\n3️⃣ Canonical Suffixes")
    print("-" * 40)
    config = ScoringConfig()
    scorer = TeacherForcedScorer(config)
    
    prompt_types = ["factual", "reasoning", "creative", "instruction", "analysis"]
    for ptype in prompt_types:
        suffix = scorer.canonical_suffixes[ptype]
        print(f"   {ptype:12s}: '{suffix}'")
    
    # Demo batch scoring
    print("\n4️⃣ Batch Scoring")
    print("-" * 40)
    
    prompts = [
        "What is machine learning?",
        "Explain quantum computing",
        "How does photosynthesis work?"
    ]
    prompt_types = ["factual", "reasoning", "factual"]
    
    config = ScoringConfig(method="delta_ce", num_positions=5)
    scorer = TeacherForcedScorer(config)
    
    results = scorer.batch_score(
        reference_model, candidate_model_similar,
        prompts, prompt_types, tokenizer,
        batch_size=2
    )
    
    print(f"   Processed {len(results)} prompts in batch")
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"   {i+1}. '{prompt[:30]}...' score: {result.score:.4f}")
    
    # Demo fast scorer with caching
    print("\n5️⃣ Fast Scorer with Caching")
    print("-" * 40)
    
    config = ScoringConfig(method="symmetric_kl")
    fast_scorer = FastTeacherForcedScorer(config)
    
    # First call - populates cache
    inputs = tokenizer("Test prompt for caching")
    model = simulate_model_outputs()()
    
    import time
    start = time.time()
    result1 = fast_scorer._get_topk_logprobs(model, inputs, 2, 5)
    time1 = time.time() - start
    
    # Second call - uses cache
    start = time.time()
    result2 = fast_scorer._get_topk_logprobs(model, inputs, 2, 5)
    time2 = time.time() - start
    
    print(f"   First call (no cache): {time1*1000:.2f}ms")
    print(f"   Second call (cached): {time2*1000:.2f}ms")
    print(f"   Speedup: {time1/time2:.1f}x")
    print(f"   Cache size: {len(fast_scorer.cache)} entries")
    
    # Demo challenge creation
    print("\n6️⃣ POT Challenge Creation")
    print("-" * 40)
    
    challenge_prompts = [
        "What is deep learning?",
        "Explain neural networks",
        "Describe backpropagation"
    ]
    challenge_types = ["factual", "reasoning", "analysis"]
    
    config = ScoringConfig(
        method="js_divergence",
        num_positions=10,
        temperature=0.8
    )
    
    challenges = create_teacher_forced_challenges(
        challenge_prompts, challenge_types, config
    )
    
    print(f"   Created {len(challenges)} challenges:")
    for challenge in challenges:
        print(f"\n   ID: {challenge['id']}")
        print(f"   Prompt: '{challenge['prompt'][:30]}...'")
        print(f"   Type: {challenge['prompt_type']}")
        print(f"   Suffix: '{challenge['canonical_suffix']}'")
        print(f"   Method: {challenge['config']['method']}")
        print(f"   Positions: {challenge['config']['num_positions']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Demo complete! Teacher-forced scoring provides:")
    print("   • Fast evaluation without generation")
    print("   • Multiple scoring methods (ΔCE, KL, JS)")
    print("   • Canonical suffixes for consistency")
    print("   • Batch processing for efficiency")
    print("   • Caching for repeated evaluations")
    print("   • Lower variance than generation-based scoring")

if __name__ == "__main__":
    main()