import torch
import pytest
from unittest.mock import Mock, MagicMock
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.scoring.teacher_forced import (
    TeacherForcedScorer,
    FastTeacherForcedScorer,
    ScoringConfig,
    ScoringResult,
    create_teacher_forced_challenges
)

def test_delta_ce_scoring():
    """Test delta cross-entropy scoring"""
    config = ScoringConfig(method="delta_ce", num_positions=5)
    scorer = TeacherForcedScorer(config)
    
    # Mock models with known outputs
    ref_model = Mock()
    cand_model = Mock()
    
    # Create mock logits
    vocab_size = 100
    seq_len = 10
    
    ref_logits = torch.randn(1, seq_len, vocab_size)
    cand_logits = ref_logits.clone()
    cand_logits += torch.randn_like(cand_logits) * 0.1  # Small perturbation
    
    ref_output = Mock(logits=ref_logits)
    cand_output = Mock(logits=cand_logits)
    ref_model.return_value = ref_output
    cand_model.return_value = cand_output
    
    # Mock tokenizer
    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": torch.randint(0, vocab_size, (1, seq_len)),
        "attention_mask": torch.ones(1, seq_len)
    }
    tokenizer.side_effect = lambda text, **kwargs: {
        "input_ids": torch.randint(0, vocab_size, (1, len(text.split()) + 3)),
        "attention_mask": torch.ones(1, len(text.split()) + 3)
    }
    
    # Score should be small for similar models
    result = scorer.score_models(ref_model, cand_model, "Test prompt", "factual", tokenizer)
    assert 0 <= result.score <= 1
    assert isinstance(result.per_position_scores, list)
    print(f"✅ Delta CE scoring test passed: score={result.score:.4f}")

def test_symmetric_kl_scoring():
    """Test symmetric KL divergence scoring"""
    config = ScoringConfig(method="symmetric_kl", num_positions=5)
    scorer = TeacherForcedScorer(config)
    
    # Test with identical distributions
    ref_logprobs = torch.log_softmax(torch.randn(5, 100), dim=-1)
    cand_logprobs = ref_logprobs.clone()
    
    result = scorer._compute_symmetric_kl(ref_logprobs, cand_logprobs)
    assert result.score < 0.01  # Should be near zero for identical
    assert len(result.per_position_scores) == 5
    print(f"✅ Symmetric KL test passed: identical distributions score={result.score:.6f}")
    
    # Test with different distributions
    cand_logprobs_diff = torch.log_softmax(torch.randn(5, 100), dim=-1)
    result_diff = scorer._compute_symmetric_kl(ref_logprobs, cand_logprobs_diff)
    assert result_diff.score > result.score  # Should be larger for different distributions
    print(f"   Different distributions score={result_diff.score:.4f}")

def test_js_divergence_scoring():
    """Test Jensen-Shannon divergence scoring"""
    config = ScoringConfig(method="js_divergence", num_positions=5)
    scorer = TeacherForcedScorer(config)
    
    # Test with identical distributions
    ref_logprobs = torch.log_softmax(torch.randn(5, 100), dim=-1)
    cand_logprobs = ref_logprobs.clone()
    
    result = scorer._compute_js_divergence(ref_logprobs, cand_logprobs)
    assert result.score < 0.01  # Should be near zero for identical
    assert 0 <= result.score <= 1  # JS divergence normalized to [0,1]
    print(f"✅ JS divergence test passed: identical distributions score={result.score:.6f}")
    
    # Test with maximally different distributions
    ref_one_hot = torch.zeros(5, 100)
    ref_one_hot[:, 0] = 1
    cand_one_hot = torch.zeros(5, 100)
    cand_one_hot[:, 99] = 1
    
    ref_logprobs_diff = torch.log(ref_one_hot + 1e-8)
    cand_logprobs_diff = torch.log(cand_one_hot + 1e-8)
    
    result_max = scorer._compute_js_divergence(ref_logprobs_diff, cand_logprobs_diff)
    assert result_max.score > 0.9  # Should be close to 1 for maximally different
    print(f"   Maximally different distributions score={result_max.score:.4f}")

def test_batch_scoring():
    """Test batch scoring efficiency"""
    config = ScoringConfig(method="delta_ce", num_positions=5)
    scorer = TeacherForcedScorer(config)
    
    prompts = ["Test 1", "Test 2", "Test 3"]
    prompt_types = ["factual", "reasoning", "creative"]
    
    # Mock models
    ref_model = Mock()
    cand_model = Mock()
    
    batch_size = 2
    vocab_size = 100
    max_len = 20
    
    # Mock batch outputs
    ref_output = Mock(logits=torch.randn(len(prompts), max_len, vocab_size))
    cand_output = Mock(logits=torch.randn(len(prompts), max_len, vocab_size))
    ref_model.return_value = ref_output
    cand_model.return_value = cand_output
    
    # Mock tokenizer
    tokenizer = Mock()
    def tokenizer_side_effect(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        return {
            "input_ids": torch.randint(0, vocab_size, (batch_size, max_len)),
            "attention_mask": torch.ones(batch_size, max_len)
        }
    tokenizer.side_effect = tokenizer_side_effect
    
    results = scorer.batch_score(
        ref_model, cand_model, prompts, prompt_types,
        tokenizer, batch_size=batch_size
    )
    
    assert len(results) == 3
    assert all(isinstance(r, ScoringResult) for r in results)
    assert all(0 <= r.score <= 1 for r in results)
    print(f"✅ Batch scoring test passed: processed {len(results)} prompts")

def test_canonical_suffixes():
    """Test canonical suffix assignment"""
    config = ScoringConfig()
    scorer = TeacherForcedScorer(config)
    
    assert "The answer is" in scorer.canonical_suffixes["factual"]
    assert "step by step" in scorer.canonical_suffixes["reasoning"]
    assert "Once upon a time" in scorer.canonical_suffixes["creative"]
    assert "how to do it" in scorer.canonical_suffixes["instruction"]
    assert "key points" in scorer.canonical_suffixes["analysis"]
    assert "response" in scorer.canonical_suffixes["default"]
    
    print("✅ Canonical suffixes test passed")
    for ptype, suffix in scorer.canonical_suffixes.items():
        print(f"   {ptype}: '{suffix}'")

def test_fast_scorer_caching():
    """Test that fast scorer properly caches results"""
    config = ScoringConfig(method="symmetric_kl")
    scorer = FastTeacherForcedScorer(config)
    
    # Mock model
    model = Mock()
    vocab_size = 100
    seq_len = 10
    
    model_output = Mock(logits=torch.randn(1, seq_len, vocab_size))
    model.return_value = model_output
    
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    
    # First call should populate cache
    result1 = scorer._get_topk_logprobs(model, inputs, 2, 3)
    assert len(scorer.cache) == 1
    assert model.call_count == 1
    
    # Second call should use cache
    result2 = scorer._get_topk_logprobs(model, inputs, 2, 3)
    assert torch.equal(result1[0], result2[0])
    assert torch.equal(result1[1], result2[1])
    assert model.call_count == 1  # Model only called once
    
    print(f"✅ Fast scorer caching test passed: cache size={len(scorer.cache)}")
    
    # Test cache clearing
    scorer.clear_cache()
    assert len(scorer.cache) == 0
    print("   Cache cleared successfully")

def test_fast_symmetric_kl():
    """Test fast approximate symmetric KL computation"""
    config = ScoringConfig()
    scorer = FastTeacherForcedScorer(config)
    
    # Create top-k results with known overlap
    positions = 3
    top_k = 10
    vocab_size = 100
    
    # Case 1: High overlap (same top tokens)
    ref_indices = torch.arange(top_k).unsqueeze(0).repeat(positions, 1)
    cand_indices = ref_indices.clone()
    cand_indices[:, 5:] = torch.arange(top_k, top_k + 5).unsqueeze(0).repeat(positions, 1)
    
    ref_logprobs = torch.log_softmax(torch.randn(positions, top_k), dim=-1)
    cand_logprobs = torch.log_softmax(torch.randn(positions, top_k), dim=-1)
    
    score = scorer._compute_fast_symmetric_kl(
        (ref_logprobs, ref_indices),
        (cand_logprobs, cand_indices)
    )
    
    assert isinstance(score, float)
    assert score >= 0  # KL divergence is non-negative
    print(f"✅ Fast symmetric KL test passed: score={score:.4f}")
    
    # Case 2: No overlap (completely different tokens)
    cand_indices_no_overlap = ref_indices + vocab_size
    score_no_overlap = scorer._compute_fast_symmetric_kl(
        (ref_logprobs, ref_indices),
        (cand_logprobs, cand_indices_no_overlap)
    )
    
    assert score_no_overlap == 1.0  # Max divergence for no overlap
    print(f"   No overlap score={score_no_overlap:.4f}")

def test_challenge_creation():
    """Test creating teacher-forced challenges"""
    prompts = ["What is AI?", "Explain gravity", "Write a story"]
    prompt_types = ["factual", "reasoning", "creative"]
    config = ScoringConfig(method="delta_ce", num_positions=10)
    
    challenges = create_teacher_forced_challenges(prompts, prompt_types, config)
    
    assert len(challenges) == 3
    
    for i, challenge in enumerate(challenges):
        assert challenge["id"] == f"tf_{i:06d}"
        assert challenge["type"] == "teacher_forced"
        assert challenge["prompt"] == prompts[i]
        assert challenge["prompt_type"] == prompt_types[i]
        assert "canonical_suffix" in challenge
        assert "full_text" in challenge
        assert challenge["full_text"] == prompts[i] + challenge["canonical_suffix"]
        assert challenge["config"]["method"] == "delta_ce"
        assert challenge["config"]["num_positions"] == 10
        assert challenge["metadata"]["index"] == i
    
    print("✅ Challenge creation test passed")
    for challenge in challenges:
        print(f"   {challenge['id']}: '{challenge['prompt'][:20]}...' -> '{challenge['canonical_suffix']}'")

def test_scoring_config():
    """Test ScoringConfig dataclass"""
    # Default config
    config1 = ScoringConfig()
    assert config1.method == "delta_ce"
    assert config1.num_positions == 10
    assert config1.temperature == 1.0
    assert config1.use_canonical_suffix == True
    
    # Custom config
    config2 = ScoringConfig(
        method="js_divergence",
        num_positions=20,
        temperature=0.8,
        use_canonical_suffix=False
    )
    assert config2.method == "js_divergence"
    assert config2.num_positions == 20
    assert config2.temperature == 0.8
    assert config2.use_canonical_suffix == False
    
    print("✅ ScoringConfig test passed")

def test_scoring_result():
    """Test ScoringResult dataclass"""
    result = ScoringResult(
        score=0.75,
        raw_score=3.5,
        per_position_scores=[0.1, 0.2, 0.3],
        metadata={"test": "value"}
    )
    
    assert result.score == 0.75
    assert result.raw_score == 3.5
    assert len(result.per_position_scores) == 3
    assert result.metadata["test"] == "value"
    
    print("✅ ScoringResult test passed")

def run_all_tests():
    """Run all teacher-forced scoring tests"""
    print("\n🧪 Running Teacher-Forced Scoring Tests\n")
    
    test_functions = [
        test_scoring_config,
        test_scoring_result,
        test_canonical_suffixes,
        test_delta_ce_scoring,
        test_symmetric_kl_scoring,
        test_js_divergence_scoring,
        test_batch_scoring,
        test_fast_scorer_caching,
        test_fast_symmetric_kl,
        test_challenge_creation
    ]
    
    for test_func in test_functions:
        try:
            print(f"\nRunning {test_func.__name__}...")
            test_func()
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n✅ All teacher-forced scoring tests passed!")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)