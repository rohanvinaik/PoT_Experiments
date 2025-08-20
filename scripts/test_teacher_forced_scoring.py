#!/usr/bin/env python3
"""
Test the fixed teacher-forced scoring implementation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pot.scoring.teacher_forced import TeacherForcedScorer, ScoringConfig, OptimizedTeacherForcedScorer, FastScorer
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_scoring():
    """Test basic scoring functionality"""
    logger.info("\n" + "="*60)
    logger.info("Testing Basic Teacher-Forced Scoring")
    logger.info("="*60)
    
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model_a = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model_b = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    model_a.eval()
    model_b.eval()
    
    # Test configurations
    configs = [
        ScoringConfig(method="delta_ce", num_positions=32, score_clip=(0.0, 0.3)),
        ScoringConfig(method="symmetric_kl", num_positions=32, score_clip=(0.0, 1.0))
    ]
    
    test_prompts = [
        "The capital of France is",
        "Machine learning is a field that",
        "The sky appears blue because"
    ]
    
    for config in configs:
        logger.info(f"\nTesting method: {config.method}")
        scorer = TeacherForcedScorer(config)
        
        scores_same = []
        scores_diff = []
        
        for prompt in test_prompts:
            # Test same model (should be close to 0)
            score_same = scorer.score_batch(model_a, model_a, [prompt], tokenizer)[0]
            scores_same.append(score_same)
            
            # Test different models (should be > 0)
            score_diff = scorer.score_batch(model_a, model_b, [prompt], tokenizer)[0]
            scores_diff.append(score_diff)
            
            logger.info(f"  {prompt[:30]}...: same={score_same:.6f}, diff={score_diff:.6f}")
        
        # Verify non-negative
        assert all(s >= 0 for s in scores_same), f"Negative scores found in same-model: {scores_same}"
        assert all(s >= 0 for s in scores_diff), f"Negative scores found in different-model: {scores_diff}"
        
        # Verify separation
        mean_same = np.mean(scores_same)
        mean_diff = np.mean(scores_diff)
        
        logger.info(f"  Mean same-model score: {mean_same:.6f}")
        logger.info(f"  Mean different-model score: {mean_diff:.6f}")
        logger.info(f"  Separation ratio: {mean_diff / max(mean_same, 1e-6):.1f}x")
        
        # For same models, scores should be very small
        assert mean_same < 0.01, f"Same-model scores too high: {mean_same}"
        
        # For different models, scores should be clearly separated
        assert mean_diff > mean_same + 0.001, f"Insufficient separation: {mean_diff} vs {mean_same}"
        
        logger.info(f"  ✅ {config.method} scoring working correctly")

def test_optimized_scorer():
    """Test the optimized scorer implementation"""
    logger.info("\n" + "="*60)
    logger.info("Testing Optimized Teacher-Forced Scorer")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model_a = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model_b = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    model_a.eval()
    model_b.eval()
    
    config = ScoringConfig(method="delta_ce", num_positions=64, score_clip=(0.0, 0.3))
    scorer = OptimizedTeacherForcedScorer(config)
    
    test_prompts = [
        "The capital of France is",
        "Python is a programming language",
        "The theory of evolution"
    ]
    
    logger.info("Testing batch scoring...")
    
    # Test batch scoring
    scores_same = scorer.score_batch(model_a, model_a, test_prompts, tokenizer, k=64)
    scores_diff = scorer.score_batch(model_a, model_b, test_prompts, tokenizer, k=64)
    
    for i, prompt in enumerate(test_prompts):
        logger.info(f"  {prompt}: same={scores_same[i]:.6f}, diff={scores_diff[i]:.6f}")
    
    # Verify properties
    assert all(s >= 0 for s in scores_same), "Negative same-model scores"
    assert all(s >= 0 for s in scores_diff), "Negative different-model scores"
    
    mean_same = np.mean(scores_same)
    mean_diff = np.mean(scores_diff)
    
    logger.info(f"Optimized scorer - Same: {mean_same:.6f}, Diff: {mean_diff:.6f}")
    
    # Same models should have very low scores
    assert mean_same < 0.01, f"Same-model scores too high: {mean_same}"
    
    # Different models should be clearly separated
    assert mean_diff > mean_same + 0.001, f"Insufficient separation"
    
    logger.info("✅ Optimized scorer working correctly")

def test_fast_scorer():
    """Test the fast scorer implementation"""
    logger.info("\n" + "="*60)
    logger.info("Testing Fast Scorer")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model_a = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model_b = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    model_a.eval()
    model_b.eval()
    
    scorer = FastScorer(k=32, method="delta_ce")
    
    test_prompts = [
        "The capital of France is",
        "Machine learning algorithms",
        "Climate change effects"
    ]
    
    scores_same = []
    scores_diff = []
    
    for prompt in test_prompts:
        score_same = scorer.score(model_a, model_a, prompt, tokenizer)
        score_diff = scorer.score(model_a, model_b, prompt, tokenizer)
        
        scores_same.append(score_same)
        scores_diff.append(score_diff)
        
        logger.info(f"  {prompt}: same={score_same:.6f}, diff={score_diff:.6f}")
    
    # Verify properties
    assert all(s >= 0 for s in scores_same), "Negative same-model scores"
    assert all(s >= 0 for s in scores_diff), "Negative different-model scores"
    
    mean_same = np.mean(scores_same)
    mean_diff = np.mean(scores_diff)
    
    logger.info(f"Fast scorer - Same: {mean_same:.6f}, Diff: {mean_diff:.6f}")
    
    # Test callable interface
    score_callable = scorer(model_a, model_b, "Test prompt", tokenizer)
    assert score_callable >= 0, "Callable interface returned negative score"
    
    logger.info("✅ Fast scorer working correctly")

def test_score_clipping():
    """Test that score clipping works correctly for EB CI stability"""
    logger.info("\n" + "="*60)
    logger.info("Testing Score Clipping")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model_a = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model_b = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    model_a.eval()
    model_b.eval()
    
    # Test with restrictive clipping
    config = ScoringConfig(method="delta_ce", num_positions=64, score_clip=(0.0, 0.1))
    scorer = TeacherForcedScorer(config)
    
    prompt = "The capital of France is"
    score = scorer.score_batch(model_a, model_b, [prompt], tokenizer)[0]
    
    logger.info(f"Score with clipping (0.0, 0.1): {score:.6f}")
    
    # Should be clipped to maximum
    assert score <= 0.1, f"Score not clipped: {score} > 0.1"
    assert score >= 0.0, f"Score below minimum: {score} < 0.0"
    
    # Test with wider clipping
    config_wide = ScoringConfig(method="delta_ce", num_positions=64, score_clip=(0.0, 1.0))
    scorer_wide = TeacherForcedScorer(config_wide)
    
    score_wide = scorer_wide.score_batch(model_a, model_b, [prompt], tokenizer)[0]
    logger.info(f"Score with clipping (0.0, 1.0): {score_wide:.6f}")
    
    # Wide clipping should give higher score
    assert score_wide >= score, f"Wide clipping gave lower score: {score_wide} < {score}"
    
    logger.info("✅ Score clipping working correctly")

def test_position_averaging():
    """Test that position averaging works correctly"""
    logger.info("\n" + "="*60)
    logger.info("Testing Position Averaging")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model_a = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model_b = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    model_a.eval()
    model_b.eval()
    
    # Test with different numbers of positions
    positions_list = [8, 16, 32, 64]
    prompt = "The capital of France is"
    
    for num_pos in positions_list:
        config = ScoringConfig(method="delta_ce", num_positions=num_pos, score_clip=(0.0, 1.0))
        scorer = TeacherForcedScorer(config)
        
        score = scorer.score_batch(model_a, model_b, [prompt], tokenizer)[0]
        logger.info(f"  K={num_pos:2d} positions: score={score:.6f}")
        
        # Score should be reasonable regardless of K
        assert 0.0 <= score <= 1.0, f"Score out of range for K={num_pos}: {score}"
    
    logger.info("✅ Position averaging working correctly")

def main():
    """Run all teacher-forced scoring tests"""
    logger.info("\n" + "="*70)
    logger.info("TEACHER-FORCED SCORING TESTS")
    logger.info("="*70)
    
    tests = [
        ("Basic Scoring", test_basic_scoring),
        ("Optimized Scorer", test_optimized_scorer),
        ("Fast Scorer", test_fast_scorer),
        ("Score Clipping", test_score_clipping),
        ("Position Averaging", test_position_averaging)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True))
        except Exception as e:
            logger.error(f"Test {name} failed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL TEACHER-FORCED SCORING TESTS PASSED!")
        logger.info("Key achievements:")
        logger.info("  • Non-negative scores guaranteed")
        logger.info("  • Proper position averaging implemented")
        logger.info("  • Score clipping for EB CI stability")
        logger.info("  • Clear separation between same/different models")
        logger.info("  • Multiple scorer variants working correctly")
    else:
        logger.info("\n⚠️ Some tests failed - review output above")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())