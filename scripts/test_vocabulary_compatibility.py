#!/usr/bin/env python3
"""
Test script for vocabulary compatibility analysis system.

Demonstrates how the framework handles models with different vocabulary sizes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pot.core.vocabulary_compatibility import (
    VocabularyCompatibilityAnalyzer,
    VocabularyMismatchBehavior
)


def test_vocabulary_compatibility():
    """Test various vocabulary size combinations"""
    
    print("=" * 70)
    print("Vocabulary Compatibility Analysis Test")
    print("=" * 70)
    
    # Initialize analyzer with default settings
    analyzer = VocabularyCompatibilityAnalyzer(
        min_overlap_ratio=0.95,
        mismatch_behavior=VocabularyMismatchBehavior.ADAPT,
        allow_extended_vocabularies=True
    )
    
    # Test cases: (vocab_a, vocab_b, model_a_name, model_b_name)
    test_cases = [
        # Identical vocabularies
        (50257, 50257, "GPT-2", "GPT-2"),
        
        # Near-identical (GPT family with minor extensions)
        (50257, 50260, "GPT-2", "GPT-2-extended"),
        
        # Same family, different sizes
        (50257, 50257, "GPT-2", "DistilGPT-2"),
        
        # Different families but high overlap
        (32000, 32768, "LLaMA", "Mistral"),
        
        # Significantly different
        (50257, 32000, "GPT-2", "LLaMA"),
        
        # Phi vs GPT-2
        (50257, 51200, "GPT-2", "Phi-2"),
        
        # Mistral vs Zephyr (real case)
        (32768, 32000, "Mistral-7B", "Zephyr-7B"),
        
        # BERT family
        (30522, 30523, "BERT-base", "BERT-cased"),
        
        # Large difference
        (50257, 100000, "GPT-2", "Custom-Large"),
    ]
    
    for vocab_a, vocab_b, model_a, model_b in test_cases:
        print(f"\n{'='*70}")
        print(f"Testing: {model_a} ({vocab_a:,} tokens) vs {model_b} ({vocab_b:,} tokens)")
        print("-" * 70)
        
        # Analyze compatibility
        report = analyzer.analyze_vocabulary_overlap(
            vocab_a, vocab_b, model_a, model_b
        )
        
        # Print report
        print(report)
        
        # Get and display strategy
        strategy = analyzer.suggest_verification_strategy(report)
        print(f"\nVerification Strategy:")
        print(f"  Can proceed: {strategy['can_proceed']}")
        print(f"  Method: {strategy['method']}")
        print(f"  Confidence adjustment: {strategy['confidence_adjustment']:.2f}")
        
        if strategy['notes']:
            print(f"  Notes:")
            for note in strategy['notes']:
                print(f"    - {note}")


def test_different_behaviors():
    """Test different mismatch behaviors"""
    
    print("\n" + "=" * 70)
    print("Testing Different Mismatch Behaviors")
    print("=" * 70)
    
    behaviors = [
        VocabularyMismatchBehavior.WARN,
        VocabularyMismatchBehavior.ADAPT,
        VocabularyMismatchBehavior.FAIL
    ]
    
    # Test case with vocabulary mismatch
    vocab_a, vocab_b = 32768, 32000
    model_a, model_b = "Mistral-7B", "Zephyr-7B"
    
    for behavior in behaviors:
        print(f"\n{'='*70}")
        print(f"Behavior: {behavior.value.upper()}")
        print("-" * 70)
        
        analyzer = VocabularyCompatibilityAnalyzer(
            min_overlap_ratio=0.95,
            mismatch_behavior=behavior,
            allow_extended_vocabularies=True
        )
        
        report = analyzer.analyze_vocabulary_overlap(
            vocab_a, vocab_b, model_a, model_b
        )
        
        strategy = analyzer.suggest_verification_strategy(report)
        
        print(f"Vocabulary sizes: {vocab_a:,} vs {vocab_b:,}")
        print(f"Overlap: {report.overlap_ratio:.1%}")
        print(f"Can proceed: {strategy['can_proceed']}")
        print(f"Strategy: {strategy['method']}")
        
        if behavior == VocabularyMismatchBehavior.FAIL and report.size_difference > 0:
            print("  → Strict mode: Failed due to vocabulary mismatch")
        elif behavior == VocabularyMismatchBehavior.ADAPT:
            print("  → Adaptive mode: Adjusting verification for mismatch")
        else:
            print("  → Warning mode: Proceeding with warnings")


def test_real_model_integration():
    """Test with actual model loading (if models are available)"""
    
    print("\n" + "=" * 70)
    print("Testing Real Model Integration")
    print("=" * 70)
    
    try:
        from pot.scoring.diff_scorer import CorrectedDifferenceScorer
        import torch
        
        # Create scorer with vocabulary compatibility
        scorer = CorrectedDifferenceScorer(
            min_vocab_overlap=0.95,
            vocab_mismatch_behavior="adapt",
            allow_extended_vocabularies=True
        )
        
        print("\n✓ CorrectedDifferenceScorer initialized with vocabulary compatibility")
        print(f"  Min overlap ratio: {scorer.vocab_analyzer.min_overlap_ratio:.1%}")
        print(f"  Mismatch behavior: {scorer.vocab_analyzer.mismatch_behavior.value}")
        print(f"  Allow extensions: {scorer.vocab_analyzer.allow_extended_vocabularies}")
        
        # Test with mock logits of different sizes
        print("\n" + "-" * 70)
        print("Testing scorer with mismatched vocabulary sizes:")
        
        # Simulate logits from models with different vocab sizes
        batch_size, seq_len = 2, 10
        
        # GPT-2 sized logits
        logits_gpt2 = torch.randn(batch_size, seq_len, 50257)
        
        # Mistral sized logits
        logits_mistral = torch.randn(batch_size, seq_len, 32768)
        
        # Phi-2 sized logits
        logits_phi = torch.randn(batch_size, seq_len, 51200)
        
        # Test GPT-2 vs Phi-2 (close sizes)
        print("\nGPT-2 (50,257) vs Phi-2 (51,200):")
        score = scorer.delta_ce_abs(logits_gpt2, logits_phi)
        print(f"  Score: {score:.4f}")
        
        # Test GPT-2 vs Mistral (different families)
        print("\nGPT-2 (50,257) vs Mistral (32,768):")
        score = scorer.delta_ce_abs(logits_gpt2, logits_mistral)
        print(f"  Score: {score:.4f}")
        
    except Exception as e:
        print(f"\n⚠ Could not test real integration: {e}")


if __name__ == "__main__":
    # Run all tests
    test_vocabulary_compatibility()
    test_different_behaviors()
    test_real_model_integration()
    
    print("\n" + "=" * 70)
    print("✅ Vocabulary Compatibility Tests Complete")
    print("=" * 70)