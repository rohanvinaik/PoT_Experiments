#!/usr/bin/env python3
"""
Test script for adaptive challenge generation with different model pairs.

Demonstrates how the framework adapts challenges to handle vocabulary differences.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import secrets
from pot.core.adaptive_challenge import (
    AdaptiveChallengeGenerator,
    AdaptiveChallengeConfig
)


def test_adaptive_generation():
    """Test adaptive challenge generation with various model pairs."""
    
    print("=" * 70)
    print("Adaptive Challenge Generation Test")
    print("=" * 70)
    
    # Initialize generator
    generator = AdaptiveChallengeGenerator(
        min_overlap_ratio=0.95,
        core_vocab_size=30000,
        enable_frequency_weighting=False
    )
    
    # Test keys
    master_key = secrets.token_hex(32)
    session_nonce = secrets.token_hex(16)
    
    # Test cases: (vocab_a, vocab_b, model_a, model_b, expected_behavior)
    test_cases = [
        # Identical vocabularies - should use standard generation
        (50257, 50257, "GPT-2", "GPT-2", "standard"),
        
        # GPT-2 vs GPT-2-medium (same family)
        (50257, 50257, "GPT-2", "GPT-2-medium", "standard"),
        
        # GPT-2 vs Phi-2 (close sizes, high overlap)
        (50257, 51200, "GPT-2", "Phi-2", "adapted"),
        
        # GPT-2 vs Mistral (different families)
        (50257, 32768, "GPT-2", "Mistral-7B", "adapted"),
        
        # Mistral vs Zephyr (close sizes)
        (32768, 32000, "Mistral-7B", "Zephyr-7B", "adapted"),
        
        # Large difference (should use fallback)
        (50257, 100000, "GPT-2", "Custom-Large", "fallback"),
        
        # Very small overlap
        (30522, 51200, "BERT", "Phi-2", "adapted")
    ]
    
    for vocab_a, vocab_b, model_a, model_b, expected in test_cases:
        print(f"\n{'='*70}")
        print(f"Testing: {model_a} ({vocab_a:,}) vs {model_b} ({vocab_b:,})")
        print(f"Expected behavior: {expected}")
        print("-" * 70)
        
        # Create adaptive config
        config = AdaptiveChallengeConfig(
            master_key_hex=master_key,
            session_nonce_hex=session_nonce,
            n=10,
            family="lm:templates",
            params={
                "templates": [
                    "The {adjective} {subject} {verb} the {object}.",
                    "Q: What is {concept}? A:",
                    "Complete: The {object} is {attribute}."
                ],
                "slots": {
                    "adjective": ["clever", "curious", "colorful"],
                    "subject": ["cat", "scientist", "robot"],
                    "verb": ["chases", "observes", "creates"],
                    "object": ["ball", "puzzle", "painting"],
                    "concept": ["gravity", "democracy", "entropy"],
                    "attribute": ["fast", "complex", "beautiful"]
                }
            },
            vocab_size_a=vocab_a,
            vocab_size_b=vocab_b,
            model_name_a=model_a,
            model_name_b=model_b,
            adaptation_strategy="shared_core"
        )
        
        # Generate adaptive challenges
        result = generator.generate_adaptive_challenges(config)
        
        # Display results
        if result.get("error"):
            print(f"❌ Error: {result['error']}")
            print(f"   Details: {result.get('details', 'N/A')}")
        else:
            print(f"✓ Challenge generation successful")
            print(f"  Challenge ID: {result.get('challenge_id', 'N/A')[:16]}...")
            print(f"  Number of challenges: {len(result.get('challenges', []))}")
            
            # Check adaptation
            if result.get("vocabulary_adapted"):
                print(f"\n  📊 Vocabulary Adaptation Applied:")
                vocab_analysis = result.get("vocabulary_analysis", {})
                print(f"    - Overlap ratio: {vocab_analysis.get('overlap_ratio', 0):.1%}")
                print(f"    - Shared tokens: {vocab_analysis.get('shared_tokens', 0):,}")
                print(f"    - Method: {vocab_analysis.get('adaptation_method', 'unknown')}")
                print(f"    - Confidence adjustment: {vocab_analysis.get('confidence_adjustment', 1.0):.2f}")
                
                print(f"\n  📈 Adaptation Details:")
                print(f"    - Strategy: {result.get('adaptation_strategy', 'unknown')}")
                print(f"    - Token coverage: {result.get('token_coverage', 0):.1%}")
                if result.get("shared_token_range"):
                    start, end = result["shared_token_range"]
                    print(f"    - Token range: [{start:,}, {end:,})")
                
                # Quality metrics
                if result.get("quality_metrics"):
                    metrics = result["quality_metrics"]
                    print(f"\n  ⭐ Quality Metrics:")
                    print(f"    - Token coverage: {metrics.get('token_coverage', 0):.1%}")
                    print(f"    - Diversity score: {metrics.get('diversity_score', 0):.2f}")
                    print(f"    - Frequency alignment: {metrics.get('frequency_alignment', 0):.2f}")
                    print(f"    - Challenge validity: {metrics.get('challenge_validity', 0):.1f}")
            
            elif result.get("fallback_used"):
                print(f"\n  ⚠️ Fallback Strategy Used:")
                print(f"    - Reason: {result.get('fallback_reason', 'unknown')}")
                print(f"    - Reduced challenges: {result.get('reduced_challenges', 0)}")
                print(f"    - Basic token limit: {result.get('basic_token_limit', 0):,}")
            
            else:
                print(f"\n  ℹ️ Standard generation (no adaptation needed)")
            
            # Show first challenge as example
            if result.get("challenges"):
                first_challenge = result["challenges"][0]
                print(f"\n  📝 Example Challenge:")
                print(f"    ID: {first_challenge.challenge_id[:16]}...")
                print(f"    Family: {first_challenge.family}")
                if "prompt" in first_challenge.parameters:
                    print(f"    Prompt: {first_challenge.parameters['prompt']}")
                elif "template" in first_challenge.parameters:
                    print(f"    Template: {first_challenge.parameters['template']}")


def test_adaptation_strategies():
    """Test different adaptation strategies."""
    
    print("\n" + "=" * 70)
    print("Testing Different Adaptation Strategies")
    print("=" * 70)
    
    strategies = ["shared_core", "common_tokens", "frequency_weighted"]
    
    # Test with GPT-2 vs Mistral (significant difference)
    vocab_a, vocab_b = 50257, 32768
    model_a, model_b = "GPT-2", "Mistral-7B"
    
    master_key = secrets.token_hex(32)
    session_nonce = secrets.token_hex(16)
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy.upper()}")
        print("-" * 70)
        
        generator = AdaptiveChallengeGenerator(
            min_overlap_ratio=0.95,
            core_vocab_size=30000,
            enable_frequency_weighting=(strategy == "frequency_weighted")
        )
        
        config = AdaptiveChallengeConfig(
            master_key_hex=master_key,
            session_nonce_hex=session_nonce,
            n=5,
            family="lm:templates",
            params={
                "templates": ["The {subject} {verb} the {object}."],
                "slots": {
                    "subject": ["cat", "dog"],
                    "verb": ["chases", "sees"],
                    "object": ["ball", "bird"]
                }
            },
            vocab_size_a=vocab_a,
            vocab_size_b=vocab_b,
            model_name_a=model_a,
            model_name_b=model_b,
            adaptation_strategy=strategy,
            min_challenges=3
        )
        
        result = generator.generate_adaptive_challenges(config)
        
        print(f"Vocabulary sizes: {vocab_a:,} vs {vocab_b:,}")
        
        if result.get("vocabulary_adapted"):
            print(f"✓ Adapted using {strategy} strategy")
            print(f"  Token coverage: {result.get('token_coverage', 0):.1%}")
            if result.get("shared_token_range"):
                start, end = result["shared_token_range"]
                print(f"  Shared range: [{start:,}, {end:,})")
                print(f"  Range size: {end - start:,} tokens")
        elif result.get("fallback_used"):
            print(f"⚠️ Fallback used: {result.get('fallback_reason', 'unknown')}")
        else:
            print(f"ℹ️ Standard generation used")
        
        # Show statistics
        stats = generator.get_adaptation_statistics()
        if stats["total_adaptations"] > 0:
            print(f"\n  Statistics:")
            print(f"    Success rate: {stats['success_rate']:.1%}")
            print(f"    Fallback rate: {stats['fallback_rate']:.1%}")


def test_quality_metrics():
    """Test challenge quality metrics."""
    
    print("\n" + "=" * 70)
    print("Testing Challenge Quality Metrics")
    print("=" * 70)
    
    generator = AdaptiveChallengeGenerator(
        min_overlap_ratio=0.95,
        core_vocab_size=30000,
        enable_frequency_weighting=True
    )
    
    # Create mock frequency data
    import numpy as np
    
    # Simulate token frequencies (higher frequency for lower token IDs)
    token_frequencies = {}
    for token_id in range(50000):
        # Zipf-like distribution
        token_frequencies[token_id] = 1.0 / (token_id + 1) ** 0.8
    
    # Normalize
    total_freq = sum(token_frequencies.values())
    token_frequencies = {k: v/total_freq for k, v in token_frequencies.items()}
    
    master_key = secrets.token_hex(32)
    session_nonce = secrets.token_hex(16)
    
    config = AdaptiveChallengeConfig(
        master_key_hex=master_key,
        session_nonce_hex=session_nonce,
        n=20,
        family="lm:templates",
        params={
            "templates": [
                "The {adjective} {subject} {verb} the {object}.",
                "When {subject} {verb}, {object} becomes {adjective}."
            ],
            "slots": {
                "adjective": ["clever", "curious", "colorful", "mysterious"],
                "subject": ["cat", "scientist", "robot", "artist"],
                "verb": ["chases", "observes", "creates", "discovers"],
                "object": ["ball", "puzzle", "painting", "equation"]
            }
        },
        vocab_size_a=50257,
        vocab_size_b=32000,
        model_name_a="GPT-2",
        model_name_b="LLaMA",
        adaptation_strategy="frequency_weighted"
    )
    
    # Generate with frequency weighting
    result = generator.generate_adaptive_challenges(config, token_frequencies)
    
    print(f"\nGenerating challenges for GPT-2 (50,257) vs LLaMA (32,000)")
    print(f"Using frequency-weighted adaptation\n")
    
    if result.get("quality_metrics"):
        metrics = result["quality_metrics"]
        print("📊 Quality Metrics:")
        print("-" * 40)
        
        # Visual representation
        def show_bar(label, value, max_val=1.0, width=30):
            filled = int(value * width / max_val)
            bar = "█" * filled + "░" * (width - filled)
            return f"{label:20} [{bar}] {value:.1%}"
        
        print(show_bar("Token Coverage", metrics.get("token_coverage", 0)))
        print(show_bar("Diversity Score", metrics.get("diversity_score", 0)))
        print(show_bar("Frequency Alignment", metrics.get("frequency_alignment", 0)))
        print(show_bar("Challenge Validity", metrics.get("challenge_validity", 0)))
        
        # Overall quality score
        quality_score = np.mean([
            metrics.get("token_coverage", 0),
            metrics.get("diversity_score", 0),
            metrics.get("frequency_alignment", 0),
            metrics.get("challenge_validity", 0)
        ])
        print("-" * 40)
        print(show_bar("Overall Quality", quality_score))
        
        # Interpretation
        print("\n📋 Interpretation:")
        if quality_score >= 0.8:
            print("  ✅ Excellent adaptation quality")
        elif quality_score >= 0.6:
            print("  ✓ Good adaptation quality")
        elif quality_score >= 0.4:
            print("  ⚠️ Moderate adaptation quality")
        else:
            print("  ❌ Poor adaptation quality - consider fallback")


if __name__ == "__main__":
    # Run all tests
    test_adaptive_generation()
    test_adaptation_strategies()
    test_quality_metrics()
    
    print("\n" + "=" * 70)
    print("✅ Adaptive Challenge Generation Tests Complete")
    print("=" * 70)