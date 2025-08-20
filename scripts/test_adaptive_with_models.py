#!/usr/bin/env python3
"""
Test adaptive challenge generation with actual models.

This script demonstrates the complete pipeline with vocabulary adaptation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import secrets
from pot.lm.models import LM
from pot.lm.verifier import LMVerifier
from pot.core.adaptive_challenge import AdaptiveChallengeConfig


def test_adaptive_verification():
    """Test adaptive challenge generation with real model verification."""
    
    print("=" * 70)
    print("Adaptive Challenge Generation with Model Verification")
    print("=" * 70)
    
    # Test configuration
    model_pairs = [
        ("gpt2", "distilgpt2", "Same family, same vocabulary"),
        ("gpt2", "microsoft/phi-2", "Different sizes, high overlap")
    ]
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    for model_a_name, model_b_name, description in model_pairs:
        print(f"\n{'='*70}")
        print(f"Testing: {model_a_name} vs {model_b_name}")
        print(f"Description: {description}")
        print("-" * 70)
        
        try:
            # Load models
            print(f"\n📥 Loading models...")
            model_a = LM(model_a_name, device=device)
            model_b = LM(model_b_name, device=device)
            
            # Get vocabulary sizes
            vocab_a = model_a.get_vocab_size() if hasattr(model_a, 'get_vocab_size') else len(model_a.tok.get_vocab())
            vocab_b = model_b.get_vocab_size() if hasattr(model_b, 'get_vocab_size') else len(model_b.tok.get_vocab())
            
            print(f"  {model_a_name}: {vocab_a:,} tokens")
            print(f"  {model_b_name}: {vocab_b:,} tokens")
            print(f"  Overlap: {min(vocab_a, vocab_b) / max(vocab_a, vocab_b):.1%}")
            
            # Create verifier with adaptive challenges
            print(f"\n🔧 Initializing verifier with adaptive challenges...")
            verifier = LMVerifier(
                reference_model=model_a,
                delta=0.01,
                use_sequential=False,
                enable_adaptive_challenges=True
            )
            
            # Generate adaptive challenges
            print(f"\n🎯 Generating adaptive challenges...")
            master_key = secrets.token_hex(32)
            session_nonce = secrets.token_hex(16)
            
            challenge_result = verifier.generate_adaptive_challenges(
                candidate_model=model_b,
                n=5,  # Small number for testing
                master_key=master_key,
                session_nonce=session_nonce
            )
            
            if challenge_result.get("error"):
                print(f"  ❌ Error generating challenges: {challenge_result['error']}")
                continue
            
            print(f"  ✓ Generated {len(challenge_result.get('challenges', []))} challenges")
            
            # Display adaptation info
            if challenge_result.get("vocabulary_adapted"):
                print(f"\n  📊 Vocabulary Adaptation:")
                vocab_analysis = challenge_result.get("vocabulary_analysis", {})
                print(f"    Overlap: {vocab_analysis.get('overlap_ratio', 0):.1%}")
                print(f"    Shared tokens: {vocab_analysis.get('shared_tokens', 0):,}")
                print(f"    Method: {vocab_analysis.get('adaptation_method', 'unknown')}")
                print(f"    Strategy: {challenge_result.get('adaptation_strategy', 'unknown')}")
                
                if challenge_result.get("quality_metrics"):
                    metrics = challenge_result["quality_metrics"]
                    print(f"\n  ⭐ Quality Metrics:")
                    print(f"    Token coverage: {metrics.get('token_coverage', 0):.1%}")
                    print(f"    Diversity: {metrics.get('diversity_score', 0):.2f}")
            
            # Extract prompts from challenges
            prompts = []
            for challenge in challenge_result.get("challenges", []):
                if "prompt" in challenge.parameters:
                    prompts.append(challenge.parameters["prompt"])
                elif "template" in challenge.parameters and "slot_values" in challenge.parameters:
                    # Fill template
                    template = challenge.parameters["template"]
                    for slot, value in challenge.parameters["slot_values"].items():
                        template = template.replace(f"{{{slot}}}", value)
                    prompts.append(template)
            
            if not prompts:
                print("  ⚠️ No prompts extracted from challenges")
                continue
            
            # Run verification
            print(f"\n🔍 Running verification with {len(prompts)} challenges...")
            
            result = verifier.verify(
                model=model_b,
                challenges=prompts[:3],  # Use first 3 for quick test
                tolerance=0.3,
                method='fuzzy'
            )
            
            print(f"\n📈 Verification Results:")
            print(f"  Accepted: {'✅ Yes' if result.accepted else '❌ No'}")
            print(f"  Distance: {result.distance:.4f}")
            print(f"  Confidence radius: {result.confidence_radius:.4f}")
            print(f"  Fuzzy similarity: {result.fuzzy_similarity:.2%}")
            
            # Show example outputs
            print(f"\n📝 Example Challenge-Response:")
            example_prompt = prompts[0]
            print(f"  Prompt: {example_prompt}")
            
            output_a = model_a.generate(example_prompt, max_new_tokens=20)
            output_b = model_b.generate(example_prompt, max_new_tokens=20)
            
            print(f"  {model_a_name}: {output_a[:50]}...")
            print(f"  {model_b_name}: {output_b[:50]}...")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()


def test_cross_family_adaptation():
    """Test adaptation between different model families."""
    
    print("\n" + "=" * 70)
    print("Cross-Family Vocabulary Adaptation Test")
    print("=" * 70)
    
    from pot.core.adaptive_challenge import AdaptiveChallengeGenerator
    
    # Create generator with relaxed constraints for cross-family
    generator = AdaptiveChallengeGenerator(
        min_overlap_ratio=0.60,  # Allow 60% overlap for cross-family
        core_vocab_size=20000,   # Focus on smaller core vocabulary
        enable_frequency_weighting=False
    )
    
    # Test configurations for different families
    test_cases = [
        (50257, 32768, "GPT-2", "Mistral-7B", "GPT vs Mistral family"),
        (50257, 32000, "GPT-2", "LLaMA", "GPT vs LLaMA family"),
        (32768, 32000, "Mistral", "Zephyr", "Mistral fine-tuning"),
    ]
    
    master_key = secrets.token_hex(32)
    session_nonce = secrets.token_hex(16)
    
    for vocab_a, vocab_b, model_a, model_b, description in test_cases:
        print(f"\n{'='*70}")
        print(f"{description}: {model_a} ({vocab_a:,}) vs {model_b} ({vocab_b:,})")
        print("-" * 70)
        
        config = AdaptiveChallengeConfig(
            master_key_hex=master_key,
            session_nonce_hex=session_nonce,
            n=10,
            family="lm:templates",
            params={
                "templates": [
                    "The {subject} is {attribute}.",
                    "Q: What is {concept}? A:"
                ],
                "slots": {
                    "subject": ["sun", "ocean", "mountain"],
                    "attribute": ["bright", "deep", "tall"],
                    "concept": ["gravity", "energy", "time"]
                }
            },
            vocab_size_a=vocab_a,
            vocab_size_b=vocab_b,
            model_name_a=model_a,
            model_name_b=model_b,
            adaptation_strategy="shared_core",
            min_challenges=5
        )
        
        result = generator.generate_adaptive_challenges(config)
        
        if result.get("error"):
            print(f"❌ Cannot adapt: {result['error']}")
        else:
            print(f"✅ Successfully adapted challenges")
            
            if result.get("vocabulary_adapted"):
                print(f"\n📊 Adaptation Details:")
                print(f"  Strategy: {result.get('adaptation_strategy', 'unknown')}")
                print(f"  Token coverage: {result.get('token_coverage', 0):.1%}")
                
                if result.get("shared_token_range"):
                    start, end = result["shared_token_range"]
                    print(f"  Shared range: [{start:,}, {end:,})")
                    print(f"  Shared tokens: {end - start:,}")
                
                vocab_analysis = result.get("vocabulary_analysis", {})
                print(f"  Overlap ratio: {vocab_analysis.get('overlap_ratio', 0):.1%}")
                print(f"  Confidence adjustment: {vocab_analysis.get('confidence_adjustment', 1.0):.2f}")
            
            elif result.get("fallback_used"):
                print(f"\n⚠️ Fallback Strategy Used:")
                print(f"  Reason: {result.get('fallback_reason', 'unknown')}")
                print(f"  Basic tokens: {result.get('basic_token_limit', 0):,}")
                print(f"  Reduced challenges: {result.get('reduced_challenges', 0)}")
            
            # Show example challenges
            if result.get("challenges"):
                print(f"\n📝 Example Challenges Generated:")
                for i, challenge in enumerate(result["challenges"][:3]):
                    if "prompt" in challenge.parameters:
                        print(f"  {i+1}. {challenge.parameters['prompt']}")
    
    # Show statistics
    print(f"\n📊 Overall Statistics:")
    stats = generator.get_adaptation_statistics()
    print(f"  Total adaptations: {stats['total_adaptations']}")
    print(f"  Success rate: {stats.get('success_rate', 0):.1%}")
    print(f"  Fallback rate: {stats.get('fallback_rate', 0):.1%}")


if __name__ == "__main__":
    # Run tests
    print("🚀 Starting Adaptive Challenge Tests with Models\n")
    
    # Test with actual models (if available)
    try:
        test_adaptive_verification()
    except ImportError as e:
        print(f"⚠️ Skipping model tests: {e}")
    
    # Test cross-family adaptation (doesn't need actual models)
    test_cross_family_adaptation()
    
    print("\n" + "=" * 70)
    print("✅ Adaptive Challenge Tests Complete")
    print("=" * 70)