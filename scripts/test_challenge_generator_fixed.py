#!/usr/bin/env python3
"""
Test the fixed challenge generator implementation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.challenges.prompt_generator import DeterministicPromptGenerator, make_prompt_generator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_unified_api():
    """Test the unified API signature"""
    
    master_key = b"test_key_for_challenge_generation_123"
    generator = DeterministicPromptGenerator(master_key)
    
    logger.info("Testing unified API...")
    
    # Test generate_challenges with required kwargs
    challenges = generator.generate_challenges(
        "gpt2", "distilgpt2",
        n=5,
        namespace="test_namespace", 
        seed=42
    )
    
    logger.info(f"Generated {len(challenges)} challenges")
    
    # Verify structure
    assert len(challenges) == 5, f"Expected 5 challenges, got {len(challenges)}"
    
    for i, challenge in enumerate(challenges):
        assert "prompt" in challenge, "Challenge missing 'prompt' field"
        assert "family" in challenge, "Challenge missing 'family' field"
        assert "idx" in challenge, "Challenge missing 'idx' field"
        assert "ref_model" in challenge, "Challenge missing 'ref_model' field"
        assert "cand_model" in challenge, "Challenge missing 'cand_model' field"
        assert "namespace" in challenge, "Challenge missing 'namespace' field"
        
        assert challenge["idx"] == i, f"Expected idx {i}, got {challenge['idx']}"
        assert challenge["ref_model"] == "gpt2", f"Expected ref_model 'gpt2', got {challenge['ref_model']}"
        assert challenge["cand_model"] == "distilgpt2", f"Expected cand_model 'distilgpt2', got {challenge['cand_model']}"
        assert challenge["namespace"] == "test_namespace", f"Expected namespace 'test_namespace', got {challenge['namespace']}"
        
        logger.info(f"Challenge {i}: {challenge['prompt'][:50]}...")
    
    logger.info("✅ Unified API test passed")
    return True

def test_determinism():
    """Test that generation is deterministic"""
    
    master_key = b"determinism_test_key_456"
    generator = DeterministicPromptGenerator(master_key)
    
    logger.info("Testing determinism...")
    
    # Generate same challenges twice
    challenges1 = generator.generate_challenges(
        "model_a", "model_b",
        n=3,
        namespace="determinism_test",
        seed=123
    )
    
    challenges2 = generator.generate_challenges(
        "model_a", "model_b", 
        n=3,
        namespace="determinism_test",
        seed=123
    )
    
    # Should be identical
    assert len(challenges1) == len(challenges2), "Challenge counts differ"
    
    for c1, c2 in zip(challenges1, challenges2):
        assert c1["prompt"] == c2["prompt"], f"Prompts differ: {c1['prompt']} vs {c2['prompt']}"
        assert c1["family"] == c2["family"], f"Families differ: {c1['family']} vs {c2['family']}"
        
    logger.info("✅ Same parameters produce identical results")
    
    # Different seeds should produce different results
    challenges3 = generator.generate_challenges(
        "model_a", "model_b",
        n=3,
        namespace="determinism_test",
        seed=456  # Different seed
    )
    
    different_prompts = 0
    for c1, c3 in zip(challenges1, challenges3):
        if c1["prompt"] != c3["prompt"]:
            different_prompts += 1
    
    assert different_prompts > 0, "Different seeds should produce different prompts"
    logger.info(f"✅ Different seed produced {different_prompts}/{len(challenges1)} different prompts")
    
    logger.info("✅ Determinism test passed")
    return True

def test_template_structure():
    """Test template structure and slot filling"""
    
    master_key = b"template_test_key_789"
    generator = DeterministicPromptGenerator(master_key)
    
    logger.info("Testing template structure...")
    
    # Generate many challenges to see variety
    challenges = generator.generate_challenges(
        "ref", "cand",
        n=20,
        namespace="template_test",
        seed=999
    )
    
    families_seen = set()
    prompt_patterns = set()
    
    for challenge in challenges:
        families_seen.add(challenge["family"])
        
        # Extract pattern (basic pattern recognition)
        prompt = challenge["prompt"]
        if "Explain" in prompt and "in" in prompt:
            prompt_patterns.add("explain_style")
        elif "What is" in prompt:
            prompt_patterns.add("what_is")
        elif "Translate to" in prompt:
            prompt_patterns.add("translate")
        
        logger.info(f"Family: {challenge['family']}, Prompt: {prompt}")
    
    logger.info(f"Families seen: {families_seen}")
    logger.info(f"Patterns seen: {prompt_patterns}")
    
    # Should see multiple families and patterns
    assert len(families_seen) > 1, f"Expected multiple families, only saw: {families_seen}"
    assert len(prompt_patterns) > 1, f"Expected multiple patterns, only saw: {prompt_patterns}"
    
    logger.info("✅ Template structure test passed")
    return True

def test_backward_compatibility():
    """Test backward compatibility wrapper"""
    
    master_key = b"backward_compat_test_key"
    generator = DeterministicPromptGenerator(master_key)
    
    logger.info("Testing backward compatibility...")
    
    # Test callable interface
    prompt1 = generator()
    prompt2 = generator()
    
    assert isinstance(prompt1, str), f"Expected string, got {type(prompt1)}"
    assert isinstance(prompt2, str), f"Expected string, got {type(prompt2)}"
    assert len(prompt1) > 0, "Prompt should not be empty"
    assert len(prompt2) > 0, "Prompt should not be empty"
    
    logger.info(f"Generated prompts: '{prompt1}' and '{prompt2}'")
    
    # Test factory function
    prompt_fn = make_prompt_generator(master_key, "test_namespace")
    
    prompt3 = prompt_fn()
    assert isinstance(prompt3, str), f"Expected string, got {type(prompt3)}"
    assert len(prompt3) > 0, "Prompt should not be empty"
    
    # Test access to generator
    assert hasattr(prompt_fn, 'generator'), "Factory should attach generator"
    challenges = prompt_fn.generator.generate_challenges(
        "test_ref", "test_cand",
        n=2, namespace="factory_test", seed=777
    )
    assert len(challenges) == 2, f"Expected 2 challenges, got {len(challenges)}"
    
    logger.info(f"Factory function prompt: '{prompt3}'")
    logger.info("✅ Backward compatibility test passed")
    return True

def test_edge_cases():
    """Test edge cases and error handling"""
    
    master_key = b"edge_case_test_key"
    generator = DeterministicPromptGenerator(master_key)
    
    logger.info("Testing edge cases...")
    
    # Test with n=0
    challenges = generator.generate_challenges(
        "ref", "cand",
        n=0,
        namespace="edge_test",
        seed=1
    )
    assert len(challenges) == 0, f"Expected 0 challenges, got {len(challenges)}"
    
    # Test with n=1
    challenges = generator.generate_challenges(
        "ref", "cand", 
        n=1,
        namespace="edge_test",
        seed=2
    )
    assert len(challenges) == 1, f"Expected 1 challenge, got {len(challenges)}"
    
    # Test with empty strings
    challenges = generator.generate_challenges(
        "", "",
        n=1,
        namespace="",
        seed=3
    )
    assert len(challenges) == 1, "Should handle empty strings"
    assert challenges[0]["ref_model"] == "", "Should preserve empty ref_model"
    assert challenges[0]["cand_model"] == "", "Should preserve empty cand_model"
    
    logger.info("✅ Edge cases test passed")
    return True

def main():
    """Run all challenge generator tests"""
    logger.info("\n" + "="*70)
    logger.info("CHALLENGE GENERATOR FIXED TESTS")
    logger.info("="*70)
    
    tests = [
        ("Unified API", test_unified_api),
        ("Determinism", test_determinism),
        ("Template Structure", test_template_structure),
        ("Backward Compatibility", test_backward_compatibility),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            logger.info(f"\n--- Testing {name} ---")
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            logger.error(f"Test {name} failed with exception: {e}")
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
        logger.info("\n🎉 ALL CHALLENGE GENERATOR TESTS PASSED!")
        logger.info("Key achievements:")
        logger.info("  • Unified API signature implemented")
        logger.info("  • Deterministic generation working")
        logger.info("  • Template structure and slot filling functional")
        logger.info("  • Backward compatibility maintained")
        logger.info("  • Edge cases handled properly")
    else:
        logger.info("\n⚠️ Some tests failed - review output above")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())