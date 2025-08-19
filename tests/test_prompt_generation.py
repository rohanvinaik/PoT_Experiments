import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.challenges.prompt_generator import (
    DeterministicPromptGenerator,
    create_prompt_challenges
)

def test_deterministic_generation():
    """Test that same key/namespace/index produces same prompt"""
    key1 = b"test_key_123"
    key2 = b"test_key_123"
    
    gen1 = DeterministicPromptGenerator(key1)
    gen2 = DeterministicPromptGenerator(key2)
    
    prompts1 = gen1.batch_generate("test_namespace", 0, 10)
    prompts2 = gen2.batch_generate("test_namespace", 0, 10)
    
    assert prompts1 == prompts2, "Same inputs should produce same prompts"
    print("✅ Deterministic generation test passed")

def test_different_seeds():
    """Test that different indices produce different prompts"""
    key = b"test_key"
    gen = DeterministicPromptGenerator(key)
    
    prompts = gen.batch_generate("test", 0, 100)
    unique_prompts = set(prompts)
    
    assert len(unique_prompts) > 85, f"Should have >85% unique prompts, got {len(unique_prompts)}"
    print(f"✅ Different seeds test passed: {len(unique_prompts)}/100 unique prompts")

def test_prompt_length():
    """Test that prompts respect max length"""
    key = b"test_key"
    gen = DeterministicPromptGenerator(key)
    
    for i in range(100):
        seed = gen.derive_seed("test", i)
        prompt = gen.seed_to_prompt(seed, max_length=50)
        assert len(prompt) <= 50, f"Prompt exceeds max length: {len(prompt)}"
    
    print("✅ Prompt length test passed")

def test_paraphrase_consistency():
    """Test that paraphrases maintain semantic consistency"""
    key = b"test_key"
    gen = DeterministicPromptGenerator(key)
    
    seed = gen.derive_seed("paraphrase_test", 0)
    paraphrases1 = gen.generate_paraphrased_set(seed, 5)
    paraphrases2 = gen.generate_paraphrased_set(seed, 5)
    
    assert paraphrases1 == paraphrases2, "Same seed should produce same paraphrases"
    print("✅ Paraphrase consistency test passed")
    print(f"   Sample paraphrases: {paraphrases1[:2]}")

def test_namespace_isolation():
    """Test that different namespaces produce different prompts"""
    key = b"test_key"
    gen = DeterministicPromptGenerator(key)
    
    prompts_ns1 = gen.batch_generate("namespace1", 0, 10)
    prompts_ns2 = gen.batch_generate("namespace2", 0, 10)
    
    # Should be different
    assert prompts_ns1 != prompts_ns2, "Different namespaces should produce different prompts"
    
    # Count how many are different
    differences = sum(1 for p1, p2 in zip(prompts_ns1, prompts_ns2) if p1 != p2)
    assert differences >= 8, f"Expected most prompts to differ, got {differences}/10"
    
    print(f"✅ Namespace isolation test passed: {differences}/10 prompts differ")

def test_smart_truncation():
    """Test that smart truncation preserves readability"""
    key = b"test_key"
    gen = DeterministicPromptGenerator(key)
    
    # Generate some prompts with very short max length
    for i in range(20):
        seed = gen.derive_seed("truncation_test", i)
        prompt = gen.seed_to_prompt(seed, max_length=30)
        
        assert len(prompt) <= 30, f"Prompt exceeds max length: {len(prompt)}"
        
        # Check that truncation looks reasonable
        if len(prompt) == 30:
            # Should end with punctuation or ellipsis
            assert prompt[-1] in '.?!…' or prompt.endswith('...'), \
                f"Truncated prompt should end properly: '{prompt}'"
    
    print("✅ Smart truncation test passed")

def test_challenge_creation():
    """Test creating POT challenges"""
    master_key = b"pot_master_key"
    family = "test_family"
    params = {"difficulty": "medium", "category": "factual"}
    n_challenges = 5
    
    challenges = create_prompt_challenges(master_key, family, params, n_challenges)
    
    assert len(challenges) == n_challenges, f"Should create {n_challenges} challenges"
    
    # Check challenge structure
    for i, challenge in enumerate(challenges):
        assert challenge["id"] == f"{family}_{i:06d}"
        assert challenge["type"] == "prompt"
        assert "content" in challenge
        assert "seed" in challenge
        assert "metadata" in challenge
        assert challenge["metadata"]["family"] == family
        assert challenge["metadata"]["index"] == i
        assert len(challenge["content"]) <= 100
    
    # Check determinism
    challenges2 = create_prompt_challenges(master_key, family, params, n_challenges)
    assert challenges == challenges2, "Same inputs should produce same challenges"
    
    print("✅ Challenge creation test passed")
    print(f"   Sample challenge: {challenges[0]['content'][:50]}...")

def test_template_variety():
    """Test that templates cover different categories"""
    key = b"test_key"
    gen = DeterministicPromptGenerator(key)
    
    # Generate many prompts to see variety
    prompts = gen.batch_generate("variety_test", 0, 200)
    
    # Check for different question types
    factual_keywords = ["What is", "Explain", "Define"]
    reasoning_keywords = ["Compare", "If"]
    creative_keywords = ["Write", "Describe"]
    instruction_keywords = ["List", "How do you"]
    analysis_keywords = ["implications", "Analyze"]
    
    categories_found = {
        "factual": any(any(kw in p for kw in factual_keywords) for p in prompts),
        "reasoning": any(any(kw in p for kw in reasoning_keywords) for p in prompts),
        "creative": any(any(kw in p for kw in creative_keywords) for p in prompts),
        "instruction": any(any(kw in p for kw in instruction_keywords) for p in prompts),
        "analysis": any(any(kw in p for kw in analysis_keywords) for p in prompts)
    }
    
    categories_present = sum(categories_found.values())
    assert categories_present >= 3, f"Should have at least 3 categories, found {categories_present}"
    
    print(f"✅ Template variety test passed: {categories_present}/5 categories found")
    print(f"   Categories: {[k for k, v in categories_found.items() if v]}")

def run_all_tests():
    """Run all tests"""
    print("\n🧪 Running Prompt Generation Tests\n")
    
    test_functions = [
        test_deterministic_generation,
        test_different_seeds,
        test_prompt_length,
        test_paraphrase_consistency,
        test_namespace_isolation,
        test_smart_truncation,
        test_challenge_creation,
        test_template_variety
    ]
    
    for test_func in test_functions:
        try:
            test_func()
        except AssertionError as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            return False
    
    print("\n✅ All prompt generation tests passed!")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)