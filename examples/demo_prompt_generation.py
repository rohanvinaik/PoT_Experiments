#!/usr/bin/env python3
"""Demo script showing deterministic prompt generation for POT verification"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.challenges.prompt_generator import (
    DeterministicPromptGenerator,
    create_prompt_challenges
)

def main():
    print("🎯 POT Deterministic Prompt Generation Demo\n")
    print("=" * 60)
    
    # Initialize generator with a master key
    master_key = b"demo_master_key_12345"
    generator = DeterministicPromptGenerator(master_key)
    
    # Demo 1: Basic prompt generation
    print("\n1️⃣ Basic Prompt Generation")
    print("-" * 40)
    namespace = "demo_namespace"
    prompts = generator.batch_generate(namespace, 0, 5)
    for i, prompt in enumerate(prompts):
        print(f"   Prompt {i}: {prompt}")
    
    # Demo 2: Show determinism
    print("\n2️⃣ Demonstrating Determinism")
    print("-" * 40)
    generator2 = DeterministicPromptGenerator(master_key)
    prompts2 = generator2.batch_generate(namespace, 0, 3)
    print("   First generator:")
    for i, p in enumerate(prompts[:3]):
        print(f"      {i}: {p}")
    print("   Second generator (same key):")
    for i, p in enumerate(prompts2):
        print(f"      {i}: {p}")
    print(f"   ✅ Identical: {prompts[:3] == prompts2}")
    
    # Demo 3: Different namespaces
    print("\n3️⃣ Different Namespaces")
    print("-" * 40)
    ns1_prompts = generator.batch_generate("namespace_A", 0, 3)
    ns2_prompts = generator.batch_generate("namespace_B", 0, 3)
    print("   Namespace A:")
    for p in ns1_prompts:
        print(f"      {p}")
    print("   Namespace B:")
    for p in ns2_prompts:
        print(f"      {p}")
    
    # Demo 4: Paraphrase generation
    print("\n4️⃣ Paraphrase Generation")
    print("-" * 40)
    seed = generator.derive_seed("paraphrase_demo", 0)
    paraphrases = generator.generate_paraphrased_set(seed, 5)
    print("   Semantically similar prompts:")
    for i, p in enumerate(paraphrases):
        print(f"      {i+1}. {p}")
    
    # Demo 5: Length control
    print("\n5️⃣ Length Control")
    print("-" * 40)
    seed = generator.derive_seed("length_demo", 0)
    short_prompt = generator.seed_to_prompt(seed, max_length=30)
    medium_prompt = generator.seed_to_prompt(seed, max_length=60)
    long_prompt = generator.seed_to_prompt(seed, max_length=100)
    print(f"   30 chars: {short_prompt} (len={len(short_prompt)})")
    print(f"   60 chars: {medium_prompt} (len={len(medium_prompt)})")
    print(f"   100 chars: {long_prompt} (len={len(long_prompt)})")
    
    # Demo 6: POT Challenge Creation
    print("\n6️⃣ POT Challenge Creation")
    print("-" * 40)
    challenges = create_prompt_challenges(
        master_key=master_key,
        family="demo_family",
        params={"difficulty": "medium", "type": "mixed"},
        n_challenges=3
    )
    
    for challenge in challenges:
        print(f"\n   Challenge ID: {challenge['id']}")
        print(f"   Content: {challenge['content']}")
        print(f"   Seed: {challenge['seed'][:16]}...")
        print(f"   Length: {challenge['metadata']['length']} chars")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete! The prompt generator provides:")
    print("   • Deterministic generation from KDF-derived seeds")
    print("   • Diverse templates across multiple categories")
    print("   • Namespace isolation for different challenge families")
    print("   • Smart truncation for length control")
    print("   • Paraphrase generation for semantic consistency testing")

if __name__ == "__main__":
    main()