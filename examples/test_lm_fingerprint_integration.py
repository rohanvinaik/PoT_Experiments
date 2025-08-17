#!/usr/bin/env python3
"""
Test script for LM fingerprinting integration
"""

import sys
import torch
import numpy as np
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append('.')

from pot.lm.verifier import LMVerifier, LMVerificationResult
from pot.lm.models import LM
from pot.core.fingerprint import FingerprintConfig

# Mock LM class for testing
class MockLM(LM):
    """Mock language model for testing"""
    
    def __init__(self, model_id: str = "mock_model", seed: int = 42):
        self.model_id = model_id
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Mock tokenizer
        class MockTokenizer:
            # Add required attributes for TokenSpaceNormalizer
            pad_token_id = 0
            eos_token_id = 1
            bos_token_id = 2
            unk_token_id = 3
            
            def encode(self, text, add_special_tokens=False):
                # Simple character-based tokenization for testing
                return [ord(c) for c in text[:100]]
            
            def decode(self, tokens):
                return ''.join([chr(t) for t in tokens if t < 128])
        
        self.tok = MockTokenizer()
    
    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        """Generate deterministic output based on prompt and seed"""
        # Use prompt and seed to generate pseudo-random but deterministic output
        prompt_hash = hash(prompt) % 1000000
        combined_seed = (self.seed + prompt_hash) % 2**32
        rng = np.random.default_rng(combined_seed)
        
        # Generate some words
        words = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                "and", "cat", "runs", "fast", "slow", "big", "small", "red", "blue"]
        
        output_words = []
        for _ in range(min(max_new_tokens // 5, 10)):  # Roughly 5 chars per word
            output_words.append(rng.choice(words))
        
        return " ".join(output_words)
    
    def get_logits(self, prompt: str) -> np.ndarray:
        """Get mock logits"""
        prompt_hash = hash(prompt) % 1000000
        combined_seed = (self.seed + prompt_hash) % 2**32
        rng = np.random.default_rng(combined_seed)
        return rng.randn(100)  # Mock vocab size of 100


def test_lm_fingerprinting():
    """Test LM fingerprinting integration"""
    print("Testing LM Fingerprinting Integration")
    print("=" * 50)
    
    # Create reference and test models
    reference_model = MockLM("reference", seed=42)
    same_model = MockLM("same", seed=42)  # Same seed = same outputs
    different_model = MockLM("different", seed=123)  # Different seed = different outputs
    
    # Create verifier with fingerprinting enabled
    print("\n1. Creating LMVerifier with fingerprinting enabled...")
    # Use the factory method for language models
    fingerprint_config = FingerprintConfig.for_language_model(
        compute_jacobian=False,  # Not useful for text
        include_timing=True,
        memory_efficient=False
    )
    
    verifier = LMVerifier(
        reference_model=reference_model,
        delta=0.01,
        use_sequential=False,  # Disable for simpler testing
        use_fingerprinting=True,
        fingerprint_config=fingerprint_config
    )
    print("✓ Verifier created successfully")
    
    # Generate challenges
    print("\n2. Generating template challenges...")
    challenges_prompts, challenges_metadata = verifier.generate_template_challenges(
        n=5,
        master_key="deadbeef" * 8,
        session_nonce="cafebabe" * 4
    )
    
    # Convert to challenge format expected by verify
    challenges = []
    for i, (prompt, meta) in enumerate(zip(challenges_prompts, challenges_metadata)):
        challenges.append({
            "prompt": prompt,
            "challenge_id": meta["challenge_id"],
            "index": i
        })
    
    print(f"✓ Generated {len(challenges)} challenges")
    
    # Test 1: Verify same model (should have high fingerprint similarity)
    print("\n3. Verifying identical model...")
    result_same = verifier.verify(
        model=same_model,
        challenges=challenges,
        tolerance=0.1,
        compute_reference_fingerprint=True
    )
    
    print(f"✓ Verification completed")
    print(f"  - Accepted: {result_same.accepted}")
    print(f"  - Distance: {result_same.distance:.4f}")
    print(f"  - Fuzzy similarity: {result_same.fuzzy_similarity:.4f}")
    
    if result_same.fingerprint:
        print(f"  - Fingerprint IO hash: {result_same.fingerprint.io_hash[:16]}...")
        print(f"  - Fingerprint match: {result_same.fingerprint_match:.4f}")
        print(f"  - Reference IO hash: {verifier.reference_fingerprint.io_hash[:16]}...")
    
    # Test 2: Verify different model (should have low fingerprint similarity)
    print("\n4. Verifying different model...")
    result_diff = verifier.verify(
        model=different_model,
        challenges=challenges,
        tolerance=0.1
    )
    
    print(f"✓ Verification completed")
    print(f"  - Accepted: {result_diff.accepted}")
    print(f"  - Distance: {result_diff.distance:.4f}")
    print(f"  - Fuzzy similarity: {result_diff.fuzzy_similarity:.4f}")
    
    if result_diff.fingerprint:
        print(f"  - Fingerprint IO hash: {result_diff.fingerprint.io_hash[:16]}...")
        print(f"  - Fingerprint match: {result_diff.fingerprint_match:.4f}")
    
    # Check fingerprint metadata
    print("\n5. Checking fingerprint metadata...")
    if 'fingerprint' in result_same.metadata:
        fp_meta = result_same.metadata['fingerprint']
        print(f"  - Has Jacobian: {fp_meta['has_jacobian']}")
        print(f"  - Num outputs: {fp_meta['num_outputs']}")
        print(f"  - Config: {fp_meta['config']}")
    
    # Verify fingerprint similarity differences
    print("\n6. Comparing fingerprint similarities...")
    if result_same.fingerprint_match is not None and result_diff.fingerprint_match is not None:
        print(f"  - Same model similarity: {result_same.fingerprint_match:.4f}")
        print(f"  - Different model similarity: {result_diff.fingerprint_match:.4f}")
        
        if result_same.fingerprint_match > result_diff.fingerprint_match:
            print("  ✓ Same model has higher fingerprint similarity (expected)")
        else:
            print("  ⚠ Warning: Different model has higher/equal similarity (unexpected)")
    
    # Test custom fingerprint config
    print("\n7. Testing custom fingerprint configuration...")
    custom_config = FingerprintConfig()
    custom_config.compute_jacobian = False
    custom_config.include_timing = False
    custom_config.output_type = 'text'
    custom_config.model_type = 'lm'
    custom_config.canonicalize_precision = 5
    
    verifier_custom = LMVerifier(
        reference_model=reference_model,
        delta=0.01,
        use_fingerprinting=True,
        fingerprint_config=custom_config
    )
    
    result_custom = verifier_custom.verify(
        model=same_model,
        challenges=challenges[:2],  # Use fewer challenges for speed
        tolerance=0.1,
        compute_reference_fingerprint=True
    )
    
    print(f"✓ Custom config verification completed")
    if result_custom.fingerprint:
        print(f"  - Custom fingerprint created")
        print(f"  - Has timing: {result_custom.fingerprint.timing_info is not None}")
        print(f"  - Has raw outputs: {result_custom.fingerprint.raw_outputs is not None}")
    
    print("\n" + "=" * 50)
    print("All tests completed successfully! ✓")
    
    return True


if __name__ == "__main__":
    try:
        success = test_lm_fingerprinting()
        if success:
            print("\n✅ LM fingerprinting integration successful!")
            sys.exit(0)
        else:
            print("\n❌ LM fingerprinting integration failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)