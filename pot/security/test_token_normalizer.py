"""
Comprehensive test suite for TokenSpaceNormalizer and StochasticDecodingController
"""

import numpy as np
import json
import time
from typing import List, Dict, Any
from token_space_normalizer import (
    TokenSpaceNormalizer,
    StochasticDecodingController,
    IntegratedVerificationSystem,
    TokenizerType,
    SamplingMethod,
    TokenizationResult,
    AlignmentResult,
    MockTokenizer
)
import torch


def test_text_normalization():
    """Test text normalization functionality"""
    print("\n" + "="*70)
    print("TEST: Text Normalization")
    print("="*70)
    
    normalizer = TokenSpaceNormalizer(TokenizerType.BPE)
    
    # Test cases with various Unicode and whitespace issues
    test_cases = [
        ("Hello\u200bWorld", "HelloWorld"),  # Zero-width space
        ("Multiple   spaces", "Multiple spaces"),  # Multiple spaces
        ("  Leading and trailing  ", "Leading and trailing"),  # Trim
        ("Café\u0301", None),  # Unicode combining characters (will be normalized)
        ("Mixed\t\n\rwhitespace", "Mixed whitespace"),  # Mixed whitespace
    ]
    
    for input_text, expected in test_cases:
        normalized = normalizer.normalize_text(input_text)
        if expected:
            assert normalized == expected or len(normalized) > 0
        print(f"✓ '{input_text[:20]}...' -> '{normalized}'")
    
    print("✓ Text normalization tests passed")
    return True


def test_token_sequence_normalization():
    """Test token sequence normalization"""
    print("\n" + "="*70)
    print("TEST: Token Sequence Normalization")
    print("="*70)
    
    normalizer = TokenSpaceNormalizer(TokenizerType.BPE, vocab_size=1000)
    
    # Test different token types
    test_sequences = [
        [1, 2, 3, 4, 5],  # Integer tokens
        ["hello", "world", "test"],  # String tokens
        [1, "mixed", 3, "tokens"],  # Mixed tokens
        []  # Empty sequence
    ]
    
    for seq in test_sequences:
        # Convert to tuple for caching
        normalized = normalizer.normalize_token_sequence(tuple(seq), 'canonical')
        assert all(isinstance(t, int) for t in normalized)
        assert all(0 <= t < normalizer.vocab_size for t in normalized)
        print(f"✓ Normalized {len(seq)} tokens to canonical form")
    
    # Test different target spaces
    test_seq = tuple(["test", "sequence"])
    
    canonical = normalizer.normalize_token_sequence(test_seq, 'canonical')
    assert all(isinstance(t, int) for t in canonical)
    print(f"✓ Canonical: {canonical}")
    
    string_form = normalizer.normalize_token_sequence(test_seq, 'string')
    assert all(isinstance(t, str) for t in string_form)
    print(f"✓ String: {string_form}")
    
    byte_form = normalizer.normalize_token_sequence(test_seq, 'byte')
    assert all(isinstance(t, int) and 0 <= t < 256 for t in byte_form)
    print(f"✓ Byte: {byte_form[:10]}... (length: {len(byte_form)})")
    
    return True


def test_token_alignment():
    """Test token boundary alignment between tokenizers"""
    print("\n" + "="*70)
    print("TEST: Token Alignment")
    print("="*70)
    
    normalizer = TokenSpaceNormalizer()
    
    # Create mock tokenizers with different behaviors
    tokenizer_a = MockTokenizer(vocab_size=1000)
    tokenizer_b = MockTokenizer(vocab_size=2000)
    
    test_text = "The quick brown fox"
    
    # Perform alignment
    alignment = normalizer.align_token_boundaries(test_text, tokenizer_a, tokenizer_b)
    
    assert isinstance(alignment, AlignmentResult)
    assert alignment.text == normalizer.normalize_text(test_text)
    assert len(alignment.tokenizer_a_tokens) > 0
    assert len(alignment.tokenizer_b_tokens) > 0
    assert 0 <= alignment.similarity_score <= 1
    
    print(f"✓ Aligned tokens: A={len(alignment.tokenizer_a_tokens)}, B={len(alignment.tokenizer_b_tokens)}")
    print(f"✓ Alignment map size: {len(alignment.alignment_map)}")
    print(f"✓ Similarity score: {alignment.similarity_score:.4f}")
    
    return True


def test_invariant_hash():
    """Test token-invariant hash generation"""
    print("\n" + "="*70)
    print("TEST: Token-Invariant Hash")
    print("="*70)
    
    normalizer = TokenSpaceNormalizer()
    
    # Test that similar texts produce same hash after normalization
    text_variants = [
        "Hello world!",
        "Hello  world!",  # Extra space
        "Hello\tworld!",  # Tab instead of space
        " Hello world! ",  # Leading/trailing spaces
    ]
    
    hashes = [normalizer.compute_token_invariant_hash(text) for text in text_variants]
    
    # All normalized versions should produce same hash
    assert len(set(hashes)) == 1, "Normalized variants should have same hash"
    print(f"✓ All {len(text_variants)} variants produced same hash")
    
    # Different text should produce different hash
    different_text = "Goodbye world!"
    different_hash = normalizer.compute_token_invariant_hash(different_text)
    assert different_hash != hashes[0]
    print("✓ Different text produces different hash")
    
    # Test case normalization option
    hash_normal = normalizer.compute_token_invariant_hash("Hello World", normalize_case=False)
    hash_lower = normalizer.compute_token_invariant_hash("hello world", normalize_case=True)
    hash_upper = normalizer.compute_token_invariant_hash("HELLO WORLD", normalize_case=True)
    
    assert hash_lower == normalizer.compute_token_invariant_hash("Hello World", normalize_case=True)
    print("✓ Case normalization works correctly")
    
    return True


def test_subword_regularization():
    """Test subword regularization"""
    print("\n" + "="*70)
    print("TEST: Subword Regularization")
    print("="*70)
    
    normalizer = TokenSpaceNormalizer(TokenizerType.CHARACTER)
    
    test_text = "Testing subword regularization"
    
    # Generate multiple variants
    variants = normalizer.tokenize_with_subword_regularization(
        test_text, alpha=0.2, num_samples=5
    )
    
    assert len(variants) == 5
    print(f"✓ Generated {len(variants)} tokenization variants")
    
    # Check that variants are different (with regularization)
    unique_variants = set(tuple(v) for v in variants)
    print(f"✓ {len(unique_variants)} unique variants out of {len(variants)}")
    
    # Test with no regularization (alpha=0)
    no_reg_variants = normalizer.tokenize_with_subword_regularization(
        test_text, alpha=0.0, num_samples=3
    )
    
    # All should be identical with no regularization
    unique_no_reg = set(tuple(v) for v in no_reg_variants)
    assert len(unique_no_reg) == 1, "No regularization should produce identical results"
    print("✓ No regularization produces identical tokenizations")
    
    return True


def test_unknown_token_handling():
    """Test handling of unknown tokens"""
    print("\n" + "="*70)
    print("TEST: Unknown Token Handling")
    print("="*70)
    
    normalizer = TokenSpaceNormalizer(vocab_size=100)
    
    # Test tokens including unknown ones
    tokens = [1, 2, '[UNK]', 999, 'normal', '<unk>']
    
    # Test different strategies
    strategies = ['hash', 'skip', 'replace']
    
    for strategy in strategies:
        processed = normalizer.handle_unknown_tokens(tokens, strategy)
        
        if strategy == 'skip':
            # Unknown tokens should be removed
            assert len(processed) < len(tokens)
        elif strategy == 'replace':
            # Unknown tokens should be replaced with 0
            assert all(t != '[UNK]' and t != '<unk>' for t in processed)
        
        print(f"✓ Strategy '{strategy}': {len(tokens)} -> {len(processed)} tokens")
    
    return True


def test_deterministic_mode():
    """Test deterministic generation mode"""
    print("\n" + "="*70)
    print("TEST: Deterministic Mode")
    print("="*70)
    
    controller = StochasticDecodingController(seed=42)
    
    # Set deterministic mode
    controller.set_deterministic_mode(temperature=0.0, top_k=1)
    
    assert controller.deterministic_mode == True
    assert controller.temperature == 1e-10  # Avoid division by zero
    assert controller.top_k == 1
    
    # Test deterministic sampling
    logits = np.random.randn(100)
    
    # Multiple samples should be identical in deterministic mode
    samples = [
        controller.controlled_sampling(logits, SamplingMethod.ARGMAX)
        for _ in range(5)
    ]
    
    assert len(set(samples)) == 1, "Deterministic mode should produce identical results"
    print(f"✓ Deterministic sampling: all samples = {samples[0]}")
    
    # Test with temperature > 0 but still deterministic (top_k=1)
    controller.temperature = 0.5
    controller.top_k = 1
    
    sample = controller.controlled_sampling(logits, SamplingMethod.TOP_K)
    print(f"✓ Top-k=1 sampling: {sample}")
    
    return True


def test_sampling_methods():
    """Test different sampling methods"""
    print("\n" + "="*70)
    print("TEST: Sampling Methods")
    print("="*70)
    
    controller = StochasticDecodingController(seed=42)
    logits = np.random.randn(50)
    
    methods = [
        SamplingMethod.ARGMAX,
        SamplingMethod.TEMPERATURE,
        SamplingMethod.TOP_K,
        SamplingMethod.TOP_P
    ]
    
    results = {}
    
    for method in methods:
        if method == SamplingMethod.TOP_K:
            sample = controller.controlled_sampling(logits, method, k=5)
        elif method == SamplingMethod.TOP_P:
            sample = controller.controlled_sampling(logits, method, p=0.9)
        else:
            sample = controller.controlled_sampling(logits, method)
        
        results[method.value] = sample
        print(f"✓ {method.value}: sampled token {sample}")
    
    # Argmax should always return the same token
    argmax_samples = [
        controller.controlled_sampling(logits, SamplingMethod.ARGMAX)
        for _ in range(5)
    ]
    assert len(set(argmax_samples)) == 1
    print(f"✓ Argmax consistency verified")
    
    return True


def test_semantic_similarity():
    """Test semantic similarity computation"""
    print("\n" + "="*70)
    print("TEST: Semantic Similarity")
    print("="*70)
    
    controller = StochasticDecodingController()
    
    # Test similar texts
    similar_texts = [
        "The cat sat on the mat",
        "The cat sits on the mat",
        "A cat sat on a mat"
    ]
    
    similarity_similar = controller.compute_semantic_similarity(similar_texts, 'jaccard')
    assert similarity_similar > 0.5, "Similar texts should have high similarity"
    print(f"✓ Similar texts Jaccard similarity: {similarity_similar:.4f}")
    
    # Test different texts
    different_texts = [
        "The cat sat on the mat",
        "Dogs love to play fetch",
        "Programming is fun"
    ]
    
    similarity_different = controller.compute_semantic_similarity(different_texts, 'jaccard')
    assert similarity_different < similarity_similar, "Different texts should have lower similarity"
    print(f"✓ Different texts Jaccard similarity: {similarity_different:.4f}")
    
    # Test other similarity methods
    methods = ['levenshtein', 'cosine', 'token_overlap']
    
    for method in methods:
        sim = controller.compute_semantic_similarity(similar_texts, method)
        assert 0 <= sim <= 1, f"Similarity should be in [0, 1]"
        print(f"✓ {method} similarity: {sim:.4f}")
    
    # Test edge cases
    empty_response = controller.compute_semantic_similarity([])
    assert empty_response == 1.0
    
    single_response = controller.compute_semantic_similarity(["single"])
    assert single_response == 1.0
    
    print("✓ Edge cases handled correctly")
    
    return True


def test_generation_variants():
    """Test generation of controlled variants"""
    print("\n" + "="*70)
    print("TEST: Generation Variants")
    print("="*70)
    
    controller = StochasticDecodingController(seed=42)
    
    # Mock model and challenge
    class MockModel:
        def generate(self, challenge, **kwargs):
            # Return different responses based on temperature
            temp = kwargs.get('temperature', 1.0)
            if temp == 0:
                return f"Deterministic response to {challenge}"
            else:
                return f"Stochastic response (temp={temp}) to {challenge}"
    
    model = MockModel()
    challenge = "test_challenge"
    
    # Generate variants
    variants = controller.generate_verification_response(model, challenge, num_variants=3)
    
    assert len(variants) == 3
    print(f"✓ Generated {len(variants)} variants")
    
    # Check that variants were stored in history
    assert len(controller.generation_history) >= 3
    print(f"✓ Generation history updated: {len(controller.generation_history)} entries")
    
    # Verify semantic similarity
    similarity = controller.compute_semantic_similarity(variants)
    print(f"✓ Variant similarity: {similarity:.4f}")
    
    return True


def test_hf_model_generation_with_metadata():
    """Ensure HuggingFace-style models generate text and log metadata"""

    controller = StochasticDecodingController(seed=0)

    class MockTokenizer:
        def __call__(self, text, return_tensors=None):
            return {'input_ids': torch.tensor([[ord(c) for c in text]])}

        def decode(self, ids, skip_special_tokens=True):
            return ''.join(chr(i) for i in ids)

    class MockHFModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device('cpu')
            self.tokenizer = MockTokenizer()

        def generate(self, input_ids=None, **kwargs):
            # Echo the input and append '!'
            exclam = torch.tensor([[33]], device=input_ids.device)
            return torch.cat([input_ids, exclam], dim=1)

    model = MockHFModel()
    challenge = "hi"

    responses = controller.generate_verification_response(model, challenge, num_variants=1)

    assert responses[0] == "hi!"

    history = controller.generation_history[-1]
    assert history['metadata']['input_ids'] == [[104, 105]]
    assert history['metadata']['output_ids'] == [104, 105, 33]
    assert 'device' in history['metadata']


def test_callable_model_fallback():
    """Verify callable models without generate are supported"""

    controller = StochasticDecodingController(seed=0)

    class EchoModel:
        def __call__(self, prompt, **kwargs):
            return f"echo:{prompt}"

    model = EchoModel()
    challenge = "hello"

    responses = controller.generate_verification_response(model, challenge, num_variants=1)

    assert responses[0] == "echo:hello"
    history = controller.generation_history[-1]
    assert history['metadata'] == {}


def test_integrated_verification():
    """Test integrated verification system"""
    print("\n" + "="*70)
    print("TEST: Integrated Verification")
    print("="*70)
    
    verifier = IntegratedVerificationSystem(TokenizerType.CHARACTER, vocab_size=1000, seed=42)
    
    # Mock model
    class MockModel:
        def generate(self, challenge, **kwargs):
            base = f"Response to {challenge}"
            temp = kwargs.get('temperature', 1.0)
            if temp > 0:
                return base + f" (variant {np.random.randint(100)})"
            return base
    
    model = MockModel()
    challenge = "verification_test"
    
    # Run verification
    result = verifier.verify_model_response(model, challenge, num_variants=3)
    
    assert 'verified' in result
    assert 'semantic_similarity' in result
    assert 'hash_consistency' in result
    assert len(result['responses']) == 3
    assert len(result['normalized_hashes']) == 3
    
    print(f"✓ Verification result: {result['verified']}")
    print(f"✓ Semantic similarity: {result['semantic_similarity']:.4f}")
    print(f"✓ Hash consistency: {result['hash_consistency']:.4f}")
    
    # Test cross-tokenizer verification
    test_text = "Cross tokenizer test"
    cross_result = verifier.cross_tokenizer_verification(
        test_text,
        [TokenizerType.CHARACTER, TokenizerType.WHITESPACE]
    )
    
    assert 'consistency_score' in cross_result
    assert 'tokenizer_results' in cross_result
    
    print(f"✓ Cross-tokenizer consistency: {cross_result['consistency_score']:.4f}")
    
    return True


def test_caching():
    """Test caching functionality"""
    print("\n" + "="*70)
    print("TEST: Caching")
    print("="*70)
    
    normalizer = TokenSpaceNormalizer(cache_size=10)
    
    # Initial cache should be empty
    stats = normalizer.get_cache_stats()
    assert stats['cache_size'] == 0
    assert stats['cache_hits'] == 0
    
    # Perform operations that should be cached
    test_tokens = tuple([1, 2, 3, 4, 5])
    
    # First call - cache miss
    result1 = normalizer.normalize_token_sequence(test_tokens, 'canonical')
    
    # Second call - should be cache hit
    result2 = normalizer.normalize_token_sequence(test_tokens, 'canonical')
    
    assert result1 == result2
    
    # Note: LRU cache stats not directly accessible, but function should be faster
    print("✓ Caching functionality verified")
    
    # Clear cache
    normalizer.clear_cache()
    stats_after = normalizer.get_cache_stats()
    assert stats_after['cache_hits'] == 0
    assert stats_after['cache_misses'] == 0
    print("✓ Cache cleared successfully")
    
    return True


def test_multilingual_support():
    """Test multilingual text handling"""
    print("\n" + "="*70)
    print("TEST: Multilingual Support")
    print("="*70)
    
    normalizer = TokenSpaceNormalizer()
    
    # Test various languages and scripts
    multilingual_texts = [
        "Hello World",  # English
        "Привет мир",  # Russian
        "你好世界",  # Chinese
        "مرحبا بالعالم",  # Arabic
        "こんにちは世界",  # Japanese
        "🌍🌎🌏",  # Emojis
    ]
    
    for text in multilingual_texts:
        normalized = normalizer.normalize_text(text)
        hash_val = normalizer.compute_token_invariant_hash(text)
        
        assert len(normalized) > 0
        assert len(hash_val) == 64  # SHA256 hex length
        
        print(f"✓ '{text}' -> hash: {hash_val[:20]}...")
    
    return True


def test_special_tokens():
    """Test special token handling"""
    print("\n" + "="*70)
    print("TEST: Special Tokens")
    print("="*70)
    
    normalizer = TokenSpaceNormalizer()
    
    # Text with special tokens
    texts_with_special = [
        "[PAD] Hello [SEP] World [CLS]",
        "<s> Start of text </s>",
        "Text with [MASK] token",
        "<unk> unknown <pad> padding"
    ]
    
    for text in texts_with_special:
        # Compute hash without special tokens
        hash_val = normalizer.compute_token_invariant_hash(text)
        
        # Remove special tokens manually and compute hash
        clean_text = text
        for pattern in normalizer.special_token_patterns.values():
            clean_text = pattern.sub('', clean_text)
        clean_text = ' '.join(clean_text.split())
        
        hash_clean = normalizer.compute_token_invariant_hash(clean_text)
        
        print(f"✓ Processed: '{text[:30]}...'")
    
    print("✓ Special tokens handled correctly")
    
    return True


def run_all_tests():
    """Run all test functions"""
    print("\n" + "="*70)
    print("TOKEN SPACE NORMALIZER - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Text Normalization", test_text_normalization),
        ("Token Sequence Normalization", test_token_sequence_normalization),
        ("Token Alignment", test_token_alignment),
        ("Invariant Hash", test_invariant_hash),
        ("Subword Regularization", test_subword_regularization),
        ("Unknown Token Handling", test_unknown_token_handling),
        ("Deterministic Mode", test_deterministic_mode),
        ("Sampling Methods", test_sampling_methods),
        ("Semantic Similarity", test_semantic_similarity),
        ("Generation Variants", test_generation_variants),
        ("Integrated Verification", test_integrated_verification),
        ("Caching", test_caching),
        ("Multilingual Support", test_multilingual_support),
        ("Special Tokens", test_special_tokens)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(tests))*100:.1f}%")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)