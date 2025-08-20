#!/usr/bin/env python3
"""
Test the fixed token space normalizer implementation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.security.token_space_normalizer import TokenSpaceNormalizer, compute_alignment_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mock_safe_tokenizer():
    """Test with mock-like tokenizer to ensure no Mock issues"""
    
    class MockTokenizer:
        def __init__(self):
            self.vocab = {"hello": 1, "world": 2, "[PAD]": 0, "test": 3}
            self.special_tokens_map = {"pad_token": "[PAD]"}
            self.pad_token_id = 0
            self.eos_token_id = None
            
        def get_vocab(self):
            return self.vocab
            
        def decode(self, ids, skip_special_tokens=False):
            tokens = [k for k, v in self.vocab.items() if v in ids]
            return " ".join(tokens)
            
        def convert_tokens_to_ids(self, token):
            return self.vocab.get(token)
    
    logger.info("Testing mock-safe tokenizer...")
    
    tokenizer = MockTokenizer()
    normalizer = TokenSpaceNormalizer(tokenizer, mode="canonical")
    
    # Test basic normalization
    test_ids = [1, 2, 0, 3]  # hello world [PAD] test
    normalized = normalizer.normalize(test_ids)
    
    logger.info(f"Input: {test_ids}")
    logger.info(f"Normalized: {normalized}")
    logger.info(f"Special tokens filtered: {0 not in normalized}")
    
    assert 0 not in normalized, "Special tokens should be filtered in canonical mode"
    assert normalized == [1, 2, 3], f"Expected [1, 2, 3], got {normalized}"
    
    logger.info("✅ Mock tokenizer test passed")
    return True

def test_preserve_mode():
    """Test preserve mode returns input as-is"""
    
    class SimpleTokenizer:
        def __init__(self):
            self.vocab = {"a": 1, "b": 2}
            
    tokenizer = SimpleTokenizer()
    normalizer = TokenSpaceNormalizer(tokenizer, mode="preserve")
    
    test_ids = [1, 2, 1]
    normalized = normalizer.normalize(test_ids)
    
    logger.info(f"Preserve mode - Input: {test_ids}")
    logger.info(f"Preserve mode - Output: {normalized}")
    
    assert normalized == test_ids, f"Preserve mode should return input as-is"
    
    logger.info("✅ Preserve mode test passed")
    return True

def test_iterable_handling():
    """Test handling of different iterable types"""
    
    class DictTokenizer:
        def __init__(self):
            self.vocab = {"x": 10, "y": 20}
            
    tokenizer = DictTokenizer()
    normalizer = TokenSpaceNormalizer(tokenizer, mode="preserve")
    
    # Test with tuple
    result1 = normalizer.normalize((10, 20))
    logger.info(f"Tuple input: {result1}")
    assert result1 == [10, 20]
    
    # Test with set (order may vary)
    result2 = normalizer.normalize({10, 20})
    logger.info(f"Set input: {result2}")
    assert set(result2) == {10, 20}
    
    # Test with generator
    def gen():
        yield 10
        yield 20
    
    result3 = normalizer.normalize(gen())
    logger.info(f"Generator input: {result3}")
    assert result3 == [10, 20]
    
    logger.info("✅ Iterable handling test passed")
    return True

def test_compute_alignment_score():
    """Test the alignment score computation"""
    
    class TestTokenizer:
        def __init__(self):
            self.vocab = {"a": 1, "b": 2, "c": 3, "d": 4}
            
    tokenizer = TestTokenizer()
    normalizer = TokenSpaceNormalizer(tokenizer, mode="preserve")
    
    # Test identical sequences
    score1 = compute_alignment_score([1, 2, 3], [1, 2, 3], normalizer)
    logger.info(f"Identical sequences score: {score1}")
    assert score1 == 1.0, "Identical sequences should have score 1.0"
    
    # Test completely different sequences  
    score2 = compute_alignment_score([1, 2], [3, 4], normalizer)
    logger.info(f"Different sequences score: {score2}")
    assert score2 == 0.0, "Different sequences should have score 0.0"
    
    # Test partial overlap
    score3 = compute_alignment_score([1, 2, 3], [2, 3, 4], normalizer)
    logger.info(f"Partial overlap score: {score3}")
    # Intersection: {2, 3}, Union: {1, 2, 3, 4}, Score: 2/4 = 0.5
    assert abs(score3 - 0.5) < 0.001, f"Expected ~0.5, got {score3}"
    
    # Test empty sequences
    score4 = compute_alignment_score([], [], normalizer)
    logger.info(f"Empty sequences score: {score4}")
    assert score4 == 1.0, "Empty sequences should have score 1.0"
    
    # Test one empty
    score5 = compute_alignment_score([1, 2], [], normalizer)
    logger.info(f"One empty score: {score5}")
    assert score5 == 0.0, "One empty sequence should have score 0.0"
    
    logger.info("✅ Alignment score test passed")
    return True

def test_error_handling():
    """Test error handling for edge cases"""
    
    class MinimalTokenizer:
        pass
    
    tokenizer = MinimalTokenizer()
    normalizer = TokenSpaceNormalizer(tokenizer, mode="canonical")
    
    # Test with empty input
    result1 = normalizer.normalize([])
    assert result1 == []
    
    # Test with None input
    result2 = normalizer.normalize(None)
    assert result2 == []
    
    # Test with string input (should not crash)
    result3 = normalizer.normalize("not_a_list")
    assert result3 == []
    
    logger.info("✅ Error handling test passed")
    return True

def main():
    """Run all token normalizer tests"""
    logger.info("\n" + "="*70)
    logger.info("TOKEN SPACE NORMALIZER FIXED TESTS")
    logger.info("="*70)
    
    tests = [
        ("Mock-Safe Tokenizer", test_mock_safe_tokenizer),
        ("Preserve Mode", test_preserve_mode), 
        ("Iterable Handling", test_iterable_handling),
        ("Alignment Score", test_compute_alignment_score),
        ("Error Handling", test_error_handling)
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
        logger.info("\n🎉 ALL TOKEN NORMALIZER TESTS PASSED!")
        logger.info("Key achievements:")
        logger.info("  • Mock-safe type handling implemented")
        logger.info("  • Mapping/Iterable abstraction working")
        logger.info("  • Special token filtering functional")
        logger.info("  • Alignment score computation accurate")
        logger.info("  • Error handling robust")
    else:
        logger.info("\n⚠️ Some tests failed - review output above")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())