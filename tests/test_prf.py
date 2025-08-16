"""
Comprehensive unit tests for PRF-based challenge derivation.

Tests:
- PRF determinism
- Nonce uniqueness guarantees
- Key derivation correctness
- Counter mode overflow handling
- Float/integer generation uniformity
"""

import pytest
import hashlib
import hmac
import struct
import numpy as np
from scipy import stats
from typing import List
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pot.core.prf import (
    prf_derive_key,
    prf_bytes,
    prf_derive_seed,
    prf_expand,
    prf_integers,
    prf_floats,
    prf_choice,
    prf_shuffle
)


class TestPRFDeterminism:
    """Test that PRF functions are deterministic."""
    
    def test_prf_derive_key_deterministic(self):
        """Test that key derivation is deterministic."""
        master_key = b"master_key_32_bytes_padded_here!"
        label = "test_label"
        nonce = b"nonce_32_bytes_padded_for_test!!"
        
        # Derive multiple times
        key1 = prf_derive_key(master_key, label, nonce)
        key2 = prf_derive_key(master_key, label, nonce)
        key3 = prf_derive_key(master_key, label, nonce)
        
        # All should be identical
        assert key1 == key2 == key3
        
        # Should be 32 bytes (SHA256 output)
        assert len(key1) == 32
        
        # Should be bytes
        assert isinstance(key1, bytes)
    
    def test_prf_bytes_deterministic(self):
        """Test that byte generation is deterministic."""
        key = b"test_key_32_bytes_padded_here!!!"
        info = b"test_info"
        
        # Generate multiple times with different lengths
        for nbytes in [16, 32, 64, 128, 256, 1000]:
            bytes1 = prf_bytes(key, info, nbytes)
            bytes2 = prf_bytes(key, info, nbytes)
            bytes3 = prf_bytes(key, info, nbytes)
            
            # All should be identical
            assert bytes1 == bytes2 == bytes3
            
            # Should have correct length
            assert len(bytes1) == nbytes
    
    def test_prf_integers_deterministic(self):
        """Test that integer generation is deterministic."""
        key = b"test_key_32_bytes_padded_here!!!"
        info = b"test_info"
        
        # Generate integers multiple times
        ints1 = prf_integers(key, info, count=100, max_value=1000)
        ints2 = prf_integers(key, info, count=100, max_value=1000)
        ints3 = prf_integers(key, info, count=100, max_value=1000)
        
        # All should be identical
        assert ints1 == ints2 == ints3
        
        # Should have correct count
        assert len(ints1) == 100
        
        # All should be in range
        assert all(0 <= x < 1000 for x in ints1)
    
    def test_prf_floats_deterministic(self):
        """Test that float generation is deterministic."""
        key = b"test_key_32_bytes_padded_here!!!"
        info = b"test_info"
        
        # Generate floats multiple times
        floats1 = prf_floats(key, info, count=50, min_val=0.0, max_val=1.0)
        floats2 = prf_floats(key, info, count=50, min_val=0.0, max_val=1.0)
        floats3 = prf_floats(key, info, count=50, min_val=0.0, max_val=1.0)
        
        # All should be identical
        assert floats1 == floats2 == floats3
        
        # Should have correct count
        assert len(floats1) == 50
        
        # All should be in range
        assert all(0.0 <= x < 1.0 for x in floats1)
    
    def test_prf_shuffle_deterministic(self):
        """Test that shuffling is deterministic."""
        key = b"test_key_32_bytes_padded_here!!!"
        info = b"test_info"
        items = list(range(100))
        
        # Shuffle multiple times
        shuffled1 = prf_shuffle(key, info, items)
        shuffled2 = prf_shuffle(key, info, items)
        shuffled3 = prf_shuffle(key, info, items)
        
        # All should be identical
        assert shuffled1 == shuffled2 == shuffled3
        
        # Should be a permutation
        assert sorted(shuffled1) == sorted(items)
        
        # Should not be the original (with high probability)
        assert shuffled1 != items


class TestKeyDerivation:
    """Test key derivation properties."""
    
    def test_different_labels_different_keys(self):
        """Test that different labels produce different keys."""
        master_key = b"master_key_32_bytes_padded_here!"
        nonce = b"nonce_32_bytes_padded_for_test!!"
        
        keys = {}
        labels = ["label1", "label2", "label3", "challenge:vision", "challenge:lm"]
        
        for label in labels:
            key = prf_derive_key(master_key, label, nonce)
            # Should be unique
            assert key not in keys.values()
            keys[label] = key
        
        # All keys should be different
        assert len(set(keys.values())) == len(labels)
    
    def test_different_nonces_different_keys(self):
        """Test that different nonces produce different keys."""
        master_key = b"master_key_32_bytes_padded_here!"
        label = "test_label"
        
        keys = []
        for i in range(10):
            nonce = struct.pack('>I', i) + b'\x00' * 28
            key = prf_derive_key(master_key, label, nonce)
            keys.append(key)
        
        # All keys should be different
        assert len(set(keys)) == 10
    
    def test_seed_derivation(self):
        """Test seed derivation for challenge generation."""
        master_key = b"master_key_32_bytes_padded_here!"
        nonce = b"nonce_32_bytes_padded_for_test!!"
        
        # Different families should produce different seeds
        seed1 = prf_derive_seed(master_key, "vision:freq", {"param": 1}, nonce)
        seed2 = prf_derive_seed(master_key, "vision:texture", {"param": 1}, nonce)
        seed3 = prf_derive_seed(master_key, "lm:templates", {"param": 1}, nonce)
        
        assert seed1 != seed2 != seed3
        assert len(seed1) == len(seed2) == len(seed3) == 32
        
        # Different params should produce different seeds
        seed4 = prf_derive_seed(master_key, "vision:freq", {"param": 2}, nonce)
        assert seed1 != seed4
        
        # Same inputs should produce same seed
        seed5 = prf_derive_seed(master_key, "vision:freq", {"param": 1}, nonce)
        assert seed1 == seed5


class TestCounterMode:
    """Test counter mode PRF expansion."""
    
    def test_small_outputs(self):
        """Test generation of small outputs."""
        key = b"test_key_32_bytes_padded_here!!!"
        info = b"test_info"
        
        # Test various small sizes
        for size in [1, 2, 4, 8, 16, 31, 32]:
            output = prf_bytes(key, info, size)
            assert len(output) == size
            assert isinstance(output, bytes)
    
    def test_large_outputs(self):
        """Test generation of large outputs."""
        key = b"test_key_32_bytes_padded_here!!!"
        info = b"test_info"
        
        # Test larger sizes that require multiple blocks
        for size in [33, 64, 100, 256, 1024, 10000]:
            output = prf_bytes(key, info, size)
            assert len(output) == size
            
            # Different parts should be different (no obvious patterns)
            if size >= 64:
                first_32 = output[:32]
                second_32 = output[32:64]
                assert first_32 != second_32
    
    def test_expansion_consistency(self):
        """Test that expansion is consistent with incremental generation."""
        key = b"test_key_32_bytes_padded_here!!!"
        info = b"test_info"
        
        # Generate 100 bytes at once
        all_at_once = prf_bytes(key, info, 100)
        
        # Generate same 100 bytes in chunks
        # Note: This won't match unless we use same counter sequence
        # This test verifies internal consistency
        first_32 = prf_bytes(key, info, 32)
        assert all_at_once[:32] == first_32
        
        # The expand function should give same result
        seed = key
        expanded = prf_expand(seed, 100)
        assert isinstance(expanded, bytes)
        assert len(expanded) == 100
    
    def test_counter_overflow_protection(self):
        """Test that counter overflow is handled."""
        key = b"test_key_32_bytes_padded_here!!!"
        info = b"test_info"
        
        # This should work (not overflow)
        output = prf_bytes(key, info, 1000000)  # 1MB
        assert len(output) == 1000000
        
        # Test error handling for invalid inputs
        with pytest.raises(ValueError):
            prf_bytes(key, info, 0)
        
        with pytest.raises(ValueError):
            prf_bytes(key, info, -1)


class TestUniformity:
    """Test uniformity of generated values."""
    
    def test_integer_uniformity(self):
        """Test that integers are uniformly distributed."""
        key = b"test_key_32_bytes_padded_here!!!"
        info = b"test_info"
        
        # Generate many integers
        max_val = 100
        count = 10000
        integers = prf_integers(key, info, count, max_val)
        
        # Chi-square test for uniformity
        observed, _ = np.histogram(integers, bins=max_val, range=(0, max_val))
        expected = count / max_val
        
        chi2, p_value = stats.chisquare(observed, f_exp=[expected] * max_val)
        
        # p-value should be reasonable (not too small)
        # Using 0.01 as threshold to avoid false positives
        assert p_value > 0.01, f"Integers not uniform: p={p_value:.4f}"
        
        # Check mean and variance
        mean = np.mean(integers)
        expected_mean = (max_val - 1) / 2
        assert abs(mean - expected_mean) < 2, f"Mean off: {mean:.2f} vs {expected_mean:.2f}"
    
    def test_float_uniformity(self):
        """Test that floats are uniformly distributed."""
        key = b"test_key_32_bytes_padded_here!!!"
        info = b"test_info"
        
        # Generate many floats
        count = 10000
        floats = prf_floats(key, info, count, 0.0, 1.0)
        
        # Kolmogorov-Smirnov test for uniformity
        d_stat, p_value = stats.kstest(floats, 'uniform')
        
        # p-value should indicate uniformity
        assert p_value > 0.01, f"Floats not uniform: p={p_value:.4f}"
        
        # Check mean and variance
        mean = np.mean(floats)
        var = np.var(floats)
        
        # For uniform [0,1]: mean=0.5, var=1/12
        assert abs(mean - 0.5) < 0.01, f"Mean off: {mean:.4f}"
        assert abs(var - 1/12) < 0.005, f"Variance off: {var:.4f}"
    
    def test_float_range(self):
        """Test float generation in custom ranges."""
        key = b"test_key_32_bytes_padded_here!!!"
        info = b"test_info"
        
        # Test various ranges
        test_cases = [
            (0.0, 1.0),
            (-1.0, 1.0),
            (10.0, 20.0),
            (-100.0, -50.0),
            (0.001, 0.002)
        ]
        
        for min_val, max_val in test_cases:
            floats = prf_floats(key, info, 1000, min_val, max_val)
            
            # All should be in range
            assert all(min_val <= x < max_val for x in floats), \
                f"Float out of range [{min_val}, {max_val})"
            
            # Check approximate mean
            expected_mean = (min_val + max_val) / 2
            actual_mean = np.mean(floats)
            tolerance = (max_val - min_val) * 0.05  # 5% tolerance
            assert abs(actual_mean - expected_mean) < tolerance, \
                f"Mean off for range [{min_val}, {max_val}]: {actual_mean:.4f}"


class TestChoice:
    """Test random choice functionality."""
    
    def test_choice_basic(self):
        """Test basic choice functionality."""
        key = b"test_key_32_bytes_padded_here!!!"
        info = b"test_info"
        
        choices = ["apple", "banana", "cherry", "date", "elderberry"]
        selected = prf_choice(key, info, choices, 10)
        
        # Should have correct count
        assert len(selected) == 10
        
        # All should be from choices
        assert all(item in choices for item in selected)
        
        # Should be deterministic
        selected2 = prf_choice(key, info, choices, 10)
        assert selected == selected2
    
    def test_choice_distribution(self):
        """Test that choices are uniformly distributed."""
        key = b"test_key_32_bytes_padded_here!!!"
        info = b"test_info"
        
        choices = list(range(10))
        selected = prf_choice(key, info, choices, 10000)
        
        # Count occurrences
        counts = {}
        for item in selected:
            counts[item] = counts.get(item, 0) + 1
        
        # All items should be selected
        assert len(counts) == len(choices)
        
        # Should be roughly uniform
        expected_count = 10000 / 10
        for count in counts.values():
            assert abs(count - expected_count) < 200  # Allow some variance


class TestShuffle:
    """Test shuffling functionality."""
    
    def test_shuffle_permutation(self):
        """Test that shuffle produces valid permutation."""
        key = b"test_key_32_bytes_padded_here!!!"
        info = b"test_info"
        
        items = list(range(100))
        shuffled = prf_shuffle(key, info, items)
        
        # Should have same length
        assert len(shuffled) == len(items)
        
        # Should contain same elements
        assert sorted(shuffled) == sorted(items)
        
        # Original should be unchanged
        assert items == list(range(100))
    
    def test_shuffle_quality(self):
        """Test quality of shuffling."""
        key = b"test_key_32_bytes_padded_here!!!"
        
        # Shuffle same list with different info
        items = list(range(50))
        shuffles = []
        
        for i in range(100):
            info = struct.pack('>I', i)
            shuffled = prf_shuffle(key, info, items)
            shuffles.append(shuffled)
        
        # Check that elements move around
        # Track where element 0 ends up
        positions_of_zero = [shuffle.index(0) for shuffle in shuffles]
        
        # Should see variety of positions
        unique_positions = len(set(positions_of_zero))
        assert unique_positions > 30, f"Element 0 only in {unique_positions} positions"
        
        # Average position should be near middle
        avg_position = np.mean(positions_of_zero)
        expected_position = len(items) / 2
        assert abs(avg_position - expected_position) < 5
    
    def test_shuffle_edge_cases(self):
        """Test shuffle with edge cases."""
        key = b"test_key_32_bytes_padded_here!!!"
        info = b"test_info"
        
        # Empty list
        assert prf_shuffle(key, info, []) == []
        
        # Single element
        assert prf_shuffle(key, info, [1]) == [1]
        
        # Two elements
        two_elem = prf_shuffle(key, info, [1, 2])
        assert sorted(two_elem) == [1, 2]


class TestNonceUniqueness:
    """Test nonce properties for uniqueness guarantees."""
    
    def test_different_nonces_different_outputs(self):
        """Test that different nonces produce different outputs."""
        master_key = b"master_key_32_bytes_padded_here!"
        label = "test"
        
        outputs = set()
        for i in range(1000):
            # Create unique nonce
            nonce = struct.pack('>Q', i) + b'\x00' * 24
            key = prf_derive_key(master_key, label, nonce)
            
            # Should be unique
            assert key not in outputs
            outputs.add(key)
    
    def test_nonce_collision_probability(self):
        """Test that random nonces have low collision probability."""
        import os
        
        nonces = set()
        for _ in range(10000):
            nonce = os.urandom(32)
            # 32 bytes = 256 bits, collision probability is negligible
            assert nonce not in nonces
            nonces.add(nonce)


def test_hmac_sha256_properties():
    """Test properties of HMAC-SHA256 used in PRF."""
    key = b"test_key"
    msg1 = b"message1"
    msg2 = b"message2"
    
    # HMAC should be deterministic
    h1 = hmac.new(key, msg1, hashlib.sha256).digest()
    h2 = hmac.new(key, msg1, hashlib.sha256).digest()
    assert h1 == h2
    
    # Different messages should give different outputs
    h3 = hmac.new(key, msg2, hashlib.sha256).digest()
    assert h1 != h3
    
    # Output should be 32 bytes
    assert len(h1) == 32


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])