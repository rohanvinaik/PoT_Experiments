"""Tests for PRF-based challenge derivation."""

import sys
import os
import struct
import numpy as np
from pathlib import Path
from collections import Counter
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pot.core.prf import (
    prf_derive_key, prf_bytes, prf_derive_seed, prf_expand,
    prf_integers, prf_floats, prf_choice, prf_shuffle
)
from pot.core.challenge import ChallengeConfig, generate_challenges, generate_challenges_legacy


def test_prf_determinism():
    """Test that PRF functions are deterministic with same inputs."""
    print("Testing PRF determinism...")
    
    master_key = b"test_master_key_32_bytes_long!!!"
    label = "test_label"
    nonce = b"test_nonce_16_bytes_for_testing"
    
    # Test prf_derive_key determinism
    key1 = prf_derive_key(master_key, label, nonce)
    key2 = prf_derive_key(master_key, label, nonce)
    assert key1 == key2, "prf_derive_key is not deterministic"
    assert len(key1) == 32, f"Expected 32 bytes, got {len(key1)}"
    
    # Test prf_bytes determinism
    info = b"test_info"
    bytes1 = prf_bytes(key1, info, 100)
    bytes2 = prf_bytes(key1, info, 100)
    assert bytes1 == bytes2, "prf_bytes is not deterministic"
    assert len(bytes1) == 100, f"Expected 100 bytes, got {len(bytes1)}"
    
    # Test prf_derive_seed determinism
    family = "vision:freq"
    params = {"freq_range": [0.1, 10.0], "contrast_range": [0.1, 1.0]}
    seed1 = prf_derive_seed(master_key, family, params, nonce)
    seed2 = prf_derive_seed(master_key, family, params, nonce)
    assert seed1 == seed2, "prf_derive_seed is not deterministic"
    
    print("✓ PRF functions are deterministic")
    return True


def test_different_nonces():
    """Test that different nonces produce different outputs."""
    print("Testing different nonces produce different sequences...")
    
    master_key = b"test_master_key_32_bytes_long!!!"
    label = "test_label"
    nonce1 = b"nonce_1_16_bytes"
    nonce2 = b"nonce_2_16_bytes"
    
    # Test with prf_derive_key
    key1 = prf_derive_key(master_key, label, nonce1)
    key2 = prf_derive_key(master_key, label, nonce2)
    assert key1 != key2, "Different nonces should produce different keys"
    
    # Test with prf_bytes
    info = b"test_info"
    bytes1 = prf_bytes(key1, info, 100)
    bytes2 = prf_bytes(key2, info, 100)
    assert bytes1 != bytes2, "Different keys should produce different bytes"
    
    # Test with challenge generation
    config1 = ChallengeConfig(
        master_key_hex=master_key.hex(),
        session_nonce_hex=nonce1.hex(),
        n=10,
        family="vision:freq",
        params={"freq_range": [0.1, 10.0], "contrast_range": [0.1, 1.0]}
    )
    
    config2 = ChallengeConfig(
        master_key_hex=master_key.hex(),
        session_nonce_hex=nonce2.hex(),
        n=10,
        family="vision:freq",
        params={"freq_range": [0.1, 10.0], "contrast_range": [0.1, 1.0]}
    )
    
    challenges1 = generate_challenges(config1)
    challenges2 = generate_challenges(config2)
    
    assert challenges1["items"] != challenges2["items"], "Different nonces should produce different challenges"
    assert challenges1["salt"] != challenges2["salt"], "Different nonces should produce different salts"
    
    print("✓ Different nonces produce completely different sequences")
    return True


def test_challenge_determinism():
    """Test that challenges are deterministic with same inputs."""
    print("Testing challenge generation determinism...")
    
    master_key = os.urandom(32)
    nonce = os.urandom(32)
    
    config = ChallengeConfig(
        master_key_hex=master_key.hex(),
        session_nonce_hex=nonce.hex(),
        n=20,
        family="vision:freq",
        params={"freq_range": [0.1, 10.0], "contrast_range": [0.1, 1.0]}
    )
    
    # Generate challenges multiple times
    challenges1 = generate_challenges(config)
    challenges2 = generate_challenges(config)
    challenges3 = generate_challenges(config)
    
    # All should be identical
    assert challenges1 == challenges2 == challenges3, "Challenges are not deterministic"
    
    # Check structure
    assert len(challenges1["items"]) == 20
    assert all("freq" in item and "theta" in item and "phase" in item and "contrast" in item 
               for item in challenges1["items"])
    
    print("✓ Challenge generation is deterministic")
    
    # Test other families
    families_and_params = [
        ("vision:texture", {"octaves": [1, 5], "scale": [0.1, 2.0]}),
        ("lm:templates", {"templates": ["template1", "template2"], 
                         "slots": {"slot1": ["a", "b"], "slot2": ["x", "y"]}})
    ]
    
    for family, params in families_and_params:
        config.family = family
        config.params = params
        
        ch1 = generate_challenges(config)
        ch2 = generate_challenges(config)
        
        assert ch1 == ch2, f"Challenges not deterministic for family {family}"
        print(f"✓ {family} challenges are deterministic")
    
    return True


def test_prf_uniformity():
    """Test that PRF output is uniformly distributed."""
    print("Testing PRF output uniformity...")
    
    key = os.urandom(32)
    
    # Test integer uniformity
    print("  Testing integer uniformity...")
    max_val = 100
    n_samples = 10000
    integers = prf_integers(key, b"uniformity_test", n_samples, 0, max_val)
    
    # Check range
    assert all(0 <= i < max_val for i in integers), "Integers out of range"
    
    # Chi-square test for uniformity
    expected_count = n_samples / max_val
    counts = Counter(integers)
    chi_square = sum((counts[i] - expected_count) ** 2 / expected_count for i in range(max_val))
    
    # Critical value for chi-square with df=99 at p=0.01 is ~135
    # We use a more lenient threshold for the test
    assert chi_square < 150, f"Integer distribution not uniform (chi-square={chi_square:.2f})"
    print(f"  ✓ Integer uniformity test passed (chi-square={chi_square:.2f})")
    
    # Test float uniformity
    print("  Testing float uniformity...")
    n_samples = 10000
    floats = prf_floats(key, b"float_test", n_samples, 0.0, 1.0)
    
    # Check range
    assert all(0.0 <= f < 1.0 for f in floats), "Floats out of range"
    
    # Test uniform distribution using bins
    n_bins = 20
    hist, _ = np.histogram(floats, bins=n_bins, range=(0, 1))
    expected_per_bin = n_samples / n_bins
    
    # Chi-square test
    chi_square = sum((count - expected_per_bin) ** 2 / expected_per_bin for count in hist)
    
    # Critical value for chi-square with df=19 at p=0.01 is ~36
    assert chi_square < 40, f"Float distribution not uniform (chi-square={chi_square:.2f})"
    print(f"  ✓ Float uniformity test passed (chi-square={chi_square:.2f})")
    
    # Test byte uniformity
    print("  Testing byte uniformity...")
    n_bytes = 10000
    random_bytes = prf_bytes(key, b"byte_test", n_bytes)
    
    # Count byte values
    byte_counts = Counter(random_bytes)
    expected_per_byte = n_bytes / 256
    
    # Chi-square test
    chi_square = sum((byte_counts[i] - expected_per_byte) ** 2 / expected_per_byte 
                     for i in range(256))
    
    # Critical value for chi-square with df=255 at p=0.01 is ~310
    assert chi_square < 320, f"Byte distribution not uniform (chi-square={chi_square:.2f})"
    print(f"  ✓ Byte uniformity test passed (chi-square={chi_square:.2f})")
    
    print("✓ PRF output is uniformly distributed")
    return True


def test_prf_independence():
    """Test that different info values produce independent sequences."""
    print("Testing PRF independence...")
    
    key = os.urandom(32)
    
    # Generate two sequences with different info
    seq1 = prf_floats(key, b"seq1", 1000)
    seq2 = prf_floats(key, b"seq2", 1000)
    
    # Calculate correlation
    correlation = np.corrcoef(seq1, seq2)[0, 1]
    
    # Correlation should be very small for independent sequences
    assert abs(correlation) < 0.05, f"Sequences not independent (correlation={correlation:.4f})"
    
    print(f"✓ Different info values produce independent sequences (correlation={correlation:.4f})")
    return True


def test_prf_functions():
    """Test additional PRF utility functions."""
    print("Testing PRF utility functions...")
    
    key = os.urandom(32)
    
    # Test prf_choice
    choices = ["apple", "banana", "cherry", "date", "elderberry"]
    # Test multiple selections
    selected = [prf_choice(key, b"choice_test" + bytes([i]), choices) for i in range(100)]
    assert len(selected) == 100
    assert all(item in choices for item in selected)
    
    # Check distribution
    counts = Counter(selected)
    expected = 100 / len(choices)
    for choice in choices:
        # Allow 50% deviation from expected
        assert abs(counts[choice] - expected) < expected * 0.5, \
            f"Choice distribution biased for {choice}"
    
    print("✓ prf_choice works correctly")
    
    # Test prf_shuffle
    items = list(range(100))
    shuffled1 = prf_shuffle(key, b"shuffle1", items)
    shuffled2 = prf_shuffle(key, b"shuffle1", items)
    shuffled3 = prf_shuffle(key, b"shuffle2", items)
    
    assert shuffled1 == shuffled2, "Shuffle not deterministic"
    assert shuffled1 != shuffled3, "Different info should produce different shuffle"
    assert sorted(shuffled1) == items, "Shuffle changed elements"
    assert shuffled1 != items, "Shuffle didn't change order"
    
    print("✓ prf_shuffle works correctly")
    
    return True


def test_compatibility():
    """Test that PRF-based and legacy generation can coexist."""
    print("Testing compatibility between PRF and legacy generation...")
    
    master_key = os.urandom(32)
    nonce = os.urandom(32)
    
    config = ChallengeConfig(
        master_key_hex=master_key.hex(),
        session_nonce_hex=nonce.hex(),
        n=10,
        family="vision:freq",
        params={"freq_range": [0.1, 10.0], "contrast_range": [0.1, 1.0]}
    )
    
    # Generate with both methods
    prf_challenges = generate_challenges(config)
    legacy_challenges = generate_challenges_legacy(config)
    
    # They should be different (different methods)
    assert prf_challenges["items"] != legacy_challenges["items"], \
        "PRF and legacy should produce different results"
    
    # But both should be valid
    assert len(prf_challenges["items"]) == 10
    assert len(legacy_challenges["items"]) == 10
    
    print("✓ PRF and legacy generation can coexist")
    return True


def visualize_distributions():
    """Visualize PRF output distributions (optional)."""
    if not HAS_MATPLOTLIB:
        print("Skipping visualization (matplotlib not available)")
        return
    
    print("\nVisualizing PRF distributions...")
    
    key = os.urandom(32)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Integer distribution
    integers = prf_integers(key, b"viz", 10000, 0, 50)
    axes[0, 0].hist(integers, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title("PRF Integer Distribution (0-49)")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Count")
    
    # Float distribution
    floats = prf_floats(key, b"viz", 10000, 0.0, 1.0)
    axes[0, 1].hist(floats, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title("PRF Float Distribution (0-1)")
    axes[0, 1].set_xlabel("Value")
    axes[0, 1].set_ylabel("Count")
    
    # Byte distribution
    random_bytes = prf_bytes(key, b"viz", 10000)
    axes[1, 0].hist(list(random_bytes), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title("PRF Byte Distribution")
    axes[1, 0].set_xlabel("Byte Value")
    axes[1, 0].set_ylabel("Count")
    
    # Sequential correlation
    seq_floats = prf_floats(key, b"seq", 1000)
    axes[1, 1].scatter(seq_floats[:-1], seq_floats[1:], alpha=0.3, s=1)
    axes[1, 1].set_title("Sequential Correlation")
    axes[1, 1].set_xlabel("Value[i]")
    axes[1, 1].set_ylabel("Value[i+1]")
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("outputs/prf_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "prf_distributions.png", dpi=150)
    print(f"✓ Saved distribution plots to {output_dir}/prf_distributions.png")
    
    plt.close()


def test_prf_bytes_edge_cases():
    """Test prf_bytes with edge cases and boundary conditions."""
    print("Testing prf_bytes edge cases...")
    
    key = b"test_key_32_bytes_padded_here!!!"
    info = b"test_info"
    
    # Test zero-length output (should raise error)
    try:
        prf_bytes(key, info, 0)
        assert False, "Should raise error for zero length"
    except ValueError:
        pass
    
    # Test negative length (should raise error)
    try:
        prf_bytes(key, info, -1)
        assert False, "Should raise error for negative length"
    except ValueError:
        pass
    
    # Test single byte
    single = prf_bytes(key, info, 1)
    assert len(single) == 1
    assert isinstance(single, bytes)
    
    # Test exact block size (32 bytes for SHA256)
    block = prf_bytes(key, info, 32)
    assert len(block) == 32
    
    # Test one byte more than block size
    block_plus = prf_bytes(key, info, 33)
    assert len(block_plus) == 33
    
    # Test large output (multiple blocks)
    large = prf_bytes(key, info, 1000)
    assert len(large) == 1000
    
    # Test very large output (stress test)
    very_large = prf_bytes(key, info, 100000)
    assert len(very_large) == 100000
    
    # Test empty info
    empty_info = prf_bytes(key, b"", 100)
    assert len(empty_info) == 100
    
    # Test different info produces different output
    info1 = prf_bytes(key, b"info1", 100)
    info2 = prf_bytes(key, b"info2", 100)
    assert info1 != info2
    
    # Test incremental generation consistency
    # First 50 bytes should match when generating 100 bytes
    first_50 = prf_bytes(key, info, 50)
    first_100 = prf_bytes(key, info, 100)
    assert first_100[:50] == first_50
    
    print("✓ prf_bytes edge cases handled correctly")
    return True


def test_prf_expand_edge_cases():
    """Test prf_expand with edge cases and key expansion properties."""
    print("Testing prf_expand edge cases...")
    
    key = b"test_key_32_bytes_padded_here!!!"
    info = b"expand_info"
    
    # Test zero-length output (should raise error)
    try:
        prf_expand(key, info, 0)
        assert False, "Should raise error for zero length"
    except ValueError:
        pass
    
    # Test negative length (should raise error)
    try:
        prf_expand(key, info, -1)
        assert False, "Should raise error for negative length"
    except ValueError:
        pass
    
    # Test small outputs (should use direct PRF)
    small = prf_expand(key, info, 16)
    assert len(small) == 16
    
    # Test exactly 32 bytes (threshold)
    exact_32 = prf_expand(key, info, 32)
    assert len(exact_32) == 32
    
    # Test just over threshold (33 bytes, should use xxhash)
    over_32 = prf_expand(key, info, 33)
    assert len(over_32) == 33
    
    # Test large expansion
    large = prf_expand(key, info, 10000)
    assert len(large) == 10000
    
    # Test determinism
    expand1 = prf_expand(key, info, 1000)
    expand2 = prf_expand(key, info, 1000)
    assert expand1 == expand2
    
    # Test different info produces different output
    expand_a = prf_expand(key, b"info_a", 1000)
    expand_b = prf_expand(key, b"info_b", 1000)
    assert expand_a != expand_b
    
    # Test key sensitivity
    key2 = b"different_key_32_bytes_padded!!"
    expand_key1 = prf_expand(key, info, 1000)
    expand_key2 = prf_expand(key2, info, 1000)
    assert expand_key1 != expand_key2
    
    # Test expansion doesn't repeat patterns
    expanded = prf_expand(key, info, 256)
    # Check no obvious 8-byte patterns (xxhash block size)
    for i in range(0, len(expanded) - 16, 8):
        assert expanded[i:i+8] != expanded[i+8:i+16], "Expansion has repeating pattern"
    
    print("✓ prf_expand edge cases handled correctly")
    return True


def test_prf_integers_edge_cases():
    """Test prf_integers with edge cases and boundary conditions."""
    print("Testing prf_integers edge cases...")
    
    key = b"test_key_32_bytes_padded_here!!!"
    info = b"int_test"
    
    # Test invalid range (min >= max)
    try:
        prf_integers(key, info, 10, 5, 5)
        assert False, "Should raise error when min_val >= max_val"
    except ValueError:
        pass
    
    try:
        prf_integers(key, info, 10, 10, 5)
        assert False, "Should raise error when min_val > max_val"
    except ValueError:
        pass
    
    # Test single value range [0, 1)
    singles = prf_integers(key, info, 100, 0, 1)
    assert all(x == 0 for x in singles)
    
    # Test small range [0, 2)
    binary = prf_integers(key, info, 1000, 0, 2)
    assert all(x in [0, 1] for x in binary)
    # Check roughly balanced
    count_0 = sum(1 for x in binary if x == 0)
    assert 400 < count_0 < 600, f"Binary distribution skewed: {count_0}/1000"
    
    # Test negative range
    negatives = prf_integers(key, info, 100, -50, -30)
    assert all(-50 <= x < -30 for x in negatives)
    
    # Test range crossing zero
    crossing = prf_integers(key, info, 100, -10, 10)
    assert all(-10 <= x < 10 for x in crossing)
    
    # Test large range
    large_range = prf_integers(key, info, 100, 0, 1000000)
    assert all(0 <= x < 1000000 for x in large_range)
    
    # Test count edge cases
    zero_count = prf_integers(key, info, 0, 0, 100)
    assert len(zero_count) == 0
    
    one_count = prf_integers(key, info, 1, 0, 100)
    assert len(one_count) == 1
    
    # Test determinism with different ranges
    ints1 = prf_integers(key, info, 50, 10, 20)
    ints2 = prf_integers(key, info, 50, 10, 20)
    assert ints1 == ints2
    
    # Test uniformity in small range
    small_range = prf_integers(key, info, 10000, 0, 10)
    counts = Counter(small_range)
    for i in range(10):
        assert 900 < counts[i] < 1100, f"Non-uniform distribution at {i}: {counts[i]}"
    
    print("✓ prf_integers edge cases handled correctly")
    return True


def test_prf_floats_edge_cases():
    """Test prf_floats with edge cases and precision tests."""
    print("Testing prf_floats edge cases...")
    
    key = b"test_key_32_bytes_padded_here!!!"
    info = b"float_test"
    
    # Test invalid range (min >= max)
    try:
        prf_floats(key, info, 10, 1.0, 1.0)
        assert False, "Should raise error when min_val >= max_val"
    except ValueError:
        pass
    
    try:
        prf_floats(key, info, 10, 2.0, 1.0)
        assert False, "Should raise error when min_val > max_val"
    except ValueError:
        pass
    
    # Test standard [0, 1) range
    standard = prf_floats(key, info, 1000)
    assert all(0.0 <= x < 1.0 for x in standard)
    assert min(standard) >= 0.0
    assert max(standard) < 1.0
    
    # Test very small range
    small_range = prf_floats(key, info, 100, 0.0, 0.001)
    assert all(0.0 <= x < 0.001 for x in small_range)
    
    # Test negative range
    negative = prf_floats(key, info, 100, -2.0, -1.0)
    assert all(-2.0 <= x < -1.0 for x in negative)
    
    # Test range crossing zero
    crossing = prf_floats(key, info, 100, -0.5, 0.5)
    assert all(-0.5 <= x < 0.5 for x in crossing)
    
    # Test large range
    large = prf_floats(key, info, 100, 0.0, 1000000.0)
    assert all(0.0 <= x < 1000000.0 for x in large)
    
    # Test precision (no exact 1.0 in [0, 1))
    many_floats = prf_floats(key, info, 10000, 0.0, 1.0)
    assert 1.0 not in many_floats
    assert all(x < 1.0 for x in many_floats)
    
    # Test determinism
    floats1 = prf_floats(key, info, 100, 0.5, 1.5)
    floats2 = prf_floats(key, info, 100, 0.5, 1.5)
    assert floats1 == floats2
    
    # Test zero count
    zero_count = prf_floats(key, info, 0, 0.0, 1.0)
    assert len(zero_count) == 0
    
    # Test distribution mean and variance
    uniform = prf_floats(key, info, 10000, 0.0, 1.0)
    mean = np.mean(uniform)
    var = np.var(uniform)
    # For uniform [0,1), mean should be ~0.5, variance ~1/12
    assert 0.48 < mean < 0.52, f"Mean {mean} not near 0.5"
    assert 0.08 < var < 0.09, f"Variance {var} not near 1/12"
    
    print("✓ prf_floats edge cases handled correctly")
    return True


def test_prf_choice_edge_cases():
    """Test prf_choice with edge cases and various list sizes."""
    print("Testing prf_choice edge cases...")
    
    key = b"test_key_32_bytes_padded_here!!!"
    info = b"choice_test"
    
    # Test empty list (should raise error)
    try:
        prf_choice(key, info, [])
        assert False, "Should raise error for empty choices"
    except ValueError:
        pass
    
    # Test single item
    single = [42]
    for i in range(10):
        chosen = prf_choice(key, info + bytes([i]), single)
        assert chosen == 42
    
    # Test two items
    binary = ["A", "B"]
    selections = [prf_choice(key, info + struct.pack('>I', i), binary) for i in range(1000)]
    count_a = sum(1 for x in selections if x == "A")
    assert 400 < count_a < 600, f"Binary choice biased: {count_a}/1000"
    
    # Test various data types
    mixed = [1, "string", 3.14, None, True, [1, 2], {"key": "value"}]
    chosen = prf_choice(key, info, mixed)
    assert chosen in mixed
    
    # Test large list
    large_list = list(range(1000))
    chosen = prf_choice(key, info, large_list)
    assert 0 <= chosen < 1000
    
    # Test determinism
    choices = ["a", "b", "c", "d", "e"]
    choice1 = prf_choice(key, info, choices)
    choice2 = prf_choice(key, info, choices)
    assert choice1 == choice2
    
    # Test different info produces different choices (usually)
    choices_made = set()
    for i in range(100):
        chosen = prf_choice(key, info + bytes([i]), choices)
        choices_made.add(chosen)
    assert len(choices_made) > 1, "Should select different items with different info"
    
    # Test uniform distribution over many selections
    choices = list(range(10))
    selections = [prf_choice(key, info + struct.pack('>I', i), choices) for i in range(10000)]
    counts = Counter(selections)
    for choice in choices:
        assert 900 < counts[choice] < 1100, f"Choice {choice} non-uniform: {counts[choice]}"
    
    print("✓ prf_choice edge cases handled correctly")
    return True


def test_prf_shuffle_edge_cases():
    """Test prf_shuffle with edge cases and permutation properties."""
    print("Testing prf_shuffle edge cases...")
    
    key = b"test_key_32_bytes_padded_here!!!"
    info = b"shuffle_test"
    
    # Test empty list
    empty = []
    shuffled = prf_shuffle(key, info, empty)
    assert shuffled == []
    
    # Test single item
    single = [42]
    shuffled = prf_shuffle(key, info, single)
    assert shuffled == [42]
    
    # Test two items
    two = [1, 2]
    # Should sometimes swap
    swapped = False
    for i in range(20):
        shuffled = prf_shuffle(key, info + bytes([i]), two)
        if shuffled == [2, 1]:
            swapped = True
            break
    assert swapped, "Two-element list never swapped in 20 tries"
    
    # Test preservation of elements
    original = list(range(100))
    shuffled = prf_shuffle(key, info, original)
    assert sorted(shuffled) == original
    assert shuffled != original, "100-element list not shuffled"
    
    # Test determinism
    items = list(range(50))
    shuffle1 = prf_shuffle(key, info, items)
    shuffle2 = prf_shuffle(key, info, items)
    assert shuffle1 == shuffle2
    
    # Test different info produces different shuffle
    shuffle_a = prf_shuffle(key, b"info_a", items)
    shuffle_b = prf_shuffle(key, b"info_b", items)
    assert shuffle_a != shuffle_b
    
    # Test that it's a proper permutation (no duplicates)
    items = list(range(1000))
    shuffled = prf_shuffle(key, info, items)
    assert len(shuffled) == len(items)
    assert len(set(shuffled)) == len(items)
    
    # Test quality of shuffle (no obvious patterns)
    # Check that elements don't stay in their original positions too often
    stay_count = sum(1 for i, x in enumerate(shuffled) if i == x)
    # For random permutation, expect ~1/e ≈ 0.368 to stay in place
    # We allow generous bounds
    assert stay_count < 50, f"Too many elements stayed in place: {stay_count}/1000"
    
    print("✓ prf_shuffle edge cases handled correctly")
    return True


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("Testing error handling...")
    
    # Test with invalid key sizes (PRF functions should handle any size)
    short_key = b"short"
    long_key = b"very" * 100
    info = b"test"
    
    # These should work with any key size
    prf_bytes(short_key, info, 10)
    prf_bytes(long_key, info, 10)
    
    # Test with None inputs where applicable
    key = b"test_key_32_bytes_padded_here!!!"
    
    # Most functions should handle empty info
    prf_bytes(key, b"", 10)
    prf_expand(key, b"", 10)
    prf_integers(key, b"", 5, 0, 10)
    prf_floats(key, b"", 5)
    
    print("✓ Error handling works correctly")
    return True


def run_all_tests():
    """Run all PRF tests."""
    print("=" * 60)
    print("Running PRF-based Challenge Derivation Tests")
    print("=" * 60)
    
    tests = [
        test_prf_determinism,
        test_different_nonces,
        test_challenge_determinism,
        test_prf_uniformity,
        test_prf_independence,
        test_prf_functions,
        test_compatibility,
        # Edge case tests
        test_prf_bytes_edge_cases,
        test_prf_expand_edge_cases,
        test_prf_integers_edge_cases,
        test_prf_floats_edge_cases,
        test_prf_choice_edge_cases,
        test_prf_shuffle_edge_cases,
        test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
    
    # Optional: visualize distributions
    try:
        visualize_distributions()
    except Exception as e:
        print(f"Note: Visualization skipped ({e})")
    
    print("=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All PRF tests passed successfully!")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)