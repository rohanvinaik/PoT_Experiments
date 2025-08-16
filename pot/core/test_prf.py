"""Tests for PRF-based challenge derivation."""

import sys
import os
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
    prf_derive_key, prf_bytes, prf_derive_seed,
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
    integers = prf_integers(key, b"uniformity_test", n_samples, max_val)
    
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
    selected = prf_choice(key, b"choice_test", choices, 100)
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
    integers = prf_integers(key, b"viz", 10000, 50)
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
        test_compatibility
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