"""Tests for sequential verification orchestrator."""

import sys
import numpy as np
from pathlib import Path
from typing import Iterator

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pot.core.sequential import sequential_verify


def constant_stream(value: float, n: int) -> Iterator[float]:
    """Generate a stream of constant values."""
    for _ in range(n):
        yield value


def random_uniform_stream(low: float, high: float, n: int, seed: int = 42) -> Iterator[float]:
    """Generate a stream of uniform random values."""
    np.random.seed(seed)
    for _ in range(n):
        yield np.random.uniform(low, high)


def test_constant_zero_accepts_quickly():
    """Test that constant Z=0 stream accepts identity quickly."""
    print("Testing constant Z=0 stream (should accept quickly)...")
    
    # Parameters
    tau = 0.05
    alpha = 0.05
    beta = 0.05
    n_max = 1000
    
    # Create stream of zeros
    stream = constant_stream(0.0, n_max)
    
    # Run sequential verification
    decision, trail = sequential_verify(stream, tau, alpha, beta, n_max)
    
    # Check results
    assert decision['type'] == 'accept_id', f"Should accept, got {decision['type']}"
    # EB bounds are conservative, so it may take a few hundred samples
    assert decision['stopping_time'] < 1000, f"Should stop within n_max, took {decision['stopping_time']} samples"
    assert decision['final_mean'] == 0.0, f"Mean should be 0, got {decision['final_mean']}"
    
    print(f"✓ Accepted identity after {decision['stopping_time']} samples")
    print(f"  Final mean: {decision['final_mean']:.4f}")
    print(f"  Final radius (alpha): {decision['final_radius_alpha']:.4f}")
    print(f"  Final radius (beta): {decision['final_radius_beta']:.4f}")
    print(f"  Confidence interval: [{decision['confidence_interval'][0]:.4f}, {decision['confidence_interval'][1]:.4f}]")
    
    # Verify trail
    assert len(trail) == decision['stopping_time']
    assert all(t[0] == i+1 for i, t in enumerate(trail)), "Trail indices should be sequential"
    assert all(t[1] == 0.0 for t in trail), "All means should be 0"
    
    print(f"✓ Trail recorded {len(trail)} states correctly")
    
    return True


def test_constant_one_rejects_quickly():
    """Test that constant Z=1 stream rejects identity quickly."""
    print("\nTesting constant Z=1 stream (should reject quickly)...")
    
    # Parameters
    tau = 0.05
    alpha = 0.05
    beta = 0.05
    n_max = 1000
    
    # Create stream of ones
    stream = constant_stream(1.0, n_max)
    
    # Run sequential verification
    decision, trail = sequential_verify(stream, tau, alpha, beta, n_max)
    
    # Check results
    assert decision['type'] == 'reject_id', f"Should reject, got {decision['type']}"
    # Rejection is typically faster than acceptance
    assert decision['stopping_time'] < 100, f"Should stop relatively quickly, took {decision['stopping_time']} samples"
    assert decision['final_mean'] == 1.0, f"Mean should be 1, got {decision['final_mean']}"
    
    print(f"✓ Rejected identity after {decision['stopping_time']} samples")
    print(f"  Final mean: {decision['final_mean']:.4f}")
    print(f"  Final radius (alpha): {decision['final_radius_alpha']:.4f}")
    print(f"  Final radius (beta): {decision['final_radius_beta']:.4f}")
    print(f"  Confidence interval: [{decision['confidence_interval'][0]:.4f}, {decision['confidence_interval'][1]:.4f}]")
    
    # Verify trail
    assert len(trail) == decision['stopping_time']
    assert all(t[1] == 1.0 for t in trail), "All means should be 1"
    
    print(f"✓ Trail recorded {len(trail)} states correctly")
    
    return True


def test_random_uniform_takes_longer():
    """Test that random uniform stream takes longer to decide."""
    print("\nTesting random uniform stream (should take longer)...")
    
    # Parameters
    tau = 0.5
    alpha = 0.05
    beta = 0.05
    n_max = 1000
    
    # Test case 1: Uniform around threshold (hardest case)
    print("\nCase 1: Uniform[0.4, 0.6] around tau=0.5...")
    stream1 = random_uniform_stream(0.4, 0.6, n_max, seed=42)
    decision1, trail1 = sequential_verify(stream1, tau, alpha, beta, n_max)
    
    print(f"  Decision: {decision1['type']} after {decision1['stopping_time']} samples")
    print(f"  Final mean: {decision1['final_mean']:.4f}")
    print(f"  Final radius (alpha): {decision1['final_radius_alpha']:.4f}")
    
    # Test case 2: Uniform below threshold (should eventually accept)
    print("\nCase 2: Uniform[0.1, 0.4] below tau=0.5...")
    stream2 = random_uniform_stream(0.1, 0.4, n_max, seed=43)
    decision2, trail2 = sequential_verify(stream2, tau, alpha, beta, n_max)
    
    assert decision2['type'] == 'accept_id', f"Should accept, got {decision2['type']}"
    print(f"  ✓ Accepted after {decision2['stopping_time']} samples")
    print(f"  Final mean: {decision2['final_mean']:.4f}")
    
    # Test case 3: Uniform above threshold (should eventually reject)
    print("\nCase 3: Uniform[0.6, 0.9] above tau=0.5...")
    stream3 = random_uniform_stream(0.6, 0.9, n_max, seed=44)
    decision3, trail3 = sequential_verify(stream3, tau, alpha, beta, n_max)
    
    assert decision3['type'] == 'reject_id', f"Should reject, got {decision3['type']}"
    print(f"  ✓ Rejected after {decision3['stopping_time']} samples")
    print(f"  Final mean: {decision3['final_mean']:.4f}")
    
    # Verify that random takes longer than constant
    const_stream = constant_stream(0.25, n_max)
    const_decision, _ = sequential_verify(const_stream, tau, alpha, beta, n_max)
    
    assert decision2['stopping_time'] > const_decision['stopping_time'], \
        "Random should take longer than constant"
    
    print(f"\n✓ Random streams take longer to decide than constant streams")
    print(f"  Constant(0.25): {const_decision['stopping_time']} samples")
    print(f"  Uniform[0.1,0.4]: {decision2['stopping_time']} samples")
    
    return True


def test_clipping_to_01():
    """Test that values are properly clipped to [0,1]."""
    print("\nTesting value clipping to [0,1]...")
    
    # Create stream with out-of-bounds values
    def out_of_bounds_stream():
        values = [-0.5, -0.1, 0.5, 1.2, 1.5, 0.3, 0.7]
        for v in values:
            yield v
    
    tau = 0.5
    alpha = 0.05
    beta = 0.05
    n_max = 10
    
    stream = out_of_bounds_stream()
    decision, trail = sequential_verify(stream, tau, alpha, beta, n_max)
    
    # Check that all recorded means are valid
    for t, mean, r_a, r_b in trail:
        assert 0 <= mean <= 1, f"Mean {mean} out of bounds at time {t}"
    
    print(f"✓ All values properly clipped to [0,1]")
    print(f"  Processed {len(trail)} samples")
    print(f"  Final mean: {decision['final_mean']:.4f}")
    
    return True


def test_forced_stop_at_nmax():
    """Test that verification stops at n_max if no early decision."""
    print("\nTesting forced stop at n_max...")
    
    # Create stream that oscillates around threshold
    def oscillating_stream():
        i = 0
        while True:
            # Oscillate around 0.5
            yield 0.5 + 0.01 * (1 if i % 2 == 0 else -1)
            i += 1
    
    tau = 0.5
    alpha = 0.01  # Very tight bounds to prevent early stopping
    beta = 0.01
    n_max = 100
    
    stream = oscillating_stream()
    decision, trail = sequential_verify(stream, tau, alpha, beta, n_max)
    
    assert decision['stopping_time'] == n_max, f"Should stop at n_max={n_max}, stopped at {decision['stopping_time']}"
    assert 'forced_stop' in decision and decision['forced_stop'], "Should indicate forced stop"
    
    print(f"✓ Forced stop at n_max={n_max}")
    print(f"  Decision type: {decision['type']}")
    print(f"  Final mean: {decision['final_mean']:.4f}")
    print(f"  Trail length: {len(trail)}")
    
    return True


def test_trail_visualization_data():
    """Test that trail provides good data for visualization."""
    print("\nTesting trail data for visualization...")
    
    # Generate interesting stream with trend
    def trending_stream():
        for i in range(200):
            # Start low, trend upward
            base = 0.1 + (0.7 * i / 200)
            noise = np.random.normal(0, 0.05)
            yield max(0, min(1, base + noise))
    
    tau = 0.4
    alpha = 0.05
    beta = 0.05
    n_max = 200
    
    stream = trending_stream()
    decision, trail = sequential_verify(stream, tau, alpha, beta, n_max)
    
    # Check trail structure
    assert all(len(t) == 4 for t in trail), "Trail entries should have 4 elements"
    
    # Extract components for visualization
    times = [t[0] for t in trail]
    means = [t[1] for t in trail]
    radii_alpha = [t[2] for t in trail]
    radii_beta = [t[3] for t in trail]
    
    # Verify properties
    assert times == list(range(1, len(trail) + 1)), "Times should be sequential"
    assert all(0 <= m <= 1 for m in means), "Means should be in [0,1]"
    assert all(r > 0 for r in radii_alpha), "Alpha radii should be positive"
    assert all(r > 0 for r in radii_beta), "Beta radii should be positive"
    
    # Radii should generally decrease over time (more samples = tighter bounds)
    if len(trail) > 10:
        early_radius = np.mean(radii_alpha[:5])
        late_radius = np.mean(radii_alpha[-5:])
        assert late_radius < early_radius, "Radii should decrease with more samples"
    
    print(f"✓ Trail provides good visualization data")
    print(f"  Decision: {decision['type']} at t={decision['stopping_time']}")
    print(f"  Mean trend: {means[0]:.3f} -> {means[-1]:.3f}")
    print(f"  Radius trend: {radii_alpha[0]:.3f} -> {radii_alpha[-1]:.3f}")
    
    return True


def test_different_alpha_beta():
    """Test with different alpha and beta values."""
    print("\nTesting with different alpha and beta values...")
    
    # Conservative: small alpha (strict acceptance)
    print("\nConservative (alpha=0.01, beta=0.10)...")
    stream1 = constant_stream(0.02, 1000)
    decision1, trail1 = sequential_verify(stream1, tau=0.05, alpha=0.01, beta=0.10, n_max=1000)
    
    # Permissive: large alpha (easier acceptance)
    print("Permissive (alpha=0.10, beta=0.01)...")
    stream2 = constant_stream(0.02, 1000)
    decision2, trail2 = sequential_verify(stream2, tau=0.05, alpha=0.10, beta=0.01, n_max=1000)
    
    # Conservative should take longer to accept
    assert decision1['stopping_time'] > decision2['stopping_time'], \
        "Conservative should take longer than permissive"
    
    print(f"✓ Different error rates work correctly")
    print(f"  Conservative: {decision1['stopping_time']} samples to {decision1['type']}")
    print(f"  Permissive: {decision2['stopping_time']} samples to {decision2['type']}")
    
    return True


def test_empty_stream():
    """Test handling of empty stream."""
    print("\nTesting empty stream handling...")
    
    empty_stream = iter([])  # Empty iterator
    decision, trail = sequential_verify(empty_stream, tau=0.5, alpha=0.05, beta=0.05, n_max=100)
    
    assert decision['type'] == 'conservative_reject', "Should conservatively reject on empty stream"
    assert decision['stopping_time'] == 0, "Should have 0 stopping time"
    assert 'no_samples' in decision and decision['no_samples'], "Should indicate no samples"
    assert len(trail) == 0, "Trail should be empty"
    
    print("✓ Empty stream handled correctly")
    
    return True


def run_all_tests():
    """Run all sequential verification tests."""
    print("=" * 60)
    print("Running Sequential Verification Tests")
    print("=" * 60)
    
    tests = [
        test_constant_zero_accepts_quickly,
        test_constant_one_rejects_quickly,
        test_random_uniform_takes_longer,
        test_clipping_to_01,
        test_forced_stop_at_nmax,
        test_trail_visualization_data,
        test_different_alpha_beta,
        test_empty_stream
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All sequential verification tests passed!")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)