#!/usr/bin/env python3
"""
Comprehensive test suite for sequential verification system.

Tests:
1. Type I error control (H0 streams)
2. Type II error control (H1 streams)  
3. Borderline behavior
4. Anytime validity
5. Numerical stability
6. Trajectory recording
"""

import pytest
import numpy as np
from typing import Iterator, List, Tuple
import random
import time
from dataclasses import dataclass

# Add parent directory to path for pot imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.core.sequential import (
    sequential_verify,
    SPRTResult,
    SequentialState,
    welford_update,
    compute_empirical_variance,
    check_stopping_condition,
    compute_anytime_p_value
)


# ============================================================================
# Fixtures for Reproducible Streams
# ============================================================================

@pytest.fixture
def fixed_seed():
    """Fixture to ensure reproducible randomness."""
    np.random.seed(42)
    random.seed(42)
    return 42


@pytest.fixture
def tau():
    """Default threshold value."""
    return 0.5


@pytest.fixture
def alpha():
    """Default Type I error rate."""
    return 0.05


@pytest.fixture
def beta():
    """Default Type II error rate."""
    return 0.05


@pytest.fixture
def max_samples():
    """Default maximum samples."""
    return 500


def generate_beta_stream(mean: float, variance: float, n: int, seed: int = None) -> Iterator[float]:
    """
    Generate stream from Beta distribution with specified mean and variance.
    
    Args:
        mean: Target mean in [0,1]
        variance: Target variance
        n: Number of samples
        seed: Random seed for reproducibility
    
    Yields:
        Values from Beta distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert mean/variance to Beta parameters
    if variance > 0 and variance < mean * (1 - mean):
        # Valid Beta parameters exist
        a = mean * (mean * (1 - mean) / variance - 1)
        b = (1 - mean) * (mean * (1 - mean) / variance - 1)
        a = max(0.1, a)  # Ensure valid parameters
        b = max(0.1, b)
    else:
        # Fallback to concentrated distribution
        if mean == 0.5:
            a = b = 100
        else:
            a = 100 * mean
            b = 100 * (1 - mean)
    
    for _ in range(n):
        yield np.clip(np.random.beta(a, b), 0, 1)


def generate_normal_stream(mean: float, std: float, n: int, seed: int = None) -> Iterator[float]:
    """
    Generate stream from truncated normal distribution.
    
    Args:
        mean: Target mean
        std: Standard deviation
        n: Number of samples
        seed: Random seed
    
    Yields:
        Values clipped to [0,1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    for _ in range(n):
        yield np.clip(np.random.normal(mean, std), 0, 1)


# ============================================================================
# Test H0 Stream (Type I Error)
# ============================================================================

def test_h0_stream(tau, alpha, beta, max_samples):
    """
    Test Type I error control with H0 streams (true mean < tau).
    
    Verifies:
    - Type I error rate ≤ alpha
    - Average stopping time is reasonable
    - Decisions are mostly H0
    """
    n_runs = 1000
    type_i_errors = 0
    stopping_times = []
    decisions = []
    
    true_mean = tau - 0.1  # 0.4 if tau=0.5
    
    for run in range(n_runs):
        stream = generate_beta_stream(
            mean=true_mean,
            variance=0.01,
            n=max_samples,
            seed=1000 + run
        )
        
        result = sequential_verify(
            stream=stream,
            tau=tau,
            alpha=alpha,
            beta=beta,
            max_samples=max_samples
        )
        
        stopping_times.append(result.stopped_at)
        decisions.append(result.decision)
        
        # Type I error: rejecting H0 when it's true
        if result.decision == 'H1':
            type_i_errors += 1
    
    # Calculate statistics
    observed_type_i = type_i_errors / n_runs
    mean_stopping = np.mean(stopping_times)
    std_stopping = np.std(stopping_times)
    
    # Assertions
    assert observed_type_i <= alpha * 1.5, \
        f"Type I error {observed_type_i:.3f} exceeds limit {alpha * 1.5:.3f}"
    
    assert mean_stopping < max_samples * 0.5, \
        f"Mean stopping time {mean_stopping:.1f} too high"
    
    # Most decisions should be H0
    h0_rate = decisions.count('H0') / n_runs
    assert h0_rate > 0.9, f"H0 acceptance rate {h0_rate:.2f} too low"
    
    print(f"✓ test_h0_stream: Type I={observed_type_i:.3f}, "
          f"mean stop={mean_stopping:.1f}±{std_stopping:.1f}")


# ============================================================================
# Test H1 Stream (Type II Error)
# ============================================================================

def test_h1_stream(tau, alpha, beta, max_samples):
    """
    Test Type II error control with H1 streams (true mean > tau).
    
    Verifies:
    - Type II error rate ≤ beta
    - Power (1 - Type II error) is high
    - Average stopping time is reasonable
    """
    n_runs = 1000
    type_ii_errors = 0
    stopping_times = []
    decisions = []
    
    true_mean = tau + 0.1  # 0.6 if tau=0.5
    
    for run in range(n_runs):
        stream = generate_beta_stream(
            mean=true_mean,
            variance=0.01,
            n=max_samples,
            seed=2000 + run
        )
        
        result = sequential_verify(
            stream=stream,
            tau=tau,
            alpha=alpha,
            beta=beta,
            max_samples=max_samples
        )
        
        stopping_times.append(result.stopped_at)
        decisions.append(result.decision)
        
        # Type II error: accepting H0 when H1 is true
        if result.decision == 'H0':
            type_ii_errors += 1
    
    # Calculate statistics
    observed_type_ii = type_ii_errors / n_runs
    power = 1 - observed_type_ii
    mean_stopping = np.mean(stopping_times)
    std_stopping = np.std(stopping_times)
    
    # Assertions
    assert observed_type_ii <= beta * 1.5, \
        f"Type II error {observed_type_ii:.3f} exceeds limit {beta * 1.5:.3f}"
    
    assert power > 0.9, f"Power {power:.2f} too low"
    
    assert mean_stopping < max_samples * 0.5, \
        f"Mean stopping time {mean_stopping:.1f} too high"
    
    # Most decisions should be H1
    h1_rate = decisions.count('H1') / n_runs
    assert h1_rate > 0.9, f"H1 rejection rate {h1_rate:.2f} too low"
    
    print(f"✓ test_h1_stream: Type II={observed_type_ii:.3f}, Power={power:.2f}, "
          f"mean stop={mean_stopping:.1f}±{std_stopping:.1f}")


# ============================================================================
# Test Borderline Stream
# ============================================================================

def test_borderline_stream(tau, alpha, beta, max_samples):
    """
    Test behavior at tau boundary.
    
    Verifies:
    - Extended stopping times near boundary
    - Decision consistency
    - Many tests reach max_samples
    """
    n_runs = 100  # Fewer runs as these take longer
    stopping_times = []
    decisions = []
    forced_stops = 0
    
    true_mean = tau  # Exactly at threshold
    
    for run in range(n_runs):
        # Use higher variance for more uncertainty
        stream = generate_beta_stream(
            mean=true_mean,
            variance=0.02,
            n=max_samples,
            seed=3000 + run
        )
        
        result = sequential_verify(
            stream=stream,
            tau=tau,
            alpha=alpha,
            beta=beta,
            max_samples=max_samples
        )
        
        stopping_times.append(result.stopped_at)
        decisions.append(result.decision)
        if result.forced_stop:
            forced_stops += 1
    
    # Calculate statistics
    mean_stopping = np.mean(stopping_times)
    std_stopping = np.std(stopping_times)
    forced_rate = forced_stops / n_runs
    
    # Assertions
    # Borderline cases should take longer
    assert mean_stopping > max_samples * 0.3, \
        f"Mean stopping time {mean_stopping:.1f} too low for borderline"
    
    # Many should hit max_samples
    assert forced_rate > 0.2, \
        f"Forced stop rate {forced_rate:.2f} too low for borderline"
    
    # Decisions should be mixed (not all one way)
    h0_rate = decisions.count('H0') / n_runs
    assert 0.2 < h0_rate < 0.8, \
        f"H0 rate {h0_rate:.2f} too extreme for borderline"
    
    print(f"✓ test_borderline_stream: mean stop={mean_stopping:.1f}±{std_stopping:.1f}, "
          f"forced={forced_rate:.2f}")


# ============================================================================
# Test Anytime Validity
# ============================================================================

def test_anytime_validity(tau, alpha, beta):
    """
    Test that confidence bounds remain valid at any stopping time.
    
    Verifies:
    - Stopping at random times maintains validity
    - Adversarial stopping rules don't break guarantees
    - Coverage remains at 1-alpha level
    """
    n_runs = 500
    
    # Test different stopping strategies
    strategies = [
        ('random', lambda t, state: random.random() < 0.1),  # Random 10% stop
        ('early', lambda t, state: t >= 20),  # Always stop early
        ('late', lambda t, state: t >= 200),  # Always stop late
        ('adversarial', lambda t, state: abs(state.mean - tau) < 0.05 and t > 50)  # Stop when close to tau
    ]
    
    for strategy_name, stop_rule in strategies:
        coverage_count = 0
        true_mean = 0.45  # Known true mean
        
        for run in range(n_runs):
            stream = generate_beta_stream(
                mean=true_mean,
                variance=0.01,
                n=500,
                seed=4000 + run * 10 + hash(strategy_name) % 100
            )
            
            # Manual verification with custom stopping
            state = SequentialState()
            for t, x in enumerate(stream, 1):
                state = welford_update(state, x)
                
                # Apply stopping rule
                if stop_rule(t, state):
                    break
            
            # Check if true mean is in confidence interval
            from pot.core.boundaries import CSState, eb_radius
            cs_state = CSState()
            cs_state.n = state.n
            cs_state.mean = state.mean
            cs_state.M2 = state.M2
            
            radius = eb_radius(cs_state, alpha)
            lower = max(0, state.mean - radius)
            upper = min(1, state.mean + radius)
            
            if lower <= true_mean <= upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_runs
        
        # Coverage should be at least 1-alpha (with some tolerance)
        expected_coverage = 1 - alpha
        tolerance = 2 * np.sqrt(alpha * (1 - alpha) / n_runs)
        
        assert coverage_rate >= expected_coverage - tolerance, \
            f"Coverage {coverage_rate:.3f} too low for {strategy_name} stopping"
        
        print(f"✓ test_anytime_validity ({strategy_name}): coverage={coverage_rate:.3f}")


# ============================================================================
# Test Numerical Stability
# ============================================================================

def test_numerical_stability():
    """
    Test numerical stability with extreme conditions.
    
    Verifies:
    - Very long sequences (>100k samples)
    - Extreme values
    - Variance computation stability
    """
    
    # Test 1: Very long sequence
    print("Testing very long sequence...")
    n_long = 100000
    true_mean = 0.5
    true_var = 0.01
    
    # Generate long stream
    np.random.seed(5000)
    long_stream = (np.clip(np.random.normal(true_mean, np.sqrt(true_var)), 0, 1) 
                   for _ in range(n_long))
    
    # Track state manually
    state = SequentialState()
    for i, x in enumerate(long_stream, 1):
        state = welford_update(state, x)
        
        # Check periodically
        if i in [1000, 10000, 50000, 100000]:
            var = compute_empirical_variance(state)
            assert var >= 0, f"Variance negative at n={i}"
            assert abs(state.mean - true_mean) < 0.1, f"Mean drift at n={i}"
    
    final_var = compute_empirical_variance(state)
    print(f"  Final: n={state.n}, mean={state.mean:.6f}, var={final_var:.6f}")
    assert final_var > 0, "Final variance should be positive"
    assert abs(state.mean - true_mean) < 0.01, "Mean should converge"
    
    # Test 2: Extreme values (near 0 and 1)
    print("Testing extreme values...")
    extreme_state = SequentialState()
    extreme_values = [1e-10, 1-1e-10, 1e-10, 1-1e-10] * 100
    
    for x in extreme_values:
        extreme_state = welford_update(extreme_state, x)
    
    extreme_var = compute_empirical_variance(extreme_state)
    assert extreme_var >= 0, "Variance should handle extreme values"
    assert 0 <= extreme_state.mean <= 1, "Mean should stay in bounds"
    
    # Test 3: Constant values (zero variance)
    print("Testing zero variance...")
    const_state = SequentialState()
    const_value = 0.7
    
    for _ in range(1000):
        const_state = welford_update(const_state, const_value)
    
    const_var = compute_empirical_variance(const_state)
    assert const_var < 1e-10, "Variance should be ~0 for constant"
    assert abs(const_state.mean - const_value) < 1e-10, "Mean should equal constant"
    
    # Test 4: Alternating values (high variance)
    print("Testing high variance...")
    alt_state = SequentialState()
    
    for i in range(10000):
        alt_state = welford_update(alt_state, i % 2)
    
    alt_var = compute_empirical_variance(alt_state)
    assert abs(alt_var - 0.25) < 0.01, "Variance should be ~0.25 for 0/1 alternating"
    assert abs(alt_state.mean - 0.5) < 0.01, "Mean should be ~0.5"
    
    print("✓ test_numerical_stability: All stability tests passed")


# ============================================================================
# Test Trajectory Recording
# ============================================================================

def test_trajectory_recording(tau, alpha, beta, max_samples):
    """
    Test trajectory recording and audit trail.
    
    Verifies:
    - Complete audit trail is maintained
    - State updates are correct
    - Memory efficiency (trajectory length matches stopped_at)
    """
    # Generate a simple stream
    stream = generate_beta_stream(
        mean=0.3,  # Clear H0 case
        variance=0.01,
        n=max_samples,
        seed=6000
    )
    
    result = sequential_verify(
        stream=stream,
        tau=tau,
        alpha=alpha,
        beta=beta,
        max_samples=max_samples
    )
    
    # Test 1: Trajectory length matches stopping point
    assert len(result.trajectory) == result.stopped_at, \
        f"Trajectory length {len(result.trajectory)} != stopped_at {result.stopped_at}"
    
    # Test 2: States are properly ordered
    for i, state in enumerate(result.trajectory, 1):
        assert state.n == i, f"State {i} has wrong n: {state.n}"
        
        if i > 1:
            # Variance should be non-negative
            assert state.variance >= 0, f"Negative variance at step {i}"
            
            # Mean should be bounded
            assert 0 <= state.mean <= 1, f"Mean out of bounds at step {i}"
    
    # Test 3: Final state matches result
    if result.trajectory:
        final_state = result.trajectory[-1]
        assert abs(final_state.mean - result.final_mean) < 1e-10, \
            "Final mean mismatch"
        assert abs(final_state.variance - result.final_variance) < 1e-10, \
            "Final variance mismatch"
    
    # Test 4: State progression is consistent
    if len(result.trajectory) > 10:
        # Check a few intermediate states
        for check_idx in [4, 9, min(19, len(result.trajectory)-1)]:
            state = result.trajectory[check_idx]
            
            # Recompute from scratch to verify
            verify_state = SequentialState()
            # Generate exactly n samples where n = index + 1 (since 0-indexed)
            n_samples = check_idx + 1
            stream_copy = list(generate_beta_stream(
                mean=0.3, variance=0.01, n=n_samples, seed=6000
            ))
            
            for x in stream_copy:
                verify_state = welford_update(verify_state, x)
            
            # States should match (approximately due to floating point)
            assert abs(state.mean - verify_state.mean) < 1e-6, \
                f"Mean mismatch at step {check_idx+1} (index {check_idx})"
    
    # Test 5: Memory efficiency
    import sys
    trajectory_size = sys.getsizeof(result.trajectory)
    state_size = sys.getsizeof(result.trajectory[0]) if result.trajectory else 0
    expected_size = state_size * len(result.trajectory) * 2  # Allow 2x overhead
    
    assert trajectory_size < expected_size, \
        f"Trajectory too large: {trajectory_size} bytes"
    
    print(f"✓ test_trajectory_recording: stopped_at={result.stopped_at}, "
          f"trajectory_len={len(result.trajectory)}, "
          f"size={trajectory_size} bytes")


# ============================================================================
# Test P-value Computation
# ============================================================================

def test_p_value_computation(tau):
    """
    Test anytime-valid p-value computation.
    
    Verifies:
    - P-values are in [0,1]
    - P-values are conservative (valid despite stopping)
    - Monotonicity with evidence strength
    """
    # Test various scenarios
    test_cases = [
        (0.3, 100, "strong_h0"),  # Strong evidence for H0
        (0.7, 100, "strong_h1"),  # Strong evidence against H0
        (0.5, 100, "neutral"),    # No evidence
        (0.45, 10, "weak_h0"),    # Weak evidence for H0
        (0.55, 10, "weak_h1"),    # Weak evidence against H0
    ]
    
    for mean, n, case_name in test_cases:
        # Create state with specified statistics
        state = SequentialState()
        np.random.seed(7000)
        
        # Generate samples with target mean
        for _ in range(n):
            # Use beta distribution for bounded samples
            if mean == 0.5:
                x = 0.5  # Constant for neutral case
            else:
                a = mean * 10
                b = (1 - mean) * 10
                x = np.random.beta(a, b)
            state = welford_update(state, x)
        
        # Compute p-value
        p_value = compute_anytime_p_value(state, tau)
        
        # Assertions
        assert 0 <= p_value <= 1, f"P-value {p_value} out of bounds for {case_name}"
        
        # Check expected ranges
        if case_name == "strong_h0":
            assert p_value > 0.5, f"P-value {p_value:.3f} too low for strong H0"
        elif case_name == "strong_h1":
            assert p_value < 0.1, f"P-value {p_value:.3f} too high for strong H1"
        elif case_name == "neutral":
            # Neutral case (mean=tau) should have high p-value (no evidence against H0)
            assert 0.3 <= p_value <= 1.0, f"P-value {p_value:.3f} unexpected for neutral"
        
        print(f"  {case_name}: mean={mean}, n={n}, p-value={p_value:.4f}")
    
    print("✓ test_p_value_computation: All p-value tests passed")


# ============================================================================
# Integration Test
# ============================================================================

def test_integration(tau, alpha, beta, max_samples):
    """
    Integration test combining multiple aspects.
    
    Verifies:
    - Complete workflow from stream to decision
    - All components work together
    - Performance is reasonable
    """
    import time
    
    # Run a batch of verifications
    n_tests = 100
    total_time = 0
    total_samples = 0
    
    for i in range(n_tests):
        # Alternate between H0 and H1 cases
        true_mean = 0.4 if i % 2 == 0 else 0.6
        
        stream = generate_beta_stream(
            mean=true_mean,
            variance=0.015,
            n=max_samples,
            seed=8000 + i
        )
        
        start = time.time()
        result = sequential_verify(
            stream=stream,
            tau=tau,
            alpha=alpha,
            beta=beta,
            max_samples=max_samples,
            compute_p_value=True
        )
        elapsed = time.time() - start
        
        total_time += elapsed
        total_samples += result.stopped_at
        
        # Basic checks
        assert result.decision in ['H0', 'H1', 'continue']
        assert result.stopped_at > 0
        assert result.final_mean >= 0 and result.final_mean <= 1
        assert result.confidence_radius > 0
        assert len(result.trajectory) == result.stopped_at
        if result.p_value is not None:
            assert 0 <= result.p_value <= 1
    
    avg_time = total_time / n_tests
    avg_samples = total_samples / n_tests
    
    print(f"✓ test_integration: {n_tests} tests, "
          f"avg_time={avg_time*1000:.1f}ms, "
          f"avg_samples={avg_samples:.1f}")
    
    # Performance assertions
    assert avg_time < 0.01, f"Average time {avg_time:.3f}s too slow"
    assert avg_samples < max_samples * 0.3, f"Average samples {avg_samples:.1f} too high"


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    """Run all tests with default parameters."""
    
    print("=" * 60)
    print("COMPREHENSIVE TEST SUITE FOR SEQUENTIAL VERIFICATION")
    print("=" * 60)
    
    # Default parameters
    tau = 0.5
    alpha = 0.05
    beta = 0.05
    max_samples = 500
    
    print(f"\nTest Parameters:")
    print(f"  tau = {tau}")
    print(f"  alpha = {alpha}")
    print(f"  beta = {beta}")
    print(f"  max_samples = {max_samples}")
    print()
    
    # Run all tests
    try:
        print("Running test_h0_stream...")
        test_h0_stream(tau, alpha, beta, max_samples)
        
        print("\nRunning test_h1_stream...")
        test_h1_stream(tau, alpha, beta, max_samples)
        
        print("\nRunning test_borderline_stream...")
        test_borderline_stream(tau, alpha, beta, max_samples)
        
        print("\nRunning test_anytime_validity...")
        test_anytime_validity(tau, alpha, beta)
        
        print("\nRunning test_numerical_stability...")
        test_numerical_stability()
        
        print("\nRunning test_trajectory_recording...")
        test_trajectory_recording(tau, alpha, beta, max_samples)
        
        print("\nRunning test_p_value_computation...")
        test_p_value_computation(tau)
        
        print("\nRunning test_integration...")
        test_integration(tau, alpha, beta, max_samples)
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        raise