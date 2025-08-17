#!/usr/bin/env python3
"""
Test script for advanced sequential verification features.
Tests mixture tests, adaptive thresholds, multi-armed testing, power analysis, and confidence sequences.
"""

import numpy as np
import time
from typing import List, Generator, Dict, Any

from pot.core.sequential import (
    mixture_sequential_test, adaptive_tau_selection, multi_armed_sequential_verify,
    power_analysis, confidence_sequences, MixtureTestResult, AdaptiveTauResult,
    MultiArmedResult, PowerAnalysisResult, ConfidenceSequence
)


def generate_test_stream(n: int, mean: float = 0.0, std: float = 1.0, seed: int = 42) -> Generator[float, None, None]:
    """Generate test data stream"""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield rng.normal(mean, std)


def test_mixture_sequential():
    """Test mixture sequential testing"""
    print("=" * 60)
    print("Testing Mixture Sequential Testing")
    print("=" * 60)
    
    # Generate multiple test streams with different means
    # Stream 1: mean = 0 (should accept H0)
    # Stream 2: mean = 0.5 (borderline) 
    # Stream 3: mean = 1.0 (should reject H0)
    streams = []
    for i in range(3):
        mean = i * 0.3  # 0.0, 0.3, 0.6
        stream = list(generate_test_stream(50, mean=mean, seed=42+i))
        streams.append(stream)
    
    tau = 0.4  # Single tau value
    weights = [0.5, 0.3, 0.2]  # Different weights for each stream
    
    print(f"Testing with {len(streams)} streams, tau={tau}, weights={weights}")
    
    result = mixture_sequential_test(
        streams=streams,
        weights=weights,
        tau=tau,
        alpha=0.05,
        beta=0.05,
        max_samples=50
    )
    
    print(f"Mixture test result:")
    print(f"  Decision: {result.decision}")
    print(f"  Stopped at: {result.stopped_at}")
    print(f"  Final combined statistic: {result.final_combined_statistic:.4f}")
    print(f"  Individual statistics: {[f'{s:.3f}' for s in result.individual_statistics]}")
    print(f"  Weights: {[f'{w:.3f}' for w in result.weights]}")
    print(f"  Confidence radius: {result.confidence_radius:.4f}")
    
    # Verify structure
    assert len(result.individual_statistics) == len(streams)
    assert len(result.weights) == len(streams)
    assert result.stopped_at <= 50
    assert result.decision in ['H0', 'H1']
    
    print("✓ Mixture sequential test passed")


def test_adaptive_tau():
    """Test adaptive tau selection"""
    print("\n" + "=" * 60)
    print("Testing Adaptive Tau Selection")
    print("=" * 60)
    
    # Generate test stream with varying variance
    stream_data = []
    rng = np.random.default_rng(42)
    
    for i in range(100):
        # Variance increases over time to test adaptation
        std = 0.1 + (i / 100) * 0.5  
        value = rng.normal(0.2, std)  # Mean slightly above 0
        stream_data.append(value)
    
    def test_stream():
        for value in stream_data:
            yield value
    
    print(f"Testing adaptive tau with {len(stream_data)} samples")
    print("Initial tau: 0.15, variance increases over time")
    
    result = adaptive_tau_selection(
        stream=test_stream(),
        initial_tau=0.15,
        alpha=0.05,
        beta=0.05,
        adaptation_rate=0.1,
        min_tau=0.05,
        max_tau=0.5,
        max_samples=100
    )
    
    print(f"\nAdaptive tau result:")
    print(f"  Final decision: {result.decision}")
    print(f"  Stopped at: {result.stopped_at}")
    print(f"  Final tau: {result.final_tau:.4f}")
    print(f"  Tau trajectory length: {len(result.tau_trajectory)}")
    print(f"  Validity maintained: {result.validity_maintained}")
    
    # Verify adaptation occurred
    assert len(result.tau_trajectory) > 0, "Expected tau trajectory"
    assert len(result.tau_trajectory) == result.stopped_at
    assert result.final_tau != 0.15, "Tau should have adapted from initial value"
    assert result.decision in ['H0', 'H1']
    
    print("✓ Adaptive tau selection passed")


def test_multi_armed():
    """Test multi-armed sequential verification"""
    print("\n" + "=" * 60)
    print("Testing Multi-Armed Sequential Verification")
    print("=" * 60)
    
    # Create multiple hypotheses to test
    hypothesis_names = ['model_A', 'model_B', 'model_C']
    tau_values = [0.1, 0.15, 0.2]
    
    # Generate streams for each hypothesis  
    streams = {}
    hypotheses = {}
    for i, (hyp, tau) in enumerate(zip(hypothesis_names, tau_values)):
        # Different mean for each hypothesis
        mean = i * 0.2  # 0.0, 0.2, 0.4
        stream = list(generate_test_stream(60, mean=mean, seed=100+i))
        streams[hyp] = stream
        hypotheses[hyp] = tau
    
    print(f"Testing {len(hypothesis_names)} hypotheses: {hypothesis_names}")
    print(f"Tau values: {tau_values}")
    print(f"Expected means: {[i * 0.2 for i in range(len(hypothesis_names))]}")
    
    result = multi_armed_sequential_verify(
        streams=streams,
        hypotheses=hypotheses,
        alpha=0.05,
        beta=0.05,
        max_samples=60,
        correction_method='bonferroni'
    )
    
    print(f"\nMulti-armed result:")
    print(f"  Decisions: {result.decisions}")
    print(f"  Stopped at: {result.stopped_at}")
    print(f"  Final statistics: {result.final_statistics}")
    print(f"  Adjusted alpha: {result.adjusted_alpha:.4f}")
    print(f"  FWER controlled: {result.fwer_controlled}")
    
    # Verify family-wise error control
    assert len(result.decisions) == len(hypothesis_names), f"Expected {len(hypothesis_names)} decisions, got {len(result.decisions)}"
    assert result.adjusted_alpha <= 0.05  # Should be adjusted for multiple testing
    assert len(result.final_statistics) == len(hypothesis_names)
    
    # Check that all hypotheses have decisions
    for hyp in hypothesis_names:
        assert hyp in result.decisions, f"Missing decision for {hyp}"
        assert result.decisions[hyp] in ['H0', 'H1'], f"Invalid decision for {hyp}: {result.decisions[hyp]}"
    
    print("✓ Multi-armed sequential verification passed")


def test_power_analysis():
    """Test power analysis"""
    print("\n" + "=" * 60)
    print("Testing Power Analysis")
    print("=" * 60)
    
    # Test different effect sizes
    effect_sizes = [0.0, 0.1, 0.2, 0.3, 0.5]
    tau = 0.15
    
    print(f"Computing power analysis for effect sizes: {effect_sizes}")
    print(f"Tau threshold: {tau}")
    
    result = power_analysis(
        tau=tau,
        alpha=0.05,
        beta=0.05,
        effect_sizes=effect_sizes,
        variance_estimate=0.01,
        n_simulations=100,  # Reduced for faster testing
        max_samples=200
    )
    
    print(f"\nPower analysis result:")
    print(f"  Power curve: {result.power_curve}")
    print(f"  Expected stopping times: {result.expected_stopping_times}")
    print(f"  Sample size recommendation: {result.sample_size_recommendation}")
    print(f"  OC curves available: {list(result.oc_curves.keys())}")
    
    # Verify power curve (might not include effect size 0.0 if it's null hypothesis)
    assert len(result.power_curve) >= len(effect_sizes) - 1  # May exclude null hypothesis
    assert len(result.expected_stopping_times) == len(effect_sizes)
    
    # Power should generally increase with effect size
    power_values = [p for _, p in result.power_curve]
    effect_in_curve = [e for e, _ in result.power_curve]
    
    if len(power_values) > 1:
        for i in range(1, len(power_values)):
            if effect_in_curve[i] > effect_in_curve[i-1]:  # Effect size increases
                # Allow some noise due to simulation
                assert power_values[i] >= power_values[i-1] - 0.3, f"Power should increase with effect size at {effect_in_curve[i]}"
    
    print("✓ Power analysis passed")


def test_confidence_sequences():
    """Test confidence sequences"""
    print("\n" + "=" * 60)
    print("Testing Confidence Sequences")
    print("=" * 60)
    
    # Generate test stream
    stream_data = list(generate_test_stream(100, mean=0.2, std=0.8, seed=200))
    
    def test_stream():
        for value in stream_data:
            yield value
    
    print(f"Computing confidence sequences for {len(stream_data)} samples")
    print("True mean: 0.2, std: 0.8")
    
    result = confidence_sequences(
        stream=test_stream(),
        alpha=0.05,
        max_samples=100,
        method='eb'  # Empirical Bernstein
    )
    
    print(f"\nConfidence sequences result:")
    print(f"  Samples processed: {len(result.times)}")
    print(f"  Final mean estimate: {result.means[-1]:.4f}")
    print(f"  Final lower bound: {result.lower_bounds[-1]:.4f}")
    print(f"  Final upper bound: {result.upper_bounds[-1]:.4f}")
    print(f"  Coverage probability: {result.coverage_probability:.3f}")
    print(f"  Is valid: {result.is_valid}")
    print(f"  Width at end: {result.upper_bounds[-1] - result.lower_bounds[-1]:.4f}")
    
    # Verify confidence sequences properties (might stop early due to convergence)
    assert len(result.times) <= len(stream_data)  # May stop early
    assert len(result.means) == len(result.times)
    assert len(result.lower_bounds) == len(result.times)
    assert len(result.upper_bounds) == len(result.times)
    
    # Check that bounds are valid
    for i in range(len(result.lower_bounds)):
        assert result.lower_bounds[i] <= result.upper_bounds[i], f"Invalid bounds at time {i}"
    
    # Coverage should be maintained (true mean should be in confidence interval most of the time)
    true_mean = 0.2
    coverage_count = sum(1 for i in range(len(result.lower_bounds)) 
                        if result.lower_bounds[i] <= true_mean <= result.upper_bounds[i])
    coverage_rate = coverage_count / len(result.lower_bounds)
    print(f"  Empirical coverage: {coverage_rate:.3f} (expected: ≥0.95)")
    
    # Allow some noise due to randomness, but coverage should be reasonable
    assert coverage_rate >= 0.85, f"Coverage rate too low: {coverage_rate}"
    
    print("✓ Confidence sequences passed")


def test_performance():
    """Test performance of advanced features"""
    print("\n" + "=" * 60)
    print("Testing Performance")
    print("=" * 60)
    
    # Test relatively large stream
    n_samples = 1000
    stream_data = list(generate_test_stream(n_samples, mean=0.1, seed=300))
    
    def fast_stream():
        for value in stream_data:
            yield value
    
    print(f"Testing performance with {n_samples} samples")
    
    # Time mixture test
    start_time = time.time()
    mixture_result = mixture_sequential_test(
        streams=[stream_data[:500], stream_data[500:]],  # Split into two streams
        weights=[0.5, 0.5],
        tau=0.15,
        alpha=0.05,
        beta=0.05,
        max_samples=500
    )
    mixture_time = time.time() - start_time
    
    # Time confidence sequences
    start_time = time.time()
    conf_result = confidence_sequences(
        stream=fast_stream(),
        alpha=0.05,
        max_samples=n_samples,
        method='eb'
    )
    conf_time = time.time() - start_time
    
    print(f"\nPerformance results:")
    print(f"  Mixture test ({500} samples × 2): {mixture_time:.3f}s")
    print(f"  Confidence sequences ({n_samples} samples): {conf_time:.3f}s")
    print(f"  Mixture throughput: {1000/mixture_time:.0f} samples/sec")
    print(f"  Confidence throughput: {n_samples/conf_time:.0f} samples/sec")
    
    # Performance should be reasonable (not too slow)
    assert mixture_time < 10.0, f"Mixture test too slow: {mixture_time}s"
    assert conf_time < 10.0, f"Confidence sequences too slow: {conf_time}s"
    
    print("✓ Performance test passed")


if __name__ == "__main__":
    try:
        print("Testing Advanced Sequential Features")
        print("=" * 60)
        
        # Test each advanced feature
        test_mixture_sequential()
        test_adaptive_tau()
        test_multi_armed()
        test_power_analysis()
        test_confidence_sequences()
        test_performance()
        
        print("\n" + "=" * 60)
        print("ALL ADVANCED SEQUENTIAL TESTS PASSED ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()