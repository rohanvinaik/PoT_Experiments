"""
Comprehensive unit tests for confidence sequence boundaries.

Tests:
- CSState mean/variance computation
- eb_radius calculation with known inputs
- False positive rate ≤ α over synthetic trials
- Median stopping time for well-separated distributions
"""

import pytest
import numpy as np
from scipy import stats
from typing import List, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pot.core.boundaries import CSState, eb_radius, decide_one_sided, SequentialTest


class TestCSState:
    """Test CSState online statistics computation."""
    
    def test_mean_computation(self):
        """Test that CSState correctly computes running mean."""
        state = CSState()
        values = [0.2, 0.5, 0.3, 0.7, 0.9, 0.1, 0.6, 0.4]
        
        for i, z in enumerate(values, 1):
            state.update(z)
            expected_mean = np.mean(values[:i])
            assert abs(state.mean - expected_mean) < 1e-10, \
                f"Mean mismatch at step {i}: {state.mean} vs {expected_mean}"
    
    def test_variance_computation(self):
        """Test that CSState correctly computes running variance."""
        state = CSState()
        np.random.seed(42)
        values = np.random.uniform(0, 1, 100)
        
        for i, z in enumerate(values):
            state.update(z)
            
            if i >= 1:  # Need at least 2 samples for variance
                expected_var = np.var(values[:i+1], ddof=1)
                assert abs(state.variance - expected_var) < 1e-10, \
                    f"Variance mismatch at step {i+1}: {state.variance} vs {expected_var}"
    
    def test_numerical_stability(self):
        """Test numerical stability with many samples."""
        state = CSState()
        np.random.seed(123)
        n_samples = 10000
        values = np.random.beta(2, 5, n_samples)  # Skewed distribution
        
        for z in values:
            state.update(z)
        
        expected_mean = np.mean(values)
        expected_var = np.var(values, ddof=1)
        
        assert abs(state.mean - expected_mean) < 1e-6, \
            f"Mean unstable: {state.mean} vs {expected_mean}"
        assert abs(state.variance - expected_var) < 1e-6, \
            f"Variance unstable: {state.variance} vs {expected_var}"
    
    def test_boundary_enforcement(self):
        """Test that values are enforced to be in [0,1]."""
        state = CSState()
        
        with pytest.raises(ValueError, match="not in"):
            state.update(-0.1)
        
        with pytest.raises(ValueError, match="not in"):
            state.update(1.1)
        
        # Valid values should work
        state.update(0.0)
        state.update(1.0)
        state.update(0.5)
        assert state.n == 3


class TestEBRadius:
    """Test empirical Bernstein radius calculations."""
    
    def test_known_inputs(self):
        """Test eb_radius with known inputs and expected outputs."""
        # Test case 1: Small sample, high variance
        state1 = CSState()
        for _ in range(5):
            state1.update(0.5)
        
        radius1 = eb_radius(state1, alpha=0.05)
        # With 5 samples of 0.5, variance is 0, so radius should be just bias term
        expected_bias = 3 * np.log(3 * np.log(10) / 0.05) / 5
        assert abs(radius1 - expected_bias) < 0.01
        
        # Test case 2: Large sample, low variance
        state2 = CSState()
        np.random.seed(42)
        for _ in range(1000):
            state2.update(np.random.normal(0.5, 0.01))  # Low variance
        
        radius2 = eb_radius(state2, alpha=0.05)
        assert radius2 < radius1, "Larger sample should have smaller radius"
        assert radius2 < 0.1, "With 1000 samples, radius should be small"
        
        # Test case 3: Verify formula components
        state3 = CSState()
        for val in [0.3, 0.4, 0.5, 0.6, 0.7]:  # Known variance
            state3.update(val)
        
        alpha = 0.01
        radius3 = eb_radius(state3, alpha)
        
        # Manual calculation
        n = state3.n
        v = state3.empirical_variance
        log_term = np.log(3 * np.log(2 * n) / alpha)
        expected = np.sqrt(2 * v * log_term / n) + 3 * log_term / n
        
        assert abs(radius3 - expected) < 1e-10, \
            f"Radius formula mismatch: {radius3} vs {expected}"
    
    def test_radius_decreases_with_samples(self):
        """Test that radius decreases as sample size increases."""
        state = CSState()
        radii = []
        
        np.random.seed(42)
        for i in range(1, 101):
            state.update(np.random.uniform(0.4, 0.6))
            if i >= 2:  # Need at least 2 samples
                radii.append(eb_radius(state, alpha=0.05))
        
        # Check that radius generally decreases
        # Allow some fluctuation due to variance changes
        decreasing_count = sum(1 for i in range(1, len(radii)) 
                              if radii[i] < radii[i-1])
        assert decreasing_count > len(radii) * 0.8, \
            "Radius should generally decrease with more samples"
    
    def test_radius_increases_with_smaller_alpha(self):
        """Test that radius increases as alpha decreases (tighter bounds)."""
        state = CSState()
        np.random.seed(42)
        for _ in range(50):
            state.update(np.random.uniform(0.3, 0.7))
        
        radius_10 = eb_radius(state, alpha=0.10)
        radius_05 = eb_radius(state, alpha=0.05)
        radius_01 = eb_radius(state, alpha=0.01)
        radius_001 = eb_radius(state, alpha=0.001)
        
        assert radius_10 < radius_05 < radius_01 < radius_001, \
            "Radius should increase with smaller alpha"


class TestFalsePositiveRate:
    """Test that false positive rate is controlled at level α."""
    
    def test_false_positive_rate_synthetic(self):
        """Test FPR ≤ α over 50k synthetic trials."""
        np.random.seed(42)
        alpha = 0.05
        n_trials = 50000
        n_samples_per_trial = 100
        threshold = 0.5
        
        false_positives = 0
        
        for trial in range(n_trials):
            # Generate data under H0 (mean = 0.3, below threshold)
            state = CSState()
            for _ in range(n_samples_per_trial):
                # Sample from distribution with mean < threshold
                z = np.random.beta(3, 7)  # Mean = 0.3
                state.update(z)
            
            # Make decision
            decision = decide_one_sided(state, threshold, alpha, "H0")
            
            if decision == "reject_id":
                false_positives += 1
        
        observed_fpr = false_positives / n_trials
        
        # Check that FPR is controlled
        # Allow some margin due to finite samples (use binomial confidence interval)
        margin = 3 * np.sqrt(alpha * (1 - alpha) / n_trials)  # 3 std devs
        
        assert observed_fpr <= alpha + margin, \
            f"FPR {observed_fpr:.4f} exceeds α={alpha} + margin={margin:.4f}"
        
        print(f"Observed FPR: {observed_fpr:.4f} (α={alpha})")
    
    def test_false_positive_rate_multiple_alphas(self):
        """Test FPR control for different alpha levels."""
        np.random.seed(123)
        alphas = [0.01, 0.05, 0.10]
        n_trials = 10000
        
        for alpha in alphas:
            false_positives = 0
            
            for _ in range(n_trials):
                state = CSState()
                # Generate 50 samples from H0
                for _ in range(50):
                    z = np.random.uniform(0.1, 0.4)  # Mean = 0.25
                    state.update(z)
                
                decision = decide_one_sided(state, threshold=0.5, 
                                          alpha=alpha, hypothesis="H0")
                
                if decision == "reject_id":
                    false_positives += 1
            
            observed_fpr = false_positives / n_trials
            margin = 3 * np.sqrt(alpha * (1 - alpha) / n_trials)
            
            assert observed_fpr <= alpha + margin, \
                f"FPR {observed_fpr:.4f} exceeds α={alpha} + margin"


class TestStoppingTime:
    """Test stopping time properties."""
    
    def test_median_stopping_time_separated(self):
        """Test median stopping time for well-separated distributions."""
        np.random.seed(42)
        n_trials = 1000
        
        # Test with well-separated distributions
        stopping_times_h0 = []
        stopping_times_h1 = []
        
        for _ in range(n_trials):
            # H0: mean = 0.2 (well below threshold)
            test_h0 = SequentialTest(threshold=0.5, alpha=0.05)
            for t in range(1, 1000):
                z = np.random.beta(2, 8)  # Mean = 0.2
                decision = test_h0.update(z)
                if decision != "continue":
                    stopping_times_h0.append(t)
                    break
            
            # H1: mean = 0.8 (well above threshold)
            test_h1 = SequentialTest(threshold=0.5, alpha=0.05)
            for t in range(1, 1000):
                z = np.random.beta(8, 2)  # Mean = 0.8
                decision = test_h1.update(z)
                if decision != "continue":
                    stopping_times_h1.append(t)
                    break
        
        median_h0 = np.median(stopping_times_h0)
        median_h1 = np.median(stopping_times_h1)
        
        # Well-separated distributions should stop quickly
        assert median_h0 < 100, f"H0 median stopping time too large: {median_h0}"
        assert median_h1 < 100, f"H1 median stopping time too large: {median_h1}"
        
        # H1 (further from threshold) might stop slightly faster
        print(f"Median stopping times: H0={median_h0:.0f}, H1={median_h1:.0f}")
    
    def test_stopping_time_near_boundary(self):
        """Test that stopping time increases near the decision boundary."""
        np.random.seed(42)
        n_trials = 100
        
        stopping_times_near = []
        stopping_times_far = []
        
        for _ in range(n_trials):
            # Near boundary: mean = 0.48 (close to 0.5 threshold)
            test_near = SequentialTest(threshold=0.5, alpha=0.05, max_samples=500)
            for t in range(1, 501):
                z = np.random.beta(48, 52)  # Mean ≈ 0.48
                decision = test_near.update(z)
                if decision != "continue":
                    stopping_times_near.append(t)
                    break
            else:
                stopping_times_near.append(500)
            
            # Far from boundary: mean = 0.2
            test_far = SequentialTest(threshold=0.5, alpha=0.05, max_samples=500)
            for t in range(1, 501):
                z = np.random.beta(2, 8)  # Mean = 0.2
                decision = test_far.update(z)
                if decision != "continue":
                    stopping_times_far.append(t)
                    break
            else:
                stopping_times_far.append(500)
        
        median_near = np.median(stopping_times_near)
        median_far = np.median(stopping_times_far)
        
        assert median_near > median_far, \
            f"Near-boundary should take longer: {median_near} vs {median_far}"
        
        print(f"Median stopping: near={median_near:.0f}, far={median_far:.0f}")


class TestAnytimeValidity:
    """Test anytime-valid properties of confidence sequences."""
    
    def test_anytime_validity(self):
        """Test that confidence sequences are valid at any stopping time."""
        np.random.seed(42)
        alpha = 0.05
        n_trials = 1000
        
        # Test with various stopping rules
        violations = 0
        
        for _ in range(n_trials):
            state = CSState()
            true_mean = 0.3
            
            # Generate data from true distribution
            for t in range(1, 101):
                z = np.random.beta(3, 7)  # Mean = 0.3
                state.update(z)
                
                # Random stopping time
                if np.random.random() < 0.1:  # 10% chance to stop
                    radius = eb_radius(state, alpha)
                    ci_lower = state.mean - radius
                    ci_upper = state.mean + radius
                    
                    # Check if true mean is in CI
                    if not (ci_lower <= true_mean <= ci_upper):
                        violations += 1
                    break
        
        violation_rate = violations / n_trials
        
        # Violation rate should be ≤ α
        assert violation_rate <= alpha * 1.5, \
            f"Violation rate {violation_rate:.3f} exceeds α={alpha}"


def test_integration():
    """Integration test combining all components."""
    np.random.seed(42)
    
    # Create a sequential test
    test = SequentialTest(threshold=0.5, alpha=0.01, max_samples=200)
    
    # Feed it data
    decisions = []
    for i in range(200):
        z = np.random.uniform(0.2, 0.4)  # Below threshold
        decision = test.update(z)
        decisions.append(decision)
        
        if decision != "continue":
            print(f"Stopped at sample {i+1} with decision: {decision}")
            break
    
    # Should accept H0 (mean is below threshold)
    assert decisions[-1] == "accept_id", "Should accept H0 for low values"
    
    # Check confidence interval
    ci = test.get_confidence_interval()
    assert ci[0] >= 0.0 and ci[1] <= 1.0, "CI should be in [0,1]"
    assert ci[1] < 0.5, "Upper bound should be below threshold"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])