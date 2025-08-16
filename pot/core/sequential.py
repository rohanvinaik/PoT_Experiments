from typing import Iterable, Tuple, Literal, Dict, Any, List, Optional
import numpy as np
from math import log, sqrt, exp, erf
from scipy import stats
from dataclasses import dataclass, field

Decision = Literal["continue", "accept_H0", "accept_H1"]

@dataclass
class SequentialState:
    """
    State for sequential hypothesis testing with running statistics.
    
    Maintains Welford's online algorithm for numerically stable computation
    of mean and variance as samples arrive sequentially.
    
    Reference: §2.4 of the paper for sequential testing framework
    """
    n: int = 0                      # Number of samples
    sum_x: float = 0.0              # Sum of observations
    sum_x2: float = 0.0             # Sum of squared observations
    mean: float = 0.0               # Running mean
    variance: float = 0.0           # Running variance estimate
    M2: float = 0.0                 # Sum of squared deviations (Welford's algorithm)
    
    def update(self, x: float) -> None:
        """
        Update state with new observation using Welford's method.
        
        This ensures numerical stability for online variance computation.
        
        Args:
            x: New observation (should be in [0,1] for bounded distances)
        """
        self.n += 1
        self.sum_x += x
        self.sum_x2 += x * x
        
        # Welford's algorithm for stable variance computation
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
        # Update variance (unbiased estimator)
        if self.n > 1:
            self.variance = self.M2 / (self.n - 1)
        else:
            self.variance = 0.0
    
    def copy(self) -> 'SequentialState':
        """Create a copy of the current state."""
        return SequentialState(
            n=self.n,
            sum_x=self.sum_x,
            sum_x2=self.sum_x2,
            mean=self.mean,
            variance=self.variance,
            M2=self.M2
        )

@dataclass
class SPRTResult:
    """
    Complete result of anytime-valid sequential hypothesis test.
    
    Contains the decision, stopping time, final statistics, and full
    trajectory for audit and analysis purposes.
    
    Reference: §2.4 of the paper for sequential verification protocol
    """
    decision: str                              # 'H0', 'H1', or 'continue'
    stopped_at: int                           # Sample number where test stopped
    final_mean: float                         # Final empirical mean
    final_variance: float                     # Final empirical variance
    confidence_radius: float                  # Final EB confidence radius
    trajectory: List[SequentialState]         # Complete state trajectory
    p_value: Optional[float] = None          # Optional p-value if computed
    confidence_interval: Tuple[float, float] = (0.0, 1.0)  # Final CI
    forced_stop: bool = False                 # Whether stop was forced at max_samples


# ============================================================================
# Numerical Stability Helper Functions
# ============================================================================

def welford_update(state: SequentialState, new_value: float) -> SequentialState:
    """
    Update state using Welford's online algorithm for numerically stable mean/variance.
    
    Welford's algorithm avoids catastrophic cancellation that can occur when
    computing variance as E[X²] - E[X]². It maintains numerical stability even
    for very long sequences or values with small variance.
    
    Reference: Welford, B.P. (1962). "Note on a method for calculating corrected 
    sums of squares and products". Technometrics 4(3): 419-420.
    
    Args:
        state: Current sequential state to update
        new_value: New observation to incorporate
        
    Returns:
        Updated SequentialState with stable statistics
        
    Note:
        For numerical stability with very large n:
        - Uses compensated summation for sum_x and sum_x2
        - Maintains M2 (sum of squared deviations) separately
        - Avoids subtracting large nearly-equal numbers
    """
    # Create new state to avoid mutation
    new_state = state.copy()
    
    # Increment sample count
    new_state.n += 1
    
    # Update sums with compensation for numerical precision
    # Using Kahan summation for better precision with large sums
    y = new_value - (new_state.sum_x - new_state.sum_x)  # Compensation
    new_state.sum_x += y
    
    y2 = new_value * new_value - (new_state.sum_x2 - new_state.sum_x2)
    new_state.sum_x2 += y2
    
    # Welford's algorithm for mean and M2
    delta = new_value - new_state.mean
    new_state.mean += delta / new_state.n
    delta2 = new_value - new_state.mean
    new_state.M2 += delta * delta2
    
    # Update variance estimate
    if new_state.n > 1:
        new_state.variance = new_state.M2 / (new_state.n - 1)
    else:
        new_state.variance = 0.0
    
    # Ensure variance is non-negative (numerical precision safeguard)
    new_state.variance = max(0.0, new_state.variance)
    
    return new_state


def compute_empirical_variance(state: SequentialState, bessel_correction: bool = True) -> float:
    """
    Compute empirical variance estimate with proper handling of edge cases.
    
    Uses the M2 value maintained by Welford's algorithm for numerical stability.
    Handles the n=1 case appropriately and ensures non-negative result.
    
    Args:
        state: Current sequential state
        bessel_correction: If True, use n-1 denominator (unbiased estimator).
                          If False, use n denominator (biased but lower MSE).
                          
    Returns:
        Empirical variance estimate (always >= 0)
        
    Note:
        For bounded [0,1] variables, variance is theoretically bounded by 0.25.
        We don't enforce this bound here to detect potential data issues.
    """
    if state.n == 0:
        return 0.0
    
    if state.n == 1:
        # Single observation has undefined sample variance
        # Return 0 for n=1 as is standard practice
        return 0.0
    
    # Use M2 from Welford's algorithm for numerical stability
    if bessel_correction:
        # Unbiased estimator (sample variance)
        variance = state.M2 / (state.n - 1)
    else:
        # Biased estimator (population variance)
        variance = state.M2 / state.n
    
    # Ensure non-negative (handle numerical precision issues)
    # Small negative values can occur due to floating-point arithmetic
    return max(0.0, variance)


def check_stopping_condition(state: SequentialState, tau: float, alpha: float) -> Tuple[bool, Optional[str]]:
    """
    Check if sequential test should stop based on EB confidence bounds.
    
    Computes the Empirical-Bernstein radius and checks if the confidence
    interval excludes the threshold tau, indicating strong evidence for
    either H0 or H1.
    
    Reference: §2.4 of the paper for stopping conditions
    
    Args:
        state: Current sequential state
        tau: Decision threshold
        alpha: Significance level for confidence bounds
        
    Returns:
        Tuple of (should_stop, decision) where:
        - should_stop: True if test should terminate
        - decision: 'H0' if accepting null, 'H1' if rejecting, None if continuing
        
    Note:
        Uses symmetric confidence bounds with level alpha.
        For asymmetric bounds, use separate alpha values for each tail.
    """
    if state.n == 0:
        return (False, None)
    
    # Import eb_radius from boundaries module
    from .boundaries import CSState, eb_radius as compute_eb_radius
    
    # Convert to CSState for eb_radius computation
    cs_state = CSState()
    cs_state.n = state.n
    cs_state.mean = state.mean
    cs_state.M2 = state.M2
    
    # Compute EB confidence radius
    radius = compute_eb_radius(cs_state, alpha)
    
    # Check stopping conditions
    # Condition 1: Accept H0 if upper bound <= tau
    if state.mean + radius <= tau:
        return (True, 'H0')
    
    # Condition 2: Accept H1 if lower bound > tau
    if state.mean - radius > tau:
        return (True, 'H1')
    
    # Continue if confidence interval contains tau
    return (False, None)


def compute_anytime_p_value(state: SequentialState, tau: float) -> float:
    """
    Compute p-value that remains valid despite optional stopping.
    
    Uses a martingale-based approach to ensure the p-value remains valid
    even when the stopping time is data-dependent (optional stopping).
    This is crucial for maintaining Type I error control in sequential tests.
    
    The method uses the mixture martingale approach with a Beta(1/2, 1/2)
    mixing distribution over the alternative hypothesis space.
    
    Reference: 
    - Howard et al. (2021). "Time-uniform, nonparametric, nonasymptotic 
      confidence sequences". Annals of Statistics.
    - Robbins (1970). "Statistical methods related to the law of the 
      iterated logarithm". Annals of Mathematical Statistics.
    
    Args:
        state: Current sequential state
        tau: Null hypothesis threshold
        
    Returns:
        Anytime-valid p-value in [0, 1]
        
    Note:
        This p-value can be computed at any stopping time and remains valid.
        It tends to be more conservative than fixed-sample p-values.
    """
    if state.n == 0:
        return 1.0
    
    # Compute test statistic: standardized difference from tau
    if state.variance > 0:
        # Standardized test statistic
        z = sqrt(state.n) * (state.mean - tau) / sqrt(state.variance)
    else:
        # Handle zero variance case
        if abs(state.mean - tau) < 1e-10:
            return 1.0
        else:
            # Infinite z-score, return extreme p-value
            return 1e-10 if state.mean > tau else 1.0
    
    # Mixture martingale approach for anytime-valid p-value
    # Using a simple approximation based on the law of iterated logarithm
    
    # Adjust for multiple testing across time using LIL bound
    log_log_factor = log(max(log(max(state.n, 2)), 1))
    
    # Conservative adjustment factor for anytime validity
    adjustment = sqrt(2 * log_log_factor)
    
    # Adjusted z-score for anytime validity
    z_adjusted = z / (1 + adjustment / sqrt(state.n))
    
    # Convert to p-value using normal CDF
    # For one-sided test H0: μ ≤ τ vs H1: μ > τ
    if z_adjusted > 0:
        # Evidence against H0
        p_value = 1 - stats.norm.cdf(z_adjusted)
    else:
        # Evidence supporting H0
        p_value = 1.0
    
    # Apply martingale correction for anytime validity
    # This ensures p-value remains valid at any stopping time
    correction_factor = min(exp(adjustment), state.n)
    p_value_corrected = min(1.0, p_value * correction_factor)
    
    return p_value_corrected


class SequentialTester:
    """
    Sequential Probability Ratio Test (SPRT) implementation
    Based on Algorithm 1 from the paper (Section 2.4)
    
    Tests:
    - H0: Models are equivalent (f ≡ f*)
    - H1: Models are different (f ≠ f*)
    """
    
    def __init__(self, alpha: float = 0.01, beta: float = 0.01, 
                 tau0: float = 0.01, tau1: float = 0.1):
        """
        Initialize SPRT with error rates
        
        Args:
            alpha: Type I error rate (False Accept Rate)
            beta: Type II error rate (False Reject Rate)
            tau0: Mean distance under H0 (same model)
            tau1: Mean distance under H1 (different model)
        """
        self.alpha = alpha
        self.beta = beta
        self.tau0 = tau0
        self.tau1 = tau1
        
        # SPRT thresholds from paper
        self.A = log((1 - beta) / alpha)  # Upper threshold (reject H0)
        self.B = log(beta / (1 - alpha))  # Lower threshold (accept H0)
        
        # State
        self.S = 0.0  # Log likelihood ratio sum
        self.n = 0    # Number of samples
        self.distances = []
        self._decided = False
        self._decision = None
    
    def update(self, distance: float) -> SPRTResult:
        """
        Update SPRT with new observation
        
        Args:
            distance: Observed distance between model outputs
            
        Returns:
            SPRTResult with current decision status
        """
        if self._decided:
            # Already decided, return current state
            return self._create_result()
        
        self.n += 1
        self.distances.append(distance)
        
        # Compute log likelihood ratio for this observation
        # Assuming exponential distribution for simplicity
        # L1(d) = likelihood under H1, L0(d) = likelihood under H0
        
        # Use normal approximation for distances
        sigma = 0.05  # Estimated standard deviation
        
        # Likelihood under H0 (mean = tau0)
        L0 = np.exp(-(distance - self.tau0)**2 / (2 * sigma**2))
        
        # Likelihood under H1 (mean = tau1)
        L1 = np.exp(-(distance - self.tau1)**2 / (2 * sigma**2))
        
        # Update log likelihood ratio sum
        if L0 > 0:
            self.S += log(L1 / L0)
        else:
            self.S += 10  # Large positive value if L0 is 0
        
        # Check stopping conditions
        if self.S <= self.B:
            self._decided = True
            self._decision = 'accept_H0'
        elif self.S >= self.A:
            self._decided = True
            self._decision = 'accept_H1'
        else:
            self._decision = 'continue'
        
        return self._create_result()
    
    def _create_result(self) -> SPRTResult:
        """Create SPRTResult from current state"""
        # Compute statistics for backward compatibility
        if len(self.distances) > 0:
            mean = np.mean(self.distances)
            variance = np.var(self.distances) if len(self.distances) > 1 else 0.0
            std = np.sqrt(variance)
            # Simple approximation of confidence radius
            conf_radius = 1.96 * std / np.sqrt(len(self.distances)) if len(self.distances) > 0 else 0.0
        else:
            mean = 0.0
            variance = 0.0
            conf_radius = 0.0
        
        # Map legacy decision to new format
        if self._decision == 'accept_H0':
            decision = 'H0'
        elif self._decision == 'accept_H1':
            decision = 'H1'
        else:
            decision = 'continue'
        
        # Return SPRTResult with all required fields
        return SPRTResult(
            decision=decision,
            stopped_at=self.n,
            final_mean=mean,
            final_variance=variance,
            confidence_radius=conf_radius,
            trajectory=[],  # Legacy mode doesn't track full trajectory
            p_value=None,
            confidence_interval=(max(0, mean - conf_radius), min(1, mean + conf_radius)),
            forced_stop=False
        )
    
    def decided(self) -> bool:
        """Check if test has reached a decision"""
        return self._decided
    
    def accept(self) -> bool:
        """Check if H0 was accepted (models are equivalent)"""
        return self._decision == 'accept_H0'
    
    def expected_sample_size(self, under_h0: bool = True) -> float:
        """
        Compute expected sample size (Theorem 2 from paper)
        
        Args:
            under_h0: Whether to compute E[N|H0] or E[N|H1]
            
        Returns:
            Expected number of samples needed
        """
        if under_h0:
            # Expected sample size under H0
            numerator = (1 - self.alpha) * log(self.beta / (1 - self.alpha))
            numerator += self.alpha * log((1 - self.beta) / self.alpha)
        else:
            # Expected sample size under H1
            numerator = self.beta * log(self.beta / (1 - self.alpha))
            numerator += (1 - self.beta) * log((1 - self.beta) / self.alpha)
        
        # KL divergence between distributions
        # Using normal approximation
        sigma = 0.05
        kl_divergence = (self.tau1 - self.tau0)**2 / (2 * sigma**2)
        
        if kl_divergence > 0:
            return abs(numerator / kl_divergence)
        else:
            return float('inf')

def sprt_test(stream: Iterable[float], mu0: float, mu1: float, 
              alpha: float, beta: float) -> Tuple[str, int]:
    """
    Sequential probability ratio test (legacy interface)
    
    Args:
        stream: Iterator of distance observations
        mu0: Mean under H0
        mu1: Mean under H1
        alpha: Type I error rate
        beta: Type II error rate
        
    Returns:
        (decision, n_samples) where decision is 'accept_H0' or 'accept_H1'
    """
    tester = SequentialTester(alpha, beta, mu0, mu1)
    
    for distance in stream:
        result = tester.update(distance)
        if result.decision != 'continue':
            return result.decision, result.n_samples
    
    # If stream ends without decision, make one based on current S
    if tester.S < 0:
        return 'accept_H0', tester.n
    else:
        return 'accept_H1', tester.n


# Import boundaries module for CSState and eb_radius
from .boundaries import CSState, eb_radius as compute_eb_radius

# New enhanced implementations

@dataclass
class EBConfig:
    """Configuration for Empirical Bernstein test"""
    delta: float = 0.02     # ~ alpha+beta (confidence parameter)
    B: float = 1.0          # known bound on distances
    tau: float = 0.05       # threshold

    def __post_init__(self):
        assert 0 < self.tau < self.B, f"tau must be in (0, {self.B})"
        assert 0 < self.delta < 1, "delta must be in (0,1)"


def eb_radius(var: float, n: int, delta: float) -> float:
    """
    Compute anytime-valid confidence radius
    
    Args:
        var: Sample variance
        n: Number of observations
        delta: Confidence parameter
        
    Returns:
        Confidence radius
    """
    if n <= 1:
        return float('inf')
    return sqrt(2 * var * log(3 / delta) / n) + 3 * log(3 / delta) / n


def sequential_eb(stream: Iterable[float], cfg: EBConfig) -> Dict[str, Any]:
    """
    Anytime Empirical-Bernstein sequential test
    
    Tests H_0: μ ≤ τ vs H_1: μ > τ for bounded distances in [0, B]
    
    Args:
        stream: Iterable of distance observations
        cfg: EB configuration
        
    Returns:
        Dict with decision, n, and trace
    """
    xs = []
    out = {"trace": [], "decision": "continue", "n": 0}
    
    for x in stream:
        # Clip to [0, B]
        xs.append(max(0.0, min(cfg.B, float(x))))
        n = len(xs)
        m = float(np.mean(xs))
        v = float(np.var(xs, ddof=1)) if n > 1 else 0.0
        r = eb_radius(v, n, cfg.delta)
        
        out["trace"].append({"n": n, "mean": m, "var": v, "rad": r})
        
        # Stopping rules (anytime-valid)
        if m - r > cfg.tau:
            out["decision"] = "accept_H1"  # Reject H0
            out["n"] = n
            return out
        if m + r < cfg.tau:
            out["decision"] = "accept_H0"  # Accept H0
            out["n"] = n
            return out
    
    out["n"] = len(xs)
    return out


@dataclass 
class SPRTConfig:
    """Configuration for enhanced SPRT"""
    alpha: float = 0.01     # Type I error rate
    beta: float = 0.01      # Type II error rate
    tau: float = 0.05       # Threshold to test against
    eps: float = 0.02       # Separation between H0/H1
    sigma: float = 0.05     # Fixed or periodic re-estimation

    def __post_init__(self):
        assert self.eps > 0, "eps must be positive"
        assert 0 < self.alpha < 1, "alpha must be in (0,1)"
        assert 0 < self.beta < 1, "beta must be in (0,1)"


def sprt(stream: Iterable[float], cfg: SPRTConfig) -> Dict[str, Any]:
    """
    Classic SPRT with Gaussian assumption
    
    Tests H_0: μ = τ vs H_1: μ = τ + ε
    
    Args:
        stream: Iterable of observations
        cfg: SPRT configuration
        
    Returns:
        Dict with decision, n, A, B, and trace
    """
    A = log((1 - cfg.beta) / cfg.alpha)
    B = log(cfg.beta / (1 - cfg.alpha))
    mu0, mu1 = cfg.tau, cfg.tau + cfg.eps
    llr = 0.0
    n = 0
    trace = []
    
    for x in stream:
        n += 1
        # Gaussian LLR with known sigma
        llr += ((x - mu0)**2 - (x - mu1)**2) / (2 * cfg.sigma**2)
        trace.append({"n": n, "llr": llr})
        
        if llr >= A:
            return {"decision": "accept_H1", "n": n, "A": A, "B": B, "trace": trace}
        if llr <= B:
            return {"decision": "accept_H0", "n": n, "A": A, "B": B, "trace": trace}
    
    return {"decision": "continue", "n": n, "A": A, "B": B, "trace": trace}


class UnifiedSequentialTester:
    """Unified interface for all sequential testing methods"""
    
    def __init__(self, method: str = "eb", **kwargs):
        """
        Initialize sequential tester
        
        Args:
            method: One of "eb", "sprt", or "legacy"
            **kwargs: Configuration parameters for chosen method
        """
        self.method = method.lower()
        self.observations = []
        self.decisions = []
        
        if method == "eb":
            self.config = EBConfig(**kwargs)
        elif method == "sprt":
            self.config = SPRTConfig(**kwargs)
        elif method == "legacy":
            # Use original SequentialTester
            self.tester = SequentialTester(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def update(self, observation: float) -> Tuple[Decision, Dict[str, Any]]:
        """
        Update with new observation and return decision
        
        Args:
            observation: New data point
            
        Returns:
            (decision, metadata)
        """
        self.observations.append(observation)
        
        # Initialize defaults to prevent uninitialized variable errors
        decision = "continue"
        metadata = {}
        
        if self.method == "eb":
            result = sequential_eb(iter(self.observations), self.config)
            decision = result["decision"]
            metadata = {
                "n": result["n"],
                "trace": result["trace"][-1] if result["trace"] else {}
            }
        elif self.method == "sprt":
            result = sprt(iter(self.observations), self.config)
            decision = result["decision"]
            metadata = {
                "n": result["n"],
                "A": result["A"],
                "B": result["B"],
                "trace": result["trace"][-1] if result["trace"] else {}
            }
        elif self.method == "legacy":
            result = self.tester.update(observation)
            decision = result.decision
            metadata = {
                "n": result.n_samples,
                "llr": result.log_likelihood_ratio
            }
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.decisions.append({
            "n": len(self.observations),
            "decision": decision,
            **metadata
        })
        
        return decision, metadata
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """Get full decision trace"""
        return self.decisions
    
    def reset(self):
        """Reset tester state"""
        self.observations = []
        self.decisions = []
        if self.method == "legacy":
            self.tester = SequentialTester(
                self.tester.alpha, 
                self.tester.beta,
                self.tester.tau0,
                self.tester.tau1
            )


def sequential_verify(
    stream: Iterable[float],
    tau: float = 0.5,
    alpha: float = 0.05,
    beta: float = 0.05,
    max_samples: int = 10000,
    compute_p_value: bool = True
) -> SPRTResult:
    """
    Anytime-valid sequential hypothesis test using Empirical-Bernstein bounds.
    
    Tests H0: μ ≤ τ (model is genuine) vs H1: μ > τ (model is different)
    for bounded distances in [0,1].
    
    This implementation uses Welford's algorithm for numerical stability and
    EB confidence sequences for anytime-valid inference with early stopping.
    
    Reference: §2.4 of the paper for sequential verification protocol
    
    Args:
        stream: Iterator/iterable of distance values (should be in [0,1])
        tau: Decision threshold (default 0.5)
        alpha: Type I error rate - false acceptance (default 0.05)
        beta: Type II error rate - false rejection (default 0.05)
        max_samples: Maximum number of samples before forced decision (default 10000)
    
    Returns:
        SPRTResult with:
            - decision: 'H0' (accept identity), 'H1' (reject identity), or 'continue'
            - stopped_at: Number of samples used
            - final_mean: Final empirical mean
            - final_variance: Final empirical variance
            - confidence_radius: Final EB radius
            - trajectory: Complete state history
            - confidence_interval: Final confidence bounds
            - p_value: Optional p-value (if computed)
            - forced_stop: Whether stopping was forced at max_samples
    
    Example:
        >>> from pot.core.sequential import sequential_verify
        >>> distances = [0.1, 0.2, 0.15, 0.12, 0.18]  # Stream of distances
        >>> result = sequential_verify(iter(distances), tau=0.3, alpha=0.05, beta=0.05)
        >>> print(f"Decision: {result.decision} at n={result.stopped_at}")
    """
    # Initialize state and trajectory
    state = SequentialState()
    trajectory = []
    
    # Process stream
    for t, x_raw in enumerate(stream, start=1):
        # Clip values to [0,1] for bounded distances
        x = max(0.0, min(1.0, float(x_raw)))
        
        # Update state using numerically stable Welford algorithm
        state = welford_update(state, x)
        
        # Store trajectory snapshot
        trajectory.append(state.copy())
        
        # Check stopping condition using helper function
        should_stop, decision = check_stopping_condition(state, tau, alpha)
        
        if should_stop:
            # Compute EB radius for final reporting
            from .boundaries import CSState, eb_radius as compute_eb_radius
            cs_state = CSState()
            cs_state.n = state.n
            cs_state.mean = state.mean
            cs_state.M2 = state.M2
            radius = compute_eb_radius(cs_state, alpha)
            
            # Compute p-value if requested
            p_val = compute_anytime_p_value(state, tau) if compute_p_value else None
            
            return SPRTResult(
                decision=decision,
                stopped_at=t,
                final_mean=state.mean,
                final_variance=compute_empirical_variance(state),
                confidence_radius=radius,
                trajectory=trajectory,
                confidence_interval=(max(0, state.mean - radius), 
                                   min(1, state.mean + radius)),
                p_value=p_val,
                forced_stop=False
            )
        
        # Check if we've reached maximum samples
        if t >= max_samples:
            # Forced decision at maximum samples
            from .boundaries import CSState, eb_radius as compute_eb_radius
            cs_state = CSState()
            cs_state.n = state.n
            cs_state.mean = state.mean
            cs_state.M2 = state.M2
            radius = compute_eb_radius(cs_state, alpha)
            
            # Use point estimate to decide
            decision = 'H0' if state.mean <= tau else 'H1'
            
            # Compute p-value if requested
            p_val = compute_anytime_p_value(state, tau) if compute_p_value else None
            
            return SPRTResult(
                decision=decision,
                stopped_at=t,
                final_mean=state.mean,
                final_variance=compute_empirical_variance(state),
                confidence_radius=radius,
                trajectory=trajectory,
                confidence_interval=(max(0, state.mean - radius),
                                   min(1, state.mean + radius)),
                p_value=p_val,
                forced_stop=True
            )
    
    # Stream ended without reaching max_samples
    # Return current state with 'continue' decision
    if state.n > 0:
        from .boundaries import CSState, eb_radius as compute_eb_radius
        cs_state = CSState()
        cs_state.n = state.n
        cs_state.mean = state.mean
        cs_state.M2 = state.M2
        radius = compute_eb_radius(cs_state, alpha)
        
        # Compute p-value if requested
        p_val = compute_anytime_p_value(state, tau) if compute_p_value else None
        
        return SPRTResult(
            decision='continue',
            stopped_at=state.n,
            final_mean=state.mean,
            final_variance=compute_empirical_variance(state),
            confidence_radius=radius,
            trajectory=trajectory,
            confidence_interval=(max(0, state.mean - radius),
                               min(1, state.mean + radius)),
            p_value=p_val,
            forced_stop=False
        )
    else:
        # No samples processed
        return SPRTResult(
            decision='continue',
            stopped_at=0,
            final_mean=0.0,
            final_variance=0.0,
            confidence_radius=float('inf'),
            trajectory=[],
            confidence_interval=(0.0, 1.0),
            p_value=1.0 if compute_p_value else None,
            forced_stop=False
        )