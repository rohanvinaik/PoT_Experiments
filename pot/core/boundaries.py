"""Anytime-valid confidence sequence boundaries for sequential decision making."""

import math
from typing import Literal, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CSState:
    """
    State for confidence sequence computation using Welford's online algorithm.
    
    Maintains running statistics for bounded values Z ∈ [0,1].
    Uses Welford's algorithm for numerically stable online mean/variance computation.
    """
    
    n: int = 0                    # Number of samples seen
    mean: float = 0.0             # Running mean
    M2: float = 0.0               # Sum of squared deviations (for variance)
    min_val: float = float('inf') # Minimum value seen
    max_val: float = float('-inf')# Maximum value seen
    sum_val: float = 0.0          # Sum of all values
    
    def update(self, z: float) -> None:
        """
        Update statistics with a new observation using Welford's algorithm.
        
        Args:
            z: New observation value (should be in [0,1])
        """
        if not (0.0 <= z <= 1.0):
            raise ValueError(f"Value {z} not in [0,1]")
        
        self.n += 1
        delta = z - self.mean
        self.mean += delta / self.n
        delta2 = z - self.mean
        self.M2 += delta * delta2
        
        # Update additional statistics
        self.min_val = min(self.min_val, z)
        self.max_val = max(self.max_val, z)
        self.sum_val += z
    
    @property
    def variance(self) -> float:
        """Compute sample variance."""
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)
    
    @property
    def std(self) -> float:
        """Compute sample standard deviation."""
        return math.sqrt(self.variance)
    
    @property
    def empirical_variance(self) -> float:
        """
        Compute empirical variance for confidence sequences.
        Uses unbiased estimator with Bessel's correction.
        """
        if self.n < 1:
            return 0.0
        # For bounded [0,1] values, we can use the sample variance
        # but cap it at 0.25 (maximum possible variance for [0,1])
        return min(self.variance, 0.25)
    
    def copy(self) -> 'CSState':
        """Create a copy of the current state."""
        return CSState(
            n=self.n,
            mean=self.mean,
            M2=self.M2,
            min_val=self.min_val,
            max_val=self.max_val,
            sum_val=self.sum_val
        )


def log_log_correction(t: int, alpha: float) -> float:
    """
    Implement the log(log(t)) correction factor for anytime validity.
    
    This correction ensures the confidence sequence remains valid
    at all stopping times (anytime-valid property).
    
    Reference: §2.4 of the paper
    
    Args:
        t: Current time/sample count (must be >= 1)
        alpha: Significance level (e.g., 0.05 for 95% confidence)
    
    Returns:
        The log-log correction factor: log(log(max(e, t)) / α)
    
    Note:
        - For small t (t < e ≈ 2.718), we use log(e) = 1 to avoid log(log(t)) < 0
        - This ensures the correction factor is always positive
    """
    if t < 1:
        raise ValueError(f"t must be >= 1, got {t}")
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"Alpha must be in (0,1), got {alpha}")
    
    # Use max(e, t) to ensure log(log(t)) >= 0
    # Since log(e) = 1, log(log(e)) = log(1) = 0
    e = math.e
    t_corrected = max(e, t)
    
    # Compute log(log(t) / α)
    log_log_term = math.log(math.log(t_corrected) / alpha)
    
    return log_log_term


def eb_radius(state: CSState, alpha: float, c: float = 1.0) -> float:
    """
    Compute Empirical-Bernstein confidence radius with log-log term.
    
    The EB radius formula from §2.4 of the paper:
    r_t(α) = sqrt(2 * σ²_t * log(log(t) / α) / t) + c * log(log(t) / α) / t
    
    Where:
    - σ²_t is the empirical variance estimate at time t
    - The log-log term ensures anytime validity
    - c is a constant (typically 1.0) for the bias/concentration term
    
    This provides anytime-valid confidence sequences that are always valid
    with probability at least 1-α, no matter when you stop.
    
    Args:
        state: Current CSState with running statistics
        alpha: Significance level (e.g., 0.05 for 95% confidence)
        c: Constant for bias term (default 1.0, can be tuned for tightness)
    
    Returns:
        Confidence radius for the empirical mean
        
    Reference:
        §2.4 of the paper for Empirical-Bernstein bounds
    """
    if state.n < 1:
        return float('inf')
    
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"Alpha must be in (0,1), got {alpha}")
    
    t = state.n
    sigma_squared_t = state.empirical_variance
    
    # Get log-log correction factor
    log_log_term = log_log_correction(t, alpha)
    
    # Compute radius components according to paper formula
    # First term: sqrt(2 * σ²_t * log(log(t) / α) / t)
    variance_term = math.sqrt(2 * sigma_squared_t * log_log_term / t)
    
    # Second term: c * log(log(t) / α) / t
    bias_term = c * log_log_term / t
    
    # Total radius
    radius = variance_term + bias_term
    
    return radius


def eb_confidence_interval(mean: float, variance: float, n: int, alpha: float, c: float = 1.0) -> Tuple[float, float]:
    """
    Return (lower, upper) confidence bounds using Empirical-Bernstein radius.
    
    This function computes anytime-valid confidence intervals for bounded
    random variables in [0, 1] using the Empirical-Bernstein inequality.
    
    Args:
        mean: Sample mean of observations
        variance: Sample variance (should be empirical variance for EB)
        n: Number of samples
        alpha: Significance level (e.g., 0.05 for 95% confidence)
        c: Constant for bias term in EB radius (default 1.0)
    
    Returns:
        (lower, upper): Tuple of confidence bounds, clipped to [0, 1]
        
    Reference:
        §2.4 of the paper for confidence interval construction
    """
    if n < 1:
        return (0.0, 1.0)  # Uninformative interval
    
    if not (0.0 <= mean <= 1.0):
        raise ValueError(f"Mean must be in [0,1], got {mean}")
    if variance < 0:
        raise ValueError(f"Variance must be non-negative, got {variance}")
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"Alpha must be in (0,1), got {alpha}")
    
    # Create a temporary state for radius computation
    temp_state = CSState()
    temp_state.n = n
    temp_state.mean = mean
    # Set M2 such that variance = M2 / (n - 1) matches input variance
    # But cap variance at 0.25 (max for [0,1] bounded variables)
    capped_variance = min(variance, 0.25)
    if n > 1:
        temp_state.M2 = capped_variance * (n - 1)
    else:
        temp_state.M2 = 0.0
    
    # Compute EB radius
    radius = eb_radius(temp_state, alpha, c)
    
    # Compute confidence bounds
    lower = mean - radius
    upper = mean + radius
    
    # Clip to [0, 1] since we know values are bounded
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    
    return (lower, upper)


def decide_one_sided(
    state: CSState,
    threshold: float,
    alpha: float,
    hypothesis: Literal["H0", "H1"] = "H0"
) -> Literal["accept_id", "reject_id", "continue"]:
    """
    Make sequential decision based on confidence sequence.
    
    For one-sided testing:
    - H0: μ ≤ threshold (null hypothesis: identity/genuine model)
    - H1: μ > threshold (alternative: different/adversarial model)
    
    Args:
        state: Current CSState with observations
        threshold: Decision threshold (boundary value)
        alpha: Significance level for confidence sequence
        hypothesis: Which hypothesis we're testing ("H0" or "H1")
    
    Returns:
        "accept_id": Accept identity (H0 true, model is genuine)
        "reject_id": Reject identity (H1 true, model is adversarial)
        "continue": Need more samples to decide
    """
    if state.n < 1:
        return "continue"
    
    # Get confidence radius
    radius = eb_radius(state, alpha)
    
    # Compute confidence bounds
    lower_bound = state.mean - radius
    upper_bound = state.mean + radius
    
    # Ensure bounds respect [0,1] constraint
    lower_bound = max(0.0, lower_bound)
    upper_bound = min(1.0, upper_bound)
    
    if hypothesis == "H0":
        # Testing H0: μ ≤ threshold
        # Reject H0 if lower bound > threshold (entire CI above threshold)
        if lower_bound > threshold:
            return "reject_id"  # Strong evidence against H0
        # Accept H0 if upper bound ≤ threshold (entire CI below/at threshold)
        elif upper_bound <= threshold:
            return "accept_id"  # Strong evidence for H0
        else:
            return "continue"  # CI contains threshold, need more data
    
    elif hypothesis == "H1":
        # Testing H1: μ > threshold
        # Accept H1 (reject H0) if lower bound > threshold
        if lower_bound > threshold:
            return "reject_id"  # Strong evidence for H1
        # Reject H1 (accept H0) if upper bound ≤ threshold
        elif upper_bound <= threshold:
            return "accept_id"  # Strong evidence against H1
        else:
            return "continue"  # CI contains threshold, need more data
    
    else:
        raise ValueError(f"hypothesis must be 'H0' or 'H1', got {hypothesis}")


def decide_two_sided(
    state_h0: CSState,
    state_h1: CSState,
    alpha: float
) -> Tuple[Literal["accept_id", "reject_id", "continue"], dict]:
    """
    Make sequential decision for two-sided testing using two confidence sequences.
    
    This implements SPRT-like logic with confidence sequences for comparing
    two hypotheses directly.
    
    Args:
        state_h0: CSState for samples under H0 (genuine model)
        state_h1: CSState for samples under H1 (adversarial model)
        alpha: Significance level for each confidence sequence
    
    Returns:
        Decision and additional info dict
    """
    if state_h0.n < 1 or state_h1.n < 1:
        return "continue", {"reason": "insufficient_samples"}
    
    # Get confidence bounds for both hypotheses
    radius_h0 = eb_radius(state_h0, alpha / 2)  # Bonferroni correction
    radius_h1 = eb_radius(state_h1, alpha / 2)
    
    # Confidence intervals
    h0_lower = max(0.0, state_h0.mean - radius_h0)
    h0_upper = min(1.0, state_h0.mean + radius_h0)
    h1_lower = max(0.0, state_h1.mean - radius_h1)
    h1_upper = min(1.0, state_h1.mean + radius_h1)
    
    info = {
        "h0_ci": (h0_lower, h0_upper),
        "h1_ci": (h1_lower, h1_upper),
        "h0_mean": state_h0.mean,
        "h1_mean": state_h1.mean,
        "h0_n": state_h0.n,
        "h1_n": state_h1.n
    }
    
    # Decision logic: check if CIs are separated
    if h0_upper < h1_lower:
        # H0 values significantly lower than H1
        return "accept_id", {**info, "reason": "h0_lower"}
    elif h1_upper < h0_lower:
        # H1 values significantly lower than H0
        return "reject_id", {**info, "reason": "h1_lower"}
    else:
        # CIs overlap, need more data
        return "continue", {**info, "reason": "overlap"}


def adaptive_threshold(
    state: CSState,
    target_far: float,
    target_frr: float,
    alpha: float
) -> float:
    """
    Compute adaptive threshold based on current observations.
    
    This adjusts the decision threshold based on the empirical distribution
    to achieve target false acceptance/rejection rates.
    
    Args:
        state: Current CSState
        target_far: Target false acceptance rate
        target_frr: Target false rejection rate
        alpha: Confidence level
    
    Returns:
        Adaptive threshold value
    """
    if state.n < 10:  # Need minimum samples
        # Default threshold
        return 0.5
    
    # Use empirical quantiles with confidence adjustment
    radius = eb_radius(state, alpha)
    
    # Adjust threshold based on target error rates
    # This is a simplified version; full implementation would use
    # empirical CDF estimation
    base_threshold = state.mean
    
    # Adjust for asymmetric error costs
    if target_far < target_frr:
        # More conservative (higher threshold)
        threshold = base_threshold + radius * (1 - target_far)
    else:
        # More permissive (lower threshold)
        threshold = base_threshold - radius * target_frr
    
    return max(0.0, min(1.0, threshold))


class SequentialTest:
    """
    Wrapper class for sequential testing with confidence sequences.
    
    Maintains state and provides high-level interface for sequential decisions.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        alpha: float = 0.05,
        max_samples: Optional[int] = None,
        hypothesis: Literal["H0", "H1"] = "H0"
    ):
        """
        Initialize sequential test.
        
        Args:
            threshold: Decision threshold
            alpha: Significance level
            max_samples: Maximum samples before forced decision
            hypothesis: Which hypothesis to test
        """
        self.threshold = threshold
        self.alpha = alpha
        self.max_samples = max_samples
        self.hypothesis = hypothesis
        self.state = CSState()
        self.decision_history = []
    
    def update(self, z: float) -> Literal["accept_id", "reject_id", "continue"]:
        """
        Update with new observation and return decision.
        
        Args:
            z: New observation in [0,1]
        
        Returns:
            Current decision
        """
        self.state.update(z)
        
        # Check if we've reached max samples
        if self.max_samples and self.state.n >= self.max_samples:
            # Forced decision based on mean
            if self.state.mean <= self.threshold:
                decision = "accept_id"
            else:
                decision = "reject_id"
        else:
            decision = decide_one_sided(
                self.state, self.threshold, self.alpha, self.hypothesis
            )
        
        self.decision_history.append({
            'n': self.state.n,
            'mean': self.state.mean,
            'radius': eb_radius(self.state, self.alpha),
            'decision': decision
        })
        
        return decision
    
    def get_confidence_interval(self) -> Tuple[float, float]:
        """Get current confidence interval."""
        if self.state.n < 1:
            return (0.0, 1.0)
        
        radius = eb_radius(self.state, self.alpha)
        lower = max(0.0, self.state.mean - radius)
        upper = min(1.0, self.state.mean + radius)
        return (lower, upper)
    
    def reset(self):
        """Reset test state."""
        self.state = CSState()
        self.decision_history = []