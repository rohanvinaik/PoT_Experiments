from typing import Iterable, Tuple, Literal, Dict, Any, List, Optional
import numpy as np
from math import log, sqrt
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
            return SPRTResult(self._decision, self.S, self.n, self.distances)
        
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
        
        return SPRTResult(self._decision, self.S, self.n, self.distances)
    
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
    max_samples: int = 10000
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
        
        # Update state with Welford's algorithm
        state.update(x)
        
        # Store trajectory snapshot
        trajectory.append(state.copy())
        
        # Compute EB radius for current state
        # Use CSState for compatibility with boundaries.eb_radius
        cs_state = CSState()
        cs_state.n = state.n
        cs_state.mean = state.mean
        cs_state.M2 = state.M2
        
        # Compute confidence radius
        radius = compute_eb_radius(cs_state, alpha)
        
        # Check stopping conditions based on EB bounds
        # Condition 1: Accept H0 if upper bound ≤ τ (strong evidence for H0)
        if state.mean + radius <= tau:
            return SPRTResult(
                decision='H0',
                stopped_at=t,
                final_mean=state.mean,
                final_variance=state.variance,
                confidence_radius=radius,
                trajectory=trajectory,
                confidence_interval=(max(0, state.mean - radius), 
                                   min(1, state.mean + radius)),
                p_value=None,  # Could compute if needed
                forced_stop=False
            )
        
        # Condition 2: Accept H1 if lower bound > τ (strong evidence against H0)
        if state.mean - radius > tau:
            return SPRTResult(
                decision='H1',
                stopped_at=t,
                final_mean=state.mean,
                final_variance=state.variance,
                confidence_radius=radius,
                trajectory=trajectory,
                confidence_interval=(max(0, state.mean - radius),
                                   min(1, state.mean + radius)),
                p_value=None,
                forced_stop=False
            )
        
        # Check if we've reached maximum samples
        if t >= max_samples:
            # Forced decision at maximum samples
            # Use point estimate to decide
            decision = 'H0' if state.mean <= tau else 'H1'
            return SPRTResult(
                decision=decision,
                stopped_at=t,
                final_mean=state.mean,
                final_variance=state.variance,
                confidence_radius=radius,
                trajectory=trajectory,
                confidence_interval=(max(0, state.mean - radius),
                                   min(1, state.mean + radius)),
                p_value=None,
                forced_stop=True
            )
    
    # Stream ended without reaching max_samples
    # Return current state with 'continue' decision
    if state.n > 0:
        cs_state = CSState()
        cs_state.n = state.n
        cs_state.mean = state.mean
        cs_state.M2 = state.M2
        radius = compute_eb_radius(cs_state, alpha)
        
        return SPRTResult(
            decision='continue',
            stopped_at=state.n,
            final_mean=state.mean,
            final_variance=state.variance,
            confidence_radius=radius,
            trajectory=trajectory,
            confidence_interval=(max(0, state.mean - radius),
                               min(1, state.mean + radius)),
            p_value=None,
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
            p_value=None,
            forced_stop=False
        )