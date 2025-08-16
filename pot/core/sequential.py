from typing import Iterable, Callable, Tuple, Optional, Literal, Dict, Any, List
import numpy as np
from math import log, sqrt
from dataclasses import dataclass

Decision = Literal["continue", "accept_H0", "accept_H1"]

@dataclass
class SPRTResult:
    """Result of SPRT sequential test"""
    decision: str  # 'accept_H0', 'accept_H1', or 'continue'
    log_likelihood_ratio: float
    n_samples: int
    distances: list
    
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
    tau: float,
    alpha: float,
    beta: float,
    n_max: int
) -> Tuple[Dict[str, Any], List[Tuple[int, float, float, float]]]:
    """
    Sequential verification orchestrator with optional stopping.
    
    Uses confidence sequences with separate radii for accept (alpha) and reject (beta) decisions.
    Implements anytime-valid inference with early stopping.
    
    Args:
        stream: Iterator/iterable of Z values (similarity scores)
        tau: Decision threshold
        alpha: Type I error rate (false acceptance)
        beta: Type II error rate (false rejection)
        n_max: Maximum number of samples before forced decision
    
    Returns:
        Tuple of:
        - Decision dict with keys:
            - 'type': 'accept_id', 'reject_id', or 'conservative_reject'
            - 'stopping_time': Number of samples used
            - 'final_mean': Final empirical mean
            - 'final_radius_alpha': Final radius for alpha
            - 'final_radius_beta': Final radius for beta
        - Trail: List of (t, mean, r_alpha, r_beta) tuples for visualization
    """
    state = CSState()
    trail = []
    
    for t, z_raw in enumerate(stream, start=1):
        # Clip Z values to [0,1]
        z = max(0.0, min(1.0, float(z_raw)))
        
        # Update state with new observation
        state.update(z)
        
        # Compute radii for both error rates
        # Alpha radius for acceptance (tighter bound)
        r_alpha = compute_eb_radius(state, alpha)
        # Beta radius for rejection (may be different)
        r_beta = compute_eb_radius(state, beta)
        
        # Record trail for visualization
        trail.append((t, state.mean, r_alpha, r_beta))
        
        # Early stopping decisions
        # Accept if upper bound with alpha is below threshold
        if state.mean + r_alpha <= tau:
            decision = {
                'type': 'accept_id',
                'stopping_time': t,
                'final_mean': state.mean,
                'final_radius_alpha': r_alpha,
                'final_radius_beta': r_beta,
                'confidence_interval': (max(0, state.mean - r_alpha), 
                                       min(1, state.mean + r_alpha))
            }
            return decision, trail
        
        # Reject if lower bound with beta is above threshold
        if state.mean - r_beta > tau:
            decision = {
                'type': 'reject_id',
                'stopping_time': t,
                'final_mean': state.mean,
                'final_radius_alpha': r_alpha,
                'final_radius_beta': r_beta,
                'confidence_interval': (max(0, state.mean - r_beta),
                                       min(1, state.mean + r_beta))
            }
            return decision, trail
        
        # Check if we've reached maximum samples
        if t >= n_max:
            # Conservative rejection at max samples
            # Decision based on which hypothesis the mean is closer to
            decision_type = 'accept_id' if state.mean <= tau else 'conservative_reject'
            decision = {
                'type': decision_type,
                'stopping_time': t,
                'final_mean': state.mean,
                'final_radius_alpha': r_alpha,
                'final_radius_beta': r_beta,
                'confidence_interval': (max(0, state.mean - r_alpha),
                                       min(1, state.mean + r_alpha)),
                'forced_stop': True
            }
            return decision, trail
    
    # If stream ends before n_max (shouldn't happen with proper usage)
    # Make conservative decision based on current state
    t = state.n
    if t > 0:
        r_alpha = compute_eb_radius(state, alpha)
        r_beta = compute_eb_radius(state, beta)
        decision_type = 'accept_id' if state.mean <= tau else 'conservative_reject'
        decision = {
            'type': decision_type,
            'stopping_time': t,
            'final_mean': state.mean,
            'final_radius_alpha': r_alpha,
            'final_radius_beta': r_beta,
            'confidence_interval': (max(0, state.mean - r_alpha),
                                   min(1, state.mean + r_alpha)),
            'stream_ended': True
        }
        return decision, trail
    else:
        # No samples processed
        decision = {
            'type': 'conservative_reject',
            'stopping_time': 0,
            'final_mean': 0.0,
            'final_radius_alpha': float('inf'),
            'final_radius_beta': float('inf'),
            'no_samples': True
        }
        return decision, []