"""
Sequential Testing Implementation for Language Model Verification
Implements Sequential Probability Ratio Testing (SPRT) for efficient verification with early stopping
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import math
from collections import deque
import json
import time


@dataclass
class SPRTState:
    """State of SPRT test"""
    log_likelihood_ratio: float = 0.0
    num_trials: int = 0
    num_successes: int = 0
    terminated: bool = False
    decision: Optional[str] = None
    confidence: float = 0.5
    history: List[bool] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)


class SequentialTester:
    """
    Sequential Probability Ratio Testing (SPRT) for efficient model verification.
    Implements Wald's sequential test with early stopping capabilities.
    """
    
    def __init__(self, 
                 alpha: float = 0.05,  # Type I error rate (false positive)
                 beta: float = 0.05,   # Type II error rate (false negative)
                 p0: float = 0.5,      # Null hypothesis success rate (H0: model is fake/modified)
                 p1: float = 0.8,      # Alternative hypothesis success rate (H1: model is genuine)
                 max_trials: int = 1000,  # Maximum number of trials
                 min_trials: int = 5):    # Minimum trials before decision
        """
        Initialize sequential tester with SPRT parameters.
        
        Args:
            alpha: Type I error rate (rejecting genuine model)
            beta: Type II error rate (accepting fake model)
            p0: Success rate under null hypothesis
            p1: Success rate under alternative hypothesis
            max_trials: Maximum number of trials before forced decision
            min_trials: Minimum trials required before making decision
        """
        # Validate parameters
        if not (0 < alpha < 1):
            raise ValueError(f"Alpha must be in (0, 1), got {alpha}")
        if not (0 < beta < 1):
            raise ValueError(f"Beta must be in (0, 1), got {beta}")
        if not (0 < p0 < 1):
            raise ValueError(f"p0 must be in (0, 1), got {p0}")
        if not (0 < p1 < 1):
            raise ValueError(f"p1 must be in (0, 1), got {p1}")
        if p0 >= p1:
            raise ValueError(f"p1 ({p1}) must be greater than p0 ({p0})")
        
        self.alpha = alpha
        self.beta = beta
        self.p0 = p0
        self.p1 = p1
        self.max_trials = max_trials
        self.min_trials = min_trials
        
        # Compute SPRT boundaries
        self.A = (1 - beta) / alpha  # Upper boundary
        self.B = beta / (1 - alpha)  # Lower boundary
        
        self.log_A = np.log(self.A)
        self.log_B = np.log(self.B)
        
        # Precompute log likelihood ratios for efficiency
        self.log_lr_success = np.log(p1 / p0)
        self.log_lr_failure = np.log((1 - p1) / (1 - p0))
        
        # Initialize state
        self.state = SPRTState()
        
        # Statistics tracking
        self.stats = {
            'total_tests': 0,
            'total_decisions': 0,
            'early_stops': 0,
            'forced_decisions': 0
        }
    
    def reset(self):
        """Reset test state for new verification."""
        self.state = SPRTState()
    
    def update(self, success: bool, timestamp: Optional[float] = None) -> Optional[str]:
        """
        Update test with new trial result.
        
        Args:
            success: Whether the trial was successful
            timestamp: Optional timestamp for the trial
            
        Returns:
            'accept' if H0 accepted (model is fake/modified)
            'reject' if H0 rejected (model is genuine)
            None if test should continue
        """
        if self.state.terminated:
            return self.state.decision
        
        # Record trial
        self.state.num_trials += 1
        if success:
            self.state.num_successes += 1
        
        self.state.history.append(success)
        self.state.timestamps.append(timestamp or time.time())
        
        # Update log likelihood ratio
        if success:
            self.state.log_likelihood_ratio += self.log_lr_success
        else:
            self.state.log_likelihood_ratio += self.log_lr_failure
        
        # Update confidence
        self.state.confidence = self.get_confidence()
        
        # Check if we have enough trials
        if self.state.num_trials < self.min_trials:
            return None
        
        # Check boundaries
        if self.state.log_likelihood_ratio >= self.log_A:
            # Reject H0, accept H1 (model is genuine)
            self.state.terminated = True
            self.state.decision = 'reject'
            self.stats['total_decisions'] += 1
            self.stats['early_stops'] += 1
            return 'reject'
            
        elif self.state.log_likelihood_ratio <= self.log_B:
            # Accept H0 (model is fake/modified)
            self.state.terminated = True
            self.state.decision = 'accept'
            self.stats['total_decisions'] += 1
            self.stats['early_stops'] += 1
            return 'accept'
        
        # Check max trials
        if self.state.num_trials >= self.max_trials:
            # Forced decision based on current likelihood
            self.state.terminated = True
            if self.state.log_likelihood_ratio > 0:
                self.state.decision = 'reject'
            else:
                self.state.decision = 'accept'
            self.stats['total_decisions'] += 1
            self.stats['forced_decisions'] += 1
            return self.state.decision
        
        return None
    
    def batch_update(self, results: List[bool]) -> Optional[str]:
        """
        Update with multiple trial results at once.
        
        Args:
            results: List of trial success/failure results
            
        Returns:
            Final decision if terminated, None otherwise
        """
        decision = None
        for result in results:
            decision = self.update(result)
            if decision is not None:
                break
        return decision
    
    def get_confidence(self) -> float:
        """
        Get current confidence level in the decision.
        
        Returns:
            Confidence score between 0 and 1
        """
        if self.state.terminated:
            # High confidence in decision
            if self.state.decision == 'reject':
                return min(1.0, 0.95 + 0.05 * (self.state.num_trials / self.min_trials))
            else:
                return max(0.0, 0.05 - 0.05 * (self.state.num_trials / self.min_trials))
        
        # Map log likelihood ratio to confidence
        # Positive LLR -> confidence towards H1 (genuine)
        # Negative LLR -> confidence towards H0 (fake)
        
        if self.state.num_trials == 0:
            return 0.5
        
        # Normalize to [0, 1] range
        range_val = self.log_A - self.log_B
        if range_val > 0:
            normalized = (self.state.log_likelihood_ratio - self.log_B) / range_val
            return np.clip(normalized, 0.0, 1.0)
        
        return 0.5
    
    def expected_sample_size(self, true_p: float) -> float:
        """
        Compute expected sample size for given true success probability.
        
        Args:
            true_p: True success probability
            
        Returns:
            Expected number of samples needed for decision
        """
        # Handle edge cases
        if true_p <= 0 or true_p >= 1:
            return self.max_trials
        
        # Compute expected value of log likelihood ratio
        if abs(true_p - self.p0) < 1e-10:
            # Under H0
            numerator = self.alpha * self.log_A + (1 - self.alpha) * self.log_B
            denominator = true_p * np.log(self.p0 / self.p1) + \
                         (1 - true_p) * np.log((1 - self.p0) / (1 - self.p1))
        elif abs(true_p - self.p1) < 1e-10:
            # Under H1
            numerator = (1 - self.beta) * self.log_A + self.beta * self.log_B
            denominator = true_p * np.log(self.p1 / self.p0) + \
                         (1 - true_p) * np.log((1 - self.p1) / (1 - self.p0))
        else:
            # General case
            if true_p > (self.p0 + self.p1) / 2:
                # Closer to H1
                prob_reject = self._compute_power(true_p)
                numerator = prob_reject * self.log_A + (1 - prob_reject) * self.log_B
            else:
                # Closer to H0
                prob_accept = 1 - self._compute_power(true_p)
                numerator = prob_accept * self.log_B + (1 - prob_accept) * self.log_A
            
            denominator = true_p * self.log_lr_success + (1 - true_p) * self.log_lr_failure
        
        if abs(denominator) < 1e-10:
            return self.max_trials
        
        expected_n = abs(numerator / denominator)
        
        # Apply bounds
        return max(self.min_trials, min(expected_n, self.max_trials))
    
    def _compute_power(self, true_p: float) -> float:
        """
        Compute statistical power for given true probability.
        
        Args:
            true_p: True success probability
            
        Returns:
            Power (probability of rejecting H0)
        """
        if true_p <= self.p0:
            return self.alpha
        elif true_p >= self.p1:
            return 1 - self.beta
        else:
            # Interpolate
            t = (true_p - self.p0) / (self.p1 - self.p0)
            return self.alpha + t * (1 - self.beta - self.alpha)
    
    def get_state(self) -> SPRTState:
        """Get current test state."""
        return self.state
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get test statistics.
        
        Returns:
            Dictionary of statistics
        """
        current_rate = (self.state.num_successes / self.state.num_trials 
                       if self.state.num_trials > 0 else 0.0)
        
        return {
            'num_trials': self.state.num_trials,
            'num_successes': self.state.num_successes,
            'success_rate': current_rate,
            'log_likelihood_ratio': self.state.log_likelihood_ratio,
            'confidence': self.state.confidence,
            'terminated': self.state.terminated,
            'decision': self.state.decision,
            'expected_trials_h0': self.expected_sample_size(self.p0),
            'expected_trials_h1': self.expected_sample_size(self.p1),
            'boundaries': {
                'upper': self.log_A,
                'lower': self.log_B,
                'current': self.state.log_likelihood_ratio
            },
            'global_stats': self.stats
        }
    
    def plot_progress(self) -> Dict[str, List]:
        """
        Get data for plotting test progress.
        
        Returns:
            Dictionary with plot data
        """
        # Compute cumulative log likelihood ratio
        cumulative_llr = []
        llr = 0
        for success in self.state.history:
            llr += self.log_lr_success if success else self.log_lr_failure
            cumulative_llr.append(llr)
        
        return {
            'trials': list(range(1, self.state.num_trials + 1)),
            'cumulative_llr': cumulative_llr,
            'upper_boundary': [self.log_A] * self.state.num_trials,
            'lower_boundary': [self.log_B] * self.state.num_trials,
            'successes': self.state.history
        }


class AdaptiveSequentialTester(SequentialTester):
    """
    Adaptive SPRT that adjusts parameters based on observed data.
    """
    
    def __init__(self, 
                 initial_alpha: float = 0.05,
                 initial_beta: float = 0.05,
                 adaptation_rate: float = 0.1,
                 window_size: int = 20,
                 **kwargs):
        """
        Initialize adaptive sequential tester.
        
        Args:
            initial_alpha: Initial Type I error rate
            initial_beta: Initial Type II error rate
            adaptation_rate: Rate of parameter adaptation
            window_size: Window for computing running statistics
            **kwargs: Additional arguments for SequentialTester
        """
        super().__init__(alpha=initial_alpha, beta=initial_beta, **kwargs)
        
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        
        # Running statistics
        self.recent_results = deque(maxlen=window_size)
        self.adaptation_history = []
    
    def update(self, success: bool, timestamp: Optional[float] = None) -> Optional[str]:
        """
        Update with adaptation.
        
        Args:
            success: Trial success
            timestamp: Optional timestamp
            
        Returns:
            Decision if terminated, None otherwise
        """
        # Record result
        self.recent_results.append(success)
        
        # Adapt parameters if we have enough data
        if len(self.recent_results) >= self.window_size // 2:
            self._adapt_parameters()
        
        # Call parent update
        return super().update(success, timestamp)
    
    def _adapt_parameters(self):
        """Adapt test parameters based on recent performance."""
        if not self.recent_results:
            return
        
        # Compute recent success rate
        recent_rate = sum(self.recent_results) / len(self.recent_results)
        
        # Estimate variance
        variance = recent_rate * (1 - recent_rate)
        
        # Adapt p1 based on observed rate
        if recent_rate > self.p1:
            # Model performing better than expected
            new_p1 = min(0.95, self.p1 + self.adaptation_rate * (recent_rate - self.p1))
            self.p1 = new_p1
            self.log_lr_success = np.log(self.p1 / self.p0)
            self.log_lr_failure = np.log((1 - self.p1) / (1 - self.p0))
        
        # Adapt error rates based on variance
        if variance < 0.1:
            # Low variance, can be more confident
            self.alpha = max(0.01, self.initial_alpha * (1 - self.adaptation_rate))
            self.beta = max(0.01, self.initial_beta * (1 - self.adaptation_rate))
        elif variance > 0.2:
            # High variance, be more conservative
            self.alpha = min(0.1, self.initial_alpha * (1 + self.adaptation_rate))
            self.beta = min(0.1, self.initial_beta * (1 + self.adaptation_rate))
        
        # Recompute boundaries
        self.A = (1 - self.beta) / self.alpha
        self.B = self.beta / (1 - self.alpha)
        self.log_A = np.log(self.A)
        self.log_B = np.log(self.B)
        
        # Record adaptation
        self.adaptation_history.append({
            'trial': self.state.num_trials,
            'recent_rate': recent_rate,
            'variance': variance,
            'alpha': self.alpha,
            'beta': self.beta,
            'p1': self.p1
        })


class GroupSequentialTester:
    """
    Group sequential testing with multiple stages and early stopping.
    """
    
    def __init__(self,
                 num_stages: int = 5,
                 trials_per_stage: int = 10,
                 alpha: float = 0.05,
                 beta: float = 0.05,
                 spending_function: str = 'obrien_fleming'):
        """
        Initialize group sequential tester.
        
        Args:
            num_stages: Number of testing stages
            trials_per_stage: Trials in each stage
            alpha: Overall Type I error rate
            beta: Overall Type II error rate
            spending_function: Alpha spending function ('obrien_fleming' or 'pocock')
        """
        self.num_stages = num_stages
        self.trials_per_stage = trials_per_stage
        self.alpha = alpha
        self.beta = beta
        self.spending_function = spending_function
        
        # Compute stage boundaries
        self.stage_boundaries = self._compute_boundaries()
        
        # Initialize state
        self.current_stage = 0
        self.stage_results = []
        self.cumulative_successes = 0
        self.cumulative_trials = 0
        self.terminated = False
        self.decision = None
    
    def _compute_boundaries(self) -> List[Tuple[float, float]]:
        """
        Compute stopping boundaries for each stage.
        
        Returns:
            List of (lower, upper) boundaries for each stage
        """
        boundaries = []
        
        for k in range(1, self.num_stages + 1):
            t = k / self.num_stages  # Information fraction
            
            if self.spending_function == 'obrien_fleming':
                # O'Brien-Fleming boundaries
                alpha_spent = 2 * (1 - self._normal_cdf(self._normal_inv(1 - self.alpha/2) / np.sqrt(t)))
                beta_spent = 2 * self._normal_cdf(self._normal_inv(self.beta/2) / np.sqrt(t))
            else:  # pocock
                # Pocock boundaries (constant)
                alpha_spent = self.alpha * t
                beta_spent = self.beta * t
            
            # Convert to z-scores
            z_upper = self._normal_inv(1 - alpha_spent / 2)
            z_lower = self._normal_inv(beta_spent / 2)
            
            boundaries.append((z_lower, z_upper))
        
        return boundaries
    
    def update_stage(self, stage_successes: int, stage_trials: int) -> Optional[str]:
        """
        Update with results from a complete stage.
        
        Args:
            stage_successes: Number of successes in stage
            stage_trials: Number of trials in stage
            
        Returns:
            Decision if terminated, None otherwise
        """
        if self.terminated:
            return self.decision
        
        # Update cumulative statistics
        self.cumulative_successes += stage_successes
        self.cumulative_trials += stage_trials
        self.current_stage += 1
        
        # Store stage results
        self.stage_results.append({
            'stage': self.current_stage,
            'successes': stage_successes,
            'trials': stage_trials,
            'cumulative_rate': self.cumulative_successes / self.cumulative_trials
        })
        
        # Compute test statistic
        p_hat = self.cumulative_successes / self.cumulative_trials
        p0 = 0.5  # Null hypothesis
        
        z_score = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / self.cumulative_trials)
        
        # Check boundaries
        lower, upper = self.stage_boundaries[self.current_stage - 1]
        
        if z_score >= upper:
            # Reject H0 (model is genuine)
            self.terminated = True
            self.decision = 'reject'
            return 'reject'
        elif z_score <= lower:
            # Accept H0 (model is fake)
            self.terminated = True
            self.decision = 'accept'
            return 'accept'
        
        # Check if final stage
        if self.current_stage >= self.num_stages:
            self.terminated = True
            self.decision = 'reject' if z_score > 0 else 'accept'
            return self.decision
        
        return None
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _normal_inv(self, p: float) -> float:
        """Inverse normal CDF (approximate)."""
        # Simplified inverse normal for demonstration
        # In practice, use scipy.stats.norm.ppf
        if p <= 0:
            return -10
        elif p >= 1:
            return 10
        else:
            # Approximation
            return math.sqrt(2) * math.sin(math.pi * (p - 0.5))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get test statistics."""
        return {
            'current_stage': self.current_stage,
            'cumulative_successes': self.cumulative_successes,
            'cumulative_trials': self.cumulative_trials,
            'success_rate': self.cumulative_successes / max(self.cumulative_trials, 1),
            'stage_results': self.stage_results,
            'terminated': self.terminated,
            'decision': self.decision,
            'boundaries': self.stage_boundaries
        }


# Integration with verification system
class SequentialVerificationSession:
    """
    Complete sequential verification session with challenge generation.
    """
    
    def __init__(self,
                 tester: Optional[SequentialTester] = None,
                 challenger: Optional[Any] = None,
                 evaluator: Optional[Any] = None,
                 model_runner: Optional[Any] = None):
        """
        Initialize verification session.
        
        Args:
            tester: Sequential tester instance
            challenger: Challenge generator
            evaluator: Challenge evaluator
            model_runner: Function to run model
        """
        self.tester = tester or SequentialTester()
        self.challenger = challenger
        self.evaluator = evaluator
        self.model_runner = model_runner
        
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def run_verification(self, 
                        max_challenges: int = 100,
                        early_stop: bool = True) -> Dict[str, Any]:
        """
        Run complete verification session.
        
        Args:
            max_challenges: Maximum number of challenges
            early_stop: Whether to stop early on decision
            
        Returns:
            Verification results
        """
        self.start_time = time.time()
        self.tester.reset()
        decision = None
        
        for i in range(max_challenges):
            # Generate challenge
            if self.challenger:
                challenges = self.challenger.generate_challenge_set(1)
                challenge = challenges[0] if challenges else None
            else:
                challenge = {'prompt': f'Test {i}', 'expected': 'test'}
            
            if not challenge:
                continue
            
            # Get model response
            if self.model_runner:
                response = self.model_runner(challenge['prompt'])
            else:
                # Simulate response
                response = challenge['expected'] if np.random.random() > 0.3 else 'wrong'
            
            # Evaluate response
            if self.evaluator:
                eval_result = self.evaluator.evaluate_response(response, challenge)
                success = eval_result.success
            else:
                success = response == challenge['expected']
            
            # Store result
            self.results.append({
                'trial': i + 1,
                'challenge': challenge,
                'response': response,
                'success': success,
                'confidence': self.tester.get_confidence()
            })
            
            # Update tester
            decision = self.tester.update(success)
            
            if decision is not None and early_stop:
                break
            
            # Update challenger if adaptive
            if hasattr(self.challenger, 'add_to_history'):
                try:
                    # Import ChallengeResult if available
                    from pot.lm.template_challenges import ChallengeResult
                    challenge_result = ChallengeResult(
                        success=success,
                        score=1.0 if success else 0.0,
                        match_type='simulation',
                        response=str(response),
                        expected=challenge.get('expected', ''),
                        difficulty=challenge.get('difficulty', 1)
                    )
                    self.challenger.add_to_history(challenge_result)
                except (ImportError, TypeError):
                    # Fallback for compatibility
                    pass
        
        self.end_time = time.time()
        
        # Compile final results
        return {
            'verified': decision == 'reject' if decision else None,
            'decision': decision,
            'confidence': self.tester.get_confidence(),
            'num_trials': self.tester.state.num_trials,
            'success_rate': (self.tester.state.num_successes / 
                           max(self.tester.state.num_trials, 1)),
            'early_stopped': decision is not None and i < max_challenges - 1,
            'duration': self.end_time - self.start_time,
            'results': self.results,
            'statistics': self.tester.get_statistics()
        }


# Utility functions
def compute_operating_characteristics(alpha_range: np.ndarray = None,
                                     beta_range: np.ndarray = None,
                                     p0: float = 0.5,
                                     p1: float = 0.8) -> Dict[str, np.ndarray]:
    """
    Compute operating characteristics for different parameter settings.
    
    Args:
        alpha_range: Range of Type I error rates
        beta_range: Range of Type II error rates
        p0: Null hypothesis success rate
        p1: Alternative hypothesis success rate
        
    Returns:
        Dictionary of operating characteristics
    """
    if alpha_range is None:
        alpha_range = np.linspace(0.01, 0.1, 10)
    if beta_range is None:
        beta_range = np.linspace(0.01, 0.1, 10)
    
    results = {
        'alpha': [],
        'beta': [],
        'expected_n_h0': [],
        'expected_n_h1': [],
        'power': []
    }
    
    for alpha in alpha_range:
        for beta in beta_range:
            tester = SequentialTester(alpha=alpha, beta=beta, p0=p0, p1=p1)
            
            results['alpha'].append(alpha)
            results['beta'].append(beta)
            results['expected_n_h0'].append(tester.expected_sample_size(p0))
            results['expected_n_h1'].append(tester.expected_sample_size(p1))
            results['power'].append(1 - beta)
    
    return {k: np.array(v) for k, v in results.items()}


def simulate_sequential_test(true_p: float,
                            num_simulations: int = 1000,
                            alpha: float = 0.05,
                            beta: float = 0.05,
                            p0: float = 0.5,
                            p1: float = 0.8) -> Dict[str, Any]:
    """
    Simulate sequential test performance.
    
    Args:
        true_p: True success probability
        num_simulations: Number of simulations
        alpha: Type I error rate
        beta: Type II error rate
        p0: Null hypothesis rate
        p1: Alternative hypothesis rate
        
    Returns:
        Simulation results
    """
    decisions = []
    sample_sizes = []
    
    for _ in range(num_simulations):
        tester = SequentialTester(alpha=alpha, beta=beta, p0=p0, p1=p1)
        
        for n in range(1, 1000):
            success = np.random.random() < true_p
            decision = tester.update(success)
            
            if decision is not None:
                decisions.append(decision)
                sample_sizes.append(n)
                break
    
    # Compute statistics
    reject_rate = sum(d == 'reject' for d in decisions) / len(decisions)
    avg_sample_size = np.mean(sample_sizes)
    
    return {
        'true_p': true_p,
        'reject_rate': reject_rate,
        'accept_rate': 1 - reject_rate,
        'avg_sample_size': avg_sample_size,
        'std_sample_size': np.std(sample_sizes),
        'min_sample_size': min(sample_sizes),
        'max_sample_size': max(sample_sizes),
        'sample_sizes': sample_sizes,
        'decisions': decisions
    }