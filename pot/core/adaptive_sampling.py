"""
Adaptive Sampling for Improved Convergence in Statistical Identity Testing
Implements dynamic batch sizing, convergence detection, and strategy switching
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive sampling"""
    initial_batch_size: int = 8
    max_batch_size: int = 32
    min_batch_size: int = 4
    
    # Convergence parameters
    mean_stability_window: int = 10
    ci_stability_threshold: float = 0.01
    rme_improvement_threshold: float = 0.05
    
    # Strategy switching thresholds
    switch_threshold: float = 0.5  # Switch strategy after using 50% of budget
    near_zero_threshold: float = 0.05  # Consider mean "near zero" below this
    high_variance_threshold: float = 0.1
    
    # Adaptive threshold parameters
    noise_window: int = 20
    noise_margin_factor: float = 2.0
    max_adaptive_factor: float = 1.5


@dataclass
class ConvergenceMetrics:
    """Track convergence of the sequential test"""
    mean_stability_window: int = 10
    ci_stability_threshold: float = 0.01
    rme_improvement_threshold: float = 0.05
    
    mean_history: List[float] = field(default_factory=list)
    ci_width_history: List[float] = field(default_factory=list)
    rme_history: List[float] = field(default_factory=list)
    decision_history: List[str] = field(default_factory=list)
    
    def update(self, mean: float, ci_width: float, rme: float, decision: str = "UNDECIDED"):
        """Update convergence metrics"""
        self.mean_history.append(mean)
        self.ci_width_history.append(ci_width)
        self.rme_history.append(rme)
        self.decision_history.append(decision)
    
    def is_converging(self) -> Tuple[bool, str]:
        """Check if metrics are converging"""
        if len(self.mean_history) < self.mean_stability_window:
            return False, "Insufficient samples"
        
        # Check mean stability
        recent_means = self.mean_history[-self.mean_stability_window:]
        mean_std = np.std(recent_means)
        
        # Check CI width improvement
        recent_ci = self.ci_width_history[-self.mean_stability_window:]
        ci_improvement = (recent_ci[0] - recent_ci[-1]) / recent_ci[0] if recent_ci[0] > 0 else 0
        
        # Check RME improvement
        if len(self.rme_history) >= 2:
            rme_improvement = (self.rme_history[-2] - self.rme_history[-1]) / self.rme_history[-2] if self.rme_history[-2] > 0 else 0
        else:
            rme_improvement = 1.0
        
        # Convergence criteria
        if mean_std < self.ci_stability_threshold:
            if ci_improvement < self.rme_improvement_threshold:
                return True, "Mean stable, CI not improving significantly"
        
        if rme_improvement < self.rme_improvement_threshold and self.rme_history[-1] < 0.5:
            return True, "RME converged to acceptable level"
        
        # Check if we're stuck in UNDECIDED
        if len(self.decision_history) >= self.mean_stability_window:
            recent_decisions = self.decision_history[-self.mean_stability_window:]
            if all(d == "UNDECIDED" for d in recent_decisions):
                if ci_improvement < 0.01:  # Very little improvement
                    return True, "Stuck in UNDECIDED with no CI improvement"
        
        return False, f"Continuing: mean_std={mean_std:.4f}, ci_imp={ci_improvement:.2f}"
    
    def get_convergence_rate(self) -> float:
        """Estimate convergence rate from CI width history"""
        if len(self.ci_width_history) < 5:
            return 0.0
        
        # Fit exponential decay to CI widths
        x = np.arange(len(self.ci_width_history))
        y = np.array(self.ci_width_history)
        
        # Log transform for linear fit (avoiding zeros)
        y_log = np.log(y + 1e-10)
        
        try:
            slope, _ = np.polyfit(x, y_log, 1)
            return -slope  # Positive value indicates convergence
        except:
            return 0.0


class AdaptiveSequentialTester:
    """Enhanced tester with adaptive sampling strategies"""
    
    def __init__(self, base_tester, config: Optional[AdaptiveConfig] = None):
        self.base_tester = base_tester
        self.config = config or AdaptiveConfig()
        self.convergence = ConvergenceMetrics(
            mean_stability_window=self.config.mean_stability_window,
            ci_stability_threshold=self.config.ci_stability_threshold,
            rme_improvement_threshold=self.config.rme_improvement_threshold
        )
        self.batch_size_history = []
        self.current_batch_size = self.config.initial_batch_size
        self.strategy_switches = []
        
    def should_increase_batch_size(self) -> bool:
        """Determine if we should increase batch size for faster convergence"""
        if not hasattr(self.base_tester, 'n') or self.base_tester.n < self.base_tester.n_min * 2:
            return False
        
        # Check if we're making slow progress
        if len(self.convergence.rme_history) >= 5:
            recent_rme = self.convergence.rme_history[-5:]
            rme_improvement = abs(recent_rme[0] - recent_rme[-1])
            
            if rme_improvement < 0.02:  # Very slow improvement
                return True
        
        # Check convergence rate
        conv_rate = self.convergence.get_convergence_rate()
        if 0 < conv_rate < 0.01:  # Very slow convergence
            return True
        
        return False
    
    def should_decrease_batch_size(self) -> bool:
        """Determine if we should decrease batch size for more granular control"""
        if not hasattr(self.base_tester, 'n'):
            return False
            
        # Near decision boundary - need finer control
        if len(self.convergence.mean_history) >= 5:
            recent_mean = np.mean(self.convergence.mean_history[-5:])
            recent_ci_width = np.mean(self.convergence.ci_width_history[-5:])
            
            # Check if we're near a decision boundary
            gamma = getattr(self.base_tester, 'gamma', 0.01)
            delta_star = getattr(self.base_tester, 'delta_star', 0.1)
            
            # Near SAME boundary
            if abs(recent_mean) < gamma * 1.5 and recent_ci_width < gamma * 2:
                return True
            
            # Near DIFFERENT boundary  
            if abs(recent_mean) > delta_star * 0.8 and recent_ci_width < abs(recent_mean) * 0.3:
                return True
        
        return False
    
    def adapt_batch_size(self) -> int:
        """Dynamically adjust batch size based on convergence"""
        if self.should_increase_batch_size():
            # Increase batch size
            self.current_batch_size = min(
                int(self.current_batch_size * 1.5), 
                self.config.max_batch_size
            )
            logger.info(f"Increased batch size to {self.current_batch_size}")
        elif self.should_decrease_batch_size():
            # Decrease batch size for finer control
            self.current_batch_size = max(
                int(self.current_batch_size * 0.75),
                self.config.min_batch_size
            )
            logger.info(f"Decreased batch size to {self.current_batch_size}")
        
        self.batch_size_history.append(self.current_batch_size)
        return self.current_batch_size
    
    def should_switch_strategy(self) -> Tuple[bool, str]:
        """Determine if we should switch scoring strategy"""
        if not hasattr(self.base_tester, 'n'):
            return False, ""
            
        n = self.base_tester.n
        n_max = getattr(self.base_tester, 'n_max', 400)
        
        # If we've used significant budget without decision
        if n > n_max * self.config.switch_threshold:
            mean = getattr(self.base_tester, 'mean', 0)
            variance = getattr(self.base_tester, 'variance', 0)
            
            # Get current CI if available
            if hasattr(self.base_tester, 'empirical_bernstein_ci'):
                (ci_lo, ci_hi), hw = self.base_tester.empirical_bernstein_ci(
                    self.base_tester.differences if hasattr(self.base_tester, 'differences') else []
                )
            else:
                hw = float('inf')
            
            # Check if we're in a difficult zone
            if abs(mean) < self.config.near_zero_threshold:  # Near zero mean
                if hw > self.config.near_zero_threshold:  # Large uncertainty
                    self.strategy_switches.append(("symmetric_kl", n))
                    return True, "symmetric_kl"  # Switch to more sensitive metric
            
            # Check if variance is too high
            if variance > self.config.high_variance_threshold:
                self.strategy_switches.append(("increase_k", n))
                return True, "increase_k"  # Increase positions per prompt
            
            # Try variance reduction through increased sampling
            if len(self.convergence.decision_history) >= 10:
                if all(d == "UNDECIDED" for d in self.convergence.decision_history[-10:]):
                    self.strategy_switches.append(("variance_reduction", n))
                    return True, "variance_reduction"
        
        return False, ""
    
    def compute_adaptive_threshold(self, threshold_type: str = "delta_star") -> float:
        """Compute adaptive threshold based on observed distribution"""
        if not hasattr(self.base_tester, 'differences') or len(self.base_tester.differences) < self.config.noise_window:
            # Return original threshold if insufficient data
            return getattr(self.base_tester, threshold_type, 0.1)
        
        # Estimate noise level from recent scores
        recent_scores = self.base_tester.differences[-self.config.noise_window:]
        noise_level = np.std(recent_scores)
        
        # Get original threshold
        original_threshold = getattr(self.base_tester, threshold_type, 0.1)
        
        # Adaptive threshold: original + noise margin
        adaptive_threshold = original_threshold + self.config.noise_margin_factor * noise_level
        
        # Cap at maximum factor
        max_threshold = original_threshold * self.config.max_adaptive_factor
        
        return min(adaptive_threshold, max_threshold)
    
    def suggest_parameter_adjustment(self) -> Dict[str, Any]:
        """Suggest parameter adjustments based on current state"""
        suggestions = {}
        
        if not hasattr(self.base_tester, 'n'):
            return suggestions
        
        n = self.base_tester.n
        n_max = getattr(self.base_tester, 'n_max', 400)
        
        # If approaching n_max without decision
        if n > n_max * 0.8 and len(self.convergence.decision_history) > 0:
            if self.convergence.decision_history[-1] == "UNDECIDED":
                # Suggest relaxing thresholds slightly
                suggestions['gamma_adjustment'] = 1.2  # Increase gamma by 20%
                suggestions['delta_star_adjustment'] = 0.9  # Decrease delta_star by 10%
                suggestions['increase_n_max'] = True
                
                logger.info(f"Suggesting parameter adjustments at n={n}: {suggestions}")
        
        # If stuck with high variance
        if len(self.convergence.rme_history) >= 5:
            recent_rme = np.mean(self.convergence.rme_history[-5:])
            if recent_rme > 0.5:  # High relative margin of error
                suggestions['increase_k'] = True  # More positions per prompt
                suggestions['use_variance_reduction'] = True
        
        return suggestions
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about adaptive sampling"""
        diagnostics = {
            'current_batch_size': self.current_batch_size,
            'batch_size_history': self.batch_size_history,
            'convergence_rate': self.convergence.get_convergence_rate(),
            'strategy_switches': self.strategy_switches,
            'mean_stability': np.std(self.convergence.mean_history[-10:]) if len(self.convergence.mean_history) >= 10 else None,
            'ci_improvement': None,
            'suggested_adjustments': self.suggest_parameter_adjustment()
        }
        
        # Calculate CI improvement
        if len(self.convergence.ci_width_history) >= 10:
            recent_ci = self.convergence.ci_width_history[-10:]
            ci_improvement = (recent_ci[0] - recent_ci[-1]) / recent_ci[0] if recent_ci[0] > 0 else 0
            diagnostics['ci_improvement'] = ci_improvement
        
        return diagnostics


class VarianceReductionStrategy:
    """Strategies for reducing variance in difficult cases"""
    
    @staticmethod
    def stratified_sampling(prompts: List[Dict], n_strata: int = 4) -> List[Dict]:
        """Stratify prompts by difficulty/type for better coverage"""
        # Group prompts by family
        stratified = []
        families = {}
        
        for prompt in prompts:
            family = prompt.get('family', 'unknown')
            if family not in families:
                families[family] = []
            families[family].append(prompt)
        
        # Sample evenly from each stratum
        samples_per_stratum = len(prompts) // len(families)
        for family, family_prompts in families.items():
            n_samples = min(samples_per_stratum, len(family_prompts))
            stratified.extend(family_prompts[:n_samples])
        
        return stratified
    
    @staticmethod
    def importance_sampling(scores: np.ndarray, target_mean: float = 0.0) -> np.ndarray:
        """Apply importance sampling weights to reduce variance"""
        if len(scores) == 0:
            return scores
        
        # Compute importance weights (favor samples near decision boundary)
        distances = np.abs(scores - target_mean)
        weights = 1.0 / (1.0 + distances)  # Higher weight for samples near target
        weights = weights / np.sum(weights)  # Normalize
        
        # Resample with weights
        n_samples = len(scores)
        indices = np.random.choice(n_samples, size=n_samples, p=weights, replace=True)
        
        return scores[indices]
    
    @staticmethod
    def control_variates(scores: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Use control variates to reduce variance"""
        if len(scores) == 0 or len(control) == 0:
            return scores
        
        # Ensure same length
        min_len = min(len(scores), len(control))
        scores = scores[:min_len]
        control = control[:min_len]
        
        # Compute optimal coefficient
        cov_matrix = np.cov(scores, control)
        if cov_matrix[1, 1] > 0:
            c_opt = -cov_matrix[0, 1] / cov_matrix[1, 1]
        else:
            c_opt = 0
        
        # Apply control variate
        control_mean = np.mean(control)
        adjusted_scores = scores + c_opt * (control - control_mean)
        
        return adjusted_scores