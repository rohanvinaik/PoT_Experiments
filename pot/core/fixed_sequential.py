import math
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class FixedSequentialTester:
    """Fixed sequential tester with proper scale handling"""
    
    config: Any
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0  # Welford's algorithm
    raw_scores: List[float] = field(default_factory=list)
    clipped_scores: List[float] = field(default_factory=list)
    differences: List[float] = field(default_factory=list)  # Added missing field
    
    def update(self, score: float) -> None:
        """Update with new score"""
        self.n += 1
        self.raw_scores.append(score)
        self.differences.append(score)  # Track differences
        
        # Clip score for stable CI
        clipped = np.clip(
            score, 
            self.config.clip_range[0], 
            self.config.clip_range[1]
        )
        self.clipped_scores.append(clipped)
        
        # Update Welford statistics on raw scores
        delta = score - self.mean
        self.mean += delta / self.n
        delta2 = score - self.mean
        self.M2 += delta * delta2
    
    @property
    def variance(self) -> float:
        """Sample variance"""
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)
    
    @property
    def std_dev(self) -> float:
        """Sample standard deviation"""
        return math.sqrt(max(self.variance, 1e-12))
    
    def compute_ci_eb(self) -> Tuple[Tuple[float, float], float]:
        """Empirical-Bernstein CI with proper effective sample size"""
        if self.n < 2:
            return ((-float("inf"), float("inf")), float("inf"))
        
        # Use clipped scores for EB
        clipped_mean = np.mean(self.clipped_scores)
        clipped_var = np.var(self.clipped_scores, ddof=1)
        
        # Effective sample size including K positions
        n_eff = self.n * self.config.k_positions
        
        # EB bound
        alpha = self.config.alpha
        log_term = math.log(3.0 / alpha)
        
        # Normalized half-width
        h_normalized = (
            math.sqrt(2 * clipped_var * log_term / n_eff) + 
            3 * log_term / max(n_eff - 1, 1)
        )
        
        # Scale to original range
        range_width = self.config.clip_range[1] - self.config.clip_range[0]
        half_width = h_normalized * range_width
        
        # Apply to raw mean
        return ((self.mean - half_width, self.mean + half_width), half_width)
    
    def compute_ci_t(self) -> Tuple[Tuple[float, float], float]:
        """T-distribution CI with effective sample size"""
        if self.n < 2:
            return ((-float("inf"), float("inf")), float("inf"))
        
        # Effective sample size
        n_eff = self.n * self.config.k_positions
        
        # Standard error
        sem = self.std_dev / math.sqrt(n_eff)
        
        # Critical value
        if self.config.confidence >= 0.99:
            t_crit = 2.576 if n_eff >= 30 else 2.845
        elif self.config.confidence >= 0.975:
            t_crit = 1.96 if n_eff >= 30 else 2.262
        else:
            t_crit = 2.0
        
        half_width = t_crit * sem
        return ((self.mean - half_width, self.mean + half_width), half_width)
    
    def check_same_decision(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check SAME with calibrated thresholds"""
        (ci_lo, ci_hi), half_width = self.compute_ci_eb()
        
        # SAME conditions with calibrated gamma
        gamma = self.config.gamma
        target_hw = self.config.half_width_target_same
        
        ci_within_band = (-gamma <= ci_lo and ci_hi <= gamma)
        precision_met = (half_width <= target_hw)
        
        if ci_within_band and precision_met:
            return True, {
                "decision": "SAME",
                "reason": f"CI [{ci_lo:.6f}, {ci_hi:.6f}] within ±{gamma:.6f} and hw={half_width:.6f} ≤ {target_hw:.6f}",
                "ci": (ci_lo, ci_hi),
                "half_width": half_width,
                "mean": self.mean,
                "n_used": self.n,
                "n_eff": self.n * self.config.k_positions
            }
        
        return False, None
    
    def check_different_decision(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check DIFFERENT with calibrated thresholds"""
        (ci_lo, ci_hi), half_width = self.compute_ci_eb()
        
        # DIFFERENT conditions with calibrated delta_star
        delta_star = self.config.delta_star
        
        # Handle both positive differences (ci_lo > delta) and negative (ci_hi < -delta)
        effect_size_met = (ci_lo >= delta_star) or (ci_hi <= -delta_star)
        
        # RME calculation (avoid division by zero)
        if abs(self.mean) > 1e-10:
            rme = half_width / abs(self.mean)
            precision_met = (rme <= self.config.epsilon_diff)
        else:
            rme = float('inf')
            precision_met = False
        
        if effect_size_met and precision_met:
            return True, {
                "decision": "DIFFERENT",
                "reason": f"Effect size met (CI excludes ±{delta_star:.4f}) and RME={rme:.3f} ≤ {self.config.epsilon_diff}",
                "ci": (ci_lo, ci_hi),
                "half_width": half_width,
                "rme": rme,
                "mean": self.mean,
                "n_used": self.n,
                "n_eff": self.n * self.config.k_positions
            }
        
        return False, None
    
    def should_stop(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Determine stopping with calibrated thresholds"""
        
        if self.n < self.config.n_min:
            return False, None
        
        # Check SAME
        same_met, same_info = self.check_same_decision()
        if same_met:
            return True, same_info
        
        # Check DIFFERENT
        diff_met, diff_info = self.check_different_decision()
        if diff_met:
            return True, diff_info
        
        # Check n_max
        if self.n >= self.config.n_max:
            (ci_lo, ci_hi), half_width = self.compute_ci_eb()
            
            return True, {
                "decision": "UNDECIDED",
                "reason": f"Reached n_max={self.config.n_max}",
                "ci": (ci_lo, ci_hi),
                "half_width": half_width,
                "mean": self.mean,
                "n_used": self.n,
                "n_eff": self.n * self.config.k_positions,
                "diagnostics": {
                    "gamma_check": f"CI not within ±{self.config.gamma:.6f}",
                    "delta_check": f"CI doesn't exclude ±{self.config.delta_star:.4f} with good RME"
                }
            }
        
        return False, None