"""
Statistical policy fixes for SAME/DIFFERENT decisions with calibrated thresholds
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from statistics import fmean
import math

@dataclass
class DiffDecisionConfig:
    """Configuration with calibrated defaults and proper policy"""
    mode: str = "AUDIT_GRADE"  # "QUICK_GATE" or "AUDIT_GRADE"
    
    # Policy thresholds (can be auto-calibrated)
    gamma: Optional[float] = None          # SAME equivalence half-band
    delta_star: Optional[float] = None     # DIFFERENT minimum effect
    epsilon_diff: float = 0.10             # RME target when calling DIFFERENT
    
    # CI controls
    ci_method: str = "eb"                  # "eb" or "t"
    score_clip_low: float = 0.0
    score_clip_high: float = 0.3
    confidence: float = 0.99
    
    # Calibration inputs (optional)
    use_calibration: bool = True
    same_model_p95: Optional[float] = None
    near_clone_p5: Optional[float] = None
    
    # Sampling
    n_min: int = 30
    n_max: int = 400
    positions_per_prompt: int = 64
    batch_size: int = 4
    
    # Early stop identical (optional)
    identical_model_n_min: int = 8
    early_stop_threshold: float = 1e-3
    
    # Precision guard
    min_effect_floor: float = 1e-4
    
    # Force decision at max (optional UI behavior)
    force_decision_at_max: bool = False
    
    def finalize(self):
        """Apply mode defaults and calibration"""
        # Defaults by mode
        if self.mode == "QUICK_GATE":
            self.confidence = getattr(self, "confidence", 0.975)
            self.gamma = self.gamma or 0.0015
            self.delta_star = self.delta_star or 0.10
            self.epsilon_diff = self.epsilon_diff or 0.20
            self.n_min = self.n_min or 12
            self.n_max = self.n_max or 120
            self.positions_per_prompt = self.positions_per_prompt or 64
        else:  # AUDIT_GRADE
            self.confidence = getattr(self, "confidence", 0.99)
            self.gamma = self.gamma or 0.0010
            self.delta_star = self.delta_star or 0.038
            self.epsilon_diff = self.epsilon_diff or 0.10
            self.n_min = self.n_min or 30
            self.n_max = self.n_max or 400
            self.positions_per_prompt = self.positions_per_prompt or 128
        
        # Auto-calibrate if provided
        if self.use_calibration and self.same_model_p95 is not None:
            self.gamma = 3 * self.same_model_p95
            if self.near_clone_p5 is not None:
                self.delta_star = (self.same_model_p95 + self.near_clone_p5) / 2.0
        
        # Back-compat fields
        self.alpha = getattr(self, "alpha", 1.0 - self.confidence)
        self.beta = getattr(self, "beta", self.alpha)  # symmetric by default


class SequentialDiffTester:
    """Sequential tester with proper policy implementation"""
    
    def __init__(self, config: DiffDecisionConfig):
        self.config = config
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Welford's algorithm
        self.history = []
    
    def add_sample(self, x: float):
        """Add sample ensuring non-negative scores"""
        # Ensure non-negative (larger = more different)
        x = max(0.0, float(x))
        self.history.append(x)
        self._update(x)
    
    def _update(self, x: float):
        """Welford's online update"""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    
    def compute_ci(self) -> Tuple[Tuple[float, float], float]:
        """Compute confidence interval"""
        if self.config.ci_method == "eb":
            # Empirical-Bernstein on bounded scores
            a, b = self.config.score_clip_low, self.config.score_clip_high
            R = max(b - a, 1e-12)
            
            # Clip history to bounds
            y = [min(max(v, a), b) for v in self.history]
            n = len(y)
            
            if n < 2:
                return ((-float("inf"), float("inf")), float("inf"))
            
            ybar = fmean(y)
            vy = sum((t - ybar) ** 2 for t in y) / (n - 1)
            
            # Special case: zero variance (identical models)
            if vy < 1e-10 and abs(ybar) < 1e-10:
                # Return tight CI around 0
                return ((0.0, 0.0), 0.0)
            
            # EB confidence radius
            ln = math.log(3.0 / self.config.alpha)
            h_raw = (math.sqrt(2 * vy * ln / n) + 3 * ln / max(n - 1, 1)) * R
            
            return ((self.mean - h_raw, self.mean + h_raw), h_raw)
        else:
            # T-distribution CI
            if self.n < 2:
                return ((-float("inf"), float("inf")), float("inf"))
            
            # Special case: zero variance (identical models)
            s2 = self.M2 / (self.n - 1)
            if s2 < 1e-10 and abs(self.mean) < 1e-10:
                return ((0.0, 0.0), 0.0)
            
            s = math.sqrt(max(s2, 1e-12))
            sem = s / math.sqrt(self.n)
            
            # Critical value for confidence level
            if self.config.confidence >= 0.99:
                z = 2.576 if self.n >= 30 else 2.821
            elif self.config.confidence >= 0.975:
                z = 1.96 if self.n >= 30 else 2.262
            else:
                z = 1.645
            
            h = z * sem
            return ((self.mean - h, self.mean + h), h)
    
    def check_same_decision(self) -> Tuple[bool, Dict[str, Any]]:
        """Check SAME decision (equivalence)"""
        if self.n < self.config.n_min:
            return False, {}
        
        (lo, hi), h = self.compute_ci()
        g = self.config.gamma
        
        # SAME: CI within [-γ, +γ] and precision met
        if (-g <= lo) and (hi <= +g) and (h <= 0.5 * g):
            return True, {
                "decision": "SAME",
                "ci": (lo, hi),
                "half_width": h,
                "mean": self.mean,
                "rule": "equivalence",
                "n_used": self.n
            }
        
        return False, {}
    
    def check_different_decision(self) -> Tuple[bool, Dict[str, Any]]:
        """Check DIFFERENT decision (minimum effect + precision)"""
        if self.n < self.config.n_min:
            return False, {}
        
        (lo, hi), h = self.compute_ci()
        
        # RME calculation
        denom = max(abs(self.mean), self.config.min_effect_floor)
        rme = h / denom
        
        # DIFFERENT: lower CI ≥ δ* and RME met
        if (lo >= self.config.delta_star) and (rme <= self.config.epsilon_diff):
            return True, {
                "decision": "DIFFERENT",
                "ci": (lo, hi),
                "half_width": h,
                "mean": self.mean,
                "rule": "min_effect",
                "rme": rme,
                "n_used": self.n
            }
        
        return False, {}
    
    def should_stop(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Determine if we should stop sampling"""
        
        # Early stop for identical models
        if self.n >= self.config.identical_model_n_min:
            if abs(self.mean) < self.config.early_stop_threshold:
                (lo, hi), h = self.compute_ci()
                if h < self.config.early_stop_threshold:
                    return True, {
                        "decision": "SAME",
                        "rule": "identical_early_stop",
                        "ci": (lo, hi),
                        "half_width": h,
                        "mean": self.mean,
                        "n_used": self.n
                    }
        
        # Check SAME
        same, info = self.check_same_decision()
        if same:
            return True, info
        
        # Check DIFFERENT
        diff, info = self.check_different_decision()
        if diff:
            return True, info
        
        # Check max samples
        if self.n >= self.config.n_max:
            (lo, hi), h = self.compute_ci()
            
            if self.config.force_decision_at_max:
                # Choose nearest side for UX
                if abs(self.mean) <= self.config.gamma:
                    decision = "SAME"
                    rule = "equivalence_forced"
                else:
                    decision = "DIFFERENT"
                    rule = "min_effect_forced"
                
                return True, {
                    "decision": decision,
                    "rule": rule,
                    "ci": (lo, hi),
                    "half_width": h,
                    "mean": self.mean,
                    "at_max": True,
                    "n_used": self.n
                }
            
            return True, {
                "decision": "UNDECIDED",
                "ci": (lo, hi),
                "half_width": h,
                "mean": self.mean,
                "n_used": self.n,
                "reason": f"Reached n_max={self.config.n_max}"
            }
        
        return False, None