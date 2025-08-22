"""
Statistical Difference Decision Framework for Model Verification

This module implements a statistical difference testing framework for model verification
with anytime stopping based on confidence intervals and relative margin of error.
Uses Empirical-Bernstein or t-distribution confidence intervals with Welford's online
algorithm for efficient streaming computation.
"""

import math
import time
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List, Optional, Literal, Callable
from statistics import fmean
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class TestingMode(Enum):
    """Testing modes with different precision/speed tradeoffs"""
    QUICK_GATE = "quick_gate"  # Fast initial check
    AUDIT_GRADE = "audit_grade"  # High precision verification

@dataclass
class DiffDecisionConfig:
    """Enhanced configuration with separate SAME/DIFFERENT rules"""
    
    # Testing mode
    mode: TestingMode = TestingMode.AUDIT_GRADE
    
    # Confidence levels (auto-set based on mode)
    confidence: Optional[float] = None  # Will be set from mode
    
    # SAME decision parameters (equivalence testing)
    gamma: Optional[float] = None  # Equivalence band width
    eta: float = 0.5  # Precision factor for SAME (half_width ≤ η·γ)
    
    # DIFFERENT decision parameters
    delta_star: Optional[float] = None  # Minimum effect size
    epsilon_diff: Optional[float] = None  # Max relative margin of error
    
    # Sampling parameters (auto-set based on mode)
    n_min: Optional[int] = None
    n_max: Optional[int] = None
    positions_per_prompt: Optional[int] = None  # K positions
    batch_size: int = 4
    
    # Score normalization
    score_clip_low: float = -100.0  # Allow negative CE differences
    score_clip_high: float = 100.0  # Allow large positive CE differences
    
    # CI method
    ci_method: Literal["eb", "t"] = "eb"  # Empirical-Bernstein recommended
    
    # Calibration parameters (optional)
    use_calibration: bool = False
    same_model_p95: Optional[float] = None  # 95th percentile of |mean| for same models
    near_clone_p5: Optional[float] = None   # 5th percentile for near-clones
    
    # Legacy compatibility fields
    alpha: Optional[float] = None  # Computed from confidence
    rel_margin_target: Optional[float] = None  # Maps to epsilon_diff
    min_effect_floor: float = 1e-4
    method: Optional[Literal["eb", "t"]] = None  # Maps to ci_method
    clip_low: Optional[float] = None  # Maps to score_clip_low
    clip_high: Optional[float] = None  # Maps to score_clip_high
    equivalence_band: Optional[float] = None  # Maps to gamma
    similar_size_ratio: float = 2.0
    early_stop_threshold: float = 1e-6
    identical_model_n_min: int = 20
    
    def __post_init__(self):
        """Set defaults based on testing mode and handle legacy fields"""
        if self.mode == TestingMode.QUICK_GATE:
            # Quick gate defaults
            self.confidence = self.confidence or 0.975
            self.gamma = self.gamma or 0.015
            self.delta_star = self.delta_star or 0.10
            self.epsilon_diff = self.epsilon_diff or 0.20
            self.n_min = self.n_min or 12
            self.n_max = self.n_max or 120
            self.positions_per_prompt = self.positions_per_prompt or 32
        else:  # AUDIT_GRADE
            # Audit grade defaults
            self.confidence = self.confidence or 0.99
            self.gamma = self.gamma or 0.010
            self.delta_star = self.delta_star or 0.10
            self.epsilon_diff = self.epsilon_diff or 0.10
            self.n_min = self.n_min or 30
            self.n_max = self.n_max or 400
            self.positions_per_prompt = self.positions_per_prompt or 64
        
        # Auto-calibrate if data provided
        if self.use_calibration and self.same_model_p95 is not None:
            self.gamma = self.same_model_p95
            if self.near_clone_p5 is not None:
                self.delta_star = (self.same_model_p95 + self.near_clone_p5) / 2
        
        # Handle legacy fields
        self.alpha = self.alpha or (1.0 - self.confidence)
        self.rel_margin_target = self.rel_margin_target or self.epsilon_diff
        self.method = self.method or self.ci_method
        self.clip_low = self.clip_low if self.clip_low is not None else self.score_clip_low
        self.clip_high = self.clip_high if self.clip_high is not None else self.score_clip_high
        self.equivalence_band = self.equivalence_band or self.gamma
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DiffDecisionConfig':
        """Create config from dictionary"""
        return cls(**d)

# ============================================================================
# ENHANCED SEQUENTIAL TESTER
# ============================================================================

class EnhancedSequentialTester:
    """Sequential tester with separate SAME/DIFFERENT decision rules
    
    Implements enhanced decision logic with:
    - Separate criteria for SAME vs DIFFERENT decisions
    - Better diagnostics for UNDECIDED cases
    - Support for both quick gate and audit grade modes
    """
    
    def __init__(self, config: DiffDecisionConfig):
        self.config = config
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Welford's algorithm
        self.raw_scores: List[float] = []
        self.clipped_scores: List[float] = []
        self.differences: List[float] = []  # Track differences for compatibility
    
    def update(self, score: float) -> None:
        """Update with new score"""
        self.n += 1
        self.raw_scores.append(score)
        self.differences.append(score)  # Track differences
        
        # Clip score for stable CI
        clipped = np.clip(score, self.config.score_clip_low, self.config.score_clip_high)
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
    
    def _compute_ci_eb(self) -> Tuple[Tuple[float, float], float]:
        """Empirical-Bernstein CI using clipped scores"""
        if self.n < 2:
            return ((-float("inf"), float("inf")), float("inf"))
        
        # Use clipped scores for EB
        clipped_mean = np.mean(self.clipped_scores)
        clipped_var = np.var(self.clipped_scores, ddof=1)
        
        # Special case: zero variance (identical models)
        if clipped_var < 1e-10 and abs(clipped_mean) < 1e-10:
            # Return tight CI around 0
            return ((0.0, 0.0), 0.0)
        
        # EB bound calculation
        alpha = self.config.alpha
        log_term = math.log(3.0 / alpha)
        
        # Effective sample size (n * K)
        n_eff = self.n * self.config.positions_per_prompt
        
        # EB half-width
        half_width_normalized = (
            math.sqrt(2 * clipped_var * log_term / n_eff) + 
            3 * log_term / max(n_eff - 1, 1)
        )
        
        # Scale back to original range
        range_scale = self.config.score_clip_high - self.config.score_clip_low
        half_width = half_width_normalized * range_scale
        
        # Apply to raw mean (not clipped mean)
        return ((self.mean - half_width, self.mean + half_width), half_width)
    
    def _compute_ci_t(self) -> Tuple[Tuple[float, float], float]:
        """T-distribution CI with effective sample size"""
        if self.n < 2:
            return ((-float("inf"), float("inf")), float("inf"))
        
        # Special case: zero variance (identical models)
        if self.variance < 1e-10 and abs(self.mean) < 1e-10:
            # Return tight CI around 0
            return ((0.0, 0.0), 0.0)
        
        # Effective sample size
        n_eff = self.n * self.config.positions_per_prompt
        
        # Standard error with effective n
        sem = self.std_dev / math.sqrt(n_eff)
        
        # Critical value based on confidence
        if self.config.confidence >= 0.99:
            t_crit = 2.576 if n_eff >= 30 else 2.845
        elif self.config.confidence >= 0.975:
            t_crit = 1.96 if n_eff >= 30 else 2.262
        else:
            t_crit = 2.576  # Default to 99%
        
        half_width = t_crit * sem
        return ((self.mean - half_width, self.mean + half_width), half_width)
    
    def compute_ci(self) -> Tuple[Tuple[float, float], float]:
        """Compute CI using configured method"""
        if self.config.ci_method == "eb":
            return self._compute_ci_eb()
        else:
            return self._compute_ci_t()
    
    def ci(self) -> Tuple[Tuple[float, float], float]:
        """Alias for compute_ci() for compatibility"""
        return self.compute_ci()
    
    def check_same_decision(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if we can declare SAME"""
        (ci_lo, ci_hi), half_width = self.compute_ci()
        gamma = self.config.gamma
        
        # SAME conditions:
        # 1. CI entirely within [-γ, +γ]
        # 2. Half-width ≤ η·γ (precision condition)
        
        ci_within_band = (ci_lo >= -gamma and ci_hi <= gamma)
        precision_met = (half_width <= self.config.eta * gamma)
        
        if ci_within_band and precision_met:
            return True, {
                "decision": "SAME",
                "reason": f"CI [{ci_lo:.6f}, {ci_hi:.6f}] within ±{gamma} and half_width {half_width:.6f} ≤ {self.config.eta * gamma:.6f}",
                "ci": (ci_lo, ci_hi),
                "half_width": half_width,
                "gamma": gamma,
                "mean": self.mean
            }
        
        return False, None
    
    def check_different_decision(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if we can declare DIFFERENT"""
        (ci_lo, ci_hi), half_width = self.compute_ci()
        delta_star = self.config.delta_star
        
        # Avoid division by zero
        if abs(self.mean) < 1e-10:
            return False, None
        
        # DIFFERENT conditions:
        # 1. |Lower CI bound| ≥ δ* (effect size requirement)
        # 2. RME ≤ ε_diff (precision requirement)
        
        effect_size_met = (abs(ci_lo) >= delta_star or abs(ci_hi) >= delta_star)
        rme = half_width / abs(self.mean)
        precision_met = (rme <= self.config.epsilon_diff)
        
        if effect_size_met and precision_met:
            direction = "higher" if self.mean > 0 else "lower"
            return True, {
                "decision": "DIFFERENT",
                "direction": direction,
                "reason": f"Effect size {max(abs(ci_lo), abs(ci_hi)):.6f} ≥ δ* {delta_star} and RME {rme:.3f} ≤ {self.config.epsilon_diff}",
                "ci": (ci_lo, ci_hi),
                "half_width": half_width,
                "rme": rme,
                "delta_star": delta_star,
                "mean": self.mean
            }
        
        return False, None
    
    def should_stop(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Determine if we should stop sampling"""
        
        # Need minimum samples
        if self.n < self.config.n_min:
            return False, None
        
        # Check for identical models (early stopping)
        if (self.n >= self.config.identical_model_n_min and 
            abs(self.mean) < self.config.early_stop_threshold):
            (ci_lo, ci_hi), half_width = self.compute_ci()
            if half_width < self.config.early_stop_threshold:
                return True, {
                    "decision": "IDENTICAL",
                    "reason": "Mean and CI indicate identical models",
                    "ci": (ci_lo, ci_hi),
                    "half_width": half_width,
                    "mean": self.mean
                }
        
        # Check SAME decision
        same_met, same_info = self.check_same_decision()
        if same_met:
            return True, same_info
        
        # Check DIFFERENT decision
        diff_met, diff_info = self.check_different_decision()
        if diff_met:
            return True, diff_info
        
        # Check if we've hit n_max
        if self.n >= self.config.n_max:
            (ci_lo, ci_hi), half_width = self.compute_ci()
            
            # Detailed diagnostics for UNDECIDED
            diagnostics = {
                "same_check": {
                    "ci_within_band": (ci_lo >= -self.config.gamma and ci_hi <= self.config.gamma),
                    "precision_met": (half_width <= self.config.eta * self.config.gamma),
                    "needed_half_width": self.config.eta * self.config.gamma
                },
                "different_check": {
                    "effect_size_met": (abs(ci_lo) >= self.config.delta_star or abs(ci_hi) >= self.config.delta_star),
                    "rme": half_width / abs(self.mean) if abs(self.mean) > 1e-10 else float('inf'),
                    "rme_target": self.config.epsilon_diff
                }
            }
            
            return True, {
                "decision": "UNDECIDED",
                "reason": "Reached n_max without meeting decision criteria",
                "ci": (ci_lo, ci_hi),
                "half_width": half_width,
                "mean": self.mean,
                "diagnostics": diagnostics,
                "suggestions": self._get_suggestions(diagnostics, half_width)
            }
        
        return False, None
    
    def _get_suggestions(self, diagnostics: Dict[str, Any], half_width: float) -> List[str]:
        """Generate specific suggestions based on diagnostics"""
        suggestions = []
        
        # For SAME path
        if not diagnostics["same_check"]["ci_within_band"]:
            suggestions.append(f"Increase K to {self.config.positions_per_prompt * 2} positions per prompt")
            suggestions.append("Ensure score clipping range is appropriate")
        elif not diagnostics["same_check"]["precision_met"]:
            current_hw = half_width
            needed_hw = diagnostics["same_check"]["needed_half_width"]
            factor = (current_hw / needed_hw) ** 2 if needed_hw > 0 else 1
            suggestions.append(f"Need ~{factor:.1f}x more effective samples (increase K or n_max)")
        
        # For DIFFERENT path
        if diagnostics["different_check"]["effect_size_met"]:
            rme = diagnostics["different_check"]["rme"]
            target = diagnostics["different_check"]["rme_target"]
            if rme > target:
                factor = (rme / target) ** 2
                suggestions.append(f"Need ~{factor:.1f}x more samples to meet RME target of {target:.2f}")
        else:
            suggestions.append("Effect size too small - may be near-clones or need different score metric")
        
        suggestions.extend([
            "Consider switching score method (delta_ce → symmetric_kl)",
            "Add more diverse challenge types",
            "Check if models are loading correctly"
        ])
        
        return suggestions
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for monitoring"""
        (lo, hi), hw = self.compute_ci()
        rel_me = hw / max(abs(self.mean), self.config.min_effect_floor)
        
        return {
            "n": self.n,
            "mean": self.mean,
            "variance": self.variance,
            "std_dev": self.std_dev,
            "ci": (lo, hi),
            "half_width": hw,
            "rel_me": rel_me,
            "mode": self.config.mode.value,
            "gamma": self.config.gamma,
            "delta_star": self.config.delta_star
        }
    
    @property
    def raw_history(self) -> List[float]:
        """Get raw score history for compatibility"""
        return self.raw_scores
    
    @property  
    def history(self) -> List[float]:
        """Get clipped score history for compatibility"""
        return self.clipped_scores

# Keep original SequentialDiffTester for backwards compatibility
class SequentialDiffTester:
    """Sequential testing with anytime-valid confidence intervals
    
    Uses Welford's online algorithm for efficient streaming computation
    of mean and variance without storing all observations.
    """
    
    def __init__(self, cfg: DiffDecisionConfig):
        self.cfg = cfg
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Welford's algorithm for variance
        self.history: List[float] = []
        self.raw_history: List[float] = []  # Keep raw scores for reporting
        
    def update(self, x: float) -> None:
        """Update statistics with new observation using Welford's method
        
        Args:
            x: New observation (score difference)
        """
        self.n += 1
        self.raw_history.append(x)
        
        # Welford's online algorithm
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
        # Store for EB computation if needed
        if self.cfg.method == "eb":
            # Clip to range for EB
            clipped = max(min(x, self.cfg.clip_high), self.cfg.clip_low)
            self.history.append(clipped)
        else:
            self.history.append(x)
    
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
    
    def _ci_t(self) -> Tuple[Tuple[float, float], float]:
        """Student t-distribution confidence interval
        
        Returns:
            ((lower, upper), half_width) confidence interval bounds and half-width
        """
        if self.n < 2:
            return ((-float("inf"), float("inf")), float("inf"))
        
        sem = self.std_dev / math.sqrt(self.n)
        
        # Use t-distribution critical values
        # For 99% CI (α=0.01), two-sided
        if self.n >= 120:
            t_crit = 2.576  # Normal approximation
        elif self.n >= 60:
            t_crit = 2.660
        elif self.n >= 30:
            t_crit = 2.750
        elif self.n >= 20:
            t_crit = 2.845
        elif self.n >= 15:
            t_crit = 2.977
        elif self.n >= 10:
            t_crit = 3.169
        elif self.n >= 5:
            t_crit = 4.032
        else:
            t_crit = 4.604  # Conservative for very small n
        
        half_width = t_crit * sem
        return ((self.mean - half_width, self.mean + half_width), half_width)
    
    def _ci_eb(self) -> Tuple[Tuple[float, float], float]:
        """Empirical-Bernstein confidence interval for bounded scores
        
        Based on Howard et al., 2021 - "Time-uniform, nonparametric, nonasymptotic
        confidence sequences" with empirical Bernstein bounds.
        
        Returns:
            ((lower, upper), half_width) confidence interval bounds and half-width
        """
        if self.n < 2:
            return ((-float("inf"), float("inf")), float("inf"))
        
        # Normalize history to [0,1]
        a, b = self.cfg.clip_low, self.cfg.clip_high
        R = max(b - a, 1e-12)
        
        # Use normalized values for EB
        y = [(x - a) / R for x in self.history]
        n = len(y)
        ybar = fmean(y)
        
        # Sample variance of normalized values
        vy = sum((t - ybar) ** 2 for t in y) / (n - 1)
        
        # EB confidence radius (Howard et al., 2021 style)
        delta = self.cfg.alpha
        ln_term = math.log(3.0 / delta)
        
        # Empirical Bernstein bound with variance term and range term
        # This gives anytime-valid confidence sequences
        variance_term = math.sqrt(2 * vy * ln_term / n)
        range_term = 3 * ln_term / max(n - 1, 1)
        h_normalized = variance_term + range_term
        
        # Map back to raw scale
        h_raw = h_normalized * R
        
        return ((self.mean - h_raw, self.mean + h_raw), h_raw)
    
    def ci(self) -> Tuple[Tuple[float, float], float]:
        """Get confidence interval using configured method
        
        Returns:
            ((lower, upper), half_width) confidence interval bounds and half-width
        """
        if self.cfg.method == "eb":
            return self._ci_eb()
        else:
            return self._ci_t()
    
    def should_stop(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Determine if we should stop sampling
        
        Returns:
            (should_stop, info_dict) where info_dict contains decision details
        """
        (lo, hi), half_width = self.ci()
        
        # Compute relative margin of error
        denom = max(abs(self.mean), self.cfg.min_effect_floor)
        rel_me = half_width / denom
        
        # Early stopping for identical models (very small differences)
        if (self.n >= self.cfg.identical_model_n_min and 
            abs(self.mean) < self.cfg.early_stop_threshold and
            half_width < self.cfg.early_stop_threshold):
            return True, {
                "decision": "IDENTICAL",
                "rel_me": rel_me,
                "half_width": half_width,
                "mean": self.mean,
                "ci": (lo, hi),
                "reason": "Mean and CI indicate identical models"
            }
        
        # Decision 1: DIFFERENT if 0 excluded and precision is good
        if (lo > 0 or hi < 0) and rel_me <= self.cfg.rel_margin_target:
            direction = "higher" if self.mean > 0 else "lower"
            return True, {
                "decision": "DIFFERENT",
                "direction": direction,
                "rel_me": rel_me,
                "half_width": half_width,
                "mean": self.mean,
                "ci": (lo, hi),
                "reason": f"CI excludes 0 with rel_me={rel_me:.3f} <= {self.cfg.rel_margin_target}"
            }
        
        # Decision 2: SAME if using equivalence band (TOST)
        if self.cfg.equivalence_band is not None:
            gamma = self.cfg.equivalence_band
            if -gamma <= lo and hi <= gamma and rel_me <= self.cfg.rel_margin_target:
                return True, {
                    "decision": "SAME",
                    "rel_me": rel_me,
                    "half_width": half_width,
                    "mean": self.mean,
                    "ci": (lo, hi),
                    "reason": f"CI within equivalence band ±{gamma}"
                }
        
        # Check if we've hit n_max
        if self.n >= self.cfg.n_max:
            # Provide detailed reason for undecided
            if not (lo > 0 or hi < 0):
                reason = f"CI [{lo:.6f}, {hi:.6f}] includes 0 after {self.n} samples"
            else:
                reason = f"Precision target not met: rel_me={rel_me:.3f} > {self.cfg.rel_margin_target}"
            
            return True, {
                "decision": "UNDECIDED",
                "rel_me": rel_me,
                "half_width": half_width,
                "mean": self.mean,
                "ci": (lo, hi),
                "reason": reason
            }
        
        # Continue sampling
        return False, None
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for monitoring"""
        (lo, hi), hw = self.ci()
        rel_me = hw / max(abs(self.mean), self.cfg.min_effect_floor)
        
        return {
            "n": self.n,
            "mean": self.mean,
            "variance": self.variance,
            "std_dev": self.std_dev,
            "ci": (lo, hi),
            "half_width": hw,
            "rel_me": rel_me
        }

# ============================================================================
# MAIN VERIFIER
# ============================================================================

class DifferenceVerifier:
    """Main orchestrator for statistical difference testing
    
    Coordinates the sequential testing process, managing prompt generation,
    scoring, and decision making.
    """
    
    def __init__(self,
                 score_fn: Callable,
                 prompt_generator: Callable,
                 cfg: DiffDecisionConfig,
                 use_enhanced: bool = True):
        """Initialize verifier
        
        Args:
            score_fn: Function(ref_model, cand_model, prompt, K) -> score
            prompt_generator: Function() -> prompt string
            cfg: Configuration for difference testing
            use_enhanced: Whether to use enhanced tester (default True)
        """
        self.score_fn = score_fn
        self.prompt_generator = prompt_generator
        self.cfg = cfg
        
        # Use enhanced tester by default for better diagnostics
        if use_enhanced and hasattr(cfg, 'mode'):
            self.tester = EnhancedSequentialTester(cfg)
        else:
            self.tester = SequentialDiffTester(cfg)
        
    def verify_difference(self,
                         ref_model: Any,
                         cand_model: Any,
                         output_dir: Optional[Path] = None,
                         verbose: bool = True) -> Dict[str, Any]:
        """Run sequential difference testing
        
        Args:
            ref_model: Reference model
            cand_model: Candidate model to compare
            output_dir: Optional directory to save results
            verbose: Whether to log progress
            
        Returns:
            Comprehensive report dictionary with decision and metrics
        """
        
        start_time = time.perf_counter()
        n_batches = math.ceil(self.cfg.n_max / self.cfg.batch_size)
        
        # Track detailed metrics
        batch_times = []
        score_times = []
        all_prompts = []
        
        logger.info(f"Starting difference testing with config: {self.cfg}")
        
        for batch_idx in range(n_batches):
            batch_start = time.perf_counter()
            
            # Generate batch of prompts
            prompts = [self.prompt_generator() for _ in range(self.cfg.batch_size)]
            all_prompts.extend(prompts)
            
            # Score each prompt
            for prompt in prompts:
                score_start = time.perf_counter()
                
                # Get score difference between models
                score = self.score_fn(
                    ref_model, 
                    cand_model, 
                    prompt, 
                    K=self.cfg.positions_per_prompt
                )
                
                score_times.append(time.perf_counter() - score_start)
                self.tester.update(score)
                
                # Log progress
                if verbose and self.tester.n % 10 == 0:
                    state = self.tester.get_state()
                    logger.info(f"n={state['n']}: mean={state['mean']:.6f}, "
                              f"CI=[{state['ci'][0]:.6f}, {state['ci'][1]:.6f}], "
                              f"rel_me={state['rel_me']:.3f}")
            
            batch_times.append(time.perf_counter() - batch_start)
            
            # Check stopping condition
            if self.tester.n >= self.cfg.n_min:
                should_stop, info = self.tester.should_stop()
                if should_stop:
                    return self._build_report(
                        info, 
                        all_prompts[:self.tester.n],
                        time.perf_counter() - start_time,
                        batch_times,
                        score_times,
                        output_dir
                    )
        
        # Shouldn't reach here, but handle gracefully
        _, info = self.tester.should_stop()
        return self._build_report(
            info or {"decision": "UNDECIDED", "reason": "Maximum samples reached"},
            all_prompts[:self.tester.n],
            time.perf_counter() - start_time,
            batch_times,
            score_times,
            output_dir
        )
    
    def _build_report(self,
                     info: Dict[str, Any],
                     prompts: List[str],
                     total_time: float,
                     batch_times: List[float],
                     score_times: List[float],
                     output_dir: Optional[Path]) -> Dict[str, Any]:
        """Build comprehensive report
        
        Args:
            info: Decision information from tester
            prompts: List of prompts used
            total_time: Total verification time
            batch_times: List of batch processing times
            score_times: List of individual scoring times
            output_dir: Optional directory to save results
            
        Returns:
            Comprehensive report dictionary
        """
        
        (lo, hi), hw = self.tester.ci()
        
        report = {
            "verifier": "stat_diff_v1",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "config": {
                "alpha": self.cfg.alpha,
                "rel_margin_target": self.cfg.rel_margin_target,
                "method": self.cfg.method,
                "n_min": self.cfg.n_min,
                "n_max": self.cfg.n_max,
                "positions_per_prompt": self.cfg.positions_per_prompt,
                "batch_size": self.cfg.batch_size,
                "equivalence_band": self.cfg.equivalence_band,
                "early_stop_threshold": self.cfg.early_stop_threshold
            },
            "results": {
                "decision": info["decision"],
                "reason": info.get("reason", ""),
                "direction": info.get("direction", None),
                "n_used": self.tester.n,
                "mean": float(self.tester.mean),
                "variance": float(self.tester.variance),
                "std_dev": float(self.tester.std_dev),
                "ci_99": [float(lo), float(hi)],
                "half_width": float(hw),
                "rel_me": float(info.get("rel_me", hw / max(abs(self.tester.mean), 1e-6)))
            },
            "timing": {
                "total_time_sec": total_time,
                "avg_batch_time_sec": np.mean(batch_times) if batch_times else 0,
                "avg_score_time_sec": np.mean(score_times) if score_times else 0,
                "total_score_time_sec": sum(score_times),
                "scores_per_second": len(score_times) / total_time if total_time > 0 else 0
            },
            "scores": {
                "raw": self.tester.raw_history,
                "summary_stats": {
                    "min": float(np.min(self.tester.raw_history)) if self.tester.raw_history else 0,
                    "max": float(np.max(self.tester.raw_history)) if self.tester.raw_history else 0,
                    "median": float(np.median(self.tester.raw_history)) if self.tester.raw_history else 0,
                    "q25": float(np.percentile(self.tester.raw_history, 25)) if self.tester.raw_history else 0,
                    "q75": float(np.percentile(self.tester.raw_history, 75)) if self.tester.raw_history else 0,
                    "iqr": float(np.percentile(self.tester.raw_history, 75) - 
                                np.percentile(self.tester.raw_history, 25)) if self.tester.raw_history else 0
                }
            }
        }
        
        # Add interpretation based on decision
        report["interpretation"] = self._get_interpretation(info["decision"], self.tester.mean)
        
        # Add next steps based on decision
        report["next_steps"] = self._get_next_steps(info["decision"])
        
        # Save report if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main report
            report_file = output_dir / f"difference_test_report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {report_file}")
            
            # Save prompts used
            prompts_file = output_dir / f"prompts_used_{int(time.time())}.json"
            with open(prompts_file, 'w') as f:
                json.dump({"prompts": prompts[:self.tester.n]}, f, indent=2)
            
            # Save summary for quick viewing
            summary_file = output_dir / f"summary_{int(time.time())}.txt"
            with open(summary_file, 'w') as f:
                f.write(self._generate_summary(report))
        
        return report
    
    def _get_interpretation(self, decision: str, mean: float) -> str:
        """Get human-readable interpretation of decision"""
        
        if decision == "DIFFERENT":
            if mean > 0:
                return "Candidate model produces statistically higher scores than reference"
            else:
                return "Candidate model produces statistically lower scores than reference"
        elif decision == "IDENTICAL":
            return "Models are statistically identical (differences within measurement noise)"
        elif decision == "SAME":
            return "Models are statistically equivalent within specified tolerance"
        else:  # UNDECIDED
            return "Unable to determine statistical difference with current sample size and precision"
    
    def _get_next_steps(self, decision: str) -> List[str]:
        """Get recommended next steps based on decision"""
        
        if decision == "DIFFERENT":
            return [
                "Confirm difference via secondary PoT metrics:",
                "- Fuzzy/TLSH behavioral hashes on the same challenges",
                "- Merkle commit of (challenge,response) pairs; compare roots",
                "- Attack detectors (distill/prune/quant/LoRA flags)",
                "- (Optional, attested mode) Jacobian/CKA sketch",
                "- Run extended test with more positions per prompt",
                "- Analyze score distributions for patterns"
            ]
        elif decision == "IDENTICAL":
            return [
                "Models appear identical. Recommended checks:",
                "- Verify file hashes match",
                "- Check model metadata and configuration",
                "- Confirm both models loaded correctly",
                "- No further statistical testing needed"
            ]
        elif decision == "SAME":
            return [
                "Models statistically equivalent within tolerance:",
                "- May be minor quantization or precision differences",
                "- Consider tighter equivalence band if needed",
                "- Check for deterministic behavior on edge cases",
                "- Verify training provenance if available"
            ]
        else:  # UNDECIDED
            return [
                f"Increase n_max beyond {self.cfg.n_max}",
                f"Increase positions_per_prompt beyond {self.cfg.positions_per_prompt}",
                "Try different scoring method (symmetric_kl, js_divergence)",
                "Diversify challenge families (add edge cases, adversarial)",
                "If still undecided, escalate to secondary PoT metrics",
                "Consider adjusting rel_margin_target for this model pair",
                "Check if models are too similar for statistical differentiation"
            ]
    
    def _generate_summary(self, report: Dict[str, Any]) -> str:
        """Generate human-readable summary"""
        
        r = report["results"]
        t = report["timing"]
        
        summary = f"""
Statistical Difference Test Summary
====================================
Decision: {r['decision']}
Reason: {r['reason']}

Statistics:
-----------
Samples used: {r['n_used']}
Mean difference: {r['mean']:.6f}
Standard deviation: {r['std_dev']:.6f}
99% CI: [{r['ci_99'][0]:.6f}, {r['ci_99'][1]:.6f}]
Relative margin: {r['rel_me']:.3f}

Performance:
------------
Total time: {t['total_time_sec']:.2f} seconds
Scores per second: {t['scores_per_second']:.2f}

Interpretation:
---------------
{report['interpretation']}

Next Steps:
-----------
"""
        for step in report['next_steps'][:3]:  # Show first 3 recommendations
            summary += f"• {step}\n"
        
        return summary

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_default_verifier(score_fn: Callable,
                           prompt_generator: Callable,
                           **config_overrides) -> DifferenceVerifier:
    """Create a verifier with default configuration
    
    Args:
        score_fn: Scoring function
        prompt_generator: Prompt generation function
        **config_overrides: Optional config parameters to override
        
    Returns:
        Configured DifferenceVerifier instance
    """
    cfg = DiffDecisionConfig(**config_overrides)
    return DifferenceVerifier(score_fn, prompt_generator, cfg)

def create_enhanced_verifier(score_fn: Callable,
                            prompt_generator: Callable,
                            mode: TestingMode = TestingMode.AUDIT_GRADE,
                            **config_overrides) -> DifferenceVerifier:
    """Create an enhanced verifier with separate SAME/DIFFERENT rules
    
    Args:
        score_fn: Scoring function
        prompt_generator: Prompt generation function
        mode: Testing mode (QUICK_GATE or AUDIT_GRADE)
        **config_overrides: Optional config parameters to override
        
    Returns:
        Configured DifferenceVerifier with enhanced tester
    """
    cfg = DiffDecisionConfig(mode=mode, **config_overrides)
    return DifferenceVerifier(score_fn, prompt_generator, cfg, use_enhanced=True)

def validate_models_compatible(ref_model: Any, 
                              cand_model: Any,
                              cfg: DiffDecisionConfig) -> Tuple[bool, str]:
    """Validate that models are compatible for comparison
    
    Args:
        ref_model: Reference model
        cand_model: Candidate model
        cfg: Configuration with size ratio limits
        
    Returns:
        (is_compatible, reason) tuple
    """
    try:
        # Check if models have parameter counts
        if hasattr(ref_model, 'num_parameters') and hasattr(cand_model, 'num_parameters'):
            try:
                # Try to call if it's a method
                ref_params = ref_model.num_parameters() if callable(ref_model.num_parameters) else ref_model.num_parameters
                cand_params = cand_model.num_parameters() if callable(cand_model.num_parameters) else cand_model.num_parameters
                
                # Convert to numbers
                ref_params = float(ref_params)
                cand_params = float(cand_params)
                
                if ref_params > 0 and cand_params > 0:
                    ratio = max(ref_params, cand_params) / min(ref_params, cand_params)
                    if ratio > cfg.similar_size_ratio:
                        return False, f"Model size ratio {ratio:.2f} exceeds limit {cfg.similar_size_ratio}"
            except (TypeError, ValueError):
                # Could not get numeric parameter counts
                pass
        
        # Check architecture compatibility if possible
        if hasattr(ref_model, 'config') and hasattr(cand_model, 'config'):
            ref_arch = getattr(ref_model.config, 'model_type', None)
            cand_arch = getattr(cand_model.config, 'model_type', None)
            
            if ref_arch and cand_arch and ref_arch != cand_arch:
                return False, f"Architecture mismatch: {ref_arch} vs {cand_arch}"
        
        return True, "Models compatible"
        
    except Exception as e:
        logger.warning(f"Could not validate model compatibility: {e}")
        return True, "Compatibility check skipped"

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage with mock models and scoring
    
    def mock_score_fn(ref_model, cand_model, prompt, K=32):
        """Mock scoring function for testing"""
        # In practice, this would compute actual model differences
        return np.random.normal(0.01, 0.05)  # Small positive bias
    
    def mock_prompt_generator():
        """Mock prompt generator for testing"""
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing",
            "Write a poem about nature",
            "Solve this equation: 2x + 3 = 7"
        ]
        return np.random.choice(prompts)
    
    # Configure and run test
    config = DiffDecisionConfig(
        n_min=20,
        n_max=100,
        rel_margin_target=0.05,
        method="eb"
    )
    
    verifier = DifferenceVerifier(
        mock_score_fn,
        mock_prompt_generator,
        config
    )
    
    # Mock models (in practice, these would be actual models)
    class MockModel:
        def __init__(self, name):
            self.name = name
    
    ref_model = MockModel("reference")
    cand_model = MockModel("candidate")
    
    # Run verification
    print("Running statistical difference test...")
    report = verifier.verify_difference(ref_model, cand_model, verbose=True)
    
    # Print results
    print(f"\nDecision: {report['results']['decision']}")
    print(f"Reason: {report['results']['reason']}")
    print(f"Samples used: {report['results']['n_used']}")
    print(f"Mean difference: {report['results']['mean']:.6f}")
    print(f"99% CI: {report['results']['ci_99']}")
    print(f"\nInterpretation: {report['interpretation']}")