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

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DiffDecisionConfig:
    """Configuration for statistical difference testing"""
    
    # Confidence & precision targets
    alpha: float = 0.01              # 99% CI (two-sided)
    rel_margin_target: float = 0.05  # ε, e.g., 5% relative half-width
    min_effect_floor: float = 1e-4   # guard against divide-by-zero when mean≈0
    
    # Sampling plan
    n_min: int = 10                  # Minimum samples before stopping
    n_max: int = 200                 # Maximum samples
    batch_size: int = 4              # Challenges per batch
    positions_per_prompt: int = 32   # K positions for teacher-forced scoring
    
    # Confidence type
    method: Literal["eb", "t"] = "eb"  # Empirical-Bernstein or t-distribution
    clip_low: float = 0.0             # For EB: clip/normalize scores into [0,1]
    clip_high: float = 0.2            # Set based on observed score scale
    
    # Optional equivalence testing
    equivalence_band: Optional[float] = None  # γ for TOST-style "SAME" decision
    
    # Model size gate (optional)
    similar_size_ratio: float = 2.0   # Only use if max(params)/min(params) <= 2.0
    
    # Early stopping for identical models
    early_stop_threshold: float = 1e-6  # Stop early if mean < this after n_min
    identical_model_n_min: int = 20     # Min samples for identical model check
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DiffDecisionConfig':
        """Create config from dictionary"""
        return cls(**d)

# ============================================================================
# SEQUENTIAL TESTER
# ============================================================================

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
                 cfg: DiffDecisionConfig):
        """Initialize verifier
        
        Args:
            score_fn: Function(ref_model, cand_model, prompt, K) -> score
            prompt_generator: Function() -> prompt string
            cfg: Configuration for difference testing
        """
        self.score_fn = score_fn
        self.prompt_generator = prompt_generator
        self.cfg = cfg
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