"""
Enhanced Difference Verifier with Mode Support

This module provides the main verification orchestrator that uses the enhanced
statistical difference testing framework with separate SAME/DIFFERENT decision rules
and support for multiple testing modes.
"""

import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import asdict

from .diff_decision import (
    DiffDecisionConfig, 
    TestingMode, 
    EnhancedSequentialTester,
    create_enhanced_verifier as create_verifier_base
)
from .calibration import load_calibration, CalibrationResult

logger = logging.getLogger(__name__)

class EnhancedDifferenceVerifier:
    """Main verifier with mode support and proper decision rules"""
    
    def __init__(self,
                 score_fn: Callable,
                 prompt_generator: Callable,
                 config: DiffDecisionConfig,
                 verbose: bool = True):
        """
        Initialize enhanced verifier.
        
        Args:
            score_fn: Function(ref_model, cand_model, prompt, K) -> score
            prompt_generator: Function() -> prompt string
            config: Configuration with mode and decision parameters
            verbose: Whether to log progress
        """
        self.score_fn = score_fn
        self.prompt_generator = prompt_generator
        self.config = config
        self.verbose = verbose
        self.tester = EnhancedSequentialTester(config)
        
    def verify_difference(self,
                         ref_model: Any,
                         cand_model: Any,
                         output_dir: Optional[Path] = None,
                         save_prompts: bool = False) -> Dict[str, Any]:
        """
        Run verification with enhanced decision rules.
        
        Args:
            ref_model: Reference model
            cand_model: Candidate model to compare
            output_dir: Optional directory to save results
            save_prompts: Whether to save prompts used
            
        Returns:
            Comprehensive report with decision and diagnostics
        """
        
        start_time = time.perf_counter()
        
        if self.verbose:
            logger.info(f"Starting {self.config.mode.value} verification")
            logger.info(f"Config: confidence={self.config.confidence}, "
                       f"γ={self.config.gamma:.6f}, δ*={self.config.delta_star:.6f}, "
                       f"ε_diff={self.config.epsilon_diff:.3f}")
        
        # Track metrics
        all_prompts = []
        score_times = []
        batch_times = []
        
        # Sampling loop
        while self.tester.n < self.config.n_max:
            batch_start = time.perf_counter()
            
            # Generate batch
            batch_prompts = [self.prompt_generator() 
                           for _ in range(self.config.batch_size)]
            all_prompts.extend(batch_prompts)
            
            # Score each prompt
            for prompt in batch_prompts:
                t_score_start = time.perf_counter()
                
                score = self.score_fn(
                    ref_model,
                    cand_model,
                    prompt,
                    K=self.config.positions_per_prompt
                )
                
                score_times.append(time.perf_counter() - t_score_start)
                self.tester.update(score)
                
                # Progress logging
                if self.verbose and self.tester.n % 20 == 0:
                    (ci_lo, ci_hi), hw = self.tester.compute_ci()
                    logger.info(f"n={self.tester.n}: mean={self.tester.mean:.6f}, "
                              f"CI=[{ci_lo:.6f}, {ci_hi:.6f}], hw={hw:.6f}")
            
            batch_times.append(time.perf_counter() - batch_start)
            
            # Check stopping
            if self.tester.n >= self.config.n_min:
                should_stop, info = self.tester.should_stop()
                if should_stop:
                    return self._build_report(
                        info,
                        all_prompts[:self.tester.n] if save_prompts else None,
                        time.perf_counter() - start_time,
                        score_times,
                        batch_times,
                        output_dir
                    )
        
        # Reached n_max without decision
        _, info = self.tester.should_stop()
        return self._build_report(
            info or {"decision": "UNDECIDED", "reason": "Maximum samples reached"},
            all_prompts[:self.tester.n] if save_prompts else None,
            time.perf_counter() - start_time,
            score_times,
            batch_times,
            output_dir
        )
    
    def _build_report(self,
                     info: Dict[str, Any],
                     prompts: Optional[List[str]],
                     total_time: float,
                     score_times: List[float],
                     batch_times: List[float],
                     output_dir: Optional[Path]) -> Dict[str, Any]:
        """Build comprehensive report with diagnostics"""
        
        # Get final CI
        (ci_lo, ci_hi), hw = self.tester.compute_ci()
        
        report = {
            "verifier": "enhanced_stat_diff_v2",
            "mode": self.config.mode.value,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            
            "config": {
                "mode": self.config.mode.value,
                "confidence": self.config.confidence,
                "gamma": self.config.gamma,
                "delta_star": self.config.delta_star,
                "epsilon_diff": self.config.epsilon_diff,
                "eta": self.config.eta,
                "n_min": self.config.n_min,
                "n_max": self.config.n_max,
                "K": self.config.positions_per_prompt,
                "batch_size": self.config.batch_size,
                "ci_method": self.config.ci_method,
                "clip_range": [self.config.score_clip_low, self.config.score_clip_high]
            },
            
            "results": {
                "decision": info["decision"],
                "reason": info.get("reason", ""),
                "n_used": self.tester.n,
                "n_eff": self.tester.n * self.config.positions_per_prompt,
                "mean": float(self.tester.mean),
                "variance": float(self.tester.variance),
                "std_dev": float(self.tester.std_dev),
                "ci": [float(ci_lo), float(ci_hi)],
                "half_width": float(hw),
                "gamma": info.get("gamma", self.config.gamma),
                "delta_star": info.get("delta_star", self.config.delta_star),
                "rme": info.get("rme"),
                "direction": info.get("direction")
            },
            
            "scores": self._compute_score_stats(),
            
            "timing": {
                "total_time_sec": total_time,
                "avg_score_time_sec": float(np.mean(score_times)) if score_times else 0,
                "total_score_time_sec": sum(score_times),
                "avg_batch_time_sec": float(np.mean(batch_times)) if batch_times else 0,
                "scores_per_second": len(score_times) / total_time if total_time > 0 else 0
            },
            
            "diagnostics": info.get("diagnostics", {}),
            "suggestions": info.get("suggestions", [])
        }
        
        # Add interpretation
        report["interpretation"] = self._interpret_decision(info["decision"], report)
        
        # Add next steps
        report["next_steps"] = self._get_next_steps(info["decision"], report)
        
        # Save if output directory provided
        if output_dir:
            self._save_results(report, prompts, output_dir)
        
        return report
    
    def _compute_score_stats(self) -> Dict[str, Any]:
        """Compute detailed score statistics"""
        if not self.tester.raw_scores:
            return {}
        
        raw = np.array(self.tester.raw_scores)
        clipped = np.array(self.tester.clipped_scores)
        
        return {
            "raw": {
                "mean": float(np.mean(raw)),
                "std": float(np.std(raw)),
                "min": float(np.min(raw)),
                "max": float(np.max(raw)),
                "percentiles": {
                    "p5": float(np.percentile(raw, 5)),
                    "p25": float(np.percentile(raw, 25)),
                    "p50": float(np.percentile(raw, 50)),
                    "p75": float(np.percentile(raw, 75)),
                    "p95": float(np.percentile(raw, 95))
                }
            },
            "clipped": {
                "mean": float(np.mean(clipped)),
                "std": float(np.std(clipped)),
                "min": float(np.min(clipped)),
                "max": float(np.max(clipped))
            }
        }
    
    def _interpret_decision(self, decision: str, report: Dict[str, Any]) -> str:
        """Provide interpretation of the decision"""
        
        results = report["results"]
        config = report["config"]
        
        if decision == "SAME":
            return (f"Models are statistically equivalent. "
                   f"The mean difference {results['mean']:.6f} with "
                   f"{config['confidence']*100:.0f}% CI [{results['ci'][0]:.6f}, {results['ci'][1]:.6f}] "
                   f"is entirely within the equivalence band ±{config['gamma']:.6f}, "
                   f"and precision requirement (half_width ≤ {config['eta']*config['gamma']:.6f}) is met. "
                   f"This indicates no material difference in behavior.")
        
        elif decision == "DIFFERENT":
            direction = results.get("direction", "different")
            return (f"Models are statistically different ({direction}). "
                   f"The mean difference {results['mean']:.6f} has "
                   f"effect size exceeding the threshold {config['delta_star']:.6f} "
                   f"with sufficient precision (RME={results.get('rme', 0):.3f} ≤ {config['epsilon_diff']:.3f}). "
                   f"This indicates a material behavioral difference.")
        
        elif decision == "IDENTICAL":
            return (f"Models appear identical. "
                   f"The mean difference {results['mean']:.6f} and CI width {results['half_width']:.6f} "
                   f"are both below the early stopping threshold {config.get('early_stop_threshold', 1e-6):.6f}. "
                   f"This suggests the same model or numerical precision differences only.")
        
        else:  # UNDECIDED
            diag = report.get("diagnostics", {})
            n_more = config['n_max'] * 2 - results['n_used']
            k_more = config['K'] * 2
            
            msg = (f"Unable to make a definitive decision with current data after {results['n_used']} samples. ")
            
            # Add specific diagnostic info
            if diag:
                same_check = diag.get("same_check", {})
                diff_check = diag.get("different_check", {})
                
                if not same_check.get("ci_within_band"):
                    msg += f"CI extends outside equivalence band ±{config['gamma']:.6f}. "
                elif not same_check.get("precision_met"):
                    msg += f"Precision requirement not met for SAME (need half_width ≤ {config['eta']*config['gamma']:.6f}). "
                
                if not diff_check.get("effect_size_met"):
                    msg += f"Effect size {results['mean']:.6f} below threshold {config['delta_star']:.6f}. "
                elif diff_check.get("rme", float('inf')) > config['epsilon_diff']:
                    msg += f"RME {diff_check.get('rme', 0):.3f} exceeds target {config['epsilon_diff']:.3f}. "
            
            msg += f"Consider: (1) increasing samples to ~{n_more}, (2) using K={k_more} positions, "
            msg += "(3) checking model loading, or (4) trying a different scoring metric."
            
            return msg
    
    def _get_next_steps(self, decision: str, report: Dict[str, Any]) -> List[str]:
        """Get recommended next steps based on decision"""
        
        if decision == "SAME":
            return [
                "Models verified as equivalent within tolerance",
                "Consider tightening γ if stricter equivalence needed",
                "May proceed with model deployment/replacement",
                "Document equivalence for audit trail"
            ]
        
        elif decision == "DIFFERENT":
            return [
                "Investigate source of behavioral difference",
                "Run secondary verification methods (fuzzy hash, provenance)",
                "Check for unauthorized modifications or attacks",
                "Do not use models interchangeably without analysis"
            ]
        
        elif decision == "IDENTICAL":
            return [
                "Verify file hashes to confirm identical models",
                "Check for numerical precision settings",
                "No further statistical testing needed"
            ]
        
        else:  # UNDECIDED
            suggestions = report.get("suggestions", [])
            if suggestions:
                return suggestions[:5]  # Top 5 suggestions
            else:
                return [
                    f"Increase n_max beyond {report['config']['n_max']}",
                    f"Increase K (positions) beyond {report['config']['K']}",
                    "Try different scoring method",
                    "Check if models are near-clones",
                    "Consider relaxing precision requirements"
                ]
    
    def _save_results(self, 
                     report: Dict[str, Any],
                     prompts: Optional[List[str]],
                     output_dir: Path):
        """Save results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        
        # Save main report
        report_file = output_dir / f"{self.config.mode.value}_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {report_file}")
        
        # Save raw scores for analysis
        if self.tester.raw_scores:
            scores_file = output_dir / f"scores_{timestamp}.json"
            with open(scores_file, 'w') as f:
                json.dump({
                    "raw": self.tester.raw_scores,
                    "clipped": self.tester.clipped_scores,
                    "stats": report["scores"]
                }, f, indent=2)
        
        # Save prompts if requested
        if prompts:
            prompts_file = output_dir / f"prompts_{timestamp}.json"
            with open(prompts_file, 'w') as f:
                json.dump({"prompts": prompts}, f, indent=2)
        
        # Save summary for quick viewing
        summary_file = output_dir / f"summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(self._generate_summary(report))
    
    def _generate_summary(self, report: Dict[str, Any]) -> str:
        """Generate human-readable summary"""
        r = report["results"]
        c = report["config"]
        t = report["timing"]
        
        summary = f"""
Enhanced Difference Verification Summary
========================================
Mode: {c['mode']}
Decision: {r['decision']}
Reason: {r['reason']}

Configuration:
--------------
Confidence: {c['confidence']*100:.0f}%
γ (SAME band): {c['gamma']:.6f}
δ* (DIFFERENT threshold): {c['delta_star']:.6f}
ε_diff (RME target): {c['epsilon_diff']:.3f}

Results:
--------
Samples used: {r['n_used']} (effective: {r['n_eff']})
Mean difference: {r['mean']:.6f}
Standard deviation: {r['std_dev']:.6f}
CI: [{r['ci'][0]:.6f}, {r['ci'][1]:.6f}]
Half-width: {r['half_width']:.6f}
"""
        if r.get('rme') is not None:
            summary += f"RME: {r['rme']:.3f}\n"
        
        summary += f"""
Performance:
------------
Total time: {t['total_time_sec']:.2f}s
Scores per second: {t['scores_per_second']:.1f}

Interpretation:
---------------
{report['interpretation']}

Next Steps:
-----------
"""
        for step in report['next_steps'][:3]:
            summary += f"• {step}\n"
        
        return summary


def create_enhanced_verifier(score_fn: Callable,
                            prompt_generator: Callable,
                            mode: TestingMode = TestingMode.AUDIT_GRADE,
                            calibration_file: Optional[str] = None,
                            **config_overrides) -> EnhancedDifferenceVerifier:
    """
    Create an enhanced verifier with optional calibration.
    
    Args:
        score_fn: Scoring function
        prompt_generator: Prompt generation function
        mode: Testing mode (QUICK_GATE or AUDIT_GRADE)
        calibration_file: Optional calibration file to load
        **config_overrides: Additional config parameters
        
    Returns:
        Configured EnhancedDifferenceVerifier
    """
    # Load calibration if provided
    if calibration_file:
        calib = load_calibration(calibration_file)
        config = DiffDecisionConfig(
            mode=mode,
            use_calibration=True,
            same_model_p95=calib.gamma,
            near_clone_p5=calib.near_clone_stats["p5"] if calib.near_clone_stats else None,
            **config_overrides
        )
    else:
        config = DiffDecisionConfig(mode=mode, **config_overrides)
    
    return EnhancedDifferenceVerifier(score_fn, prompt_generator, config)


def run_quick_verification(ref_model: Any,
                          cand_model: Any,
                          score_fn: Callable,
                          prompt_generator: Callable,
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a quick gate verification.
    
    Args:
        ref_model: Reference model
        cand_model: Candidate model
        score_fn: Scoring function
        prompt_generator: Prompt generator
        output_dir: Optional output directory
        
    Returns:
        Verification report
    """
    verifier = create_enhanced_verifier(
        score_fn,
        prompt_generator,
        mode=TestingMode.QUICK_GATE
    )
    
    return verifier.verify_difference(
        ref_model,
        cand_model,
        Path(output_dir) if output_dir else None
    )


def run_audit_verification(ref_model: Any,
                          cand_model: Any,
                          score_fn: Callable,
                          prompt_generator: Callable,
                          calibration_file: Optional[str] = None,
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run an audit-grade verification.
    
    Args:
        ref_model: Reference model
        cand_model: Candidate model
        score_fn: Scoring function
        prompt_generator: Prompt generator
        calibration_file: Optional calibration file
        output_dir: Optional output directory
        
    Returns:
        Verification report
    """
    verifier = create_enhanced_verifier(
        score_fn,
        prompt_generator,
        mode=TestingMode.AUDIT_GRADE,
        calibration_file=calibration_file
    )
    
    return verifier.verify_difference(
        ref_model,
        cand_model,
        Path(output_dir) if output_dir else None,
        save_prompts=True  # Save prompts for audit
    )