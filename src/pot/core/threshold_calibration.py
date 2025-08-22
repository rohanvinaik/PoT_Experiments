"""
Threshold Calibration for Real Model Pairs
Calibrates decision thresholds based on actual model behavior to avoid UNDECIDED outcomes
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional, Callable
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for threshold calibration"""
    n_samples_same: int = 30
    n_samples_different: int = 30
    positions_per_prompt: int = 32
    use_cached_models: bool = True
    save_scores: bool = True
    confidence_level: float = 0.99
    
    # Percentiles for threshold setting
    same_model_percentile: float = 99.0  # For γ
    different_model_percentile: float = 10.0  # For δ*
    
    # Margin factors
    gamma_margin: float = 1.5  # Multiply same-model P99 by this for γ
    delta_margin: float = 0.8  # Multiply separation point by this for δ*


class ThresholdCalibrator:
    """Calibrate decision thresholds based on actual model pairs"""
    
    def __init__(self, scorer: Optional[Any] = None, prompt_generator: Optional[Callable] = None):
        self.scorer = scorer
        self.prompt_generator = prompt_generator or self._default_prompt_generator
        self.model_cache = {}
        self.tokenizer_cache = {}
        
    def _default_prompt_generator(self) -> str:
        """Default prompt generator for calibration"""
        import random
        prompts = [
            "The capital of France is",
            "To make a good pizza, you need",
            "The theory of relativity states that",
            "Machine learning is a field that",
            "The weather today is",
            "In the beginning, there was",
            "The most important thing in life is",
            "Scientists have discovered that",
            "The future of technology will",
            "According to recent studies,",
            "The best way to learn is",
            "Climate change is affecting"
        ]
        return random.choice(prompts)
    
    def load_model_cached(self, model_path: str):
        """Load model with caching"""
        if model_path not in self.model_cache:
            logger.info(f"Loading model: {model_path}")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Determine device and dtype
            if torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.float16
            elif torch.backends.mps.is_available():
                device_map = None
                torch_dtype = torch.float32
            else:
                device_map = None
                torch_dtype = torch.float32
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map
            )
            model.eval()
            
            # Move to MPS if available and not using device_map
            if torch.backends.mps.is_available() and device_map is None:
                model = model.to("mps")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            self.model_cache[model_path] = model
            self.tokenizer_cache[model_path] = tokenizer
            
        return self.model_cache[model_path], self.tokenizer_cache[model_path]
    
    def quick_calibration(self,
                         model_path: str,
                         config: Optional[CalibrationConfig] = None) -> Dict[str, float]:
        """Quick calibration for same-model baseline"""
        
        config = config or CalibrationConfig()
        logger.info(f"Quick calibration with {model_path}, n_samples={config.n_samples_same}")
        
        # Load model
        model, tokenizer = self.load_model_cached(model_path)
        
        # Use optimized scorer if available
        if self.scorer is None:
            from pot.scoring.optimized_scorer import FastScorer
            self.scorer = FastScorer(k=config.positions_per_prompt, top_k=100)
        
        # Score model against itself
        scores = []
        t_start = time.time()
        
        for i in range(config.n_samples_same):
            prompt = self.prompt_generator()
            
            if hasattr(self.scorer, 'score'):
                # FastScorer interface
                score = self.scorer.score(model, model, prompt, tokenizer)
            else:
                # Generic scorer interface
                score = self.scorer(model, model, prompt, tokenizer, K=config.positions_per_prompt)
            
            scores.append(score)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Processed {i+1}/{config.n_samples_same} samples")
        
        t_elapsed = time.time() - t_start
        logger.info(f"Calibration completed in {t_elapsed:.2f}s")
        
        # Compute statistics
        scores_array = np.array(scores)
        abs_scores = np.abs(scores_array)
        
        # Add small epsilon to avoid zero thresholds
        abs_scores = abs_scores + 1e-6
        
        # Key percentiles for threshold setting
        p95 = float(np.percentile(abs_scores, 95))
        p99 = float(np.percentile(abs_scores, config.same_model_percentile))
        
        calibration = {
            "model": model_path,
            "n_samples": config.n_samples_same,
            "same_model_mean": float(np.mean(scores_array)),
            "same_model_std": float(np.std(scores_array)),
            "same_model_median": float(np.median(abs_scores)),
            "same_model_p95": p95,
            "same_model_p99": p99,
            "same_model_max": float(np.max(abs_scores)),
            
            # Recommended thresholds based on same-model distribution
            "recommended_gamma": p99 * config.gamma_margin,
            "recommended_delta_star_conservative": p99 * 10,
            "recommended_delta_star_moderate": p99 * 5,
            "recommended_delta_star_aggressive": p99 * 3,
            
            # Expected false positive rates
            "expected_fp_rate_at_gamma": 1.0 - config.same_model_percentile / 100,
            
            # Raw scores for analysis
            "raw_scores": scores if config.save_scores else None
        }
        
        logger.info(f"Same-model calibration: mean={calibration['same_model_mean']:.6f}, "
                   f"std={calibration['same_model_std']:.6f}, "
                   f"p99={p99:.6f}")
        logger.info(f"Recommended γ={calibration['recommended_gamma']:.6f}")
        
        return calibration
    
    def calibrate_pair(self,
                      ref_path: str,
                      cand_path: str,
                      config: Optional[CalibrationConfig] = None) -> Dict[str, float]:
        """Calibrate a specific model pair"""
        
        config = config or CalibrationConfig()
        logger.info(f"Calibrating pair: {ref_path} vs {cand_path}")
        
        # Load models
        ref_model, ref_tokenizer = self.load_model_cached(ref_path)
        cand_model, cand_tokenizer = self.load_model_cached(cand_path)
        
        # Use ref tokenizer for both
        tokenizer = ref_tokenizer
        
        # Use optimized scorer if available
        if self.scorer is None:
            from pot.scoring.optimized_scorer import FastScorer
            self.scorer = FastScorer(k=config.positions_per_prompt, top_k=100)
        
        # Score pairs
        scores = []
        t_start = time.time()
        
        for i in range(config.n_samples_different):
            prompt = self.prompt_generator()
            
            if hasattr(self.scorer, 'score'):
                # FastScorer interface
                score = self.scorer.score(ref_model, cand_model, prompt, tokenizer)
            else:
                # Generic scorer interface
                score = self.scorer(ref_model, cand_model, prompt, tokenizer, K=config.positions_per_prompt)
            
            scores.append(score)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Processed {i+1}/{config.n_samples_different} samples")
        
        t_elapsed = time.time() - t_start
        logger.info(f"Pair calibration completed in {t_elapsed:.2f}s")
        
        # Compute statistics
        scores_array = np.array(scores)
        abs_scores = np.abs(scores_array)
        
        return {
            "ref_model": ref_path,
            "cand_model": cand_path,
            "n_samples": config.n_samples_different,
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "median": float(np.median(scores_array)),
            "abs_mean": float(np.mean(abs_scores)),
            "abs_median": float(np.median(abs_scores)),
            "p5": float(np.percentile(abs_scores, 5)),
            "p10": float(np.percentile(abs_scores, config.different_model_percentile)),
            "p25": float(np.percentile(abs_scores, 25)),
            "min": float(np.min(abs_scores)),
            "max": float(np.max(abs_scores)),
            "raw_scores": scores if config.save_scores else None
        }
    
    def full_calibration(self,
                        same_model_paths: List[str],
                        different_model_pairs: List[Tuple[str, str]],
                        config: Optional[CalibrationConfig] = None) -> Dict[str, Any]:
        """Full calibration with multiple model pairs"""
        
        config = config or CalibrationConfig()
        logger.info(f"Starting full calibration: {len(same_model_paths)} same-model, "
                   f"{len(different_model_pairs)} different-model pairs")
        
        # Same-model calibration
        same_model_results = []
        all_same_scores = []
        
        for model_path in same_model_paths:
            calib = self.quick_calibration(model_path, config)
            same_model_results.append(calib)
            if calib.get("raw_scores"):
                all_same_scores.extend(np.abs(calib["raw_scores"]))
        
        # Different-model calibration
        different_results = []
        all_different_scores = []
        
        for ref_path, cand_path in different_model_pairs:
            calib = self.calibrate_pair(ref_path, cand_path, config)
            different_results.append(calib)
            if calib.get("raw_scores"):
                all_different_scores.extend(np.abs(calib["raw_scores"]))
        
        # Compute aggregate statistics
        same_array = np.array(all_same_scores) if all_same_scores else np.array([0])
        diff_array = np.array(all_different_scores) if all_different_scores else np.array([1])
        
        # Find optimal separation thresholds
        same_p99 = np.percentile(same_array, config.same_model_percentile)
        diff_p10 = np.percentile(diff_array, config.different_model_percentile) if len(diff_array) > 0 else same_p99 * 10
        
        # Compute overlap region
        overlap_start = np.percentile(same_array, 95)
        overlap_end = np.percentile(diff_array, 5) if len(diff_array) > 0 else overlap_start * 10
        
        calibration = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "n_samples_same": config.n_samples_same,
                "n_samples_different": config.n_samples_different,
                "positions_per_prompt": config.positions_per_prompt,
                "confidence_level": config.confidence_level
            },
            "same_model": {
                "n_models": len(same_model_paths),
                "n_total_samples": len(all_same_scores),
                "mean": float(np.mean(same_array)),
                "std": float(np.std(same_array)),
                "median": float(np.median(same_array)),
                "p95": float(np.percentile(same_array, 95)),
                "p99": float(same_p99),
                "max": float(np.max(same_array))
            },
            "different_model": {
                "n_pairs": len(different_model_pairs),
                "n_total_samples": len(all_different_scores),
                "mean": float(np.mean(diff_array)) if len(diff_array) > 0 else None,
                "std": float(np.std(diff_array)) if len(diff_array) > 0 else None,
                "median": float(np.median(diff_array)) if len(diff_array) > 0 else None,
                "p5": float(np.percentile(diff_array, 5)) if len(diff_array) > 0 else None,
                "p10": float(diff_p10) if len(diff_array) > 0 else None,
                "p25": float(np.percentile(diff_array, 25)) if len(diff_array) > 0 else None,
                "min": float(np.min(diff_array)) if len(diff_array) > 0 else None
            },
            "separation": {
                "overlap_region": [float(overlap_start), float(overlap_end)],
                "separation_gap": float(overlap_end - overlap_start) if overlap_end > overlap_start else 0,
                "separation_ratio": float(overlap_end / overlap_start) if overlap_start > 0 else float('inf')
            },
            "optimal_thresholds": self._compute_optimal_thresholds(
                same_p99, diff_p10, overlap_start, overlap_end, config
            ),
            "detailed_results": {
                "same_model": same_model_results if config.save_scores else None,
                "different_pairs": different_results if config.save_scores else None
            }
        }
        
        # Add recommendations
        calibration["recommendations"] = self._generate_recommendations(
            calibration["optimal_thresholds"],
            calibration["separation"]["separation_ratio"]
        )
        
        return calibration
    
    def _compute_optimal_thresholds(self, 
                                   same_p99: float,
                                   diff_p10: float,
                                   overlap_start: float,
                                   overlap_end: float,
                                   config: CalibrationConfig) -> Dict[str, float]:
        """Compute optimal thresholds based on distributions"""
        
        # Ensure minimum thresholds to avoid zero values
        same_p99 = max(same_p99, 0.01)
        diff_p10 = max(diff_p10, 0.1)
        overlap_start = max(overlap_start, 0.01)
        overlap_end = max(overlap_end, 0.1)
        
        # γ (gamma) for SAME decision: Should accept most same-model scores
        gamma = max(same_p99 * config.gamma_margin, 0.01)
        
        # δ* (delta_star) for DIFFERENT decision: Should separate distributions
        if overlap_end > overlap_start * 2:
            # Good separation exists
            delta_star = max((overlap_start + overlap_end) / 2 * config.delta_margin, 0.05)
        else:
            # Poor separation - use conservative threshold
            delta_star = max(same_p99 * 5, 0.1)
        
        # η (eta) for precision requirement
        eta = 0.5  # Half-width should be at most 50% of gamma for SAME
        
        # ε_diff for relative margin of error
        if diff_p10 > 0:
            epsilon_diff = min(0.3, 2 * same_p99 / diff_p10)  # Allow 30% relative error
        else:
            epsilon_diff = 0.3
        
        return {
            "gamma": float(gamma),
            "delta_star": float(delta_star),
            "eta": float(eta),
            "epsilon_diff": float(epsilon_diff),
            "epsilon_same": float(eta * gamma),  # Absolute precision for SAME
            
            # Additional context
            "expected_fp_rate": float(1 - config.same_model_percentile / 100),
            "expected_fn_rate": float(config.different_model_percentile / 100),
            "confidence_level": config.confidence_level
        }
    
    def _generate_recommendations(self, 
                                 thresholds: Dict[str, float],
                                 separation_ratio: float) -> Dict[str, Any]:
        """Generate configuration recommendations based on calibration"""
        
        if separation_ratio > 3:
            # Excellent separation
            return {
                "assessment": "excellent_separation",
                "mode": "quick_gate",
                "n_min": 20,
                "n_max": 100,
                "positions_per_prompt": 32,
                "expected_samples_same": 30,
                "expected_samples_different": 25,
                "confidence": 0.975,
                "use_thresholds": thresholds,
                "notes": "Models show clear separation - quick decisions possible"
            }
        elif separation_ratio > 2:
            # Good separation
            return {
                "assessment": "good_separation",
                "mode": "standard",
                "n_min": 30,
                "n_max": 200,
                "positions_per_prompt": 32,
                "expected_samples_same": 50,
                "expected_samples_different": 40,
                "confidence": 0.99,
                "use_thresholds": thresholds,
                "notes": "Models show good separation - standard testing sufficient"
            }
        elif separation_ratio > 1.5:
            # Moderate separation
            return {
                "assessment": "moderate_separation",
                "mode": "audit_grade",
                "n_min": 50,
                "n_max": 400,
                "positions_per_prompt": 64,
                "expected_samples_same": 80,
                "expected_samples_different": 70,
                "confidence": 0.99,
                "use_thresholds": thresholds,
                "notes": "Models show moderate overlap - more samples needed for confidence"
            }
        else:
            # Poor separation
            return {
                "assessment": "poor_separation",
                "mode": "high_sensitivity",
                "n_min": 100,
                "n_max": 800,
                "positions_per_prompt": 128,
                "expected_samples_same": 150,
                "expected_samples_different": 120,
                "confidence": 0.995,
                "use_thresholds": {
                    **thresholds,
                    "gamma": thresholds["gamma"] * 0.8,  # Tighter SAME threshold
                    "delta_star": thresholds["delta_star"] * 1.2  # Looser DIFFERENT threshold
                },
                "notes": "Models show significant overlap - consider alternative scoring methods",
                "alternative_suggestions": [
                    "Use different scoring metric (e.g., KL divergence)",
                    "Increase positions per prompt to 256",
                    "Use ensemble of scoring methods",
                    "Consider model-specific calibration"
                ]
            }
    
    def save_calibration(self, calibration: Dict[str, Any], filepath: str):
        """Save calibration to file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Remove raw scores if present to reduce file size
        if "detailed_results" in calibration:
            for results in [calibration["detailed_results"].get("same_model", []),
                          calibration["detailed_results"].get("different_pairs", [])]:
                if results:
                    for result in results:
                        if "raw_scores" in result:
                            result["raw_scores"] = None
        
        with open(filepath, 'w') as f:
            json.dump(calibration, f, indent=2)
        logger.info(f"Calibration saved to {filepath}")
    
    def load_calibration(self, filepath: str) -> Dict[str, Any]:
        """Load calibration from file"""
        with open(filepath, 'r') as f:
            calibration = json.load(f)
        logger.info(f"Calibration loaded from {filepath}")
        return calibration
    
    def apply_calibration(self, config: Any, calibration: Dict[str, Any]) -> Any:
        """Apply calibration to a configuration object"""
        
        thresholds = calibration.get("optimal_thresholds", {})
        recommendations = calibration.get("recommendations", {})
        
        # Update thresholds
        if hasattr(config, 'gamma'):
            config.gamma = thresholds.get("gamma", config.gamma)
        if hasattr(config, 'delta_star'):
            config.delta_star = thresholds.get("delta_star", config.delta_star)
        if hasattr(config, 'eta'):
            config.eta = thresholds.get("eta", config.eta)
        if hasattr(config, 'epsilon_diff'):
            config.epsilon_diff = thresholds.get("epsilon_diff", config.epsilon_diff)
        
        # Update sampling parameters
        if hasattr(config, 'n_min'):
            config.n_min = recommendations.get("n_min", config.n_min)
        if hasattr(config, 'n_max'):
            config.n_max = recommendations.get("n_max", config.n_max)
        if hasattr(config, 'positions_per_prompt'):
            config.positions_per_prompt = recommendations.get("positions_per_prompt", config.positions_per_prompt)
        
        logger.info(f"Applied calibration: γ={config.gamma:.6f}, δ*={config.delta_star:.6f}")
        
        return config


class AutoCalibrator:
    """Automatic calibration for common model pairs"""
    
    @staticmethod
    def calibrate_gpt2_family() -> Dict[str, Any]:
        """Calibrate for GPT-2 family models"""
        
        calibrator = ThresholdCalibrator()
        
        # Quick calibration for GPT-2 vs itself
        same_model_paths = ["gpt2"]
        different_model_pairs = [("gpt2", "distilgpt2")]
        
        config = CalibrationConfig(
            n_samples_same=50,
            n_samples_different=50,
            positions_per_prompt=32
        )
        
        return calibrator.full_calibration(
            same_model_paths,
            different_model_pairs,
            config
        )
    
    @staticmethod
    def quick_auto_calibrate(model_name: str = "gpt2") -> Dict[str, float]:
        """Quick automatic calibration for a single model"""
        
        calibrator = ThresholdCalibrator()
        config = CalibrationConfig(n_samples_same=30)
        
        return calibrator.quick_calibration(model_name, config)