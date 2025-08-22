"""
Calibration System for Statistical Difference Testing

This module provides automatic calibration of γ (equivalence band) and δ* (minimum 
effect size) parameters from pilot runs on same-model and near-clone comparisons.
"""

import numpy as np
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class CalibrationResult:
    """Results from calibration runs"""
    gamma: float  # 95th percentile of same-model |mean|
    delta_star: float  # Midpoint between same P95 and near-clone P5
    same_model_stats: Dict[str, float]
    near_clone_stats: Optional[Dict[str, float]]
    n_same_pairs: int
    n_near_clone_pairs: int
    calibration_time: float
    n_samples_per_pair: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationResult':
        """Create from dictionary"""
        return cls(**data)
    
    def get_config_recommendations(self) -> Dict[str, Dict[str, float]]:
        """Get recommended configurations for different modes"""
        return {
            "quick_gate": {
                "gamma": self.gamma * 1.5,  # More lenient for quick checks
                "delta_star": self.delta_star,
                "eta": 0.5  # Half-width precision factor
            },
            "audit_grade": {
                "gamma": self.gamma,
                "delta_star": self.delta_star,
                "eta": 0.3  # Tighter precision requirement
            },
            "conservative": {
                "gamma": self.gamma * 0.8,  # Stricter equivalence band
                "delta_star": self.delta_star * 1.2,  # Higher effect threshold
                "eta": 0.2  # Very tight precision
            }
        }

class ModelCalibrator:
    """Calibrate γ and δ* from pilot runs"""
    
    def __init__(self, 
                 scorer: Optional[Callable] = None,
                 prompt_generator: Optional[Callable] = None,
                 n_samples_per_pair: int = 50):
        """
        Initialize calibrator.
        
        Args:
            scorer: Function(ref_model, cand_model, prompt, K) -> score
            prompt_generator: Function() -> prompt string
            n_samples_per_pair: Number of samples for each model pair
        """
        self.scorer = scorer
        self.prompt_generator = prompt_generator
        self.n_samples = n_samples_per_pair
        self.calibration_cache = {}
        
    def run_same_model_calibration(self, 
                                   models: List[Any],
                                   model_names: Optional[List[str]] = None,
                                   n_runs_per_model: int = 5,
                                   use_mock: bool = False) -> Dict[str, float]:
        """
        Run same-model comparisons to calibrate γ.
        
        Args:
            models: List of model objects or paths
            model_names: Optional names for logging
            n_runs_per_model: Number of runs per model with different seeds
            use_mock: Use mock scoring for testing
            
        Returns:
            Dictionary of statistics from same-model comparisons
        """
        if model_names is None:
            model_names = [f"model_{i}" for i in range(len(models))]
            
        logger.info(f"Running same-model calibration on {len(models)} models")
        
        all_means = []
        all_scores = []
        
        for model_idx, model in enumerate(models):
            model_name = model_names[model_idx]
            logger.info(f"Calibrating {model_name}")
            
            # Run multiple same-model comparisons (different random seeds)
            for run in range(n_runs_per_model):
                scores = []
                
                # Set different seed for each run
                np.random.seed(42 + run)
                
                for _ in range(self.n_samples):
                    if use_mock:
                        # Mock scoring for testing
                        score = np.random.normal(0.0, 0.001)
                    else:
                        prompt = self.prompt_generator()
                        # Score model against itself
                        score = self.scorer(model, model, prompt, K=64)
                    
                    scores.append(score)
                    all_scores.append(score)
                
                run_mean = np.mean(scores)
                all_means.append(abs(run_mean))
                
                logger.debug(f"  Run {run+1}: mean = {run_mean:.6f}, std = {np.std(scores):.6f}")
        
        # Compute statistics
        stats = {
            "mean": float(np.mean(all_means)),
            "std": float(np.std(all_means)),
            "p50": float(np.percentile(all_means, 50)),
            "p75": float(np.percentile(all_means, 75)),
            "p90": float(np.percentile(all_means, 90)),
            "p95": float(np.percentile(all_means, 95)),
            "p99": float(np.percentile(all_means, 99)),
            "max": float(np.max(all_means)),
            "n_samples_total": len(all_scores),
            "score_mean": float(np.mean(all_scores)),
            "score_std": float(np.std(all_scores))
        }
        
        logger.info(f"Same-model stats: P95 = {stats['p95']:.6f}, Max = {stats['max']:.6f}")
        
        return stats
    
    def run_near_clone_calibration(self,
                                   model_pairs: List[Tuple[Any, Any]],
                                   pair_names: Optional[List[str]] = None,
                                   use_mock: bool = False) -> Dict[str, float]:
        """
        Run near-clone comparisons to calibrate δ*.
        
        Args:
            model_pairs: List of (reference, clone) model pairs
            pair_names: Optional names for logging
            use_mock: Use mock scoring for testing
            
        Returns:
            Dictionary of statistics from near-clone comparisons
        """
        if pair_names is None:
            pair_names = [f"pair_{i}" for i in range(len(model_pairs))]
            
        logger.info(f"Running near-clone calibration on {len(model_pairs)} pairs")
        
        all_means = []
        all_scores = []
        
        for pair_idx, (ref_model, clone_model) in enumerate(model_pairs):
            pair_name = pair_names[pair_idx]
            logger.info(f"Comparing {pair_name}")
            
            # Score differences
            scores = []
            np.random.seed(42 + pair_idx)  # Reproducible results
            
            for _ in range(self.n_samples):
                if use_mock:
                    # Mock scoring for testing (near-clones have moderate differences)
                    score = np.random.normal(0.08, 0.02)
                else:
                    prompt = self.prompt_generator()
                    score = self.scorer(ref_model, clone_model, prompt, K=64)
                
                scores.append(score)
                all_scores.append(score)
            
            pair_mean = np.mean(scores)
            all_means.append(abs(pair_mean))
            
            logger.info(f"  Pair mean = {pair_mean:.6f}, std = {np.std(scores):.6f}")
        
        # Compute statistics
        stats = {
            "mean": float(np.mean(all_means)),
            "std": float(np.std(all_means)),
            "p5": float(np.percentile(all_means, 5)),
            "p10": float(np.percentile(all_means, 10)),
            "p25": float(np.percentile(all_means, 25)),
            "p50": float(np.percentile(all_means, 50)),
            "p75": float(np.percentile(all_means, 75)),
            "p95": float(np.percentile(all_means, 95)),
            "min": float(np.min(all_means)),
            "max": float(np.max(all_means)),
            "n_samples_total": len(all_scores),
            "score_mean": float(np.mean(all_scores)),
            "score_std": float(np.std(all_scores))
        }
        
        logger.info(f"Near-clone stats: P5 = {stats['p5']:.6f}, P50 = {stats['p50']:.6f}")
        
        return stats
    
    def calibrate(self,
                 same_models: List[Any],
                 near_clone_pairs: Optional[List[Tuple[Any, Any]]] = None,
                 same_model_names: Optional[List[str]] = None,
                 pair_names: Optional[List[str]] = None,
                 output_file: Optional[str] = None,
                 use_mock: bool = False) -> CalibrationResult:
        """
        Run full calibration.
        
        Args:
            same_models: List of models for same-model calibration
            near_clone_pairs: Optional list of (ref, clone) pairs
            same_model_names: Optional names for same models
            pair_names: Optional names for pairs
            output_file: Optional file to save calibration results
            use_mock: Use mock scoring for testing
            
        Returns:
            CalibrationResult with recommended γ and δ* values
        """
        start_time = time.time()
        
        # Same-model calibration
        same_stats = self.run_same_model_calibration(
            same_models, 
            same_model_names,
            use_mock=use_mock
        )
        gamma = same_stats["p95"]
        
        # Near-clone calibration (optional)
        if near_clone_pairs:
            near_stats = self.run_near_clone_calibration(
                near_clone_pairs,
                pair_names,
                use_mock=use_mock
            )
            # Set δ* as midpoint between same P95 and near-clone P5
            delta_star = (same_stats["p95"] + near_stats["p5"]) / 2
            
            # Validate separation
            if near_stats["p5"] <= same_stats["p95"]:
                logger.warning(
                    f"Poor separation: near-clone P5 ({near_stats['p5']:.6f}) <= "
                    f"same-model P95 ({same_stats['p95']:.6f}). "
                    "Consider using different near-clone models."
                )
        else:
            # Conservative default: 10x the same-model P95
            near_stats = None
            delta_star = gamma * 10
            logger.warning(f"No near-clone pairs provided, using conservative δ* = {delta_star:.6f}")
        
        calibration_time = time.time() - start_time
        
        result = CalibrationResult(
            gamma=gamma,
            delta_star=delta_star,
            same_model_stats=same_stats,
            near_clone_stats=near_stats,
            n_same_pairs=len(same_models),
            n_near_clone_pairs=len(near_clone_pairs) if near_clone_pairs else 0,
            calibration_time=calibration_time,
            n_samples_per_pair=self.n_samples
        )
        
        # Save calibration results
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Calibration saved to {output_file}")
        
        # Print recommendations
        self._print_recommendations(result)
        
        return result
    
    def _print_recommendations(self, result: CalibrationResult):
        """Print calibration recommendations"""
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        print(f"Recommended γ (equivalence band): {result.gamma:.6f}")
        print(f"Recommended δ* (min effect size): {result.delta_star:.6f}")
        
        if result.near_clone_stats:
            separation = result.near_clone_stats["p5"] / result.same_model_stats["p95"]
            print(f"Separation factor: {separation:.2f}x")
            
            if separation < 2:
                print("⚠️  Warning: Poor separation between same and near-clone models")
            elif separation < 5:
                print("✓ Adequate separation between same and near-clone models")
            else:
                print("✓ Excellent separation between same and near-clone models")
        
        print("\nSuggested configurations:")
        configs = result.get_config_recommendations()
        for mode, params in configs.items():
            print(f"\n  {mode.upper()}:")
            for key, value in params.items():
                print(f"    {key}: {value:.6f}")
        
        print("\nUsage example:")
        print(f"""
from pot.core.diff_decision import DiffDecisionConfig, TestingMode
from pot.core.calibration import load_calibration

# Load calibration
calib = load_calibration("{result.gamma:.6f}_calibration.json")

# Use in config
config = DiffDecisionConfig(
    mode=TestingMode.AUDIT_GRADE,
    use_calibration=True,
    same_model_p95=calib.gamma,
    near_clone_p5=calib.near_clone_stats["p5"] if calib.near_clone_stats else None
)
""")
        print("="*60)

def load_calibration(filepath: str) -> CalibrationResult:
    """Load calibration from file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return CalibrationResult.from_dict(data)

def create_mock_calibrator() -> ModelCalibrator:
    """Create a calibrator with mock scoring for testing"""
    def mock_scorer(ref_model, cand_model, prompt, K=32):
        # Simulate scoring based on model comparison
        if ref_model == cand_model:
            return np.random.normal(0.0, 0.001)
        elif "clone" in str(cand_model):
            return np.random.normal(0.08, 0.02)
        else:
            return np.random.normal(0.15, 0.03)
    
    def mock_prompt_gen():
        prompts = ["Test prompt 1", "Test prompt 2", "Test prompt 3"]
        return np.random.choice(prompts)
    
    return ModelCalibrator(
        scorer=mock_scorer,
        prompt_generator=mock_prompt_gen,
        n_samples_per_pair=30
    )