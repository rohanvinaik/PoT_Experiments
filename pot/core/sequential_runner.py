import time
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

from .boundaries import CSState, eb_radius, decide_one_sided
from .prf import prf_derive_key
from ..challenges.prompt_generator import DeterministicPromptGenerator
from ..scoring.teacher_forced import TeacherForcedScorer, ScoringConfig

logger = logging.getLogger(__name__)

@dataclass
class TestMetrics:
    """Metrics for a sequential test run"""
    alpha: float
    beta: float
    n_used: int
    n_max: int
    statistic_mean: float
    statistic_var: float
    boundary_type: str  # "EB" or "MB"
    decision: str  # "accept_id", "reject_id", or "undecided"
    hypothesis: str  # "H0" (null/impostor) or "H1" (identity match)
    stopping_time: int
    
    # Time breakdown
    t_load: float  # Model loading time
    t_infer_total: float  # Total inference time
    t_per_query: float  # Average time per query
    t_setup: float  # Setup/initialization time
    t_scoring: float  # Scoring computation time
    t_total: float  # Total wall time
    
    # Additional statistics
    per_query_times: List[float]
    per_query_scores: List[float]
    confidence_intervals: List[Tuple[float, float]]
    
    def to_json_safe_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        result = {}
        for key, value in asdict(self).items():
            # Handle numpy types
            if isinstance(value, np.bool_):
                result[key] = bool(value)
            elif isinstance(value, np.integer):
                result[key] = int(value)
            elif isinstance(value, np.floating):
                result[key] = float(value)
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                # Recursively convert list/tuple elements
                result[key] = [
                    self._convert_value(v) for v in value
                ]
            else:
                result[key] = value
        return result
    
    def _convert_value(self, value: Any) -> Any:
        """Convert individual values to JSON-safe types"""
        if isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, tuple):
            return list(value)
        else:
            return value

class SequentialTestRunner:
    """Runs real sequential tests without mocks"""
    
    def __init__(self, 
                 reference_model_path: str,
                 scoring_config: ScoringConfig,
                 master_key: bytes,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.reference_model_path = reference_model_path
        self.scoring_config = scoring_config
        self.master_key = master_key
        self.device = device
        self.reference_model = None
        self.tokenizer = None
        self.prompt_generator = DeterministicPromptGenerator(master_key)
        self.scorer = TeacherForcedScorer(scoring_config)
        
    def load_models(self, candidate_model_path: str) -> Tuple[float, Any]:
        """Load reference and candidate models, return load time"""
        t_start = time.perf_counter()
        
        # Load reference model if not already loaded
        if self.reference_model is None:
            logger.info(f"Loading reference model from {self.reference_model_path}")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.reference_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            self.reference_model.eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.reference_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load candidate model
        logger.info(f"Loading candidate model from {candidate_model_path}")
        from transformers import AutoModelForCausalLM
        
        candidate_model = AutoModelForCausalLM.from_pretrained(
            candidate_model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        candidate_model.eval()
        
        t_load = time.perf_counter() - t_start
        logger.info(f"Models loaded in {t_load:.2f}s")
        
        return t_load, candidate_model
    
    def run_sequential_test(self,
                           candidate_model_path: str,
                           alpha: float = 0.01,
                           beta: float = 0.01,
                           tau: float = 0.05,
                           n_max: int = 512,
                           boundary_type: str = "EB",
                           namespace: str = "verification",
                           output_dir: Optional[str] = None) -> TestMetrics:
        """Run a real sequential test and return comprehensive metrics"""
        
        t_total_start = time.perf_counter()
        
        # Setup output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped subdirectory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = output_path / f"session_{timestamp}"
            session_dir.mkdir(exist_ok=True)
        else:
            session_dir = Path(".")
        
        # Load models
        t_load, candidate_model = self.load_models(candidate_model_path)
        
        # Initialize test state
        cs = CSState()
        per_query_times = []
        per_query_scores = []
        confidence_intervals = []
        
        # Setup timing
        t_setup_start = time.perf_counter()
        
        # Generate challenges
        prompts = self.prompt_generator.batch_generate(namespace, 0, n_max)
        prompt_types = ["factual"] * len(prompts)  # Can vary this
        
        t_setup = time.perf_counter() - t_setup_start
        
        # Run sequential test
        t_infer_start = time.perf_counter()
        decision = None
        hypothesis = None
        stopping_time = n_max
        
        logger.info(f"Starting sequential test with α={alpha}, β={beta}, τ={tau}, n_max={n_max}")
        
        for i in range(n_max):
            # Time individual query
            t_query_start = time.perf_counter()
            
            # Score this challenge
            result = self.scorer.score_models(
                self.reference_model,
                candidate_model,
                prompts[i],
                prompt_types[i],
                self.tokenizer
            )
            
            t_query = time.perf_counter() - t_query_start
            per_query_times.append(t_query)
            
            # Update statistics
            z = result.score
            per_query_scores.append(z)
            cs.update(z)
            
            # Compute confidence bounds
            if boundary_type == "EB":
                r_accept = eb_radius(cs.t, cs.var, alpha)
                r_reject = eb_radius(cs.t, cs.var, beta)
            else:
                # Mixture-Bernstein would go here
                raise NotImplementedError(f"Boundary type {boundary_type} not implemented")
            
            confidence_intervals.append((cs.mean - r_reject, cs.mean + r_accept))
            
            # Make decision
            if cs.mean + r_accept <= tau:
                decision = "accept_id"
                hypothesis = "H1"
                stopping_time = i + 1
                logger.info(f"Accepted identity at n={stopping_time}, mean={cs.mean:.4f}")
                break
            elif cs.mean - r_reject >= tau:
                decision = "reject_id"
                hypothesis = "H0"
                stopping_time = i + 1
                logger.info(f"Rejected (impostor) at n={stopping_time}, mean={cs.mean:.4f}")
                break
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"n={i+1}: mean={cs.mean:.4f}, "
                          f"CI=[{cs.mean - r_reject:.4f}, {cs.mean + r_accept:.4f}], "
                          f"τ={tau:.4f}")
        
        t_infer_total = time.perf_counter() - t_infer_start
        
        # If no decision made
        if decision is None:
            decision = "undecided"
            hypothesis = "inconclusive"
            logger.warning(f"No decision after n_max={n_max}, mean={cs.mean:.4f}")
        
        # Compute scoring time
        t_scoring = sum(per_query_times) - t_infer_total  # Approximate
        
        # Create metrics
        metrics = TestMetrics(
            alpha=alpha,
            beta=beta,
            n_used=stopping_time,
            n_max=n_max,
            statistic_mean=float(cs.mean),
            statistic_var=float(cs.var),
            boundary_type=boundary_type,
            decision=decision,
            hypothesis=hypothesis,
            stopping_time=stopping_time,
            t_load=t_load,
            t_infer_total=t_infer_total,
            t_per_query=t_infer_total / stopping_time if stopping_time > 0 else 0,
            t_setup=t_setup,
            t_scoring=t_scoring,
            t_total=time.perf_counter() - t_total_start,
            per_query_times=per_query_times[:stopping_time],
            per_query_scores=per_query_scores[:stopping_time],
            confidence_intervals=confidence_intervals[:stopping_time]
        )
        
        # Save results (NOT mocks)
        self._save_results(metrics, session_dir)
        
        return metrics
    
    def _save_results(self, metrics: TestMetrics, output_dir: Path):
        """Save test results to JSON files"""
        
        # Main results file
        results_file = output_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(metrics.to_json_safe_dict(), f, indent=2)
        logger.info(f"Saved results to {results_file}")
        
        # Summary file with key metrics only
        summary = {
            "timestamp": datetime.now().isoformat(),
            "decision": metrics.decision,
            "hypothesis": metrics.hypothesis,
            "alpha": metrics.alpha,
            "beta": metrics.beta,
            "n_used": metrics.n_used,
            "n_max": metrics.n_max,
            "statistic_mean": metrics.statistic_mean,
            "statistic_var": metrics.statistic_var,
            "stopping_time": metrics.stopping_time,
            "time_total_seconds": metrics.t_total,
            "time_per_query_ms": metrics.t_per_query * 1000
        }
        
        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Detailed timing breakdown
        timing = {
            "t_load": metrics.t_load,
            "t_setup": metrics.t_setup,
            "t_infer_total": metrics.t_infer_total,
            "t_scoring": metrics.t_scoring,
            "t_total": metrics.t_total,
            "t_per_query_avg": metrics.t_per_query,
            "per_query_times": metrics.per_query_times
        }
        
        timing_file = output_dir / "timing.json"
        with open(timing_file, 'w') as f:
            json.dump(timing, f, indent=2)