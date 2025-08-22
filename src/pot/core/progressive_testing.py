"""
Progressive Testing Strategy for Efficient Model Verification

Implements a multi-stage testing approach that starts with quick checks
and escalates to more thorough testing only when needed.
"""

import logging
import time
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class TestingStage(Enum):
    QUICK_CHECK = "quick_check"
    STANDARD_TEST = "standard_test"
    DEEP_ANALYSIS = "deep_analysis"
    EXHAUSTIVE = "exhaustive"

@dataclass
class StageConfig:
    """Configuration for each testing stage"""
    stage: TestingStage
    n_samples: int
    k_positions: int
    batch_size: int
    confidence: float
    scoring_method: str
    timeout_seconds: Optional[float] = None
    gamma: Optional[float] = None
    delta_star: Optional[float] = None
    eta: Optional[float] = None
    epsilon_diff: Optional[float] = None

class ProgressiveVerifier:
    """Progressive testing that escalates through stages"""
    
    def __init__(self, base_config=None):
        self.base_config = base_config
        self.stages = self._define_stages()
        self.current_stage_idx = 0
        self.convergence_history = []
        
    def _define_stages(self) -> List[StageConfig]:
        """Define testing stages from quick to exhaustive"""
        return [
            # Stage 1: Quick check (5-10 seconds)
            StageConfig(
                stage=TestingStage.QUICK_CHECK,
                n_samples=10,
                k_positions=16,
                batch_size=5,
                confidence=0.95,
                scoring_method="delta_ce",
                timeout_seconds=10,
                gamma=0.40,  # Relaxed for quick check
                delta_star=0.15,  # Lower threshold for quick detection
                eta=0.6,
                epsilon_diff=0.30
            ),
            
            # Stage 2: Standard test (30-60 seconds)
            StageConfig(
                stage=TestingStage.STANDARD_TEST,
                n_samples=50,
                k_positions=32,
                batch_size=10,
                confidence=0.975,
                scoring_method="delta_ce",
                timeout_seconds=60,
                gamma=0.35,  # Standard calibrated values
                delta_star=0.20,
                eta=0.5,
                epsilon_diff=0.20
            ),
            
            # Stage 3: Deep analysis (2-5 minutes)
            StageConfig(
                stage=TestingStage.DEEP_ANALYSIS,
                n_samples=200,
                k_positions=64,
                batch_size=20,
                confidence=0.99,
                scoring_method="symmetric_kl",
                timeout_seconds=300,
                gamma=0.30,  # Tighter for deep analysis
                delta_star=0.25,
                eta=0.4,
                epsilon_diff=0.15
            ),
            
            # Stage 4: Exhaustive (5-10 minutes)
            StageConfig(
                stage=TestingStage.EXHAUSTIVE,
                n_samples=500,
                k_positions=128,
                batch_size=25,
                confidence=0.995,
                scoring_method="symmetric_kl",
                timeout_seconds=600,
                gamma=0.25,  # Strictest thresholds
                delta_star=0.30,
                eta=0.3,
                epsilon_diff=0.10
            )
        ]
    
    def should_escalate(self, 
                       current_result: Dict[str, Any],
                       stage: StageConfig) -> Tuple[bool, str]:
        """Determine if we should escalate to next stage"""
        
        decision = current_result.get("decision", "UNDECIDED")
        
        # Don't escalate if we have a clear decision with good confidence
        if decision in ["SAME", "DIFFERENT"]:
            rme = current_result.get("rme", 1.0)
            half_width = current_result.get("half_width", 1.0)
            mean = abs(current_result.get("mean", 0))
            
            # Check if decision is reliable enough
            if stage.stage == TestingStage.QUICK_CHECK:
                # Always confirm quick check results unless very clear
                if decision == "SAME" and mean < 0.05 and half_width < 0.02:
                    return False, "Clear self-consistency detected"
                elif decision == "DIFFERENT" and mean > 0.40 and rme < 0.05:
                    return False, "Clear difference detected"
                else:
                    return True, "Confirming quick check result"
                    
            elif stage.stage == TestingStage.STANDARD_TEST:
                if rme < 0.15:  # Good precision
                    return False, "Sufficient precision achieved"
                else:
                    return True, "Need better precision"
                    
            else:
                # Later stages - only escalate if really uncertain
                if rme < 0.10:
                    return False, "High confidence decision"
                elif decision == "DIFFERENT" and mean > stage.delta_star * 1.5:
                    return False, "Strong difference signal"
                    
        # UNDECIDED - check if it's worth escalating
        if decision == "UNDECIDED":
            mean = abs(current_result.get("mean", 0))
            ci_width = current_result.get("half_width", 1.0)
            
            # If mean is very small and CI is tight, likely same
            if mean < 0.01 and ci_width < 0.01:
                return False, "Likely identical, no need to escalate"
            
            # If we're in early stages, escalate
            if stage.stage in [TestingStage.QUICK_CHECK, TestingStage.STANDARD_TEST]:
                return True, "Undecided - escalating for clarity"
            
            # In later stages, check if we're making progress
            if self.current_stage_idx >= 2:
                # Check convergence
                if self._check_convergence():
                    return False, "Converged to stable estimate"
                else:
                    return True, "Still undecided - final escalation"
        
        return False, "No escalation needed"
    
    def _check_convergence(self) -> bool:
        """Check if estimates are converging"""
        if len(self.convergence_history) < 2:
            return False
        
        # Check if mean estimates are stabilizing
        recent_means = [h["mean"] for h in self.convergence_history[-3:]]
        if len(recent_means) >= 2:
            mean_change = abs(recent_means[-1] - recent_means[-2])
            if mean_change < 0.001:
                return True
        
        # Check if CI is shrinking
        recent_widths = [h["half_width"] for h in self.convergence_history[-3:]]
        if len(recent_widths) >= 2:
            width_ratio = recent_widths[-1] / recent_widths[-2]
            if width_ratio > 0.95:  # Not improving much
                return True
        
        return False
    
    def run_progressive_test(self, ref_model, cand_model, scorer, prompt_gen) -> Dict[str, Any]:
        """Run progressive testing through stages"""
        
        results_history = []
        final_result = None
        total_time = 0
        total_samples = 0
        
        logger.info("Starting progressive testing strategy")
        
        for stage_idx, stage in enumerate(self.stages):
            self.current_stage_idx = stage_idx
            
            print(f"\n🔍 STAGE {stage_idx + 1}: {stage.stage.value.upper()}")
            print(f"   Samples: {stage.n_samples}, K: {stage.k_positions}")
            print(f"   Confidence: {stage.confidence:.1%}, Timeout: {stage.timeout_seconds}s")
            
            stage_start = time.time()
            
            # Run test for this stage
            result = self._run_stage_test(
                ref_model, cand_model, scorer, prompt_gen, stage
            )
            
            stage_time = time.time() - stage_start
            total_time += stage_time
            total_samples += result.get("n_used", 0)
            
            # Store convergence info
            self.convergence_history.append({
                "stage": stage.stage.value,
                "mean": result.get("mean", 0),
                "half_width": result.get("half_width", 1.0),
                "rme": result.get("rme", 1.0)
            })
            
            result["stage"] = stage.stage.value
            result["stage_time"] = stage_time
            results_history.append(result)
            
            print(f"   Decision: {result['decision']}")
            print(f"   Mean: {result.get('mean', 0):.6f}, RME: {result.get('rme', 1.0):.3f}")
            print(f"   Time: {stage_time:.1f}s")
            
            # Check if we should escalate
            should_esc, reason = self.should_escalate(result, stage)
            
            print(f"   Escalation: {should_esc} - {reason}")
            
            if not should_esc:
                final_result = result
                break
            
            # Adapt next stage if needed
            if stage_idx + 1 < len(self.stages):
                self._adapt_next_stage(result, stage_idx)
        
        # If we exhausted all stages, use last result
        if final_result is None:
            final_result = results_history[-1]
        
        # Add progression metadata
        final_result["progression"] = {
            "stages_used": len(results_history),
            "total_samples": total_samples,
            "total_time": total_time,
            "avg_time_per_sample": total_time / total_samples if total_samples > 0 else 0,
            "history": results_history,
            "convergence": self.convergence_history
        }
        
        # Final summary
        print(f"\n📊 PROGRESSIVE TESTING COMPLETE")
        print(f"   Final Decision: {final_result['decision']}")
        print(f"   Stages Used: {len(results_history)}/{len(self.stages)}")
        print(f"   Total Samples: {total_samples}")
        print(f"   Total Time: {total_time:.1f}s")
        
        return final_result
    
    def _run_stage_test(self, ref_model, cand_model, scorer, prompt_gen, 
                       stage: StageConfig) -> Dict[str, Any]:
        """Run test for a single stage"""
        
        from .diff_decision import DiffDecisionConfig, EnhancedSequentialTester, TestingMode
        
        # Create config for this stage
        if stage.stage in [TestingStage.QUICK_CHECK, TestingStage.STANDARD_TEST]:
            mode = TestingMode.QUICK_GATE
        else:
            mode = TestingMode.AUDIT_GRADE
        
        config = DiffDecisionConfig(mode=mode)
        
        # Apply stage-specific thresholds
        if stage.gamma is not None:
            config.gamma = stage.gamma
        if stage.delta_star is not None:
            config.delta_star = stage.delta_star
        if stage.eta is not None:
            config.eta = stage.eta
        if stage.epsilon_diff is not None:
            config.epsilon_diff = stage.epsilon_diff
        
        config.n_min = min(5, stage.n_samples)
        config.n_max = stage.n_samples
        config.positions_per_prompt = stage.k_positions
        config.confidence = stage.confidence
        
        # Initialize tester
        tester = EnhancedSequentialTester(config)
        
        # Collect scores
        scores = []
        n_samples = 0
        start_time = time.time()
        
        while n_samples < stage.n_samples:
            # Check timeout
            if stage.timeout_seconds and (time.time() - start_time) > stage.timeout_seconds:
                logger.warning(f"Stage timeout reached after {n_samples} samples")
                break
            
            # Get prompt and score
            prompt = prompt_gen() if callable(prompt_gen) else prompt_gen[n_samples % len(prompt_gen)]
            
            # Get tokenizer from scorer
            tokenizer = scorer.tokenizer if hasattr(scorer, 'tokenizer') else None
            
            # Score based on method
            if stage.scoring_method == "delta_ce":
                score = scorer.score(ref_model, cand_model, prompt, tokenizer)
            else:  # symmetric_kl or other methods
                # For now, use same scoring (can be extended)
                score = scorer.score(ref_model, cand_model, prompt, tokenizer)
            
            scores.append(score)
            tester.update(score)
            n_samples += 1
            
            # Early stopping check
            if n_samples >= config.n_min and n_samples % 5 == 0:
                if self._check_early_stop(tester, config):
                    break
        
        # Get final statistics
        scores_array = np.array(scores)
        mean = np.mean(scores_array)
        std = np.std(scores_array, ddof=1) if len(scores_array) > 1 else 0
        
        # Compute confidence interval
        from scipy import stats
        if config.confidence == 0.95:
            z_score = 1.96
        elif config.confidence == 0.975:
            z_score = 2.24
        elif config.confidence == 0.99:
            z_score = 2.576
        else:
            z_score = 2.807  # 0.995
        
        margin = z_score * std / np.sqrt(len(scores_array)) if len(scores_array) > 0 else float('inf')
        ci_low, ci_high = mean - margin, mean + margin
        half_width = margin
        
        # Make decision
        same_ci = (ci_low >= -config.gamma) and (ci_high <= config.gamma)
        same_precision = half_width <= (config.eta * config.gamma)
        
        effect_size = abs(mean)
        relative_me = half_width / effect_size if effect_size > 0 else float('inf')
        diff_effect = effect_size >= config.delta_star
        diff_precision = relative_me <= config.epsilon_diff
        
        if same_ci and same_precision:
            decision = "SAME"
        elif diff_effect and diff_precision:
            decision = "DIFFERENT"
        else:
            decision = "UNDECIDED"
        
        return {
            "decision": decision,
            "mean": float(mean),
            "std": float(std),
            "ci": [float(ci_low), float(ci_high)],
            "half_width": float(half_width),
            "rme": float(relative_me),
            "n_used": n_samples,
            "scores": scores[:10],  # Store first 10 for debugging
            "thresholds": {
                "gamma": config.gamma,
                "delta_star": config.delta_star,
                "eta": config.eta,
                "epsilon_diff": config.epsilon_diff
            }
        }
    
    def _check_early_stop(self, tester, config) -> bool:
        """Check if we can stop early"""
        if len(tester.clipped_scores) < config.n_min:
            return False
        
        scores_array = np.array(tester.clipped_scores)
        mean = np.mean(scores_array)
        std = np.std(scores_array, ddof=1) if len(scores_array) > 1 else 0
        margin = 2.576 * std / np.sqrt(len(scores_array)) if len(scores_array) > 0 else float('inf')
        
        ci_low, ci_high = mean - margin, mean + margin
        
        # Strong SAME signal
        if abs(ci_high) < config.gamma * 0.5:
            return True
        
        # Strong DIFFERENT signal
        if abs(ci_low) > config.delta_star * 1.5:
            return True
        
        return False
    
    def _adapt_next_stage(self, current_result: Dict[str, Any], stage_idx: int):
        """Adapt next stage configuration based on current results"""
        
        if stage_idx + 1 < len(self.stages):
            next_stage = self.stages[stage_idx + 1]
            
            # If variance is high, increase K
            std = current_result.get("std", 0)
            if std > 0.1:
                next_stage.k_positions = min(next_stage.k_positions * 2, 256)
                logger.info(f"High variance detected (std={std:.3f}) - increasing K to {next_stage.k_positions}")
            
            # If mean is near zero, use more sensitive scoring
            mean = abs(current_result.get("mean", 0))
            if mean < 0.01:
                next_stage.scoring_method = "symmetric_kl"
                logger.info(f"Near-zero mean ({mean:.6f}) - switching to symmetric KL")
            
            # If we're borderline, increase samples
            if current_result.get("decision") == "UNDECIDED":
                rme = current_result.get("rme", 1.0)
                if rme > 0.25:
                    next_stage.n_samples = int(next_stage.n_samples * 1.5)
                    logger.info(f"High uncertainty (RME={rme:.3f}) - increasing samples to {next_stage.n_samples}")


class ProgressiveTestRunner:
    """Convenience runner for progressive testing"""
    
    @staticmethod
    def run(model_a_name: str, model_b_name: str, 
            n_prompts: int = 10,
            save_results: bool = True) -> Dict[str, Any]:
        """Run progressive test between two models"""
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ..scoring.optimized_scorer import FastScorer
        import torch
        
        print(f"\n🚀 PROGRESSIVE TESTING: {model_a_name} vs {model_b_name}")
        print("=" * 60)
        
        # Load models
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        tokenizer = AutoTokenizer.from_pretrained(model_a_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model_a = AutoModelForCausalLM.from_pretrained(model_a_name).to(device)
        model_b = AutoModelForCausalLM.from_pretrained(model_b_name).to(device)
        
        # Initialize scorer with tokenizer
        scorer = FastScorer(k=32, top_k=100)
        scorer.tokenizer = tokenizer  # Attach tokenizer to scorer
        
        # Create prompts
        prompts = [
            "The capital of France is",
            "To make a sandwich, you need",
            "The sky is blue because",
            "Water freezes at",
            "The largest planet is",
            "Machine learning is",
            "The future of technology",
            "Climate change is caused by",
            "The human brain",
            "Artificial intelligence will"
        ][:n_prompts]
        
        # Create prompt generator
        def prompt_gen():
            import random
            return random.choice(prompts)
        
        # Run progressive test
        verifier = ProgressiveVerifier()
        result = verifier.run_progressive_test(
            model_a, model_b, scorer, prompt_gen
        )
        
        # Save results if requested
        if save_results:
            output_dir = Path("experimental_results/progressive")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"progressive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_file}")
        
        return result