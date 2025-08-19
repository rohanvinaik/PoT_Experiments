"""
Complete POT (Proof of Training) Verification System for Google Colab
Run this entire block in a single Colab cell
"""

# ============================================================================
# SETUP AND INSTALLATIONS
# ============================================================================

import os
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

# Install required packages (if not already installed)
try:
    import torch
    import transformers
except ImportError:
    print("Installing required packages...")
    !pip install -q torch transformers accelerate

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# CORE POT COMPONENTS
# ============================================================================

# 1. Boundaries Module
# ----------------------------------------------------------------------------

@dataclass
class CSState:
    """Confidence sequence state tracking"""
    t: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    
    @property
    def mean(self) -> float:
        return self.sum / self.t if self.t > 0 else 0.0
    
    @property
    def var(self) -> float:
        if self.t <= 1:
            return 0.0
        mean_sq = (self.sum / self.t) ** 2
        return max(0, self.sum_sq / self.t - mean_sq)
    
    def update(self, z: float):
        self.t += 1
        self.sum += z
        self.sum_sq += z * z

def eb_radius(t: int, var: float, alpha: float) -> float:
    """Empirical Bernstein radius"""
    if t == 0:
        return float('inf')
    
    # Variance term
    var_term = np.sqrt(2 * var * np.log(3 / alpha) / t)
    
    # Range term (assuming [0,1] bounded)
    range_term = 3 * np.log(3 / alpha) / t
    
    return var_term + range_term

# 2. Prompt Generator Module
# ----------------------------------------------------------------------------

import hmac
import hashlib
import random

@dataclass
class PromptTemplate:
    template: str
    slots: List[str]
    category: str

class DeterministicPromptGenerator:
    def __init__(self, master_key: bytes):
        self.master_key = master_key
        self.templates = self._init_template_bank()
        self.filler_banks = self._init_filler_banks()
        self.concept_banks = self._init_concept_banks()
    
    def derive_seed(self, namespace: str, idx: int) -> bytes:
        message = f"{namespace}||{idx}".encode('utf-8')
        return hmac.new(self.master_key, message, hashlib.sha256).digest()
    
    def seed_to_prompt(self, seed: bytes, max_length: int = 100) -> str:
        rng = random.Random(int.from_bytes(seed[:8], 'big'))
        template = rng.choice(self.templates)
        
        filled_slots = {}
        for slot in template.slots:
            if slot == "filler":
                filled_slots[slot] = self._select_filler(rng, template.category)
            elif slot == "concept":
                filled_slots[slot] = self._select_concept(rng, template.category)
            else:
                filled_slots[slot] = self._generate_slot_content(rng, slot)
        
        prompt = template.template.format(**filled_slots)
        
        if len(prompt) > max_length:
            prompt = self._smart_truncate(prompt, max_length)
        
        return prompt
    
    def batch_generate(self, namespace: str, start_idx: int, count: int) -> List[str]:
        prompts = []
        for i in range(start_idx, start_idx + count):
            seed = self.derive_seed(namespace, i)
            prompt = self.seed_to_prompt(seed)
            prompts.append(prompt)
        return prompts
    
    def _init_template_bank(self) -> List[PromptTemplate]:
        return [
            PromptTemplate("What is {concept}? {filler}", ["concept", "filler"], "factual"),
            PromptTemplate("Explain {concept} in simple terms. {filler}", ["concept", "filler"], "factual"),
            PromptTemplate("How do you {task}? {filler}", ["task", "filler"], "instruction"),
        ]
    
    def _init_filler_banks(self) -> Dict[str, List[str]]:
        return {
            "factual": ["Please be specific.", "Be concise.", "Focus on main points."],
            "instruction": ["Be practical.", "Keep it simple.", "Be clear."],
        }
    
    def _init_concept_banks(self) -> Dict[str, List[str]]:
        return {
            "factual": ["gravity", "democracy", "evolution", "photosynthesis"],
            "instruction": ["solve problems", "learn skills", "save time"],
        }
    
    def _select_filler(self, rng: random.Random, category: str) -> str:
        fillers = self.filler_banks.get(category, self.filler_banks["factual"])
        return rng.choice(fillers)
    
    def _select_concept(self, rng: random.Random, category: str) -> str:
        concepts = self.concept_banks.get(category, self.concept_banks["factual"])
        return rng.choice(concepts)
    
    def _generate_slot_content(self, rng: random.Random, slot: str) -> str:
        generators = {
            "task": lambda: rng.choice(["solve complex problems", "make decisions", "build systems"])
        }
        return generators.get(slot, lambda: f"[{slot}]")()
    
    def _smart_truncate(self, prompt: str, max_length: int) -> str:
        if len(prompt) <= max_length:
            return prompt
        return prompt[:max_length-3] + "..."

# 3. Teacher-Forced Scoring Module
# ----------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    method: str = "delta_ce"
    num_positions: int = 10
    temperature: float = 1.0
    epsilon: float = 1e-8
    use_canonical_suffix: bool = True
    max_seq_length: int = 512

@dataclass
class ScoringResult:
    score: float
    raw_score: float
    per_position_scores: List[float]
    metadata: Dict[str, Any]

class TeacherForcedScorer:
    def __init__(self, config: ScoringConfig):
        self.config = config
        self.canonical_suffixes = {
            "factual": " The answer is",
            "reasoning": " Let me think step by step.",
            "instruction": " Here's how to do it:",
            "default": " The response is"
        }
    
    def score_models(self, reference_model: Any, candidate_model: Any,
                     prompt: str, prompt_type: str = "default",
                     tokenizer: Any = None) -> ScoringResult:
        
        suffix = self.canonical_suffixes.get(prompt_type, self.canonical_suffixes["default"])
        full_text = prompt + suffix
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
        full_inputs = tokenizer(full_text, return_tensors="pt", padding=False, truncation=False)
        
        prompt_length = inputs["input_ids"].shape[1]
        total_length = full_inputs["input_ids"].shape[1]
        eval_positions = min(self.config.num_positions, total_length - prompt_length)
        
        ref_logprobs = self._get_logprobs(reference_model, full_inputs, prompt_length, eval_positions)
        cand_logprobs = self._get_logprobs(candidate_model, full_inputs, prompt_length, eval_positions)
        
        if self.config.method == "delta_ce":
            return self._compute_delta_ce(ref_logprobs, cand_logprobs, full_inputs,
                                         prompt_length, eval_positions)
        elif self.config.method == "symmetric_kl":
            return self._compute_symmetric_kl(ref_logprobs, cand_logprobs)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
    
    def _get_logprobs(self, model: Any, inputs: Dict[str, torch.Tensor],
                     prompt_length: int, eval_positions: int) -> torch.Tensor:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits / self.config.temperature
            log_probs = F.log_softmax(logits, dim=-1)
            eval_log_probs = log_probs[0, prompt_length:prompt_length+eval_positions, :]
        return eval_log_probs
    
    def _compute_delta_ce(self, ref_logprobs: torch.Tensor, cand_logprobs: torch.Tensor,
                         inputs: Dict[str, torch.Tensor], prompt_length: int,
                         eval_positions: int) -> ScoringResult:
        
        target_ids = inputs["input_ids"][0, prompt_length+1:prompt_length+1+eval_positions]
        per_position_scores = []
        
        for pos in range(min(eval_positions, len(target_ids))):
            target_id = target_ids[pos]
            ref_lp = ref_logprobs[pos, target_id].item()
            cand_lp = cand_logprobs[pos, target_id].item()
            ce_diff = (-cand_lp) - (-ref_lp)
            per_position_scores.append(ce_diff)
        
        raw_score = np.mean(per_position_scores) if per_position_scores else 0.0
        normalized_score = np.clip(np.abs(raw_score) / 10.0, 0.0, 1.0)
        
        return ScoringResult(
            score=normalized_score,
            raw_score=raw_score,
            per_position_scores=per_position_scores,
            metadata={"method": "delta_ce", "eval_positions": eval_positions}
        )
    
    def _compute_symmetric_kl(self, ref_logprobs: torch.Tensor,
                             cand_logprobs: torch.Tensor) -> ScoringResult:
        per_position_scores = []
        
        for pos in range(ref_logprobs.shape[0]):
            ref_probs = torch.exp(ref_logprobs[pos]) + self.config.epsilon
            cand_probs = torch.exp(cand_logprobs[pos]) + self.config.epsilon
            
            ref_probs = ref_probs / ref_probs.sum()
            cand_probs = cand_probs / cand_probs.sum()
            
            kl_ref_cand = (ref_probs * (torch.log(ref_probs) - torch.log(cand_probs))).sum()
            kl_cand_ref = (cand_probs * (torch.log(cand_probs) - torch.log(ref_probs))).sum()
            
            sym_kl = 0.5 * (kl_ref_cand + kl_cand_ref)
            per_position_scores.append(sym_kl.item())
        
        raw_score = np.mean(per_position_scores) if per_position_scores else 0.0
        normalized_score = np.clip(raw_score / 5.0, 0.0, 1.0)
        
        return ScoringResult(
            score=normalized_score,
            raw_score=raw_score,
            per_position_scores=per_position_scores,
            metadata={"method": "symmetric_kl", "eval_positions": len(per_position_scores)}
        )

# 4. Sequential Test Runner Module
# ----------------------------------------------------------------------------

@dataclass
class TestMetrics:
    alpha: float
    beta: float
    n_used: int
    n_max: int
    statistic_mean: float
    statistic_var: float
    boundary_type: str
    decision: str
    hypothesis: str
    stopping_time: int
    t_load: float
    t_infer_total: float
    t_per_query: float
    t_setup: float
    t_total: float
    per_query_times: List[float]
    per_query_scores: List[float]
    confidence_intervals: List[Tuple[float, float]]

class SequentialTestRunner:
    def __init__(self, reference_model_path: str, scoring_config: ScoringConfig,
                 master_key: bytes, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.reference_model_path = reference_model_path
        self.scoring_config = scoring_config
        self.master_key = master_key
        self.device = device
        self.reference_model = None
        self.tokenizer = None
        self.prompt_generator = DeterministicPromptGenerator(master_key)
        self.scorer = TeacherForcedScorer(scoring_config)
    
    def load_models(self, candidate_model_path: str) -> Tuple[float, Any]:
        t_start = time.perf_counter()
        
        if self.reference_model is None:
            print(f"Loading reference model: {self.reference_model_path}")
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.reference_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            self.reference_model.eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.reference_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading candidate model: {candidate_model_path}")
        candidate_model = AutoModelForCausalLM.from_pretrained(
            candidate_model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        candidate_model.eval()
        
        t_load = time.perf_counter() - t_start
        print(f"Models loaded in {t_load:.2f}s")
        
        return t_load, candidate_model
    
    def run_sequential_test(self, candidate_model_path: str, alpha: float = 0.01,
                           beta: float = 0.01, tau: float = 0.05, n_max: int = 100,
                           boundary_type: str = "EB", namespace: str = "verification") -> TestMetrics:
        
        t_total_start = time.perf_counter()
        
        # Load models
        t_load, candidate_model = self.load_models(candidate_model_path)
        
        # Initialize
        cs = CSState()
        per_query_times = []
        per_query_scores = []
        confidence_intervals = []
        
        # Setup
        t_setup_start = time.perf_counter()
        prompts = self.prompt_generator.batch_generate(namespace, 0, n_max)
        prompt_types = ["factual"] * len(prompts)
        t_setup = time.perf_counter() - t_setup_start
        
        # Run test
        t_infer_start = time.perf_counter()
        decision = None
        hypothesis = None
        stopping_time = n_max
        
        print(f"\nRunning sequential test (α={alpha}, β={beta}, τ={tau}, n_max={n_max})")
        print("-" * 60)
        
        for i in range(n_max):
            t_query_start = time.perf_counter()
            
            result = self.scorer.score_models(
                self.reference_model, candidate_model,
                prompts[i], prompt_types[i], self.tokenizer
            )
            
            t_query = time.perf_counter() - t_query_start
            per_query_times.append(t_query)
            
            z = result.score
            per_query_scores.append(z)
            cs.update(z)
            
            r_accept = eb_radius(cs.t, cs.var, alpha)
            r_reject = eb_radius(cs.t, cs.var, beta)
            
            confidence_intervals.append((cs.mean - r_reject, cs.mean + r_accept))
            
            # Decision logic
            if cs.mean + r_accept <= tau:
                decision = "accept_id"
                hypothesis = "H1"
                stopping_time = i + 1
                print(f"\n✅ Accepted identity at n={stopping_time}, mean={cs.mean:.4f}")
                break
            elif cs.mean - r_reject >= tau:
                decision = "reject_id"
                hypothesis = "H0"
                stopping_time = i + 1
                print(f"\n❌ Rejected (impostor) at n={stopping_time}, mean={cs.mean:.4f}")
                break
            
            if (i + 1) % 10 == 0:
                print(f"n={i+1}: mean={cs.mean:.4f}, CI=[{cs.mean - r_reject:.4f}, {cs.mean + r_accept:.4f}]")
        
        t_infer_total = time.perf_counter() - t_infer_start
        
        if decision is None:
            decision = "undecided"
            hypothesis = "inconclusive"
            print(f"\n⚠️ No decision after n_max={n_max}, mean={cs.mean:.4f}")
        
        return TestMetrics(
            alpha=alpha, beta=beta, n_used=stopping_time, n_max=n_max,
            statistic_mean=float(cs.mean), statistic_var=float(cs.var),
            boundary_type=boundary_type, decision=decision, hypothesis=hypothesis,
            stopping_time=stopping_time, t_load=t_load, t_infer_total=t_infer_total,
            t_per_query=t_infer_total / stopping_time if stopping_time > 0 else 0,
            t_setup=t_setup, t_total=time.perf_counter() - t_total_start,
            per_query_times=per_query_times[:stopping_time],
            per_query_scores=per_query_scores[:stopping_time],
            confidence_intervals=confidence_intervals[:stopping_time]
        )

# ============================================================================
# MAIN EXECUTION FOR COLAB
# ============================================================================

def run_pot_verification_demo():
    """Main function to run POT verification in Colab"""
    
    print("🚀 POT (Proof of Training) Verification System")
    print("=" * 60)
    
    # Configuration
    REFERENCE_MODEL = "gpt2"  # Use GPT-2 as reference
    CANDIDATE_MODEL = "distilgpt2"  # Compare with DistilGPT-2
    
    # You can change these to test other models:
    # REFERENCE_MODEL = "microsoft/DialoGPT-small"
    # CANDIDATE_MODEL = "gpt2"
    
    print(f"\nReference Model: {REFERENCE_MODEL}")
    print(f"Candidate Model: {CANDIDATE_MODEL}")
    print("-" * 60)
    
    # Setup
    master_key = b"colab_test_key_123456789012345678"  # 32 bytes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Scoring configuration
    scoring_config = ScoringConfig(
        method="symmetric_kl",  # or "delta_ce"
        num_positions=5,  # Number of tokens to check
        temperature=1.0
    )
    
    # Initialize runner
    runner = SequentialTestRunner(
        reference_model_path=REFERENCE_MODEL,
        scoring_config=scoring_config,
        master_key=master_key,
        device=device
    )
    
    # Run sequential test
    metrics = runner.run_sequential_test(
        candidate_model_path=CANDIDATE_MODEL,
        alpha=0.01,  # 1% false accept rate
        beta=0.01,   # 1% false reject rate
        tau=0.10,    # Threshold for acceptance
        n_max=50,    # Maximum number of tests
        boundary_type="EB",
        namespace="colab_demo"
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Decision: {metrics.decision}")
    print(f"Hypothesis: {metrics.hypothesis}")
    print(f"Samples used: {metrics.n_used}/{metrics.n_max}")
    print(f"Mean score: {metrics.statistic_mean:.4f}")
    print(f"Variance: {metrics.statistic_var:.6f}")
    print(f"\nTiming:")
    print(f"  Model loading: {metrics.t_load:.1f}s")
    print(f"  Inference: {metrics.t_infer_total:.1f}s")
    print(f"  Per query: {metrics.t_per_query*1000:.0f}ms")
    print(f"  Total: {metrics.t_total:.1f}s")
    
    # Plot results
    if metrics.per_query_scores:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Score progression
        ax1.plot(range(1, len(metrics.per_query_scores) + 1), metrics.per_query_scores, 'b-', alpha=0.7)
        ax1.axhline(y=metrics.statistic_mean, color='r', linestyle='--', label=f'Mean={metrics.statistic_mean:.3f}')
        ax1.axhline(y=0.10, color='g', linestyle=':', label='Threshold τ=0.10')
        ax1.set_xlabel('Query Number')
        ax1.set_ylabel('Score')
        ax1.set_title('Score Progression')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confidence intervals
        if metrics.confidence_intervals:
            ci_lower = [ci[0] for ci in metrics.confidence_intervals]
            ci_upper = [ci[1] for ci in metrics.confidence_intervals]
            x = range(1, len(ci_lower) + 1)
            
            ax2.fill_between(x, ci_lower, ci_upper, alpha=0.3, color='blue', label='Confidence Interval')
            ax2.plot(x, [metrics.statistic_mean] * len(x), 'r-', alpha=0.7, label='Running Mean')
            ax2.axhline(y=0.10, color='g', linestyle=':', label='Threshold τ')
            ax2.set_xlabel('Query Number')
            ax2.set_ylabel('Score')
            ax2.set_title('Confidence Intervals')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print("\n✅ POT Verification Complete!")
    
    return metrics

# Run the demo
if __name__ == "__main__":
    metrics = run_pot_verification_demo()