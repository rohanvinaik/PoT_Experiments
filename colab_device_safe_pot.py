"""
Complete POT (Proof of Training) Verification System for Google Colab
WITH PROPER DEVICE MANAGEMENT - No CPU/GPU mismatch errors
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

# Device setup and verification
def get_device():
    """Get the best available device and print info"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    return device

DEVICE = get_device()

print(f"\n📦 Package versions:")
print(f"   PyTorch: {torch.__version__}")
print(f"   Transformers: {transformers.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")

# ============================================================================
# CORE POT COMPONENTS WITH DEVICE MANAGEMENT
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
        if np.isnan(z):
            print(f"Warning: NaN value detected, using 0.5")
            z = 0.5
        self.t += 1
        self.sum += z
        self.sum_sq += z * z

def eb_radius(t: int, var: float, alpha: float) -> float:
    """Empirical Bernstein radius"""
    if t == 0:
        return float('inf')
    
    var = max(0, min(var, 1.0))
    var_term = np.sqrt(2 * var * np.log(3 / alpha) / t)
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

# 3. Teacher-Forced Scoring Module WITH DEVICE SAFETY
# ----------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    method: str = "delta_ce"
    num_positions: int = 5
    temperature: float = 1.0
    epsilon: float = 1e-8
    use_canonical_suffix: bool = True
    max_seq_length: int = 128

@dataclass
class ScoringResult:
    score: float
    raw_score: float
    per_position_scores: List[float]
    metadata: Dict[str, Any]

class TeacherForcedScorer:
    def __init__(self, config: ScoringConfig, device: torch.device):
        self.config = config
        self.device = device  # Store device
        self.canonical_suffixes = {
            "factual": " The answer is",
            "reasoning": " Let me think step by step.",
            "instruction": " Here's how to do it:",
            "default": " The response is"
        }
    
    def score_models(self, reference_model: Any, candidate_model: Any,
                     prompt: str, prompt_type: str = "default",
                     tokenizer: Any = None) -> ScoringResult:
        
        try:
            suffix = self.canonical_suffixes.get(prompt_type, self.canonical_suffixes["default"])
            full_text = prompt + suffix
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
            full_inputs = tokenizer(full_text, return_tensors="pt", 
                                   truncation=True, max_length=self.config.max_seq_length)
            
            # CRITICAL: Move inputs to device
            full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}
            
            prompt_length = inputs["input_ids"].shape[1]
            total_length = full_inputs["input_ids"].shape[1]
            eval_positions = min(self.config.num_positions, total_length - prompt_length)
            
            if eval_positions <= 0:
                return ScoringResult(score=0.5, raw_score=0.0, per_position_scores=[], 
                                    metadata={"error": "no_eval_positions"})
            
            # Get log probabilities (inputs already on correct device)
            ref_logprobs = self._get_logprobs(reference_model, full_inputs, prompt_length, eval_positions)
            cand_logprobs = self._get_logprobs(candidate_model, full_inputs, prompt_length, eval_positions)
            
            if self.config.method == "delta_ce":
                return self._compute_delta_ce(ref_logprobs, cand_logprobs, full_inputs,
                                             prompt_length, eval_positions)
            elif self.config.method == "symmetric_kl":
                return self._compute_symmetric_kl(ref_logprobs, cand_logprobs)
            else:
                return self._compute_simple_difference(ref_logprobs, cand_logprobs)
                
        except RuntimeError as e:
            if "device" in str(e).lower():
                print(f"Device error detected: {e}")
                print(f"Model device: {next(reference_model.parameters()).device}")
                print(f"Input device: {self.device}")
            return ScoringResult(score=0.5, raw_score=0.0, per_position_scores=[], 
                               metadata={"error": str(e)})
        except Exception as e:
            print(f"Error in scoring: {e}")
            return ScoringResult(score=0.5, raw_score=0.0, per_position_scores=[], 
                               metadata={"error": str(e)})
    
    def _get_logprobs(self, model: Any, inputs: Dict[str, torch.Tensor],
                     prompt_length: int, eval_positions: int) -> torch.Tensor:
        try:
            # Double-check inputs are on correct device
            model_device = next(model.parameters()).device
            for key, tensor in inputs.items():
                if tensor.device != model_device:
                    print(f"Warning: Moving {key} from {tensor.device} to {model_device}")
                    inputs[key] = tensor.to(model_device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits / self.config.temperature
                
                if torch.isnan(logits).any():
                    print("Warning: NaN in logits")
                    logits = torch.zeros_like(logits)
                
                log_probs = F.log_softmax(logits, dim=-1)
                eval_log_probs = log_probs[0, prompt_length:prompt_length+eval_positions, :]
                
            return eval_log_probs
        except Exception as e:
            print(f"Error getting logprobs: {e}")
            vocab_size = 50257  # GPT-2 vocab size
            return torch.log(torch.ones(eval_positions, vocab_size) / vocab_size).to(self.device)
    
    def _compute_simple_difference(self, ref_logprobs: torch.Tensor, 
                                  cand_logprobs: torch.Tensor) -> ScoringResult:
        """Simple L2 distance between distributions"""
        try:
            per_position_scores = []
            
            for pos in range(min(ref_logprobs.shape[0], cand_logprobs.shape[0])):
                ref_probs = torch.exp(ref_logprobs[pos])
                cand_probs = torch.exp(cand_logprobs[pos])
                
                diff = torch.norm(ref_probs - cand_probs).item()
                per_position_scores.append(diff)
            
            raw_score = np.mean(per_position_scores) if per_position_scores else 0.5
            normalized_score = np.clip(raw_score, 0.0, 1.0)
            
            return ScoringResult(
                score=normalized_score,
                raw_score=raw_score,
                per_position_scores=per_position_scores,
                metadata={"method": "l2_distance"}
            )
        except Exception as e:
            print(f"Error in simple difference: {e}")
            return ScoringResult(score=0.5, raw_score=0.5, per_position_scores=[], 
                               metadata={"error": str(e)})
    
    def _compute_delta_ce(self, ref_logprobs: torch.Tensor, cand_logprobs: torch.Tensor,
                         inputs: Dict[str, torch.Tensor], prompt_length: int,
                         eval_positions: int) -> ScoringResult:
        
        try:
            target_ids = inputs["input_ids"][0, prompt_length+1:prompt_length+1+eval_positions]
            per_position_scores = []
            
            for pos in range(min(eval_positions, len(target_ids), ref_logprobs.shape[0])):
                target_id = target_ids[pos]
                
                if target_id >= ref_logprobs.shape[1]:
                    continue
                    
                ref_lp = ref_logprobs[pos, target_id].item()
                cand_lp = cand_logprobs[pos, target_id].item()
                
                if np.isnan(ref_lp) or np.isnan(cand_lp):
                    continue
                    
                ce_diff = abs((-cand_lp) - (-ref_lp))
                per_position_scores.append(ce_diff)
            
            if not per_position_scores:
                return ScoringResult(score=0.5, raw_score=0.5, per_position_scores=[], 
                                   metadata={"error": "no_valid_scores"})
            
            raw_score = np.mean(per_position_scores)
            normalized_score = np.clip(raw_score / 10.0, 0.0, 1.0)
            
            return ScoringResult(
                score=normalized_score,
                raw_score=raw_score,
                per_position_scores=per_position_scores,
                metadata={"method": "delta_ce", "eval_positions": len(per_position_scores)}
            )
        except Exception as e:
            print(f"Error in delta_ce: {e}")
            return ScoringResult(score=0.5, raw_score=0.5, per_position_scores=[], 
                               metadata={"error": str(e)})
    
    def _compute_symmetric_kl(self, ref_logprobs: torch.Tensor,
                             cand_logprobs: torch.Tensor) -> ScoringResult:
        try:
            per_position_scores = []
            
            for pos in range(min(ref_logprobs.shape[0], cand_logprobs.shape[0])):
                ref_probs = torch.exp(ref_logprobs[pos]) + self.config.epsilon
                cand_probs = torch.exp(cand_logprobs[pos]) + self.config.epsilon
                
                ref_probs = ref_probs / ref_probs.sum()
                cand_probs = cand_probs / cand_probs.sum()
                
                if torch.isnan(ref_probs).any() or torch.isnan(cand_probs).any():
                    continue
                
                kl_ref_cand = (ref_probs * (torch.log(ref_probs) - torch.log(cand_probs))).sum()
                kl_cand_ref = (cand_probs * (torch.log(cand_probs) - torch.log(ref_probs))).sum()
                
                sym_kl = 0.5 * (kl_ref_cand + kl_cand_ref)
                score = sym_kl.item()
                
                if not np.isnan(score):
                    per_position_scores.append(score)
            
            if not per_position_scores:
                return ScoringResult(score=0.5, raw_score=0.5, per_position_scores=[], 
                                   metadata={"error": "no_valid_kl_scores"})
            
            raw_score = np.mean(per_position_scores)
            normalized_score = np.clip(raw_score / 5.0, 0.0, 1.0)
            
            return ScoringResult(
                score=normalized_score,
                raw_score=raw_score,
                per_position_scores=per_position_scores,
                metadata={"method": "symmetric_kl", "eval_positions": len(per_position_scores)}
            )
        except Exception as e:
            print(f"Error in symmetric_kl: {e}")
            return ScoringResult(score=0.5, raw_score=0.5, per_position_scores=[], 
                               metadata={"error": str(e)})

# 4. Sequential Test Runner Module WITH DEVICE MANAGEMENT
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
                 master_key: bytes, device: torch.device):
        self.reference_model_path = reference_model_path
        self.scoring_config = scoring_config
        self.master_key = master_key
        self.device = device
        self.reference_model = None
        self.tokenizer = None
        self.prompt_generator = DeterministicPromptGenerator(master_key)
        self.scorer = TeacherForcedScorer(scoring_config, device)  # Pass device to scorer
    
    def load_models(self, candidate_model_path: str) -> Tuple[float, Any]:
        t_start = time.perf_counter()
        
        if self.reference_model is None:
            print(f"📥 Loading reference model: {self.reference_model_path}")
            try:
                # Load model with proper device placement
                self.reference_model = AutoModelForCausalLM.from_pretrained(
                    self.reference_model_path,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
                
                # Move to device
                self.reference_model = self.reference_model.to(self.device)
                self.reference_model.eval()
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.reference_model_path)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                print(f"✅ Reference model loaded on {self.reference_model.device}")
                print(f"   Model device: {next(self.reference_model.parameters()).device}")
                
            except Exception as e:
                print(f"❌ Error loading reference model: {e}")
                raise
        
        print(f"📥 Loading candidate model: {candidate_model_path}")
        try:
            # Load candidate model with proper device placement
            candidate_model = AutoModelForCausalLM.from_pretrained(
                candidate_model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move to device
            candidate_model = candidate_model.to(self.device)
            candidate_model.eval()
            
            print(f"✅ Candidate model loaded on {candidate_model.device}")
            print(f"   Model device: {next(candidate_model.parameters()).device}")
            
        except Exception as e:
            print(f"❌ Error loading candidate model: {e}")
            raise
        
        t_load = time.perf_counter() - t_start
        print(f"⏱️ Models loaded in {t_load:.2f}s")
        
        # Verify models work
        print("🧪 Testing models...")
        test_text = "Hello world"
        test_input = self.tokenizer(test_text, return_tensors="pt")
        test_input = {k: v.to(self.device) for k, v in test_input.items()}  # Move to device
        
        with torch.no_grad():
            ref_out = self.reference_model(**test_input)
            cand_out = candidate_model(**test_input)
            
        print(f"✅ Models working")
        print(f"   Reference output shape: {ref_out.logits.shape}")
        print(f"   Candidate output shape: {cand_out.logits.shape}")
        
        # Clear cache if using GPU
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            print(f"🧹 GPU cache cleared")
        
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
        
        print(f"\n🔬 Running sequential test")
        print(f"   α={alpha}, β={beta}, τ={tau}, n_max={n_max}")
        print("-" * 60)
        
        print(f"📝 First prompt: '{prompts[0][:50]}...'")
        
        for i in range(n_max):
            t_query_start = time.perf_counter()
            
            # Score models
            result = self.scorer.score_models(
                self.reference_model, candidate_model,
                prompts[i], prompt_types[i], self.tokenizer
            )
            
            t_query = time.perf_counter() - t_query_start
            per_query_times.append(t_query)
            
            z = result.score
            per_query_scores.append(z)
            
            # Show first few scores
            if i < 3:
                print(f"   Query {i+1}: score={z:.4f}, time={t_query*1000:.1f}ms")
            
            cs.update(z)
            
            r_accept = eb_radius(cs.t, cs.var, alpha)
            r_reject = eb_radius(cs.t, cs.var, beta)
            
            confidence_intervals.append((cs.mean - r_reject, cs.mean + r_accept))
            
            # Decision logic
            if cs.mean + r_accept <= tau:
                decision = "accept_id"
                hypothesis = "H1"
                stopping_time = i + 1
                print(f"\n✅ ACCEPTED identity at n={stopping_time}, mean={cs.mean:.4f}")
                break
            elif cs.mean - r_reject >= tau:
                decision = "reject_id"
                hypothesis = "H0"
                stopping_time = i + 1
                print(f"\n❌ REJECTED (impostor) at n={stopping_time}, mean={cs.mean:.4f}")
                break
            
            if (i + 1) % 10 == 0:
                print(f"📊 n={i+1}: mean={cs.mean:.4f}, CI=[{cs.mean - r_reject:.4f}, {cs.mean + r_accept:.4f}]")
            
            # Clear GPU cache periodically
            if self.device.type == "cuda" and (i + 1) % 20 == 0:
                torch.cuda.empty_cache()
        
        t_infer_total = time.perf_counter() - t_infer_start
        
        if decision is None:
            decision = "undecided"
            hypothesis = "inconclusive"
            print(f"\n⚠️ UNDECIDED after n_max={n_max}, mean={cs.mean:.4f}")
        
        return TestMetrics(
            alpha=alpha, beta=beta, n_used=stopping_time, n_max=n_max,
            statistic_mean=float(cs.mean) if not np.isnan(cs.mean) else 0.5,
            statistic_var=float(cs.var) if not np.isnan(cs.var) else 0.0,
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
    
    print("=" * 70)
    print("🚀 POT (Proof of Training) Verification System")
    print("   Device-Safe Version - No CPU/GPU Mismatch Errors")
    print("=" * 70)
    
    # Models to compare
    REFERENCE_MODEL = "gpt2"  # GPT-2 (124M params)
    CANDIDATE_MODEL = "distilgpt2"  # DistilGPT-2 (82M params)
    
    # Alternative models you can try:
    # REFERENCE_MODEL = "microsoft/DialoGPT-small"
    # CANDIDATE_MODEL = "gpt2"
    
    print(f"\n📋 Configuration:")
    print(f"   Reference Model: {REFERENCE_MODEL}")
    print(f"   Candidate Model: {CANDIDATE_MODEL}")
    print("-" * 70)
    
    # Setup
    master_key = b"colab_test_key_123456789012345678"  # 32 bytes
    
    # Scoring configuration
    scoring_config = ScoringConfig(
        method="delta_ce",  # or "symmetric_kl"
        num_positions=5,    # Number of tokens to check
        temperature=1.0
    )
    
    print(f"\n⚙️ Settings:")
    print(f"   Scoring method: {scoring_config.method}")
    print(f"   Positions evaluated: {scoring_config.num_positions}")
    print(f"   Device: {DEVICE}")
    
    # Initialize runner with device
    runner = SequentialTestRunner(
        reference_model_path=REFERENCE_MODEL,
        scoring_config=scoring_config,
        master_key=master_key,
        device=DEVICE  # Pass the global device
    )
    
    # Run sequential test
    print("\n" + "=" * 70)
    metrics = runner.run_sequential_test(
        candidate_model_path=CANDIDATE_MODEL,
        alpha=0.01,  # 1% false accept rate
        beta=0.01,   # 1% false reject rate
        tau=0.10,    # Threshold for acceptance
        n_max=30,    # Maximum queries
        boundary_type="EB",
        namespace="colab_demo"
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"Decision: {metrics.decision}")
    print(f"Hypothesis: {metrics.hypothesis}")
    print(f"Samples used: {metrics.n_used}/{metrics.n_max}")
    print(f"Mean score: {metrics.statistic_mean:.4f}")
    print(f"Variance: {metrics.statistic_var:.6f}")
    print(f"Threshold (τ): 0.10")
    
    print(f"\n⏱️ Performance:")
    print(f"   Model loading: {metrics.t_load:.1f}s")
    print(f"   Total inference: {metrics.t_infer_total:.1f}s")
    print(f"   Per query: {metrics.t_per_query*1000:.0f}ms")
    print(f"   Total time: {metrics.t_total:.1f}s")
    
    # Show scores
    if metrics.per_query_scores:
        valid_scores = [s for s in metrics.per_query_scores if not np.isnan(s)]
        if valid_scores:
            print(f"\n📈 Score Statistics:")
            print(f"   Min: {min(valid_scores):.4f}")
            print(f"   Max: {max(valid_scores):.4f}")
            print(f"   Mean: {np.mean(valid_scores):.4f}")
            print(f"   First 5 scores: {[f'{s:.3f}' for s in valid_scores[:5]]}")
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        if metrics.per_query_scores and len(metrics.per_query_scores) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Score progression
            valid_scores = [s for s in metrics.per_query_scores if not np.isnan(s)]
            if valid_scores:
                ax1.plot(range(1, len(valid_scores) + 1), valid_scores, 'b-', alpha=0.7, linewidth=2)
                ax1.axhline(y=metrics.statistic_mean, color='r', linestyle='--', 
                           label=f'Mean={metrics.statistic_mean:.3f}', linewidth=2)
                ax1.axhline(y=0.10, color='g', linestyle=':', 
                           label='Threshold τ=0.10', linewidth=2)
                ax1.set_xlabel('Query Number')
                ax1.set_ylabel('Score')
                ax1.set_title('Score Progression')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim([0, max(0.2, max(valid_scores) * 1.1)])
            
            # Confidence intervals
            if metrics.confidence_intervals:
                ci_lower = [ci[0] for ci in metrics.confidence_intervals if not np.isnan(ci[0])]
                ci_upper = [ci[1] for ci in metrics.confidence_intervals if not np.isnan(ci[1])]
                if ci_lower and ci_upper:
                    x = range(1, len(ci_lower) + 1)
                    ax2.fill_between(x, ci_lower, ci_upper, alpha=0.3, color='blue', 
                                    label='Confidence Interval')
                    ax2.plot(x, [metrics.statistic_mean] * len(x), 'r-', alpha=0.7, 
                            label='Running Mean', linewidth=2)
                    ax2.axhline(y=0.10, color='g', linestyle=':', 
                              label='Threshold τ', linewidth=2)
                    ax2.set_xlabel('Query Number')
                    ax2.set_ylabel('Score')
                    ax2.set_title('Confidence Intervals')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"⚠️ Could not create plots: {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    if metrics.decision == "accept_id":
        print("✅ RESULT: Models are SIMILAR (identity accepted)")
    elif metrics.decision == "reject_id":
        print("❌ RESULT: Models are DIFFERENT (identity rejected)")
    else:
        print("⚠️ RESULT: INCONCLUSIVE (need more samples)")
    print("=" * 70)
    
    print("\n✅ POT Verification Complete!")
    
    # Clear GPU memory if used
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleared")
    
    return metrics

# Run the demo
if __name__ == "__main__":
    metrics = run_pot_verification_demo()