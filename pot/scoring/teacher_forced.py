import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class ScoringConfig:
    method: str = "delta_ce"  # or "symmetric_kl"
    num_positions: int = 64
    score_clip: Tuple[float, float] = (0.0, 0.3)  # for EB CI
    temperature: float = 1.0
    epsilon: float = 1e-8

class TeacherForcedScorer:
    """Teacher-forced scorer with non-negative outputs"""
    
    def __init__(self, config: ScoringConfig):
        self.config = config
        
    def score(self, ref_model, cand_model, inputs, prompt_len: int) -> float:
        """Score models ensuring non-negative result"""
        K = self.config.num_positions
        
        with torch.no_grad():
            # Get model outputs
            ref_outputs = ref_model(**inputs)
            cand_outputs = cand_model(**inputs)
            
            # Extract logits for K positions after prompt
            max_len = min(prompt_len + K, ref_outputs.logits.shape[1] - 1)
            ref_logits = ref_outputs.logits[0, prompt_len:max_len]
            cand_logits = cand_outputs.logits[0, prompt_len:max_len]
            
            # Get target tokens
            targets = inputs["input_ids"][0, prompt_len + 1:max_len + 1]
            
            if self.config.method == "delta_ce":
                # Compute CE difference (always non-negative via abs)
                ref_log_probs = F.log_softmax(ref_logits / self.config.temperature, dim=-1)
                cand_log_probs = F.log_softmax(cand_logits / self.config.temperature, dim=-1)
                
                # Get log probs for actual next tokens
                ref_ce = -ref_log_probs.gather(1, targets.unsqueeze(1)).squeeze().mean()
                cand_ce = -cand_log_probs.gather(1, targets.unsqueeze(1)).squeeze().mean()
                
                # Absolute difference ensures non-negative
                score = abs((cand_ce - ref_ce).item())
                
            elif self.config.method == "symmetric_kl":
                # Symmetric KL (always non-negative by construction)
                ref_probs = F.softmax(ref_logits / self.config.temperature, dim=-1)
                cand_probs = F.softmax(cand_logits / self.config.temperature, dim=-1)
                
                # Add epsilon for stability
                ref_probs = ref_probs + self.config.epsilon
                cand_probs = cand_probs + self.config.epsilon
                
                # Renormalize
                ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True)
                cand_probs = cand_probs / cand_probs.sum(dim=-1, keepdim=True)
                
                # Symmetric KL
                kl_ref_cand = (ref_probs * (torch.log(ref_probs) - torch.log(cand_probs))).sum(dim=-1)
                kl_cand_ref = (cand_probs * (torch.log(cand_probs) - torch.log(ref_probs))).sum(dim=-1)
                
                score = 0.5 * (kl_ref_cand.mean() + kl_cand_ref.mean()).item()
                
            else:
                raise ValueError(f"Unknown method: {self.config.method}")
        
        # Clip to configured range for EB CI stability
        a, b = self.config.score_clip
        return float(min(max(score, a), b))
    
    def score_batch(self, ref_model, cand_model, prompts, tokenizer) -> list:
        """Score multiple prompts efficiently"""
        scores = []
        
        for prompt in prompts:
            # Add canonical suffix
            full_text = prompt + " The answer is"
            
            # Tokenize
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            
            # Move to model device
            inputs = {k: v.to(ref_model.device) for k, v in inputs.items()}
            
            # Get prompt length
            prompt_tokens = tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_tokens["input_ids"].shape[1]
            
            # Score
            score = self.score(ref_model, cand_model, inputs, prompt_len)
            scores.append(score)
            
        return scores


# Legacy compatibility classes for existing tests
@dataclass
class ScoringResult:
    """Legacy result class for backward compatibility"""
    score: float  # Final score in [0, 1]
    raw_score: float  # Unclipped score
    per_position_scores: List[float]  # Score at each position
    metadata: Dict[str, Any]


class OptimizedTeacherForcedScorer:
    """Optimized scorer with top-k approximation and penalty removal"""
    
    def __init__(self, config: ScoringConfig):
        self.config = config
        
    def score_models(self, ref_model, cand_model, prompt: str, tokenizer, k: int = 64) -> float:
        """Score models with proper non-negative output"""
        
        # Add canonical suffix for consistency
        full_text = prompt + " The answer is"
        
        # Tokenize
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]
        
        # Move to device
        device = next(ref_model.parameters()).device
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            # Get model outputs
            ref_outputs = ref_model(**inputs)
            cand_outputs = cand_model(**inputs)
            
            # Get logits for positions after prompt
            seq_len = inputs["input_ids"].shape[1]
            max_positions = min(k, seq_len - prompt_len - 1)
            
            if max_positions <= 0:
                return 0.0  # No positions to evaluate
            
            # Extract logits for evaluation positions
            ref_logits = ref_outputs.logits[0, prompt_len:prompt_len + max_positions]
            cand_logits = cand_outputs.logits[0, prompt_len:prompt_len + max_positions]
            
            # Get target tokens
            targets = inputs["input_ids"][0, prompt_len + 1:prompt_len + 1 + max_positions]
            
            if self.config.method == "delta_ce":
                # Cross-entropy difference with absolute value
                ref_log_probs = F.log_softmax(ref_logits / self.config.temperature, dim=-1)
                cand_log_probs = F.log_softmax(cand_logits / self.config.temperature, dim=-1)
                
                # Get log probabilities for actual next tokens
                ref_target_lp = ref_log_probs.gather(1, targets.unsqueeze(1)).squeeze()
                cand_target_lp = cand_log_probs.gather(1, targets.unsqueeze(1)).squeeze()
                
                # Cross-entropy values (negative log prob)
                ref_ce = -ref_target_lp.mean()
                cand_ce = -cand_target_lp.mean()
                
                # Absolute difference ensures non-negative
                score = abs(cand_ce - ref_ce).item()
                
            elif self.config.method == "symmetric_kl":
                # Symmetric KL divergence
                ref_probs = F.softmax(ref_logits / self.config.temperature, dim=-1)
                cand_probs = F.softmax(cand_logits / self.config.temperature, dim=-1)
                
                # Add epsilon and renormalize
                ref_probs = ref_probs + self.config.epsilon
                cand_probs = cand_probs + self.config.epsilon
                ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True)
                cand_probs = cand_probs / cand_probs.sum(dim=-1, keepdim=True)
                
                # Symmetric KL: 0.5 * (KL(p||q) + KL(q||p))
                kl_ref_cand = (ref_probs * (torch.log(ref_probs) - torch.log(cand_probs))).sum(dim=-1)
                kl_cand_ref = (cand_probs * (torch.log(cand_probs) - torch.log(ref_probs))).sum(dim=-1)
                
                score = 0.5 * (kl_ref_cand.mean() + kl_cand_ref.mean()).item()
                
            else:
                raise ValueError(f"Unknown method: {self.config.method}")
        
        # Clip to configured range for stability
        a, b = self.config.score_clip
        return float(min(max(score, a), b))
    
    def score_batch(self, ref_model, cand_model, prompts: List[str], tokenizer, k: int = 64) -> List[float]:
        """Score multiple prompts efficiently"""
        scores = []
        
        for prompt in prompts:
            score = self.score_models(ref_model, cand_model, prompt, tokenizer, k)
            scores.append(score)
            
        return scores


class FastScorer:
    """Fast approximation scorer with top-k optimization"""
    
    def __init__(self, k: int = 50, temperature: float = 1.0, method: str = "delta_ce"):
        self.k = k
        self.temperature = temperature
        self.method = method
        
    def score(self, ref_model, cand_model, prompt: str, tokenizer) -> float:
        """Fast scoring with top-k approximation"""
        
        # Add suffix for consistency
        full_text = prompt + " The answer is"
        
        # Tokenize
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]
        
        # Move to device
        device = next(ref_model.parameters()).device
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            # Get outputs
            ref_outputs = ref_model(**inputs)
            cand_outputs = cand_model(**inputs)
            
            # Evaluate at most k positions after prompt
            seq_len = inputs["input_ids"].shape[1]
            eval_positions = min(self.k, seq_len - prompt_len - 1)
            
            if eval_positions <= 0:
                return 0.0
            
            # Get logits for evaluation
            ref_logits = ref_outputs.logits[0, prompt_len:prompt_len + eval_positions]
            cand_logits = cand_outputs.logits[0, prompt_len:prompt_len + eval_positions]
            
            # Get targets
            targets = inputs["input_ids"][0, prompt_len + 1:prompt_len + 1 + eval_positions]
            
            if self.method == "delta_ce":
                # Simple cross-entropy difference
                ref_log_probs = F.log_softmax(ref_logits / self.temperature, dim=-1)
                cand_log_probs = F.log_softmax(cand_logits / self.temperature, dim=-1)
                
                ref_nll = -ref_log_probs.gather(1, targets.unsqueeze(1)).squeeze()
                cand_nll = -cand_log_probs.gather(1, targets.unsqueeze(1)).squeeze()
                
                # Return absolute difference
                return abs((cand_nll - ref_nll).mean().item())
                
            else:
                # Default to delta_ce
                return self.score(ref_model, cand_model, prompt, tokenizer)
    
    def __call__(self, ref_model, cand_model, prompt: str, tokenizer, K: int = None) -> float:
        """Callable interface for compatibility"""
        return self.score(ref_model, cand_model, prompt, tokenizer)


def create_teacher_forced_challenges(
    prompts: List[str],
    prompt_types: List[str] = None,
    config: ScoringConfig = None
) -> List[Dict[str, Any]]:
    """Create teacher-forced challenges for POT verification"""
    
    if config is None:
        config = ScoringConfig()
    
    if prompt_types is None:
        prompt_types = ["default"] * len(prompts)
    
    canonical_suffixes = {
        "factual": " The answer is",
        "reasoning": " Let me think step by step.",
        "creative": " Once upon a time,",
        "instruction": " Here's how to do it:",
        "analysis": " The key points are:",
        "default": " The response is"
    }
    
    challenges = []
    
    for i, (prompt, ptype) in enumerate(zip(prompts, prompt_types)):
        # Get canonical suffix for this prompt type
        suffix = canonical_suffixes.get(ptype, canonical_suffixes["default"])
        
        challenge = {
            "id": f"tf_{i:06d}",
            "type": "teacher_forced",
            "prompt": prompt,
            "prompt_type": ptype,
            "canonical_suffix": suffix,
            "full_text": prompt + suffix,
            "config": {
                "method": config.method,
                "num_positions": config.num_positions,
                "temperature": config.temperature
            },
            "metadata": {
                "index": i,
                "prompt_length": len(prompt),
                "suffix_length": len(suffix)
            }
        }
        challenges.append(challenge)
    
    return challenges