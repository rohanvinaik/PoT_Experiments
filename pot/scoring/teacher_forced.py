import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import json

@dataclass
class ScoringConfig:
    method: str = "delta_ce"  # "delta_ce", "symmetric_kl", "js_divergence"
    num_positions: int = 10  # Number of token positions to evaluate
    temperature: float = 1.0  # Temperature for softmax
    epsilon: float = 1e-8  # Numerical stability
    use_canonical_suffix: bool = True  # Use fixed suffix vs model's own
    max_seq_length: int = 512  # Max sequence length

@dataclass
class ScoringResult:
    score: float  # Final score in [0, 1]
    raw_score: float  # Unclipped score
    per_position_scores: List[float]  # Score at each position
    metadata: Dict[str, Any]

class TeacherForcedScorer:
    def __init__(self, config: ScoringConfig):
        self.config = config
        self.canonical_suffixes = self._init_canonical_suffixes()
        
    def _init_canonical_suffixes(self) -> Dict[str, str]:
        """Initialize canonical suffixes for different prompt types"""
        return {
            "factual": " The answer is",
            "reasoning": " Let me think step by step.",
            "creative": " Once upon a time,",
            "instruction": " Here's how to do it:",
            "analysis": " The key points are:",
            "default": " The response is"
        }
    
    def score_models(self,
                     reference_model: Any,
                     candidate_model: Any,
                     prompt: str,
                     prompt_type: str = "default",
                     tokenizer: Any = None) -> ScoringResult:
        """Score candidate against reference using teacher forcing"""
        
        # Get canonical continuation
        if self.config.use_canonical_suffix:
            suffix = self.canonical_suffixes.get(prompt_type, self.canonical_suffixes["default"])
            full_text = prompt + suffix
        else:
            # Use reference model's generation (more expensive)
            with torch.no_grad():
                suffix = self._generate_suffix(reference_model, prompt, tokenizer)
                full_text = prompt + suffix
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
        full_inputs = tokenizer(full_text, return_tensors="pt", padding=False, truncation=False)
        
        prompt_length = inputs["input_ids"].shape[1]
        total_length = full_inputs["input_ids"].shape[1]
        
        # Get positions to evaluate (after prompt)
        eval_positions = min(self.config.num_positions, total_length - prompt_length)
        
        # Get log probabilities from both models
        ref_logprobs = self._get_logprobs(reference_model, full_inputs, 
                                         prompt_length, eval_positions)
        cand_logprobs = self._get_logprobs(candidate_model, full_inputs,
                                          prompt_length, eval_positions)
        
        # Compute score based on method
        if self.config.method == "delta_ce":
            score = self._compute_delta_ce(ref_logprobs, cand_logprobs, full_inputs,
                                          prompt_length, eval_positions)
        elif self.config.method == "symmetric_kl":
            score = self._compute_symmetric_kl(ref_logprobs, cand_logprobs)
        elif self.config.method == "js_divergence":
            score = self._compute_js_divergence(ref_logprobs, cand_logprobs)
        else:
            raise ValueError(f"Unknown scoring method: {self.config.method}")
        
        return score
    
    def _get_logprobs(self,
                     model: Any,
                     inputs: Dict[str, torch.Tensor],
                     prompt_length: int,
                     eval_positions: int) -> torch.Tensor:
        """Get log probabilities for specified positions"""
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Apply temperature
            logits = logits / self.config.temperature
            
            # Get log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Extract positions after prompt
            eval_log_probs = log_probs[0, prompt_length:prompt_length+eval_positions, :]
            
        return eval_log_probs
    
    def _compute_delta_ce(self,
                         ref_logprobs: torch.Tensor,
                         cand_logprobs: torch.Tensor,
                         inputs: Dict[str, torch.Tensor],
                         prompt_length: int,
                         eval_positions: int) -> ScoringResult:
        """Compute difference in cross-entropy: CE(candidate) - CE(reference)"""
        
        # Get target token IDs
        target_ids = inputs["input_ids"][0, prompt_length+1:prompt_length+1+eval_positions]
        
        per_position_scores = []
        
        for pos in range(min(eval_positions, len(target_ids))):
            target_id = target_ids[pos]
            
            # Get log probs for actual next token
            ref_lp = ref_logprobs[pos, target_id].item()
            cand_lp = cand_logprobs[pos, target_id].item()
            
            # CE difference (negative log prob difference)
            ce_diff = (-cand_lp) - (-ref_lp)  # CE(cand) - CE(ref)
            per_position_scores.append(ce_diff)
        
        # Average across positions
        raw_score = np.mean(per_position_scores) if per_position_scores else 0.0
        
        # Normalize to [0, 1] - assuming max CE diff of 10
        normalized_score = np.clip(np.abs(raw_score) / 10.0, 0.0, 1.0)
        
        return ScoringResult(
            score=normalized_score,
            raw_score=raw_score,
            per_position_scores=per_position_scores,
            metadata={
                "method": "delta_ce",
                "eval_positions": eval_positions,
                "mean_ce_diff": raw_score
            }
        )
    
    def _compute_symmetric_kl(self,
                             ref_logprobs: torch.Tensor,
                             cand_logprobs: torch.Tensor) -> ScoringResult:
        """Compute symmetric KL divergence at each position"""
        
        per_position_scores = []
        
        for pos in range(ref_logprobs.shape[0]):
            # Convert to probabilities
            ref_probs = torch.exp(ref_logprobs[pos])
            cand_probs = torch.exp(cand_logprobs[pos])
            
            # Add epsilon for numerical stability
            ref_probs = ref_probs + self.config.epsilon
            cand_probs = cand_probs + self.config.epsilon
            
            # Renormalize
            ref_probs = ref_probs / ref_probs.sum()
            cand_probs = cand_probs / cand_probs.sum()
            
            # KL(ref || cand)
            kl_ref_cand = (ref_probs * (torch.log(ref_probs) - torch.log(cand_probs))).sum()
            
            # KL(cand || ref)
            kl_cand_ref = (cand_probs * (torch.log(cand_probs) - torch.log(ref_probs))).sum()
            
            # Symmetric KL
            sym_kl = 0.5 * (kl_ref_cand + kl_cand_ref)
            per_position_scores.append(sym_kl.item())
        
        # Average across positions
        raw_score = np.mean(per_position_scores) if per_position_scores else 0.0
        
        # Normalize to [0, 1] - assuming max KL of 5
        normalized_score = np.clip(raw_score / 5.0, 0.0, 1.0)
        
        return ScoringResult(
            score=normalized_score,
            raw_score=raw_score,
            per_position_scores=per_position_scores,
            metadata={
                "method": "symmetric_kl",
                "eval_positions": len(per_position_scores),
                "mean_kl": raw_score
            }
        )
    
    def _compute_js_divergence(self,
                               ref_logprobs: torch.Tensor,
                               cand_logprobs: torch.Tensor) -> ScoringResult:
        """Compute Jensen-Shannon divergence at each position"""
        
        per_position_scores = []
        
        for pos in range(ref_logprobs.shape[0]):
            # Convert to probabilities
            ref_probs = torch.exp(ref_logprobs[pos])
            cand_probs = torch.exp(cand_logprobs[pos])
            
            # Add epsilon and renormalize
            ref_probs = (ref_probs + self.config.epsilon)
            ref_probs = ref_probs / ref_probs.sum()
            cand_probs = (cand_probs + self.config.epsilon)
            cand_probs = cand_probs / cand_probs.sum()
            
            # Average distribution
            avg_probs = 0.5 * (ref_probs + cand_probs)
            
            # JS divergence
            kl_ref_avg = (ref_probs * (torch.log(ref_probs) - torch.log(avg_probs))).sum()
            kl_cand_avg = (cand_probs * (torch.log(cand_probs) - torch.log(avg_probs))).sum()
            js_div = 0.5 * (kl_ref_avg + kl_cand_avg)
            
            per_position_scores.append(js_div.item())
        
        # Average across positions
        raw_score = np.mean(per_position_scores) if per_position_scores else 0.0
        
        # JS divergence is already in [0, log(2)], normalize to [0, 1]
        normalized_score = np.clip(raw_score / np.log(2), 0.0, 1.0)
        
        return ScoringResult(
            score=normalized_score,
            raw_score=raw_score,
            per_position_scores=per_position_scores,
            metadata={
                "method": "js_divergence",
                "eval_positions": len(per_position_scores),
                "mean_js": raw_score
            }
        )
    
    def batch_score(self,
                   reference_model: Any,
                   candidate_model: Any,
                   prompts: List[str],
                   prompt_types: List[str],
                   tokenizer: Any,
                   batch_size: int = 8) -> List[ScoringResult]:
        """Batch scoring for efficiency"""
        
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_types = prompt_types[i:i+batch_size]
            
            # Prepare batch with canonical suffixes
            batch_texts = []
            prompt_lengths = []
            for prompt, ptype in zip(batch_prompts, batch_types):
                suffix = self.canonical_suffixes.get(ptype, self.canonical_suffixes["default"])
                batch_texts.append(prompt + suffix)
                # Store prompt length for later
                prompt_inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
                prompt_lengths.append(prompt_inputs["input_ids"].shape[1])
            
            # Tokenize batch
            batch_inputs = tokenizer(batch_texts, return_tensors="pt", 
                                    padding=True, truncation=True,
                                    max_length=self.config.max_seq_length)
            
            # Process batch
            with torch.no_grad():
                ref_outputs = reference_model(**batch_inputs)
                cand_outputs = candidate_model(**batch_inputs)
                
                # Score each item in batch
                for j, (prompt, ptype, prompt_len) in enumerate(zip(batch_prompts, batch_types, prompt_lengths)):
                    # Extract individual scores
                    result = self._score_single_from_batch(
                        ref_outputs, cand_outputs, batch_inputs,
                        j, prompt_len, ptype
                    )
                    results.append(result)
        
        return results
    
    def _score_single_from_batch(self,
                                 ref_outputs: Any,
                                 cand_outputs: Any,
                                 batch_inputs: Dict[str, torch.Tensor],
                                 batch_idx: int,
                                 prompt_length: int,
                                 prompt_type: str) -> ScoringResult:
        """Score a single item from batch outputs"""
        
        # Extract logits for this item
        ref_logits = ref_outputs.logits[batch_idx:batch_idx+1]
        cand_logits = cand_outputs.logits[batch_idx:batch_idx+1]
        
        # Apply temperature and get log probs
        ref_logits = ref_logits / self.config.temperature
        cand_logits = cand_logits / self.config.temperature
        
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        cand_log_probs = F.log_softmax(cand_logits, dim=-1)
        
        # Get sequence length for this item
        seq_len = (batch_inputs["attention_mask"][batch_idx] == 1).sum().item()
        eval_positions = min(self.config.num_positions, seq_len - prompt_length)
        
        # Extract evaluation positions
        ref_eval = ref_log_probs[0, prompt_length:prompt_length+eval_positions, :]
        cand_eval = cand_log_probs[0, prompt_length:prompt_length+eval_positions, :]
        
        # Compute score based on method
        if self.config.method == "delta_ce":
            # Need the actual token IDs for CE
            single_inputs = {
                "input_ids": batch_inputs["input_ids"][batch_idx:batch_idx+1]
            }
            return self._compute_delta_ce(ref_eval, cand_eval, single_inputs,
                                         prompt_length, eval_positions)
        elif self.config.method == "symmetric_kl":
            return self._compute_symmetric_kl(ref_eval, cand_eval)
        elif self.config.method == "js_divergence":
            return self._compute_js_divergence(ref_eval, cand_eval)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
    
    def _generate_suffix(self, model: Any, prompt: str, tokenizer: Any, max_length: int = 20) -> str:
        """Generate a suffix using the model (fallback when not using canonical)"""
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=self.config.temperature,
                do_sample=False,  # Greedy for determinism
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
        suffix = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return suffix


class FastTeacherForcedScorer(TeacherForcedScorer):
    """Optimized version with caching and approximations"""
    
    def __init__(self, config: ScoringConfig):
        super().__init__(config)
        self.cache = {}  # Cache computed distributions
        self.use_top_k = True  # Only consider top-k tokens
        self.top_k = 100
        
    def _get_topk_logprobs(self,
                           model: Any,
                           inputs: Dict[str, torch.Tensor],
                           prompt_length: int,
                           eval_positions: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get only top-k log probabilities for efficiency"""
        
        cache_key = (id(model), inputs["input_ids"].cpu().numpy().tobytes())
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits / self.config.temperature
            
            # Get top-k for each position
            eval_logits = logits[0, prompt_length:prompt_length+eval_positions, :]
            
            # Get top-k values and indices
            topk_values, topk_indices = torch.topk(eval_logits, self.top_k, dim=-1)
            topk_log_probs = F.log_softmax(topk_values, dim=-1)
            
            result = (topk_log_probs, topk_indices)
            self.cache[cache_key] = result
            
        return result
    
    def _compute_fast_symmetric_kl(self,
                                   ref_topk: Tuple[torch.Tensor, torch.Tensor],
                                   cand_topk: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Fast approximate symmetric KL using only top-k tokens"""
        
        ref_logprobs, ref_indices = ref_topk
        cand_logprobs, cand_indices = cand_topk
        
        scores = []
        
        for pos in range(ref_logprobs.shape[0]):
            # Find common tokens in top-k
            ref_idx_set = set(ref_indices[pos].tolist())
            cand_idx_set = set(cand_indices[pos].tolist())
            common_indices = ref_idx_set.intersection(cand_idx_set)
            
            if len(common_indices) < 10:  # Too few common tokens
                # Use full computation fallback
                scores.append(1.0)  # Max divergence
                continue
            
            # Approximate KL on common tokens
            approx_kl = 0.0
            for idx in common_indices:
                ref_pos = (ref_indices[pos] == idx).nonzero().item()
                cand_pos = (cand_indices[pos] == idx).nonzero().item()
                
                ref_lp = ref_logprobs[pos, ref_pos]
                cand_lp = cand_logprobs[pos, cand_pos]
                
                approx_kl += 0.5 * (
                    torch.exp(ref_lp) * (ref_lp - cand_lp) +
                    torch.exp(cand_lp) * (cand_lp - ref_lp)
                ).item()
            
            scores.append(approx_kl)
        
        return np.mean(scores)
    
    def clear_cache(self):
        """Clear the cache to free memory"""
        self.cache.clear()


def create_teacher_forced_challenges(
    prompts: List[str],
    prompt_types: List[str],
    config: ScoringConfig
) -> List[Dict[str, Any]]:
    """Create teacher-forced challenges for POT verification"""
    
    scorer = TeacherForcedScorer(config)
    challenges = []
    
    for i, (prompt, ptype) in enumerate(zip(prompts, prompt_types)):
        # Get canonical suffix for this prompt type
        suffix = scorer.canonical_suffixes.get(ptype, scorer.canonical_suffixes["default"])
        
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