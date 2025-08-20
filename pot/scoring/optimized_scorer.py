"""
Optimized Teacher-Forced Scoring for Faster Inference
Reduces inference time through batching, caching, and top-k optimization
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import time
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass 
class OptimizedScoringConfig:
    """Optimized scoring configuration"""
    method: str = "delta_ce"
    positions_per_prompt: int = 32
    use_cached_embeddings: bool = True
    use_top_k_only: bool = True
    top_k: int = 100
    batch_positions: bool = True
    batch_size: int = 8  # Process prompts in batches
    temperature: float = 1.0
    max_length: int = 256  # Reduced for speed
    use_amp: bool = True  # Use automatic mixed precision
    compile_model: bool = False  # Use torch.compile if available
    
class OptimizedTeacherForcedScorer:
    """Optimized scorer with caching and batching"""
    
    def __init__(self, config: Optional[OptimizedScoringConfig] = None):
        self.config = config or OptimizedScoringConfig()
        self.embedding_cache = {}
        self.logit_cache = {}
        self.device = None
        self.amp_enabled = self.config.use_amp and torch.cuda.is_available()
        
    def setup_device(self, model: Any):
        """Setup device and optimization settings"""
        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.amp_enabled = False  # AMP not supported on MPS
            else:
                self.device = torch.device("cpu")
                self.amp_enabled = False
            
            logger.info(f"Using device: {self.device}, AMP: {self.amp_enabled}")
        
        # Move model to device if needed
        if next(model.parameters()).device != self.device:
            model = model.to(self.device)
            
        # Compile model if requested and available (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
                
        return model
        
    def score_batch(self,
                    ref_model: Any,
                    cand_model: Any, 
                    prompts: List[str],
                    tokenizer: Any) -> List[float]:
        """Score multiple prompts in a batch for efficiency"""
        
        start_time = time.time()
        
        # Setup devices
        ref_model = self.setup_device(ref_model)
        cand_model = self.setup_device(cand_model)
        
        # Process in smaller batches for memory efficiency
        all_scores = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_scores = self._score_batch_internal(
                ref_model, cand_model, batch_prompts, tokenizer
            )
            all_scores.extend(batch_scores)
        
        elapsed = time.time() - start_time
        logger.debug(f"Scored {len(prompts)} prompts in {elapsed:.3f}s ({elapsed/len(prompts):.3f}s per prompt)")
        
        return all_scores
    
    def _score_batch_internal(self,
                             ref_model: Any,
                             cand_model: Any,
                             prompts: List[str],
                             tokenizer: Any) -> List[float]:
        """Internal batch scoring with optimizations"""
        
        # Prepare prompts with canonical suffixes
        suffixes = [" The answer is"] * len(prompts)
        full_texts = [p + s for p, s in zip(prompts, suffixes)]
        
        # Batch tokenization with optimized settings
        inputs = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prompt lengths for masking
        prompt_lengths = []
        for prompt in prompts:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_lengths.append(len(tokens))
        
        # Use automatic mixed precision if available
        if self.amp_enabled:
            with torch.amp.autocast('cuda'):
                scores = self._compute_scores_amp(
                    ref_model, cand_model, inputs, prompt_lengths
                )
        else:
            with torch.no_grad():
                scores = self._compute_scores_no_amp(
                    ref_model, cand_model, inputs, prompt_lengths
                )
        
        return scores
    
    def _compute_scores_amp(self,
                           ref_model: Any,
                           cand_model: Any,
                           inputs: Dict,
                           prompt_lengths: List[int]) -> List[float]:
        """Compute scores with automatic mixed precision"""
        
        with torch.no_grad():
            # Get logits from both models
            ref_outputs = ref_model(**inputs)
            cand_outputs = cand_model(**inputs)
            
            ref_logits = ref_outputs.logits.float()  # Convert to float32 for stability
            cand_logits = cand_outputs.logits.float()
            
            return self._process_logits(ref_logits, cand_logits, inputs["input_ids"], prompt_lengths)
    
    def _compute_scores_no_amp(self,
                              ref_model: Any,
                              cand_model: Any,
                              inputs: Dict,
                              prompt_lengths: List[int]) -> List[float]:
        """Compute scores without AMP"""
        
        # Get logits from both models
        ref_outputs = ref_model(**inputs)
        cand_outputs = cand_model(**inputs)
        
        ref_logits = ref_outputs.logits
        cand_logits = cand_outputs.logits
        
        return self._process_logits(ref_logits, cand_logits, inputs["input_ids"], prompt_lengths)
    
    def _process_logits(self,
                       ref_logits: torch.Tensor,
                       cand_logits: torch.Tensor,
                       input_ids: torch.Tensor,
                       prompt_lengths: List[int]) -> List[float]:
        """Process logits to compute scores"""
        
        scores = []
        batch_size = ref_logits.shape[0]
        
        for i in range(batch_size):
            if i < len(prompt_lengths):
                score = self._compute_score_optimized(
                    ref_logits[i],
                    cand_logits[i],
                    input_ids[i],
                    prompt_lengths[i]
                )
                scores.append(score)
        
        return scores
    
    def _compute_score_optimized(self,
                                 ref_logits: torch.Tensor,
                                 cand_logits: torch.Tensor,
                                 input_ids: torch.Tensor,
                                 prompt_length: int) -> float:
        """Optimized score computation with top-k approximation"""
        
        # Get evaluation positions
        seq_len = len(input_ids)
        eval_start = prompt_length
        eval_end = min(prompt_length + self.config.positions_per_prompt, seq_len - 1)
        
        if eval_end <= eval_start:
            return 0.0
        
        eval_positions = range(eval_start, eval_end)
        
        if self.config.use_top_k_only:
            # Fast top-k approximation
            return self._compute_top_k_score(
                ref_logits, cand_logits, input_ids, eval_positions
            )
        else:
            # Full computation
            return self._compute_full_score(
                ref_logits, cand_logits, input_ids, eval_positions
            )
    
    def _compute_top_k_score(self,
                            ref_logits: torch.Tensor,
                            cand_logits: torch.Tensor,
                            input_ids: torch.Tensor,
                            eval_positions: range) -> float:
        """Compute score using only top-k tokens for speed"""
        
        scores = []
        
        for pos in eval_positions:
            if pos + 1 >= len(input_ids):
                continue
                
            target = input_ids[pos + 1]
            
            # Get top-k logits from reference model
            ref_top_k_values, ref_top_k_indices = torch.topk(
                ref_logits[pos], min(self.config.top_k, ref_logits[pos].shape[-1])
            )
            
            # Check if target is in top-k
            if target in ref_top_k_indices:
                # Get corresponding candidate logits
                cand_logits_subset = cand_logits[pos][ref_top_k_indices]
                
                # Find target position in top-k
                target_mask = ref_top_k_indices == target
                target_idx = target_mask.nonzero(as_tuple=True)[0][0]
                
                # Compute log softmax on subset
                ref_log_probs = F.log_softmax(ref_top_k_values / self.config.temperature, dim=-1)
                cand_log_probs = F.log_softmax(cand_logits_subset / self.config.temperature, dim=-1)
                
                # Get CE difference
                ref_lp = ref_log_probs[target_idx].item()
                cand_lp = cand_log_probs[target_idx].item()
                ce_diff = abs(ref_lp - cand_lp)  # Fixed: should be 0 for identical models
                scores.append(ce_diff)
            # Skip tokens not in top-k instead of penalizing
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _compute_full_score(self,
                           ref_logits: torch.Tensor,
                           cand_logits: torch.Tensor,
                           input_ids: torch.Tensor,
                           eval_positions: range) -> float:
        """Full CE difference computation"""
        
        scores = []
        
        for pos in eval_positions:
            if pos + 1 >= len(input_ids):
                continue
                
            target = input_ids[pos + 1]
            
            # Full softmax computation
            ref_log_probs = F.log_softmax(ref_logits[pos] / self.config.temperature, dim=-1)
            cand_log_probs = F.log_softmax(cand_logits[pos] / self.config.temperature, dim=-1)
            
            # Get target log probabilities
            ref_lp = ref_log_probs[target].item()
            cand_lp = cand_log_probs[target].item()
            
            # CE difference
            ce_diff = abs(-cand_lp + ref_lp)
            scores.append(ce_diff)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def warmup_cache(self, model: Any, prompts: List[str], tokenizer: Any):
        """Pre-compute embeddings for common prompts to speed up inference"""
        
        if not self.config.use_cached_embeddings:
            return
            
        logger.info(f"Warming up cache with {min(len(prompts), 10)} prompts")
        
        model = self.setup_device(model)
        
        with torch.no_grad():
            for prompt in prompts[:10]:  # Cache first 10 prompts
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get outputs with hidden states
                outputs = model(**inputs, output_hidden_states=True)
                
                # Cache embeddings
                cache_key = hash(prompt[:100])  # Use first 100 chars as key
                self.embedding_cache[cache_key] = {
                    'hidden_states': outputs.hidden_states[-1].cpu(),
                    'logits': outputs.logits.cpu()
                }
        
        logger.info(f"Cache warmed up with {len(self.embedding_cache)} entries")
    
    def clear_cache(self):
        """Clear all caches to free memory"""
        self.embedding_cache.clear()
        self.logit_cache.clear()
        logger.info("Caches cleared")
    
    @staticmethod
    def benchmark_configurations() -> Dict[str, OptimizedScoringConfig]:
        """Return different configuration presets for benchmarking"""
        return {
            "fastest": OptimizedScoringConfig(
                use_top_k_only=True,
                top_k=50,
                batch_size=16,
                positions_per_prompt=16,
                max_length=128
            ),
            "balanced": OptimizedScoringConfig(
                use_top_k_only=True,
                top_k=100,
                batch_size=8,
                positions_per_prompt=32,
                max_length=256
            ),
            "accurate": OptimizedScoringConfig(
                use_top_k_only=False,
                batch_size=4,
                positions_per_prompt=64,
                max_length=512
            )
        }


class FastScorer:
    """Simplified fast scorer for maximum speed"""
    
    def __init__(self, k: int = 32, top_k: int = 50):
        self.k = k
        self.top_k = top_k
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def score(self,
              ref_model: Any,
              cand_model: Any,
              prompt: str,
              tokenizer: Any) -> float:
        """Fast single prompt scoring - compares model outputs directly"""
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        model_device = next(ref_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # For scoring, we'll compare the logits at each position
        # No need for artificial continuations
        
        with torch.no_grad():
            # Get logits
            ref_out = ref_model(**inputs)
            cand_out = cand_model(**inputs)
            
            # Compare logits at each position in the prompt
            scores = []
            seq_len = inputs["input_ids"].shape[1]
            
            # Score all positions except the last (no next token for it)
            for i in range(min(self.k, seq_len - 1)):
                # Get top-k logits for efficiency
                ref_topk, ref_idx = torch.topk(ref_out.logits[0, i], min(self.top_k, ref_out.logits.shape[-1]))
                cand_topk, cand_idx = torch.topk(cand_out.logits[0, i], min(self.top_k, cand_out.logits.shape[-1]))
                
                # Convert to probabilities
                ref_probs = F.softmax(ref_topk, dim=-1)
                cand_probs = F.softmax(cand_topk, dim=-1)
                
                # Compute KL divergence approximation on top-k
                # For identical models, this should be 0
                kl = 0.0
                for j in range(len(ref_idx)):
                    if ref_idx[j] in cand_idx:
                        cand_j = (cand_idx == ref_idx[j]).nonzero(as_tuple=True)[0]
                        if len(cand_j) > 0:
                            kl += ref_probs[j] * (ref_probs[j].log() - cand_probs[cand_j[0]].log())
                
                scores.append(float(kl.abs().item()))
            
            return float(np.mean(scores)) if scores else 0.0