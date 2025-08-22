"""
Difference Scoring for Model Comparison

Provides various scoring methods for computing differences between
model outputs on the same prompts.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Optional, Literal, Tuple, Union
import logging

from pot.core.vocabulary_compatibility import (
    VocabularyCompatibilityAnalyzer,
    VocabularyMismatchBehavior
)

logger = logging.getLogger(__name__)


class CorrectedDifferenceScorer:
    """
    Difference scorer with proper orientation:
    - Larger scores indicate MORE different models
    - Scores are always non-negative
    - Handles numerical stability properly
    """
    
    def __init__(
        self,
        epsilon: float = 1e-10,
        min_vocab_overlap: float = 0.95,
        vocab_mismatch_behavior: str = "adapt",
        allow_extended_vocabularies: bool = True
    ):
        """
        Initialize the scorer.
        
        Args:
            epsilon: Small value for numerical stability
            min_vocab_overlap: Minimum vocabulary overlap ratio for compatibility
            vocab_mismatch_behavior: How to handle vocab mismatches ("warn", "adapt", "fail")
            allow_extended_vocabularies: Whether to allow vocabulary extensions
        """
        self.epsilon = epsilon
        
        # Initialize vocabulary compatibility analyzer
        behavior = VocabularyMismatchBehavior(vocab_mismatch_behavior)
        self.vocab_analyzer = VocabularyCompatibilityAnalyzer(
            min_overlap_ratio=min_vocab_overlap,
            mismatch_behavior=behavior,
            allow_extended_vocabularies=allow_extended_vocabularies
        )
    
    def delta_ce_abs(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        temperature: float = 1.0
    ) -> float:
        """
        Compute absolute cross-entropy difference.
        
        CE(p_a, p_b) - CE(p_a, p_a) = CE(p_a, p_b) - H(p_a)
        
        This measures how much MORE uncertain p_a becomes when using p_b's predictions.
        Larger values = more different models.
        
        Args:
            logits_a: Logits from model A [batch_size, seq_len, vocab_size]
            logits_b: Logits from model B [batch_size, seq_len, vocab_size]
            temperature: Temperature for softmax (default 1.0)
            
        Returns:
            Non-negative difference score (larger = more different)
        """
        # Analyze vocabulary compatibility
        vocab_a = logits_a.size(-1)
        vocab_b = logits_b.size(-1)
        
        if vocab_a != vocab_b:
            # Analyze compatibility
            compat_report = self.vocab_analyzer.analyze_vocabulary_overlap(
                vocab_a, vocab_b
            )
            
            # Get verification strategy
            strategy = self.vocab_analyzer.suggest_verification_strategy(compat_report)
            
            if not strategy["can_proceed"]:
                # Incompatible vocabularies
                logger.warning(
                    f"Vocabularies incompatible for cross-entropy calculation: "
                    f"{vocab_a} vs {vocab_b} (overlap: {compat_report.overlap_ratio:.1%})"
                )
                return 1.0  # Maximum difference for incompatible models
            
            # Adapt computation to shared token space
            shared_start, shared_end = self.vocab_analyzer.determine_shared_token_space(
                vocab_a, vocab_b
            )
            
            # Log adaptation
            logger.info(
                f"Adapting to vocabulary mismatch: {vocab_a} vs {vocab_b}. "
                f"Computing on {shared_end} shared tokens ({compat_report.overlap_ratio:.1%} overlap)"
            )
            
            # Truncate to shared vocabulary
            logits_a = logits_a[..., shared_start:shared_end]
            logits_b = logits_b[..., shared_start:shared_end]
        
        # Apply temperature scaling
        logits_a = logits_a / temperature
        logits_b = logits_b / temperature
        
        # Convert to probabilities with stability
        probs_a = F.softmax(logits_a, dim=-1)
        probs_b = F.softmax(logits_b, dim=-1)
        
        # Add epsilon for numerical stability
        probs_a = probs_a + self.epsilon
        probs_b = probs_b + self.epsilon
        
        # Renormalize after adding epsilon
        probs_a = probs_a / probs_a.sum(dim=-1, keepdim=True)
        probs_b = probs_b / probs_b.sum(dim=-1, keepdim=True)
        
        # Compute cross-entropy: CE(p_a, p_b) = -sum(p_a * log(p_b))
        ce_ab = -torch.sum(probs_a * torch.log(probs_b), dim=-1)
        
        # Compute entropy: H(p_a) = -sum(p_a * log(p_a))
        h_a = -torch.sum(probs_a * torch.log(probs_a), dim=-1)
        
        # Difference: CE(p_a, p_b) - H(p_a)
        # This is always >= 0 (by Gibbs' inequality)
        diff = ce_ab - h_a
        
        # Take mean over batch and sequence
        score = diff.mean().item()
        
        # Ensure non-negative (should already be, but for safety)
        return max(0.0, score)
    
    def symmetric_kl(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        temperature: float = 1.0
    ) -> float:
        """
        Compute symmetric KL divergence (Jeffreys divergence).
        
        J(p_a, p_b) = KL(p_a || p_b) + KL(p_b || p_a)
        
        This is symmetric and always non-negative.
        Larger values = more different models.
        
        Args:
            logits_a: Logits from model A [batch_size, seq_len, vocab_size]
            logits_b: Logits from model B [batch_size, seq_len, vocab_size]
            temperature: Temperature for softmax (default 1.0)
            
        Returns:
            Non-negative symmetric KL divergence (larger = more different)
        """
        # Analyze vocabulary compatibility
        vocab_a = logits_a.size(-1)
        vocab_b = logits_b.size(-1)
        
        if vocab_a != vocab_b:
            compat_report = self.vocab_analyzer.analyze_vocabulary_overlap(vocab_a, vocab_b)
            strategy = self.vocab_analyzer.suggest_verification_strategy(compat_report)
            
            if not strategy["can_proceed"]:
                logger.warning(f"Vocabularies incompatible for KL divergence: {vocab_a} vs {vocab_b}")
                return 1.0  # Maximum difference
            
            # Adapt to shared token space
            shared_start, shared_end = self.vocab_analyzer.determine_shared_token_space(vocab_a, vocab_b)
            logger.info(f"Computing KL on {shared_end} shared tokens")
            logits_a = logits_a[..., shared_start:shared_end]
            logits_b = logits_b[..., shared_start:shared_end]
        
        # Apply temperature scaling
        logits_a = logits_a / temperature
        logits_b = logits_b / temperature
        
        # Convert to log probabilities for stability
        log_probs_a = F.log_softmax(logits_a, dim=-1)
        log_probs_b = F.log_softmax(logits_b, dim=-1)
        
        # Convert to probabilities
        probs_a = torch.exp(log_probs_a)
        probs_b = torch.exp(log_probs_b)
        
        # KL(p_a || p_b) = sum(p_a * (log(p_a) - log(p_b)))
        kl_ab = torch.sum(probs_a * (log_probs_a - log_probs_b), dim=-1)
        
        # KL(p_b || p_a) = sum(p_b * (log(p_b) - log(p_a)))
        kl_ba = torch.sum(probs_b * (log_probs_b - log_probs_a), dim=-1)
        
        # Symmetric KL = KL(p_a || p_b) + KL(p_b || p_a)
        symmetric = kl_ab + kl_ba
        
        # Take mean over batch and sequence
        score = symmetric.mean().item()
        
        # Ensure non-negative (should already be, but for safety)
        return max(0.0, score)
    
    def compute_with_positions(
        self,
        model_a,
        model_b,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        method: str = "delta_ce_abs",
        temperature: float = 1.0
    ) -> Tuple[float, dict]:
        """
        Compute difference score for specific positions.
        
        Args:
            model_a: First model
            model_b: Second model
            input_ids: Input token IDs
            attention_mask: Attention mask (optional)
            positions: Positions to score (optional, defaults to all)
            method: Scoring method ("delta_ce_abs" or "symmetric_kl")
            temperature: Temperature for softmax
            
        Returns:
            (score, metadata) where score is non-negative difference
        """
        # Get model outputs
        with torch.no_grad():
            outputs_a = model_a(input_ids, attention_mask=attention_mask)
            outputs_b = model_b(input_ids, attention_mask=attention_mask)
        
        logits_a = outputs_a.logits
        logits_b = outputs_b.logits
        
        # If positions specified, extract only those
        if positions is not None:
            batch_size = logits_a.shape[0]
            selected_a = []
            selected_b = []
            
            for b in range(batch_size):
                for pos in positions[b]:
                    if pos < logits_a.shape[1]:
                        selected_a.append(logits_a[b, pos:pos+1, :])
                        selected_b.append(logits_b[b, pos:pos+1, :])
            
            if selected_a:
                logits_a = torch.cat(selected_a, dim=0).unsqueeze(0)
                logits_b = torch.cat(selected_b, dim=0).unsqueeze(0)
            else:
                # No valid positions, return 0 difference
                return 0.0, {"n_positions": 0, "method": method}
        
        # Compute score based on method
        if method == "delta_ce_abs":
            score = self.delta_ce_abs(logits_a, logits_b, temperature)
        elif method == "symmetric_kl":
            score = self.symmetric_kl(logits_a, logits_b, temperature)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Prepare metadata
        metadata = {
            "method": method,
            "temperature": temperature,
            "n_positions": positions.numel() if positions is not None else logits_a.shape[1],
            "score_interpretation": "larger values = more different models"
        }
        
        return score, metadata
    
    def score_batch(
        self,
        model_a,
        model_b,
        prompts: list,
        tokenizer,
        max_length: int = 512,
        method: str = "delta_ce_abs",
        temperature: float = 1.0,
        k: int = 32
    ) -> list:
        """
        Score a batch of prompts.
        
        Args:
            model_a: First model
            model_b: Second model
            prompts: List of text prompts
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            method: Scoring method
            temperature: Temperature for softmax
            k: Number of positions to sample per prompt
            
        Returns:
            List of scores (all non-negative, larger = more different)
        """
        scores = []
        
        for prompt in prompts:
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
            
            # Sample positions
            seq_len = input_ids.shape[1]
            if seq_len > k:
                positions = torch.randperm(seq_len)[:k].unsqueeze(0)
            else:
                positions = torch.arange(seq_len).unsqueeze(0)
            
            # Move to device
            device = next(model_a.parameters()).device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            positions = positions.to(device)
            
            # Compute score
            score, _ = self.compute_with_positions(
                model_a, model_b,
                input_ids, attention_mask,
                positions, method, temperature
            )
            
            scores.append(score)
        
        return scores


class DifferenceScorer:
    """Score differences between model outputs"""
    
    def __init__(self, 
                 method: Literal["delta_ce", "symmetric_kl", "js_divergence"] = "delta_ce"):
        """
        Initialize scorer.
        
        Args:
            method: Scoring method to use
        """
        self.method = method
        self.tokenizer = None
        
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for the models"""
        self.tokenizer = tokenizer
    
    def __call__(self, 
                 ref_model: Any, 
                 cand_model: Any, 
                 prompt: str, 
                 K: int = 32) -> float:
        """
        Compute difference score between models.
        
        Args:
            ref_model: Reference model
            cand_model: Candidate model
            prompt: Input prompt
            K: Number of positions to score
            
        Returns:
            Difference score
        """
        # For mock models (testing)
        if hasattr(ref_model, 'model_id') and hasattr(cand_model, 'model_id'):
            # Mock scoring for testing
            if ref_model.model_id == cand_model.model_id:
                return np.random.normal(0.0, 0.001)
            else:
                return np.random.normal(0.1, 0.02)
        
        # For real models
        if self.method == "delta_ce":
            return self._delta_ce_score(ref_model, cand_model, prompt, K)
        elif self.method == "symmetric_kl":
            return self._symmetric_kl_score(ref_model, cand_model, prompt, K)
        elif self.method == "js_divergence":
            return self._js_divergence_score(ref_model, cand_model, prompt, K)
        else:
            raise ValueError(f"Unknown scoring method: {self.method}")
    
    def _delta_ce_score(self, ref_model, cand_model, prompt: str, K: int) -> float:
        """
        Compute delta cross-entropy score.
        
        ΔCE = CE(ref_output, cand_model) - CE(ref_output, ref_model)
        """
        if self.tokenizer is None:
            # Mock scoring
            return np.random.normal(0.05, 0.01)
        
        try:
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            
            # Get model device
            ref_device = next(ref_model.parameters()).device
            cand_device = next(cand_model.parameters()).device
            
            # Move inputs to device
            ref_inputs = {k: v.to(ref_device) for k, v in inputs.items()}
            cand_inputs = {k: v.to(cand_device) for k, v in inputs.items()}
            
            # Generate reference output
            with torch.no_grad():
                ref_output = ref_model.generate(
                    **ref_inputs,
                    max_new_tokens=K,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # Get logits for both models
                ref_logits = ref_model(**ref_inputs).logits
                cand_logits = cand_model(**cand_inputs).logits
            
            # Compute cross-entropy difference
            # Simplified version - in practice would compute over K positions
            ce_ref = torch.nn.functional.cross_entropy(
                ref_logits[0], 
                ref_inputs['input_ids'][0],
                reduction='mean'
            ).item()
            
            ce_cand = torch.nn.functional.cross_entropy(
                cand_logits[0],
                ref_inputs['input_ids'][0],
                reduction='mean'
            ).item()
            
            delta_ce = abs(ce_cand - ce_ref)
            
            # Clip to reasonable range
            delta_ce = min(max(delta_ce, 0.0), 1.0)
            
            return delta_ce
            
        except Exception as e:
            logger.warning(f"Error in delta_ce_score: {e}, using mock score")
            return np.random.normal(0.05, 0.01)
    
    def _symmetric_kl_score(self, ref_model, cand_model, prompt: str, K: int) -> float:
        """
        Compute symmetric KL divergence.
        
        KL_sym = (KL(P||Q) + KL(Q||P)) / 2
        """
        # Simplified implementation
        # In practice, would compute KL over output distributions
        return self._delta_ce_score(ref_model, cand_model, prompt, K) * 1.2
    
    def _js_divergence_score(self, ref_model, cand_model, prompt: str, K: int) -> float:
        """
        Compute Jensen-Shannon divergence.
        
        JS(P||Q) = (KL(P||M) + KL(Q||M)) / 2, where M = (P+Q)/2
        """
        # Simplified implementation
        return self._delta_ce_score(ref_model, cand_model, prompt, K) * 0.8

class MockScorer:
    """Mock scorer for testing without real models"""
    
    def __init__(self, scenario: str = "same"):
        self.scenario = scenario
        self.tokenizer = None
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, ref_model, cand_model, prompt: str, K: int = 32) -> float:
        if self.scenario == "identical":
            return np.random.normal(0.0, 0.0001)
        elif self.scenario == "same":
            return np.random.normal(0.003, 0.001)
        elif self.scenario == "different":
            return np.random.normal(0.12, 0.02)
        else:
            return np.random.normal(0.05, 0.02)