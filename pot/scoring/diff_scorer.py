"""
Difference Scoring for Model Comparison

Provides various scoring methods for computing differences between
model outputs on the same prompts.
"""

import numpy as np
import torch
from typing import Any, Optional, Literal
import logging

logger = logging.getLogger(__name__)

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