"""
Language Model Verifier for Proof-of-Training
Implements verification protocol from paper Section 3
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
import time

from transformers import AutoTokenizer

from ..core.stats import empirical_bernstein_bound, t_statistic
from ..core.sequential import SequentialTester, SPRTResult
from ..core.challenge import generate_challenges, ChallengeConfig
from .fuzzy_hash import TokenSpaceNormalizer, NGramFuzzyHasher, AdvancedFuzzyHasher
from .models import LM


@dataclass
class LMVerificationResult:
    """Result of language model verification"""
    accepted: bool
    distance: float
    confidence_radius: float
    n_challenges: int
    fuzzy_similarity: float
    time_elapsed: float
    metadata: Dict[str, Any]


class LMVerifier:
    """
    Language Model Verifier
    Implements the verification protocol for language models from paper Section 3
    """
    
    def __init__(self, reference_model: LM, delta: float = 0.01, 
                 use_sequential: bool = True):
        """
        Initialize LM verifier
        
        Args:
            reference_model: Reference language model f*
            delta: Confidence parameter (1-delta confidence)
            use_sequential: Whether to use SPRT for early stopping
        """
        self.reference_model = reference_model
        self.delta = delta
        self.use_sequential = use_sequential
        
        # Initialize tokenizer and normalizer
        self.tokenizer = reference_model.tok
        self.normalizer = TokenSpaceNormalizer(self.tokenizer)
        self.fuzzy_hasher = NGramFuzzyHasher()
        
        # Try to use advanced fuzzy hashing if available
        try:
            self.advanced_hasher = AdvancedFuzzyHasher()
        except ImportError:
            self.advanced_hasher = None
        
        # Sequential tester if enabled
        if use_sequential:
            self.sequential_tester = SequentialTester(
                alpha=delta,
                beta=delta,
                tau0=0.05,  # Expected distance for same model
                tau1=0.2    # Expected distance for different model
            )
    
    def generate_template_challenges(self, n: int, 
                                    master_key: str,
                                    session_nonce: str) -> List[Dict[str, Any]]:
        """
        Generate template-based challenges for language models
        From paper Section 3.2
        """
        # Define template families
        templates = [
            "Complete the following: The {object} is {attribute}",
            "Q: What is {concept}? A:",
            "Translate to {language}: {text}",
            "Summarize: {passage}",
            "Continue the story: {beginning}",
        ]
        
        # Slot values for variation
        slots = {
            "object": ["cat", "house", "tree", "computer", "ocean"],
            "attribute": ["large", "blue", "ancient", "mysterious", "simple"],
            "concept": ["gravity", "democracy", "evolution", "entropy", "recursion"],
            "language": ["French", "Spanish", "German", "Italian", "Portuguese"],
            "text": ["Hello world", "Good morning", "Thank you", "How are you"],
            "passage": ["The quick brown fox...", "Once upon a time...", 
                       "In the beginning...", "It was the best of times..."],
            "beginning": ["The door creaked open", "She looked at the stars",
                         "The letter arrived", "He woke up suddenly"]
        }
        
        config = ChallengeConfig(
            master_key_hex=master_key,
            session_nonce_hex=session_nonce,
            n=n,
            family="lm:templates",
            params={"templates": templates, "slots": slots}
        )
        
        result = generate_challenges(config)
        return result["items"]
    
    def compute_output_distance(self, output1: str, output2: str,
                               method: str = 'fuzzy') -> float:
        """
        Compute distance between two model outputs
        
        Args:
            output1: First model output
            output2: Second model output
            method: Distance computation method
            
        Returns:
            Distance in [0, 1]
        """
        # Tokenize outputs
        tokens1 = self.tokenizer.encode(output1, add_special_tokens=False)
        tokens2 = self.tokenizer.encode(output2, add_special_tokens=False)
        
        # Compute distance using normalizer
        distance = self.normalizer.compute_distance(tokens1, tokens2, method=method)
        
        return distance
    
    def evaluate_challenge(self, model: LM, challenge: Dict[str, Any]) -> Tuple[str, float]:
        """
        Evaluate a single challenge on the model
        
        Returns:
            (model_output, distance_from_reference)
        """
        # Construct prompt from challenge
        if "template" in challenge and "slot_values" in challenge:
            prompt = challenge["template"]
            for slot, value in challenge["slot_values"].items():
                prompt = prompt.replace(f"{{{slot}}}", value)
        else:
            # Fallback for simple text challenges
            prompt = challenge.get("prompt", "Hello")
        
        # Get outputs from both models
        model_output = model.generate(prompt, max_new_tokens=64)
        reference_output = self.reference_model.generate(prompt, max_new_tokens=64)
        
        # Compute distance
        distance = self.compute_output_distance(model_output, reference_output)
        
        return model_output, distance
    
    def verify(self, model: LM, challenges: List[Dict[str, Any]], 
              tolerance: float = 0.1) -> LMVerificationResult:
        """
        Verify a language model against reference
        
        Args:
            model: Model to verify (f)
            challenges: List of challenges to evaluate
            tolerance: Maximum acceptable average distance
            
        Returns:
            LMVerificationResult with verification outcome
        """
        start_time = time.time()
        distances = []
        fuzzy_similarities = []
        
        # Reset sequential tester if using
        if self.use_sequential:
            self.sequential_tester = SequentialTester(
                alpha=self.delta, beta=self.delta,
                tau0=tolerance/2, tau1=tolerance*2
            )
        
        for i, challenge in enumerate(challenges):
            # Evaluate challenge
            model_output, distance = self.evaluate_challenge(model, challenge)
            distances.append(distance)
            
            # Compute fuzzy similarity for additional validation
            ref_output = self.reference_model.generate(
                self._challenge_to_prompt(challenge), 
                max_new_tokens=64
            )
            
            tokens_model = self.tokenizer.encode(model_output, add_special_tokens=False)
            tokens_ref = self.tokenizer.encode(ref_output, add_special_tokens=False)
            
            hash_model = self.fuzzy_hasher.compute_fuzzy_hash(tokens_model)
            hash_ref = self.fuzzy_hasher.compute_fuzzy_hash(tokens_ref)
            
            similarity = self.fuzzy_hasher.jaccard_similarity(hash_model, hash_ref)
            fuzzy_similarities.append(similarity)
            
            # Sequential testing for early stopping
            if self.use_sequential:
                result = self.sequential_tester.update(distance)
                if result.decision != 'continue':
                    # Early stopping
                    distances = result.distances
                    break
        
        # Compute test statistic and confidence radius
        distances_array = np.array(distances)
        test_statistic = t_statistic(distances_array)
        conf_radius = empirical_bernstein_bound(distances_array, self.delta)
        
        # Decision: accept if test statistic + radius <= tolerance
        accepted = (test_statistic + conf_radius) <= tolerance
        
        # If using sequential testing, override with its decision
        if self.use_sequential and self.sequential_tester.decided():
            accepted = self.sequential_tester.accept()
        
        elapsed = time.time() - start_time
        
        metadata = {
            "test_statistic": float(test_statistic),
            "tolerance": tolerance,
            "n_evaluated": len(distances),
            "distance_stats": {
                "mean": float(np.mean(distances_array)),
                "std": float(np.std(distances_array)),
                "min": float(np.min(distances_array)),
                "max": float(np.max(distances_array))
            },
            "fuzzy_stats": {
                "mean": float(np.mean(fuzzy_similarities)),
                "std": float(np.std(fuzzy_similarities))
            }
        }
        
        # Add advanced fuzzy hashing results if available
        if self.advanced_hasher:
            sample_idx = min(5, len(challenges))  # Sample a few for advanced hashing
            advanced_results = []
            
            for idx in range(sample_idx):
                challenge = challenges[idx]
                model_out, _ = self.evaluate_challenge(model, challenge)
                ref_out = self.reference_model.generate(
                    self._challenge_to_prompt(challenge), max_new_tokens=64
                )
                
                tokens_m = self.tokenizer.encode(model_out, add_special_tokens=False)
                tokens_r = self.tokenizer.encode(ref_out, add_special_tokens=False)
                
                sim_scores = self.advanced_hasher.compute_combined_similarity(
                    tokens_m, tokens_r
                )
                advanced_results.append(sim_scores)
            
            metadata["advanced_fuzzy"] = advanced_results
        
        return LMVerificationResult(
            accepted=accepted,
            distance=float(test_statistic),
            confidence_radius=float(conf_radius),
            n_challenges=len(distances),
            fuzzy_similarity=float(np.mean(fuzzy_similarities)),
            time_elapsed=elapsed,
            metadata=metadata
        )
    
    def _challenge_to_prompt(self, challenge: Dict[str, Any]) -> str:
        """Convert challenge dict to prompt string"""
        if "template" in challenge and "slot_values" in challenge:
            prompt = challenge["template"]
            for slot, value in challenge["slot_values"].items():
                prompt = prompt.replace(f"{{{slot}}}", value)
            return prompt
        return challenge.get("prompt", "Hello")
    
    def verify_with_time_tolerance(self, model: LM, 
                                  challenges: List[Dict[str, Any]],
                                  base_tolerance: float = 0.1,
                                  days_elapsed: int = 0,
                                  drift_rate: float = 0.001) -> LMVerificationResult:
        """
        Verify with time-aware tolerance for version drift
        From paper Section 5 on handling model updates
        
        Args:
            model: Model to verify
            challenges: Verification challenges
            base_tolerance: Base tolerance at time 0
            days_elapsed: Days since reference model snapshot
            drift_rate: Expected drift per day
            
        Returns:
            Verification result with adjusted tolerance
        """
        # Adjust tolerance based on time elapsed
        # τ(t) = τ_0 + λ * t
        adjusted_tolerance = base_tolerance + drift_rate * days_elapsed
        
        # Run verification with adjusted tolerance
        result = self.verify(model, challenges, adjusted_tolerance)
        
        # Add time tolerance info to metadata
        result.metadata["time_tolerance"] = {
            "base_tolerance": base_tolerance,
            "days_elapsed": days_elapsed,
            "drift_rate": drift_rate,
            "adjusted_tolerance": adjusted_tolerance
        }
        
        return result


class BatchLMVerifier:
    """
    Batch verification for multiple language models
    Implements efficient batch processing from paper
    """
    
    def __init__(self, reference_model: LM, delta: float = 0.01):
        self.verifier = LMVerifier(reference_model, delta)
    
    def verify_batch(self, models: List[LM], 
                    challenges: List[Dict[str, Any]],
                    tolerance: float = 0.1) -> List[LMVerificationResult]:
        """
        Verify multiple models in batch
        
        Args:
            models: List of models to verify
            challenges: Common challenge set
            tolerance: Acceptance threshold
            
        Returns:
            List of verification results
        """
        results = []
        
        for i, model in enumerate(models):
            print(f"Verifying model {i+1}/{len(models)}...")
            result = self.verifier.verify(model, challenges, tolerance)
            results.append(result)
            
            # Early termination if too many failures
            failures = sum(1 for r in results if not r.accepted)
            if failures > len(models) * 0.5:
                print(f"High failure rate ({failures}/{i+1}), stopping batch")
                break
        
        return results
    
    def adaptive_verify(self, model: LM, 
                       min_challenges: int = 10,
                       max_challenges: int = 100,
                       tolerance: float = 0.1,
                       master_key: str = None,
                       session_nonce: str = None) -> LMVerificationResult:
        """
        Adaptive verification with dynamic challenge count
        Uses sequential testing to determine when to stop
        """
        if master_key is None:
            master_key = "0" * 64  # Default for testing
        if session_nonce is None:
            session_nonce = "1" * 32
        
        # Start with minimum challenges
        challenges = self.verifier.generate_template_challenges(
            min_challenges, master_key, session_nonce
        )
        
        # Initial verification
        result = self.verifier.verify(model, challenges, tolerance)
        
        # If inconclusive, add more challenges
        while (not result.accepted and 
               result.n_challenges < max_challenges and
               abs(result.distance - tolerance) < result.confidence_radius):
            
            # Generate additional challenges
            additional = min(min_challenges, max_challenges - result.n_challenges)
            new_challenges = self.verifier.generate_template_challenges(
                additional, master_key, session_nonce + str(result.n_challenges)
            )
            
            challenges.extend(new_challenges)
            
            # Re-verify with expanded challenge set
            result = self.verifier.verify(model, challenges, tolerance)
        
        return result