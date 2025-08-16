"""
Language Model Verifier for Proof-of-Training
Implements verification protocol from paper Section 3
"""

import torch
import numpy as np
import difflib
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
import time

from transformers import AutoTokenizer

from ..core.stats import empirical_bernstein_bound, t_statistic
from ..core.sequential import SequentialTester, SPRTResult
from ..core.challenge import generate_challenges, ChallengeConfig
from ..core.fingerprint import fingerprint_run, FingerprintConfig, FingerprintResult, compare_jacobian_sketches
from ..core.canonicalize import canonicalize_text, canonicalize_logits
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
    fingerprint: Optional[FingerprintResult]  # Behavioral fingerprint
    fingerprint_match: Optional[float]  # Similarity score if reference fingerprint exists
    metadata: Dict[str, Any]


class LMVerifier:
    """
    Language Model Verifier
    Implements the verification protocol for language models from paper Section 3
    """
    
    def __init__(self, reference_model: LM, delta: float = 0.01, 
                 use_sequential: bool = True, use_fingerprinting: bool = True,
                 fingerprint_config: Optional[FingerprintConfig] = None):
        """
        Initialize LM verifier
        
        Args:
            reference_model: Reference language model f*
            delta: Confidence parameter (1-delta confidence)
            use_sequential: Whether to use SPRT for early stopping
            use_fingerprinting: Whether to compute behavioral fingerprints
            fingerprint_config: Configuration for fingerprinting (uses default LM config if None)
        """
        self.reference_model = reference_model
        self.delta = delta
        self.use_sequential = use_sequential
        self.use_fingerprinting = use_fingerprinting
        
        # Set up fingerprinting configuration
        if use_fingerprinting:
            if fingerprint_config is None:
                # Use default language model configuration
                self.fingerprint_config = FingerprintConfig.for_language_model(
                    compute_jacobian=False,  # Jacobians less useful for text
                    include_timing=True,
                    canonicalize_outputs=True
                )
            else:
                self.fingerprint_config = fingerprint_config
            
            # Will compute reference fingerprint lazily
            self.reference_fingerprint = None
        
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
                                    session_nonce: str,
                                    model_id: Optional[str] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Generate template-based challenges for language models
        From paper Section 3.2
        
        Args:
            n: Number of challenges
            master_key: Master key for challenge generation
            session_nonce: Session-specific nonce
            model_id: Optional model identifier for model-specific challenges
            
        Returns:
            Tuple of (prompts, challenge metadata including IDs)
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
            params={"templates": templates, "slots": slots},
            model_id=model_id
        )
        
        result = generate_challenges(config)
        prompts = []
        metadata = []
        
        # Use Challenge objects if available
        challenges = result.get("challenges", None)
        if challenges:
            # New format with Challenge objects
            for challenge in challenges:
                params = challenge.parameters
                # Use the generated prompt
                prompt = params.get("prompt", None)
                if not prompt:
                    # Fallback: construct prompt from template and slots
                    template = params["template"]
                    slot_values = params["slot_values"]
                    prompt = template
                    for slot_name, slot_value in slot_values.items():
                        prompt = prompt.replace(f"{{{slot_name}}}", slot_value)
                
                prompts.append(prompt)
                
                # Store metadata including challenge ID
                metadata.append({
                    "challenge_id": challenge.challenge_id,
                    "index": challenge.index,
                    "template": params.get("template", ""),
                    "slot_values": params.get("slot_values", {}),
                    "prompt": prompt
                })
        else:
            # Backward compatibility: use items
            for idx, item in enumerate(result["items"]):
                # Construct prompt from template if needed
                if "prompt" in item:
                    prompt = item["prompt"]
                else:
                    template = item.get("template", "")
                    slot_values = item.get("slot_values", {})
                    prompt = template
                    for slot_name, slot_value in slot_values.items():
                        prompt = prompt.replace(f"{{{slot_name}}}", slot_value)
                
                prompts.append(prompt)
                metadata.append({
                    "challenge_id": f"legacy_{idx}",
                    "index": idx,
                    "template": item.get("template", ""),
                    "slot_values": item.get("slot_values", {}),
                    "prompt": prompt
                })
        
        return prompts, metadata
    
    def compute_output_distance(self, output1: str, output2: str,
                               method: str = 'fuzzy') -> float:
        """
        Compute distance between two model outputs.

        Args:
            output1: First model output
            output2: Second model output
            method: Distance computation method. Supported values:
                - ``'fuzzy'`` (default): Token-level fuzzy Jaccard distance
                - ``'exact'``: Exact token match
                - ``'weighted'``: Weighted n-gram distance
                - ``'edit'``: Normalized Levenshtein edit distance
                - ``'embedding'``: Cosine distance between token-count embeddings

        Returns:
            Distance in ``[0, 1]`` where 0 indicates identical outputs.
        """
        if method in {"fuzzy", "exact", "weighted"}:
            tokens1 = self.tokenizer.encode(output1, add_special_tokens=False)
            tokens2 = self.tokenizer.encode(output2, add_special_tokens=False)
            distance = self.normalizer.compute_distance(tokens1, tokens2, method=method)
            return distance

        if method == "edit":
            # Normalized edit distance using SequenceMatcher
            return 1.0 - difflib.SequenceMatcher(None, output1, output2).ratio()

        if method == "embedding":
            tokens1 = self.tokenizer.encode(output1, add_special_tokens=False)
            tokens2 = self.tokenizer.encode(output2, add_special_tokens=False)
            norm1 = self.normalizer.normalize_tokens(tokens1)
            norm2 = self.normalizer.normalize_tokens(tokens2)

            vec1 = Counter(norm1)
            vec2 = Counter(norm2)
            all_tokens = set(vec1) | set(vec2)
            if not all_tokens:
                return 0.0

            v1 = np.array([vec1[t] for t in all_tokens], dtype=float)
            v2 = np.array([vec2[t] for t in all_tokens], dtype=float)

            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                return 1.0

            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return 1.0 - cos_sim

        raise ValueError(f"Unknown method: {method}")
    
    def evaluate_challenge(self, model: LM, challenge: Dict[str, Any],
                            method: str = 'fuzzy') -> Tuple[str, float]:
        """
        Evaluate a single challenge on the model.

        Args:
            model: Model to query
            challenge: Challenge specification
            method: Distance computation method (default ``'fuzzy'``)

        Returns:
            Tuple of ``(model_output, distance_from_reference)``
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
        distance = self.compute_output_distance(model_output, reference_output, method=method)
        
        return model_output, distance
    
    def verify(self, model: LM, challenges: List[Dict[str, Any]],
              tolerance: float = 0.1, method: str = 'fuzzy',
              compute_reference_fingerprint: bool = False) -> LMVerificationResult:
        """
        Verify a language model against reference.

        Args:
            model: Model to verify (f)
            challenges: List of challenges to evaluate
            tolerance: Maximum acceptable average distance
            method: Distance computation method (default ``'fuzzy'``)
            compute_reference_fingerprint: Whether to compute reference fingerprint if not already done

        Returns:
            LMVerificationResult with verification outcome
        """
        start_time = time.time()
        distances = []
        fuzzy_similarities = []
        
        # Compute behavioral fingerprint if enabled
        model_fingerprint = None
        fingerprint_similarity = None
        
        if self.use_fingerprinting:
            # Convert challenges to prompts for fingerprinting
            prompts = [self._challenge_to_prompt(ch) for ch in challenges]
            
            # Compute reference fingerprint if needed
            if compute_reference_fingerprint or self.reference_fingerprint is None:
                self.compute_reference_fingerprint(prompts)
            
            # Create a wrapper function for the model being verified
            def model_wrapper(prompt_batch):
                if isinstance(prompt_batch, str):
                    prompt_batch = [prompt_batch]
                
                outputs = []
                for prompt in prompt_batch:
                    output = model.generate(prompt, max_new_tokens=64)
                    # Canonicalize the text output
                    canonical_output = canonicalize_text(output)
                    outputs.append(canonical_output)
                
                return outputs if len(outputs) > 1 else outputs[0]
            
            # Compute fingerprint for the model being verified
            fingerprint_start = time.time()
            model_fingerprint = fingerprint_run(
                model_wrapper,
                prompts,
                self.fingerprint_config
            )
            fingerprint_time = time.time() - fingerprint_start
            
            # Compare fingerprints if reference exists
            if self.reference_fingerprint is not None:
                fingerprint_similarity = self.compute_fingerprint_similarity(
                    self.reference_fingerprint,
                    model_fingerprint
                )
                
                # Log fingerprint metrics for debugging
                print(f"[LM Fingerprint] IO Hash Match: {model_fingerprint.io_hash == self.reference_fingerprint.io_hash}")
                print(f"[LM Fingerprint] Similarity: {fingerprint_similarity:.4f}")
                print(f"[LM Fingerprint] Time: {fingerprint_time:.3f}s")
                
                # Early rejection based on fingerprint (optional, with conservative threshold)
                if fingerprint_similarity < 0.3:  # Very low threshold for text
                    print("[LM Fingerprint] Warning: Low similarity detected")
        
        # Reset sequential tester if using
        if self.use_sequential:
            self.sequential_tester = SequentialTester(
                alpha=self.delta, beta=self.delta,
                tau0=tolerance/2, tau1=tolerance*2
            )
        
        for i, challenge in enumerate(challenges):
            # Evaluate challenge
            model_output, distance = self.evaluate_challenge(model, challenge, method=method)
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
        
        # Add fingerprint information to metadata if available
        if model_fingerprint is not None:
            metadata['fingerprint'] = {
                'io_hash': model_fingerprint.io_hash,
                'has_jacobian': model_fingerprint.jacobian_sketch is not None,
                'similarity': fingerprint_similarity if fingerprint_similarity is not None else 'N/A',
                'num_outputs': len(model_fingerprint.raw_outputs),
                'config': model_fingerprint.metadata.get('fingerprint_config', {})
            }
            if self.reference_fingerprint is not None:
                metadata['fingerprint']['reference_io_hash'] = self.reference_fingerprint.io_hash
        
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
            fingerprint=model_fingerprint,
            fingerprint_match=fingerprint_similarity,
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
    
    def compute_reference_fingerprint(self, prompts: List[str]) -> FingerprintResult:
        """
        Compute and store reference model fingerprint
        
        Args:
            prompts: Text prompts to use for fingerprinting
            
        Returns:
            FingerprintResult for the reference model
        """
        if not self.use_fingerprinting:
            return None
        
        # Create a wrapper function for the reference model
        def model_wrapper(prompt_batch):
            # Handle both single prompts and batches
            if isinstance(prompt_batch, str):
                prompt_batch = [prompt_batch]
            
            outputs = []
            for prompt in prompt_batch:
                output = self.reference_model.generate(prompt, max_new_tokens=64)
                # Canonicalize the text output
                canonical_output = canonicalize_text(output)
                outputs.append(canonical_output)
            
            return outputs if len(outputs) > 1 else outputs[0]
        
        # Compute fingerprint
        self.reference_fingerprint = fingerprint_run(
            model_wrapper,
            prompts,
            self.fingerprint_config
        )
        
        return self.reference_fingerprint
    
    def compute_fingerprint_similarity(self, fingerprint1: FingerprintResult,
                                      fingerprint2: FingerprintResult) -> float:
        """
        Compute similarity between two fingerprints
        
        Args:
            fingerprint1: First fingerprint
            fingerprint2: Second fingerprint
            
        Returns:
            Similarity score between 0 and 1 (1 = identical)
        """
        if fingerprint1 is None or fingerprint2 is None:
            return 0.0
        
        # Compare IO hashes (exact match)
        if fingerprint1.io_hash == fingerprint2.io_hash:
            return 1.0
        
        # If Jacobian sketches exist, compare them (though less useful for text)
        if fingerprint1.jacobian_sketch and fingerprint2.jacobian_sketch:
            sketch1_bytes = bytes.fromhex(fingerprint1.jacobian_sketch)
            sketch2_bytes = bytes.fromhex(fingerprint2.jacobian_sketch)
            similarity = compare_jacobian_sketches(sketch1_bytes, sketch2_bytes, method='hamming')
            return similarity
        
        # Compare raw outputs using fuzzy text matching
        if fingerprint1.raw_outputs and fingerprint2.raw_outputs:
            similarities = []
            for out1, out2 in zip(fingerprint1.raw_outputs, fingerprint2.raw_outputs):
                # Convert to strings if needed
                s1 = str(out1) if not isinstance(out1, str) else out1
                s2 = str(out2) if not isinstance(out2, str) else out2
                
                # Use SequenceMatcher for similarity
                import difflib
                similarity = difflib.SequenceMatcher(None, s1, s2).ratio()
                similarities.append(similarity)
            
            if similarities:
                return float(np.mean(similarities))
        
        # Default: different fingerprints
        return 0.0
    
    def verify_with_time_tolerance(self, model: LM,
                                  challenges: List[Dict[str, Any]],
                                  base_tolerance: float = 0.1,
                                  days_elapsed: int = 0,
                                  drift_rate: float = 0.001,
                                  drift_model: str = "linear",
                                  max_tolerance: Optional[float] = None) -> LMVerificationResult:
        """
        Verify with time-aware tolerance for version drift
        From paper Section 5 on handling model updates
        
        Args:
            model: Model to verify
            challenges: Verification challenges
            base_tolerance: Base tolerance at time 0
            days_elapsed: Days since reference model snapshot
            drift_rate: Expected drift parameter
            drift_model: Drift adjustment model ('linear', 'quadratic', 'exponential')
            max_tolerance: Optional cap on adjusted tolerance
            
        Returns:
            Verification result with adjusted tolerance
        """
        # Adjust tolerance based on time elapsed using specified drift model
        if drift_model == "linear":
            adjusted_tolerance = base_tolerance + drift_rate * days_elapsed
        elif drift_model == "quadratic":
            adjusted_tolerance = base_tolerance + drift_rate * (days_elapsed ** 2)
        elif drift_model == "exponential":
            adjusted_tolerance = base_tolerance * ((1 + drift_rate) ** days_elapsed)
        else:
            raise ValueError(f"Unknown drift_model: {drift_model}")

        cap_applied = False
        if max_tolerance is not None and adjusted_tolerance > max_tolerance:
            adjusted_tolerance = max_tolerance
            cap_applied = True
        
        # Run verification with adjusted tolerance (pass through fingerprinting)
        result = self.verify(model, challenges, adjusted_tolerance, 
                           compute_reference_fingerprint=True)
        
        # Add time tolerance info to metadata
        result.metadata["time_tolerance"] = {
            "base_tolerance": base_tolerance,
            "days_elapsed": days_elapsed,
            "drift_rate": drift_rate,
            "drift_model": drift_model,
            "max_tolerance": max_tolerance,
            "adjusted_tolerance": adjusted_tolerance,
            "justification": (
                f"Tolerance adjusted using {drift_model} model with rate {drift_rate} "
                f"over {days_elapsed} days" +
                ("; capped at maximum tolerance " + str(max_tolerance) if cap_applied else "")
            )
        }
        
        return result


class BatchLMVerifier:
    """
    Batch verification for multiple language models
    Implements efficient batch processing from paper
    """
    
    def __init__(self, reference_model: LM, delta: float = 0.01,
                 use_fingerprinting: bool = True, fingerprint_config: Optional[FingerprintConfig] = None):
        self.verifier = LMVerifier(reference_model, delta, 
                                  use_fingerprinting=use_fingerprinting,
                                  fingerprint_config=fingerprint_config)
    
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