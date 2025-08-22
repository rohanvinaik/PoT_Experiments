"""
Core vision verification implementation.
This module contains the main VisionVerifier and EnhancedVisionVerifier classes.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings

from .verifier_base import (
    BaseVerifier, 
    VerificationConfig, 
    VerificationResult,
    VerificationMethod,
    ChallengeType,
    VerifierRegistry
)
from .verifier_challenges import ChallengeGenerator, ChallengeLibrary, PatternType
from .verifier_hooks import HookManager, ActivationRecorder, GradientMonitor


class VisionVerifier(BaseVerifier):
    """
    Main vision model verifier implementation.
    This consolidates the core verification logic from the original large file.
    """
    
    def __init__(self, 
                 reference_model: Optional[nn.Module] = None,
                 config: Optional[VerificationConfig] = None):
        """
        Initialize vision verifier.
        
        Args:
            reference_model: Optional reference model for comparison
            config: Verification configuration
        """
        super().__init__(config)
        self.reference_model = reference_model
        self.challenge_generator = ChallengeGenerator(device=str(self.device))
        self.hook_manager = HookManager() if config.enable_hooks else None
        
        # Initialize metrics
        self.metrics = {
            'challenges_passed': 0,
            'challenges_total': 0,
            'average_confidence': 0.0,
            'execution_times': []
        }
    
    def verify(self, 
              model: nn.Module,
              num_challenges: Optional[int] = None,
              **kwargs) -> VerificationResult:
        """
        Perform verification on a vision model.
        
        Args:
            model: Model to verify
            num_challenges: Number of challenges to use
            **kwargs: Additional verification parameters
            
        Returns:
            Verification result
        """
        start_time = time.time()
        
        # Setup
        model.eval()
        num_challenges = num_challenges or self.config.num_challenges
        
        # Generate challenges
        challenges = self._generate_challenge_set(num_challenges)
        
        # Attach hooks if enabled
        if self.config.enable_hooks and self.hook_manager:
            self.hook_manager.register_hooks_recursive(model)
        
        # Run verification
        if self.config.method == VerificationMethod.STATISTICAL:
            result = self._statistical_verification(model, challenges)
        elif self.config.method == VerificationMethod.BEHAVIORAL:
            result = self._behavioral_verification(model, challenges)
        elif self.config.method == VerificationMethod.EXACT:
            result = self._exact_verification(model, challenges)
        else:
            result = self._batch_verification(model, challenges)
        
        # Cleanup hooks
        if self.hook_manager:
            self.hook_manager.remove_all_hooks()
        
        # Add timing
        result.execution_time = time.time() - start_time
        
        return result
    
    def _generate_challenge_set(self, num_challenges: int) -> List[torch.Tensor]:
        """Generate a set of challenges"""
        challenges = []
        
        # Mix of different challenge types
        for i in range(num_challenges):
            if i % 5 == 0:
                # Standard pattern
                pattern = self.challenge_generator.generate_sine_grating((224, 224))
            elif i % 5 == 1:
                # Checkerboard
                pattern = self.challenge_generator.generate_checkerboard((224, 224))
            elif i % 5 == 2:
                # Noise
                pattern = self.challenge_generator.generate_gaussian_noise((224, 224))
            elif i % 5 == 3:
                # Gabor
                pattern = self.challenge_generator.generate_gabor_filter((224, 224))
            else:
                # Radial gradient
                pattern = self.challenge_generator.generate_radial_gradient((224, 224))
            
            challenges.append(pattern)
        
        return challenges
    
    def _batch_verification(self, 
                          model: nn.Module,
                          challenges: List[torch.Tensor]) -> VerificationResult:
        """
        Perform batch verification.
        This consolidates the duplicate _batch_verification implementations
        from lines 1283, 1400, and 1994.
        
        Args:
            model: Model to verify
            challenges: List of challenge inputs
            
        Returns:
            Verification result
        """
        passed = 0
        total = len(challenges)
        confidences = []
        
        # Process in batches
        batch_size = self.config.batch_size
        
        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch = challenges[i:i+batch_size]
                
                # Stack batch
                if len(batch) > 1:
                    batch_tensor = torch.cat(batch, dim=0)
                else:
                    batch_tensor = batch[0]
                
                batch_tensor = batch_tensor.to(self.device)
                
                # Get model output
                try:
                    output = model(batch_tensor)
                    
                    # Evaluate each output
                    for j, out in enumerate(output):
                        # Simple confidence based on output statistics
                        confidence = self._compute_confidence(out)
                        confidences.append(confidence)
                        
                        if confidence > self.config.confidence_threshold:
                            passed += 1
                            
                except Exception as e:
                    warnings.warn(f"Batch verification error: {e}")
                    continue
        
        # Compute overall result
        avg_confidence = np.mean(confidences) if confidences else 0.0
        verified = passed / total >= self.config.confidence_threshold
        
        return VerificationResult(
            verified=verified,
            confidence=avg_confidence,
            method="batch",
            challenges_passed=passed,
            challenges_total=total,
            details={
                'confidences': confidences,
                'pass_rate': passed / total if total > 0 else 0
            }
        )
    
    def _statistical_verification(self,
                                 model: nn.Module,
                                 challenges: List[torch.Tensor]) -> VerificationResult:
        """Perform statistical verification"""
        # Use batch verification as base
        result = self._batch_verification(model, challenges)
        
        # Add statistical analysis
        if result.details.get('confidences'):
            confidences = result.details['confidences']
            result.details['statistics'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'median': np.median(confidences)
            }
        
        result.method = "statistical"
        return result
    
    def _behavioral_verification(self,
                                model: nn.Module,
                                challenges: List[torch.Tensor]) -> VerificationResult:
        """Perform behavioral verification"""
        if not self.reference_model:
            # Fall back to batch verification
            return self._batch_verification(model, challenges)
        
        # Compare behaviors
        passed = 0
        total = len(challenges)
        similarities = []
        
        with torch.no_grad():
            for challenge in challenges:
                challenge = challenge.to(self.device)
                
                # Get outputs
                output_test = model(challenge)
                output_ref = self.reference_model(challenge)
                
                # Compute similarity
                similarity = self.evaluate_response(output_test, output_ref)
                similarities.append(similarity)
                
                if similarity > self.config.confidence_threshold:
                    passed += 1
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        return VerificationResult(
            verified=passed / total >= self.config.confidence_threshold,
            confidence=avg_similarity,
            method="behavioral",
            challenges_passed=passed,
            challenges_total=total,
            details={'similarities': similarities}
        )
    
    def _exact_verification(self,
                          model: nn.Module,
                          challenges: List[torch.Tensor]) -> VerificationResult:
        """Perform exact verification"""
        # For exact verification, we need perfect matches
        original_threshold = self.config.confidence_threshold
        self.config.confidence_threshold = 0.99
        
        result = self._batch_verification(model, challenges)
        result.method = "exact"
        
        # Restore threshold
        self.config.confidence_threshold = original_threshold
        
        return result
    
    def _compute_confidence(self, output: torch.Tensor) -> float:
        """Compute confidence score for an output"""
        # Simple heuristic based on output statistics
        if output.numel() == 0:
            return 0.0
        
        # Check for valid range
        if torch.isnan(output).any() or torch.isinf(output).any():
            return 0.0
        
        # Compute statistics
        mean_val = float(output.mean())
        std_val = float(output.std())
        
        # Higher variance often indicates more structured output
        confidence = min(1.0, std_val * 2)
        
        # Penalize extreme values
        if abs(mean_val) > 10:
            confidence *= 0.5
        
        return confidence


class EnhancedVisionVerifier(VisionVerifier):
    """
    Enhanced vision verifier with advanced features.
    This extends the base VisionVerifier with additional capabilities.
    """
    
    def __init__(self,
                 reference_model: Optional[nn.Module] = None,
                 config: Optional[VerificationConfig] = None):
        """
        Initialize enhanced verifier.
        
        Args:
            reference_model: Optional reference model
            config: Verification configuration
        """
        super().__init__(reference_model, config)
        self.activation_recorder = ActivationRecorder()
        self.gradient_monitor = GradientMonitor()
        
    def verify_with_analysis(self,
                            model: nn.Module,
                            **kwargs) -> Tuple[VerificationResult, Dict[str, Any]]:
        """
        Perform verification with detailed analysis.
        
        Args:
            model: Model to verify
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (verification result, analysis data)
        """
        # Attach recorders
        self.activation_recorder.attach_to_model(model)
        
        # Run standard verification
        result = self.verify(model, **kwargs)
        
        # Collect analysis data
        analysis = {
            'activations': self.activation_recorder.get_recordings(),
            'metrics': self.metrics
        }
        
        # Add gradient analysis if model is in training mode
        if model.training:
            self.gradient_monitor.attach_to_model(model)
            # Run a backward pass
            dummy_loss = self._compute_dummy_loss(model)
            dummy_loss.backward()
            
            analysis['gradient_stats'] = self.gradient_monitor.get_gradient_stats()
            analysis['gradient_alerts'] = self.gradient_monitor.check_gradients()
            
            self.gradient_monitor.detach()
        
        # Cleanup
        self.activation_recorder.detach()
        
        return result, analysis
    
    def _compute_dummy_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute a dummy loss for gradient analysis"""
        # Generate a simple input
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        output = model(dummy_input)
        
        # Simple loss
        if isinstance(output, torch.Tensor):
            return output.mean()
        else:
            return torch.tensor(0.0, device=self.device)
    
    def adaptive_verification(self,
                            model: nn.Module,
                            min_challenges: int = 5,
                            max_challenges: int = 50,
                            early_stop_threshold: float = 0.95) -> VerificationResult:
        """
        Adaptive verification that adjusts number of challenges based on confidence.
        
        Args:
            model: Model to verify
            min_challenges: Minimum number of challenges
            max_challenges: Maximum number of challenges
            early_stop_threshold: Confidence threshold for early stopping
            
        Returns:
            Verification result
        """
        challenges_used = 0
        confidences = []
        
        for i in range(max_challenges):
            # Generate single challenge
            challenge = self._generate_challenge_set(1)[0]
            
            # Evaluate
            with torch.no_grad():
                output = model(challenge.to(self.device))
                confidence = self._compute_confidence(output)
                confidences.append(confidence)
            
            challenges_used += 1
            
            # Check early stopping
            if challenges_used >= min_challenges:
                avg_confidence = np.mean(confidences)
                
                if avg_confidence >= early_stop_threshold:
                    # High confidence, can stop early
                    break
                elif avg_confidence < 0.5 and challenges_used >= min_challenges * 2:
                    # Low confidence, no point continuing
                    break
        
        # Compute final result
        avg_confidence = np.mean(confidences) if confidences else 0.0
        passed = sum(c > self.config.confidence_threshold for c in confidences)
        
        return VerificationResult(
            verified=avg_confidence >= self.config.confidence_threshold,
            confidence=avg_confidence,
            method="adaptive",
            challenges_passed=passed,
            challenges_total=challenges_used,
            details={
                'confidences': confidences,
                'early_stopped': challenges_used < max_challenges
            }
        )


# Register verifiers
VerifierRegistry.register("vision", VisionVerifier)
VerifierRegistry.register("vision_enhanced", EnhancedVisionVerifier)