"""
Vision Model Verifier for Proof-of-Training
Implements verification protocol for vision models from paper Section 3
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import time
from PIL import Image
import torchvision.transforms as transforms

from ..core.stats import empirical_bernstein_bound, t_statistic
from ..core.sequential import SequentialTester, SPRTResult
from ..core.challenge import generate_challenges, ChallengeConfig
from ..core.wrapper_detection import WrapperAttackDetector
from .models import VisionModel


@dataclass
class VisionVerificationResult:
    """Result of vision model verification"""
    accepted: bool
    distance: float
    confidence_radius: float
    n_challenges: int
    perceptual_similarity: float
    time_elapsed: float
    wrapper_detection: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


class VisionVerifier:
    """
    Vision Model Verifier
    Implements the verification protocol for vision models from paper Section 3
    """
    
    def __init__(self, reference_model: VisionModel, delta: float = 0.01,
                 use_sequential: bool = True, detect_wrappers: bool = True):
        """
        Initialize vision verifier
        
        Args:
            reference_model: Reference vision model f*
            delta: Confidence parameter (1-delta confidence)
            use_sequential: Whether to use SPRT for early stopping
            detect_wrappers: Whether to detect wrapper attacks
        """
        self.reference_model = reference_model
        self.delta = delta
        self.use_sequential = use_sequential
        self.detect_wrappers = detect_wrappers
        
        # Initialize components
        if use_sequential:
            self.sequential_tester = SequentialTester(
                alpha=delta,
                beta=delta,
                tau0=0.02,  # Expected distance for same model
                tau1=0.1    # Expected distance for different model
            )
        
        if detect_wrappers:
            self.wrapper_detector = WrapperAttackDetector()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def generate_frequency_challenges(self, n: int,
                                     master_key: str,
                                     session_nonce: str) -> List[torch.Tensor]:
        """
        Generate frequency-based challenges (sine gratings)
        From paper Section 3.3: Vision-specific challenges
        """
        config = ChallengeConfig(
            master_key_hex=master_key,
            session_nonce_hex=session_nonce,
            n=n,
            family="vision:freq",
            params={
                "freq_range": (0.5, 10.0),  # Spatial frequency range
                "contrast_range": (0.2, 1.0)  # Contrast range
            }
        )
        
        challenge_data = generate_challenges(config)
        challenges = []
        
        for item in challenge_data["items"]:
            # Generate sine grating image
            image = self._generate_sine_grating(
                size=(224, 224),
                frequency=item["freq"],
                orientation=item["theta"],
                phase=item["phase"],
                contrast=item["contrast"]
            )
            challenges.append(image)
        
        return challenges
    
    def generate_texture_challenges(self, n: int,
                                  master_key: str,
                                  session_nonce: str) -> List[torch.Tensor]:
        """
        Generate texture-based challenges using Perlin noise
        """
        config = ChallengeConfig(
            master_key_hex=master_key,
            session_nonce_hex=session_nonce,
            n=n,
            family="vision:texture",
            params={
                "octaves": (1, 4),
                "scale": (0.01, 0.1)
            }
        )
        
        challenge_data = generate_challenges(config)
        challenges = []
        
        for item in challenge_data["items"]:
            # Generate Perlin noise texture
            image = self._generate_perlin_noise(
                size=(224, 224),
                octaves=item["octaves"],
                scale=item["scale"]
            )
            challenges.append(image)
        
        return challenges
    
    def _generate_sine_grating(self, size: Tuple[int, int],
                              frequency: float,
                              orientation: float,
                              phase: float,
                              contrast: float) -> torch.Tensor:
        """Generate sine grating pattern"""
        h, w = size
        x = np.linspace(-np.pi, np.pi, w)
        y = np.linspace(-np.pi, np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        Xr = X * np.cos(orientation) + Y * np.sin(orientation)
        
        # Generate grating
        grating = np.sin(frequency * Xr + phase)
        
        # Apply contrast
        grating = grating * contrast
        
        # Normalize to [0, 1]
        grating = (grating + 1) / 2
        
        # Convert to RGB tensor
        grating_rgb = np.stack([grating, grating, grating], axis=0)
        return torch.tensor(grating_rgb, dtype=torch.float32)
    
    def _generate_perlin_noise(self, size: Tuple[int, int],
                              octaves: int,
                              scale: float) -> torch.Tensor:
        """Generate Perlin noise texture"""
        h, w = size
        noise = np.zeros((h, w))
        
        for octave in range(octaves):
            freq = 2 ** octave
            amplitude = 1 / freq
            
            # Generate random gradients
            grad_h = int(h * scale * freq) + 1
            grad_w = int(w * scale * freq) + 1
            gradients = np.random.randn(grad_h, grad_w, 2)
            
            # Interpolate to full resolution
            x = np.linspace(0, grad_w - 1, w)
            y = np.linspace(0, grad_h - 1, h)
            
            # Simple bilinear interpolation (placeholder)
            octave_noise = np.random.randn(h, w) * amplitude
            noise += octave_noise
        
        # Normalize to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        
        # Convert to RGB tensor
        noise_rgb = np.stack([noise, noise, noise], axis=0)
        return torch.tensor(noise_rgb, dtype=torch.float32)
    
    def compute_perceptual_distance(self, features1: torch.Tensor,
                                   features2: torch.Tensor,
                                   metric: str = 'cosine') -> float:
        """
        Compute perceptual distance between feature representations
        
        Args:
            features1: First feature tensor
            features2: Second feature tensor
            metric: Distance metric ('cosine', 'l2', 'l1')
            
        Returns:
            Distance in [0, 1]
        """
        with torch.no_grad():
            if metric == 'cosine':
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    features1.flatten(), features2.flatten(), dim=0
                )
                distance = 1 - cos_sim.item()
            elif metric == 'l2':
                # L2 distance
                distance = torch.norm(features1 - features2, p=2).item()
                # Normalize (rough approximation)
                distance = min(1.0, distance / 10.0)
            elif metric == 'l1':
                # L1 distance
                distance = torch.norm(features1 - features2, p=1).item()
                # Normalize
                distance = min(1.0, distance / 100.0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        return distance
    
    def evaluate_challenge(self, model: VisionModel,
                         challenge: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """
        Evaluate a single challenge on the model
        
        Returns:
            (model_features, distance_from_reference, response_time)
        """
        start_time = time.time()
        
        # Get model output
        with torch.no_grad():
            # Add batch dimension if needed
            if challenge.dim() == 3:
                challenge = challenge.unsqueeze(0)
            
            model_features = model.get_features(challenge)
            reference_features = self.reference_model.get_features(challenge)
        
        response_time = time.time() - start_time
        
        # Compute distance
        distance = self.compute_perceptual_distance(
            model_features, reference_features, metric='cosine'
        )
        
        return model_features, distance, response_time
    
    def verify(self, model: VisionModel,
              challenges: List[torch.Tensor],
              tolerance: float = 0.05,
              challenge_types: List[str] = None) -> VisionVerificationResult:
        """
        Verify a vision model against reference
        
        Args:
            model: Model to verify (f)
            challenges: List of challenge images
            tolerance: Maximum acceptable average distance
            challenge_types: Optional list indicating challenge type for each challenge
            
        Returns:
            VisionVerificationResult with verification outcome
        """
        start_time = time.time()
        distances = []
        perceptual_similarities = []
        response_times = []
        challenge_responses = []
        
        # Reset sequential tester if using
        if self.use_sequential:
            self.sequential_tester = SequentialTester(
                alpha=self.delta, beta=self.delta,
                tau0=tolerance/2, tau1=tolerance*2
            )
        
        for i, challenge in enumerate(challenges):
            # Evaluate challenge
            features, distance, resp_time = self.evaluate_challenge(model, challenge)
            
            distances.append(distance)
            response_times.append(resp_time)
            challenge_responses.append(features)
            
            # Compute perceptual similarity
            perceptual_sim = 1 - distance
            perceptual_similarities.append(perceptual_sim)
            
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
        
        # Wrapper detection if enabled
        wrapper_result = None
        if self.detect_wrappers and len(response_times) > 10:
            # Separate challenge and regular responses (if types provided)
            if challenge_types:
                chal_resp = [r for r, t in zip(challenge_responses, challenge_types) 
                           if t == 'challenge']
                reg_resp = [r for r, t in zip(challenge_responses, challenge_types)
                          if t == 'regular']
            else:
                # Assume all are challenges
                chal_resp = challenge_responses
                reg_resp = []
            
            detection = self.wrapper_detector.comprehensive_detection(
                challenge_responses=chal_resp,
                regular_responses=reg_resp,
                timing_data=response_times
            )
            
            wrapper_result = {
                'is_wrapper': detection.is_wrapper,
                'confidence': detection.confidence,
                'anomaly_score': detection.anomaly_score
            }
        
        elapsed = time.time() - start_time
        
        metadata = {
            'test_statistic': float(test_statistic),
            'tolerance': tolerance,
            'n_evaluated': len(distances),
            'distance_stats': {
                'mean': float(np.mean(distances_array)),
                'std': float(np.std(distances_array)),
                'min': float(np.min(distances_array)),
                'max': float(np.max(distances_array))
            },
            'timing_stats': {
                'mean': float(np.mean(response_times)),
                'std': float(np.std(response_times)),
                'min': float(np.min(response_times)),
                'max': float(np.max(response_times))
            }
        }
        
        return VisionVerificationResult(
            accepted=accepted,
            distance=float(test_statistic),
            confidence_radius=float(conf_radius),
            n_challenges=len(distances),
            perceptual_similarity=float(np.mean(perceptual_similarities)),
            time_elapsed=elapsed,
            wrapper_detection=wrapper_result,
            metadata=metadata
        )
    
    def verify_with_augmentations(self, model: VisionModel,
                                 base_challenges: List[torch.Tensor],
                                 augmentation_types: List[str] = ['rotation', 'scale', 'noise'],
                                 tolerance: float = 0.05) -> VisionVerificationResult:
        """
        Verify model with augmented challenges for robustness
        Tests invariance to common transformations
        """
        all_challenges = []
        challenge_types = []
        
        # Add base challenges
        all_challenges.extend(base_challenges)
        challenge_types.extend(['base'] * len(base_challenges))
        
        # Generate augmented versions
        for aug_type in augmentation_types:
            augmented = self._augment_challenges(base_challenges, aug_type)
            all_challenges.extend(augmented)
            challenge_types.extend([aug_type] * len(augmented))
        
        # Run verification with full challenge set
        result = self.verify(model, all_challenges, tolerance, challenge_types)
        
        # Add augmentation-specific analysis to metadata
        result.metadata['augmentation_analysis'] = self._analyze_augmentation_results(
            distances=result.metadata['distance_stats'],
            challenge_types=challenge_types
        )
        
        return result
    
    def _augment_challenges(self, challenges: List[torch.Tensor],
                          aug_type: str) -> List[torch.Tensor]:
        """Apply augmentation to challenges"""
        augmented = []
        
        for challenge in challenges:
            if aug_type == 'rotation':
                # Random rotation between -30 and 30 degrees
                angle = np.random.uniform(-30, 30)
                aug = transforms.functional.rotate(challenge, angle)
            elif aug_type == 'scale':
                # Random scale between 0.8 and 1.2
                scale = np.random.uniform(0.8, 1.2)
                h, w = challenge.shape[-2:]
                new_h, new_w = int(h * scale), int(w * scale)
                aug = transforms.functional.resize(challenge, (new_h, new_w))
                # Crop or pad back to original size
                aug = transforms.functional.center_crop(aug, (h, w))
            elif aug_type == 'noise':
                # Add Gaussian noise
                noise = torch.randn_like(challenge) * 0.05
                aug = challenge + noise
                aug = torch.clamp(aug, 0, 1)
            else:
                aug = challenge
            
            augmented.append(aug)
        
        return augmented
    
    def _analyze_augmentation_results(self, distances: Dict[str, float],
                                     challenge_types: List[str]) -> Dict[str, Any]:
        """Analyze results per augmentation type"""
        analysis = {}
        
        for aug_type in set(challenge_types):
            type_indices = [i for i, t in enumerate(challenge_types) if t == aug_type]
            if type_indices:
                # Get distances for this type
                # Note: This is simplified; in practice would track per-challenge distances
                analysis[aug_type] = {
                    'count': len(type_indices),
                    'proportion': len(type_indices) / len(challenge_types)
                }
        
        return analysis


class BatchVisionVerifier:
    """
    Batch verification for multiple vision models
    """
    
    def __init__(self, reference_model: VisionModel, delta: float = 0.01):
        self.verifier = VisionVerifier(reference_model, delta)
    
    def verify_batch(self, models: List[VisionModel],
                    challenges: List[torch.Tensor],
                    tolerance: float = 0.05,
                    parallel: bool = False) -> List[VisionVerificationResult]:
        """
        Verify multiple models in batch
        
        Args:
            models: List of models to verify
            challenges: Common challenge set  
            tolerance: Acceptance threshold
            parallel: Whether to process in parallel (requires multiprocessing)
            
        Returns:
            List of verification results
        """
        results = []
        
        if parallel and len(models) > 1:
            # Parallel processing (simplified - would use multiprocessing in practice)
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self.verifier.verify, model, challenges, tolerance)
                    for model in models
                ]
                results = [future.result() for future in futures]
        else:
            # Sequential processing
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
    
    def compare_models(self, models: List[VisionModel],
                       n_challenges: int = 50,
                       master_key: str = None) -> np.ndarray:
        """
        Compare multiple models pairwise
        
        Returns:
            Distance matrix between models
        """
        n_models = len(models)
        distance_matrix = np.zeros((n_models, n_models))
        
        # Generate common challenges
        if master_key is None:
            master_key = "0" * 64
        session_nonce = "1" * 32
        
        challenges = self.verifier.generate_frequency_challenges(
            n_challenges, master_key, session_nonce
        )
        
        # Compute pairwise distances
        for i in range(n_models):
            for j in range(i + 1, n_models):
                distances = []
                
                for challenge in challenges:
                    with torch.no_grad():
                        features_i = models[i].get_features(challenge.unsqueeze(0))
                        features_j = models[j].get_features(challenge.unsqueeze(0))
                    
                    dist = self.verifier.compute_perceptual_distance(
                        features_i, features_j
                    )
                    distances.append(dist)
                
                avg_distance = np.mean(distances)
                distance_matrix[i, j] = avg_distance
                distance_matrix[j, i] = avg_distance
        
        return distance_matrix