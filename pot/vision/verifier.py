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
import xxhash

from ..core.stats import empirical_bernstein_bound, t_statistic
from ..core.sequential import (
    SequentialTester, SPRTResult, sequential_verify,
    SequentialState, welford_update, compute_empirical_variance
)
from ..core.challenge import generate_challenges, ChallengeConfig
from ..core.wrapper_detection import WrapperAttackDetector
from ..core.fingerprint import fingerprint_run, FingerprintConfig, FingerprintResult, compare_jacobian_sketches
from .models import VisionModel

# Semantic verification imports (optional)
try:
    from ..semantic.library import ConceptLibrary
    from ..semantic.match import SemanticMatcher
    from ..semantic.utils import extract_embeddings_from_logits, normalize_embeddings
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False


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
    fingerprint: Optional[FingerprintResult]  # Behavioral fingerprint
    fingerprint_match: Optional[float]  # Similarity score if reference fingerprint exists
    sequential_result: Optional[SPRTResult]  # Sequential verification result with trajectory
    semantic_score: Optional[float] = None  # Semantic verification score
    combined_score: Optional[float] = None  # Combined distance and semantic score
    metadata: Dict[str, Any] = None


class VisionVerifier:
    """
    Vision Model Verifier
    Implements the verification protocol for vision models from paper Section 3
    """
    
    def __init__(self, reference_model: VisionModel, delta: float = 0.01,
                 use_sequential: bool = True, sequential_mode: str = 'legacy',
                 detect_wrappers: bool = True,
                 use_fingerprinting: bool = True, fingerprint_config: Optional[FingerprintConfig] = None,
                 semantic_library: Optional['ConceptLibrary'] = None,
                 semantic_weight: float = 0.3):
        """
        Initialize vision verifier
        
        Args:
            reference_model: Reference vision model f*
            delta: Confidence parameter (1-delta confidence)
            use_sequential: Whether to use sequential testing for early stopping
            sequential_mode: 'legacy' for old SPRT, 'enhanced' for new EB-based sequential verification
            detect_wrappers: Whether to detect wrapper attacks
            use_fingerprinting: Whether to compute behavioral fingerprints
            fingerprint_config: Configuration for fingerprinting (uses default vision config if None)
            semantic_library: Optional ConceptLibrary for semantic verification
            semantic_weight: Weight for semantic score in combined verification (1-weight for distance)
        """
        self.reference_model = reference_model
        self.delta = delta
        self.use_sequential = use_sequential
        self.sequential_mode = sequential_mode
        self.detect_wrappers = detect_wrappers
        self.use_fingerprinting = use_fingerprinting
        
        # Set up fingerprinting configuration
        if use_fingerprinting:
            if fingerprint_config is None:
                # Use default vision model configuration
                self.fingerprint_config = FingerprintConfig.for_vision_model(
                    compute_jacobian=True,
                    include_timing=True
                )
            else:
                self.fingerprint_config = fingerprint_config
            
            # Compute reference fingerprint if fingerprinting is enabled
            self.reference_fingerprint = None
        
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
        
        # Initialize semantic verification if library provided
        self.semantic_library = semantic_library
        self.semantic_matcher = None
        self.semantic_weight = semantic_weight
        
        if semantic_library is not None and SEMANTIC_AVAILABLE:
            try:
                self.semantic_matcher = SemanticMatcher(
                    library=semantic_library,
                    threshold=0.7  # Default threshold for semantic matching
                )
            except Exception as e:
                print(f"Warning: Could not initialize semantic matcher: {e}")
                self.semantic_matcher = None
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def generate_frequency_challenges(self, n: int,
                                     master_key: str,
                                     session_nonce: str,
                                     model_id: Optional[str] = None) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        """
        Generate frequency-based challenges (sine gratings)
        From paper Section 3.3: Vision-specific challenges
        
        Args:
            n: Number of challenges
            master_key: Master key for challenge generation
            session_nonce: Session-specific nonce
            model_id: Optional model identifier for model-specific challenges
            
        Returns:
            Tuple of (challenge images, challenge metadata including IDs)
        """
        config = ChallengeConfig(
            master_key_hex=master_key,
            session_nonce_hex=session_nonce,
            n=n,
            family="vision:freq",
            params={
                "freq_range": (0.5, 10.0),  # Spatial frequency range
                "contrast_range": (0.2, 1.0)  # Contrast range
            },
            model_id=model_id
        )
        
        challenge_data = generate_challenges(config)
        images = []
        metadata = []
        
        # Use Challenge objects if available, otherwise fall back to items
        challenges = challenge_data.get("challenges", None)
        if challenges:
            # New format with Challenge objects
            for challenge in challenges:
                params = challenge.parameters
                # Generate sine grating image
                # Note: theta_rad is in radians, theta is in degrees
                image = self._generate_sine_grating(
                    size=(224, 224),
                    frequency=params["freq"],
                    orientation=params.get("theta_rad", params["theta"]),  # Use radians if available
                    phase=params["phase"],
                    contrast=params["contrast"]
                )
                images.append(image)
                
                # Store metadata including challenge ID
                metadata.append({
                    "challenge_id": challenge.challenge_id,
                    "index": challenge.index,
                    "parameters": params
                })
        else:
            # Backward compatibility: use items
            for idx, item in enumerate(challenge_data["items"]):
                image = self._generate_sine_grating(
                    size=(224, 224),
                    frequency=item["freq"],
                    orientation=item["theta"],
                    phase=item["phase"],
                    contrast=item["contrast"]
                )
                images.append(image)
                metadata.append({
                    "challenge_id": f"legacy_{idx}",
                    "index": idx,
                    "parameters": item
                })
        
        return images, metadata
    
    def generate_texture_challenges(self, n: int,
                                  master_key: str,
                                  session_nonce: str,
                                  model_id: Optional[str] = None) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        """
        Generate texture-based challenges using Perlin noise
        
        Args:
            n: Number of challenges
            master_key: Master key for challenge generation
            session_nonce: Session-specific nonce
            model_id: Optional model identifier for model-specific challenges
            
        Returns:
            Tuple of (challenge images, challenge metadata including IDs)
        """
        config = ChallengeConfig(
            master_key_hex=master_key,
            session_nonce_hex=session_nonce,
            n=n,
            family="vision:texture",
            params={
                "octaves": (1, 4),
                "scale": (0.01, 0.1)
            },
            model_id=model_id
        )
        
        challenge_data = generate_challenges(config)
        images = []
        metadata = []
        
        # Use Challenge objects if available
        challenges = challenge_data.get("challenges", None)
        if challenges:
            for challenge in challenges:
                params = challenge.parameters
                # Use challenge ID for seed generation (more deterministic)
                seed = xxhash.xxh64(challenge.challenge_id.encode()).intdigest()
                
                # Generate Perlin noise texture
                image = self._generate_perlin_noise(
                    size=(224, 224),
                    octaves=params["octaves"],
                    scale=params["scale"],
                    seed=seed,
                )
                images.append(image)
                
                metadata.append({
                    "challenge_id": challenge.challenge_id,
                    "index": challenge.index,
                    "parameters": params
                })
        else:
            # Backward compatibility
            for idx, item in enumerate(challenge_data["items"]):
                seed_input = f"{master_key}:{session_nonce}:{idx}:{item['octaves']}:{item['scale']}"
                seed = xxhash.xxh64(seed_input).intdigest()
                
                image = self._generate_perlin_noise(
                    size=(224, 224),
                    octaves=item["octaves"],
                    scale=item["scale"],
                    seed=seed,
                )
                images.append(image)
                metadata.append({
                    "challenge_id": f"legacy_{idx}",
                    "index": idx,
                    "parameters": item
                })
        
        return images, metadata
    
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
                              scale: float,
                              seed: Optional[int] = None) -> torch.Tensor:
        """Generate Perlin noise texture using Perlin's gradient interpolation."""
        h, w = size
        rng = np.random.default_rng(seed)
        noise = np.zeros((h, w))

        def fade(t: np.ndarray) -> np.ndarray:
            return t * t * t * (t * (t * 6 - 15) + 10)

        for octave in range(octaves):
            freq = 2 ** octave
            amplitude = 1 / freq

            grad_h = int(h * scale * freq) + 1
            grad_w = int(w * scale * freq) + 1

            angles = rng.uniform(0, 2 * np.pi, size=(grad_h, grad_w))
            gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

            x = np.linspace(0, grad_w - 1, w, endpoint=False)
            y = np.linspace(0, grad_h - 1, h, endpoint=False)
            xi = x.astype(int)
            yi = y.astype(int)
            xf = x - xi
            yf = y - yi

            xi = xi[np.newaxis, :]
            yi = yi[:, np.newaxis]
            xf = xf[np.newaxis, :]
            yf = yf[:, np.newaxis]

            g00 = gradients[yi, xi]
            g10 = gradients[yi, xi + 1]
            g01 = gradients[yi + 1, xi]
            g11 = gradients[yi + 1, xi + 1]

            x_b = np.broadcast_to(xf, (h, w))
            y_b = np.broadcast_to(yf, (h, w))

            d00 = np.stack((x_b, y_b), axis=-1)
            d10 = np.stack((x_b - 1, y_b), axis=-1)
            d01 = np.stack((x_b, y_b - 1), axis=-1)
            d11 = np.stack((x_b - 1, y_b - 1), axis=-1)

            dot00 = np.sum(g00 * d00, axis=-1)
            dot10 = np.sum(g10 * d10, axis=-1)
            dot01 = np.sum(g01 * d01, axis=-1)
            dot11 = np.sum(g11 * d11, axis=-1)

            u = fade(xf)
            v = fade(yf)

            nx0 = dot00 * (1 - u) + dot10 * u
            nx1 = dot01 * (1 - u) + dot11 * u
            value = (nx0 * (1 - v) + nx1 * v) * amplitude
            noise += value

        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

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
    
    def _compute_semantic_score(self, features: torch.Tensor) -> float:
        """
        Compute semantic verification score for vision features.
        
        Args:
            features: Model feature tensor
            
        Returns:
            Semantic similarity score in [0, 1]
        """
        if self.semantic_matcher is None:
            return 0.5  # Neutral score if no semantic verification
        
        try:
            # Normalize features
            normalized = normalize_embeddings(features, method='l2')
            
            # Match to library concepts
            matches = self.semantic_matcher.match_to_library(normalized)
            
            if not matches:
                return 0.0
            
            # Return best match score
            best_score = max(matches.values())
            return float(best_score)
            
        except Exception as e:
            print(f"Warning: Semantic scoring failed: {e}")
            return 0.5  # Neutral score on error
    
    def compute_reference_fingerprint(self, challenges: List[torch.Tensor]) -> FingerprintResult:
        """
        Compute and store reference model fingerprint
        
        Args:
            challenges: Challenges to use for fingerprinting
            
        Returns:
            FingerprintResult for the reference model
        """
        if not self.use_fingerprinting:
            return None
            
        # Create a wrapper function for the reference model
        def model_wrapper(x):
            with torch.no_grad():
                return self.reference_model.get_features(x)
        
        # Compute fingerprint
        self.reference_fingerprint = fingerprint_run(
            model_wrapper,
            challenges,
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
        
        # If Jacobian sketches exist, compare them
        if fingerprint1.jacobian_sketch and fingerprint2.jacobian_sketch:
            # Convert hex strings back to bytes for comparison
            sketch1_bytes = bytes.fromhex(fingerprint1.jacobian_sketch)
            sketch2_bytes = bytes.fromhex(fingerprint2.jacobian_sketch)
            
            # Use Hamming similarity by default
            similarity = compare_jacobian_sketches(sketch1_bytes, sketch2_bytes, method='hamming')
            return similarity
        
        # Fall back to comparing raw outputs if available
        if fingerprint1.raw_outputs and fingerprint2.raw_outputs:
            # Simple cosine similarity on flattened outputs
            try:
                import numpy as np
                flat1 = np.concatenate([np.array(o).flatten() for o in fingerprint1.raw_outputs])
                flat2 = np.concatenate([np.array(o).flatten() for o in fingerprint2.raw_outputs])
                
                # Normalize and compute cosine similarity
                norm1 = np.linalg.norm(flat1)
                norm2 = np.linalg.norm(flat2)
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(flat1, flat2) / (norm1 * norm2)
                    return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            except:
                pass
        
        # Default: different fingerprints
        return 0.0
    
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
              challenge_types: List[str] = None,
              compute_reference_fingerprint: bool = False,
              alpha: float = None,
              beta: float = None) -> VisionVerificationResult:
        """
        Verify a vision model against reference
        
        Args:
            model: Model to verify (f)
            challenges: List of challenge images
            tolerance: Maximum acceptable average distance (tau threshold)
            challenge_types: Optional list indicating challenge type for each challenge
            compute_reference_fingerprint: Whether to compute reference fingerprint if not already done
            alpha: Type I error rate for sequential testing (defaults to self.delta)
            beta: Type II error rate for sequential testing (defaults to self.delta)
            
        Returns:
            VisionVerificationResult with verification outcome
        """
        start_time = time.time()
        distances = []
        perceptual_similarities = []
        response_times = []
        challenge_responses = []
        
        # Set error rates if not provided
        if alpha is None:
            alpha = self.delta
        if beta is None:
            beta = self.delta
        
        # Compute behavioral fingerprint if enabled
        model_fingerprint = None
        fingerprint_similarity = None
        
        if self.use_fingerprinting:
            # Compute reference fingerprint if needed
            if compute_reference_fingerprint or self.reference_fingerprint is None:
                self.compute_reference_fingerprint(challenges)
            
            # Create a wrapper function for the model being verified
            def model_wrapper(x):
                with torch.no_grad():
                    return model.get_features(x)
            
            # Compute fingerprint for the model being verified
            fingerprint_start = time.time()
            model_fingerprint = fingerprint_run(
                model_wrapper,
                challenges,
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
                print(f"[Fingerprint] IO Hash Match: {model_fingerprint.io_hash == self.reference_fingerprint.io_hash}")
                print(f"[Fingerprint] Similarity: {fingerprint_similarity:.4f}")
                print(f"[Fingerprint] Time: {fingerprint_time:.3f}s")
                
                # Early rejection based on fingerprint (optional)
                # if fingerprint_similarity < 0.5:  # Threshold can be configurable
                #     print("[Fingerprint] Early rejection: similarity too low")
        
        # Initialize sequential testing based on mode
        sequential_result = None
        
        if self.use_sequential and self.sequential_mode == 'enhanced':
            # Use new EB-based sequential verification
            def distance_stream():
                for challenge in challenges:
                    features, distance, resp_time = self.evaluate_challenge(model, challenge)
                    distances.append(distance)
                    response_times.append(resp_time)
                    challenge_responses.append(features)
                    perceptual_similarities.append(1 - distance)
                    yield distance
            
            # Run sequential verification
            sequential_result = sequential_verify(
                stream=distance_stream(),
                tau=tolerance,
                alpha=alpha,
                beta=beta,
                max_samples=len(challenges),
                compute_p_value=True
            )
            
            # Early stopping achieved
            if sequential_result.stopped_at < len(challenges):
                print(f"[Sequential] Early stopping at n={sequential_result.stopped_at}")
                
        elif self.use_sequential and self.sequential_mode == 'legacy':
            # Use legacy SPRT
            self.sequential_tester = SequentialTester(
                alpha=alpha, beta=beta,
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
                result = self.sequential_tester.update(distance)
                if result.decision != 'continue':
                    # Early stopping
                    break
        else:
            # No sequential testing - evaluate all challenges
            for challenge in challenges:
                features, distance, resp_time = self.evaluate_challenge(model, challenge)
                distances.append(distance)
                response_times.append(resp_time)
                challenge_responses.append(features)
                perceptual_similarities.append(1 - distance)
        
        # Compute test statistic and confidence radius
        distances_array = np.array(distances)
        test_statistic = t_statistic(distances_array)
        conf_radius = empirical_bernstein_bound(distances_array, self.delta)
        
        # Compute semantic score if enabled
        semantic_score = None
        combined_score = test_statistic
        
        if self.semantic_matcher is not None and SEMANTIC_AVAILABLE:
            try:
                # Extract semantic scores from model features
                semantic_scores = []
                for features in challenge_responses[:min(10, len(challenge_responses))]:  # Sample for efficiency
                    if features is not None:
                        sem_score = self._compute_semantic_score(features)
                        semantic_scores.append(sem_score)
                
                if semantic_scores:
                    semantic_score = float(np.mean(semantic_scores))
                    # Combine distance and semantic scores
                    # Lower distance is better, higher semantic score is better
                    # So we invert semantic score for combination
                    semantic_distance = 1.0 - semantic_score
                    combined_score = (1 - self.semantic_weight) * test_statistic + self.semantic_weight * semantic_distance
            except Exception as e:
                print(f"Warning: Semantic scoring failed: {e}")
                semantic_score = None
                combined_score = test_statistic
        
        # Decision logic based on sequential mode
        if sequential_result is not None:
            # Use enhanced sequential result
            accepted = sequential_result.decision == 'H0'
            test_statistic = sequential_result.final_mean
            conf_radius = sequential_result.confidence_radius
        elif self.use_sequential and self.sequential_mode == 'legacy' and hasattr(self, 'sequential_tester'):
            # Use legacy SPRT decision
            if self.sequential_tester.decided():
                accepted = self.sequential_tester.accept()
        else:
            # Fixed-sample decision: accept if combined score + radius <= tolerance
            # Use combined score if semantic verification is enabled
            decision_score = combined_score if semantic_score is not None else test_statistic
            accepted = (decision_score + conf_radius) <= tolerance
        
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
        
        # Add sequential result info to metadata if available
        if sequential_result is not None:
            metadata['sequential'] = {
                'mode': 'enhanced',
                'stopped_at': sequential_result.stopped_at,
                'decision': sequential_result.decision,
                'p_value': sequential_result.p_value,
                'trajectory_length': len(sequential_result.trajectory) if sequential_result.trajectory else 0,
                'forced_stop': sequential_result.forced_stop
            }
        elif self.use_sequential and self.sequential_mode == 'legacy':
            metadata['sequential'] = {
                'mode': 'legacy',
                'stopped_at': len(distances)
            }
        
        return VisionVerificationResult(
            accepted=accepted,
            distance=float(test_statistic),
            confidence_radius=float(conf_radius),
            n_challenges=len(distances),
            perceptual_similarity=float(np.mean(perceptual_similarities)) if perceptual_similarities else 0.0,
            time_elapsed=elapsed,
            wrapper_detection=wrapper_result,
            fingerprint=model_fingerprint,
            fingerprint_match=fingerprint_similarity,
            sequential_result=sequential_result,
            semantic_score=semantic_score,
            combined_score=float(combined_score) if semantic_score is not None else None,
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
        result = self.verify(model, all_challenges, tolerance, challenge_types, 
                            compute_reference_fingerprint=True)
        
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
    
    def __init__(self, reference_model: VisionModel, delta: float = 0.01,
                 use_fingerprinting: bool = True, fingerprint_config: Optional[FingerprintConfig] = None,
                 semantic_library: Optional['ConceptLibrary'] = None,
                 semantic_weight: float = 0.3):
        self.verifier = VisionVerifier(reference_model, delta, 
                                      use_fingerprinting=use_fingerprinting,
                                      fingerprint_config=fingerprint_config,
                                      semantic_library=semantic_library,
                                      semantic_weight=semantic_weight)
    
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