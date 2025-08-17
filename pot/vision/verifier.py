"""
Vision Model Verifier for Proof-of-Training
Implements verification protocol for vision models from paper Section 3
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
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

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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
    
    def _generate_mid_frequency(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate mid-frequency pattern."""
        h, w = size
        
        # Create edge patterns using Gabor filters
        x = torch.linspace(-1, 1, w)
        y = torch.linspace(-1, 1, h)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        # Multiple oriented Gabor patterns
        pattern = torch.zeros_like(xx)
        frequencies = [2, 4, 6]  # Mid-range frequencies
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for freq in frequencies:
            for orient in orientations:
                # Gabor filter
                rotated_x = xx * np.cos(orient) + yy * np.sin(orient)
                envelope = torch.exp(-(xx**2 + yy**2) / 0.5)
                gabor = envelope * torch.sin(2 * np.pi * freq * rotated_x)
                pattern += gabor
        
        # Normalize to [0, 1]
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        
        # Convert to RGB
        image = pattern.unsqueeze(0).repeat(3, 1, 1)
        
        return image
    
    def _generate_mixed_frequency(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate mixed frequency pattern."""
        h, w = size
        
        # Combine low, mid, and high frequency components
        low_freq = self._generate_low_frequency(size)[0]  # Take first channel
        mid_freq = self._generate_mid_frequency(size)[0]
        high_freq = self._generate_high_frequency(size)[0]
        
        # Weight combination
        pattern = 0.4 * low_freq + 0.4 * mid_freq + 0.2 * high_freq
        
        # Normalize to [0, 1]
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        
        # Convert to RGB
        image = pattern.unsqueeze(0).repeat(3, 1, 1)
        
        return image
    
    def generate_texture_challenges(self, 
                                   num_challenges: int = 10,
                                   image_size: Tuple[int, int] = (224, 224),
                                   texture_types: List[str] = None) -> List[torch.Tensor]:
        """Generate texture-based challenge images."""
        texture_types = texture_types or ['perlin', 'voronoi', 'cellular', 'fractal']
        challenges = []
        
        for i in range(num_challenges):
            texture_type = texture_types[i % len(texture_types)]
            
            if texture_type == 'perlin':
                challenge = self._generate_perlin_texture(image_size)
            elif texture_type == 'voronoi':
                challenge = self._generate_voronoi_texture(image_size)
            elif texture_type == 'cellular':
                challenge = self._generate_cellular_texture(image_size)
            elif texture_type == 'fractal':
                challenge = self._generate_fractal_texture(image_size)
            else:
                challenge = self._generate_perlin_texture(image_size)
                
            challenges.append(challenge)
            
        return challenges
    
    def generate_natural_challenges(self,
                                   num_challenges: int = 10,
                                   image_size: Tuple[int, int] = (224, 224)) -> List[torch.Tensor]:
        """Generate natural image-like challenges."""
        challenges = []
        
        for i in range(num_challenges):
            # Create naturalistic patterns
            challenge = self._generate_natural_pattern(image_size, seed=i)
            challenges.append(challenge)
            
        return challenges
    
    def _generate_perlin_texture(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate Perlin noise texture."""
        return self._generate_perlin_noise(size, octaves=3, scale=0.05)
    
    def _generate_voronoi_texture(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate Voronoi diagram texture."""
        h, w = size
        
        # Generate random seed points
        n_points = 20
        seed_points = torch.rand(n_points, 2) * torch.tensor([w, h])
        
        # Create coordinate grid
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        coords = torch.stack([x, y], dim=-1).float()
        
        # Compute distances to all seed points
        distances = torch.cdist(coords.view(-1, 2), seed_points)
        
        # Find closest seed point for each pixel
        closest = torch.argmin(distances, dim=1)
        voronoi = closest.view(h, w).float() / n_points
        
        # Convert to RGB
        image = voronoi.unsqueeze(0).repeat(3, 1, 1)
        
        return image
    
    def _generate_cellular_texture(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate cellular automata texture."""
        h, w = size
        
        # Initialize random grid
        grid = torch.rand(h, w) > 0.5
        
        # Apply cellular automata rules for several iterations
        for _ in range(5):
            new_grid = torch.zeros_like(grid)
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    # Count neighbors
                    neighbors = torch.sum(grid[i-1:i+2, j-1:j+2]) - grid[i, j]
                    
                    # Conway's Game of Life rules
                    if grid[i, j]:  # Cell is alive
                        new_grid[i, j] = neighbors in [2, 3]
                    else:  # Cell is dead
                        new_grid[i, j] = neighbors == 3
            
            grid = new_grid
        
        # Convert to float and RGB
        pattern = grid.float()
        image = pattern.unsqueeze(0).repeat(3, 1, 1)
        
        return image
    
    def _generate_fractal_texture(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate fractal texture using Mandelbrot set."""
        h, w = size
        
        # Create complex plane
        x = torch.linspace(-2, 2, w)
        y = torch.linspace(-2, 2, h)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        c = xx + 1j * yy
        
        # Mandelbrot iteration
        z = torch.zeros_like(c)
        pattern = torch.zeros(h, w)
        
        for i in range(50):
            mask = torch.abs(z) <= 2
            z[mask] = z[mask]**2 + c[mask]
            pattern[mask] = i
        
        # Normalize
        pattern = pattern / pattern.max()
        
        # Convert to RGB
        image = pattern.unsqueeze(0).repeat(3, 1, 1)
        
        return image
    
    def _generate_natural_pattern(self, size: Tuple[int, int], seed: int = 0) -> torch.Tensor:
        """Generate natural-looking pattern."""
        h, w = size
        np.random.seed(seed)
        
        # Combine multiple noise octaves for natural look
        pattern = torch.zeros(h, w)
        
        # Large-scale structure
        pattern += 0.5 * torch.from_numpy(
            np.random.normal(0, 1, (h//4, w//4))
        ).repeat_interleave(4, dim=0).repeat_interleave(4, dim=1)[:h, :w]
        
        # Medium-scale details
        pattern += 0.3 * torch.from_numpy(
            np.random.normal(0, 1, (h//2, w//2))
        ).repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)[:h, :w]
        
        # Fine details
        pattern += 0.2 * torch.from_numpy(np.random.normal(0, 1, (h, w)))
        
        # Apply smooth transitions if scipy available
        if SCIPY_AVAILABLE:
            pattern_np = gaussian_filter(pattern.numpy(), sigma=2)
            pattern = torch.from_numpy(pattern_np)
        
        # Normalize to [0, 1]
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        
        # Convert to RGB with slight color variation
        r_channel = pattern * 0.9 + 0.05
        g_channel = pattern * 0.95 + 0.025
        b_channel = pattern * 0.85 + 0.075
        
        image = torch.stack([r_channel, g_channel, b_channel], dim=0)
        
        return image
    
    def run_model(self, 
                  inputs: Union[torch.Tensor, List[torch.Tensor]],
                  return_intermediates: bool = True) -> Dict[str, Any]:
        """Run model on inputs and collect comprehensive outputs."""
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        
        inputs = inputs.to(self.device)
        
        # Ensure batch dimension
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        
        # Hook to capture intermediates
        intermediates = {}
        hooks = []
        
        if return_intermediates:
            def hook_fn(name):
                def hook(module, input, output):
                    intermediates[name] = output.detach()
                return hook
            
            # Register hooks on key layers
            for name, module in self.model.named_modules():
                if self._is_key_layer(name, module):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Run inference with timing
        start_time = time.perf_counter()
        
        with torch.no_grad():
            logits = self.model(inputs)
            
        inference_time = time.perf_counter() - start_time
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Extract embeddings from probe points
        embeddings = self.probe_extractor.extract_embeddings(intermediates)
        
        return {
            'logits': logits,
            'embeddings': embeddings,
            'intermediates': intermediates,
            'inference_time': inference_time,
            'samples_per_second': inputs.shape[0] / inference_time
        }
    
    def _is_key_layer(self, name: str, module: nn.Module) -> bool:
        """Determine if layer should be probed."""
        # Probe convolutional layers
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            return True
        # Probe pooling layers
        if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            return True
        # Probe normalization layers
        if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            return 'final' in name or 'last' in name
        # Probe attention layers
        if 'attention' in name.lower() or 'attn' in name.lower():
            return True
        # Probe final layers
        if isinstance(module, nn.Linear) and ('head' in name or 'fc' in name or 'classifier' in name):
            return True
        return False
    
    def logits_to_canonical_form(self, 
                                 logits: torch.Tensor,
                                 normalize: bool = True) -> torch.Tensor:
        """Convert logits to canonical form for comparison."""
        # Apply temperature scaling if configured
        temperature = self.config.get('temperature', 1.0)
        logits = logits / temperature
        
        if normalize:
            # Option 1: Softmax normalization
            if self.config.get('normalization', 'softmax') == 'softmax':
                canonical = torch.nn.functional.softmax(logits, dim=-1)
            
            # Option 2: Z-score normalization
            elif self.config.get('normalization') == 'zscore':
                mean = logits.mean(dim=-1, keepdim=True)
                std = logits.std(dim=-1, keepdim=True)
                canonical = (logits - mean) / (std + 1e-8)
            
            # Option 3: Min-max normalization
            elif self.config.get('normalization') == 'minmax':
                min_val = logits.min(dim=-1, keepdim=True)[0]
                max_val = logits.max(dim=-1, keepdim=True)[0]
                canonical = (logits - min_val) / (max_val - min_val + 1e-8)
        else:
            canonical = logits
            
        return canonical
    
    def verify_session(self, 
                      num_challenges: int = 10,
                      threshold: float = None,
                      challenge_types: List[str] = None) -> Dict[str, Any]:
        """Run complete verification session."""
        challenge_types = challenge_types or ['frequency', 'texture', 'natural']
        
        if self.config.get('verification_method') == 'sequential':
            return self._sequential_verification(num_challenges, threshold, challenge_types)
        else:
            return self._batch_verification(num_challenges, threshold, challenge_types)
    
    def _sequential_verification(self, 
                                num_challenges: int,
                                threshold: float,
                                challenge_types: List[str]) -> Dict[str, Any]:
        """Implement sequential verification with early stopping."""
        from pot.core.sequential import SequentialTester
        
        tester = SequentialTester(
            alpha=self.config.get('sprt_alpha', 0.05),
            beta=self.config.get('sprt_beta', 0.05),
            tau0=self.config.get('sprt_p0', 0.5),
            tau1=self.config.get('sprt_p1', 0.8)
        )
        
        results = []
        decision = None
        
        for i in range(num_challenges):
            # Select challenge type
            challenge_type = challenge_types[i % len(challenge_types)]
            
            # Generate challenge
            if challenge_type == 'frequency':
                challenge = self.generate_frequency_challenges(1)[0]
            elif challenge_type == 'texture':
                challenge = self.generate_texture_challenges(1)[0]
            else:  # natural
                challenge = self.generate_natural_challenges(1)[0]
            
            # Run model
            output = self.run_model(challenge)
            
            # Evaluate response
            success = self._evaluate_challenge_response(output, challenge_type)
            
            results.append({
                'challenge_type': challenge_type,
                'output': output,
                'success': success
            })
            
            # Update sequential test
            decision = tester.update(success)
            
            if decision is not None:
                break
        
        return {
            'verified': decision == 'reject',  # Reject H0 means genuine
            'confidence': tester.get_confidence(),
            'num_challenges': tester.num_trials,
            'results': results,
            'early_stopped': decision is not None
        }
    
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
    
    def _batch_verification(self, 
                           num_challenges: int,
                           threshold: float,
                           challenge_types: List[str]) -> Dict[str, Any]:
        """Implement batch verification without early stopping."""
        challenges = []
        challenge_metadata = []
        
        # Generate all challenges
        for i in range(num_challenges):
            challenge_type = challenge_types[i % len(challenge_types)]
            
            if challenge_type == 'frequency':
                challenge = self.generate_frequency_challenges(1)[0]
            elif challenge_type == 'texture':
                challenge = self.generate_texture_challenges(1)[0]
            else:  # natural
                challenge = self.generate_natural_challenges(1)[0]
            
            challenges.append(challenge)
            challenge_metadata.append({
                'type': challenge_type,
                'index': i
            })
        
        # Run all challenges
        outputs = []
        successes = []
        
        for challenge, metadata in zip(challenges, challenge_metadata):
            output = self.run_model(challenge)
            success = self._evaluate_challenge_response(output, metadata['type'])
            
            outputs.append(output)
            successes.append(success)
        
        # Compute statistics
        success_rate = sum(successes) / len(successes)
        confidence = self._compute_confidence(successes)
        
        return {
            'verified': success_rate >= (1 - threshold),
            'confidence': confidence,
            'success_rate': success_rate,
            'num_challenges': len(challenges),
            'results': list(zip(challenges, outputs, successes)),
            'early_stopped': False
        }
    
    def _evaluate_challenge_response(self, output: Dict[str, Any], challenge_type: str) -> bool:
        """Evaluate if model response to challenge is successful."""
        # Extract logits and compute canonical form
        logits = output['logits']
        canonical_logits = self.logits_to_canonical_form(logits)
        
        # Simple success criterion based on output confidence
        confidence = torch.max(torch.nn.functional.softmax(canonical_logits, dim=-1))
        
        # Type-specific thresholds
        if challenge_type == 'frequency':
            threshold = 0.1  # Lower threshold for frequency patterns
        elif challenge_type == 'texture':
            threshold = 0.15  # Medium threshold for textures
        else:  # natural
            threshold = 0.2  # Higher threshold for natural patterns
        
        return confidence.item() > threshold
    
    def _compute_confidence(self, successes: List[bool]) -> float:
        """Compute confidence interval for success rate."""
        n = len(successes)
        if n == 0:
            return 0.0
        
        p = sum(successes) / n
        # Wilson score interval
        z = 1.96  # 95% confidence
        denominator = 1 + z**2 / n
        centre_adjusted_probability = p + z**2 / (2 * n)
        adjusted_standard_deviation = torch.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
        
        lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
        upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
        
        # Return width of confidence interval (smaller is more confident)
        return float(upper_bound - lower_bound)
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration for vision verifier."""
        return {
            'temperature': 1.0,
            'normalization': 'softmax',
            'verification_method': 'sequential',
            'sprt_alpha': 0.05,
            'sprt_beta': 0.05,
            'sprt_p0': 0.5,
            'sprt_p1': 0.8,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'batch_size': 32,
            'num_workers': 4
        }
    
    def _load_model(self, model: Union[nn.Module, str]) -> nn.Module:
        """Load model from module or path."""
        if isinstance(model, str):
            # Load from path
            if model.endswith('.pth') or model.endswith('.pt'):
                loaded_model = torch.load(model, map_location='cpu')
                if isinstance(loaded_model, dict):
                    # Assume it's a state dict
                    raise ValueError("Cannot load model from state dict without model architecture")
                return loaded_model
            else:
                raise ValueError(f"Unsupported model format: {model}")
        else:
            return model

    def _batch_verification(self, 
                           num_challenges: int,
                           threshold: float,
                           challenge_types: List[str]) -> Dict[str, Any]:
        """
        Batch verification - run all challenges then evaluate.
        
        Args:
            num_challenges: Number of challenges to run
            threshold: Success threshold for verification
            challenge_types: Types of challenges to use
            
        Returns:
            Dictionary with verification results
        """
        challenges = []
        outputs = []
        
        # Generate all challenges
        for i in range(num_challenges):
            challenge_type = challenge_types[i % len(challenge_types)]
            
            if challenge_type == 'frequency':
                challenge = self.generate_frequency_challenges(1)[0]
            elif challenge_type == 'texture':
                challenge = self.generate_texture_challenges(1)[0]
            else:  # natural
                challenge = self.generate_natural_challenges(1)[0]
                
            challenges.append(challenge)
        
        # Run model on all challenges
        batch = torch.stack(challenges)
        batch_output = self.run_model(batch)
        
        # Evaluate all responses
        successes = []
        for i, challenge_type in enumerate(challenge_types[:num_challenges]):
            # Extract individual output
            output = {
                'logits': batch_output['logits'][i:i+1],
                'embeddings': {k: v[i:i+1] for k, v in batch_output['embeddings'].items()},
                'inference_time': batch_output['inference_time'] / num_challenges
            }
            
            success = self._evaluate_challenge_response(output, challenge_type)
            successes.append(success)
            
            outputs.append({
                'challenge_type': challenge_type,
                'output': output,
                'success': success
            })
        
        # Compute verification result
        success_rate = sum(successes) / len(successes)
        threshold = threshold or self.config.get('success_threshold', 0.7)
        
        return {
            'verified': success_rate >= threshold,
            'confidence': success_rate,
            'num_challenges': num_challenges,
            'success_rate': success_rate,
            'threshold': threshold,
            'results': outputs
        }

    def _evaluate_challenge_response(self, 
                                    output: Dict[str, Any],
                                    challenge_type: str) -> bool:
        """
        Evaluate if model response is consistent with expected behavior.
        
        Args:
            output: Model output dictionary
            challenge_type: Type of challenge being evaluated
            
        Returns:
            True if response indicates genuine model
        """
        # Get reference statistics for challenge type
        ref_stats = self._get_reference_statistics(challenge_type)
        
        # Extract key metrics from output
        logits = output['logits']
        embeddings = output['embeddings']
        
        # Canonicalize logits
        canonical_logits = self.logits_to_canonical_form(logits)
        
        # Compute distances
        checks = []
        
        # Check 1: Logit distribution
        if 'logit_mean' in ref_stats:
            logit_mean = canonical_logits.mean().item()
            logit_std = canonical_logits.std().item()
            
            mean_diff = abs(logit_mean - ref_stats['logit_mean'])
            std_diff = abs(logit_std - ref_stats['logit_std'])
            
            checks.append(mean_diff < ref_stats.get('mean_tolerance', 0.1))
            checks.append(std_diff < ref_stats.get('std_tolerance', 0.1))
        
        # Check 2: Embedding consistency
        if 'penultimate' in embeddings and 'embedding_norm' in ref_stats:
            emb_norm = torch.norm(embeddings['penultimate'], p=2).item()
            norm_ratio = emb_norm / ref_stats['embedding_norm']
            
            checks.append(0.8 < norm_ratio < 1.2)  # Within 20% of expected
        
        # Check 3: Activation patterns
        if challenge_type == 'frequency':
            # Frequency challenges should activate edge detectors
            if 'early' in embeddings:
                early_activation = embeddings['early'].abs().mean().item()
                checks.append(early_activation > ref_stats.get('min_early_activation', 0.1))
                
        elif challenge_type == 'texture':
            # Texture challenges should activate mid-level features
            if 'mid' in embeddings:
                mid_activation = embeddings['mid'].abs().mean().item()
                checks.append(mid_activation > ref_stats.get('min_mid_activation', 0.15))
        
        # Check 4: Temporal consistency (inference time)
        if 'inference_time' in output:
            time_ratio = output['inference_time'] / ref_stats.get('expected_time', 0.01)
            checks.append(0.5 < time_ratio < 2.0)  # Within 2x of expected
        
        # Require majority of checks to pass
        return sum(checks) >= len(checks) * 0.6 if checks else False

    def _get_reference_statistics(self, challenge_type: str) -> Dict[str, float]:
        """
        Get reference statistics for challenge type.
        These should be calibrated on genuine model.
        
        Args:
            challenge_type: Type of challenge
            
        Returns:
            Dictionary of reference statistics
        """
        # Check if we have calibrated statistics
        if hasattr(self, '_reference_statistics') and self._reference_statistics:
            return self._reference_statistics.get(challenge_type, {})
        
        # Default statistics - should be loaded from calibration
        default_stats = {
            'frequency': {
                'logit_mean': 0.0,
                'logit_std': 1.0,
                'embedding_norm': 10.0,
                'min_early_activation': 0.1,
                'expected_time': 0.01,
                'mean_tolerance': 0.15,
                'std_tolerance': 0.2
            },
            'texture': {
                'logit_mean': 0.0,
                'logit_std': 1.0,
                'embedding_norm': 12.0,
                'min_mid_activation': 0.15,
                'expected_time': 0.01,
                'mean_tolerance': 0.15,
                'std_tolerance': 0.2
            },
            'natural': {
                'logit_mean': 0.0,
                'logit_std': 1.0,
                'embedding_norm': 15.0,
                'expected_time': 0.01,
                'mean_tolerance': 0.2,
                'std_tolerance': 0.25
            }
        }
        
        return default_stats.get(challenge_type, default_stats['natural'])


class ProbeExtractor:
    """Extract feature probes from model intermediates."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def extract_embeddings(self, intermediates: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract embeddings from intermediate activations."""
        embeddings = {}
        
        for name, activation in intermediates.items():
            # Global average pooling for spatial features
            if activation.dim() == 4:  # (B, C, H, W)
                embedding = torch.nn.functional.adaptive_avg_pool2d(activation, (1, 1))
                embedding = embedding.flatten(1)
            elif activation.dim() == 3:  # (B, L, D) - sequence
                embedding = activation.mean(dim=1)
            else:  # Already flattened
                embedding = activation
                
            embeddings[name] = embedding
            
        return embeddings
    
    def extract_with_hooks(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract embeddings using forward hooks."""
        # Simple implementation - store intermediates and extract embeddings
        intermediates = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                intermediates[name] = output.detach()
            return hook
        
        # Register hooks for major module types
        try:
            module_count = 0
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d)):
                    hooks.append(module.register_forward_hook(hook_fn(f"{name}_{module_count}")))
                    module_count += 1
                    if module_count >= 5:  # Limit number of hooks
                        break
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(x)
            
            # Extract embeddings from intermediates
            embeddings = self.extract_embeddings(intermediates)
            
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
        
        return embeddings


class VisionDistanceMetrics:
    """Distance metrics for vision model verification."""
    
    @staticmethod
    def cosine_distance(x1: torch.Tensor, x2: torch.Tensor) -> float:
        """Compute cosine distance between tensors."""
        cos_sim = torch.nn.functional.cosine_similarity(
            x1.flatten(), x2.flatten(), dim=0
        )
        return 1.0 - cos_sim.item()
    
    @staticmethod
    def euclidean_distance(x1: torch.Tensor, x2: torch.Tensor) -> float:
        """Compute normalized Euclidean distance."""
        dist = torch.norm(x1 - x2, p=2).item()
        # Normalize by tensor size
        return dist / torch.numel(x1)**0.5
    
    @staticmethod
    def manhattan_distance(x1: torch.Tensor, x2: torch.Tensor) -> float:
        """Compute normalized Manhattan distance."""
        dist = torch.norm(x1 - x2, p=1).item()
        return dist / torch.numel(x1)
    
    @staticmethod
    def kl_divergence(x1: torch.Tensor, x2: torch.Tensor) -> float:
        """Compute KL divergence for probability distributions."""
        p = torch.nn.functional.softmax(x1.flatten(), dim=0)
        q = torch.nn.functional.softmax(x2.flatten(), dim=0)
        
        # Add small epsilon for numerical stability
        eps = 1e-8
        p = p + eps
        q = q + eps
        
        kl = torch.nn.functional.kl_div(
            torch.log(q), p, reduction='sum'
        )
        return kl.item()


class FrequencyChallenger:
    """Generate frequency-based challenges."""
    
    def generate(self, config: Dict[str, Any]) -> torch.Tensor:
        return self._generate_sine_grating(**config)
    
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


class TextureChallenger:
    """Generate texture-based challenges."""
    
    def generate(self, config: Dict[str, Any]) -> torch.Tensor:
        return self._generate_perlin_noise(**config)
    
    def _generate_perlin_noise(self, size: Tuple[int, int],
                              octaves: int,
                              scale: float,
                              seed: Optional[int] = None) -> torch.Tensor:
        """Generate Perlin noise texture"""
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


# Enhanced VisionVerifier class with full implementation
class EnhancedVisionVerifier:
    """Enhanced VisionVerifier with complete implementation from user specification."""
    
    def __init__(self, 
                 model: Union[nn.Module, str],
                 config: Optional[Dict[str, Any]] = None,
                 device: str = 'cuda'):
        """Initialize Vision verifier with model and configuration."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model).to(self.device)
        self.model.eval()
        
        self.config = config or self._default_config()
        self.challenge_generator = FrequencyChallenger()
        self.texture_generator = TextureChallenger()
        self.probe_extractor = ProbeExtractor(self.model)
        self.distance_metrics = VisionDistanceMetrics()
    
    def generate_frequency_challenges(self, 
                                     num_challenges: int = 10,
                                     image_size: Tuple[int, int] = (224, 224),
                                     frequency_bands: List[str] = None) -> List[torch.Tensor]:
        """Generate frequency-based challenge images."""
        frequency_bands = frequency_bands or ['low', 'mid', 'high', 'mixed']
        challenges = []
        
        for i in range(num_challenges):
            band = frequency_bands[i % len(frequency_bands)]
            
            if band == 'low':
                challenge = self._generate_low_frequency(image_size)
            elif band == 'mid':
                challenge = self._generate_mid_frequency(image_size)
            elif band == 'high':
                challenge = self._generate_high_frequency(image_size)
            elif band == 'mixed':
                challenge = self._generate_mixed_frequency(image_size)
                
            challenges.append(challenge)
            
        return challenges
    
    def _generate_low_frequency(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate low-frequency pattern."""
        h, w = size
        
        # Create gradient patterns
        x = torch.linspace(-1, 1, w)
        y = torch.linspace(-1, 1, h)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        # Radial gradient
        pattern = torch.sqrt(xx**2 + yy**2)
        
        # Add some sinusoidal patterns
        pattern += 0.5 * torch.sin(2 * np.pi * xx)
        pattern += 0.5 * torch.cos(2 * np.pi * yy)
        
        # Normalize to [0, 1]
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        
        # Convert to RGB
        image = pattern.unsqueeze(0).repeat(3, 1, 1)
        
        return image
    
    def _generate_high_frequency(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate high-frequency pattern."""
        h, w = size
        
        # Generate noise patterns
        pattern = torch.randn(h, w)
        
        # Add checkerboard pattern
        checker_size = 4
        for i in range(0, h, checker_size*2):
            for j in range(0, w, checker_size*2):
                if i+checker_size < h and j+checker_size < w:
                    pattern[i:i+checker_size, j:j+checker_size] *= -1
                if i+checker_size < h and j+checker_size*2 < w:
                    pattern[i:i+checker_size, j+checker_size:j+checker_size*2] *= -1
                    
        # Normalize
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        
        # Convert to RGB
        image = pattern.unsqueeze(0).repeat(3, 1, 1)
        
        return image
    
    def _load_model(self, model: Union[nn.Module, str]) -> nn.Module:
        """Load model from module or path."""
        if isinstance(model, str):
            # Load from path
            if model.endswith('.pth') or model.endswith('.pt'):
                loaded_model = torch.load(model, map_location='cpu')
                if isinstance(loaded_model, dict):
                    # Assume it's a state dict
                    raise ValueError("Cannot load model from state dict without model architecture")
                return loaded_model
            else:
                raise ValueError(f"Unsupported model format: {model}")
        else:
            return model
    
    def _batch_verification(self, 
                           num_challenges: int,
                           threshold: float,
                           challenge_types: List[str]) -> Dict[str, Any]:
        """
        Batch verification - run all challenges then evaluate.
        
        Args:
            num_challenges: Number of challenges to run
            threshold: Success threshold for verification
            challenge_types: Types of challenges to use
            
        Returns:
            Dictionary with verification results
        """
        challenges = []
        outputs = []
        
        # Generate all challenges
        for i in range(num_challenges):
            challenge_type = challenge_types[i % len(challenge_types)]
            
            if challenge_type == 'frequency':
                challenge = self.generate_frequency_challenges(1)[0]
            elif challenge_type == 'texture':
                challenge = self.generate_texture_challenges(1)[0]
            else:  # natural
                challenge = self.generate_natural_challenges(1)[0]
                
            challenges.append(challenge)
        
        # Run model on all challenges
        batch = torch.stack(challenges)
        batch_output = self.run_model(batch)
        
        # Evaluate all responses
        successes = []
        for i, challenge_type in enumerate(challenge_types[:num_challenges]):
            # Extract individual output
            output = {
                'logits': batch_output['logits'][i:i+1],
                'embeddings': {k: v[i:i+1] for k, v in batch_output['embeddings'].items()},
                'inference_time': batch_output['inference_time'] / num_challenges
            }
            
            success = self._evaluate_challenge_response(output, challenge_type)
            successes.append(success)
            
            outputs.append({
                'challenge_type': challenge_type,
                'output': output,
                'success': success
            })
        
        # Compute verification result
        success_rate = sum(successes) / len(successes)
        threshold = threshold or self.config.get('success_threshold', 0.7)
        
        return {
            'verified': success_rate >= threshold,
            'confidence': success_rate,
            'num_challenges': num_challenges,
            'success_rate': success_rate,
            'threshold': threshold,
            'results': outputs
        }

    def _evaluate_challenge_response(self, 
                                    output: Dict[str, Any],
                                    challenge_type: str) -> bool:
        """
        Evaluate if model response is consistent with expected behavior.
        
        Args:
            output: Model output dictionary
            challenge_type: Type of challenge being evaluated
            
        Returns:
            True if response indicates genuine model
        """
        # Get reference statistics for challenge type
        ref_stats = self._get_reference_statistics(challenge_type)
        
        # Extract key metrics from output
        logits = output['logits']
        embeddings = output['embeddings']
        
        # Canonicalize logits
        canonical_logits = self.logits_to_canonical_form(logits)
        
        # Compute distances
        checks = []
        
        # Check 1: Logit distribution
        if 'logit_mean' in ref_stats:
            logit_mean = canonical_logits.mean().item()
            logit_std = canonical_logits.std().item()
            
            mean_diff = abs(logit_mean - ref_stats['logit_mean'])
            std_diff = abs(logit_std - ref_stats['logit_std'])
            
            checks.append(mean_diff < ref_stats.get('mean_tolerance', 0.1))
            checks.append(std_diff < ref_stats.get('std_tolerance', 0.1))
        
        # Check 2: Embedding consistency
        if 'penultimate' in embeddings and 'embedding_norm' in ref_stats:
            emb_norm = torch.norm(embeddings['penultimate'], p=2).item()
            norm_ratio = emb_norm / ref_stats['embedding_norm']
            
            checks.append(0.8 < norm_ratio < 1.2)  # Within 20% of expected
        
        # Check 3: Activation patterns
        if challenge_type == 'frequency':
            # Frequency challenges should activate edge detectors
            if 'early' in embeddings:
                early_activation = embeddings['early'].abs().mean().item()
                checks.append(early_activation > ref_stats.get('min_early_activation', 0.1))
                
        elif challenge_type == 'texture':
            # Texture challenges should activate mid-level features
            if 'mid' in embeddings:
                mid_activation = embeddings['mid'].abs().mean().item()
                checks.append(mid_activation > ref_stats.get('min_mid_activation', 0.15))
        
        # Check 4: Temporal consistency (inference time)
        if 'inference_time' in output:
            time_ratio = output['inference_time'] / ref_stats.get('expected_time', 0.01)
            checks.append(0.5 < time_ratio < 2.0)  # Within 2x of expected
        
        # Require majority of checks to pass
        return sum(checks) >= len(checks) * 0.6 if checks else False

    def _get_reference_statistics(self, challenge_type: str) -> Dict[str, float]:
        """
        Get reference statistics for challenge type.
        These should be calibrated on genuine model.
        
        Args:
            challenge_type: Type of challenge
            
        Returns:
            Dictionary of reference statistics
        """
        # Check if we have calibrated statistics
        if hasattr(self, '_reference_statistics') and self._reference_statistics:
            return self._reference_statistics.get(challenge_type, {})
        
        # Default statistics - should be loaded from calibration
        default_stats = {
            'frequency': {
                'logit_mean': 0.0,
                'logit_std': 1.0,
                'embedding_norm': 10.0,
                'min_early_activation': 0.1,
                'expected_time': 0.01,
                'mean_tolerance': 0.15,
                'std_tolerance': 0.2
            },
            'texture': {
                'logit_mean': 0.0,
                'logit_std': 1.0,
                'embedding_norm': 12.0,
                'min_mid_activation': 0.15,
                'expected_time': 0.01,
                'mean_tolerance': 0.15,
                'std_tolerance': 0.2
            },
            'natural': {
                'logit_mean': 0.0,
                'logit_std': 1.0,
                'embedding_norm': 15.0,
                'expected_time': 0.01,
                'mean_tolerance': 0.2,
                'std_tolerance': 0.25
            }
        }
        
        return default_stats.get(challenge_type, default_stats['natural'])
    
    def run_model(self, inputs: torch.Tensor) -> Dict[str, Any]:
        """Run model on inputs and return outputs with embeddings."""
        with torch.no_grad():
            # Run model
            logits = self.model(inputs)
            
            # Extract embeddings using probe extractor
            embeddings = self.probe_extractor.extract_with_hooks(inputs)
            
            # Measure inference time
            start_time = time.time()
            _ = self.model(inputs)
            inference_time = time.time() - start_time
            
            return {
                'logits': logits,
                'embeddings': embeddings or {},
                'inference_time': inference_time
            }
    
    def generate_texture_challenges(self, 
                                  num_challenges: int = 10,
                                  image_size: Tuple[int, int] = (224, 224)) -> List[torch.Tensor]:
        """Generate texture-based challenge images."""
        challenges = []
        
        for i in range(num_challenges):
            # Use the texture generator to create varied patterns
            if hasattr(self.texture_generator, 'generate_perlin_noise'):
                pattern = self.texture_generator.generate_perlin_noise(
                    size=image_size,
                    octaves=2 + (i % 3),
                    scale=10 + (i % 20)
                )
            else:
                # Fallback to simple pattern
                pattern = self._generate_simple_texture(image_size, i)
            
            challenges.append(pattern)
        
        return challenges
    
    def _generate_simple_texture(self, size: Tuple[int, int], seed: int) -> torch.Tensor:
        """Generate simple texture pattern as fallback."""
        h, w = size
        torch.manual_seed(seed)
        
        # Create random texture
        pattern = torch.randn(h, w)
        
        # Add some structure
        pattern = torch.sin(pattern * 5) + torch.cos(pattern * 3)
        
        # Normalize
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        
        # Convert to RGB
        return pattern.unsqueeze(0).repeat(3, 1, 1)
    
    def verify_session(self, 
                      num_challenges: int = 5,
                      challenge_types: List[str] = None) -> Dict[str, Any]:
        """Verify model using batch verification."""
        challenge_types = challenge_types or ['frequency', 'texture']
        
        return self._batch_verification(
            num_challenges=num_challenges,
            threshold=0.5,
            challenge_types=challenge_types
        )
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'temperature': 1.0,
            'normalization': 'softmax',
            'verification_method': 'batch',
            'device': 'cpu'
        }
    
    def logits_to_canonical_form(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to canonical form (normalized)."""
        # Apply temperature scaling
        temperature = self.config.get('temperature', 1.0)
        logits = logits / temperature
        
        # Apply normalization
        normalization = self.config.get('normalization', 'softmax')
        
        if normalization == 'softmax':
            return torch.softmax(logits, dim=-1)
        elif normalization == 'sigmoid':
            return torch.sigmoid(logits)
        elif normalization == 'none':
            return logits
        else:
            # Default to softmax
            return torch.softmax(logits, dim=-1)


def compute_activation_statistics(activations: torch.Tensor) -> Dict[str, float]:
    """Compute statistics for activation tensors."""
    flat = activations.flatten()
    
    return {
        'mean': float(torch.mean(flat)),
        'std': float(torch.std(flat)),
        'min': float(torch.min(flat)),
        'max': float(torch.max(flat)),
        'l2_norm': float(torch.norm(flat, p=2)),
        'sparsity': float(torch.sum(flat == 0) / flat.numel())
    }


# Additional utility functions for complete implementation

def preprocess_image(image: torch.Tensor, size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Preprocess image tensor for vision models."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Resize if needed
    if image.shape[-2:] != size:
        image = torch.nn.functional.interpolate(image, size=size, mode='bilinear', align_corners=False)
    
    return image


class VisionVerifierCalibrator:
    """Calibrate verifier on genuine model."""
    
    def __init__(self, verifier: VisionVerifier):
        """
        Initialize calibrator.
        
        Args:
            verifier: VisionVerifier instance to calibrate
        """
        self.verifier = verifier
        self.statistics = {}
        
    def calibrate(self, 
                 num_samples: int = 100,
                 challenge_types: List[str] = None) -> Dict[str, Dict]:
        """
        Calibrate verifier by collecting statistics on genuine model.
        
        Args:
            num_samples: Number of samples per challenge type
            challenge_types: Types of challenges to calibrate on
            
        Returns:
            Dictionary of calibrated statistics per challenge type
        """
        challenge_types = challenge_types or ['frequency', 'texture', 'natural']
        
        print(f"Calibrating verifier on {num_samples} samples per challenge type...")
        
        for challenge_type in challenge_types:
            print(f"Calibrating {challenge_type} challenges...")
            self.statistics[challenge_type] = self._calibrate_challenge_type(
                challenge_type, num_samples
            )
        
        # Set calibrated statistics on verifier
        self.verifier._reference_statistics = self.statistics
        
        print("✓ Calibration completed")
        return self.statistics
    
    def _calibrate_challenge_type(self, 
                                 challenge_type: str,
                                 num_samples: int) -> Dict[str, float]:
        """
        Calibrate for specific challenge type.
        
        Args:
            challenge_type: Type of challenge to calibrate
            num_samples: Number of samples to collect
            
        Returns:
            Dictionary of statistics for this challenge type
        """
        logit_means = []
        logit_stds = []
        embedding_norms = []
        activation_levels = {}
        inference_times = []
        
        for i in range(num_samples):
            try:
                # Generate challenge
                if challenge_type == 'frequency':
                    challenge = self.verifier.generate_frequency_challenges(1)[0]
                elif challenge_type == 'texture':
                    challenge = self.verifier.generate_texture_challenges(1)[0]
                else:  # natural
                    challenge = self.verifier.generate_natural_challenges(1)[0]
                
                # Run model
                output = self.verifier.run_model(challenge.unsqueeze(0))
                
                # Collect statistics
                canonical_logits = self.verifier.logits_to_canonical_form(output['logits'])
                logit_means.append(canonical_logits.mean().item())
                logit_stds.append(canonical_logits.std().item())
                
                if 'penultimate' in output['embeddings']:
                    embedding_norms.append(
                        torch.norm(output['embeddings']['penultimate'], p=2).item()
                    )
                
                for layer_name, embedding in output['embeddings'].items():
                    if layer_name not in activation_levels:
                        activation_levels[layer_name] = []
                    activation_levels[layer_name].append(embedding.abs().mean().item())
                
                inference_times.append(output['inference_time'])
                
                # Progress indicator
                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{num_samples} samples")
                    
            except Exception as e:
                print(f"  Warning: Failed to process sample {i}: {e}")
                continue
        
        # Compute aggregate statistics
        stats = {
            'logit_mean': float(np.mean(logit_means)) if logit_means else 0.0,
            'logit_std': float(np.mean(logit_stds)) if logit_stds else 1.0,
            'logit_mean_std': float(np.std(logit_means)) if logit_means else 0.1,
            'logit_std_std': float(np.std(logit_stds)) if logit_stds else 0.1,
            'embedding_norm': float(np.mean(embedding_norms)) if embedding_norms else 10.0,
            'expected_time': float(np.median(inference_times)) if inference_times else 0.01,
            'mean_tolerance': float(2 * np.std(logit_means)) if logit_means else 0.15,
            'std_tolerance': float(2 * np.std(logit_stds)) if logit_stds else 0.15
        }
        
        # Add activation statistics
        for layer_name, activations in activation_levels.items():
            if activations:
                stats[f'mean_{layer_name}_activation'] = float(np.mean(activations))
                stats[f'min_{layer_name}_activation'] = float(np.percentile(activations, 10))
                stats[f'max_{layer_name}_activation'] = float(np.percentile(activations, 90))
        
        return stats
    
    def save_calibration(self, path: str):
        """
        Save calibration statistics to file.
        
        Args:
            path: Path to save calibration file
        """
        import json
        
        with open(path, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        
        print(f"✓ Calibration saved to {path}")
    
    def load_calibration(self, path: str):
        """
        Load calibration statistics from file.
        
        Args:
            path: Path to calibration file
        """
        import json
        
        with open(path, 'r') as f:
            self.statistics = json.load(f)
            
        # Set loaded statistics on verifier
        self.verifier._reference_statistics = self.statistics
        
        print(f"✓ Calibration loaded from {path}")
        print(f"  Challenge types: {list(self.statistics.keys())}")
    
    def validate_calibration(self, 
                           num_validation_samples: int = 50) -> Dict[str, float]:
        """
        Validate calibration by running verification on genuine model.
        
        Args:
            num_validation_samples: Number of validation samples
            
        Returns:
            Dictionary with validation results
        """
        print(f"Validating calibration with {num_validation_samples} samples...")
        
        validation_results = {}
        
        for challenge_type in self.statistics.keys():
            successes = []
            
            for i in range(num_validation_samples):
                try:
                    # Generate challenge
                    if challenge_type == 'frequency':
                        challenge = self.verifier.generate_frequency_challenges(1)[0]
                    elif challenge_type == 'texture':
                        challenge = self.verifier.generate_texture_challenges(1)[0]
                    else:  # natural
                        challenge = self.verifier.generate_natural_challenges(1)[0]
                    
                    # Run model and evaluate
                    output = self.verifier.run_model(challenge.unsqueeze(0))
                    success = self.verifier._evaluate_challenge_response(output, challenge_type)
                    successes.append(success)
                    
                except Exception as e:
                    print(f"  Warning: Validation sample {i} failed: {e}")
                    continue
            
            success_rate = sum(successes) / len(successes) if successes else 0.0
            validation_results[challenge_type] = success_rate
            
            print(f"  {challenge_type}: {success_rate:.2%} success rate")
        
        overall_success = np.mean(list(validation_results.values()))
        validation_results['overall'] = overall_success
        
        print(f"✓ Overall validation success rate: {overall_success:.2%}")
        
        return validation_results
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """
        Get summary of calibration statistics.
        
        Returns:
            Dictionary with calibration summary
        """
        if not self.statistics:
            return {"error": "No calibration data available"}
        
        summary = {
            'challenge_types': list(self.statistics.keys()),
            'num_challenge_types': len(self.statistics),
            'statistics_per_type': {}
        }
        
        for challenge_type, stats in self.statistics.items():
            summary['statistics_per_type'][challenge_type] = {
                'logit_mean': stats.get('logit_mean', 0.0),
                'logit_std': stats.get('logit_std', 1.0),
                'embedding_norm': stats.get('embedding_norm', 10.0),
                'expected_time': stats.get('expected_time', 0.01),
                'mean_tolerance': stats.get('mean_tolerance', 0.15),
                'std_tolerance': stats.get('std_tolerance', 0.15)
            }
        
        return summary