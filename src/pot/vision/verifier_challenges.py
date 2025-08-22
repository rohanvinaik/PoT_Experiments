"""
Challenge generation utilities for vision verification.
This module contains all challenge generation functions used by verifiers.
"""

import numpy as np
import torch
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum
import math


class PatternType(Enum):
    """Types of patterns for challenge generation"""
    SINE_GRATING = "sine_grating"
    CHECKERBOARD = "checkerboard"
    GAUSSIAN_NOISE = "gaussian_noise"
    PERLIN_NOISE = "perlin_noise"
    GABOR_FILTER = "gabor_filter"
    RADIAL_GRADIENT = "radial_gradient"


class ChallengeGenerator:
    """Unified challenge generator for vision models"""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize challenge generator.
        
        Args:
            device: Device to generate challenges on
        """
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed=42)
    
    def generate_sine_grating(self, 
                             size: Tuple[int, int],
                             frequency: float = 0.1,
                             angle: float = 0.0,
                             phase: float = 0.0) -> torch.Tensor:
        """
        Generate a sine grating pattern.
        This consolidates the duplicate _generate_sine_grating implementations.
        
        Args:
            size: (height, width) of the pattern
            frequency: Spatial frequency of the grating
            angle: Angle of the grating in radians
            phase: Phase offset of the sine wave
            
        Returns:
            Tensor containing the sine grating pattern
        """
        h, w = size
        
        # Create coordinate grids
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        
        # Rotate coordinates
        x_rot = xx * np.cos(angle) - yy * np.sin(angle)
        
        # Generate sine grating
        grating = np.sin(2 * np.pi * frequency * x_rot + phase)
        
        # Normalize to [0, 1]
        grating = (grating + 1) / 2
        
        # Convert to tensor
        pattern = torch.from_numpy(grating).float().to(self.device)
        
        # Add batch and channel dimensions if needed
        if pattern.dim() == 2:
            pattern = pattern.unsqueeze(0).unsqueeze(0)
        
        return pattern
    
    def generate_checkerboard(self,
                            size: Tuple[int, int],
                            square_size: int = 8) -> torch.Tensor:
        """
        Generate a checkerboard pattern.
        
        Args:
            size: (height, width) of the pattern
            square_size: Size of each square
            
        Returns:
            Tensor containing the checkerboard pattern
        """
        h, w = size
        
        # Create checkerboard
        board = np.zeros((h, w))
        for i in range(0, h, square_size * 2):
            for j in range(0, w, square_size * 2):
                board[i:i+square_size, j:j+square_size] = 1
                board[i+square_size:i+2*square_size, j+square_size:j+2*square_size] = 1
        
        pattern = torch.from_numpy(board).float().to(self.device)
        
        if pattern.dim() == 2:
            pattern = pattern.unsqueeze(0).unsqueeze(0)
        
        return pattern
    
    def generate_gaussian_noise(self,
                               size: Tuple[int, int],
                               mean: float = 0.5,
                               std: float = 0.1) -> torch.Tensor:
        """
        Generate Gaussian noise pattern.
        
        Args:
            size: (height, width) of the pattern
            mean: Mean of the Gaussian distribution
            std: Standard deviation
            
        Returns:
            Tensor containing Gaussian noise
        """
        h, w = size
        noise = torch.randn(1, 1, h, w, device=self.device) * std + mean
        noise = torch.clamp(noise, 0, 1)
        return noise
    
    def generate_gabor_filter(self,
                             size: Tuple[int, int],
                             frequency: float = 0.1,
                             theta: float = 0.0,
                             sigma: float = 1.0,
                             phase: float = 0.0) -> torch.Tensor:
        """
        Generate a Gabor filter pattern.
        
        Args:
            size: (height, width) of the pattern
            frequency: Frequency of the sinusoidal component
            theta: Orientation angle
            sigma: Standard deviation of Gaussian envelope
            phase: Phase offset
            
        Returns:
            Tensor containing Gabor filter pattern
        """
        h, w = size
        
        # Create coordinate grids
        x = np.linspace(-2, 2, w)
        y = np.linspace(-2, 2, h)
        xx, yy = np.meshgrid(x, y)
        
        # Rotate coordinates
        x_rot = xx * np.cos(theta) + yy * np.sin(theta)
        y_rot = -xx * np.sin(theta) + yy * np.cos(theta)
        
        # Gaussian envelope
        gaussian = np.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2))
        
        # Sinusoidal carrier
        sinusoid = np.cos(2 * np.pi * frequency * x_rot + phase)
        
        # Gabor filter
        gabor = gaussian * sinusoid
        
        # Normalize to [0, 1]
        gabor = (gabor - gabor.min()) / (gabor.max() - gabor.min() + 1e-8)
        
        pattern = torch.from_numpy(gabor).float().to(self.device)
        
        if pattern.dim() == 2:
            pattern = pattern.unsqueeze(0).unsqueeze(0)
        
        return pattern
    
    def generate_radial_gradient(self,
                                size: Tuple[int, int],
                                center: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """
        Generate a radial gradient pattern.
        
        Args:
            size: (height, width) of the pattern
            center: Center point (relative coordinates 0-1)
            
        Returns:
            Tensor containing radial gradient
        """
        h, w = size
        
        if center is None:
            center = (0.5, 0.5)
        
        # Create coordinate grids
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        
        # Calculate distance from center
        dist = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
        
        # Normalize
        gradient = 1 - (dist / dist.max())
        
        pattern = torch.from_numpy(gradient).float().to(self.device)
        
        if pattern.dim() == 2:
            pattern = pattern.unsqueeze(0).unsqueeze(0)
        
        return pattern
    
    def generate_batch(self,
                      pattern_type: PatternType,
                      batch_size: int,
                      size: Tuple[int, int],
                      **kwargs) -> torch.Tensor:
        """
        Generate a batch of patterns.
        
        Args:
            pattern_type: Type of pattern to generate
            batch_size: Number of patterns
            size: Size of each pattern
            **kwargs: Additional arguments for pattern generation
            
        Returns:
            Batch of patterns
        """
        patterns = []
        
        for _ in range(batch_size):
            if pattern_type == PatternType.SINE_GRATING:
                pattern = self.generate_sine_grating(size, **kwargs)
            elif pattern_type == PatternType.CHECKERBOARD:
                pattern = self.generate_checkerboard(size, **kwargs)
            elif pattern_type == PatternType.GAUSSIAN_NOISE:
                pattern = self.generate_gaussian_noise(size, **kwargs)
            elif pattern_type == PatternType.GABOR_FILTER:
                pattern = self.generate_gabor_filter(size, **kwargs)
            elif pattern_type == PatternType.RADIAL_GRADIENT:
                pattern = self.generate_radial_gradient(size, **kwargs)
            else:
                raise ValueError(f"Unknown pattern type: {pattern_type}")
            
            patterns.append(pattern)
        
        return torch.cat(patterns, dim=0)
    
    def generate_adversarial_pattern(self,
                                    model: torch.nn.Module,
                                    base_input: torch.Tensor,
                                    target: torch.Tensor,
                                    epsilon: float = 0.01,
                                    steps: int = 10) -> torch.Tensor:
        """
        Generate adversarial pattern using FGSM or PGD.
        
        Args:
            model: Target model
            base_input: Base input to perturb
            target: Target output
            epsilon: Perturbation budget
            steps: Number of PGD steps
            
        Returns:
            Adversarial pattern
        """
        perturbed = base_input.clone().detach().requires_grad_(True)
        
        for _ in range(steps):
            output = model(perturbed)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            
            # Update perturbation
            with torch.no_grad():
                perturbation = epsilon * perturbed.grad.sign()
                perturbed = perturbed + perturbation
                perturbed = torch.clamp(perturbed, 0, 1)
                perturbed = perturbed.detach().requires_grad_(True)
        
        return perturbed.detach()


class ChallengeLibrary:
    """Library of pre-defined challenge sets"""
    
    @staticmethod
    def get_standard_challenges(size: Tuple[int, int] = (224, 224)) -> List[torch.Tensor]:
        """Get standard set of challenges for testing"""
        generator = ChallengeGenerator()
        challenges = []
        
        # Add various patterns
        challenges.append(generator.generate_sine_grating(size, frequency=0.05))
        challenges.append(generator.generate_sine_grating(size, frequency=0.1, angle=np.pi/4))
        challenges.append(generator.generate_checkerboard(size, square_size=16))
        challenges.append(generator.generate_gaussian_noise(size))
        challenges.append(generator.generate_gabor_filter(size))
        challenges.append(generator.generate_radial_gradient(size))
        
        return challenges
    
    @staticmethod
    def get_edge_case_challenges(size: Tuple[int, int] = (224, 224)) -> List[torch.Tensor]:
        """Get edge case challenges"""
        generator = ChallengeGenerator()
        challenges = []
        
        # Extreme values
        challenges.append(torch.zeros(1, 1, *size))  # All black
        challenges.append(torch.ones(1, 1, *size))   # All white
        
        # High frequency patterns
        challenges.append(generator.generate_sine_grating(size, frequency=0.5))
        
        # Very low frequency
        challenges.append(generator.generate_sine_grating(size, frequency=0.01))
        
        # Random noise
        challenges.append(torch.rand(1, 1, *size))
        
        return challenges