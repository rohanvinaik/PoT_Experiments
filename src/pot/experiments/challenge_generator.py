"""
Challenge Generators for PoT Framework

This module implements challenge families matching the paper specifications:
- Vision challenges: adversarial patches, style transfer, compression
- Language challenges: paraphrasing, substitution, perturbation  
- Multimodal challenges: cross-modal verification

Features:
- Configurable difficulty levels (0.0 = easy, 1.0 = hard)
- Deterministic generation with seed management
- Comprehensive metadata tracking
- Distance-based verification
- Baseline challenges for testing
"""

import os
import sys
import abc
import json
import hashlib
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance

# Optional imports with graceful fallbacks
try:
    import torchvision.transforms as transforms
    from torchvision.transforms import functional as TF
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    warnings.warn("Torchvision not available - some vision challenges will use fallbacks")

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available - language challenges will use fallbacks")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not available - some vision effects will use PIL fallbacks")

class ChallengeType(Enum):
    """Types of challenges available."""
    # Vision challenges
    ADVERSARIAL_PATCH = "adversarial_patch"
    STYLE_TRANSFER = "style_transfer"
    COMPRESSION = "compression"
    NOISE_INJECTION = "noise_injection"
    
    # Language challenges
    PARAPHRASING = "paraphrasing"
    SUBSTITUTION = "substitution"
    PERTURBATION = "perturbation"
    SYNONYM_REPLACEMENT = "synonym_replacement"
    
    # Multimodal challenges
    CROSS_MODAL = "cross_modal"
    IMAGE_TEXT_MISMATCH = "image_text_mismatch"
    
    # Baseline challenges
    BASELINE_VISION = "baseline_vision"
    BASELINE_LANGUAGE = "baseline_language"

@dataclass
class ChallengeResult:
    """Result of challenge generation and verification."""
    challenge_input: Any
    expected_output: Any
    metadata: Dict[str, Any]
    challenge_type: ChallengeType
    difficulty: float
    seed: int
    generation_time: float
    
    # Verification results (set after verify_response)
    model_output: Optional[Any] = None
    passed: Optional[bool] = None
    distance: Optional[float] = None
    verification_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result_dict = asdict(self)
        result_dict['challenge_type'] = self.challenge_type.value
        return result_dict

@dataclass
class ChallengeConfig:
    """Configuration for challenge generation."""
    challenge_type: ChallengeType
    difficulty: float = 0.5  # 0.0 = easy, 1.0 = hard
    seed: Optional[int] = None
    model_type: str = "auto"  # "vision", "language", "multimodal", "auto"
    
    # Challenge-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Generation settings
    deterministic: bool = True
    save_metadata: bool = True
    verify_result: bool = True
    
    def __post_init__(self):
        if self.seed is None:
            self.seed = np.random.randint(0, 2**31 - 1)

class ChallengeGenerator(abc.ABC):
    """
    Base class for all challenge generators.
    
    Provides the core interface for generating and verifying challenges
    with configurable difficulty levels and deterministic behavior.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize challenge generator."""
        self.logger = logger or self._setup_logger()
        self.generation_history: List[ChallengeResult] = []
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for challenge generation."""
        logger = logging.getLogger(f"challenge_generator_{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abc.abstractmethod
    def generate_challenge(self, model: nn.Module, input_data: Any, 
                          config: ChallengeConfig) -> ChallengeResult:
        """
        Generate a challenge for the given model and input.
        
        Args:
            model: Target model to generate challenge for
            input_data: Input data to base challenge on
            config: Challenge configuration including difficulty
            
        Returns:
            ChallengeResult with challenge input, expected output, and metadata
        """
        pass
    
    @abc.abstractmethod
    def verify_response(self, model_output: Any, expected_output: Any,
                       metadata: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Verify model response against expected output.
        
        Args:
            model_output: Actual model output
            expected_output: Expected output from challenge
            metadata: Challenge metadata for verification
            
        Returns:
            Tuple of (passed: bool, distance: float)
        """
        pass
    
    def _set_seed(self, seed: int) -> None:
        """Set random seeds for deterministic generation."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _compute_distance(self, output1: torch.Tensor, output2: torch.Tensor,
                         metric: str = "cosine") -> float:
        """Compute distance between two outputs."""
        if output1.shape != output2.shape:
            # Flatten if shapes don't match
            output1 = output1.flatten()
            output2 = output2.flatten()
            
        if metric == "cosine":
            return 1 - F.cosine_similarity(output1, output2, dim=-1).mean().item()
        elif metric == "euclidean":
            return F.mse_loss(output1, output2).sqrt().item()
        elif metric == "l1":
            return F.l1_loss(output1, output2).item()
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
    
    def generate_and_verify(self, model: nn.Module, input_data: Any,
                           config: ChallengeConfig) -> ChallengeResult:
        """Generate challenge and verify model response."""
        start_time = time.time()
        
        # Generate challenge
        result = self.generate_challenge(model, input_data, config)
        
        if config.verify_result:
            # Run model on challenge input
            model.eval()
            with torch.no_grad():
                if isinstance(result.challenge_input, torch.Tensor):
                    model_output = model(result.challenge_input.unsqueeze(0) 
                                       if result.challenge_input.dim() < 4 else result.challenge_input)
                else:
                    model_output = model(result.challenge_input)
            
            # Verify response
            verify_start = time.time()
            passed, distance = self.verify_response(model_output, result.expected_output, result.metadata)
            verification_time = time.time() - verify_start
            
            # Update result
            result.model_output = model_output
            result.passed = passed
            result.distance = distance
            result.verification_time = verification_time
        
        # Store in history
        self.generation_history.append(result)
        
        distance_str = f"{result.distance:.4f}" if result.distance is not None else "N/A"
        self.logger.info(f"Generated {config.challenge_type.value} challenge "
                        f"(difficulty={config.difficulty:.2f}, "
                        f"distance={distance_str})")
        
        return result

class AdversarialChallengeGenerator(ChallengeGenerator):
    """
    Generator for adversarial patch challenges.
    
    Creates adversarial patches that can fool vision models while being
    imperceptible to humans. Difficulty controls patch size and optimization strength.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patch_sizes = {
            0.0: (8, 8),    # Easy: very small patch
            0.5: (16, 16),  # Medium: small patch
            1.0: (32, 32)   # Hard: larger patch
        }
    
    def generate_challenge(self, model: nn.Module, input_data: torch.Tensor,
                          config: ChallengeConfig) -> ChallengeResult:
        """Generate adversarial patch challenge."""
        start_time = time.time()
        self._set_seed(config.seed)
        
        # Determine patch size based on difficulty
        patch_h, patch_w = self._interpolate_patch_size(config.difficulty)
        
        # Get clean model output
        model.eval()
        with torch.no_grad():
            clean_output = model(input_data.unsqueeze(0) if input_data.dim() < 4 else input_data)
            clean_pred = clean_output.argmax(dim=-1)
        
        # Initialize adversarial patch
        patch = torch.rand(3, patch_h, patch_w, requires_grad=True)
        
        # Optimization parameters based on difficulty
        learning_rate = 0.01 + config.difficulty * 0.09  # 0.01 to 0.1
        num_iterations = int(10 + config.difficulty * 40)  # 10 to 50 iterations
        
        optimizer = torch.optim.Adam([patch], lr=learning_rate)
        
        # Generate adversarial patch
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # Apply patch to image
            patched_image = self._apply_patch(input_data, patch, config.parameters)
            
            # Compute loss
            output = model(patched_image.unsqueeze(0) if patched_image.dim() < 4 else patched_image)
            
            # Loss: maximize prediction change
            target_class = (clean_pred + 1) % output.shape[-1]  # Different class
            loss = -F.cross_entropy(output, target_class.unsqueeze(0) if target_class.dim() == 0 else target_class)
            
            loss.backward()
            optimizer.step()
            
            # Clamp patch values
            with torch.no_grad():
                patch.clamp_(0, 1)
        
        # Create final challenge input
        challenge_input = self._apply_patch(input_data, patch.detach(), config.parameters)
        
        metadata = {
            "patch_size": (patch_h, patch_w),
            "learning_rate": learning_rate,
            "iterations": num_iterations,
            "clean_prediction": clean_pred.item() if clean_pred.numel() == 1 else clean_pred.tolist(),
            "patch_location": config.parameters.get("patch_location", "random"),
            "seed": config.seed
        }
        
        generation_time = time.time() - start_time
        
        return ChallengeResult(
            challenge_input=challenge_input,
            expected_output=clean_output,
            metadata=metadata,
            challenge_type=config.challenge_type,
            difficulty=config.difficulty,
            seed=config.seed,
            generation_time=generation_time
        )
    
    def _interpolate_patch_size(self, difficulty: float) -> Tuple[int, int]:
        """Interpolate patch size based on difficulty."""
        if difficulty <= 0.5:
            # Interpolate between easy and medium
            t = difficulty * 2
            easy_size = self.patch_sizes[0.0]
            medium_size = self.patch_sizes[0.5]
            h = int(easy_size[0] + t * (medium_size[0] - easy_size[0]))
            w = int(easy_size[1] + t * (medium_size[1] - easy_size[1]))
        else:
            # Interpolate between medium and hard
            t = (difficulty - 0.5) * 2
            medium_size = self.patch_sizes[0.5]
            hard_size = self.patch_sizes[1.0]
            h = int(medium_size[0] + t * (hard_size[0] - medium_size[0]))
            w = int(medium_size[1] + t * (hard_size[1] - medium_size[1]))
        
        return h, w
    
    def _apply_patch(self, image: torch.Tensor, patch: torch.Tensor,
                     parameters: Dict[str, Any]) -> torch.Tensor:
        """Apply patch to image at specified or random location."""
        patched_image = image.clone()
        
        # Get patch location
        location = parameters.get("patch_location", "random")
        if location == "random":
            # Random location
            h_start = np.random.randint(0, max(1, image.shape[-2] - patch.shape[-2]))
            w_start = np.random.randint(0, max(1, image.shape[-1] - patch.shape[-1]))
        elif location == "center":
            # Center location
            h_start = (image.shape[-2] - patch.shape[-2]) // 2
            w_start = (image.shape[-1] - patch.shape[-1]) // 2
        else:
            # Specific coordinates
            h_start, w_start = location
        
        # Apply patch
        h_end = min(h_start + patch.shape[-2], image.shape[-2])
        w_end = min(w_start + patch.shape[-1], image.shape[-1])
        
        patched_image[..., h_start:h_end, w_start:w_end] = patch[..., :h_end-h_start, :w_end-w_start]
        
        return patched_image
    
    def verify_response(self, model_output: torch.Tensor, expected_output: torch.Tensor,
                       metadata: Dict[str, Any]) -> Tuple[bool, float]:
        """Verify adversarial challenge response."""
        # Compute prediction change
        clean_pred = expected_output.argmax(dim=-1)
        adv_pred = model_output.argmax(dim=-1)
        
        # Challenge passes if prediction changes
        passed = not torch.equal(clean_pred, adv_pred)
        
        # Distance based on output difference
        distance = self._compute_distance(model_output, expected_output, "cosine")
        
        return passed, distance

class StyleTransferChallengeGenerator(ChallengeGenerator):
    """
    Generator for style transfer challenges.
    
    Applies various style transformations to images while preserving semantic content.
    Difficulty controls the strength of style transfer effects.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.style_effects = [
            "blur", "sharpen", "emboss", "edge_enhance", 
            "color_shift", "brightness", "contrast", "saturation"
        ]
    
    def generate_challenge(self, model: nn.Module, input_data: torch.Tensor,
                          config: ChallengeConfig) -> ChallengeResult:
        """Generate style transfer challenge."""
        start_time = time.time()
        self._set_seed(config.seed)
        
        # Get clean model output
        model.eval()
        with torch.no_grad():
            clean_output = model(input_data.unsqueeze(0) if input_data.dim() < 4 else input_data)
        
        # Select style effect
        effect = config.parameters.get("style_effect", 
                                     np.random.choice(self.style_effects))
        
        # Apply style transfer based on difficulty
        styled_image = self._apply_style_effect(input_data, effect, config.difficulty)
        
        metadata = {
            "style_effect": effect,
            "difficulty": config.difficulty,
            "seed": config.seed,
            "original_shape": input_data.shape,
            "effect_parameters": self._get_effect_parameters(effect, config.difficulty)
        }
        
        generation_time = time.time() - start_time
        
        return ChallengeResult(
            challenge_input=styled_image,
            expected_output=clean_output,
            metadata=metadata,
            challenge_type=config.challenge_type,
            difficulty=config.difficulty,
            seed=config.seed,
            generation_time=generation_time
        )
    
    def _apply_style_effect(self, image: torch.Tensor, effect: str, 
                           difficulty: float) -> torch.Tensor:
        """Apply style effect with given difficulty."""
        # Convert to PIL for easier manipulation
        if image.dim() == 3:
            pil_image = TF.to_pil_image(image) if TORCHVISION_AVAILABLE else self._tensor_to_pil(image)
        else:
            pil_image = TF.to_pil_image(image[0]) if TORCHVISION_AVAILABLE else self._tensor_to_pil(image[0])
        
        # Apply effect based on type and difficulty
        if effect == "blur":
            radius = 0.5 + difficulty * 2.0  # 0.5 to 2.5
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        elif effect == "sharpen":
            factor = 1.0 + difficulty * 2.0  # 1.0 to 3.0
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(factor)
        
        elif effect == "brightness":
            factor = 0.5 + difficulty * 1.0  # 0.5 to 1.5
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(factor)
        
        elif effect == "contrast":
            factor = 0.5 + difficulty * 1.5  # 0.5 to 2.0
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(factor)
        
        elif effect == "saturation":
            factor = 0.2 + difficulty * 1.6  # 0.2 to 1.8
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(factor)
        
        elif effect == "color_shift":
            # Simple color channel shift
            img_array = np.array(pil_image)
            shift = int(difficulty * 50)  # 0 to 50
            img_array = np.roll(img_array, shift, axis=0)
            pil_image = Image.fromarray(img_array)
        
        else:
            # Default: slight blur
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Convert back to tensor
        if TORCHVISION_AVAILABLE:
            return TF.to_tensor(pil_image)
        else:
            return self._pil_to_tensor(pil_image)
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL image without torchvision."""
        # Clamp and scale to 0-255
        tensor = torch.clamp(tensor, 0, 1)
        tensor = (tensor * 255).byte()
        
        # Convert to numpy and PIL
        if tensor.shape[0] == 3:  # RGB
            img_array = tensor.permute(1, 2, 0).numpy()
            return Image.fromarray(img_array, mode='RGB')
        else:  # Grayscale
            img_array = tensor.squeeze().numpy()
            return Image.fromarray(img_array, mode='L')
    
    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor without torchvision."""
        img_array = np.array(pil_image)
        
        if len(img_array.shape) == 3:  # RGB
            tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        else:  # Grayscale
            tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        return tensor.float() / 255.0
    
    def _get_effect_parameters(self, effect: str, difficulty: float) -> Dict[str, Any]:
        """Get parameters used for the effect."""
        if effect == "blur":
            return {"radius": 0.5 + difficulty * 2.0}
        elif effect == "sharpen":
            return {"factor": 1.0 + difficulty * 2.0}
        elif effect == "brightness":
            return {"factor": 0.5 + difficulty * 1.0}
        elif effect == "contrast":
            return {"factor": 0.5 + difficulty * 1.5}
        elif effect == "saturation":
            return {"factor": 0.2 + difficulty * 1.6}
        elif effect == "color_shift":
            return {"shift": int(difficulty * 50)}
        else:
            return {}
    
    def verify_response(self, model_output: torch.Tensor, expected_output: torch.Tensor,
                       metadata: Dict[str, Any]) -> Tuple[bool, float]:
        """Verify style transfer challenge response."""
        # Compute output similarity
        distance = self._compute_distance(model_output, expected_output, "cosine")
        
        # Challenge passes if output is similar (robust to style changes)
        # Threshold based on difficulty - harder challenges allow more change
        threshold = 0.1 + metadata["difficulty"] * 0.3  # 0.1 to 0.4
        passed = distance < threshold
        
        return passed, distance

class CompressionChallengeGenerator(ChallengeGenerator):
    """
    Generator for compression challenges.
    
    Applies various compression techniques to test model robustness.
    Difficulty controls compression strength.
    """
    
    def generate_challenge(self, model: nn.Module, input_data: torch.Tensor,
                          config: ChallengeConfig) -> ChallengeResult:
        """Generate compression challenge."""
        start_time = time.time()
        self._set_seed(config.seed)
        
        # Get clean model output
        model.eval()
        with torch.no_grad():
            clean_output = model(input_data.unsqueeze(0) if input_data.dim() < 4 else input_data)
        
        # Apply compression
        compression_type = config.parameters.get("compression_type", "jpeg")
        compressed_image = self._apply_compression(input_data, compression_type, config.difficulty)
        
        metadata = {
            "compression_type": compression_type,
            "difficulty": config.difficulty,
            "seed": config.seed,
            "quality": self._get_compression_quality(config.difficulty),
            "original_shape": input_data.shape
        }
        
        generation_time = time.time() - start_time
        
        return ChallengeResult(
            challenge_input=compressed_image,
            expected_output=clean_output,
            metadata=metadata,
            challenge_type=config.challenge_type,
            difficulty=config.difficulty,
            seed=config.seed,
            generation_time=generation_time
        )
    
    def _apply_compression(self, image: torch.Tensor, compression_type: str,
                          difficulty: float) -> torch.Tensor:
        """Apply compression based on type and difficulty."""
        if compression_type == "jpeg":
            return self._apply_jpeg_compression(image, difficulty)
        elif compression_type == "quantization":
            return self._apply_quantization(image, difficulty)
        else:
            # Default: simple quantization
            return self._apply_quantization(image, difficulty)
    
    def _apply_jpeg_compression(self, image: torch.Tensor, difficulty: float) -> torch.Tensor:
        """Apply JPEG-like compression."""
        # Simulate JPEG compression by quantizing DCT coefficients
        # For simplicity, we'll use aggressive quantization
        
        # Convert to PIL and back to simulate compression
        if TORCHVISION_AVAILABLE:
            pil_image = TF.to_pil_image(image if image.dim() == 3 else image[0])
        else:
            pil_image = self._tensor_to_pil(image if image.dim() == 3 else image[0])
        
        # Quality based on difficulty (lower difficulty = higher quality)
        quality = max(10, int(100 - difficulty * 85))  # 100 to 15 quality
        
        # Save and reload to apply compression (simulate with quantization)
        # Since we can't easily save/load in memory, we'll use quantization
        quantization_levels = max(8, int(256 - difficulty * 240))  # 256 to 16 levels
        
        img_array = np.array(pil_image)
        quantized = np.round(img_array / (256 / quantization_levels)) * (256 / quantization_levels)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        
        compressed_pil = Image.fromarray(quantized)
        
        if TORCHVISION_AVAILABLE:
            return TF.to_tensor(compressed_pil)
        else:
            return self._pil_to_tensor(compressed_pil)
    
    def _apply_quantization(self, image: torch.Tensor, difficulty: float) -> torch.Tensor:
        """Apply simple quantization."""
        # Number of quantization levels (lower = more compression)
        levels = max(4, int(256 - difficulty * 240))  # 256 to 16 levels
        
        # Quantize
        quantized = torch.round(image * (levels - 1)) / (levels - 1)
        return torch.clamp(quantized, 0, 1)
    
    def _get_compression_quality(self, difficulty: float) -> int:
        """Get compression quality parameter."""
        return max(10, int(100 - difficulty * 85))
    
    def verify_response(self, model_output: torch.Tensor, expected_output: torch.Tensor,
                       metadata: Dict[str, Any]) -> Tuple[bool, float]:
        """Verify compression challenge response."""
        # Compute output similarity
        distance = self._compute_distance(model_output, expected_output, "cosine")
        
        # Challenge passes if output is similar (robust to compression)
        threshold = 0.05 + metadata["difficulty"] * 0.25  # 0.05 to 0.3
        passed = distance < threshold
        
        return passed, distance

class ParaphraseChallengeGenerator(ChallengeGenerator):
    """
    Generator for paraphrasing challenges.
    
    Creates paraphrased versions of text inputs while preserving semantic meaning.
    Difficulty controls the degree of paraphrasing.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.paraphrase_strategies = [
            "synonym_replacement", "sentence_reordering", "word_insertion",
            "word_deletion", "passive_voice", "question_conversion"
        ]
    
    def generate_challenge(self, model: nn.Module, input_data: Union[str, torch.Tensor],
                          config: ChallengeConfig) -> ChallengeResult:
        """Generate paraphrasing challenge."""
        start_time = time.time()
        self._set_seed(config.seed)
        
        # Handle input data
        if isinstance(input_data, str):
            original_text = input_data
            # For language models, we need to tokenize
            tokenized_input = self._tokenize_text(original_text, config.parameters)
        else:
            # Assume it's already tokenized
            tokenized_input = input_data
            original_text = config.parameters.get("original_text", "<unknown>")
        
        # Get clean model output
        model.eval()
        with torch.no_grad():
            if isinstance(tokenized_input, torch.Tensor):
                clean_output = model(tokenized_input.unsqueeze(0) if tokenized_input.dim() == 1 else tokenized_input)
            else:
                clean_output = model(tokenized_input)
        
        # Generate paraphrase
        strategy = config.parameters.get("strategy", 
                                       np.random.choice(self.paraphrase_strategies))
        
        paraphrased_text = self._apply_paraphrasing(original_text, strategy, config.difficulty)
        paraphrased_input = self._tokenize_text(paraphrased_text, config.parameters)
        
        metadata = {
            "original_text": original_text,
            "paraphrased_text": paraphrased_text,
            "strategy": strategy,
            "difficulty": config.difficulty,
            "seed": config.seed,
            "num_changes": self._count_changes(original_text, paraphrased_text)
        }
        
        generation_time = time.time() - start_time
        
        return ChallengeResult(
            challenge_input=paraphrased_input,
            expected_output=clean_output,
            metadata=metadata,
            challenge_type=config.challenge_type,
            difficulty=config.difficulty,
            seed=config.seed,
            generation_time=generation_time
        )
    
    def _tokenize_text(self, text: str, parameters: Dict[str, Any]) -> torch.Tensor:
        """Tokenize text for model input."""
        # Simple word-based tokenization as fallback
        max_length = parameters.get("max_length", 128)
        vocab_size = parameters.get("vocab_size", 1000)
        
        # Convert to simple token IDs
        words = text.lower().split()[:max_length]
        token_ids = [hash(word) % vocab_size for word in words]
        
        # Pad to max_length
        while len(token_ids) < max_length:
            token_ids.append(0)  # Padding token
        
        return torch.tensor(token_ids[:max_length], dtype=torch.long)
    
    def _apply_paraphrasing(self, text: str, strategy: str, difficulty: float) -> str:
        """Apply paraphrasing strategy with given difficulty."""
        words = text.split()
        
        if strategy == "synonym_replacement":
            return self._synonym_replacement(words, difficulty)
        elif strategy == "word_insertion":
            return self._word_insertion(words, difficulty)
        elif strategy == "word_deletion":
            return self._word_deletion(words, difficulty)
        elif strategy == "sentence_reordering":
            return self._sentence_reordering(text, difficulty)
        else:
            # Default: synonym replacement
            return self._synonym_replacement(words, difficulty)
    
    def _synonym_replacement(self, words: List[str], difficulty: float) -> str:
        """Replace words with simple synonyms."""
        synonyms = {
            "good": ["great", "excellent", "fine", "nice"],
            "bad": ["poor", "terrible", "awful", "horrible"],
            "big": ["large", "huge", "enormous", "massive"],
            "small": ["tiny", "little", "mini", "compact"],
            "fast": ["quick", "rapid", "swift", "speedy"],
            "slow": ["sluggish", "gradual", "unhurried", "leisurely"]
        }
        
        # Number of replacements based on difficulty
        num_replacements = int(len(words) * difficulty * 0.3)  # Up to 30% of words
        
        result = words.copy()
        indices = np.random.choice(len(words), min(num_replacements, len(words)), replace=False)
        
        for idx in indices:
            word = words[idx].lower()
            if word in synonyms:
                result[idx] = np.random.choice(synonyms[word])
        
        return " ".join(result)
    
    def _word_insertion(self, words: List[str], difficulty: float) -> str:
        """Insert additional words."""
        insertion_words = ["very", "quite", "really", "extremely", "somewhat", "rather"]
        
        num_insertions = int(len(words) * difficulty * 0.2)  # Up to 20% insertions
        result = words.copy()
        
        for _ in range(num_insertions):
            if result:
                insert_pos = np.random.randint(0, len(result))
                insert_word = np.random.choice(insertion_words)
                result.insert(insert_pos, insert_word)
        
        return " ".join(result)
    
    def _word_deletion(self, words: List[str], difficulty: float) -> str:
        """Delete some words."""
        # Delete up to 20% of words based on difficulty
        num_deletions = int(len(words) * difficulty * 0.2)
        
        if len(words) <= num_deletions:
            return " ".join(words[:max(1, len(words) - 1)])  # Keep at least one word
        
        indices_to_keep = np.random.choice(len(words), len(words) - num_deletions, replace=False)
        indices_to_keep.sort()
        
        return " ".join([words[i] for i in indices_to_keep])
    
    def _sentence_reordering(self, text: str, difficulty: float) -> str:
        """Reorder sentences."""
        sentences = text.split('. ')
        if len(sentences) < 2:
            return text
        
        # Only reorder if difficulty is high enough
        if difficulty > 0.5 and len(sentences) > 1:
            np.random.shuffle(sentences)
        
        return '. '.join(sentences)
    
    def _count_changes(self, original: str, modified: str) -> int:
        """Count number of word changes between texts."""
        orig_words = set(original.lower().split())
        mod_words = set(modified.lower().split())
        return len(orig_words.symmetric_difference(mod_words))
    
    def verify_response(self, model_output: torch.Tensor, expected_output: torch.Tensor,
                       metadata: Dict[str, Any]) -> Tuple[bool, float]:
        """Verify paraphrasing challenge response."""
        # Compute output similarity
        distance = self._compute_distance(model_output, expected_output, "cosine")
        
        # Challenge passes if semantic meaning is preserved
        # More lenient threshold for higher difficulty paraphrases
        base_threshold = 0.2
        difficulty_adjustment = metadata["difficulty"] * 0.3
        threshold = base_threshold + difficulty_adjustment
        
        passed = distance < threshold
        
        return passed, distance

class CrossModalChallengeGenerator(ChallengeGenerator):
    """
    Generator for cross-modal challenges.
    
    Creates challenges that test consistency between different modalities
    (e.g., image-text alignment, vision-language understanding).
    """
    
    def generate_challenge(self, model: nn.Module, input_data: Tuple[torch.Tensor, str],
                          config: ChallengeConfig) -> ChallengeResult:
        """Generate cross-modal challenge."""
        start_time = time.time()
        self._set_seed(config.seed)
        
        if not isinstance(input_data, tuple) or len(input_data) != 2:
            raise ValueError("CrossModalChallengeGenerator requires input_data as (image, text) tuple")
        
        image, text = input_data
        
        # Ensure image has proper shape
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image must be a torch.Tensor")
        
        # Get clean model output for both modalities
        model.eval()
        with torch.no_grad():
            # Assume model can handle both image and text
            if hasattr(model, 'encode_image') and hasattr(model, 'encode_text'):
                # CLIP-like model
                image_features = model.encode_image(image.unsqueeze(0) if image.dim() < 4 else image)
                text_features = model.encode_text(text)
                clean_output = (image_features, text_features)
            else:
                # Fallback: use full model with tuple input
                clean_output = model((image, text))
        
        # Create mismatched challenge based on difficulty
        challenge_type = config.parameters.get("mismatch_type", "text_substitution")
        
        if challenge_type == "text_substitution":
            # Replace text with semantically different text
            modified_text = self._substitute_text(text, config.difficulty)
            challenge_input = (image, modified_text)
        
        elif challenge_type == "image_modification":
            # Modify image while keeping text
            modified_image = self._modify_image(image, config.difficulty)
            challenge_input = (modified_image, text)
        
        else:
            # Both modifications
            modified_image = self._modify_image(image, config.difficulty * 0.5)
            modified_text = self._substitute_text(text, config.difficulty * 0.5)
            challenge_input = (modified_image, modified_text)
        
        metadata = {
            "mismatch_type": challenge_type,
            "original_text": text,
            "modified_text": challenge_input[1] if isinstance(challenge_input[1], str) else text,
            "difficulty": config.difficulty,
            "seed": config.seed,
            "semantic_distance": self._compute_semantic_distance(text, challenge_input[1] if isinstance(challenge_input[1], str) else text)
        }
        
        generation_time = time.time() - start_time
        
        return ChallengeResult(
            challenge_input=challenge_input,
            expected_output=clean_output,
            metadata=metadata,
            challenge_type=config.challenge_type,
            difficulty=config.difficulty,
            seed=config.seed,
            generation_time=generation_time
        )
    
    def _substitute_text(self, text: str, difficulty: float) -> str:
        """Substitute text with semantically different content."""
        # Simple substitution based on difficulty
        substitutions = {
            0.2: {"cat": "dog", "car": "bike", "house": "apartment"},
            0.5: {"cat": "elephant", "car": "airplane", "house": "castle"},
            0.8: {"cat": "computer", "car": "mountain", "house": "ocean"}
        }
        
        # Find appropriate substitution level
        if difficulty <= 0.3:
            sub_dict = substitutions[0.2]
        elif difficulty <= 0.6:
            sub_dict = substitutions[0.5]
        else:
            sub_dict = substitutions[0.8]
        
        words = text.lower().split()
        for i, word in enumerate(words):
            if word in sub_dict:
                words[i] = sub_dict[word]
                break  # Only substitute one word for now
        
        return " ".join(words)
    
    def _modify_image(self, image: torch.Tensor, difficulty: float) -> torch.Tensor:
        """Modify image to create cross-modal mismatch."""
        # Apply color transformation based on difficulty
        if difficulty < 0.3:
            # Slight color shift
            modified = image + torch.randn_like(image) * 0.05
        elif difficulty < 0.6:
            # Significant color change
            # Swap color channels
            modified = image[[2, 1, 0], :, :] if image.dim() == 3 else image
        else:
            # Major transformation - invert colors
            modified = 1.0 - image
        
        return torch.clamp(modified, 0, 1)
    
    def _compute_semantic_distance(self, text1: str, text2: str) -> float:
        """Compute semantic distance between texts."""
        # Simple word overlap metric
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return 1.0 - (intersection / union if union > 0 else 0.0)
    
    def verify_response(self, model_output: Any, expected_output: Any,
                       metadata: Dict[str, Any]) -> Tuple[bool, float]:
        """Verify cross-modal challenge response."""
        if isinstance(model_output, tuple) and isinstance(expected_output, tuple):
            # Compare both modalities
            img_distance = self._compute_distance(model_output[0], expected_output[0], "cosine")
            
            if len(model_output) > 1 and len(expected_output) > 1:
                text_distance = self._compute_distance(model_output[1], expected_output[1], "cosine")
                distance = (img_distance + text_distance) / 2
            else:
                distance = img_distance
        else:
            # Single output comparison
            distance = self._compute_distance(model_output, expected_output, "cosine")
        
        # Challenge passes if model detects the mismatch (high distance)
        threshold = 0.3 + metadata["difficulty"] * 0.4  # 0.3 to 0.7
        passed = distance > threshold  # Note: reversed logic for mismatch detection
        
        return passed, distance

class SubstitutionChallengeGenerator(ChallengeGenerator):
    """
    Generator for word/token substitution challenges.
    
    Replaces specific words or tokens in text with alternatives
    to test model robustness to lexical variations.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.substitution_types = ["synonym", "antonym", "random", "similar_form"]
    
    def generate_challenge(self, model: nn.Module, input_data: Union[str, torch.Tensor],
                          config: ChallengeConfig) -> ChallengeResult:
        """Generate substitution challenge."""
        start_time = time.time()
        self._set_seed(config.seed)
        
        # Handle input data
        if isinstance(input_data, str):
            original_text = input_data
            tokenized_input = self._tokenize_text(original_text, config.parameters)
        else:
            tokenized_input = input_data
            original_text = config.parameters.get("original_text", "<unknown>")
        
        # Get clean model output
        model.eval()
        with torch.no_grad():
            if isinstance(tokenized_input, torch.Tensor):
                clean_output = model(tokenized_input.unsqueeze(0) if tokenized_input.dim() == 1 else tokenized_input)
            else:
                clean_output = model(tokenized_input)
        
        # Apply substitution
        substitution_type = config.parameters.get("substitution_type", 
                                                np.random.choice(self.substitution_types))
        
        substituted_text = self._apply_substitution(original_text, substitution_type, config.difficulty)
        substituted_input = self._tokenize_text(substituted_text, config.parameters)
        
        metadata = {
            "original_text": original_text,
            "substituted_text": substituted_text,
            "substitution_type": substitution_type,
            "difficulty": config.difficulty,
            "seed": config.seed,
            "num_substitutions": self._count_substitutions(original_text, substituted_text)
        }
        
        generation_time = time.time() - start_time
        
        return ChallengeResult(
            challenge_input=substituted_input,
            expected_output=clean_output,
            metadata=metadata,
            challenge_type=config.challenge_type,
            difficulty=config.difficulty,
            seed=config.seed,
            generation_time=generation_time
        )
    
    def _tokenize_text(self, text: str, parameters: Dict[str, Any]) -> torch.Tensor:
        """Tokenize text for model input."""
        max_length = parameters.get("max_length", 128)
        vocab_size = parameters.get("vocab_size", 1000)
        
        words = text.lower().split()[:max_length]
        token_ids = [hash(word) % vocab_size for word in words]
        
        while len(token_ids) < max_length:
            token_ids.append(0)
        
        return torch.tensor(token_ids[:max_length], dtype=torch.long)
    
    def _apply_substitution(self, text: str, substitution_type: str, difficulty: float) -> str:
        """Apply word substitution based on type and difficulty."""
        words = text.split()
        
        # Number of substitutions based on difficulty
        num_subs = max(1, int(len(words) * difficulty * 0.5))  # Up to 50% of words
        
        if len(words) == 0:
            return text
        
        indices = np.random.choice(len(words), min(num_subs, len(words)), replace=False)
        result = words.copy()
        
        for idx in indices:
            original_word = words[idx].lower()
            
            if substitution_type == "synonym":
                result[idx] = self._get_synonym(original_word)
            elif substitution_type == "antonym":
                result[idx] = self._get_antonym(original_word)
            elif substitution_type == "similar_form":
                result[idx] = self._get_similar_form(original_word)
            else:  # random
                result[idx] = self._get_random_word()
        
        return " ".join(result)
    
    def _get_synonym(self, word: str) -> str:
        """Get synonym for word."""
        synonyms = {
            "good": "excellent", "bad": "terrible", "big": "large", "small": "tiny",
            "fast": "quick", "slow": "sluggish", "happy": "joyful", "sad": "melancholy",
            "beautiful": "gorgeous", "ugly": "hideous", "smart": "intelligent", "stupid": "foolish"
        }
        return synonyms.get(word, word + "_syn")
    
    def _get_antonym(self, word: str) -> str:
        """Get antonym for word."""
        antonyms = {
            "good": "bad", "big": "small", "fast": "slow", "happy": "sad",
            "beautiful": "ugly", "smart": "stupid", "hot": "cold", "light": "dark"
        }
        return antonyms.get(word, "not_" + word)
    
    def _get_similar_form(self, word: str) -> str:
        """Get word with similar form but different meaning."""
        # Simple morphological variations
        if word.endswith("ing"):
            return word[:-3] + "ed"
        elif word.endswith("ed"):
            return word[:-2] + "ing"
        elif word.endswith("s"):
            return word[:-1]
        else:
            return word + "s"
    
    def _get_random_word(self) -> str:
        """Get random word for substitution."""
        random_words = ["random", "substitute", "replacement", "alternative", "different", "other"]
        return np.random.choice(random_words)
    
    def _count_substitutions(self, original: str, modified: str) -> int:
        """Count number of substitutions made."""
        orig_words = original.split()
        mod_words = modified.split()
        
        if len(orig_words) != len(mod_words):
            return abs(len(orig_words) - len(mod_words))
        
        return sum(1 for o, m in zip(orig_words, mod_words) if o != m)
    
    def verify_response(self, model_output: torch.Tensor, expected_output: torch.Tensor,
                       metadata: Dict[str, Any]) -> Tuple[bool, float]:
        """Verify substitution challenge response."""
        distance = self._compute_distance(model_output, expected_output, "cosine")
        
        # Challenge passes if output is similar despite substitutions
        threshold = 0.15 + metadata["difficulty"] * 0.25  # 0.15 to 0.4
        passed = distance < threshold
        
        return passed, distance

class PerturbationChallengeGenerator(ChallengeGenerator):
    """
    Generator for perturbation challenges.
    
    Applies small perturbations to inputs to test model robustness
    to noise and minor variations.
    """
    
    def generate_challenge(self, model: nn.Module, input_data: torch.Tensor,
                          config: ChallengeConfig) -> ChallengeResult:
        """Generate perturbation challenge."""
        start_time = time.time()
        self._set_seed(config.seed)
        
        # Get clean model output
        model.eval()
        with torch.no_grad():
            clean_output = model(input_data.unsqueeze(0) if input_data.dim() < 4 else input_data)
        
        # Apply perturbation based on input type
        perturbation_type = config.parameters.get("perturbation_type", "gaussian_noise")
        
        if perturbation_type == "gaussian_noise":
            perturbed_input = self._add_gaussian_noise(input_data, config.difficulty)
        elif perturbation_type == "uniform_noise":
            perturbed_input = self._add_uniform_noise(input_data, config.difficulty)
        elif perturbation_type == "salt_pepper":
            perturbed_input = self._add_salt_pepper_noise(input_data, config.difficulty)
        else:
            # Default: gaussian noise
            perturbed_input = self._add_gaussian_noise(input_data, config.difficulty)
        
        metadata = {
            "perturbation_type": perturbation_type,
            "difficulty": config.difficulty,
            "seed": config.seed,
            "noise_strength": self._get_noise_strength(config.difficulty),
            "original_shape": input_data.shape
        }
        
        generation_time = time.time() - start_time
        
        return ChallengeResult(
            challenge_input=perturbed_input,
            expected_output=clean_output,
            metadata=metadata,
            challenge_type=config.challenge_type,
            difficulty=config.difficulty,
            seed=config.seed,
            generation_time=generation_time
        )
    
    def _add_gaussian_noise(self, input_data: torch.Tensor, difficulty: float) -> torch.Tensor:
        """Add Gaussian noise to input."""
        noise_std = 0.01 + difficulty * 0.1  # 0.01 to 0.11
        noise = torch.randn_like(input_data) * noise_std
        return torch.clamp(input_data + noise, 0, 1)
    
    def _add_uniform_noise(self, input_data: torch.Tensor, difficulty: float) -> torch.Tensor:
        """Add uniform noise to input."""
        noise_magnitude = 0.01 + difficulty * 0.1  # 0.01 to 0.11
        noise = (torch.rand_like(input_data) - 0.5) * 2 * noise_magnitude
        return torch.clamp(input_data + noise, 0, 1)
    
    def _add_salt_pepper_noise(self, input_data: torch.Tensor, difficulty: float) -> torch.Tensor:
        """Add salt and pepper noise to input."""
        noise_prob = 0.01 + difficulty * 0.1  # 0.01 to 0.11 probability
        
        result = input_data.clone()
        
        # Salt noise (white pixels)
        salt_mask = torch.rand_like(input_data) < noise_prob / 2
        result[salt_mask] = 1.0
        
        # Pepper noise (black pixels)
        pepper_mask = torch.rand_like(input_data) < noise_prob / 2
        result[pepper_mask] = 0.0
        
        return result
    
    def _get_noise_strength(self, difficulty: float) -> float:
        """Get noise strength parameter."""
        return 0.01 + difficulty * 0.1
    
    def verify_response(self, model_output: torch.Tensor, expected_output: torch.Tensor,
                       metadata: Dict[str, Any]) -> Tuple[bool, float]:
        """Verify perturbation challenge response."""
        distance = self._compute_distance(model_output, expected_output, "cosine")
        
        # Challenge passes if output is robust to perturbations
        threshold = 0.05 + metadata["difficulty"] * 0.2  # 0.05 to 0.25
        passed = distance < threshold
        
        return passed, distance

class BaselineChallengeGenerator(ChallengeGenerator):
    """
    Generator for baseline challenges for testing.
    
    Creates simple, predictable challenges to verify the system works correctly.
    """
    
    def generate_challenge(self, model: nn.Module, input_data: Any,
                          config: ChallengeConfig) -> ChallengeResult:
        """Generate baseline challenge."""
        start_time = time.time()
        self._set_seed(config.seed)
        
        # Get clean model output
        model.eval()
        with torch.no_grad():
            clean_output = model(input_data.unsqueeze(0) if hasattr(input_data, 'dim') and input_data.dim() < 4 else input_data)
        
        # Apply minimal transformation based on type
        if config.challenge_type == ChallengeType.BASELINE_VISION:
            # Add small amount of noise
            noise_std = 0.01 + config.difficulty * 0.05  # 0.01 to 0.06
            challenge_input = input_data + torch.randn_like(input_data) * noise_std
            challenge_input = torch.clamp(challenge_input, 0, 1)
        else:
            # For language or other types, return input unchanged
            challenge_input = input_data
        
        metadata = {
            "transformation": "minimal_noise" if config.challenge_type == ChallengeType.BASELINE_VISION else "none",
            "difficulty": config.difficulty,
            "seed": config.seed,
            "noise_std": noise_std if config.challenge_type == ChallengeType.BASELINE_VISION else 0.0
        }
        
        generation_time = time.time() - start_time
        
        return ChallengeResult(
            challenge_input=challenge_input,
            expected_output=clean_output,
            metadata=metadata,
            challenge_type=config.challenge_type,
            difficulty=config.difficulty,
            seed=config.seed,
            generation_time=generation_time
        )
    
    def verify_response(self, model_output: torch.Tensor, expected_output: torch.Tensor,
                       metadata: Dict[str, Any]) -> Tuple[bool, float]:
        """Verify baseline challenge response."""
        # Compute output similarity
        distance = self._compute_distance(model_output, expected_output, "cosine")
        
        # Baseline challenges should have very similar outputs
        threshold = 0.05 + metadata["difficulty"] * 0.1  # 0.05 to 0.15
        passed = distance < threshold
        
        return passed, distance

# Factory function for creating challenge generators
def create_challenge_generator(challenge_type: ChallengeType, **kwargs) -> ChallengeGenerator:
    """
    Factory function to create appropriate challenge generator.
    
    Args:
        challenge_type: Type of challenge to generate
        **kwargs: Additional arguments for generator
        
    Returns:
        Appropriate ChallengeGenerator instance
    """
    generators = {
        # Vision challenges
        ChallengeType.ADVERSARIAL_PATCH: AdversarialChallengeGenerator,
        ChallengeType.STYLE_TRANSFER: StyleTransferChallengeGenerator,
        ChallengeType.COMPRESSION: CompressionChallengeGenerator,
        ChallengeType.NOISE_INJECTION: PerturbationChallengeGenerator,
        
        # Language challenges
        ChallengeType.PARAPHRASING: ParaphraseChallengeGenerator,
        ChallengeType.SUBSTITUTION: SubstitutionChallengeGenerator,
        ChallengeType.PERTURBATION: PerturbationChallengeGenerator,
        ChallengeType.SYNONYM_REPLACEMENT: SubstitutionChallengeGenerator,
        
        # Multimodal challenges
        ChallengeType.CROSS_MODAL: CrossModalChallengeGenerator,
        ChallengeType.IMAGE_TEXT_MISMATCH: CrossModalChallengeGenerator,
        
        # Baseline challenges
        ChallengeType.BASELINE_VISION: BaselineChallengeGenerator,
        ChallengeType.BASELINE_LANGUAGE: BaselineChallengeGenerator,
    }
    
    if challenge_type not in generators:
        raise ValueError(f"Unknown challenge type: {challenge_type}")
    
    return generators[challenge_type](**kwargs)

# Convenience functions for common use cases
def generate_vision_challenge(model: nn.Module, image: torch.Tensor,
                            challenge_type: str = "adversarial_patch",
                            difficulty: float = 0.5, **kwargs) -> ChallengeResult:
    """Generate a vision challenge with default settings."""
    config = ChallengeConfig(
        challenge_type=ChallengeType(challenge_type),
        difficulty=difficulty,
        parameters=kwargs
    )
    
    generator = create_challenge_generator(config.challenge_type)
    return generator.generate_and_verify(model, image, config)

def generate_language_challenge(model: nn.Module, text: str,
                              challenge_type: str = "paraphrasing",
                              difficulty: float = 0.5, **kwargs) -> ChallengeResult:
    """Generate a language challenge with default settings."""
    config = ChallengeConfig(
        challenge_type=ChallengeType(challenge_type),
        difficulty=difficulty,
        parameters=kwargs
    )
    
    generator = create_challenge_generator(config.challenge_type)
    return generator.generate_and_verify(model, text, config)

def generate_baseline_challenge(model: nn.Module, input_data: Any,
                              model_type: str = "vision",
                              difficulty: float = 0.1) -> ChallengeResult:
    """Generate a baseline challenge for testing."""
    challenge_type = (ChallengeType.BASELINE_VISION if model_type == "vision" 
                     else ChallengeType.BASELINE_LANGUAGE)
    
    config = ChallengeConfig(
        challenge_type=challenge_type,
        difficulty=difficulty
    )
    
    generator = create_challenge_generator(challenge_type)
    return generator.generate_and_verify(model, input_data, config)

def generate_multimodal_challenge(model: nn.Module, input_data: Tuple[torch.Tensor, str],
                                 challenge_type: str = "cross_modal",
                                 difficulty: float = 0.5, **kwargs) -> ChallengeResult:
    """Generate a multimodal challenge with default settings."""
    config = ChallengeConfig(
        challenge_type=ChallengeType(challenge_type),
        difficulty=difficulty,
        parameters=kwargs
    )
    
    generator = create_challenge_generator(config.challenge_type)
    return generator.generate_and_verify(model, input_data, config)

def generate_adversarial_challenge(model: nn.Module, image: torch.Tensor,
                                 difficulty: float = 0.5, **kwargs) -> ChallengeResult:
    """Generate an adversarial patch challenge."""
    return generate_vision_challenge(model, image, "adversarial_patch", difficulty, **kwargs)

def generate_style_challenge(model: nn.Module, image: torch.Tensor,
                           difficulty: float = 0.5, **kwargs) -> ChallengeResult:
    """Generate a style transfer challenge."""
    return generate_vision_challenge(model, image, "style_transfer", difficulty, **kwargs)

def generate_compression_challenge(model: nn.Module, image: torch.Tensor,
                                 difficulty: float = 0.5, **kwargs) -> ChallengeResult:
    """Generate a compression challenge."""
    return generate_vision_challenge(model, image, "compression", difficulty, **kwargs)

def generate_substitution_challenge(model: nn.Module, text: str,
                                  difficulty: float = 0.5, **kwargs) -> ChallengeResult:
    """Generate a word substitution challenge."""
    return generate_language_challenge(model, text, "substitution", difficulty, **kwargs)

def generate_perturbation_challenge(model: nn.Module, input_data: torch.Tensor,
                                  difficulty: float = 0.5, **kwargs) -> ChallengeResult:
    """Generate a perturbation challenge."""
    config = ChallengeConfig(
        challenge_type=ChallengeType.PERTURBATION,
        difficulty=difficulty,
        parameters=kwargs
    )
    
    generator = create_challenge_generator(config.challenge_type)
    return generator.generate_and_verify(model, input_data, config)

# Batch challenge generation
def generate_challenge_batch(model: nn.Module, input_data: Any,
                           challenge_types: List[str], difficulty: float = 0.5,
                           seed_offset: int = 0, **kwargs) -> List[ChallengeResult]:
    """
    Generate a batch of challenges of different types.
    
    Args:
        model: Model to generate challenges for
        input_data: Input data for challenges
        challenge_types: List of challenge type names
        difficulty: Difficulty level for all challenges
        seed_offset: Offset for deterministic seed generation
        **kwargs: Additional parameters for challenge generation
        
    Returns:
        List of ChallengeResult objects
    """
    results = []
    
    for i, challenge_type in enumerate(challenge_types):
        try:
            config = ChallengeConfig(
                challenge_type=ChallengeType(challenge_type),
                difficulty=difficulty,
                seed=42 + seed_offset + i,  # Deterministic but different seeds
                parameters=kwargs
            )
            
            generator = create_challenge_generator(config.challenge_type)
            result = generator.generate_and_verify(model, input_data, config)
            results.append(result)
            
        except Exception as e:
            # Log error but continue with other challenges
            print(f"Warning: Failed to generate {challenge_type} challenge: {e}")
            continue
    
    return results

def get_available_challenge_types() -> Dict[str, List[str]]:
    """Get available challenge types organized by modality."""
    return {
        "vision": [
            "adversarial_patch", "style_transfer", "compression", 
            "noise_injection", "baseline_vision"
        ],
        "language": [
            "paraphrasing", "substitution", "perturbation",
            "synonym_replacement", "baseline_language"
        ],
        "multimodal": [
            "cross_modal", "image_text_mismatch"
        ]
    }