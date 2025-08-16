"""
Comprehensive unit tests for equivalence transformations.

Tests:
- Transform acceptance rates
- FP32/FP16/INT8 equivalence
- Vision transforms (rotation, flip, crop)
- Language model equivalences
- Wrapper detection
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Mock equivalence transform functions for testing
class EquivalenceTransforms:
    """Mock equivalence transforms for testing."""
    
    @staticmethod
    def rotate90(x: torch.Tensor) -> torch.Tensor:
        """Rotate image 90 degrees clockwise."""
        if len(x.shape) == 4:  # Batch dimension
            return torch.rot90(x, k=1, dims=[2, 3])
        elif len(x.shape) == 3:  # Single image
            return torch.rot90(x, k=1, dims=[1, 2])
        return x
    
    @staticmethod
    def flip_horizontal(x: torch.Tensor) -> torch.Tensor:
        """Flip image horizontally."""
        if len(x.shape) == 4:
            return torch.flip(x, dims=[3])
        elif len(x.shape) == 3:
            return torch.flip(x, dims=[2])
        return x
    
    @staticmethod
    def flip_vertical(x: torch.Tensor) -> torch.Tensor:
        """Flip image vertically."""
        if len(x.shape) == 4:
            return torch.flip(x, dims=[2])
        elif len(x.shape) == 3:
            return torch.flip(x, dims=[1])
        return x
    
    @staticmethod
    def crop_center(x: torch.Tensor, crop_fraction: float = 0.9) -> torch.Tensor:
        """Crop center portion of image."""
        if len(x.shape) == 4:
            b, c, h, w = x.shape
            new_h = int(h * crop_fraction)
            new_w = int(w * crop_fraction)
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            return x[:, :, start_h:start_h+new_h, start_w:start_w+new_w]
        return x
    
    @staticmethod
    def add_noise(x: torch.Tensor, std: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * std
        return x + noise


class TestVisionTransforms:
    """Test vision-specific equivalence transforms."""
    
    def test_rotation_invariance(self):
        """Test that rotation preserves essential features."""
        # Create a test image with clear structure
        image = torch.zeros(1, 3, 32, 32)
        # Add a diagonal line
        for i in range(32):
            if i < 30:
                image[0, :, i, i] = 1.0
        
        # Apply rotation
        rotated = EquivalenceTransforms.rotate90(image)
        
        # Check dimensions preserved
        assert rotated.shape == image.shape
        
        # Check that content is rotated (not just copied)
        assert not torch.allclose(rotated, image)
        
        # Apply 4 rotations should return to original
        result = image
        for _ in range(4):
            result = EquivalenceTransforms.rotate90(result)
        assert torch.allclose(result, image, atol=1e-6)
    
    def test_flip_invariance(self):
        """Test that flips preserve essential features."""
        # Create asymmetric test image
        image = torch.zeros(1, 3, 32, 32)
        image[0, :, :, :16] = 1.0  # Left half white
        
        # Test horizontal flip
        h_flipped = EquivalenceTransforms.flip_horizontal(image)
        assert h_flipped.shape == image.shape
        assert not torch.allclose(h_flipped, image)
        
        # Double flip should return original
        h_double = EquivalenceTransforms.flip_horizontal(h_flipped)
        assert torch.allclose(h_double, image)
        
        # Test vertical flip
        v_flipped = EquivalenceTransforms.flip_vertical(image)
        assert v_flipped.shape == image.shape
        
        # Double flip should return original
        v_double = EquivalenceTransforms.flip_vertical(v_flipped)
        assert torch.allclose(v_double, image)
    
    def test_crop_preserves_center(self):
        """Test that center crop preserves central features."""
        # Create image with center pattern
        image = torch.zeros(1, 3, 32, 32)
        image[0, :, 14:18, 14:18] = 1.0  # Center square
        
        # Apply center crop
        cropped = EquivalenceTransforms.crop_center(image, 0.5)
        
        # Should be smaller
        assert cropped.shape[2] == 16
        assert cropped.shape[3] == 16
        
        # Center should still have the pattern
        center_val = cropped[0, 0, 7:9, 7:9].mean()
        assert center_val > 0.5  # Should still be bright
    
    def test_noise_addition(self):
        """Test controlled noise addition."""
        torch.manual_seed(42)
        image = torch.ones(1, 3, 32, 32) * 0.5
        
        # Add small noise
        noisy = EquivalenceTransforms.add_noise(image, std=0.01)
        
        # Shape preserved
        assert noisy.shape == image.shape
        
        # Should be different but close
        assert not torch.allclose(noisy, image)
        assert torch.allclose(noisy, image, atol=0.1)
        
        # Check noise statistics
        noise = noisy - image
        assert abs(noise.mean()) < 0.01
        assert abs(noise.std() - 0.01) < 0.005


class TestQuantizationEquivalence:
    """Test quantization-based equivalences."""
    
    def test_fp32_to_fp16_conversion(self):
        """Test FP32 to FP16 conversion preserves values."""
        # Create test tensor with various values
        fp32_tensor = torch.tensor([
            1.0, 0.5, 0.25, 0.125,
            1e-4, 1e-5, 1e-6,
            -1.0, -0.5, -0.25
        ], dtype=torch.float32)
        
        # Convert to FP16 and back
        fp16_tensor = fp32_tensor.to(torch.float16)
        back_to_fp32 = fp16_tensor.to(torch.float32)
        
        # Should be close but not identical
        assert torch.allclose(back_to_fp32, fp32_tensor, atol=1e-3)
        
        # Large values should be preserved better
        large_vals = fp32_tensor[torch.abs(fp32_tensor) > 0.1]
        large_fp16 = large_vals.to(torch.float16).to(torch.float32)
        assert torch.allclose(large_fp16, large_vals, rtol=1e-3)
    
    def test_int8_quantization(self):
        """Test INT8 quantization with scale and zero point."""
        # Create test tensor
        fp32_tensor = torch.randn(100) * 2.0
        
        # Compute scale and zero point
        min_val = fp32_tensor.min()
        max_val = fp32_tensor.max()
        scale = (max_val - min_val) / 255
        zero_point = -min_val / scale
        
        # Quantize
        int8_tensor = ((fp32_tensor - min_val) / scale).round().clamp(0, 255).to(torch.uint8)
        
        # Dequantize
        dequantized = int8_tensor.to(torch.float32) * scale + min_val
        
        # Should be close
        assert torch.allclose(dequantized, fp32_tensor, atol=scale)
        
        # Check range preservation
        assert dequantized.min() >= min_val - scale
        assert dequantized.max() <= max_val + scale
    
    def test_mixed_precision_model(self):
        """Test model behavior under mixed precision."""
        torch.manual_seed(42)
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        # Test input
        x = torch.randn(5, 10)
        
        # FP32 output
        with torch.no_grad():
            fp32_out = model(x)
        
        # Convert model to FP16
        model_fp16 = model.half()
        x_fp16 = x.half()
        
        with torch.no_grad():
            fp16_out = model_fp16(x_fp16).float()
        
        # Should be close but not identical
        assert torch.allclose(fp32_out, fp16_out, rtol=1e-2, atol=1e-3)
        
        # Relative error should be small
        rel_error = torch.abs(fp32_out - fp16_out) / (torch.abs(fp32_out) + 1e-8)
        assert rel_error.mean() < 0.01


class TestLanguageModelEquivalence:
    """Test language model specific equivalences."""
    
    def test_tokenization_invariance(self):
        """Test that different tokenizations can be equivalent."""
        # Simulate different tokenization of same text
        text = "Hello world"
        
        # Tokenization 1: character-level
        tokens1 = list(text)
        
        # Tokenization 2: word-level
        tokens2 = text.split()
        
        # Tokenization 3: subword (mock)
        tokens3 = ["Hel", "lo", " ", "wor", "ld"]
        
        # All represent same text
        assert "".join(tokens1) == text
        assert " ".join(tokens2) == text
        assert "".join(tokens3) == text
    
    def test_prompt_template_equivalence(self):
        """Test that different prompt templates can be equivalent."""
        # Base content
        question = "What is 2+2?"
        
        # Different templates
        template1 = f"Question: {question}\nAnswer:"
        template2 = f"Q: {question}\nA:"
        template3 = f"{question}"
        
        # All contain the question
        assert question in template1
        assert question in template2
        assert question in template3
        
        # Different formats but same content
        assert len(set([template1, template2, template3])) == 3
    
    def test_output_normalization(self):
        """Test output normalization for comparison."""
        # Different but equivalent outputs
        outputs = [
            "The answer is 4.",
            "4",
            "  4  ",
            "4\n",
            "Answer: 4"
        ]
        
        def normalize(text: str) -> str:
            """Simple normalization."""
            # Extract number
            import re
            numbers = re.findall(r'\d+', text)
            return numbers[0] if numbers else ""
        
        # All should normalize to same value
        normalized = [normalize(out) for out in outputs]
        assert all(n == "4" for n in normalized)


class TestWrapperDetection:
    """Test wrapper model detection."""
    
    def test_wrapper_behavior_detection(self):
        """Test detection of wrapper-like behavior."""
        torch.manual_seed(42)
        
        # Create base model
        base_model = nn.Linear(10, 10)
        
        # Create wrapper that adds constant
        class WrapperModel(nn.Module):
            def __init__(self, base, offset=0.1):
                super().__init__()
                self.base = base
                self.offset = offset
            
            def forward(self, x):
                return self.base(x) + self.offset
        
        wrapper = WrapperModel(base_model)
        
        # Test inputs
        test_inputs = [torch.randn(1, 10) for _ in range(10)]
        
        # Collect outputs
        base_outputs = []
        wrapper_outputs = []
        
        with torch.no_grad():
            for x in test_inputs:
                base_outputs.append(base_model(x))
                wrapper_outputs.append(wrapper(x))
        
        # Check for systematic difference
        differences = []
        for b, w in zip(base_outputs, wrapper_outputs):
            diff = (w - b).mean().item()
            differences.append(diff)
        
        # Wrapper adds constant offset
        assert all(abs(d - 0.1) < 1e-6 for d in differences)
        
        # Detection: consistent offset indicates wrapper
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        is_wrapper = std_diff < 0.01  # Very consistent difference
        assert is_wrapper
    
    def test_proxy_fraction_calculation(self):
        """Test calculation of proxy fraction for wrapper detection."""
        # Simulate outputs from original and potentially wrapped model
        n_samples = 100
        
        # Case 1: Identical model (not wrapped)
        original_outputs = [np.random.randn(10) for _ in range(n_samples)]
        identical_outputs = original_outputs.copy()
        
        proxy_fraction = 0
        for orig, test in zip(original_outputs, identical_outputs):
            if np.allclose(orig, test, rtol=1e-5):
                proxy_fraction += 1
        proxy_fraction /= n_samples
        
        assert proxy_fraction == 1.0  # All outputs match
        
        # Case 2: Wrapped model (systematic changes)
        wrapped_outputs = [out + 0.1 for out in original_outputs]
        
        proxy_fraction = 0
        for orig, test in zip(original_outputs, wrapped_outputs):
            if np.allclose(orig, test, rtol=1e-5):
                proxy_fraction += 1
        proxy_fraction /= n_samples
        
        assert proxy_fraction == 0.0  # No exact matches
        
        # Case 3: Partially wrapped (some pass-through)
        mixed_outputs = []
        for i, out in enumerate(original_outputs):
            if i % 2 == 0:
                mixed_outputs.append(out)  # Pass through
            else:
                mixed_outputs.append(out + 0.1)  # Modified
        
        proxy_fraction = 0
        for orig, test in zip(original_outputs, mixed_outputs):
            if np.allclose(orig, test, rtol=1e-5):
                proxy_fraction += 1
        proxy_fraction /= n_samples
        
        assert abs(proxy_fraction - 0.5) < 0.01  # About half match


class TestTransformComposition:
    """Test composition of multiple transforms."""
    
    def test_transform_composition_order(self):
        """Test that transform order matters."""
        torch.manual_seed(42)
        image = torch.randn(1, 3, 32, 32)
        
        # Apply in different orders
        # Order 1: rotate then flip
        result1 = EquivalenceTransforms.rotate90(image)
        result1 = EquivalenceTransforms.flip_horizontal(result1)
        
        # Order 2: flip then rotate
        result2 = EquivalenceTransforms.flip_horizontal(image)
        result2 = EquivalenceTransforms.rotate90(result2)
        
        # Results should be different
        assert not torch.allclose(result1, result2)
    
    def test_inverse_transforms(self):
        """Test that some transforms have inverses."""
        torch.manual_seed(42)
        image = torch.randn(1, 3, 32, 32)
        
        # Rotation: 4x should return to original
        result = image
        for _ in range(4):
            result = EquivalenceTransforms.rotate90(result)
        assert torch.allclose(result, image)
        
        # Flip: 2x should return to original
        result = EquivalenceTransforms.flip_horizontal(image)
        result = EquivalenceTransforms.flip_horizontal(result)
        assert torch.allclose(result, image)
        
        # Vertical flip: 2x should return to original
        result = EquivalenceTransforms.flip_vertical(image)
        result = EquivalenceTransforms.flip_vertical(result)
        assert torch.allclose(result, image)
    
    def test_transform_pipeline(self):
        """Test pipeline of multiple transforms."""
        torch.manual_seed(42)
        
        # Define transform pipeline
        def transform_pipeline(x, transforms):
            for transform in transforms:
                x = transform(x)
            return x
        
        # Test image
        image = torch.randn(1, 3, 32, 32)
        
        # Pipeline 1: Multiple transforms
        pipeline1 = [
            EquivalenceTransforms.rotate90,
            lambda x: EquivalenceTransforms.add_noise(x, 0.01),
            EquivalenceTransforms.flip_horizontal
        ]
        
        result1 = transform_pipeline(image, pipeline1)
        
        # Should be different from original
        assert not torch.allclose(result1, image)
        
        # Should have same shape
        assert result1.shape == image.shape
        
        # Apply same pipeline again for consistency
        result2 = transform_pipeline(image, pipeline1)
        
        # Results should be similar (except for random noise)
        # The structural transforms should be identical
        diff = torch.abs(result1 - result2)
        assert diff.mean() < 0.02  # Small difference due to noise


class TestAcceptanceRates:
    """Test transform acceptance rates."""
    
    def test_acceptance_rate_calculation(self):
        """Test calculation of transform acceptance rates."""
        np.random.seed(42)
        
        # Simulate verification with transforms
        n_trials = 1000
        threshold = 0.1
        
        # Simulate distances for original
        original_distances = np.random.exponential(0.05, n_trials)
        
        # Simulate distances for transformed (slightly higher)
        transform_distances = np.random.exponential(0.07, n_trials)
        
        # Calculate acceptance rates
        original_accept = np.mean(original_distances < threshold)
        transform_accept = np.mean(transform_distances < threshold)
        
        # Transform should have lower acceptance (higher distances)
        assert transform_accept < original_accept
        
        # Calculate relative acceptance
        relative_accept = transform_accept / original_accept if original_accept > 0 else 0
        assert 0 < relative_accept < 1
    
    def test_multiple_transform_rates(self):
        """Test acceptance rates for multiple transforms."""
        np.random.seed(42)
        n_trials = 1000
        threshold = 0.1
        
        # Base distances
        base_distances = np.random.exponential(0.05, n_trials)
        
        # Different transforms have different impacts
        transforms = {
            "identity": lambda d: d,  # No change
            "mild": lambda d: d * 1.2,  # 20% increase
            "moderate": lambda d: d * 1.5,  # 50% increase
            "severe": lambda d: d * 2.0,  # 100% increase
        }
        
        acceptance_rates = {}
        for name, transform in transforms.items():
            transformed = transform(base_distances)
            acceptance_rates[name] = np.mean(transformed < threshold)
        
        # Identity should have highest acceptance
        assert acceptance_rates["identity"] == max(acceptance_rates.values())
        
        # Severe should have lowest acceptance
        assert acceptance_rates["severe"] == min(acceptance_rates.values())
        
        # Should be monotonic
        assert (acceptance_rates["identity"] >= 
                acceptance_rates["mild"] >= 
                acceptance_rates["moderate"] >= 
                acceptance_rates["severe"])


def test_equivalence_budget():
    """Test equivalence budget tracking."""
    # Budget for proxy behavior
    budget = 0.15  # 15% allowed proxy fraction
    
    # Simulate verification with potential wrapper
    n_challenges = 100
    proxy_responses = 0
    
    for i in range(n_challenges):
        # Simulate checking if response is proxy
        is_proxy = i < 10  # First 10% are proxy
        if is_proxy:
            proxy_responses += 1
        
        # Check if budget exceeded
        current_fraction = proxy_responses / (i + 1)
        if current_fraction > budget:
            print(f"Budget exceeded at challenge {i+1}")
            break
    
    final_fraction = proxy_responses / n_challenges
    assert final_fraction <= budget


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])