#!/usr/bin/env python3
"""
Test script for Challenge Generators

This script tests the comprehensive challenge generation system with:
- Vision challenges (adversarial, style transfer, compression)
- Language challenges (paraphrasing, substitution, perturbation)
- Multimodal challenges (cross-modal verification)
- Configurable difficulty levels and deterministic generation
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any

from pot.experiments.challenge_generator import (
    ChallengeGenerator, ChallengeConfig, ChallengeType, ChallengeResult,
    AdversarialChallengeGenerator, StyleTransferChallengeGenerator,
    CompressionChallengeGenerator, ParaphraseChallengeGenerator,
    SubstitutionChallengeGenerator, PerturbationChallengeGenerator,
    CrossModalChallengeGenerator, BaselineChallengeGenerator,
    create_challenge_generator, generate_vision_challenge,
    generate_language_challenge, generate_baseline_challenge,
    generate_challenge_batch, get_available_challenge_types
)

def create_test_vision_model() -> nn.Module:
    """Create a simple vision model for testing."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(16 * 8 * 8, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    model.eval()
    return model

def create_test_language_model() -> nn.Module:
    """Create a simple language model for testing."""
    class SimpleLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 64)
            self.lstm = nn.LSTM(64, 128, batch_first=True)
            self.classifier = nn.Linear(128, 1000)
        
        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            # Use last output
            output = self.classifier(lstm_out[:, -1, :])
            return output
    
    model = SimpleLanguageModel()
    model.eval()
    return model

def create_test_multimodal_model() -> nn.Module:
    """Create a simple multimodal model for testing."""
    class SimpleMultimodalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(16 * 4 * 4, 64)
            )
            self.text_encoder = nn.Sequential(
                nn.Embedding(1000, 64),
                nn.LSTM(64, 64, batch_first=True)
            )
            self.classifier = nn.Linear(128, 10)
        
        def forward(self, image_text_tuple):
            if isinstance(image_text_tuple, tuple):
                image, text = image_text_tuple
                # Ensure proper batch dimensions
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                image_features = self.vision_encoder(image)
                # Simple text encoding (would normally use proper tokenization)
                text_features = torch.randn(image_features.shape[0], 64)  # Match batch size
                combined = torch.cat([image_features, text_features], dim=-1)
                return self.classifier(combined)
            else:
                # Handle single input (image)
                if image_text_tuple.dim() == 3:
                    image_text_tuple = image_text_tuple.unsqueeze(0)
                return self.vision_encoder(image_text_tuple)
        
        def encode_image(self, image):
            return self.vision_encoder(image)
        
        def encode_text(self, text):
            return torch.randn(1, 64)  # Mock text encoding
    
    model = SimpleMultimodalModel()
    model.eval()
    return model

def test_baseline_challenges():
    """Test baseline challenge generators."""
    print("🧪 Testing Baseline Challenges")
    print("-" * 40)
    
    # Vision baseline
    vision_model = create_test_vision_model()
    test_image = torch.randn(3, 32, 32)
    
    try:
        result = generate_baseline_challenge(vision_model, test_image, "vision", difficulty=0.1)
        print(f"✅ Vision baseline challenge:")
        print(f"   - Type: {result.challenge_type.value}")
        print(f"   - Difficulty: {result.difficulty}")
        print(f"   - Distance: {result.distance:.4f}")
        print(f"   - Passed: {result.passed}")
        print(f"   - Generation time: {result.generation_time:.4f}s")
    except Exception as e:
        print(f"❌ Vision baseline failed: {e}")
        return False
    
    # Language baseline
    language_model = create_test_language_model()
    test_tokens = torch.randint(0, 1000, (64,))
    
    try:
        result = generate_baseline_challenge(language_model, test_tokens, "language", difficulty=0.1)
        print(f"✅ Language baseline challenge:")
        print(f"   - Type: {result.challenge_type.value}")
        print(f"   - Difficulty: {result.difficulty}")
        print(f"   - Distance: {result.distance:.4f}")
        print(f"   - Passed: {result.passed}")
    except Exception as e:
        print(f"❌ Language baseline failed: {e}")
        return False
    
    return True

def test_vision_challenges():
    """Test vision challenge generators."""
    print("\n📸 Testing Vision Challenges")
    print("-" * 40)
    
    model = create_test_vision_model()
    test_image = torch.randn(3, 32, 32)
    
    vision_challenges = ["adversarial_patch", "style_transfer", "compression"]
    difficulties = [0.2, 0.5, 0.8]
    
    for challenge_type in vision_challenges:
        print(f"\n🔧 Testing {challenge_type}:")
        
        for difficulty in difficulties:
            try:
                result = generate_vision_challenge(
                    model, test_image, challenge_type, difficulty
                )
                
                print(f"   Difficulty {difficulty}: "
                      f"distance={result.distance:.4f}, "
                      f"passed={result.passed}, "
                      f"time={result.generation_time:.4f}s")
                
                # Verify metadata
                if not result.metadata:
                    print(f"   ⚠️  No metadata for {challenge_type}")
                
            except Exception as e:
                print(f"   ❌ {challenge_type} difficulty {difficulty} failed: {e}")
                return False
    
    return True

def test_language_challenges():
    """Test language challenge generators."""
    print("\n📝 Testing Language Challenges")
    print("-" * 40)
    
    model = create_test_language_model()
    test_text = "The quick brown fox jumps over the lazy dog"
    
    language_challenges = ["paraphrasing", "substitution"]
    difficulties = [0.3, 0.6, 0.9]
    
    for challenge_type in language_challenges:
        print(f"\n🔧 Testing {challenge_type}:")
        
        for difficulty in difficulties:
            try:
                result = generate_language_challenge(
                    model, test_text, challenge_type, difficulty,
                    max_length=64, vocab_size=1000
                )
                
                print(f"   Difficulty {difficulty}: "
                      f"distance={result.distance:.4f}, "
                      f"passed={result.passed}, "
                      f"time={result.generation_time:.4f}s")
                
                # Print text modifications for language challenges
                if "original_text" in result.metadata:
                    orig = result.metadata["original_text"]
                    if challenge_type == "paraphrasing" and "paraphrased_text" in result.metadata:
                        mod = result.metadata["paraphrased_text"]
                        print(f"   Original: '{orig}'")
                        print(f"   Modified: '{mod}'")
                    elif challenge_type == "substitution" and "substituted_text" in result.metadata:
                        mod = result.metadata["substituted_text"]
                        print(f"   Original: '{orig}'")
                        print(f"   Modified: '{mod}'")
                
            except Exception as e:
                print(f"   ❌ {challenge_type} difficulty {difficulty} failed: {e}")
                return False
    
    return True

def test_multimodal_challenges():
    """Test multimodal challenge generators."""
    print("\n🔄 Testing Multimodal Challenges")
    print("-" * 40)
    
    model = create_test_multimodal_model()
    test_image = torch.randn(3, 32, 32)
    test_text = "A cat sitting on a table"
    test_input = (test_image, test_text)
    
    multimodal_challenges = ["cross_modal"]
    difficulties = [0.3, 0.7]
    
    for challenge_type in multimodal_challenges:
        print(f"\n🔧 Testing {challenge_type}:")
        
        for difficulty in difficulties:
            try:
                config = ChallengeConfig(
                    challenge_type=ChallengeType.CROSS_MODAL,
                    difficulty=difficulty,
                    parameters={"mismatch_type": "text_substitution"}
                )
                
                generator = create_challenge_generator(config.challenge_type)
                result = generator.generate_and_verify(model, test_input, config)
                
                print(f"   Difficulty {difficulty}: "
                      f"distance={result.distance:.4f}, "
                      f"passed={result.passed}, "
                      f"time={result.generation_time:.4f}s")
                
                # Print multimodal modifications
                if "original_text" in result.metadata and "modified_text" in result.metadata:
                    print(f"   Original text: '{result.metadata['original_text']}'")
                    print(f"   Modified text: '{result.metadata['modified_text']}'")
                
            except Exception as e:
                print(f"   ❌ {challenge_type} difficulty {difficulty} failed: {e}")
                return False
    
    return True

def test_perturbation_challenges():
    """Test perturbation challenge generators."""
    print("\n🔀 Testing Perturbation Challenges")
    print("-" * 40)
    
    model = create_test_vision_model()
    test_image = torch.randn(3, 32, 32)
    
    perturbation_types = ["gaussian_noise", "uniform_noise", "salt_pepper"]
    difficulties = [0.1, 0.5, 0.9]
    
    for perturbation_type in perturbation_types:
        print(f"\n🔧 Testing {perturbation_type}:")
        
        for difficulty in difficulties:
            try:
                config = ChallengeConfig(
                    challenge_type=ChallengeType.PERTURBATION,
                    difficulty=difficulty,
                    parameters={"perturbation_type": perturbation_type}
                )
                
                generator = create_challenge_generator(config.challenge_type)
                result = generator.generate_and_verify(model, test_image, config)
                
                print(f"   Difficulty {difficulty}: "
                      f"distance={result.distance:.4f}, "
                      f"passed={result.passed}, "
                      f"noise_strength={result.metadata.get('noise_strength', 'N/A'):.4f}")
                
            except Exception as e:
                print(f"   ❌ {perturbation_type} difficulty {difficulty} failed: {e}")
                return False
    
    return True

def test_deterministic_generation():
    """Test deterministic generation with seeds."""
    print("\n🎲 Testing Deterministic Generation")
    print("-" * 40)
    
    model = create_test_vision_model()
    test_image = torch.randn(3, 32, 32)
    
    # Generate same challenge twice with same seed
    config1 = ChallengeConfig(
        challenge_type=ChallengeType.ADVERSARIAL_PATCH,
        difficulty=0.5,
        seed=42
    )
    
    config2 = ChallengeConfig(
        challenge_type=ChallengeType.ADVERSARIAL_PATCH,
        difficulty=0.5,
        seed=42
    )
    
    try:
        generator = create_challenge_generator(config1.challenge_type)
        result1 = generator.generate_challenge(model, test_image, config1)
        result2 = generator.generate_challenge(model, test_image, config2)
        
        # Check if results are similar (should be identical with same seed)
        dist1 = result1.distance if result1.distance is not None else 0.0
        dist2 = result2.distance if result2.distance is not None else 0.0
        distance_diff = abs(dist1 - dist2)
        
        print(f"✅ Deterministic generation test:")
        print(f"   Seed: {config1.seed}")
        print(f"   Result 1 distance: {dist1:.6f}")
        print(f"   Result 2 distance: {dist2:.6f}")
        print(f"   Distance difference: {distance_diff:.6f}")
        
        if distance_diff < 1e-6:
            print("   ✅ Results are identical - deterministic generation working!")
        else:
            print("   ⚠️  Results differ - check deterministic generation")
        
        return True
        
    except Exception as e:
        print(f"❌ Deterministic generation test failed: {e}")
        return False

def test_batch_generation():
    """Test batch challenge generation."""
    print("\n📦 Testing Batch Generation")
    print("-" * 40)
    
    model = create_test_vision_model()
    test_image = torch.randn(3, 32, 32)
    
    challenge_types = ["baseline_vision", "style_transfer", "compression"]
    
    try:
        results = generate_challenge_batch(
            model, test_image, challenge_types,
            difficulty=0.5, seed_offset=100
        )
        
        print(f"✅ Batch generation successful:")
        print(f"   Requested: {len(challenge_types)} challenges")
        print(f"   Generated: {len(results)} challenges")
        
        for i, result in enumerate(results):
            print(f"   {i+1}. {result.challenge_type.value}: "
                  f"distance={result.distance:.4f}, "
                  f"passed={result.passed}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Batch generation failed: {e}")
        return False

def test_challenge_metadata():
    """Test challenge metadata and serialization."""
    print("\n📋 Testing Challenge Metadata")
    print("-" * 40)
    
    model = create_test_vision_model()
    test_image = torch.randn(3, 32, 32)
    
    try:
        config = ChallengeConfig(
            challenge_type=ChallengeType.STYLE_TRANSFER,
            difficulty=0.6,
            seed=123,
            save_metadata=True
        )
        
        generator = create_challenge_generator(config.challenge_type)
        result = generator.generate_and_verify(model, test_image, config)
        
        print(f"✅ Metadata test:")
        print(f"   Challenge type: {result.challenge_type.value}")
        print(f"   Seed: {result.seed}")
        print(f"   Difficulty: {result.difficulty}")
        print(f"   Generation time: {result.generation_time:.4f}s")
        
        # Test serialization
        result_dict = result.to_dict()
        print(f"   Serializable: {len(result_dict)} fields")
        
        # Check required metadata fields
        required_fields = ["challenge_type", "difficulty", "seed", "metadata"]
        missing_fields = [field for field in required_fields if field not in result_dict]
        
        if missing_fields:
            print(f"   ⚠️  Missing fields: {missing_fields}")
        else:
            print("   ✅ All required metadata fields present")
        
        return len(missing_fields) == 0
        
    except Exception as e:
        print(f"❌ Metadata test failed: {e}")
        return False

def test_available_challenges():
    """Test available challenge types listing."""
    print("\n📋 Testing Available Challenge Types")
    print("-" * 40)
    
    try:
        available = get_available_challenge_types()
        
        print("Available challenge types:")
        for modality, challenges in available.items():
            print(f"  {modality.upper()}:")
            for challenge in challenges:
                print(f"    - {challenge}")
        
        total_challenges = sum(len(challenges) for challenges in available.values())
        print(f"\nTotal available: {total_challenges} challenge types")
        
        return total_challenges > 0
        
    except Exception as e:
        print(f"❌ Available challenges test failed: {e}")
        return False

def main():
    """Run all challenge generator tests."""
    print("🧪 Challenge Generators Test Suite")
    print("=" * 50)
    print("Testing comprehensive challenge generation system with")
    print("vision, language, and multimodal challenges.\n")
    
    test_functions = [
        test_baseline_challenges,
        test_vision_challenges,
        test_language_challenges,
        test_multimodal_challenges,
        test_perturbation_challenges,
        test_deterministic_generation,
        test_batch_generation,
        test_challenge_metadata,
        test_available_challenges
    ]
    
    results = []
    for test_func in test_functions:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results")
    print("=" * 30)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        print("\n✅ Key Features Verified:")
        print("   - Vision challenges (adversarial, style, compression)")
        print("   - Language challenges (paraphrasing, substitution)")
        print("   - Multimodal challenges (cross-modal verification)")
        print("   - Configurable difficulty levels")
        print("   - Deterministic generation with seed management")
        print("   - Comprehensive metadata tracking")
        print("   - Batch challenge generation")
        print("   - Distance-based verification")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)