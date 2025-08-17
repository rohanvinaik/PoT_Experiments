"""Integration tests for verifier updates to use centralized challenge generation."""

import pytest
import torch
import numpy as np
from pot.vision.verifier import VisionVerifier
from pot.lm.verifier import LMVerifier
from pot.core.challenge import ChallengeConfig, generate_challenges

# Add parent directory to path for pot imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



class TestVisionVerifierIntegration:
    """Test vision verifier with centralized challenge generation."""
    
    @pytest.fixture
    def dummy_vision_model(self):
        """Create a dummy vision model for testing."""
        class DummyVisionModel:
            def __init__(self):
                self.device = torch.device("cpu")
            
            def forward(self, x):
                return torch.randn(x.shape[0], 1000)
        
        return DummyVisionModel()
    
    def test_frequency_challenges_with_model_id(self, dummy_vision_model):
        """Test frequency challenge generation with model ID."""
        verifier = VisionVerifier(dummy_vision_model, use_sequential=False, detect_wrappers=False)
        
        master_key = "0123456789abcdef" * 4
        session_nonce = "fedcba9876543210" * 2
        model_id = "resnet50_v1"
        
        images, metadata = verifier.generate_frequency_challenges(
            5, master_key, session_nonce, model_id
        )
        
        assert len(images) == 5
        assert len(metadata) == 5
        
        # Check metadata structure
        for meta in metadata:
            assert "challenge_id" in meta
            assert "index" in meta
            assert "parameters" in meta
            assert len(meta["challenge_id"]) > 0
            assert meta["challenge_id"] != f"legacy_{meta['index']}"  # Should use real IDs
    
    def test_texture_challenges_consistency(self, dummy_vision_model):
        """Test texture challenge generation consistency."""
        verifier = VisionVerifier(dummy_vision_model, use_sequential=False, detect_wrappers=False)
        
        master_key = "deadbeef" * 8
        session_nonce = "cafebabe" * 4
        
        # Generate twice with same parameters
        images1, meta1 = verifier.generate_texture_challenges(3, master_key, session_nonce)
        images2, meta2 = verifier.generate_texture_challenges(3, master_key, session_nonce)
        
        # Should be identical
        for m1, m2 in zip(meta1, meta2):
            assert m1["challenge_id"] == m2["challenge_id"]
            assert m1["parameters"] == m2["parameters"]
    
    def test_backward_compatibility(self, dummy_vision_model):
        """Test backward compatibility when challenges field is not present."""
        verifier = VisionVerifier(dummy_vision_model, use_sequential=False, detect_wrappers=False)
        
        # This should still work even if the centralized generation
        # doesn't provide Challenge objects (backward compatibility)
        images, metadata = verifier.generate_frequency_challenges(
            2, "abcd" * 16, "1234" * 8
        )
        
        assert len(images) == 2
        assert len(metadata) == 2


class TestLMVerifierIntegration:
    """Test LM verifier with centralized challenge generation."""
    
    @pytest.fixture
    def dummy_lm_model(self):
        """Create a dummy LM model for testing."""
        class DummyTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(c) for c in text[:100]]
            
            def decode(self, tokens):
                return ''.join([chr(t % 128) for t in tokens])
        
        class DummyLM:
            def __init__(self):
                self.tok = DummyTokenizer()
                self.device = torch.device("cpu")
            
            def generate(self, prompt):
                return "Mock response"
        
        return DummyLM()
    
    def test_template_challenges_with_model_id(self, dummy_lm_model):
        """Test template challenge generation with model ID."""
        verifier = LMVerifier(dummy_lm_model, use_sequential=False)
        
        master_key = "0123456789abcdef" * 4
        session_nonce = "fedcba9876543210" * 2
        model_id = "gpt4_v1"
        
        prompts, metadata = verifier.generate_template_challenges(
            5, master_key, session_nonce, model_id
        )
        
        assert len(prompts) == 5
        assert len(metadata) == 5
        
        # Check metadata structure
        for meta in metadata:
            assert "challenge_id" in meta
            assert "index" in meta
            assert "prompt" in meta
            assert len(meta["challenge_id"]) > 0
            assert meta["challenge_id"] != f"legacy_{meta['index']}"  # Should use real IDs
        
        # Check prompts are properly generated
        for prompt in prompts:
            assert len(prompt) > 0
            assert "{" not in prompt  # No unfilled slots
            assert "}" not in prompt
    
    def test_prompt_consistency(self, dummy_lm_model):
        """Test that prompts are consistently generated."""
        verifier = LMVerifier(dummy_lm_model, use_sequential=False)
        
        master_key = "deadbeef" * 8
        session_nonce = "cafebabe" * 4
        model_id = "test_model"
        
        # Generate twice
        prompts1, meta1 = verifier.generate_template_challenges(
            3, master_key, session_nonce, model_id
        )
        prompts2, meta2 = verifier.generate_template_challenges(
            3, master_key, session_nonce, model_id
        )
        
        # Should be identical
        assert prompts1 == prompts2
        for m1, m2 in zip(meta1, meta2):
            assert m1["challenge_id"] == m2["challenge_id"]
            assert m1["prompt"] == m2["prompt"]
    
    def test_model_id_influence(self, dummy_lm_model):
        """Test that different model IDs produce different challenges."""
        verifier = LMVerifier(dummy_lm_model, use_sequential=False)
        
        master_key = "abcd" * 16
        session_nonce = "1234" * 8
        
        prompts_a, _ = verifier.generate_template_challenges(
            3, master_key, session_nonce, "model_a"
        )
        prompts_b, _ = verifier.generate_template_challenges(
            3, master_key, session_nonce, "model_b"
        )
        
        # At least some prompts should be different
        assert prompts_a != prompts_b


class TestChallengeIntegration:
    """Test integration between verifiers and centralized challenge generation."""
    
    def test_vision_freq_direct_vs_verifier(self):
        """Test that direct generation matches verifier generation for vision:freq."""
        master_key = "0123456789abcdef" * 4
        session_nonce = "fedcba9876543210" * 2
        model_id = "test_model"
        
        # Direct generation
        config = ChallengeConfig(
            master_key_hex=master_key,
            session_nonce_hex=session_nonce,
            n=3,
            family="vision:freq",
            params={
                "freq_range": (0.5, 10.0),
                "contrast_range": (0.2, 1.0)
            },
            model_id=model_id
        )
        direct_result = generate_challenges(config)
        
        # Verifier generation
        class DummyModel:
            device = torch.device("cpu")
            def forward(self, x): return x
        
        verifier = VisionVerifier(DummyModel(), use_sequential=False, detect_wrappers=False)
        _, verifier_metadata = verifier.generate_frequency_challenges(
            3, master_key, session_nonce, model_id
        )
        
        # Compare challenge IDs
        direct_ids = [c.challenge_id for c in direct_result["challenges"]]
        verifier_ids = [m["challenge_id"] for m in verifier_metadata]
        
        assert direct_ids == verifier_ids
    
    def test_lm_templates_direct_vs_verifier(self):
        """Test that direct generation matches verifier generation for lm:templates."""
        master_key = "deadbeef" * 8
        session_nonce = "cafebabe" * 4
        model_id = "test_lm"
        
        # The verifier defines specific templates and slots
        templates = [
            "Complete the following: The {object} is {attribute}",
            "Q: What is {concept}? A:",
            "Translate to {language}: {text}",
            "Summarize: {passage}",
            "Continue the story: {beginning}",
        ]
        
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
        
        # Direct generation with same parameters
        config = ChallengeConfig(
            master_key_hex=master_key,
            session_nonce_hex=session_nonce,
            n=3,
            family="lm:templates",
            params={"templates": templates, "slots": slots},
            model_id=model_id
        )
        direct_result = generate_challenges(config)
        
        # Verifier generation
        class DummyTokenizer:
            def encode(self, text, add_special_tokens=False): return []
            def decode(self, tokens): return ""
        
        class DummyLM:
            tok = DummyTokenizer()
            device = torch.device("cpu")
        
        verifier = LMVerifier(DummyLM(), use_sequential=False)
        prompts, verifier_metadata = verifier.generate_template_challenges(
            3, master_key, session_nonce, model_id
        )
        
        # Compare prompts
        direct_prompts = [c.parameters["prompt"] for c in direct_result["challenges"]]
        
        assert direct_prompts == prompts