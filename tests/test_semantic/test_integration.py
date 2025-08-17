"""
Test suite for semantic verification integration with existing verifiers.
Tests integration with LM and Vision verifiers.
"""

import pytest
import torch
import numpy as np
from typing import Optional

from pot.semantic.library import ConceptLibrary
from pot.semantic.match import SemanticMatcher
from pot.semantic.config import (
    SemanticVerificationConfig,
    load_semantic_config,
    create_semantic_components,
    integrate_with_verifier
)
from pot.lm.verifier import LMVerifier
from pot.lm.models import LM
from pot.vision.verifier import VisionVerifier
from pot.vision.models import VisionModel


class MockTokenizer:
    """Mock tokenizer for testing."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.bos_token_id = 3
        self.sep_token_id = 4
        self.cls_token_id = 5
    
    def encode(self, text: str, add_special_tokens: bool = True):
        return [ord(c) for c in text[:100]]
    
    def decode(self, token_ids):
        return ''.join([chr(t) if 32 <= t < 127 else '?' for t in token_ids])


class MockLM(LM):
    """Mock language model for testing."""
    def __init__(self):
        self.tok = MockTokenizer()
    
    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        return f"Response to: {prompt[:20]}..."
    
    def get_hidden_states(self, prompt: str) -> Optional[torch.Tensor]:
        return torch.randn(1, 768)


class MockVisionModel(VisionModel):
    """Mock vision model for testing."""
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0] if x.dim() > 3 else 1
        return torch.randn(batch_size, 512)


class TestLMVerifierIntegration:
    """Test integration with LM verifier."""
    
    def test_lm_verifier_without_semantic(self):
        """Test LM verifier works without semantic library."""
        ref_model = MockLM()
        verifier = LMVerifier(ref_model, delta=0.01)
        
        assert verifier.semantic_library is None
        assert verifier.semantic_matcher is None
        
        # Should work normally
        challenges = [{"prompt": "Hello"}]
        test_model = MockLM()
        result = verifier.verify(test_model, challenges, tolerance=0.5)
        
        assert result.semantic_score is None
        assert result.combined_score is None
    
    def test_lm_verifier_with_semantic(self):
        """Test LM verifier with semantic library."""
        # Create semantic library
        library = ConceptLibrary(dim=768, method='gaussian')
        
        # Add concepts
        for i in range(3):
            embeddings = torch.randn(10, 768) + torch.ones(768) * i
            library.add_concept(f'lm_concept_{i}', embeddings)
        
        # Create verifier with semantic
        ref_model = MockLM()
        verifier = LMVerifier(
            ref_model,
            delta=0.01,
            semantic_library=library,
            semantic_weight=0.3
        )
        
        assert verifier.semantic_library is not None
        assert verifier.semantic_matcher is not None
        assert verifier.semantic_weight == 0.3
        
        # Test verification
        challenges = [
            {"template": "Complete: The {object} is", "slot_values": {"object": "cat"}},
            {"template": "Complete: The {object} is", "slot_values": {"object": "dog"}}
        ]
        
        test_model = MockLM()
        result = verifier.verify(test_model, challenges, tolerance=0.5)
        
        assert result.semantic_score is not None
        assert 0.0 <= result.semantic_score <= 1.0
        
        if result.semantic_score is not None:
            assert result.combined_score is not None
            # Combined score should be weighted average
            expected_combined = (1 - 0.3) * result.distance + 0.3 * (1 - result.semantic_score)
            assert abs(result.combined_score - expected_combined) < 0.01
    
    def test_lm_semantic_weight_effect(self):
        """Test that semantic weight affects decisions."""
        library = ConceptLibrary(dim=768, method='gaussian')
        embeddings = torch.randn(10, 768)
        library.add_concept('test_concept', embeddings)
        
        ref_model = MockLM()
        
        # Low semantic weight
        verifier_low = LMVerifier(
            ref_model,
            semantic_library=library,
            semantic_weight=0.1
        )
        
        # High semantic weight
        verifier_high = LMVerifier(
            ref_model,
            semantic_library=library,
            semantic_weight=0.9
        )
        
        challenges = [{"prompt": "Test"}]
        test_model = MockLM()
        
        result_low = verifier_low.verify(test_model, challenges, tolerance=0.5)
        result_high = verifier_high.verify(test_model, challenges, tolerance=0.5)
        
        # Combined scores should differ based on weight
        if result_low.semantic_score and result_high.semantic_score:
            assert result_low.combined_score != result_high.combined_score


class TestVisionVerifierIntegration:
    """Test integration with Vision verifier."""
    
    def test_vision_verifier_without_semantic(self):
        """Test Vision verifier works without semantic library."""
        ref_model = MockVisionModel()
        verifier = VisionVerifier(ref_model, delta=0.01)
        
        assert verifier.semantic_library is None
        assert verifier.semantic_matcher is None
        
        # Should work normally
        challenges = [torch.randn(3, 224, 224)]
        test_model = MockVisionModel()
        result = verifier.verify(test_model, challenges, tolerance=0.5)
        
        assert result.semantic_score is None
        assert result.combined_score is None
    
    def test_vision_verifier_with_semantic(self):
        """Test Vision verifier with semantic library."""
        # Create semantic library
        library = ConceptLibrary(dim=512, method='gaussian')
        
        # Add concepts
        for i in range(3):
            embeddings = torch.randn(10, 512) + torch.ones(512) * i
            library.add_concept(f'vision_concept_{i}', embeddings)
        
        # Create verifier with semantic
        ref_model = MockVisionModel()
        verifier = VisionVerifier(
            ref_model,
            delta=0.01,
            semantic_library=library,
            semantic_weight=0.3
        )
        
        assert verifier.semantic_library is not None
        assert verifier.semantic_matcher is not None
        assert verifier.semantic_weight == 0.3
        
        # Test verification
        challenges = [torch.randn(3, 224, 224) for _ in range(3)]
        test_model = MockVisionModel()
        result = verifier.verify(test_model, challenges, tolerance=0.5)
        
        assert result.semantic_score is not None
        assert 0.0 <= result.semantic_score <= 1.0
        
        if result.semantic_score is not None:
            assert result.combined_score is not None
    
    def test_batch_vision_verifier_with_semantic(self):
        """Test BatchVisionVerifier with semantic library."""
        from pot.vision.verifier import BatchVisionVerifier
        
        library = ConceptLibrary(dim=512, method='gaussian')
        embeddings = torch.randn(10, 512)
        library.add_concept('batch_concept', embeddings)
        
        ref_model = MockVisionModel()
        batch_verifier = BatchVisionVerifier(
            ref_model,
            delta=0.01,
            semantic_library=library,
            semantic_weight=0.3
        )
        
        assert batch_verifier.verifier.semantic_library is not None
        
        # Test batch verification
        models = [MockVisionModel() for _ in range(2)]
        challenges = [torch.randn(3, 224, 224) for _ in range(2)]
        
        results = batch_verifier.verify_batch(models, challenges, tolerance=0.5)
        
        assert len(results) == 2
        for result in results:
            assert result.semantic_score is not None


class TestConfigurationIntegration:
    """Test configuration-based integration."""
    
    def test_semantic_config_creation(self):
        """Test creating semantic configuration."""
        config = SemanticVerificationConfig()
        
        assert config.enabled
        assert config.semantic_weight == 0.3
        assert config.library_method == 'gaussian'
        assert config.library_dimension == 768
        
        # Validate config
        assert config.validate()
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'semantic_verification': {
                'enabled': True,
                'semantic_weight': 0.5,
                'library': {
                    'method': 'hypervector',
                    'dimension': 512
                },
                'matching': {
                    'threshold': 0.8,
                    'primary_method': 'hamming'
                }
            }
        }
        
        config = SemanticVerificationConfig.from_dict(config_dict)
        
        assert config.enabled
        assert config.semantic_weight == 0.5
        assert config.library_method == 'hypervector'
        assert config.library_dimension == 512
        assert config.matching_threshold == 0.8
        assert config.matching_primary_method == 'hamming'
    
    def test_create_semantic_components(self):
        """Test creating semantic components from config."""
        config = SemanticVerificationConfig(
            enabled=True,
            library_dimension=128,
            library_method='gaussian'
        )
        
        library, matcher = create_semantic_components(config)
        
        assert library is not None
        assert matcher is not None
        assert library.dim == 128
        assert library.method == 'gaussian'
        assert matcher.threshold == config.matching_threshold
    
    def test_create_components_disabled(self):
        """Test creating components when disabled."""
        config = SemanticVerificationConfig(enabled=False)
        
        library, matcher = create_semantic_components(config)
        
        assert library is None
        assert matcher is None
    
    def test_integrate_with_verifier(self):
        """Test integrating semantic with existing verifier."""
        # Create config
        config = SemanticVerificationConfig(
            enabled=True,
            lm_enabled=True,
            lm_dimension=768,
            lm_semantic_weight=0.4
        )
        
        # Create verifier
        ref_model = MockLM()
        verifier = LMVerifier(ref_model)
        
        # Initially no semantic
        assert verifier.semantic_library is None
        
        # Integrate semantic
        integrate_with_verifier('lm', verifier, config)
        
        # Now has semantic
        assert verifier.semantic_library is not None
        assert verifier.semantic_weight == 0.4
    
    def test_integrate_when_disabled(self):
        """Test integration when semantic is disabled."""
        config = SemanticVerificationConfig(
            enabled=False
        )
        
        ref_model = MockVisionModel()
        verifier = VisionVerifier(ref_model)
        
        integrate_with_verifier('vision', verifier, config)
        
        # Should remain None
        assert verifier.semantic_library is None


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_lm_verifier_backward_compat(self):
        """Test LM verifier works without semantic parameters."""
        ref_model = MockLM()
        
        # Old-style initialization
        verifier = LMVerifier(ref_model)
        
        assert verifier.semantic_library is None
        assert verifier.semantic_matcher is None
        
        # Old-style verification
        test_model = MockLM()
        challenges = [{"prompt": "Test"}]
        result = verifier.verify(test_model, challenges)
        
        assert result.accepted is not None
        assert result.distance is not None
        assert result.semantic_score is None
    
    def test_vision_verifier_backward_compat(self):
        """Test Vision verifier works without semantic parameters."""
        ref_model = MockVisionModel()
        
        # Old-style initialization
        verifier = VisionVerifier(ref_model)
        
        assert verifier.semantic_library is None
        assert verifier.semantic_matcher is None
        
        # Old-style verification
        test_model = MockVisionModel()
        challenges = [torch.randn(3, 224, 224)]
        result = verifier.verify(test_model, challenges)
        
        assert result.accepted is not None
        assert result.distance is not None
        assert result.semantic_score is None
    
    def test_result_dataclass_backward_compat(self):
        """Test result dataclasses are backward compatible."""
        from pot.lm.verifier import LMVerificationResult
        from pot.vision.verifier import VisionVerificationResult
        
        # Create results without semantic fields
        lm_result = LMVerificationResult(
            accepted=True,
            distance=0.1,
            confidence_radius=0.01,
            n_challenges=10,
            fuzzy_similarity=0.9,
            time_elapsed=1.0,
            fingerprint=None,
            fingerprint_match=None,
            sequential_result=None
        )
        
        # Should have default None for semantic fields
        assert lm_result.semantic_score is None
        assert lm_result.combined_score is None
        
        vision_result = VisionVerificationResult(
            accepted=True,
            distance=0.1,
            confidence_radius=0.01,
            n_challenges=10,
            perceptual_similarity=0.9,
            time_elapsed=1.0,
            wrapper_detection=None,
            fingerprint=None,
            fingerprint_match=None,
            sequential_result=None
        )
        
        assert vision_result.semantic_score is None
        assert vision_result.combined_score is None


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_complete_workflow_lm(self):
        """Test complete semantic verification workflow for LM."""
        # Create and populate library
        library = ConceptLibrary(dim=768, method='gaussian')
        
        # Add concepts from "training"
        for i in range(3):
            training_embeddings = torch.randn(20, 768) + torch.ones(768) * i
            library.add_concept(f'training_concept_{i}', training_embeddings)
        
        # Create reference and test models
        ref_model = MockLM()
        test_model = MockLM()
        
        # Create verifier with semantic
        verifier = LMVerifier(
            ref_model,
            semantic_library=library,
            semantic_weight=0.3
        )
        
        # Generate challenges
        challenges = []
        for i in range(5):
            challenges.append({
                "template": f"Test prompt {i}",
                "slot_values": {}
            })
        
        # Verify
        result = verifier.verify(test_model, challenges, tolerance=0.5)
        
        # Check complete result
        assert result.accepted is not None
        assert result.distance >= 0
        assert result.confidence_radius >= 0
        assert result.n_challenges == 5
        assert result.semantic_score is not None
        assert result.combined_score is not None
        
        # Metadata should include semantic info
        assert 'test_statistic' in result.metadata
    
    def test_complete_workflow_vision(self):
        """Test complete semantic verification workflow for Vision."""
        # Create and populate library
        library = ConceptLibrary(dim=512, method='gaussian')
        
        # Add visual concepts
        for i in range(3):
            visual_embeddings = torch.randn(20, 512) + torch.ones(512) * i * 0.5
            library.add_concept(f'visual_class_{i}', visual_embeddings)
        
        # Create reference and test models
        ref_model = MockVisionModel()
        test_model = MockVisionModel()
        
        # Create verifier with semantic
        verifier = VisionVerifier(
            ref_model,
            semantic_library=library,
            semantic_weight=0.3
        )
        
        # Generate challenges
        challenges = [torch.randn(3, 224, 224) for _ in range(5)]
        
        # Verify
        result = verifier.verify(test_model, challenges, tolerance=0.5)
        
        # Check complete result
        assert result.accepted is not None
        assert result.distance >= 0
        assert result.confidence_radius >= 0
        assert result.n_challenges == 5
        assert result.semantic_score is not None
        assert result.combined_score is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])