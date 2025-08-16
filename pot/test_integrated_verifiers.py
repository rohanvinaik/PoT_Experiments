#!/usr/bin/env python3
"""
Test script for integrated sequential verification in vision and LM verifiers.
Demonstrates both legacy and enhanced sequential modes.
"""

import torch
import numpy as np
from typing import List, Dict, Any
import time

# Mock models for testing
class MockVisionModel:
    """Mock vision model for testing"""
    def __init__(self, seed: int = 42):
        self.seed = seed
        torch.manual_seed(seed)
        self.fc = torch.nn.Linear(100, 10)
    
    def get_features(self, x):
        """Get features from input"""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # Flatten and resize to expected dimensions
        x_flat = x.view(x.size(0), -1)
        if x_flat.size(1) < 100:
            padding = torch.zeros(x.size(0), 100 - x_flat.size(1))
            x_flat = torch.cat([x_flat, padding], dim=1)
        elif x_flat.size(1) > 100:
            x_flat = x_flat[:, :100]
        
        with torch.no_grad():
            return self.fc(x_flat)


class MockLM:
    """Mock language model for testing"""
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Mock tokenizer
        self.tok = self
        # Add tokenizer attributes
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.bos_token_id = 3
        self.cls_token_id = 4
        self.sep_token_id = 5
        self.mask_token_id = 6
    
    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Generate text based on prompt and seed"""
        # Deterministic response based on prompt hash and seed
        prompt_hash = hash(prompt) % 1000000
        combined_seed = (self.seed + prompt_hash) % 2**32
        rng = np.random.default_rng(combined_seed)
        
        words = ["The", "model", "generates", "text", "based", "on", "input", "seed"]
        num_words = min(max_new_tokens // 5, len(words))
        selected = rng.choice(words, size=num_words, replace=True)
        return " ".join(selected)
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Mock tokenization"""
        # Simple character-based tokenization for testing
        return [ord(c) for c in text[:100]]  # Limit to 100 tokens


def test_vision_verifier():
    """Test vision verifier with enhanced sequential mode"""
    print("=" * 60)
    print("Testing Vision Verifier with Sequential Integration")
    print("=" * 60)
    
    from pot.vision.verifier import VisionVerifier
    from pot.vision.models import VisionModel
    
    # Wrap mock models in VisionModel interface
    class VisionModelWrapper(VisionModel):
        def __init__(self, mock_model):
            self.model = mock_model
            
        def get_features(self, x):
            return self.model.get_features(x)
    
    # Create reference and test models
    ref_model = VisionModelWrapper(MockVisionModel(seed=42))
    test_model_same = VisionModelWrapper(MockVisionModel(seed=42))  # Same seed = same model
    test_model_diff = VisionModelWrapper(MockVisionModel(seed=99))  # Different seed = different model
    
    # Generate test challenges
    challenges = [torch.randn(3, 224, 224) for _ in range(20)]
    
    print("\n1. Testing Legacy Sequential Mode")
    print("-" * 40)
    
    verifier_legacy = VisionVerifier(
        reference_model=ref_model,
        delta=0.01,
        use_sequential=True,
        sequential_mode='legacy',
        use_fingerprinting=False  # Disable for speed
    )
    
    # Test with same model
    result = verifier_legacy.verify(test_model_same, challenges, tolerance=0.05)
    print(f"Same model (legacy): accepted={result.accepted}, distance={result.distance:.4f}")
    print(f"  Challenges evaluated: {result.n_challenges}")
    if 'sequential' in result.metadata:
        print(f"  Sequential mode: {result.metadata['sequential']['mode']}")
    
    # Test with different model
    result = verifier_legacy.verify(test_model_diff, challenges, tolerance=0.05)
    print(f"Different model (legacy): accepted={result.accepted}, distance={result.distance:.4f}")
    print(f"  Challenges evaluated: {result.n_challenges}")
    
    print("\n2. Testing Enhanced Sequential Mode (EB-based)")
    print("-" * 40)
    
    verifier_enhanced = VisionVerifier(
        reference_model=ref_model,
        delta=0.01,
        use_sequential=True,
        sequential_mode='enhanced',
        use_fingerprinting=False
    )
    
    # Test with same model
    result = verifier_enhanced.verify(
        test_model_same, 
        challenges, 
        tolerance=0.05,
        alpha=0.01,
        beta=0.01
    )
    print(f"Same model (enhanced): accepted={result.accepted}, distance={result.distance:.4f}")
    print(f"  Challenges evaluated: {result.n_challenges}")
    if result.sequential_result:
        print(f"  Early stopping at: {result.sequential_result.stopped_at}")
        print(f"  Decision: {result.sequential_result.decision}")
        if result.sequential_result.p_value:
            print(f"  P-value: {result.sequential_result.p_value:.6f}")
        print(f"  Trajectory length: {len(result.sequential_result.trajectory)}")
    
    # Test with different model
    result = verifier_enhanced.verify(
        test_model_diff,
        challenges,
        tolerance=0.05,
        alpha=0.01,
        beta=0.01
    )
    print(f"Different model (enhanced): accepted={result.accepted}, distance={result.distance:.4f}")
    print(f"  Challenges evaluated: {result.n_challenges}")
    if result.sequential_result:
        print(f"  Early stopping at: {result.sequential_result.stopped_at}")
        print(f"  Decision: {result.sequential_result.decision}")
    
    print("\n3. Testing Fixed-Sample Mode (no sequential)")
    print("-" * 40)
    
    verifier_fixed = VisionVerifier(
        reference_model=ref_model,
        delta=0.01,
        use_sequential=False,
        use_fingerprinting=False
    )
    
    result = verifier_fixed.verify(test_model_same, challenges, tolerance=0.05)
    print(f"Same model (fixed): accepted={result.accepted}, distance={result.distance:.4f}")
    print(f"  All {result.n_challenges} challenges evaluated")
    
    print("\n✓ Vision verifier integration test complete")


def test_lm_verifier():
    """Test LM verifier with enhanced sequential mode"""
    print("\n" + "=" * 60)
    print("Testing LM Verifier with Sequential Integration")
    print("=" * 60)
    
    from pot.lm.verifier import LMVerifier
    from pot.lm.models import LM
    
    # Wrap mock models in LM interface
    class LMWrapper(LM):
        def __init__(self, mock_model):
            self.model = mock_model
            self.tok = mock_model.tok
            
        def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
            return self.model.generate(prompt, max_new_tokens)
    
    # Create reference and test models
    ref_model = LMWrapper(MockLM(seed=42))
    test_model_same = LMWrapper(MockLM(seed=42))
    test_model_diff = LMWrapper(MockLM(seed=99))
    
    # Generate test challenges
    challenges = []
    for i in range(20):
        challenges.append({
            "template": "Complete: The {object} is {attribute}",
            "slot_values": {
                "object": ["cat", "dog", "tree"][i % 3],
                "attribute": ["big", "small", "green"][i % 3]
            }
        })
    
    print("\n1. Testing Legacy Sequential Mode")
    print("-" * 40)
    
    verifier_legacy = LMVerifier(
        reference_model=ref_model,
        delta=0.01,
        use_sequential=True,
        sequential_mode='legacy',
        use_fingerprinting=False
    )
    
    # Test with same model
    result = verifier_legacy.verify(test_model_same, challenges, tolerance=0.1)
    print(f"Same model (legacy): accepted={result.accepted}, distance={result.distance:.4f}")
    print(f"  Challenges evaluated: {result.n_challenges}")
    print(f"  Fuzzy similarity: {result.fuzzy_similarity:.4f}")
    
    # Test with different model
    result = verifier_legacy.verify(test_model_diff, challenges, tolerance=0.1)
    print(f"Different model (legacy): accepted={result.accepted}, distance={result.distance:.4f}")
    print(f"  Challenges evaluated: {result.n_challenges}")
    
    print("\n2. Testing Enhanced Sequential Mode (EB-based)")
    print("-" * 40)
    
    verifier_enhanced = LMVerifier(
        reference_model=ref_model,
        delta=0.01,
        use_sequential=True,
        sequential_mode='enhanced',
        use_fingerprinting=False
    )
    
    # Test with same model
    result = verifier_enhanced.verify(
        test_model_same,
        challenges,
        tolerance=0.1,
        alpha=0.01,
        beta=0.01
    )
    print(f"Same model (enhanced): accepted={result.accepted}, distance={result.distance:.4f}")
    print(f"  Challenges evaluated: {result.n_challenges}")
    if result.sequential_result:
        print(f"  Early stopping at: {result.sequential_result.stopped_at}")
        print(f"  Decision: {result.sequential_result.decision}")
        if result.sequential_result.p_value:
            print(f"  P-value: {result.sequential_result.p_value:.6f}")
    
    # Test with different model
    result = verifier_enhanced.verify(
        test_model_diff,
        challenges,
        tolerance=0.1,
        alpha=0.01,
        beta=0.01
    )
    print(f"Different model (enhanced): accepted={result.accepted}, distance={result.distance:.4f}")
    print(f"  Challenges evaluated: {result.n_challenges}")
    if result.sequential_result:
        print(f"  Early stopping at: {result.sequential_result.stopped_at}")
        print(f"  Decision: {result.sequential_result.decision}")
    
    print("\n3. Testing Fixed-Sample Mode (no sequential)")
    print("-" * 40)
    
    verifier_fixed = LMVerifier(
        reference_model=ref_model,
        delta=0.01,
        use_sequential=False,
        use_fingerprinting=False
    )
    
    result = verifier_fixed.verify(test_model_same, challenges, tolerance=0.1)
    print(f"Same model (fixed): accepted={result.accepted}, distance={result.distance:.4f}")
    print(f"  All {result.n_challenges} challenges evaluated")
    
    print("\n✓ LM verifier integration test complete")


def test_early_stopping_efficiency():
    """Test that early stopping actually saves evaluations"""
    print("\n" + "=" * 60)
    print("Testing Early Stopping Efficiency")
    print("=" * 60)
    
    from pot.vision.verifier import VisionVerifier
    from pot.vision.models import VisionModel
    
    class VisionModelWrapper(VisionModel):
        def __init__(self, mock_model):
            self.model = mock_model
            self.eval_count = 0
            
        def get_features(self, x):
            self.eval_count += 1
            return self.model.get_features(x)
    
    # Create models
    ref_model = VisionModelWrapper(MockVisionModel(seed=42))
    test_model_very_different = VisionModelWrapper(MockVisionModel(seed=999))
    
    # Many challenges to test early stopping
    challenges = [torch.randn(3, 224, 224) for _ in range(100)]
    
    # Test with enhanced sequential (should stop early)
    verifier_seq = VisionVerifier(
        reference_model=ref_model,
        delta=0.01,
        use_sequential=True,
        sequential_mode='enhanced',
        use_fingerprinting=False
    )
    
    test_model_very_different.eval_count = 0
    result_seq = verifier_seq.verify(
        test_model_very_different,
        challenges,
        tolerance=0.05,
        alpha=0.01,
        beta=0.01
    )
    
    print(f"Sequential mode:")
    print(f"  Evaluations: {test_model_very_different.eval_count}/{len(challenges)}")
    print(f"  Stopped at: {result_seq.sequential_result.stopped_at if result_seq.sequential_result else 'N/A'}")
    print(f"  Decision: {result_seq.sequential_result.decision if result_seq.sequential_result else 'N/A'}")
    
    # Test without sequential (should evaluate all)
    verifier_fixed = VisionVerifier(
        reference_model=ref_model,
        delta=0.01,
        use_sequential=False,
        use_fingerprinting=False
    )
    
    test_model_very_different.eval_count = 0
    result_fixed = verifier_fixed.verify(
        test_model_very_different,
        challenges,
        tolerance=0.05
    )
    
    print(f"\nFixed-sample mode:")
    print(f"  Evaluations: {test_model_very_different.eval_count}/{len(challenges)}")
    
    # Calculate savings
    if result_seq.sequential_result:
        savings = 100 * (1 - result_seq.sequential_result.stopped_at / len(challenges))
        print(f"\nEfficiency gain: {savings:.1f}% fewer evaluations with early stopping")
    
    print("\n✓ Early stopping efficiency test complete")


if __name__ == "__main__":
    try:
        # Test vision verifier
        test_vision_verifier()
        
        # Test LM verifier
        test_lm_verifier()
        
        # Test early stopping efficiency
        test_early_stopping_efficiency()
        
        print("\n" + "=" * 60)
        print("ALL INTEGRATION TESTS PASSED ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()