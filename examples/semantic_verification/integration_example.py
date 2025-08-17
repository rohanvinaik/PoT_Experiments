#!/usr/bin/env python3
"""
Integration example showing how to use semantic verification with LM and Vision verifiers.
Demonstrates enhanced verification with semantic analysis.
"""

import torch
import numpy as np
from typing import Optional

# Import semantic components
from pot.semantic import (
    ConceptLibrary,
    SemanticVerificationConfig,
    create_semantic_components,
    integrate_with_verifier
)

# Import verifiers
from pot.lm.verifier import LMVerifier
from pot.lm.models import LM
from pot.vision.verifier import VisionVerifier
from pot.vision.models import VisionModel


# Mock models for demonstration
class DemoTokenizer:
    """Demo tokenizer for LM examples."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.bos_token_id = 3
        self.sep_token_id = 4
        self.cls_token_id = 5
    
    def encode(self, text: str, add_special_tokens: bool = True):
        return [ord(c) % 100 for c in text[:100]]
    
    def decode(self, token_ids):
        return ''.join([chr(t + 32) for t in token_ids[:100]])


class DemoLM(LM):
    """Demo language model for examples."""
    def __init__(self, behavior_type='normal'):
        self.tok = DemoTokenizer()
        self.behavior_type = behavior_type
    
    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        if self.behavior_type == 'normal':
            return f"Normal response to: {prompt[:30]}..."
        elif self.behavior_type == 'shifted':
            return f"SHIFTED RESPONSE: {prompt[:30].upper()}!!!"
        else:
            return f"Anomalous output ### {prompt[:20]} ###"
    
    def get_hidden_states(self, prompt: str) -> Optional[torch.Tensor]:
        # Return mock hidden states based on behavior
        if self.behavior_type == 'normal':
            return torch.randn(1, 768) * 0.5
        elif self.behavior_type == 'shifted':
            return torch.randn(1, 768) * 0.5 + torch.ones(768) * 2.0
        else:
            return torch.randn(1, 768) * 2.0 + torch.ones(768) * 5.0


class DemoVisionModel(VisionModel):
    """Demo vision model for examples."""
    def __init__(self, behavior_type='normal'):
        self.behavior_type = behavior_type
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0] if x.dim() > 3 else 1
        
        if self.behavior_type == 'normal':
            return torch.randn(batch_size, 512) * 0.5
        elif self.behavior_type == 'shifted':
            return torch.randn(batch_size, 512) * 0.5 + torch.ones(512) * 1.5
        else:
            return torch.randn(batch_size, 512) * 2.0 + torch.ones(512) * 4.0


def setup_semantic_library_lm():
    """Set up semantic library for language model verification."""
    print("Setting up semantic library for LM verification...")
    
    # Create library for LM embeddings
    library = ConceptLibrary(dim=768, method='gaussian')
    
    # Add semantic concepts from "training data"
    # In practice, these would come from actual model training
    
    # Concept: Normal language patterns
    normal_embeddings = torch.randn(100, 768) * 0.5
    library.add_concept('normal_language', normal_embeddings)
    
    # Concept: Formal language
    formal_embeddings = torch.randn(80, 768) * 0.5 + torch.ones(768) * 0.5
    library.add_concept('formal_language', formal_embeddings)
    
    # Concept: Informal language
    informal_embeddings = torch.randn(80, 768) * 0.5 - torch.ones(768) * 0.5
    library.add_concept('informal_language', informal_embeddings)
    
    print(f"  ✓ Created library with {len(library.list_concepts())} concepts")
    return library


def setup_semantic_library_vision():
    """Set up semantic library for vision model verification."""
    print("Setting up semantic library for vision verification...")
    
    # Create library for vision features
    library = ConceptLibrary(dim=512, method='gaussian')
    
    # Add visual concepts
    # In practice, these would be extracted from training images
    
    # Concept: Object class A
    class_a_features = torch.randn(100, 512) * 0.5
    library.add_concept('object_class_a', class_a_features)
    
    # Concept: Object class B
    class_b_features = torch.randn(100, 512) * 0.5 + torch.ones(512) * 1.0
    library.add_concept('object_class_b', class_b_features)
    
    # Concept: Background
    background_features = torch.randn(100, 512) * 0.5 - torch.ones(512) * 0.5
    library.add_concept('background', background_features)
    
    print(f"  ✓ Created library with {len(library.list_concepts())} concepts")
    return library


def lm_verification_example():
    """Example of LM verification with semantic analysis."""
    print("\n" + "=" * 70)
    print("LANGUAGE MODEL VERIFICATION WITH SEMANTIC ANALYSIS")
    print("=" * 70)
    
    # Set up semantic library
    semantic_library = setup_semantic_library_lm()
    
    # Create reference model
    reference_model = DemoLM(behavior_type='normal')
    
    # Create verifier with semantic verification
    print("\nCreating LM verifier with semantic verification...")
    verifier = LMVerifier(
        reference_model=reference_model,
        delta=0.01,
        semantic_library=semantic_library,
        semantic_weight=0.3  # 30% weight for semantic score
    )
    print(f"  ✓ Verifier created with semantic_weight={verifier.semantic_weight}")
    
    # Generate test challenges
    challenges = [
        {"template": "Complete the sentence: The weather is", "slot_values": {}},
        {"template": "Answer the question: What is {concept}?", 
         "slot_values": {"concept": "machine learning"}},
        {"template": "Translate to {language}: {text}",
         "slot_values": {"language": "Spanish", "text": "Hello"}},
    ]
    
    print(f"\nTesting with {len(challenges)} challenges...")
    
    # Test 1: Normal model (should pass)
    print("\n1. Testing normal model:")
    normal_model = DemoLM(behavior_type='normal')
    result_normal = verifier.verify(normal_model, challenges, tolerance=0.5)
    
    print(f"   - Distance: {result_normal.distance:.3f}")
    print(f"   - Semantic score: {result_normal.semantic_score:.3f}" if result_normal.semantic_score else "   - Semantic score: N/A")
    print(f"   - Combined score: {result_normal.combined_score:.3f}" if result_normal.combined_score else "   - Combined score: N/A")
    print(f"   - Accepted: {result_normal.accepted}")
    
    # Test 2: Shifted model (might fail)
    print("\n2. Testing shifted model:")
    shifted_model = DemoLM(behavior_type='shifted')
    result_shifted = verifier.verify(shifted_model, challenges, tolerance=0.5)
    
    print(f"   - Distance: {result_shifted.distance:.3f}")
    print(f"   - Semantic score: {result_shifted.semantic_score:.3f}" if result_shifted.semantic_score else "   - Semantic score: N/A")
    print(f"   - Combined score: {result_shifted.combined_score:.3f}" if result_shifted.combined_score else "   - Combined score: N/A")
    print(f"   - Accepted: {result_shifted.accepted}")
    
    # Test 3: Anomalous model (should fail)
    print("\n3. Testing anomalous model:")
    anomalous_model = DemoLM(behavior_type='anomalous')
    result_anomalous = verifier.verify(anomalous_model, challenges, tolerance=0.5)
    
    print(f"   - Distance: {result_anomalous.distance:.3f}")
    print(f"   - Semantic score: {result_anomalous.semantic_score:.3f}" if result_anomalous.semantic_score else "   - Semantic score: N/A")
    print(f"   - Combined score: {result_anomalous.combined_score:.3f}" if result_anomalous.combined_score else "   - Combined score: N/A")
    print(f"   - Accepted: {result_anomalous.accepted}")
    
    # Analysis
    print("\n" + "-" * 50)
    print("Analysis:")
    if result_normal.accepted and not result_anomalous.accepted:
        print("  ✓ Semantic verification correctly distinguished normal from anomalous")
    else:
        print("  ⚠️  Results may vary due to randomness in demo models")


def vision_verification_example():
    """Example of vision model verification with semantic analysis."""
    print("\n" + "=" * 70)
    print("VISION MODEL VERIFICATION WITH SEMANTIC ANALYSIS")
    print("=" * 70)
    
    # Set up semantic library
    semantic_library = setup_semantic_library_vision()
    
    # Create reference model
    reference_model = DemoVisionModel(behavior_type='normal')
    
    # Create verifier with semantic verification
    print("\nCreating Vision verifier with semantic verification...")
    verifier = VisionVerifier(
        reference_model=reference_model,
        delta=0.01,
        semantic_library=semantic_library,
        semantic_weight=0.3
    )
    print(f"  ✓ Verifier created with semantic_weight={verifier.semantic_weight}")
    
    # Generate test challenges (mock images)
    challenges = [torch.randn(3, 224, 224) for _ in range(5)]
    print(f"\nTesting with {len(challenges)} image challenges...")
    
    # Test different models
    model_types = ['normal', 'shifted', 'anomalous']
    results = {}
    
    for model_type in model_types:
        print(f"\nTesting {model_type} model:")
        test_model = DemoVisionModel(behavior_type=model_type)
        result = verifier.verify(test_model, challenges, tolerance=0.5)
        results[model_type] = result
        
        print(f"   - Distance: {result.distance:.3f}")
        print(f"   - Semantic score: {result.semantic_score:.3f}" if result.semantic_score else "   - Semantic score: N/A")
        print(f"   - Combined score: {result.combined_score:.3f}" if result.combined_score else "   - Combined score: N/A")
        print(f"   - Accepted: {result.accepted}")
    
    # Analysis
    print("\n" + "-" * 50)
    print("Analysis:")
    if results['normal'].accepted and not results['anomalous'].accepted:
        print("  ✓ Semantic verification correctly distinguished normal from anomalous")
    else:
        print("  ⚠️  Results may vary due to randomness in demo models")


def configuration_example():
    """Example of using configuration for semantic verification."""
    print("\n" + "=" * 70)
    print("CONFIGURATION-BASED SEMANTIC VERIFICATION")
    print("=" * 70)
    
    # Create configuration
    print("\nCreating semantic verification configuration...")
    config = SemanticVerificationConfig(
        enabled=True,
        semantic_weight=0.4,
        library_method='gaussian',
        library_dimension=768,
        matching_threshold=0.7,
        matching_primary_method='cosine',
        lm_enabled=True,
        lm_semantic_weight=0.35,
        vision_enabled=True,
        vision_semantic_weight=0.3
    )
    
    print("Configuration settings:")
    print(f"  - Enabled: {config.enabled}")
    print(f"  - Library method: {config.library_method}")
    print(f"  - Library dimension: {config.library_dimension}")
    print(f"  - Matching threshold: {config.matching_threshold}")
    print(f"  - LM weight: {config.lm_semantic_weight}")
    print(f"  - Vision weight: {config.vision_semantic_weight}")
    
    # Validate configuration
    try:
        config.validate()
        print("  ✓ Configuration is valid")
    except ValueError as e:
        print(f"  ✗ Configuration error: {e}")
    
    # Create semantic components from config
    print("\nCreating semantic components from configuration...")
    library, matcher = create_semantic_components(config)
    
    if library and matcher:
        print(f"  ✓ Created library (dim={library.dim}) and matcher (threshold={matcher.threshold})")
        
        # Add some concepts
        library.add_concept('config_concept', torch.randn(10, config.library_dimension))
        print(f"  ✓ Added concept to library")
    else:
        print("  ⚠️  Components not created (may be disabled in config)")
    
    # Integrate with existing verifier
    print("\nIntegrating with existing verifier...")
    ref_model = DemoLM()
    verifier = LMVerifier(ref_model)  # Create without semantic
    
    print(f"  Before integration: semantic_library = {verifier.semantic_library}")
    
    integrate_with_verifier('lm', verifier, config)
    
    print(f"  After integration: semantic_library = {verifier.semantic_library is not None}")
    if verifier.semantic_library:
        print(f"  After integration: semantic_weight = {verifier.semantic_weight}")


def semantic_weight_comparison():
    """Compare verification with different semantic weights."""
    print("\n" + "=" * 70)
    print("SEMANTIC WEIGHT COMPARISON")
    print("=" * 70)
    
    # Set up
    semantic_library = setup_semantic_library_lm()
    reference_model = DemoLM(behavior_type='normal')
    test_model = DemoLM(behavior_type='shifted')
    
    challenges = [
        {"template": "Test prompt {i}", "slot_values": {"i": str(i)}}
        for i in range(3)
    ]
    
    print("\nTesting shifted model with different semantic weights...")
    print("-" * 50)
    
    weights = [0.0, 0.3, 0.5, 0.7, 1.0]
    results = []
    
    for weight in weights:
        # Create verifier with specific weight
        if weight == 0.0:
            # No semantic verification
            verifier = LMVerifier(reference_model=reference_model)
        else:
            verifier = LMVerifier(
                reference_model=reference_model,
                semantic_library=semantic_library,
                semantic_weight=weight
            )
        
        # Verify
        result = verifier.verify(test_model, challenges, tolerance=0.5)
        results.append((weight, result))
        
        print(f"Weight {weight:.1f}:")
        print(f"  - Distance: {result.distance:.3f}")
        if result.semantic_score is not None:
            print(f"  - Semantic: {result.semantic_score:.3f}")
            print(f"  - Combined: {result.combined_score:.3f}")
        print(f"  - Accepted: {result.accepted}")
        print()
    
    # Analysis
    print("-" * 50)
    print("Analysis:")
    print("  Higher semantic weight gives more importance to semantic similarity")
    print("  Lower semantic weight relies more on distance metrics")
    print("  Choose weight based on your verification requirements")


def main():
    """Run all integration examples."""
    print("\n" + "=" * 80)
    print(" " * 25 + "SEMANTIC VERIFICATION INTEGRATION EXAMPLES")
    print("=" * 80)
    
    # 1. LM verification with semantic
    lm_verification_example()
    
    # 2. Vision verification with semantic
    vision_verification_example()
    
    # 3. Configuration-based setup
    configuration_example()
    
    # 4. Semantic weight comparison
    semantic_weight_comparison()
    
    print("\n" + "=" * 80)
    print(" " * 30 + "INTEGRATION EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nThese examples demonstrate how to integrate semantic verification")
    print("with existing LM and Vision verifiers for enhanced verification.")
    print("\nFor drift detection examples, see drift_detection.py")


if __name__ == "__main__":
    main()