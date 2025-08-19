"""
Example: Statistical Difference Testing for Model Verification

This example demonstrates how to use the statistical difference decision framework
to verify whether two models are statistically different, identical, or equivalent.
"""

import numpy as np
import torch
import time
from pathlib import Path
from typing import Tuple, Optional

# Import the framework
from pot.core.diff_decision import (
    DiffDecisionConfig,
    DifferenceVerifier,
    create_default_verifier
)

# Import PoT components if available
try:
    from pot.core.prf import PRFKeyDerivation
except ImportError:
    # Create a simple PRF replacement for the demo
    class PRFKeyDerivation:
        def __init__(self, master_key):
            self.master_key = master_key
            
        def derive_int(self, namespace, index, max_val):
            import hashlib
            data = f"{namespace}:{index}".encode()
            h = hashlib.sha256(self.master_key + data).digest()
            return int.from_bytes(h[:4], 'big') % max_val

# ===============================================================================
# SCORING FUNCTIONS
# ===============================================================================

def compute_delta_ce_score(ref_model, cand_model, prompt: str, K: int = 32) -> float:
    """Compute delta cross-entropy score between models
    
    Args:
        ref_model: Reference model
        cand_model: Candidate model
        prompt: Input prompt
        K: Number of positions to score
        
    Returns:
        Average score difference across K positions
    """
    # In practice, this would:
    # 1. Tokenize the prompt
    # 2. Get logits from both models for K positions
    # 3. Compute cross-entropy difference
    # 4. Return average score
    
    # For demonstration, simulate scoring
    if hasattr(ref_model, 'score_offset') and hasattr(cand_model, 'score_offset'):
        # Use model-specific offsets if available
        base_score = ref_model.score_offset - cand_model.score_offset
    else:
        base_score = 0.0
    
    # Add position-dependent noise
    scores = []
    for _ in range(K):
        score = base_score + np.random.normal(0, 0.01)
        scores.append(score)
    
    return np.mean(scores)

def compute_symmetric_kl_score(ref_model, cand_model, prompt: str, K: int = 32) -> float:
    """Compute symmetric KL divergence score between models
    
    Returns average symmetric KL divergence across K positions
    """
    # Similar to delta_ce but uses symmetric KL
    base_score = 0.0
    if hasattr(ref_model, 'score_offset') and hasattr(cand_model, 'score_offset'):
        base_score = abs(ref_model.score_offset - cand_model.score_offset) * 2
    
    scores = []
    for _ in range(K):
        score = base_score + np.random.normal(0, 0.015)
        scores.append(max(0, score))  # KL is non-negative
    
    return np.mean(scores)

# ===============================================================================
# PROMPT GENERATION
# ===============================================================================

class DeterministicPromptGenerator:
    """Generate deterministic prompts using PRF"""
    
    def __init__(self, master_key: bytes, seed: int = 42):
        self.prf = PRFKeyDerivation(master_key)
        self.counter = 0
        self.seed = seed
        
        # Template prompts
        self.templates = [
            "Explain the concept of {} in simple terms.",
            "What are the main differences between {} and {}?",
            "How does {} work in practice?",
            "Provide a detailed analysis of {}.",
            "What are the implications of {} for {}?",
            "Describe the historical development of {}.",
            "Compare and contrast {} with {}.",
            "What are the key challenges in {}?",
        ]
        
        self.topics = [
            "machine learning", "quantum computing", "climate change",
            "artificial intelligence", "renewable energy", "genetics",
            "blockchain", "neural networks", "evolution", "democracy"
        ]
    
    def generate(self) -> str:
        """Generate next deterministic prompt"""
        # Use PRF to select template and topics
        template_idx = self.prf.derive_int(
            namespace="template",
            index=self.counter,
            max_val=len(self.templates)
        )
        
        topic1_idx = self.prf.derive_int(
            namespace="topic1",
            index=self.counter,
            max_val=len(self.topics)
        )
        
        topic2_idx = self.prf.derive_int(
            namespace="topic2",
            index=self.counter,
            max_val=len(self.topics)
        )
        
        template = self.templates[template_idx]
        topic1 = self.topics[topic1_idx]
        topic2 = self.topics[topic2_idx]
        
        # Format template
        if "{}" in template and "{}" in template.replace("{}", "X", 1):
            # Two placeholders
            prompt = template.format(topic1, topic2)
        else:
            # One placeholder
            prompt = template.format(topic1)
        
        self.counter += 1
        return prompt

# ===============================================================================
# MODEL EXAMPLES
# ===============================================================================

class MockModel:
    """Mock model for testing"""
    
    def __init__(self, name: str, score_offset: float = 0.0, num_params: int = 1000000):
        self.name = name
        self.score_offset = score_offset
        self.num_params = num_params
        self.config = type('Config', (), {'model_type': 'mock'})()
    
    def num_parameters(self) -> int:
        return self.num_params
    
    def __repr__(self):
        return f"MockModel(name={self.name}, offset={self.score_offset})"

# ===============================================================================
# VERIFICATION SCENARIOS
# ===============================================================================

def test_identical_models():
    """Test verification of identical models"""
    print("\n" + "="*60)
    print("SCENARIO 1: Identical Models")
    print("="*60)
    
    # Create identical models
    ref_model = MockModel("model_v1", score_offset=0.0)
    cand_model = MockModel("model_v1_copy", score_offset=0.0)
    
    # Configure for identical model detection
    config = DiffDecisionConfig(
        n_min=10,
        n_max=50,
        identical_model_n_min=10,
        early_stop_threshold=0.001,
        rel_margin_target=0.05,
        method="eb"
    )
    
    # Create verifier
    prompt_gen = DeterministicPromptGenerator(b"test_key")
    verifier = DifferenceVerifier(
        compute_delta_ce_score,
        prompt_gen.generate,
        config
    )
    
    # Run verification
    report = verifier.verify_difference(ref_model, cand_model)
    
    # Display results
    print(f"Decision: {report['results']['decision']}")
    print(f"Samples used: {report['results']['n_used']}")
    print(f"Mean difference: {report['results']['mean']:.6f}")
    print(f"99% CI: [{report['results']['ci_99'][0]:.6f}, {report['results']['ci_99'][1]:.6f}]")
    print(f"\n{report['interpretation']}")
    
    return report

def test_different_models():
    """Test verification of clearly different models"""
    print("\n" + "="*60)
    print("SCENARIO 2: Different Models")
    print("="*60)
    
    # Create different models
    ref_model = MockModel("model_v1", score_offset=0.0)
    cand_model = MockModel("model_v2", score_offset=0.05)  # Clear difference
    
    # Configure for difference detection
    config = DiffDecisionConfig(
        n_min=15,
        n_max=100,
        rel_margin_target=0.1,
        method="t",
        batch_size=5
    )
    
    # Create verifier
    prompt_gen = DeterministicPromptGenerator(b"test_key_2")
    verifier = DifferenceVerifier(
        compute_delta_ce_score,
        prompt_gen.generate,
        config
    )
    
    # Run verification
    report = verifier.verify_difference(ref_model, cand_model)
    
    # Display results
    print(f"Decision: {report['results']['decision']}")
    print(f"Samples used: {report['results']['n_used']}")
    print(f"Mean difference: {report['results']['mean']:.6f}")
    print(f"99% CI: [{report['results']['ci_99'][0]:.6f}, {report['results']['ci_99'][1]:.6f}]")
    print(f"Relative margin: {report['results']['rel_me']:.3f}")
    print(f"\n{report['interpretation']}")
    
    return report

def test_equivalent_models():
    """Test verification with equivalence band (TOST)"""
    print("\n" + "="*60)
    print("SCENARIO 3: Equivalent Models (within tolerance)")
    print("="*60)
    
    # Create slightly different models
    ref_model = MockModel("model_v1", score_offset=0.0)
    cand_model = MockModel("model_v1_quantized", score_offset=0.008)  # Small difference
    
    # Configure with equivalence band
    config = DiffDecisionConfig(
        n_min=20,
        n_max=150,
        equivalence_band=0.02,  # Consider equivalent if difference < 0.02
        rel_margin_target=0.05,
        method="eb",
        clip_high=0.1  # Adjust for scale
    )
    
    # Create verifier
    prompt_gen = DeterministicPromptGenerator(b"test_key_3")
    verifier = DifferenceVerifier(
        compute_delta_ce_score,
        prompt_gen.generate,
        config
    )
    
    # Run verification
    report = verifier.verify_difference(ref_model, cand_model)
    
    # Display results
    print(f"Decision: {report['results']['decision']}")
    print(f"Samples used: {report['results']['n_used']}")
    print(f"Mean difference: {report['results']['mean']:.6f}")
    print(f"99% CI: [{report['results']['ci_99'][0]:.6f}, {report['results']['ci_99'][1]:.6f}]")
    print(f"Equivalence band: ±{config.equivalence_band}")
    print(f"\n{report['interpretation']}")
    
    return report

def test_high_variance_models():
    """Test verification with high variance (challenging case)"""
    print("\n" + "="*60)
    print("SCENARIO 4: High Variance Models (challenging)")
    print("="*60)
    
    # Create models with high variance scoring
    class HighVarianceModel(MockModel):
        def __init__(self, name, offset):
            super().__init__(name, offset)
            self.variance_multiplier = 5.0
    
    ref_model = HighVarianceModel("model_noisy_v1", 0.0)
    cand_model = HighVarianceModel("model_noisy_v2", 0.02)
    
    # Configure with larger sample size for high variance
    config = DiffDecisionConfig(
        n_min=30,
        n_max=200,
        rel_margin_target=0.15,  # More lenient due to variance
        method="t",
        batch_size=10
    )
    
    # Modified scoring with higher variance
    def high_variance_score(ref, cand, prompt, K=32):
        base = compute_delta_ce_score(ref, cand, prompt, K)
        # Add extra noise
        return base + np.random.normal(0, 0.03)
    
    # Create verifier
    prompt_gen = DeterministicPromptGenerator(b"test_key_4")
    verifier = DifferenceVerifier(
        high_variance_score,
        prompt_gen.generate,
        config
    )
    
    # Run verification
    report = verifier.verify_difference(ref_model, cand_model)
    
    # Display results
    print(f"Decision: {report['results']['decision']}")
    print(f"Samples used: {report['results']['n_used']}")
    print(f"Mean difference: {report['results']['mean']:.6f}")
    print(f"Standard deviation: {report['results']['std_dev']:.6f}")
    print(f"99% CI: [{report['results']['ci_99'][0]:.6f}, {report['results']['ci_99'][1]:.6f}]")
    print(f"Relative margin: {report['results']['rel_me']:.3f}")
    print(f"\n{report['interpretation']}")
    
    return report

def test_performance_analysis():
    """Analyze performance characteristics"""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Test with different configurations
    configs = [
        ("Fast (small n)", DiffDecisionConfig(n_min=5, n_max=20, batch_size=5)),
        ("Standard", DiffDecisionConfig(n_min=20, n_max=100, batch_size=10)),
        ("Thorough", DiffDecisionConfig(n_min=50, n_max=300, batch_size=20)),
    ]
    
    ref_model = MockModel("reference", 0.0)
    cand_model = MockModel("candidate", 0.03)
    
    for name, config in configs:
        prompt_gen = DeterministicPromptGenerator(b"perf_test")
        verifier = DifferenceVerifier(
            compute_delta_ce_score,
            prompt_gen.generate,
            config
        )
        
        start_time = time.perf_counter()
        report = verifier.verify_difference(ref_model, cand_model, verbose=False)
        elapsed = time.perf_counter() - start_time
        
        print(f"\n{name}:")
        print(f"  Decision: {report['results']['decision']}")
        print(f"  Samples: {report['results']['n_used']}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Throughput: {report['results']['n_used']/elapsed:.1f} samples/sec")

# ===============================================================================
# MAIN
# ===============================================================================

def main():
    """Run all verification scenarios"""
    print("\n" + "="*80)
    print(" STATISTICAL DIFFERENCE TESTING FRAMEWORK DEMONSTRATION")
    print("="*80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run scenarios
    results = {}
    
    print("\nRunning verification scenarios...")
    
    # Test identical models
    results['identical'] = test_identical_models()
    
    # Test different models
    results['different'] = test_different_models()
    
    # Test equivalent models
    results['equivalent'] = test_equivalent_models()
    
    # Test high variance
    results['high_variance'] = test_high_variance_models()
    
    # Performance analysis
    test_performance_analysis()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for scenario, report in results.items():
        decision = report['results']['decision']
        n_used = report['results']['n_used']
        mean = report['results']['mean']
        print(f"\n{scenario.upper()}:")
        print(f"  Decision: {decision}")
        print(f"  Samples: {n_used}")
        print(f"  Mean difference: {mean:.6f}")
    
    print("\n" + "="*80)
    print("Demonstration complete!")
    
    return results

if __name__ == "__main__":
    results = main()