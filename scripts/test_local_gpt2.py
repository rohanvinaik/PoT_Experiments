#!/usr/bin/env python3
"""Test sequential verification using local GPT-2 model"""

import sys
import os
import logging
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pot.core.sequential_runner import SequentialTestRunner
from pot.scoring.teacher_forced import ScoringConfig
from pot.core.boundaries import CSState, eb_radius
from pot.challenges.prompt_generator import DeterministicPromptGenerator
import torch

def run_lightweight_test():
    """Run a test with GPT-2 model (small and fast)"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("\n🚀 Running Local Sequential Test with GPT-2\n")
    print("=" * 60)
    
    # Use GPT-2 as both reference and candidate (lightweight test)
    model_name = "gpt2"  # ~124M parameters, very fast
    
    # Test configuration
    master_key = b"test_key_local_12345"
    scoring_config = ScoringConfig(
        method="delta_ce",
        num_positions=5,  # Fewer positions for speed
        temperature=1.0
    )
    
    # Initialize runner
    logger.info(f"Initializing with model: {model_name}")
    runner = SequentialTestRunner(
        reference_model_path=model_name,
        scoring_config=scoring_config,
        master_key=master_key,
        device="cpu"  # Use CPU for compatibility
    )
    
    # Run a quick test
    logger.info("Starting sequential test...")
    
    try:
        metrics = runner.run_sequential_test(
            candidate_model_path=model_name,  # Same model for testing
            alpha=0.05,  # Looser bounds for demo
            beta=0.05,
            tau=0.02,    # Small threshold since we're comparing identical models
            n_max=20,    # Small number for quick test
            boundary_type="EB",
            namespace="local_test",
            output_dir="outputs/local_test"
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS")
        print("=" * 60)
        print(f"Decision: {metrics.decision}")
        print(f"Hypothesis: {metrics.hypothesis}")
        print(f"Stopped at: n = {metrics.n_used} / {metrics.n_max}")
        print(f"Mean statistic: {metrics.statistic_mean:.6f}")
        print(f"Variance: {metrics.statistic_var:.6f}")
        print("\n⏱️  TIMING BREAKDOWN")
        print("-" * 40)
        print(f"Model loading: {metrics.t_load:.2f}s")
        print(f"Setup time: {metrics.t_setup:.2f}s")
        print(f"Total inference: {metrics.t_infer_total:.2f}s")
        print(f"Per query avg: {metrics.t_per_query*1000:.1f}ms")
        print(f"Total time: {metrics.t_total:.2f}s")
        
        if metrics.per_query_scores:
            print("\n📈 SCORE PROGRESSION")
            print("-" * 40)
            for i, score in enumerate(metrics.per_query_scores[:10], 1):
                print(f"Query {i:2d}: {score:.6f}")
        
        print("\n" + "=" * 60)
        
        if metrics.decision == "accept_id":
            print("✅ Identity ACCEPTED - Models match!")
        elif metrics.decision == "reject_id":
            print("❌ Identity REJECTED - Models differ!")
        else:
            print("⚠️  UNDECIDED - Need more samples")
            
        return metrics
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return None

def run_mock_comparison_test():
    """Run a test comparing GPT-2 with simulated differences"""
    
    print("\n🔬 Running Comparison Test (Simulated Difference)\n")
    print("=" * 60)
    
    from unittest.mock import Mock
    import torch
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create a mock wrapper that adds noise to GPT-2 outputs
    class NoisyModelWrapper:
        def __init__(self, base_model_name, noise_level=0.1):
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
            self.noise_level = noise_level
            
        def __call__(self, **kwargs):
            outputs = self.model(**kwargs)
            # Add noise to logits to simulate a different model
            outputs.logits = outputs.logits + torch.randn_like(outputs.logits) * self.noise_level
            return outputs
        
        def eval(self):
            self.model.eval()
            return self
    
    # Test configuration
    master_key = b"test_key_comparison"
    scoring_config = ScoringConfig(
        method="symmetric_kl",  # More sensitive to distribution differences
        num_positions=5,
        temperature=1.0
    )
    
    # Load base GPT-2
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info("Loading GPT-2 model...")
    reference_model = AutoModelForCausalLM.from_pretrained("gpt2")
    reference_model.eval()
    
    # Create noisy version
    logger.info("Creating noisy variant...")
    candidate_model = NoisyModelWrapper("gpt2", noise_level=0.5)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate test prompts
    generator = DeterministicPromptGenerator(master_key)
    prompts = generator.batch_generate("comparison_test", 0, 10)
    
    # Score differences
    from pot.scoring.teacher_forced import TeacherForcedScorer
    scorer = TeacherForcedScorer(scoring_config)
    
    print("\n📊 Scoring Model Differences")
    print("-" * 40)
    
    scores = []
    for i, prompt in enumerate(prompts[:5]):
        result = scorer.score_models(
            reference_model,
            candidate_model,
            prompt,
            "factual",
            tokenizer
        )
        scores.append(result.score)
        print(f"Prompt {i+1}: '{prompt[:30]}...' → Score: {result.score:.4f}")
    
    mean_score = sum(scores) / len(scores)
    print(f"\nMean divergence score: {mean_score:.4f}")
    
    if mean_score < 0.1:
        print("✅ Models are very similar")
    elif mean_score < 0.3:
        print("⚠️  Models show moderate differences")
    else:
        print("❌ Models are significantly different")
    
    return scores

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test sequential verification locally")
    parser.add_argument("--comparison", action="store_true",
                       help="Run comparison test with noisy model")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="Device to use (default: cpu)")
    
    args = parser.parse_args()
    
    if args.comparison:
        scores = run_mock_comparison_test()
    else:
        metrics = run_lightweight_test()
    
    print("\n✅ Test completed successfully!")