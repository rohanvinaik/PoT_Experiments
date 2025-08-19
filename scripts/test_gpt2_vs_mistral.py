#!/usr/bin/env python3
"""Real test comparing GPT-2 vs Mistral-7B locally"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path  
sys.path.insert(0, str(Path(__file__).parent.parent))

from pot.core.sequential_runner import SequentialTestRunner
from pot.scoring.teacher_forced import ScoringConfig

def main():
    """Run real test comparing GPT-2 and Mistral"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("\n🔬 REAL Sequential Test: GPT-2 vs Mistral-7B\n")
    print("=" * 60)
    
    # Models to compare
    reference_model = "gpt2"  # GPT-2 as reference
    candidate_model = "mistralai/Mistral-7B-Instruct-v0.3"  # Mistral as candidate
    
    print(f"Reference: {reference_model}")
    print(f"Candidate: {candidate_model}")
    print("=" * 60)
    
    # Configuration
    master_key = bytes.fromhex("deadbeef" * 4)  # 32 byte key
    
    scoring_config = ScoringConfig(
        method="delta_ce",  # Cross-entropy difference
        num_positions=5,     # Check 5 token positions
        temperature=1.0,
        use_canonical_suffix=True  # Use fixed suffixes for consistency
    )
    
    # Initialize runner
    logger.info("Initializing sequential test runner...")
    runner = SequentialTestRunner(
        reference_model_path=reference_model,
        scoring_config=scoring_config,
        master_key=master_key,
        device="cpu"  # Force CPU for compatibility
    )
    
    # Run the test
    logger.info("Starting sequential verification test...")
    
    try:
        metrics = runner.run_sequential_test(
            candidate_model_path=candidate_model,
            alpha=0.01,     # 1% false accept rate
            beta=0.01,      # 1% false reject rate  
            tau=0.5,        # Threshold (expecting high difference)
            n_max=30,       # Max 30 queries
            boundary_type="EB",
            namespace="gpt2_vs_mistral",
            output_dir="outputs/gpt2_vs_mistral"
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS")
        print("=" * 60)
        print(f"Decision: {metrics.decision}")
        print(f"Hypothesis: {metrics.hypothesis}")
        print(f"Samples used: {metrics.n_used} / {metrics.n_max}")
        print(f"Mean divergence: {metrics.statistic_mean:.4f}")
        print(f"Variance: {metrics.statistic_var:.6f}")
        
        print("\n⏱️  PERFORMANCE")
        print("-" * 40)
        print(f"Model loading: {metrics.t_load:.1f}s")
        print(f"Total inference: {metrics.t_infer_total:.1f}s")
        print(f"Per query: {metrics.t_per_query*1000:.0f}ms")
        print(f"Total time: {metrics.t_total:.1f}s")
        
        # Show score progression
        if metrics.per_query_scores:
            print("\n📈 DIVERGENCE SCORES")
            print("-" * 40)
            for i, score in enumerate(metrics.per_query_scores[:10], 1):
                bar = "█" * int(score * 20)
                print(f"Query {i:2d}: {score:.3f} {bar}")
        
        print("\n" + "=" * 60)
        if metrics.decision == "reject_id":
            print("✅ EXPECTED: Models are DIFFERENT (GPT-2 ≠ Mistral)")
        elif metrics.decision == "accept_id":
            print("❌ UNEXPECTED: Models marked as same (should differ)")
        else:
            print("⚠️  UNDECIDED after", metrics.n_max, "samples")
            
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    import torch
    print(f"Using device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
    main()