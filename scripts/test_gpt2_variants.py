#!/usr/bin/env python3
"""Real test comparing GPT-2 with distilgpt2 (a smaller variant)"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path  
sys.path.insert(0, str(Path(__file__).parent.parent))

from pot.core.sequential_runner import SequentialTestRunner
from pot.scoring.teacher_forced import ScoringConfig

def main():
    """Run real test comparing GPT-2 variants"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("\n🔬 REAL Sequential Test: GPT-2 vs DistilGPT-2\n")
    print("=" * 60)
    
    # Models to compare - both are small enough to run on CPU
    reference_model = "gpt2"  # Standard GPT-2 (124M params)
    candidate_model = "distilgpt2"  # DistilGPT-2 (82M params, distilled version)
    
    print(f"Reference: {reference_model}")
    print(f"Candidate: {candidate_model}")
    print("=" * 60)
    print("\nNote: DistilGPT-2 is a distilled version of GPT-2,")
    print("so they should be similar but not identical.\n")
    
    # Configuration
    master_key = bytes.fromhex("deadbeef" * 4)  # 32 byte key
    
    scoring_config = ScoringConfig(
        method="symmetric_kl",  # KL divergence to measure distribution difference
        num_positions=8,        # Check 8 token positions
        temperature=1.0,
        use_canonical_suffix=True
    )
    
    # Initialize runner
    logger.info("Initializing sequential test runner...")
    runner = SequentialTestRunner(
        reference_model_path=reference_model,
        scoring_config=scoring_config,
        master_key=master_key,
        device="cpu"  # Use CPU for compatibility
    )
    
    # Run the test
    logger.info("Starting sequential verification test...")
    
    try:
        metrics = runner.run_sequential_test(
            candidate_model_path=candidate_model,
            alpha=0.01,     # 1% false accept rate
            beta=0.01,      # 1% false reject rate  
            tau=0.15,       # Moderate threshold (expecting some difference)
            n_max=50,       # Max 50 queries
            boundary_type="EB",
            namespace="gpt2_vs_distilgpt2",
            output_dir="outputs/gpt2_comparison"
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
        print(f"Threshold (τ): {0.15:.4f}")
        
        print("\n⏱️  PERFORMANCE")
        print("-" * 40)
        print(f"Model loading: {metrics.t_load:.1f}s")
        print(f"Total inference: {metrics.t_infer_total:.1f}s")
        print(f"Per query: {metrics.t_per_query*1000:.0f}ms")
        print(f"Total time: {metrics.t_total:.1f}s")
        
        # Show score progression
        if metrics.per_query_scores:
            print("\n📈 KL DIVERGENCE SCORES")
            print("-" * 40)
            for i, score in enumerate(metrics.per_query_scores[:15], 1):
                bar = "█" * min(int(score * 40), 40)
                print(f"Query {i:2d}: {score:.4f} {bar}")
            
            if len(metrics.per_query_scores) > 15:
                print(f"... ({len(metrics.per_query_scores) - 15} more)")
        
        # Show confidence intervals
        if metrics.confidence_intervals and len(metrics.confidence_intervals) > 0:
            print("\n📉 CONFIDENCE INTERVALS")
            print("-" * 40)
            for i in [0, len(metrics.confidence_intervals)//2, -1]:
                if i < len(metrics.confidence_intervals):
                    ci = metrics.confidence_intervals[i]
                    n = i + 1 if i >= 0 else metrics.n_used
                    print(f"n={n:3d}: [{ci[0]:.4f}, {ci[1]:.4f}]")
        
        print("\n" + "=" * 60)
        print("INTERPRETATION:")
        if metrics.decision == "accept_id":
            print("✅ Models are SIMILAR enough (within threshold)")
            print("   DistilGPT-2 successfully mimics GPT-2 behavior")
        elif metrics.decision == "reject_id":
            print("⚠️  Models are DIFFERENT (exceed threshold)")
            print("   DistilGPT-2 diverges significantly from GPT-2")
        else:
            print("❓ UNDECIDED after", metrics.n_max, "samples")
            print("   Need more data to make a decision")
            
        # Save summary
        print(f"\n📁 Results saved to: {Path('outputs/gpt2_comparison').absolute()}")
            
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()