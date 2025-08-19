#!/usr/bin/env python3
"""Run real sequential verification test"""

import argparse
import logging
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pot.core.sequential_runner import SequentialTestRunner
from pot.scoring.teacher_forced import ScoringConfig
from pot.utils.json_utils import safe_json_dump, validate_no_mocks

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('test_run.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Run real sequential verification test")
    
    # Model paths
    parser.add_argument("--reference-model", required=True, 
                       help="Path to reference model")
    parser.add_argument("--candidate-model", required=True,
                       help="Path to candidate model to verify")
    
    # Test parameters
    parser.add_argument("--alpha", type=float, default=0.01,
                       help="Type I error bound (false accept)")
    parser.add_argument("--beta", type=float, default=0.01,
                       help="Type II error bound (false reject)")
    parser.add_argument("--tau", type=float, default=0.05,
                       help="Identity threshold")
    parser.add_argument("--n-max", type=int, default=512,
                       help="Maximum number of challenges")
    parser.add_argument("--boundary", choices=["EB", "MB"], default="EB",
                       help="Boundary type (Empirical Bernstein or Mixture)")
    
    # Scoring configuration
    parser.add_argument("--scoring-method", choices=["delta_ce", "symmetric_kl", "js_divergence"],
                       default="delta_ce", help="Scoring method")
    parser.add_argument("--num-positions", type=int, default=10,
                       help="Number of token positions to evaluate")
    
    # Security parameters
    parser.add_argument("--master-key", required=True,
                       help="Master key for PRF (hex string)")
    parser.add_argument("--namespace", default="verification",
                       help="Namespace for challenge generation")
    
    # Output
    parser.add_argument("--output-dir", default="outputs/real_tests",
                       help="Output directory for results")
    parser.add_argument("--no-validate", action="store_true",
                       help="Skip validation for mock data")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not Path(args.reference_model).exists():
        logger.error(f"Reference model not found: {args.reference_model}")
        sys.exit(1)
    
    if not Path(args.candidate_model).exists():
        logger.error(f"Candidate model not found: {args.candidate_model}")
        sys.exit(1)
    
    # Parse master key
    try:
        master_key = bytes.fromhex(args.master_key)
    except ValueError:
        logger.error("Invalid master key format (must be hex)")
        sys.exit(1)
    
    # Create scoring config
    scoring_config = ScoringConfig(
        method=args.scoring_method,
        num_positions=args.num_positions
    )
    
    # Initialize runner
    runner = SequentialTestRunner(
        reference_model_path=args.reference_model,
        scoring_config=scoring_config,
        master_key=master_key
    )
    
    # Run test
    logger.info("Starting sequential verification test")
    logger.info(f"Reference: {args.reference_model}")
    logger.info(f"Candidate: {args.candidate_model}")
    logger.info(f"Parameters: α={args.alpha}, β={args.beta}, τ={args.tau}, n_max={args.n_max}")
    
    try:
        metrics = runner.run_sequential_test(
            candidate_model_path=args.candidate_model,
            alpha=args.alpha,
            beta=args.beta,
            tau=args.tau,
            n_max=args.n_max,
            boundary_type=args.boundary,
            namespace=args.namespace,
            output_dir=args.output_dir
        )
        
        # Validate results are real (not mock)
        if not args.no_validate:
            results_dict = metrics.to_json_safe_dict()
            if not validate_no_mocks(results_dict):
                logger.warning("Results may contain mock data!")
        
        # Print summary
        logger.info("=" * 50)
        logger.info("TEST COMPLETE")
        logger.info(f"Decision: {metrics.decision}")
        logger.info(f"Hypothesis: {metrics.hypothesis}")
        logger.info(f"Stopped at n={metrics.n_used}/{metrics.n_max}")
        logger.info(f"Mean statistic: {metrics.statistic_mean:.4f}")
        logger.info(f"Variance: {metrics.statistic_var:.4f}")
        logger.info(f"Total time: {metrics.t_total:.2f}s")
        logger.info(f"Inference time: {metrics.t_infer_total:.2f}s")
        logger.info(f"Per-query time: {metrics.t_per_query*1000:.2f}ms")
        logger.info("=" * 50)
        
        # Return appropriate exit code
        if metrics.decision == "accept_id":
            sys.exit(0)  # Success - identity verified
        elif metrics.decision == "reject_id":
            sys.exit(1)  # Rejection - impostor detected
        else:
            sys.exit(2)  # Undecided
            
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        
        # Save error info
        error_file = Path(args.output_dir) / "error.json"
        safe_json_dump({
            "error": str(e),
            "type": type(e).__name__,
            "args": vars(args)
        }, error_file)
        
        sys.exit(3)

if __name__ == "__main__":
    main()