#!/usr/bin/env python3
"""
End-to-End Validation Pipeline CLI

This script provides a command-line interface for running the complete
PoT validation pipeline with comprehensive monitoring and reporting.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.pot.validation.e2e_pipeline import (
        PipelineOrchestrator,
        PipelineConfig,
        TestingMode,
        VerificationMode
    )
    from src.pot.validation.reporting import ReportGenerator
except ImportError:
    print("Error: Could not import validation modules.")
    print("Please ensure you're running from the PoT_Experiments root directory.")
    sys.exit(1)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Run end-to-end validation pipeline for PoT framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation with default settings
  %(prog)s --ref-model gpt2 --cand-model distilgpt2

  # Quick validation with dry run
  %(prog)s --ref-model gpt2 --cand-model distilgpt2 --mode quick --dry-run

  # Full audit with custom output directory
  %(prog)s --ref-model /path/to/model1 --cand-model /path/to/model2 \\
           --mode audit --output-dir results/my_validation

  # Benchmark mode with multiple runs
  %(prog)s --ref-model gpt2 --cand-model distilgpt2 --benchmark --benchmark-runs 5

  # API mode validation
  %(prog)s --ref-model http://api1.example.com --cand-model http://api2.example.com \\
           --verification-mode api --n-challenges 50
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--ref-model',
        type=str,
        required=True,
        help='Path or identifier for reference model'
    )
    
    parser.add_argument(
        '--cand-model',
        type=str,
        required=True,
        help='Path or identifier for candidate model'
    )
    
    # Testing mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['quick', 'audit', 'extended'],
        default='audit',
        help='Testing mode (default: audit)'
    )
    
    # Verification mode
    parser.add_argument(
        '--verification-mode',
        type=str,
        choices=['local', 'api', 'hybrid'],
        default='local',
        help='Verification mode (default: local)'
    )
    
    # Pipeline options
    parser.add_argument(
        '--n-challenges',
        type=int,
        default=32,
        help='Number of challenges to generate (default: 32)'
    )
    
    parser.add_argument(
        '--max-queries',
        type=int,
        default=400,
        help='Maximum number of queries (default: 400)'
    )
    
    parser.add_argument(
        '--hmac-key',
        type=str,
        default=None,
        help='HMAC key for challenge generation (auto-generated if not provided)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/validation_reports',
        help='Output directory for reports (default: outputs/validation_reports)'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        default=True,
        help='Generate HTML report (default: True)'
    )
    
    parser.add_argument(
        '--no-report',
        dest='generate_report',
        action='store_false',
        help='Skip HTML report generation'
    )
    
    # Execution modes
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry-run mode (simulate without actual model loading)'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run in benchmark mode (multiple runs for performance testing)'
    )
    
    parser.add_argument(
        '--benchmark-runs',
        type=int,
        default=3,
        help='Number of benchmark runs (default: 3)'
    )
    
    # Performance options
    parser.add_argument(
        '--enable-zk',
        action='store_true',
        help='Enable ZK proof generation'
    )
    
    parser.add_argument(
        '--disable-memory-tracking',
        action='store_true',
        help='Disable memory tracking'
    )
    
    parser.add_argument(
        '--disable-cpu-tracking',
        action='store_true',
        help='Disable CPU tracking'
    )
    
    # Logging options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    # Config file option
    parser.add_argument(
        '--config',
        type=str,
        help='Load configuration from JSON file'
    )
    
    return parser.parse_args()


def load_config_file(config_path: str) -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}


def create_pipeline_config(args) -> PipelineConfig:
    """Create pipeline configuration from arguments"""
    # Map string mode to TestingMode enum
    mode_map = {
        'quick': TestingMode.QUICK_GATE,
        'audit': TestingMode.AUDIT_GRADE,
        'extended': TestingMode.EXTENDED
    }
    
    # Map verification mode
    verification_map = {
        'local': VerificationMode.LOCAL_WEIGHTS,
        'api': VerificationMode.API_BLACK_BOX,
        'hybrid': VerificationMode.HYBRID
    }
    
    config = PipelineConfig(
        testing_mode=mode_map[args.mode],
        verification_mode=verification_map[args.verification_mode],
        enable_zk_proof=args.enable_zk,
        enable_memory_tracking=not args.disable_memory_tracking,
        enable_cpu_tracking=not args.disable_cpu_tracking,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        benchmark_mode=args.benchmark,
        max_queries=args.max_queries,
        hmac_key=args.hmac_key,
        verbose=args.verbose and not args.quiet
    )
    
    return config


def print_summary(results: dict, logger: logging.Logger):
    """Print results summary to console"""
    if results.get('success'):
        logger.info("=" * 60)
        logger.info("VALIDATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Decision: {results['decision']}")
        logger.info(f"Confidence: {results['confidence']:.4f}")
        logger.info(f"Queries Executed: {results['n_queries']}")
        logger.info(f"Total Duration: {results['total_duration']:.2f} seconds")
        logger.info(f"Peak Memory: {results['peak_memory_mb']:.2f} MB")
        logger.info("-" * 60)
        logger.info(f"Evidence Bundle: {results['evidence_bundle_path']}")
        if results.get('zk_proof_path'):
            logger.info(f"ZK Proof: {results['zk_proof_path']}")
    else:
        logger.error("=" * 60)
        logger.error("VALIDATION FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {results.get('error', 'Unknown error')}")
        logger.error(f"Failed at stage: {results.get('stage_failed', 'Unknown')}")


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.verbose and not args.quiet)
    
    # Load config from file if provided
    if args.config:
        file_config = load_config_file(args.config)
        # Merge with command-line arguments (CLI takes precedence)
        for key, value in file_config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Create pipeline configuration
    config = create_pipeline_config(args)
    
    # Print configuration summary
    if not args.quiet:
        logger.info("=" * 60)
        logger.info("POT E2E VALIDATION PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Reference Model: {args.ref_model}")
        logger.info(f"Candidate Model: {args.cand_model}")
        logger.info(f"Testing Mode: {args.mode}")
        logger.info(f"Verification Mode: {args.verification_mode}")
        logger.info(f"Number of Challenges: {args.n_challenges}")
        logger.info(f"Output Directory: {args.output_dir}")
        
        if args.dry_run:
            logger.info(">>> DRY RUN MODE <<<")
        if args.benchmark:
            logger.info(f">>> BENCHMARK MODE ({args.benchmark_runs} runs) <<<")
        
        logger.info("=" * 60)
    
    try:
        # Create orchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # Run pipeline
        if args.benchmark:
            # Benchmark mode
            logger.info(f"Starting benchmark with {args.benchmark_runs} runs...")
            results = orchestrator.benchmark_pipeline(
                ref_model_path=args.ref_model,
                cand_model_path=args.cand_model,
                n_runs=args.benchmark_runs
            )
            
            # Print benchmark summary
            if not args.quiet:
                logger.info("=" * 60)
                logger.info("BENCHMARK RESULTS")
                logger.info("=" * 60)
                logger.info(f"Successful Runs: {results['n_successful']}/{results['n_runs']}")
                if results['n_successful'] > 0:
                    logger.info(f"Average Duration: {results['avg_duration_seconds']:.2f} seconds")
                    logger.info(f"Average Peak Memory: {results['avg_peak_memory_mb']:.2f} MB")
                    logger.info(f"Average Queries: {results['avg_queries']:.1f}")
        else:
            # Normal mode
            results = orchestrator.run_complete_pipeline(
                ref_model_path=args.ref_model,
                cand_model_path=args.cand_model,
                n_challenges=args.n_challenges
            )
            
            # Print summary
            if not args.quiet:
                print_summary(results, logger)
            
            # Generate HTML report if requested
            if args.generate_report and results.get('success'):
                logger.info("Generating HTML report...")
                generator = ReportGenerator(output_dir=Path(args.output_dir))
                
                # Load evidence bundle
                evidence_bundle = None
                if results.get('evidence_bundle_path'):
                    try:
                        with open(results['evidence_bundle_path'], 'r') as f:
                            evidence_bundle = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load evidence bundle: {e}")
                
                # Generate report
                report_path = generator.generate_html_report(results, evidence_bundle)
                logger.info(f"HTML Report: {report_path}")
                
                # Also generate JSON summary
                summary_path = Path(args.output_dir) / f"summary_{results['run_id']}.json"
                generator.generate_summary_json(results, summary_path)
                logger.info(f"JSON Summary: {summary_path}")
        
        # Exit with appropriate code
        if results.get('success', False):
            sys.exit(0)
        else:
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()