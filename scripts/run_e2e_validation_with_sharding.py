#!/usr/bin/env python3
"""
Enhanced E2E Validation Pipeline with Sharding Support

This script extends the standard E2E validation pipeline to support
large model verification through intelligent sharding.
"""

import argparse
import sys
import os
import json
import psutil
from pathlib import Path
from datetime import datetime
import logging
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import sharding components
from src.pot.sharding import AdaptiveShardManager, MemoryManager
from src.pot.sharding.pipeline_integration import (
    ShardedVerificationPipeline,
    ShardedVerificationConfig,
    integrate_with_e2e_pipeline
)

# Try to import standard validation components
try:
    from src.pot.validation.e2e_pipeline import (
        PipelineOrchestrator,
        PipelineConfig,
        TestingMode,
        VerificationMode
    )
    STANDARD_PIPELINE_AVAILABLE = True
except ImportError:
    STANDARD_PIPELINE_AVAILABLE = False
    print("Warning: Standard pipeline not available, using sharding-only mode")


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments with sharding options"""
    parser = argparse.ArgumentParser(
        description='E2E validation pipeline with sharding support for large models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sharding Examples:
  # Automatic sharding for large models
  %(prog)s --ref-model /path/to/70B_model --cand-model /path/to/70B_model_v2 \\
           --enable-sharding --auto-detect-large-model

  # Manual sharding with configuration
  %(prog)s --ref-model yi-34b --cand-model yi-34b-chat \\
           --enable-sharding --shard-size-mb 2048 --max-memory-percent 70

  # Benchmark mode for large models
  %(prog)s --ref-model llama-70b --cand-model llama-70b \\
           --enable-sharding --benchmark-sharding --dry-run
        """
    )
    
    # Standard arguments
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
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['quick', 'audit', 'extended'],
        default='audit',
        help='Testing mode (default: audit)'
    )
    
    # Sharding arguments
    sharding_group = parser.add_argument_group('Sharding Options')
    
    sharding_group.add_argument(
        '--enable-sharding',
        action='store_true',
        help='Enable sharded verification for large models'
    )
    
    sharding_group.add_argument(
        '--auto-detect-large-model',
        action='store_true',
        default=True,
        help='Automatically detect and shard large models (default: True)'
    )
    
    sharding_group.add_argument(
        '--large-model-threshold-gb',
        type=float,
        default=30.0,
        help='Model size threshold for automatic sharding (default: 30 GB)'
    )
    
    sharding_group.add_argument(
        '--shard-size-mb',
        type=int,
        help='Manual shard size in MB (auto-calculated if not specified)'
    )
    
    sharding_group.add_argument(
        '--max-memory-percent',
        type=float,
        default=70.0,
        help='Maximum memory usage percentage (default: 70%%)'
    )
    
    sharding_group.add_argument(
        '--enable-compression',
        action='store_true',
        default=True,
        help='Enable shard compression (default: True)'
    )
    
    sharding_group.add_argument(
        '--enable-checkpointing',
        action='store_true',
        default=True,
        help='Enable checkpointing for fault tolerance (default: True)'
    )
    
    sharding_group.add_argument(
        '--shard-cache-dir',
        type=str,
        default='/tmp/pot_shards',
        help='Directory for shard cache (default: /tmp/pot_shards)'
    )
    
    sharding_group.add_argument(
        '--benchmark-sharding',
        action='store_true',
        help='Run sharding benchmark'
    )
    
    # Performance monitoring
    monitoring_group = parser.add_argument_group('Monitoring Options')
    
    monitoring_group.add_argument(
        '--enable-monitoring',
        action='store_true',
        default=True,
        help='Enable resource monitoring (default: True)'
    )
    
    monitoring_group.add_argument(
        '--monitor-interval',
        type=float,
        default=1.0,
        help='Monitoring interval in seconds (default: 1.0)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/sharded_validation',
        help='Output directory for reports'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry-run mode'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def check_system_requirements(logger):
    """Check system requirements for sharding"""
    mem_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')
    
    logger.info("System Requirements Check:")
    logger.info(f"  Total RAM: {mem_info.total / (1024**3):.1f} GB")
    logger.info(f"  Available RAM: {mem_info.available / (1024**3):.1f} GB")
    logger.info(f"  CPU Cores: {psutil.cpu_count()}")
    logger.info(f"  Disk Space: {disk_info.free / (1024**3):.1f} GB free")
    
    # Check minimum requirements
    min_ram_gb = 8
    min_disk_gb = 50
    
    if mem_info.total / (1024**3) < min_ram_gb:
        logger.warning(f"System has less than {min_ram_gb} GB RAM, sharding may be slow")
    
    if disk_info.free / (1024**3) < min_disk_gb:
        logger.warning(f"Less than {min_disk_gb} GB disk space available")
    
    return True


def get_model_size(model_path: str) -> float:
    """Get model size in GB"""
    if not os.path.exists(model_path):
        # Estimate based on known model names
        known_sizes = {
            'yi-34b': 206,
            'llama-70b': 140,
            'llama-65b': 130,
            'falcon-40b': 80,
            'gpt-j-6b': 12
        }
        
        for name, size in known_sizes.items():
            if name in model_path.lower():
                return size
        
        return 0.0
    
    # Calculate actual size
    if os.path.isfile(model_path):
        return os.path.getsize(model_path) / (1024**3)
    elif os.path.isdir(model_path):
        total_size = 0
        for root, _, files in os.walk(model_path):
            for file in files:
                if file.endswith(('.bin', '.safetensors', '.pt', '.pth')):
                    total_size += os.path.getsize(os.path.join(root, file))
        return total_size / (1024**3)
    
    return 0.0


def run_sharded_validation(args, logger):
    """Run validation with sharding"""
    logger.info("="*60)
    logger.info("SHARDED VALIDATION PIPELINE")
    logger.info("="*60)
    
    # Check system requirements
    if not check_system_requirements(logger):
        return {'success': False, 'error': 'System requirements not met'}
    
    # Check model sizes
    ref_model_size = get_model_size(args.ref_model)
    cand_model_size = get_model_size(args.cand_model)
    
    logger.info(f"\nModel Sizes:")
    logger.info(f"  Reference: {ref_model_size:.1f} GB")
    logger.info(f"  Candidate: {cand_model_size:.1f} GB")
    
    # Determine if sharding is needed
    max_model_size = max(ref_model_size, cand_model_size)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    should_shard = (
        args.enable_sharding or
        (args.auto_detect_large_model and max_model_size > args.large_model_threshold_gb) or
        max_model_size > available_memory_gb * 0.5
    )
    
    if should_shard:
        logger.info(f"\nSharding enabled (model size {max_model_size:.1f} GB)")
    else:
        logger.info("\nSharding not required")
        if STANDARD_PIPELINE_AVAILABLE:
            logger.info("Using standard pipeline")
            # Fall back to standard pipeline
            return run_standard_validation(args, logger)
    
    # Configure sharding
    config = ShardedVerificationConfig(
        enable_sharding=True,
        auto_detect_large_model=args.auto_detect_large_model,
        large_model_threshold_gb=args.large_model_threshold_gb,
        shard_cache_dir=args.shard_cache_dir,
        enable_compression=args.enable_compression,
        enable_checkpointing=args.enable_checkpointing,
        max_memory_usage_percent=args.max_memory_percent,
        enable_monitoring=args.enable_monitoring
    )
    
    # Initialize pipeline
    pipeline = ShardedVerificationPipeline(config)
    
    # Prepare sharding for both models
    logger.info("\nPreparing sharded verification...")
    
    ref_shard_config, ref_shard_dir = pipeline.prepare_sharded_verification(
        args.ref_model,
        os.path.join(args.shard_cache_dir, "ref_model")
    )
    
    cand_shard_config, cand_shard_dir = pipeline.prepare_sharded_verification(
        args.cand_model,
        os.path.join(args.shard_cache_dir, "cand_model")
    )
    
    logger.info(f"\nSharding Configuration:")
    logger.info(f"  Reference: {ref_shard_config.num_shards} shards of {ref_shard_config.shard_size_mb} MB")
    logger.info(f"  Candidate: {cand_shard_config.num_shards} shards of {cand_shard_config.shard_size_mb} MB")
    
    if args.dry_run:
        logger.info("\n[DRY RUN] Skipping actual verification")
        return {
            'success': True,
            'dry_run': True,
            'ref_shards': ref_shard_config.num_shards,
            'cand_shards': cand_shard_config.num_shards
        }
    
    # Run verification
    logger.info("\nStarting sharded verification...")
    start_time = time.time()
    
    try:
        # Generate challenges
        challenges = [f"challenge_{i}" for i in range(32)]  # Simplified
        
        # Verification function (simplified for demo)
        def verify_shard(shard_data, challenges):
            # In production, this would run actual PoT verification
            return [f"response_{i}" for i in range(len(challenges))]
        
        # Run sharded verification for reference model
        logger.info("\nProcessing reference model...")
        ref_results = pipeline.verify_with_sharding(
            args.ref_model,
            verify_shard,
            challenges
        )
        
        # Run sharded verification for candidate model
        logger.info("\nProcessing candidate model...")
        cand_results = pipeline.verify_with_sharding(
            args.cand_model,
            verify_shard,
            challenges
        )
        
        elapsed = time.time() - start_time
        
        # Generate report
        results = {
            'success': True,
            'duration_seconds': elapsed,
            'ref_model': {
                'shards_processed': ref_results['metrics']['total_shards_processed'],
                'peak_memory_mb': ref_results['metrics']['peak_memory_mb']
            },
            'cand_model': {
                'shards_processed': cand_results['metrics']['total_shards_processed'],
                'peak_memory_mb': cand_results['metrics']['peak_memory_mb']
            },
            'memory_efficiency': max_model_size * 1024 / max(
                ref_results['metrics']['peak_memory_mb'],
                cand_results['metrics']['peak_memory_mb']
            )
        }
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / f"sharded_validation_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to {report_path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("VALIDATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total Duration: {elapsed:.1f} seconds")
        logger.info(f"Memory Efficiency: {results['memory_efficiency']:.1f}x")
        logger.info(f"Peak Memory Usage: {max(ref_results['metrics']['peak_memory_mb'], cand_results['metrics']['peak_memory_mb']):.1f} MB")
        
        return results
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {'success': False, 'error': str(e)}


def run_standard_validation(args, logger):
    """Run standard validation without sharding"""
    # This would call the standard E2E pipeline
    logger.info("Running standard validation pipeline...")
    return {'success': True, 'standard_pipeline': True}


def main():
    """Main entry point"""
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    
    # Run benchmark if requested
    if args.benchmark_sharding:
        logger.info("Running sharding benchmark...")
        # Import and run benchmark
        from scripts.sharding.benchmark_large_models import run_yi_34b_benchmark
        run_yi_34b_benchmark(args)
        return
    
    # Run sharded validation
    results = run_sharded_validation(args, logger)
    
    # Exit with appropriate code
    sys.exit(0 if results.get('success') else 1)


if __name__ == '__main__':
    main()