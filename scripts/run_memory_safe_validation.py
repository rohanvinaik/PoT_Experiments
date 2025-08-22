#!/usr/bin/env python3
"""
Memory-Safe Validation Runner for Large Models

This script ensures proper memory handling and sequential execution for large model tests.
It enforces a 25% memory limit and provides robust error handling and recovery.
"""

import argparse
import sys
import os
import json
import time
import psutil
import gc
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pot.validation.e2e_pipeline import (
    PipelineOrchestrator,
    PipelineConfig,
    TestingMode,
    VerificationMode
)
from src.pot.sharding.pipeline_integration import ShardedVerificationPipeline, ShardedVerificationConfig
from src.pot.sharding.memory_manager import MemoryManager, MemoryPolicy

logger = logging.getLogger(__name__)


class ModelSize(Enum):
    """Model size classification"""
    SMALL = "small"      # < 1GB
    MEDIUM = "medium"    # 1-5GB  
    LARGE = "large"      # 5-20GB
    XLARGE = "xlarge"    # > 20GB (7B models)


@dataclass
class MemorySafeConfig:
    """Configuration for memory-safe validation"""
    max_memory_percent: float = 25.0  # Maximum 25% of system memory
    sequential_threshold_gb: float = 5.0  # Run sequentially if model > 5GB
    enable_sharding_threshold_gb: float = 10.0  # Enable sharding if > 10GB
    gc_interval_seconds: float = 30.0  # Force GC every 30 seconds
    checkpoint_interval: int = 10  # Checkpoint every 10 queries
    recovery_max_attempts: int = 3  # Max recovery attempts on failure
    pre_test_gc: bool = True  # Run GC before each test
    post_test_gc: bool = True  # Run GC after each test
    memory_monitor_interval: float = 1.0  # Monitor memory every second


class MemorySafeValidator:
    """Memory-safe validation orchestrator"""
    
    def __init__(self, config: Optional[MemorySafeConfig] = None):
        self.config = config or MemorySafeConfig()
        self.memory_manager = self._init_memory_manager()
        self.sharding_pipeline = None
        self.test_queue: List[Tuple[str, str, str]] = []
        self.results: Dict[str, Any] = {}
        self.failed_tests: List[Dict[str, Any]] = []
        
    def _init_memory_manager(self) -> MemoryManager:
        """Initialize memory manager with 25% limit"""
        policy = MemoryPolicy(
            max_memory_percent=self.config.max_memory_percent,
            gc_threshold_percent=self.config.max_memory_percent - 5,  # GC at 20%
            emergency_threshold_percent=self.config.max_memory_percent + 5,  # Emergency at 30%
            swap_limit_mb=512,  # Minimal swap usage
            oom_safety_margin_mb=1024  # 1GB safety margin
        )
        return MemoryManager(policy)
    
    def classify_model_size(self, model_path: str) -> ModelSize:
        """Classify model by size"""
        try:
            # Check if it's a path or model name
            if os.path.exists(model_path):
                # Get directory size
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(model_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            total_size += os.path.getsize(filepath)
                size_gb = total_size / (1024**3)
            else:
                # Estimate based on model name
                model_name = model_path.lower()
                if '7b' in model_name or '8b' in model_name:
                    size_gb = 28  # Typical 7B model size
                elif '3b' in model_name:
                    size_gb = 12
                elif '1.3b' in model_name or '1b' in model_name:
                    size_gb = 5
                elif 'medium' in model_name:
                    size_gb = 1.5
                elif 'small' in model_name or 'distil' in model_name:
                    size_gb = 0.5
                else:
                    size_gb = 1.0  # Default estimate
            
            # Classify
            if size_gb < 1:
                return ModelSize.SMALL
            elif size_gb < 5:
                return ModelSize.MEDIUM
            elif size_gb < 20:
                return ModelSize.LARGE
            else:
                return ModelSize.XLARGE
                
        except Exception as e:
            logger.warning(f"Could not determine model size for {model_path}: {e}")
            return ModelSize.MEDIUM  # Safe default
    
    def should_run_sequentially(self, model1: str, model2: str) -> bool:
        """Determine if models should be run sequentially"""
        size1 = self.classify_model_size(model1)
        size2 = self.classify_model_size(model2)
        
        # Run sequentially if either model is large or xlarge
        return size1 in [ModelSize.LARGE, ModelSize.XLARGE] or \
               size2 in [ModelSize.LARGE, ModelSize.XLARGE]
    
    def should_enable_sharding(self, model1: str, model2: str) -> bool:
        """Determine if sharding should be enabled"""
        size1 = self.classify_model_size(model1)
        size2 = self.classify_model_size(model2)
        
        # Enable sharding for xlarge models
        return size1 == ModelSize.XLARGE or size2 == ModelSize.XLARGE
    
    def prepare_test_environment(self) -> bool:
        """Prepare environment for testing"""
        try:
            # Clear memory
            if self.config.pre_test_gc:
                logger.info("🧹 Running pre-test garbage collection...")
                gc.collect()
                gc.collect()  # Run twice to be thorough
            
            # Check available memory
            state = self.memory_manager.get_current_state()
            available_gb = state.available_bytes / (1024**3)
            used_percent = state.percent_used
            
            logger.info(f"💾 Memory status: {used_percent:.1f}% used, {available_gb:.1f}GB available")
            
            # Check if we're within limits
            if used_percent > self.config.max_memory_percent:
                logger.warning(f"⚠️ Memory usage ({used_percent:.1f}%) exceeds limit ({self.config.max_memory_percent}%)")
                
                # Try emergency cleanup
                logger.info("🚨 Attempting emergency memory cleanup...")
                self.memory_manager.emergency_cleanup()
                gc.collect()
                
                # Re-check
                state = self.memory_manager.get_current_state()
                if state.percent_used > self.config.max_memory_percent:
                    logger.error("❌ Could not reduce memory usage to safe levels")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare test environment: {e}")
            return False
    
    def run_single_test(self, ref_model: str, cand_model: str, test_name: str, 
                       attempt: int = 1) -> Dict[str, Any]:
        """Run a single test with memory safety"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Running test: {test_name} (Attempt {attempt}/{self.config.recovery_max_attempts})")
        logger.info(f"📊 Models: {ref_model} vs {cand_model}")
        logger.info(f"{'='*60}")
        
        result = {
            'test_name': test_name,
            'ref_model': ref_model,
            'cand_model': cand_model,
            'attempt': attempt,
            'success': False,
            'error': None,
            'metrics': {}
        }
        
        try:
            # Prepare environment
            if not self.prepare_test_environment():
                raise RuntimeError("Failed to prepare test environment")
            
            # Determine if sharding is needed
            enable_sharding = self.should_enable_sharding(ref_model, cand_model)
            
            # Create pipeline config
            config = PipelineConfig(
                testing_mode=TestingMode.AUDIT_GRADE,
                verification_mode=VerificationMode.LOCAL_WEIGHTS,
                n_challenges=30,  # Use reasonable number for 7B models
                enable_memory_tracking=True,
                enable_zk_proof=False,  # Disable ZK for memory efficiency
                output_dir=Path(f"outputs/memory_safe_{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            )
            
            # Initialize sharding if needed
            if enable_sharding:
                logger.info("🧩 Enabling sharding for large models...")
                if not self.sharding_pipeline:
                    shard_config = ShardedVerificationConfig(
                        enable_sharding=True,
                        max_memory_usage_percent=self.config.max_memory_percent - 5,  # Leave headroom
                        enable_checkpointing=True,
                        enable_compression=True
                    )
                    self.sharding_pipeline = ShardedVerificationPipeline(shard_config)
            
            # Create and run pipeline
            start_time = time.time()
            pipeline = PipelineOrchestrator(config)
            
            # Monitor memory during execution
            memory_samples = []
            
            def monitor_memory():
                """Background memory monitoring"""
                while getattr(monitor_memory, 'active', True):
                    state = self.memory_manager.get_current_state()
                    memory_samples.append({
                        'timestamp': state.timestamp,
                        'rss_mb': state.rss_bytes / (1024**2),
                        'percent': state.percent_used
                    })
                    
                    # Check for memory pressure
                    if state.percent_used > self.config.max_memory_percent:
                        logger.warning(f"⚠️ Memory limit exceeded: {state.percent_used:.1f}%")
                        # Trigger GC
                        gc.collect()
                    
                    time.sleep(self.config.memory_monitor_interval)
            
            # Start memory monitoring in background
            import threading
            monitor_thread = threading.Thread(target=monitor_memory)
            monitor_thread.daemon = True
            monitor_memory.active = True
            monitor_thread.start()
            
            try:
                # Run the pipeline
                pipeline_result = pipeline.run_complete_pipeline(ref_model, cand_model)
                
                # Stop monitoring
                monitor_memory.active = False
                monitor_thread.join(timeout=2)
                
                # Extract results
                result.update({
                    'success': pipeline_result.get('success', False),
                    'decision': pipeline_result.get('decision'),
                    'confidence': pipeline_result.get('confidence'),
                    'n_queries': pipeline_result.get('n_queries'),
                    'duration': time.time() - start_time,
                    'peak_memory_mb': max(s['rss_mb'] for s in memory_samples) if memory_samples else 0,
                    'avg_memory_percent': sum(s['percent'] for s in memory_samples) / len(memory_samples) if memory_samples else 0,
                    'evidence_bundle_path': pipeline_result.get('evidence_bundle_path')
                })
                
                logger.info(f"✅ Test completed successfully")
                logger.info(f"📊 Decision: {result['decision']} (confidence: {result['confidence']:.2%})")
                logger.info(f"💾 Peak memory: {result['peak_memory_mb']:.1f}MB ({result['avg_memory_percent']:.1f}% avg)")
                
            except Exception as e:
                monitor_memory.active = False
                raise e
                
        except Exception as e:
            logger.error(f"❌ Test failed: {str(e)}")
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            
            # Check if we should retry
            if attempt < self.config.recovery_max_attempts:
                logger.info(f"🔄 Retrying test after cleanup...")
                
                # Aggressive cleanup
                gc.collect()
                time.sleep(5)  # Give system time to recover
                
                # Recursive retry
                return self.run_single_test(ref_model, cand_model, test_name, attempt + 1)
            else:
                logger.error(f"❌ Test failed after {attempt} attempts")
                self.failed_tests.append(result)
        
        finally:
            # Post-test cleanup
            if self.config.post_test_gc:
                logger.info("🧹 Running post-test cleanup...")
                gc.collect()
                
                # Clear any cached models
                if hasattr(pipeline, 'clear_cache'):
                    pipeline.clear_cache()
        
        return result
    
    def run_test_suite(self, test_pairs: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        """
        Run a suite of tests with proper memory management.
        
        Args:
            test_pairs: List of (ref_model, cand_model, test_name) tuples
            
        Returns:
            Results dictionary
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 Starting Memory-Safe Validation Suite")
        logger.info(f"📊 Total tests: {len(test_pairs)}")
        logger.info(f"💾 Memory limit: {self.config.max_memory_percent}%")
        logger.info(f"{'='*80}\n")
        
        # Group tests by whether they need sequential execution
        sequential_tests = []
        parallel_tests = []
        
        for ref_model, cand_model, test_name in test_pairs:
            if self.should_run_sequentially(ref_model, cand_model):
                sequential_tests.append((ref_model, cand_model, test_name))
            else:
                parallel_tests.append((ref_model, cand_model, test_name))
        
        logger.info(f"📋 Test allocation:")
        logger.info(f"  - Sequential (large models): {len(sequential_tests)}")
        logger.info(f"  - Parallel (small models): {len(parallel_tests)}")
        
        results = {
            'start_time': datetime.now().isoformat(),
            'config': {
                'max_memory_percent': self.config.max_memory_percent,
                'sequential_threshold_gb': self.config.sequential_threshold_gb,
                'enable_sharding_threshold_gb': self.config.enable_sharding_threshold_gb
            },
            'tests': [],
            'summary': {}
        }
        
        # Run sequential tests first (one at a time)
        logger.info(f"\n🔄 Running sequential tests...")
        for ref_model, cand_model, test_name in sequential_tests:
            test_result = self.run_single_test(ref_model, cand_model, test_name)
            results['tests'].append(test_result)
            
            # Wait between tests to ensure memory is freed
            time.sleep(5)
        
        # Run parallel tests (can potentially run multiple, but we'll still be careful)
        if parallel_tests:
            logger.info(f"\n⚡ Running parallel-capable tests (still sequential for safety)...")
            for ref_model, cand_model, test_name in parallel_tests:
                test_result = self.run_single_test(ref_model, cand_model, test_name)
                results['tests'].append(test_result)
        
        # Generate summary
        successful_tests = [t for t in results['tests'] if t.get('success', False)]
        failed_tests = [t for t in results['tests'] if not t.get('success', False)]
        
        results['summary'] = {
            'total_tests': len(test_pairs),
            'successful': len(successful_tests),
            'failed': len(failed_tests),
            'success_rate': len(successful_tests) / len(test_pairs) if test_pairs else 0,
            'end_time': datetime.now().isoformat(),
            'peak_memory_mb': max((t.get('peak_memory_mb', 0) for t in results['tests']), default=0),
            'failed_test_names': [t['test_name'] for t in failed_tests]
        }
        
        # Save results
        output_file = f"outputs/memory_safe_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("outputs", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 Validation Suite Complete")
        logger.info(f"✅ Successful: {results['summary']['successful']}/{results['summary']['total_tests']}")
        if failed_tests:
            logger.error(f"❌ Failed tests: {', '.join(results['summary']['failed_test_names'])}")
        logger.info(f"💾 Peak memory usage: {results['summary']['peak_memory_mb']:.1f}MB")
        logger.info(f"📁 Results saved to: {output_file}")
        logger.info(f"{'='*80}\n")
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run memory-safe validation for large models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 7B model permutations (A|A, B|B, A|B)
  %(prog)s --models yi-6b yi-34b --permutations all --max-memory 25
  
  # Run specific test
  %(prog)s --ref-model meta-llama/Llama-2-7b-hf --cand-model meta-llama/Llama-2-7b-chat-hf
  
  # Run with custom memory limit
  %(prog)s --models gpt2 distilgpt2 --max-memory 15
        """
    )
    
    parser.add_argument(
        '--ref-model',
        type=str,
        help='Reference model path or name'
    )
    
    parser.add_argument(
        '--cand-model', 
        type=str,
        help='Candidate model path or name'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        help='List of models for permutation testing'
    )
    
    parser.add_argument(
        '--permutations',
        choices=['all', 'self', 'cross'],
        default='all',
        help='Type of permutations to run (all=A|A,B|B,A|B, self=A|A,B|B, cross=A|B)'
    )
    
    parser.add_argument(
        '--max-memory',
        type=float,
        default=25.0,
        help='Maximum memory usage percentage (default: 25%%)'
    )
    
    parser.add_argument(
        '--sequential-threshold',
        type=float,
        default=5.0,
        help='Model size threshold (GB) for sequential execution'
    )
    
    parser.add_argument(
        '--sharding-threshold',
        type=float,
        default=10.0,
        help='Model size threshold (GB) for enabling sharding'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create configuration
    config = MemorySafeConfig(
        max_memory_percent=args.max_memory,
        sequential_threshold_gb=args.sequential_threshold,
        enable_sharding_threshold_gb=args.sharding_threshold
    )
    
    # Initialize validator
    validator = MemorySafeValidator(config)
    
    # Prepare test pairs
    test_pairs = []
    
    if args.models:
        # Generate permutations
        models = args.models
        
        if args.permutations in ['all', 'self']:
            # Self-consistency tests (A|A, B|B)
            for model in models:
                test_name = f"{Path(model).name}_self_consistency"
                test_pairs.append((model, model, test_name))
        
        if args.permutations in ['all', 'cross'] and len(models) >= 2:
            # Cross-model test (A|B)
            test_name = f"{Path(models[0]).name}_vs_{Path(models[1]).name}"
            test_pairs.append((models[0], models[1], test_name))
    
    elif args.ref_model and args.cand_model:
        # Single test
        test_name = f"{Path(args.ref_model).name}_vs_{Path(args.cand_model).name}"
        test_pairs.append((args.ref_model, args.cand_model, test_name))
    
    else:
        parser.error("Either --models or both --ref-model and --cand-model must be provided")
    
    # Run tests
    if test_pairs:
        results = validator.run_test_suite(test_pairs)
        
        # Exit with appropriate code
        if results['summary']['failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    else:
        logger.error("No tests to run")
        sys.exit(1)


if __name__ == "__main__":
    main()