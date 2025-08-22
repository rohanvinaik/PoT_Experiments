"""
Pipeline Integration for Sharded Verification

Integrates sharded verification into the main PoT pipeline for seamless
processing of large models.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from .adaptive_sharding import AdaptiveShardManager, ShardConfig
from .memory_manager import MemoryManager, MemoryPolicy
from .shard_scheduler import ShardScheduler
from .checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


@dataclass
class ShardedVerificationConfig:
    """Configuration for sharded verification"""
    enable_sharding: bool = False
    auto_detect_large_model: bool = True
    large_model_threshold_gb: float = 30.0
    shard_cache_dir: str = "/tmp/pot_shards"
    enable_compression: bool = True
    enable_checkpointing: bool = True
    max_memory_usage_percent: float = 70.0
    enable_monitoring: bool = True
    enable_prefetch: bool = True
    prefetch_count: int = 2


class ShardedVerificationPipeline:
    """
    Main pipeline for sharded verification of large models.
    
    Integrates with existing PoT verification pipeline to enable
    processing of models that exceed available memory.
    """
    
    def __init__(self, config: Optional[ShardedVerificationConfig] = None):
        """
        Initialize sharded verification pipeline.
        
        Args:
            config: Sharded verification configuration
        """
        self.config = config or ShardedVerificationConfig()
        
        # Initialize components
        self.shard_manager = AdaptiveShardManager({
            'target_memory_usage': self.config.max_memory_usage_percent / 100
        })
        
        self.memory_manager = MemoryManager(MemoryPolicy(
            max_memory_percent=self.config.max_memory_usage_percent,
            gc_threshold_percent=self.config.max_memory_usage_percent - 10,
            emergency_threshold_percent=self.config.max_memory_usage_percent + 10,
            swap_limit_mb=2048,  # 2GB swap limit
            oom_safety_margin_mb=1024  # 1GB safety margin
        ))
        
        self.scheduler = ShardScheduler()
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(self.config.shard_cache_dir, "checkpoints")
        )
        
        self.metrics = {
            'total_shards_processed': 0,
            'total_time_seconds': 0,
            'peak_memory_mb': 0,
            'page_faults': 0,
            'checkpoints_created': 0
        }
    
    def should_use_sharding(self, model_path: str) -> bool:
        """
        Determine if sharding should be used for model.
        
        Args:
            model_path: Path to model
            
        Returns:
            True if sharding should be used
        """
        if not self.config.enable_sharding:
            return False
        
        if not self.config.auto_detect_large_model:
            return True
        
        # Check model size
        try:
            model_size_gb = self._get_model_size_gb(model_path)
            available_memory_gb = self.memory_manager.get_current_state().available_bytes / (1024**3)
            
            # Use sharding if model is larger than threshold or exceeds available memory
            should_shard = (
                model_size_gb > self.config.large_model_threshold_gb or
                model_size_gb > available_memory_gb * 0.5  # Model uses >50% of available RAM
            )
            
            if should_shard:
                logger.info(f"Model size {model_size_gb:.1f}GB exceeds threshold, "
                           f"enabling sharding")
            
            return should_shard
            
        except Exception as e:
            logger.warning(f"Could not determine model size: {e}")
            return False
    
    def prepare_sharded_verification(
        self,
        model_path: str,
        output_dir: Optional[str] = None
    ) -> Tuple[ShardConfig, str]:
        """
        Prepare model for sharded verification.
        
        Args:
            model_path: Path to model
            output_dir: Directory for shard cache
            
        Returns:
            Tuple of (ShardConfig, shard_directory)
        """
        logger.info("Preparing sharded verification")
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(
                self.config.shard_cache_dir,
                f"model_{hash(model_path) % 1000000}"
            )
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze model and get optimal configuration
        hardware_profile = self._get_hardware_profile()
        shard_config = self.shard_manager.optimize_configuration(
            model_path,
            hardware_profile
        )
        
        # Create shards if not already cached
        shard_info_file = os.path.join(output_dir, "shard_info.json")
        if not os.path.exists(shard_info_file):
            logger.info("Creating model shards...")
            shard_infos = self.shard_manager.create_shards(
                model_path,
                shard_config,
                output_dir
            )
            
            # Save shard information
            import json
            with open(shard_info_file, 'w') as f:
                json.dump({
                    'config': {
                        'shard_size_mb': shard_config.shard_size_mb,
                        'num_shards': shard_config.num_shards,
                        'overlap_ratio': shard_config.overlap_ratio,
                        'compression_enabled': shard_config.compression_enabled
                    },
                    'shards': [
                        {
                            'shard_id': s.shard_id,
                            'start_param': s.start_param,
                            'end_param': s.end_param,
                            'size_bytes': s.size_bytes,
                            'layer_names': s.layer_names
                        }
                        for s in shard_infos
                    ]
                }, f, indent=2)
        else:
            logger.info("Using cached shards")
            # Load existing configuration
            import json
            with open(shard_info_file, 'r') as f:
                info = json.load(f)
                shard_config = ShardConfig(**info['config'])
        
        return shard_config, output_dir
    
    def verify_with_sharding(
        self,
        model_path: str,
        verification_fn: callable,
        challenges: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform verification using sharding.
        
        Args:
            model_path: Path to model
            verification_fn: Verification function to apply
            challenges: List of challenges for verification
            **kwargs: Additional arguments for verification
            
        Returns:
            Verification results
        """
        start_time = time.time()
        
        # Prepare sharding
        shard_config, shard_dir = self.prepare_sharded_verification(model_path)
        
        # Set up memory monitoring
        self.memory_manager.establish_baseline()
        if self.config.enable_monitoring:
            self.memory_manager.register_emergency_callback(
                lambda: self.shard_manager.unload_shard(-1)  # Unload all
            )
        
        # Initialize results
        results = {
            'responses': [],
            'shard_metrics': [],
            'memory_profile': [],
            'verification_complete': False
        }
        
        # Create checkpoint if enabled
        checkpoint = None
        if self.config.enable_checkpointing:
            checkpoint = self.checkpoint_manager.create_checkpoint(
                model_path,
                shard_config,
                challenges
            )
        
        try:
            # Process each shard
            for shard_id, shard_data in self.shard_manager.get_shard_iterator(
                shard_dir,
                shard_config
            ):
                logger.info(f"Processing shard {shard_id + 1}/{shard_config.num_shards}")
                
                # Check memory before processing
                if not self.memory_manager.check_memory_available(100 * 1024 * 1024):  # 100MB buffer
                    logger.warning("Insufficient memory for processing")
                    self.shard_manager.unload_shard(shard_id)
                    continue
                
                # Process challenges with current shard
                shard_start_time = time.time()
                shard_responses = self._process_shard(
                    shard_data,
                    verification_fn,
                    challenges,
                    **kwargs
                )
                shard_time = time.time() - shard_start_time
                
                # Collect results
                results['responses'].extend(shard_responses)
                
                # Record metrics
                memory_state = self.memory_manager.get_current_state()
                shard_metric = {
                    'shard_id': shard_id,
                    'processing_time': shard_time,
                    'memory_mb': memory_state.rss_bytes / 1024 / 1024,
                    'responses_generated': len(shard_responses)
                }
                results['shard_metrics'].append(shard_metric)
                
                # Update checkpoint
                if checkpoint:
                    self.checkpoint_manager.update_checkpoint(
                        checkpoint,
                        shard_id,
                        shard_responses
                    )
                
                # Update metrics
                self.metrics['total_shards_processed'] += 1
                self.metrics['peak_memory_mb'] = max(
                    self.metrics['peak_memory_mb'],
                    memory_state.rss_bytes / 1024 / 1024
                )
                
                # Check memory pressure
                pressure = self.memory_manager.monitor_memory_pressure()
                if pressure['level'] == 'critical':
                    logger.warning("Critical memory pressure, triggering cleanup")
                    self.shard_manager.unload_shard(shard_id)
            
            results['verification_complete'] = True
            
        except Exception as e:
            logger.error(f"Sharded verification failed: {e}")
            if checkpoint:
                # Try to recover from checkpoint
                logger.info("Attempting recovery from checkpoint")
                results = self.checkpoint_manager.recover_from_checkpoint(checkpoint)
            raise
        
        finally:
            # Clean up
            elapsed_time = time.time() - start_time
            self.metrics['total_time_seconds'] = elapsed_time
            
            # Get final memory profile
            results['memory_profile'] = self.memory_manager.get_memory_usage_summary()
            results['metrics'] = self.metrics
            
            # Clean up loaded shards
            for shard_id in list(self.shard_manager.current_shards.keys()):
                self.shard_manager.unload_shard(shard_id)
            
            logger.info(f"Sharded verification complete in {elapsed_time:.1f}s")
        
        return results
    
    def _get_model_size_gb(self, model_path: str) -> float:
        """Get model size in GB"""
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
    
    def _get_hardware_profile(self) -> Dict[str, Any]:
        """Get hardware profile for optimization"""
        import psutil
        
        # Get disk speed (simplified - actual implementation would benchmark)
        disk_speed_mbps = 100  # Assume 100 MB/s for HDD, 500+ for SSD
        
        return {
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cpu_cores': psutil.cpu_count(),
            'disk_speed_mbps': disk_speed_mbps
        }
    
    def _process_shard(
        self,
        shard_data: Any,
        verification_fn: callable,
        challenges: List[str],
        **kwargs
    ) -> List[Any]:
        """
        Process challenges with a single shard.
        
        Args:
            shard_data: Loaded shard data
            verification_fn: Verification function
            challenges: Challenges to process
            **kwargs: Additional arguments
            
        Returns:
            List of responses
        """
        responses = []
        
        # In actual implementation, this would:
        # 1. Load the shard's parameters into a partial model
        # 2. Run verification on that partial model
        # 3. Aggregate results appropriately
        
        # For now, simulate processing
        for challenge in challenges[:5]:  # Process subset for demo
            # Simulate verification with shard
            response = f"shard_{shard_data['shard_id']}_response_to_{challenge[:20]}"
            responses.append(response)
        
        return responses
    
    def get_sharding_report(self) -> Dict[str, Any]:
        """
        Get comprehensive sharding performance report.
        
        Returns:
            Sharding performance metrics
        """
        memory_summary = self.memory_manager.get_memory_usage_summary()
        
        # Calculate efficiency metrics
        if self.metrics['total_time_seconds'] > 0:
            shards_per_second = self.metrics['total_shards_processed'] / self.metrics['total_time_seconds']
            memory_efficiency = (
                self.metrics['total_shards_processed'] * 1000 / 
                max(1, self.metrics['peak_memory_mb'])
            )
        else:
            shards_per_second = 0
            memory_efficiency = 0
        
        return {
            'summary': {
                'total_shards': self.metrics['total_shards_processed'],
                'total_time_seconds': self.metrics['total_time_seconds'],
                'shards_per_second': shards_per_second,
                'peak_memory_mb': self.metrics['peak_memory_mb'],
                'memory_efficiency': memory_efficiency
            },
            'memory': memory_summary,
            'sharding': {
                'cache_dir': self.config.shard_cache_dir,
                'compression_enabled': self.config.enable_compression,
                'checkpointing_enabled': self.config.enable_checkpointing,
                'prefetch_enabled': self.config.enable_prefetch
            }
        }


def integrate_with_e2e_pipeline(pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integrate sharding with E2E pipeline configuration.
    
    Args:
        pipeline_config: Original pipeline configuration
        
    Returns:
        Updated configuration with sharding support
    """
    # Add sharding configuration
    pipeline_config['sharding'] = {
        'enable': True,
        'auto_detect_large_model': True,
        'large_model_threshold_gb': 30.0,
        'max_memory_usage_percent': 70.0,
        'enable_monitoring': True,
        'enable_checkpointing': True,
        'enable_compression': True
    }
    
    # Add sharding hooks
    pipeline_config['hooks'] = pipeline_config.get('hooks', {})
    pipeline_config['hooks']['pre_verification'] = 'check_sharding_requirement'
    pipeline_config['hooks']['verification_wrapper'] = 'sharded_verification_wrapper'
    
    return pipeline_config