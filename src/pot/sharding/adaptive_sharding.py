"""
Adaptive Sharding Manager

Dynamically optimizes shard sizes based on available memory and model characteristics.
"""

import os
import gc
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass
import logging
import time
from pathlib import Path
import torch
import json

logger = logging.getLogger(__name__)


@dataclass
class ShardConfig:
    """Configuration for model sharding"""
    shard_size_mb: int
    num_shards: int
    overlap_ratio: float  # For boundary handling
    compression_enabled: bool
    prefetch_count: int
    checkpoint_interval: int


@dataclass
class ShardInfo:
    """Information about a model shard"""
    shard_id: int
    start_param: int
    end_param: int
    size_bytes: int
    layer_names: List[str]
    dependencies: List[int]  # Other shard IDs this depends on


class AdaptiveShardManager:
    """
    Manages adaptive sharding for large model verification.
    
    Features:
    - Dynamic shard size optimization
    - Memory-aware loading strategies
    - Boundary handling for layer dependencies
    - Compression support
    - Fault tolerance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adaptive shard manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.target_memory_usage = self.config.get('target_memory_usage', 0.7)  # 70% of available RAM
        self.min_shard_size_mb = self.config.get('min_shard_size_mb', 100)
        self.max_shard_size_mb = self.config.get('max_shard_size_mb', 4096)
        self.compression_threshold_mb = self.config.get('compression_threshold_mb', 1024)
        
        self.current_shards: Dict[int, Any] = {}
        self.shard_metadata: Dict[int, ShardInfo] = {}
        self.memory_history: List[float] = []
        self.performance_metrics: Dict[str, Any] = {}
        
    def analyze_model(self, model_path: str) -> ShardConfig:
        """
        Analyze model and determine optimal sharding configuration.
        
        Args:
            model_path: Path to model
            
        Returns:
            Optimal ShardConfig
        """
        logger.info(f"Analyzing model at {model_path}")
        
        # Get model size
        model_size = self._get_model_size(model_path)
        
        # Get available memory
        available_memory = self._get_available_memory()
        
        # Calculate optimal shard size
        optimal_shard_size = self._calculate_optimal_shard_size(
            model_size, 
            available_memory
        )
        
        # Determine number of shards
        num_shards = max(1, int(np.ceil(model_size / (optimal_shard_size * 1024 * 1024))))
        
        # Check if compression is beneficial
        compression_enabled = optimal_shard_size > self.compression_threshold_mb
        
        # Calculate prefetch count based on available memory
        prefetch_count = min(3, int(available_memory / (optimal_shard_size * 1024 * 1024) - 1))
        
        config = ShardConfig(
            shard_size_mb=optimal_shard_size,
            num_shards=num_shards,
            overlap_ratio=0.05,  # 5% overlap for boundary handling
            compression_enabled=compression_enabled,
            prefetch_count=max(1, prefetch_count),
            checkpoint_interval=max(1, num_shards // 10)
        )
        
        logger.info(f"Sharding config: {num_shards} shards of {optimal_shard_size}MB")
        return config
    
    def create_shards(
        self, 
        model_path: str,
        config: ShardConfig,
        output_dir: str
    ) -> List[ShardInfo]:
        """
        Create model shards based on configuration.
        
        Args:
            model_path: Path to model
            config: Sharding configuration
            output_dir: Directory to save shards
            
        Returns:
            List of ShardInfo objects
        """
        os.makedirs(output_dir, exist_ok=True)
        shards = []
        
        try:
            # Load model metadata
            model_info = self._load_model_metadata(model_path)
            total_params = model_info['total_params']
            layers = model_info['layers']
            
            # Calculate parameters per shard
            params_per_shard = total_params // config.num_shards
            overlap_params = int(params_per_shard * config.overlap_ratio)
            
            current_param = 0
            for shard_id in range(config.num_shards):
                # Calculate shard boundaries
                start_param = max(0, current_param - overlap_params)
                end_param = min(total_params, current_param + params_per_shard)
                
                # Find corresponding layers
                shard_layers = self._find_layers_in_range(
                    layers, 
                    start_param, 
                    end_param
                )
                
                # Calculate dependencies
                dependencies = []
                if shard_id > 0:
                    dependencies.append(shard_id - 1)
                if shard_id < config.num_shards - 1:
                    dependencies.append(shard_id + 1)
                
                # Create shard info
                shard_info = ShardInfo(
                    shard_id=shard_id,
                    start_param=start_param,
                    end_param=end_param,
                    size_bytes=(end_param - start_param) * 4,  # Assuming float32
                    layer_names=shard_layers,
                    dependencies=dependencies
                )
                
                shards.append(shard_info)
                self.shard_metadata[shard_id] = shard_info
                
                # Save shard
                self._save_shard(
                    model_path,
                    shard_info,
                    output_dir,
                    config.compression_enabled
                )
                
                current_param = end_param - overlap_params
                
        except Exception as e:
            logger.error(f"Failed to create shards: {e}")
            raise
        
        return shards
    
    def load_shard(
        self,
        shard_id: int,
        shard_dir: str,
        compressed: bool = False
    ) -> Any:
        """
        Load a specific shard into memory.
        
        Args:
            shard_id: ID of shard to load
            shard_dir: Directory containing shards
            compressed: Whether shard is compressed
            
        Returns:
            Loaded shard data
        """
        # Check memory before loading
        self._ensure_memory_available(shard_id)
        
        shard_path = os.path.join(shard_dir, f"shard_{shard_id}.pt")
        if compressed:
            shard_path += ".gz"
        
        logger.info(f"Loading shard {shard_id} from {shard_path}")
        
        try:
            if compressed:
                import gzip
                with gzip.open(shard_path, 'rb') as f:
                    shard_data = torch.load(f, map_location='cpu')
            else:
                shard_data = torch.load(shard_path, map_location='cpu')
            
            self.current_shards[shard_id] = shard_data
            
            # Update memory tracking
            self.memory_history.append(psutil.Process().memory_info().rss / 1024 / 1024)
            
            return shard_data
            
        except Exception as e:
            logger.error(f"Failed to load shard {shard_id}: {e}")
            raise
    
    def unload_shard(self, shard_id: int) -> None:
        """
        Unload a shard from memory.
        
        Args:
            shard_id: ID of shard to unload
        """
        if shard_id in self.current_shards:
            logger.info(f"Unloading shard {shard_id}")
            del self.current_shards[shard_id]
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_shard_iterator(
        self,
        shard_dir: str,
        config: ShardConfig
    ) -> Iterator[Tuple[int, Any]]:
        """
        Get iterator for processing shards sequentially.
        
        Args:
            shard_dir: Directory containing shards
            config: Sharding configuration
            
        Yields:
            Tuple of (shard_id, shard_data)
        """
        for shard_id in range(config.num_shards):
            # Prefetch next shards if configured
            if config.prefetch_count > 0:
                self._prefetch_shards(
                    shard_id,
                    config.prefetch_count,
                    shard_dir,
                    config.compression_enabled
                )
            
            # Load current shard
            shard_data = self.load_shard(
                shard_id,
                shard_dir,
                config.compression_enabled
            )
            
            yield shard_id, shard_data
            
            # Unload if not needed for dependencies
            if not self._is_needed_for_dependencies(shard_id):
                self.unload_shard(shard_id)
    
    def optimize_configuration(
        self,
        model_path: str,
        hardware_profile: Dict[str, Any]
    ) -> ShardConfig:
        """
        Optimize sharding configuration for specific hardware.
        
        Args:
            model_path: Path to model
            hardware_profile: Hardware specifications
            
        Returns:
            Optimized ShardConfig
        """
        # Extract hardware info
        total_memory_gb = hardware_profile.get('memory_gb', 64)
        disk_speed_mbps = hardware_profile.get('disk_speed_mbps', 100)
        cpu_cores = hardware_profile.get('cpu_cores', 8)
        
        # Get model size
        model_size = self._get_model_size(model_path)
        model_size_gb = model_size / (1024 ** 3)
        
        # Calculate optimal shard size based on hardware
        # Balance between memory usage and I/O overhead
        memory_per_shard_gb = total_memory_gb * self.target_memory_usage / 2
        io_optimal_size_gb = disk_speed_mbps * 0.01  # 1% of disk speed
        
        optimal_size_gb = min(
            memory_per_shard_gb,
            max(io_optimal_size_gb, 0.5)  # At least 500MB
        )
        
        optimal_size_mb = int(optimal_size_gb * 1024)
        optimal_size_mb = max(self.min_shard_size_mb, 
                              min(self.max_shard_size_mb, optimal_size_mb))
        
        # Calculate shards
        num_shards = max(1, int(np.ceil(model_size_gb / optimal_size_gb)))
        
        # Determine prefetch based on available memory
        memory_for_prefetch = (total_memory_gb * self.target_memory_usage - optimal_size_gb)
        prefetch_count = max(0, int(memory_for_prefetch / optimal_size_gb))
        
        # Enable compression for large shards or slow disks
        compression_enabled = (
            optimal_size_mb > self.compression_threshold_mb or
            disk_speed_mbps < 50
        )
        
        config = ShardConfig(
            shard_size_mb=optimal_size_mb,
            num_shards=num_shards,
            overlap_ratio=0.05,
            compression_enabled=compression_enabled,
            prefetch_count=min(3, prefetch_count),
            checkpoint_interval=max(1, num_shards // 10)
        )
        
        logger.info(f"Optimized config for {total_memory_gb}GB RAM: "
                   f"{num_shards} shards of {optimal_size_mb}MB")
        
        return config
    
    def _get_model_size(self, model_path: str) -> int:
        """Get total size of model in bytes"""
        if os.path.isfile(model_path):
            return os.path.getsize(model_path)
        elif os.path.isdir(model_path):
            total_size = 0
            for root, _, files in os.walk(model_path):
                for file in files:
                    if file.endswith(('.bin', '.safetensors', '.pt', '.pth')):
                        total_size += os.path.getsize(os.path.join(root, file))
            return total_size
        else:
            raise ValueError(f"Invalid model path: {model_path}")
    
    def _get_available_memory(self) -> int:
        """Get available system memory in bytes"""
        mem = psutil.virtual_memory()
        # Consider both available memory and target usage
        return int(mem.available * self.target_memory_usage)
    
    def _calculate_optimal_shard_size(
        self,
        model_size: int,
        available_memory: int
    ) -> int:
        """
        Calculate optimal shard size in MB.
        
        Args:
            model_size: Total model size in bytes
            available_memory: Available memory in bytes
            
        Returns:
            Optimal shard size in MB
        """
        # Target using 50% of available memory per shard (leaving room for processing)
        target_shard_size = available_memory // 2
        
        # But don't exceed 1/10 of model size (at least 10 shards)
        max_shard_size = model_size // 10
        
        optimal_size = min(target_shard_size, max_shard_size)
        optimal_size_mb = optimal_size // (1024 * 1024)
        
        # Apply bounds
        optimal_size_mb = max(self.min_shard_size_mb,
                              min(self.max_shard_size_mb, optimal_size_mb))
        
        return optimal_size_mb
    
    def _load_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """Load model metadata for sharding"""
        metadata = {
            'total_params': 0,
            'layers': []
        }
        
        # Try to load model config
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Extract model dimensions
            hidden_size = config.get('hidden_size', 4096)
            num_layers = config.get('num_hidden_layers', 32)
            vocab_size = config.get('vocab_size', 32000)
            
            # Estimate total parameters
            # Simplified estimation for transformer models
            params_per_layer = 12 * hidden_size * hidden_size  # Attention + FFN
            total_params = (
                vocab_size * hidden_size +  # Embedding
                params_per_layer * num_layers +  # Transformer layers
                vocab_size * hidden_size  # Output layer
            )
            
            metadata['total_params'] = total_params
            
            # Generate layer names
            for i in range(num_layers):
                metadata['layers'].append(f"layer_{i}")
        else:
            # Fallback: estimate from file size
            model_size = self._get_model_size(model_path)
            metadata['total_params'] = model_size // 4  # Assuming float32
            
            # Generate generic layer names
            num_layers = 32  # Default assumption
            for i in range(num_layers):
                metadata['layers'].append(f"layer_{i}")
        
        return metadata
    
    def _find_layers_in_range(
        self,
        layers: List[str],
        start_param: int,
        end_param: int
    ) -> List[str]:
        """Find which layers fall within parameter range"""
        # Simplified: distribute layers evenly
        total_layers = len(layers)
        if total_layers == 0:
            return []
        
        # Assuming uniform distribution of parameters
        params_per_layer = (end_param - start_param) / total_layers
        
        start_layer = int(start_param / params_per_layer)
        end_layer = int(end_param / params_per_layer)
        
        return layers[start_layer:end_layer + 1]
    
    def _save_shard(
        self,
        model_path: str,
        shard_info: ShardInfo,
        output_dir: str,
        compress: bool
    ) -> None:
        """Save a model shard to disk"""
        shard_path = os.path.join(output_dir, f"shard_{shard_info.shard_id}.pt")
        
        # Create mock shard data for now
        # In production, this would extract actual model parameters
        shard_data = {
            'shard_id': shard_info.shard_id,
            'layers': shard_info.layer_names,
            'params': {
                'start': shard_info.start_param,
                'end': shard_info.end_param
            },
            'metadata': {
                'created': time.time(),
                'model_path': model_path
            }
        }
        
        if compress:
            import gzip
            shard_path += ".gz"
            with gzip.open(shard_path, 'wb') as f:
                torch.save(shard_data, f)
        else:
            torch.save(shard_data, shard_path)
        
        logger.info(f"Saved shard {shard_info.shard_id} to {shard_path}")
    
    def _ensure_memory_available(self, shard_id: int) -> None:
        """Ensure enough memory is available to load shard"""
        if shard_id not in self.shard_metadata:
            return
        
        required_memory = self.shard_metadata[shard_id].size_bytes
        available_memory = psutil.virtual_memory().available
        
        if required_memory > available_memory * 0.9:  # Leave 10% buffer
            # Free up memory by unloading other shards
            for loaded_shard_id in list(self.current_shards.keys()):
                if loaded_shard_id != shard_id:
                    self.unload_shard(loaded_shard_id)
                    
                    # Check if enough memory freed
                    if psutil.virtual_memory().available > required_memory * 1.1:
                        break
    
    def _prefetch_shards(
        self,
        current_shard: int,
        prefetch_count: int,
        shard_dir: str,
        compressed: bool
    ) -> None:
        """Prefetch upcoming shards in background"""
        # Simple implementation - in production would use threading
        pass
    
    def _is_needed_for_dependencies(self, shard_id: int) -> bool:
        """Check if shard is needed for dependencies"""
        if shard_id not in self.shard_metadata:
            return False
        
        # Check if any loaded shards depend on this one
        for loaded_id in self.current_shards:
            if loaded_id in self.shard_metadata:
                if shard_id in self.shard_metadata[loaded_id].dependencies:
                    return True
        
        return False
    
    def get_memory_profile(self) -> Dict[str, Any]:
        """Get current memory profile"""
        process = psutil.Process()
        mem_info = process.memory_info()
        
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vms_mb': mem_info.vms / 1024 / 1024,
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'percent_used': psutil.virtual_memory().percent,
            'loaded_shards': list(self.current_shards.keys()),
            'history': self.memory_history[-100:]  # Last 100 measurements
        }