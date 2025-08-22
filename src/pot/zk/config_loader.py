"""
Configuration loader for ZK proof system.

This module loads and manages deployment configurations from YAML files,
providing mode-specific settings for development, production, and benchmarking.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import socket
import multiprocessing as mp

logger = logging.getLogger(__name__)


@dataclass
class ZKConfig:
    """ZK proof system configuration."""
    
    # Mode
    mode: str = "development"
    
    # Circuit settings
    max_constraints: int = 10000
    optimization_level: int = 0
    debug_mode: bool = True
    witness_generation_checks: bool = True
    custom_gates: bool = False
    lookup_tables: bool = False
    
    # Prover settings
    num_workers: int = 2
    use_processes: bool = False
    batch_size: int = 4
    max_proof_time_ms: int = 30000
    prefetch_witnesses: bool = False
    
    # Cache settings
    enable_memory_cache: bool = True
    memory_cache_size_mb: int = 50
    enable_disk_cache: bool = False
    disk_cache_path: str = "./.zk_cache"
    cache_ttl_seconds: int = 300
    
    # Aggregation settings
    enable_aggregation: bool = False
    max_aggregation_batch: int = 4
    recursive_depth: int = 1
    
    # Monitoring settings
    enable_monitoring: bool = True
    log_level: str = "DEBUG"
    metrics_interval_seconds: int = 5
    persist_metrics: bool = False
    metrics_path: str = "./zk_metrics"
    
    # LoRA settings
    lora_supported_ranks: List[int] = field(default_factory=lambda: [4, 8])
    lora_precompute: bool = False
    lora_alpha_scaling: bool = True
    
    # Resource limits
    max_memory_gb: float = 2.0
    max_cpu_percent: int = 50
    timeout_minutes: int = 5
    
    # Alert thresholds
    alert_proof_time_ms: int = 10000
    alert_failure_rate: float = 0.1
    alert_cpu_percent: int = 90
    alert_memory_gb: float = 4.0
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=dict)


class ConfigLoader:
    """
    Loads and manages ZK system configuration.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path or Path("configs/zk_config.yaml")
        self.config_data: Dict[str, Any] = {}
        self.current_mode: str = "development"
        self.config: ZKConfig = ZKConfig()
        
        # Load configuration
        if self.config_path.exists():
            self.load_config()
        else:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
    
    def load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config_data = yaml.safe_load(f)
            
            # Determine mode
            self.current_mode = self._determine_mode()
            logger.info(f"Loading configuration in {self.current_mode} mode")
            
            # Load mode-specific config
            self.config = self._build_config(self.current_mode)
            
            # Apply overrides
            self._apply_overrides()
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = ZKConfig()
    
    def _determine_mode(self) -> str:
        """Determine which mode to use."""
        # Check environment variable
        if 'ZK_MODE' in os.environ:
            return os.environ['ZK_MODE']
        
        # Check auto-detection rules
        if self.config_data.get('mode_selection', {}).get('auto_detect'):
            # Production detection
            if os.environ.get('NODE_ENV') == 'production':
                return 'production'
            
            # Benchmark detection
            if os.environ.get('BENCHMARK') == 'true':
                return 'benchmarking'
            
            # CI detection
            if os.environ.get('CI') == 'true':
                return 'development'
            
            # Hostname detection
            hostname = socket.gethostname().lower()
            if 'prod' in hostname:
                return 'production'
            elif 'dev' in hostname:
                return 'development'
        
        # Use default
        return self.config_data.get('default_mode', 'development')
    
    def _build_config(self, mode: str) -> ZKConfig:
        """Build configuration for specified mode."""
        if mode not in self.config_data:
            logger.warning(f"Mode {mode} not found, using development")
            mode = 'development'
        
        mode_config = self.config_data[mode]
        config = ZKConfig(mode=mode)
        
        # Circuit settings
        circuit = mode_config.get('circuit', {})
        config.max_constraints = circuit.get('max_constraints', 10000)
        config.optimization_level = circuit.get('optimization_level', 0)
        config.debug_mode = circuit.get('debug_mode', True)
        config.witness_generation_checks = circuit.get('witness_generation_checks', True)
        config.custom_gates = circuit.get('custom_gates', False)
        config.lookup_tables = circuit.get('lookup_tables', False)
        
        # Prover settings
        prover = mode_config.get('prover', {})
        num_workers = prover.get('num_workers', 2)
        config.num_workers = mp.cpu_count() if num_workers == -1 else num_workers
        config.use_processes = prover.get('use_processes', False)
        config.batch_size = prover.get('batch_size', 4)
        config.max_proof_time_ms = prover.get('max_proof_time_ms', 30000)
        config.prefetch_witnesses = prover.get('prefetch_witnesses', False)
        
        # Cache settings
        cache = mode_config.get('cache', {})
        config.enable_memory_cache = cache.get('enable_memory_cache', True)
        config.memory_cache_size_mb = cache.get('memory_cache_size_mb', 50)
        config.enable_disk_cache = cache.get('enable_disk_cache', False)
        config.disk_cache_path = cache.get('disk_cache_path', './.zk_cache')
        config.cache_ttl_seconds = cache.get('cache_ttl_seconds', 300)
        
        # Aggregation settings
        aggregation = mode_config.get('aggregation', {})
        config.enable_aggregation = aggregation.get('enable', False)
        config.max_aggregation_batch = aggregation.get('max_batch_size', 4)
        config.recursive_depth = aggregation.get('recursive_depth', 1)
        
        # Monitoring settings
        monitoring = mode_config.get('monitoring', {})
        config.enable_monitoring = monitoring.get('enable', True)
        config.log_level = monitoring.get('log_level', 'DEBUG')
        config.metrics_interval_seconds = monitoring.get('metrics_interval_seconds', 5)
        config.persist_metrics = monitoring.get('persist_metrics', False)
        config.metrics_path = monitoring.get('metrics_path', './zk_metrics')
        
        # Alert thresholds
        alerts = monitoring.get('alert_thresholds', {})
        config.alert_proof_time_ms = alerts.get('proof_time_ms', 10000)
        config.alert_failure_rate = alerts.get('failure_rate', 0.1)
        config.alert_cpu_percent = alerts.get('cpu_percent', 90)
        config.alert_memory_gb = alerts.get('memory_gb', 4.0)
        
        # LoRA settings
        lora = mode_config.get('lora', {})
        config.lora_supported_ranks = lora.get('supported_ranks', [4, 8])
        config.lora_precompute = lora.get('precompute_circuits', False)
        config.lora_alpha_scaling = lora.get('alpha_scaling', True)
        
        # Resource limits
        resources = mode_config.get('resources', {})
        config.max_memory_gb = resources.get('max_memory_gb', 2.0)
        config.max_cpu_percent = resources.get('max_cpu_percent', 50)
        config.timeout_minutes = resources.get('timeout_minutes', 5)
        
        # Feature flags
        features_data = self.config_data.get('features', {})
        config.features = {
            'lora_support': features_data.get('stable', {}).get('lora_support', True),
            'proof_aggregation': features_data.get('stable', {}).get('proof_aggregation', True),
            'parallel_proving': features_data.get('stable', {}).get('parallel_proving', True),
            'poseidon_hash': features_data.get('stable', {}).get('poseidon_hash', True),
            'adaptive_batching': features_data.get('beta', {}).get('adaptive_batching', False),
            'smart_caching': features_data.get('beta', {}).get('smart_caching', False),
            'proof_compression': features_data.get('beta', {}).get('proof_compression', False),
            'recursive_snarks': features_data.get('experimental', {}).get('recursive_snarks', False),
            'gpu_acceleration': features_data.get('experimental', {}).get('gpu_acceleration', False),
        }
        
        return config
    
    def _apply_overrides(self):
        """Apply environment variable and CLI overrides."""
        # Environment variable overrides
        overrides = self.config_data.get('overrides', {}).get('env_vars', [])
        
        for override in overrides:
            env_name = override['name']
            if env_name in os.environ:
                value = os.environ[env_name]
                self._apply_override_value(override['path'], value)
        
        # Specific known overrides
        if 'ZK_NUM_WORKERS' in os.environ:
            self.config.num_workers = int(os.environ['ZK_NUM_WORKERS'])
        
        if 'ZK_CACHE_SIZE' in os.environ:
            self.config.memory_cache_size_mb = int(os.environ['ZK_CACHE_SIZE'])
        
        if 'ZK_LOG_LEVEL' in os.environ:
            self.config.log_level = os.environ['ZK_LOG_LEVEL']
        
        if 'ZK_BATCH_SIZE' in os.environ:
            self.config.batch_size = int(os.environ['ZK_BATCH_SIZE'])
    
    def _apply_override_value(self, path: str, value: Any):
        """Apply override value to config path."""
        # Simple implementation - extend as needed
        if path == "$.prover.num_workers":
            self.config.num_workers = int(value)
        elif path == "$.cache.memory_cache_size_mb":
            self.config.memory_cache_size_mb = int(value)
        elif path == "$.monitoring.log_level":
            self.config.log_level = value
    
    def get_hardware_optimizations(self) -> Dict[str, bool]:
        """Get hardware-specific optimizations."""
        hw_opts = self.config_data.get('hardware_optimizations', {})
        
        # Detect CPU type
        cpu_info = self._get_cpu_info()
        
        if 'intel' in cpu_info.lower():
            opts = hw_opts.get('intel', {})
        elif 'amd' in cpu_info.lower():
            opts = hw_opts.get('amd', {})
        elif 'apple' in cpu_info.lower():
            opts = hw_opts.get('apple_silicon', {})
        else:
            opts = {}
        
        return {
            'avx2': opts.get('enable_avx2', False),
            'avx512': opts.get('enable_avx512', False),
            'neon': opts.get('enable_neon', False),
            'numa_aware': opts.get('numa_aware', False),
            'gpu': hw_opts.get('gpu', {}).get('enable', False)
        }
    
    def _get_cpu_info(self) -> str:
        """Get CPU information."""
        try:
            import platform
            return platform.processor()
        except:
            return "unknown"
    
    def get_cloud_config(self, provider: str) -> Dict[str, Any]:
        """Get cloud-specific configuration."""
        cloud = self.config_data.get('cloud', {})
        
        if provider == 'aws':
            return cloud.get('aws', {})
        elif provider == 'gcp':
            return cloud.get('gcp', {})
        elif provider == 'azure':
            return cloud.get('azure', {})
        elif provider == 'kubernetes':
            return cloud.get('kubernetes', {})
        
        return {}
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return self.config_data.get('security', {})
    
    def validate(self) -> List[str]:
        """Validate configuration and return warnings."""
        warnings = []
        
        # Check worker count
        if self.config.num_workers > mp.cpu_count():
            warnings.append(f"num_workers ({self.config.num_workers}) exceeds CPU count ({mp.cpu_count()})")
        
        # Check memory limits
        total_memory = self._get_total_memory_gb()
        if self.config.max_memory_gb > total_memory:
            warnings.append(f"max_memory_gb ({self.config.max_memory_gb}) exceeds system memory ({total_memory:.1f}GB)")
        
        # Check cache size
        if self.config.memory_cache_size_mb > self.config.max_memory_gb * 1024:
            warnings.append("Cache size exceeds memory limit")
        
        # Check feature compatibility
        if self.config.enable_aggregation and not self.config.features.get('proof_aggregation'):
            warnings.append("Aggregation enabled but feature flag is disabled")
        
        # Production mode checks
        if self.current_mode == 'production':
            if self.config.debug_mode:
                warnings.append("Debug mode enabled in production")
            if self.config.witness_generation_checks:
                warnings.append("Witness checks enabled in production (impacts performance)")
        
        return warnings
    
    def _get_total_memory_gb(self) -> float:
        """Get total system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except:
            return 8.0  # Default assumption
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'mode': self.config.mode,
            'circuit': {
                'max_constraints': self.config.max_constraints,
                'optimization_level': self.config.optimization_level,
                'debug_mode': self.config.debug_mode,
            },
            'prover': {
                'num_workers': self.config.num_workers,
                'use_processes': self.config.use_processes,
                'batch_size': self.config.batch_size,
                'max_proof_time_ms': self.config.max_proof_time_ms,
            },
            'cache': {
                'enable_memory_cache': self.config.enable_memory_cache,
                'memory_cache_size_mb': self.config.memory_cache_size_mb,
                'enable_disk_cache': self.config.enable_disk_cache,
            },
            'features': self.config.features
        }
    
    def print_config(self):
        """Print current configuration."""
        print(f"\n{'='*60}")
        print(f"ZK System Configuration - {self.current_mode.upper()} Mode")
        print(f"{'='*60}")
        
        print(f"\n📋 Circuit Settings:")
        print(f"  Max constraints:    {self.config.max_constraints:,}")
        print(f"  Optimization level: {self.config.optimization_level}")
        print(f"  Debug mode:         {self.config.debug_mode}")
        
        print(f"\n⚙️  Prover Settings:")
        print(f"  Workers:       {self.config.num_workers}")
        print(f"  Use processes: {self.config.use_processes}")
        print(f"  Batch size:    {self.config.batch_size}")
        print(f"  Max time:      {self.config.max_proof_time_ms}ms")
        
        print(f"\n💾 Cache Settings:")
        print(f"  Memory cache: {self.config.enable_memory_cache} ({self.config.memory_cache_size_mb}MB)")
        print(f"  Disk cache:   {self.config.enable_disk_cache}")
        
        print(f"\n🚀 Features:")
        for feature, enabled in self.config.features.items():
            status = "✓" if enabled else "✗"
            print(f"  [{status}] {feature}")
        
        # Validate and show warnings
        warnings = self.validate()
        if warnings:
            print(f"\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        print(f"\n{'='*60}")


# Global config instance
_global_config: Optional[ConfigLoader] = None


def get_config(config_path: Optional[Path] = None) -> ZKConfig:
    """Get global configuration."""
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader(config_path)
    return _global_config.config


def reload_config(config_path: Optional[Path] = None):
    """Reload configuration from file."""
    global _global_config
    _global_config = ConfigLoader(config_path)
    return _global_config.config


def set_mode(mode: str):
    """Set configuration mode."""
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader()
    
    _global_config.current_mode = mode
    _global_config.config = _global_config._build_config(mode)
    _global_config._apply_overrides()
    
    logger.info(f"Switched to {mode} mode")


def print_current_config():
    """Print current configuration."""
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader()
    
    _global_config.print_config()


if __name__ == "__main__":
    # Example usage
    print("Loading ZK configuration...")
    
    # Load default config
    config = get_config()
    print_current_config()
    
    # Try different modes
    print("\nTesting mode switching...")
    for mode in ['development', 'production', 'benchmarking']:
        print(f"\nSwitching to {mode} mode:")
        set_mode(mode)
        config = get_config()
        print(f"  Workers: {config.num_workers}, Batch: {config.batch_size}, Cache: {config.memory_cache_size_mb}MB")