"""
Vision Verifier Configuration
Provides dataclass-based configuration for vision model verification.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class VisionVerifierConfig:
    """Configuration for Vision Verifier."""
    
    # Model settings
    model_name: str = "resnet18"
    device: str = "cuda"
    model_path: Optional[str] = None
    
    # Challenge settings
    num_challenges: int = 10
    challenge_types: List[str] = None
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    
    # Frequency challenge settings
    frequency_bands: List[str] = None
    frequency_range: Tuple[float, float] = (0.1, 10.0)
    frequency_components: int = 5
    gabor_orientations: int = 8
    gabor_scales: int = 4
    
    # Texture challenge settings
    texture_types: List[str] = None
    perlin_octaves: int = 4
    perlin_persistence: float = 0.5
    voronoi_points: int = 50
    fractal_iterations: int = 100
    
    # Natural image challenge settings
    natural_scene_types: List[str] = None
    landscape_complexity: float = 0.7
    cloud_density: float = 0.5
    
    # Verification settings
    verification_method: str = "sequential"  # 'sequential' or 'batch'
    distance_metric: str = "combined"  # 'cosine', 'kl', 'js', 'combined'
    distance_threshold: float = 0.2
    confidence_threshold: float = 0.95
    early_stopping: bool = True
    max_challenges: int = 100
    
    # Sequential testing parameters (SPRT)
    sprt_alpha: float = 0.05  # Type I error rate
    sprt_beta: float = 0.05   # Type II error rate
    sprt_p0: float = 0.5      # Null hypothesis probability
    sprt_p1: float = 0.8      # Alternative hypothesis probability
    
    # Canonicalization settings
    normalization: str = "softmax"  # 'softmax', 'zscore', 'minmax', 'none'
    temperature: float = 1.0
    smoothing: float = 0.1    # Label smoothing for stability
    
    # Probe settings
    probe_layers: List[str] = None
    pool_method: str = "adaptive_avg"  # 'adaptive_avg', 'adaptive_max', 'global_avg'
    normalize_embeddings: bool = True
    embedding_dim: Optional[int] = None
    stability_mode: bool = True
    
    # Distance computation settings
    distance_config: Dict[str, Any] = None
    
    # Performance settings
    num_workers: int = 4
    pin_memory: bool = True
    use_mixed_precision: bool = False
    
    # Logging and output settings
    verbose: bool = True
    save_intermediates: bool = False
    output_dir: Optional[str] = None
    log_level: str = "INFO"
    
    # Security and robustness settings
    max_input_size: Tuple[int, int] = (512, 512)
    input_validation: bool = True
    deterministic: bool = True
    random_seed: int = 42
    
    def __post_init__(self):
        """Initialize default values for list fields."""
        if self.challenge_types is None:
            self.challenge_types = ['frequency', 'texture', 'natural']
        
        if self.frequency_bands is None:
            self.frequency_bands = ['low', 'mid', 'high', 'mixed']
        
        if self.texture_types is None:
            self.texture_types = ['perlin', 'voronoi', 'fractal', 'cellular']
        
        if self.natural_scene_types is None:
            self.natural_scene_types = ['landscape', 'clouds', 'abstract', 'water']
        
        if self.probe_layers is None:
            self.probe_layers = ['early', 'mid', 'late', 'penultimate', 'final']
        
        if self.distance_config is None:
            self.distance_config = {
                'epsilon': 1e-10,
                'temperature': self.temperature,
                'normalize_features': self.normalize_embeddings,
                'batch_size': 1000
            }
        
        # Validation
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.num_challenges <= 0:
            raise ValueError("num_challenges must be positive")
        
        if not 0 < self.sprt_alpha < 1:
            raise ValueError("sprt_alpha must be between 0 and 1")
        
        if not 0 < self.sprt_beta < 1:
            raise ValueError("sprt_beta must be between 0 and 1")
        
        if not 0 < self.sprt_p0 < 1:
            raise ValueError("sprt_p0 must be between 0 and 1")
        
        if not 0 < self.sprt_p1 < 1:
            raise ValueError("sprt_p1 must be between 0 and 1")
        
        if self.sprt_p0 >= self.sprt_p1:
            raise ValueError("sprt_p1 must be greater than sprt_p0")
        
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        
        if self.distance_threshold < 0:
            raise ValueError("distance_threshold must be non-negative")
        
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if len(self.image_size) != 2 or any(s <= 0 for s in self.image_size):
            raise ValueError("image_size must be a tuple of two positive integers")
        
        if len(self.max_input_size) != 2 or any(s <= 0 for s in self.max_input_size):
            raise ValueError("max_input_size must be a tuple of two positive integers")
        
        if self.verification_method not in ['sequential', 'batch']:
            raise ValueError("verification_method must be 'sequential' or 'batch'")
        
        if self.normalization not in ['softmax', 'zscore', 'minmax', 'none']:
            raise ValueError("normalization must be one of 'softmax', 'zscore', 'minmax', 'none'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VisionVerifierConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'VisionVerifierConfig':
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Convert specific lists back to tuples
            tuple_fields = ['image_size', 'max_input_size', 'frequency_range']
            for field in tuple_fields:
                if field in config_dict and isinstance(config_dict[field], list):
                    config_dict[field] = tuple(config_dict[field])
            
            return cls.from_dict(config_dict)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML configuration files")
    
    @classmethod
    def from_json(cls, json_path: str) -> 'VisionVerifierConfig':
        """Load configuration from JSON file."""
        import json
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        try:
            import yaml
            
            # Convert tuples to lists for YAML serialization
            config_dict = self.to_dict()
            def convert_tuples(obj):
                if isinstance(obj, tuple):
                    return list(obj)
                elif isinstance(obj, dict):
                    return {k: convert_tuples(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_tuples(item) for item in obj]
                return obj
            
            yaml_dict = convert_tuples(config_dict)
            
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_dict, f, default_flow_style=False, indent=2)
        except ImportError:
            raise ImportError("PyYAML is required to save YAML configuration files")
    
    def save_json(self, json_path: str):
        """Save configuration to JSON file."""
        import json
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_challenge_config(self, challenge_type: str) -> Dict[str, Any]:
        """Get configuration for specific challenge type."""
        if challenge_type == 'frequency':
            return {
                'frequency_bands': self.frequency_bands,
                'frequency_range': self.frequency_range,
                'frequency_components': self.frequency_components,
                'gabor_orientations': self.gabor_orientations,
                'gabor_scales': self.gabor_scales,
                'image_size': self.image_size
            }
        elif challenge_type == 'texture':
            return {
                'texture_types': self.texture_types,
                'perlin_octaves': self.perlin_octaves,
                'perlin_persistence': self.perlin_persistence,
                'voronoi_points': self.voronoi_points,
                'fractal_iterations': self.fractal_iterations,
                'image_size': self.image_size
            }
        elif challenge_type == 'natural':
            return {
                'natural_scene_types': self.natural_scene_types,
                'landscape_complexity': self.landscape_complexity,
                'cloud_density': self.cloud_density,
                'image_size': self.image_size
            }
        else:
            raise ValueError(f"Unknown challenge type: {challenge_type}")
    
    def get_probe_config(self) -> Dict[str, Any]:
        """Get probe configuration."""
        return {
            'extract_from': ['backbone', 'neck', 'head'],
            'layer_types': ['Conv2d', 'Linear', 'LayerNorm', 'BatchNorm2d'],
            'specific_layers': self.probe_layers,
            'pool_method': self.pool_method,
            'normalize': self.normalize_embeddings,
            'max_layers': len(self.probe_layers),
            'embedding_dim': self.embedding_dim,
            'stability_mode': self.stability_mode
        }
    
    def get_sequential_config(self) -> Dict[str, Any]:
        """Get sequential testing configuration."""
        return {
            'alpha': self.sprt_alpha,
            'beta': self.sprt_beta,
            'p0': self.sprt_p0,
            'p1': self.sprt_p1,
            'early_stopping': self.early_stopping,
            'max_challenges': self.max_challenges
        }
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"VisionVerifierConfig has no attribute '{key}'")
        
        # Re-validate after update
        self._validate_config()
    
    def copy(self) -> 'VisionVerifierConfig':
        """Create a copy of the configuration."""
        return VisionVerifierConfig.from_dict(self.to_dict())


# Predefined configurations for common use cases
class VisionConfigPresets:
    """Predefined configuration presets."""
    
    @staticmethod
    def quick_verification() -> VisionVerifierConfig:
        """Quick verification with minimal challenges."""
        return VisionVerifierConfig(
            num_challenges=5,
            challenge_types=['frequency'],
            verification_method='batch',
            early_stopping=False,
            verbose=False
        )
    
    @staticmethod
    def standard_verification() -> VisionVerifierConfig:
        """Standard verification configuration."""
        return VisionVerifierConfig(
            num_challenges=10,
            challenge_types=['frequency', 'texture'],
            verification_method='sequential',
            early_stopping=True,
            confidence_threshold=0.90
        )
    
    @staticmethod
    def comprehensive_verification() -> VisionVerifierConfig:
        """Comprehensive verification with all challenge types."""
        return VisionVerifierConfig(
            num_challenges=25,
            challenge_types=['frequency', 'texture', 'natural'],
            verification_method='sequential',
            early_stopping=True,
            confidence_threshold=0.95,
            max_challenges=50,
            save_intermediates=True
        )
    
    @staticmethod
    def research_verification() -> VisionVerifierConfig:
        """Research-grade verification with extensive probing."""
        return VisionVerifierConfig(
            num_challenges=50,
            challenge_types=['frequency', 'texture', 'natural'],
            verification_method='sequential',
            probe_layers=['early', 'early_mid', 'mid', 'late', 'penultimate', 'final'],
            distance_metric='combined',
            save_intermediates=True,
            confidence_threshold=0.99,
            max_challenges=100
        )
    
    @staticmethod
    def production_verification() -> VisionVerifierConfig:
        """Production-ready verification optimized for speed."""
        return VisionVerifierConfig(
            num_challenges=8,
            challenge_types=['frequency', 'texture'],
            verification_method='batch',
            early_stopping=False,
            use_mixed_precision=True,
            verbose=False,
            save_intermediates=False
        )


# Utility functions for configuration management
def load_config(config_path: str) -> VisionVerifierConfig:
    """Load configuration from file (auto-detect format)."""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        return VisionVerifierConfig.from_yaml(config_path)
    elif config_path.endswith('.json'):
        return VisionVerifierConfig.from_json(config_path)
    else:
        raise ValueError("Unsupported configuration file format. Use .yaml, .yml, or .json")


def create_config_template(output_path: str, format: str = 'yaml'):
    """Create a configuration template file."""
    config = VisionVerifierConfig()
    
    if format.lower() == 'yaml':
        config.save_yaml(output_path)
    elif format.lower() == 'json':
        config.save_json(output_path)
    else:
        raise ValueError("Format must be 'yaml' or 'json'")


def validate_config_file(config_path: str) -> bool:
    """Validate a configuration file."""
    try:
        config = load_config(config_path)
        config._validate_config()
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


# Export commonly used configurations
__all__ = [
    'VisionVerifierConfig',
    'VisionConfigPresets', 
    'load_config',
    'create_config_template',
    'validate_config_file'
]