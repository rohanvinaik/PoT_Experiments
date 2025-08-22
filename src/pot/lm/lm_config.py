"""
Configuration management for Language Model Verification
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List


@dataclass
class LMVerifierConfig:
    """Configuration for LM Verifier."""
    
    # Model settings
    model_name: str = "gpt2"
    device: str = "cuda"
    
    # Challenge settings
    num_challenges: int = 10
    challenge_types: Optional[List[str]] = None  # None means all types
    difficulty_curve: str = "adaptive"
    
    # Verification settings
    verification_method: str = "sequential"  # 'sequential' or 'batch'
    distance_metric: str = "combined"
    distance_threshold: float = 0.15
    
    # Sequential testing parameters
    sprt_alpha: float = 0.05
    sprt_beta: float = 0.05
    sprt_p0: float = 0.5
    sprt_p1: float = 0.8
    max_trials: int = 100
    min_trials: int = 5
    
    # Normalization settings
    normalization_mode: str = "canonical"
    
    # Fuzzy matching settings
    fuzzy_threshold: float = 0.85
    fuzzy_method: str = "token_set_ratio"
    
    # Hash settings
    hash_type: str = "simhash"
    hash_size: int = 64
    
    # Template challenge settings
    template_categories: Optional[List[str]] = None
    use_dynamic_challenges: bool = True
    dynamic_topics: Optional[List[str]] = None
    
    # Output settings
    output_format: str = "json"
    save_detailed_results: bool = True
    plot_progress: bool = False
    
    def __post_init__(self):
        """Set default values for None fields."""
        if self.challenge_types is None:
            self.challenge_types = ['factual', 'reasoning', 'arithmetic', 'completion']
        
        if self.template_categories is None:
            self.template_categories = ['factual', 'reasoning', 'arithmetic', 'completion', 'logic', 'language']
        
        if self.dynamic_topics is None:
            self.dynamic_topics = ['math', 'logic', 'knowledge', 'coding', 'pattern']
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LMVerifierConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'LMVerifierConfig':
        """Load configuration from file."""
        import json
        import yaml
        from pathlib import Path
        
        path = Path(config_path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return cls.from_dict(config_dict)
    
    def save_to_file(self, config_path: str):
        """Save configuration to file."""
        import json
        import yaml
        from pathlib import Path
        
        path = Path(config_path)
        config_dict = self.to_dict()
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate numeric ranges
        if not (0 < self.sprt_alpha < 1):
            issues.append(f"sprt_alpha must be in (0, 1), got {self.sprt_alpha}")
        
        if not (0 < self.sprt_beta < 1):
            issues.append(f"sprt_beta must be in (0, 1), got {self.sprt_beta}")
        
        if not (0 < self.sprt_p0 < 1):
            issues.append(f"sprt_p0 must be in (0, 1), got {self.sprt_p0}")
        
        if not (0 < self.sprt_p1 < 1):
            issues.append(f"sprt_p1 must be in (0, 1), got {self.sprt_p1}")
        
        if self.sprt_p0 >= self.sprt_p1:
            issues.append(f"sprt_p1 ({self.sprt_p1}) must be greater than sprt_p0 ({self.sprt_p0})")
        
        if not (0 < self.fuzzy_threshold <= 1):
            issues.append(f"fuzzy_threshold must be in (0, 1], got {self.fuzzy_threshold}")
        
        if not (0 < self.distance_threshold <= 1):
            issues.append(f"distance_threshold must be in (0, 1], got {self.distance_threshold}")
        
        if self.num_challenges <= 0:
            issues.append(f"num_challenges must be positive, got {self.num_challenges}")
        
        if self.max_trials <= 0:
            issues.append(f"max_trials must be positive, got {self.max_trials}")
        
        if self.min_trials <= 0:
            issues.append(f"min_trials must be positive, got {self.min_trials}")
        
        if self.min_trials > self.max_trials:
            issues.append(f"min_trials ({self.min_trials}) must be <= max_trials ({self.max_trials})")
        
        # Validate enum values
        valid_verification_methods = ['sequential', 'batch']
        if self.verification_method not in valid_verification_methods:
            issues.append(f"verification_method must be one of {valid_verification_methods}, got {self.verification_method}")
        
        valid_difficulty_curves = ['linear', 'exponential', 'adaptive', 'random']
        if self.difficulty_curve not in valid_difficulty_curves:
            issues.append(f"difficulty_curve must be one of {valid_difficulty_curves}, got {self.difficulty_curve}")
        
        valid_distance_metrics = ['euclidean', 'cosine', 'combined', 'token_space']
        if self.distance_metric not in valid_distance_metrics:
            issues.append(f"distance_metric must be one of {valid_distance_metrics}, got {self.distance_metric}")
        
        valid_hash_types = ['simhash', 'minhash', 'lsh']
        if self.hash_type not in valid_hash_types:
            issues.append(f"hash_type must be one of {valid_hash_types}, got {self.hash_type}")
        
        valid_output_formats = ['json', 'yaml', 'csv']
        if self.output_format not in valid_output_formats:
            issues.append(f"output_format must be one of {valid_output_formats}, got {self.output_format}")
        
        return issues
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


# Preset configurations for common use cases
class PresetConfigs:
    """Preset configurations for common verification scenarios."""
    
    @staticmethod
    def quick_test() -> LMVerifierConfig:
        """Quick test configuration for development."""
        return LMVerifierConfig(
            num_challenges=5,
            verification_method="sequential",
            sprt_alpha=0.1,
            sprt_beta=0.1,
            max_trials=20,
            min_trials=3
        )
    
    @staticmethod
    def standard_verification() -> LMVerifierConfig:
        """Standard verification configuration."""
        return LMVerifierConfig(
            num_challenges=25,
            verification_method="sequential",
            sprt_alpha=0.05,
            sprt_beta=0.05,
            max_trials=100,
            min_trials=5
        )
    
    @staticmethod
    def comprehensive_verification() -> LMVerifierConfig:
        """Comprehensive verification configuration."""
        return LMVerifierConfig(
            num_challenges=50,
            verification_method="sequential",
            sprt_alpha=0.01,
            sprt_beta=0.01,
            max_trials=200,
            min_trials=10,
            save_detailed_results=True,
            plot_progress=True
        )
    
    @staticmethod
    def batch_verification() -> LMVerifierConfig:
        """Batch verification configuration (no early stopping)."""
        return LMVerifierConfig(
            num_challenges=30,
            verification_method="batch",
            distance_threshold=0.1,
            fuzzy_threshold=0.9
        )
    
    @staticmethod
    def high_security() -> LMVerifierConfig:
        """High security verification configuration."""
        return LMVerifierConfig(
            num_challenges=100,
            verification_method="sequential",
            sprt_alpha=0.001,
            sprt_beta=0.001,
            max_trials=500,
            min_trials=20,
            distance_threshold=0.05,
            fuzzy_threshold=0.95
        )


def load_config(config_source: Optional[str] = None) -> LMVerifierConfig:
    """
    Load configuration from various sources.
    
    Args:
        config_source: Path to config file, preset name, or None for default
        
    Returns:
        LMVerifierConfig instance
    """
    if config_source is None:
        return LMVerifierConfig()
    
    # Check if it's a preset name
    if hasattr(PresetConfigs, config_source):
        return getattr(PresetConfigs, config_source)()
    
    # Try to load from file
    try:
        return LMVerifierConfig.from_file(config_source)
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(f"Could not load config from '{config_source}': {e}")


def create_config_template(output_path: str = "lm_verifier_config.yaml"):
    """Create a configuration template file."""
    config = LMVerifierConfig()
    config.save_to_file(output_path)
    print(f"Configuration template created at: {output_path}")


if __name__ == "__main__":
    # Demo usage
    print("LM Verifier Configuration Demo")
    print("=" * 40)
    
    # Default config
    config = LMVerifierConfig()
    print(f"\nDefault config is valid: {config.is_valid()}")
    
    # Preset configs
    print("\nAvailable presets:")
    for preset_name in ['quick_test', 'standard_verification', 'comprehensive_verification', 'batch_verification', 'high_security']:
        preset_config = getattr(PresetConfigs, preset_name)()
        print(f"  {preset_name}: {preset_config.num_challenges} challenges, {preset_config.verification_method} method")
    
    # Validation demo
    print("\nValidation demo:")
    invalid_config = LMVerifierConfig(sprt_alpha=1.5, sprt_p0=0.8, sprt_p1=0.5)
    issues = invalid_config.validate()
    print(f"Invalid config has {len(issues)} issues:")
    for issue in issues:
        print(f"  - {issue}")