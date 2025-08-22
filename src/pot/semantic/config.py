"""
Configuration management for semantic verification.
Handles loading and validation of semantic verification settings from YAML files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging

from .library import ConceptLibrary
from .match import SemanticMatcher

logger = logging.getLogger(__name__)


@dataclass
class SemanticVerificationConfig:
    """Configuration for semantic verification."""
    
    # Core settings
    enabled: bool = True
    semantic_weight: float = 0.3
    
    # Library settings
    library_method: str = 'gaussian'
    library_dimension: int = 768
    library_path: Optional[str] = None
    
    # Matching settings
    matching_threshold: float = 0.7
    matching_primary_method: str = 'cosine'
    matching_fallback_method: str = 'euclidean'
    matching_use_cache: bool = True
    matching_cache_size: int = 10000
    
    # Drift detection
    drift_enabled: bool = True
    drift_max: float = 0.3
    drift_samples: int = 100
    
    # LM verifier settings
    lm_enabled: bool = True
    lm_dimension: int = 768
    lm_semantic_weight: float = 0.3
    lm_use_hidden_states: bool = True
    lm_max_samples: int = 10
    
    # Vision verifier settings
    vision_enabled: bool = True
    vision_dimension: int = 512
    vision_semantic_weight: float = 0.3
    vision_feature_layer: str = 'penultimate'
    vision_max_samples: int = 10
    
    # Reporting
    reporting_enabled: bool = True
    reporting_format: str = 'json'
    reporting_include_plots: bool = True
    reporting_output_dir: str = 'outputs/semantic_reports/'
    
    # Backward compatibility
    fallback_mode: str = 'distance_only'
    log_warnings: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SemanticVerificationConfig':
        """
        Create config from dictionary (e.g., from YAML).
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            SemanticVerificationConfig instance
        """
        config = cls()
        
        # Parse semantic_verification section
        if 'semantic_verification' in config_dict:
            sv = config_dict['semantic_verification']
            
            config.enabled = sv.get('enabled', True)
            config.semantic_weight = sv.get('semantic_weight', 0.3)
            
            # Library settings
            if 'library' in sv:
                lib = sv['library']
                config.library_method = lib.get('method', 'gaussian')
                config.library_dimension = lib.get('dimension', 768)
                config.library_path = lib.get('library_path')
            
            # Matching settings
            if 'matching' in sv:
                match = sv['matching']
                config.matching_threshold = match.get('threshold', 0.7)
                config.matching_primary_method = match.get('primary_method', 'cosine')
                config.matching_fallback_method = match.get('fallback_method', 'euclidean')
                config.matching_use_cache = match.get('use_cache', True)
                config.matching_cache_size = match.get('cache_size', 10000)
            
            # Drift detection
            if 'drift_detection' in sv:
                drift = sv['drift_detection']
                config.drift_enabled = drift.get('enabled', True)
                config.drift_max = drift.get('max_drift', 0.3)
                config.drift_samples = drift.get('drift_samples', 100)
            
            # Verifier integration
            if 'verifier_integration' in sv:
                vi = sv['verifier_integration']
                
                if 'lm' in vi:
                    lm = vi['lm']
                    config.lm_enabled = lm.get('enabled', True)
                    config.lm_dimension = lm.get('dimension', 768)
                    config.lm_semantic_weight = lm.get('semantic_weight', 0.3)
                    config.lm_use_hidden_states = lm.get('use_hidden_states', True)
                    config.lm_max_samples = lm.get('max_semantic_samples', 10)
                
                if 'vision' in vi:
                    vision = vi['vision']
                    config.vision_enabled = vision.get('enabled', True)
                    config.vision_dimension = vision.get('dimension', 512)
                    config.vision_semantic_weight = vision.get('semantic_weight', 0.3)
                    config.vision_feature_layer = vision.get('feature_layer', 'penultimate')
                    config.vision_max_samples = vision.get('max_semantic_samples', 10)
            
            # Reporting
            if 'reporting' in sv:
                report = sv['reporting']
                config.reporting_enabled = report.get('generate_reports', True)
                config.reporting_format = report.get('format', 'json')
                config.reporting_include_plots = report.get('include_plots', True)
                config.reporting_output_dir = report.get('output_dir', 'outputs/semantic_reports/')
        
        # Parse backward_compatibility section
        if 'backward_compatibility' in config_dict:
            bc = config_dict['backward_compatibility']
            config.fallback_mode = bc.get('fallback_mode', 'distance_only')
            config.log_warnings = bc.get('log_warnings', True)
        
        return config
    
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate semantic weight
        if not 0.0 <= self.semantic_weight <= 1.0:
            raise ValueError(f"semantic_weight must be in [0, 1], got {self.semantic_weight}")
        
        # Validate library method
        if self.library_method not in ['gaussian', 'hypervector']:
            raise ValueError(f"Invalid library_method: {self.library_method}")
        
        # Validate dimensions
        if self.library_dimension <= 0:
            raise ValueError(f"library_dimension must be positive, got {self.library_dimension}")
        
        # Validate matching methods
        valid_methods = ['cosine', 'euclidean', 'mahalanobis', 'hamming']
        if self.matching_primary_method not in valid_methods:
            raise ValueError(f"Invalid matching_primary_method: {self.matching_primary_method}")
        if self.matching_fallback_method not in valid_methods:
            raise ValueError(f"Invalid matching_fallback_method: {self.matching_fallback_method}")
        
        # Validate threshold
        if not 0.0 <= self.matching_threshold <= 1.0:
            raise ValueError(f"matching_threshold must be in [0, 1], got {self.matching_threshold}")
        
        # Validate fallback mode
        if self.fallback_mode not in ['distance_only', 'error', 'warning']:
            raise ValueError(f"Invalid fallback_mode: {self.fallback_mode}")
        
        return True


def load_semantic_config(config_path: Union[str, Path]) -> SemanticVerificationConfig:
    """
    Load semantic verification configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        SemanticVerificationConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = SemanticVerificationConfig.from_dict(config_dict)
        config.validate()
        
        logger.info(f"Loaded semantic verification config from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML config: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load config: {e}")


def create_semantic_components(config: SemanticVerificationConfig) -> tuple:
    """
    Create semantic verification components from configuration.
    
    Args:
        config: Semantic verification configuration
        
    Returns:
        Tuple of (ConceptLibrary, SemanticMatcher) or (None, None) if disabled
    """
    if not config.enabled:
        logger.info("Semantic verification is disabled in configuration")
        return None, None
    
    try:
        # Create or load concept library
        if config.library_path and os.path.exists(config.library_path):
            # Load existing library
            logger.info(f"Loading concept library from {config.library_path}")
            library = ConceptLibrary(config.library_path)
        else:
            # Create new library
            logger.info(f"Creating new concept library (dim={config.library_dimension}, method={config.library_method})")
            library = ConceptLibrary(
                dim=config.library_dimension,
                method=config.library_method
            )
        
        # Create semantic matcher
        matcher = SemanticMatcher(
            library=library,
            threshold=config.matching_threshold
        )
        
        # Configure matcher cache
        if not config.matching_use_cache:
            matcher._similarity_cache.clear()
        
        logger.info("Successfully created semantic verification components")
        return library, matcher
        
    except Exception as e:
        if config.fallback_mode == 'error':
            raise
        elif config.fallback_mode == 'warning' and config.log_warnings:
            logger.warning(f"Failed to create semantic components: {e}")
        
        return None, None


def integrate_with_verifier(verifier_type: str, verifier_instance: Any, 
                           config: SemanticVerificationConfig) -> None:
    """
    Integrate semantic verification with an existing verifier instance.
    
    Args:
        verifier_type: Type of verifier ('lm' or 'vision')
        verifier_instance: Verifier instance to enhance
        config: Semantic verification configuration
    """
    if not config.enabled:
        return
    
    # Check if verifier type is enabled
    if verifier_type == 'lm' and not config.lm_enabled:
        return
    elif verifier_type == 'vision' and not config.vision_enabled:
        return
    
    # Create semantic components
    library, matcher = create_semantic_components(config)
    
    if library is None or matcher is None:
        return
    
    # Set semantic attributes on verifier
    verifier_instance.semantic_library = library
    verifier_instance.semantic_matcher = matcher
    
    if verifier_type == 'lm':
        verifier_instance.semantic_weight = config.lm_semantic_weight
        logger.info(f"Integrated semantic verification with LM verifier (weight={config.lm_semantic_weight})")
    elif verifier_type == 'vision':
        verifier_instance.semantic_weight = config.vision_semantic_weight
        logger.info(f"Integrated semantic verification with Vision verifier (weight={config.vision_semantic_weight})")


def get_default_config() -> SemanticVerificationConfig:
    """
    Get default semantic verification configuration.
    
    Returns:
        Default SemanticVerificationConfig instance
    """
    return SemanticVerificationConfig()


# Example usage functions

def example_usage():
    """Example of how to use semantic verification configuration."""
    
    # Load config from file
    config = load_semantic_config('configs/semantic_verification.yaml')
    
    # Create semantic components
    library, matcher = create_semantic_components(config)
    
    if library and matcher:
        # Use with verifiers
        from pot.lm.verifier import LMVerifier
        from pot.lm.models import LM
        
        # Create a mock reference model
        class MockLM(LM):
            def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
                return f"Response: {prompt}"
        
        ref_model = MockLM()
        
        # Create verifier with semantic verification
        if config.lm_enabled:
            verifier = LMVerifier(
                ref_model,
                semantic_library=library,
                semantic_weight=config.lm_semantic_weight
            )
        else:
            verifier = LMVerifier(ref_model)
        
        print(f"Created LM verifier with semantic verification: {config.lm_enabled}")


if __name__ == "__main__":
    # Run example if executed directly
    example_usage()