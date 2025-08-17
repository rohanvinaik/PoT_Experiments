"""
Configuration system for reliable PoT validation experiments.
Provides standardized settings for reproducible testing.
"""

import time
import hashlib
from typing import Dict, Any, List
from dataclasses import dataclass
from .test_models import create_test_model


@dataclass
class ValidationConfig:
    """Configuration for validation experiments."""
    
    # Model settings
    model_type: str = "deterministic"
    model_seed: int = 42
    model_count: int = 5
    
    # Verification settings
    verification_types: List[str] = None
    verification_depths: List[str] = None
    
    # Challenge settings
    challenge_dimensions: List[int] = None
    challenge_count: int = 3
    
    # Performance settings
    performance_iterations: int = 3
    max_history_size: int = 100
    
    # Output settings
    generate_reports: bool = True
    save_detailed_logs: bool = True
    
    def __post_init__(self):
        """Set default values for list fields."""
        if self.verification_types is None:
            self.verification_types = ['exact', 'fuzzy', 'statistical']
        if self.verification_depths is None:
            self.verification_depths = ['quick', 'standard', 'comprehensive']
        if self.challenge_dimensions is None:
            self.challenge_dimensions = [100, 500, 1000]


def generate_session_seed() -> int:
    """
    Generate a session-based seed for reproducible but varied testing.
    Uses current timestamp to create different seeds while maintaining
    reproducibility within a session.
    
    Returns:
        Integer seed based on current time
    """
    # Use current time (rounded to nearest minute) to create session-based seed
    timestamp = int(time.time() // 60)  # Changes every minute
    # Hash to get better distribution and limit to reasonable range
    hash_obj = hashlib.md5(str(timestamp).encode())
    seed = int(hash_obj.hexdigest()[:8], 16) % 10000  # Limit to 0-9999 range
    return seed


def get_reliable_test_config() -> ValidationConfig:
    """
    Get configuration optimized for reliable, reproducible testing.
    Uses session-based seeding for varied but reproducible results.
    
    Returns:
        Validation configuration with deterministic settings
    """
    session_seed = generate_session_seed()
    
    return ValidationConfig(
        model_type="deterministic",
        model_seed=session_seed,
        model_count=3,  # Smaller count for faster, reliable tests
        verification_types=['fuzzy'],  # Focus on most important type
        verification_depths=['quick', 'standard'],  # Skip comprehensive for speed
        challenge_dimensions=[100, 500],  # Smaller dimensions for speed
        challenge_count=2,
        performance_iterations=2,
        max_history_size=50,
        generate_reports=True,
        save_detailed_logs=False  # Reduce log volume
    )


def get_comprehensive_test_config() -> ValidationConfig:
    """
    Get configuration for comprehensive system validation.
    
    Returns:
        Validation configuration with full testing coverage
    """
    return ValidationConfig(
        model_type="deterministic",
        model_seed=42,
        model_count=10,
        verification_types=['exact', 'fuzzy', 'statistical'],
        verification_depths=['quick', 'standard', 'comprehensive'],
        challenge_dimensions=[100, 500, 1000, 5000],
        challenge_count=5,
        performance_iterations=5,
        max_history_size=100,
        generate_reports=True,
        save_detailed_logs=True
    )


def create_test_models_from_config(config: ValidationConfig) -> List[Any]:
    """
    Create test models based on configuration.
    
    Args:
        config: Validation configuration
        
    Returns:
        List of test model instances
    """
    models = []
    
    for i in range(config.model_count):
        model_kwargs = {
            'model_id': f"test_model_{i}",
            'seed': config.model_seed + i  # Ensure each model is unique but deterministic
        }
        
        if config.model_type == "linear":
            model_kwargs.update({
                'input_dim': 10,
                'output_dim': 10
            })
        
        model = create_test_model(config.model_type, **model_kwargs)
        models.append(model)
    
    return models