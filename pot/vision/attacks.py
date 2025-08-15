"""Attack implementations for vision models."""

# Import shared attack implementations
from pot.core.attacks import (
    targeted_finetune,
    limited_distillation,
    wrapper_attack,
    extraction_attack,
    WrapperConfig,
)

def wrapper_map(logits, config: WrapperConfig = WrapperConfig()):
    """Apply temperature scaling and bias shift to vision logits."""
    return logits / config.temperature + config.bias


def inverse_wrapper_map(logits, config: WrapperConfig = WrapperConfig()):
    """Reverses the transformation applied by ``wrapper_map``."""
    return (logits - config.bias) * config.temperature

# Re-export for backward compatibility
__all__ = [
    'targeted_finetune', 
    'limited_distillation', 
    'wrapper_attack', 
    'extraction_attack',
    'wrapper_map',
    'inverse_wrapper_map',
    'WrapperConfig',
]