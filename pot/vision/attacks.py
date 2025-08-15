"""Attack implementations for vision models."""

# Import shared attack implementations
from pot.core.attacks import (
    targeted_finetune,
    limited_distillation,
    wrapper_attack,
    extraction_attack
)

# Vision-specific wrapper (for backward compatibility)
def wrapper_map(logits):
    """Simple wrapper for logit transformation.
    e.g., temperature scaling + bias shift placeholder
    """
    return logits

# Re-export for backward compatibility
__all__ = [
    'targeted_finetune', 
    'limited_distillation', 
    'wrapper_attack', 
    'extraction_attack',
    'wrapper_map'
]