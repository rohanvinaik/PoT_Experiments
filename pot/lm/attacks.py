"""Attack implementations for language models."""

# Import shared attack implementations
from pot.core.attacks import (
    targeted_finetune,
    limited_distillation,
    wrapper_attack,
    extraction_attack
)

# LM-specific wrapper (for backward compatibility)
def wrapper_map(outputs):
    """Simple wrapper that passes through outputs."""
    return outputs

# Re-export for backward compatibility
__all__ = [
    'targeted_finetune', 
    'limited_distillation', 
    'wrapper_attack', 
    'extraction_attack',
    'wrapper_map'
]