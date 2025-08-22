"""
Refactored vision verifier module.
This is the new entry point that imports from the modularized components.
Import this instead of the old monolithic verifier.py.
"""

# Import all public interfaces
from .verifier_base import (
    VerificationConfig,
    VerificationResult,
    VerificationMethod,
    ChallengeType,
    IVerifier,
    BaseVerifier,
    VerifierRegistry
)

from .verifier_core import (
    VisionVerifier,
    EnhancedVisionVerifier
)

from .verifier_challenges import (
    ChallengeGenerator,
    ChallengeLibrary,
    PatternType
)

from .verifier_hooks import (
    HookManager,
    ActivationRecorder,
    GradientMonitor,
    create_hook_filter
)

# Convenience functions
def create_verifier(verifier_type: str = "vision",
                   config: VerificationConfig = None) -> IVerifier:
    """
    Create a verifier instance.
    
    Args:
        verifier_type: Type of verifier ("vision" or "vision_enhanced")
        config: Verification configuration
        
    Returns:
        Verifier instance
    """
    return VerifierRegistry.create(verifier_type, config)


def get_default_config() -> VerificationConfig:
    """Get default verification configuration"""
    return VerificationConfig()


# Export main classes for backward compatibility
__all__ = [
    # Base classes
    'VerificationConfig',
    'VerificationResult', 
    'VerificationMethod',
    'ChallengeType',
    'IVerifier',
    'BaseVerifier',
    'VerifierRegistry',
    
    # Core verifiers
    'VisionVerifier',
    'EnhancedVisionVerifier',
    
    # Challenge generation
    'ChallengeGenerator',
    'ChallengeLibrary',
    'PatternType',
    
    # Hook management
    'HookManager',
    'ActivationRecorder',
    'GradientMonitor',
    'create_hook_filter',
    
    # Convenience functions
    'create_verifier',
    'get_default_config'
]