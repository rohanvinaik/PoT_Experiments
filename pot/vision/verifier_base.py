"""
Base classes and interfaces for vision verification system.
This module provides the foundational classes that other verifier modules build upon.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from enum import Enum
import numpy as np
import torch
import torch.nn as nn


class VerificationMethod(Enum):
    """Verification methods available"""
    EXACT = "exact"
    FUZZY = "fuzzy"
    STATISTICAL = "statistical"
    BEHAVIORAL = "behavioral"
    ENSEMBLE = "ensemble"


class ChallengeType(Enum):
    """Types of challenges for verification"""
    RANDOM = "random"
    ADVERSARIAL = "adversarial"
    EDGE_CASE = "edge_case"
    PATTERN = "pattern"
    TEXTURE = "texture"
    GEOMETRIC = "geometric"


@dataclass
class VerificationResult:
    """Result of a verification attempt"""
    verified: bool
    confidence: float
    method: str
    details: Dict[str, Any] = field(default_factory=dict)
    challenges_passed: int = 0
    challenges_total: int = 0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'verified': self.verified,
            'confidence': self.confidence,
            'method': self.method,
            'details': self.details,
            'challenges_passed': self.challenges_passed,
            'challenges_total': self.challenges_total,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }


@dataclass 
class VerificationConfig:
    """Configuration for verification process"""
    method: VerificationMethod = VerificationMethod.STATISTICAL
    num_challenges: int = 10
    confidence_threshold: float = 0.95
    batch_size: int = 32
    device: str = "cpu"
    timeout: Optional[float] = None
    enable_hooks: bool = False
    save_activations: bool = False
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'method': self.method.value,
            'num_challenges': self.num_challenges,
            'confidence_threshold': self.confidence_threshold,
            'batch_size': self.batch_size,
            'device': self.device,
            'timeout': self.timeout,
            'enable_hooks': self.enable_hooks,
            'save_activations': self.save_activations,
            'verbose': self.verbose
        }


class IVerifier(ABC):
    """Abstract interface for all verifiers"""
    
    @abstractmethod
    def verify(self, model: nn.Module, **kwargs) -> VerificationResult:
        """Perform verification on a model"""
        pass
    
    @abstractmethod
    def generate_challenge(self, challenge_type: ChallengeType) -> torch.Tensor:
        """Generate a challenge input"""
        pass
    
    @abstractmethod
    def evaluate_response(self, output: torch.Tensor, expected: torch.Tensor) -> float:
        """Evaluate model response against expected output"""
        pass


class BaseVerifier(IVerifier):
    """Base implementation of verifier with common functionality"""
    
    def __init__(self, config: Optional[VerificationConfig] = None):
        """
        Initialize base verifier.
        
        Args:
            config: Verification configuration
        """
        self.config = config or VerificationConfig()
        self.device = torch.device(self.config.device)
        self.activations = {}
        self.hooks = []
        
    def verify(self, model: nn.Module, **kwargs) -> VerificationResult:
        """Base verification implementation"""
        # This will be overridden by subclasses
        raise NotImplementedError("Subclasses must implement verify()")
    
    def generate_challenge(self, challenge_type: ChallengeType) -> torch.Tensor:
        """Base challenge generation"""
        # This will be overridden by subclasses
        raise NotImplementedError("Subclasses must implement generate_challenge()")
    
    def evaluate_response(self, output: torch.Tensor, expected: torch.Tensor) -> float:
        """Base response evaluation"""
        # Simple cosine similarity by default
        output_flat = output.flatten()
        expected_flat = expected.flatten()
        
        if output_flat.shape != expected_flat.shape:
            return 0.0
        
        similarity = torch.nn.functional.cosine_similarity(
            output_flat.unsqueeze(0),
            expected_flat.unsqueeze(0)
        )
        
        return float(similarity.item())
    
    def cleanup(self):
        """Clean up resources"""
        self.remove_hooks()
        self.activations.clear()
    
    def remove_hooks(self):
        """Remove all hooks from model"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
        return False


class VerifierRegistry:
    """Registry for managing different verifier implementations"""
    
    _verifiers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, verifier_class: type):
        """Register a verifier implementation"""
        if not issubclass(verifier_class, IVerifier):
            raise TypeError(f"{verifier_class} must implement IVerifier interface")
        cls._verifiers[name] = verifier_class
    
    @classmethod
    def get(cls, name: str) -> type:
        """Get a verifier class by name"""
        if name not in cls._verifiers:
            raise KeyError(f"Verifier '{name}' not registered")
        return cls._verifiers[name]
    
    @classmethod
    def create(cls, name: str, config: Optional[VerificationConfig] = None) -> IVerifier:
        """Create a verifier instance"""
        verifier_class = cls.get(name)
        return verifier_class(config)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered verifiers"""
        return list(cls._verifiers.keys())