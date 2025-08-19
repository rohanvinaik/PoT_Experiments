"""
Deterministic test models for reliable PoT system validation.
These models provide consistent outputs for reproducible testing.
"""

import numpy as np
import hashlib
from typing import Any, Dict, List, Union


class DeterministicMockModel:
    """Mock model with consistent, deterministic outputs."""
    
    def __init__(self, model_id: str = "test_model", seed: int = 42):
        """
        Initialize deterministic mock model.
        
        Args:
            model_id: Unique identifier for this model
            seed: Random seed for reproducible outputs
        """
        self.model_id = model_id
        self.seed = seed
        self.base_seed = seed
        
    def forward(self, x: Any) -> np.ndarray:
        """
        Generate deterministic output based on input.
        
        Args:
            x: Input data (any type)
            
        Returns:
            Deterministic 10-dimensional output vector
        """
        # Create deterministic seed from input
        if isinstance(x, np.ndarray):
            input_hash = hashlib.md5(x.tobytes()).hexdigest()
        else:
            input_hash = hashlib.md5(str(x).encode()).hexdigest()
        
        # Use first 8 characters of hash as seed modifier
        seed_modifier = int(input_hash[:8], 16) % 1000000
        deterministic_seed = (self.base_seed + seed_modifier) % (2**32)
        
        # Generate consistent output
        np.random.seed(deterministic_seed)
        return np.random.randn(10)
    
    def predict(self, x: Any) -> np.ndarray:
        """Alias for forward method."""
        return self.forward(x)
    
    def state_dict(self) -> Dict[str, Any]:
        """Return model state dictionary."""
        return {
            'model_id': self.model_id,
            'seed': self.seed,
            'architecture': 'deterministic_mock'
        }
    
    def __call__(self, x: Any) -> np.ndarray:
        """Make model callable."""
        return self.forward(x)
    
    def num_parameters(self) -> int:
        """Return mock parameter count for compatibility."""
        return 100  # Mock value for testing


class LinearTestModel:
    """Simple linear model for testing mathematical consistency."""
    
    def __init__(self, input_dim: int = 10, output_dim: int = 10, seed: int = 42):
        """
        Initialize linear test model.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            seed: Random seed for weight initialization
        """
        np.random.seed(seed)
        self.weights = np.random.randn(input_dim, output_dim) * 0.1
        self.bias = np.random.randn(output_dim) * 0.01
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, x: Union[np.ndarray, List]) -> np.ndarray:
        """
        Forward pass through linear layer.
        
        Args:
            x: Input data
            
        Returns:
            Linear transformation output
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        # Ensure input has correct shape
        if x.ndim == 1:
            if len(x) != self.input_dim:
                # Pad or truncate to match input_dim
                if len(x) < self.input_dim:
                    x = np.pad(x, (0, self.input_dim - len(x)))
                else:
                    x = x[:self.input_dim]
        else:
            # Handle batch inputs
            if x.shape[-1] != self.input_dim:
                if x.shape[-1] < self.input_dim:
                    pad_width = [(0, 0)] * (x.ndim - 1) + [(0, self.input_dim - x.shape[-1])]
                    x = np.pad(x, pad_width)
                else:
                    x = x[..., :self.input_dim]
        
        return np.dot(x, self.weights) + self.bias
    
    def predict(self, x: Any) -> np.ndarray:
        """Alias for forward method."""
        return self.forward(x)
    
    def state_dict(self) -> Dict[str, Any]:
        """Return model state dictionary."""
        return {
            'weights': self.weights.tolist(),
            'bias': self.bias.tolist(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'architecture': 'linear_test'
        }
    
    def __call__(self, x: Any) -> np.ndarray:
        """Make model callable."""
        return self.forward(x)


class ConsistentHashModel:
    """Model that always returns the same hash for the same input."""
    
    def __init__(self, model_id: str = "hash_model"):
        """Initialize consistent hash model."""
        self.model_id = model_id
        
    def forward(self, x: Any) -> str:
        """
        Generate consistent hash output.
        
        Args:
            x: Input data
            
        Returns:
            Consistent hash string
        """
        if isinstance(x, np.ndarray):
            input_data = x.tobytes()
        else:
            input_data = str(x).encode()
            
        # Generate consistent hash
        hash_obj = hashlib.sha256(input_data + self.model_id.encode())
        return hash_obj.hexdigest()[:16]  # 16-character hash
    
    def predict(self, x: Any) -> str:
        """Alias for forward method."""
        return self.forward(x)
    
    def state_dict(self) -> Dict[str, Any]:
        """Return model state dictionary."""
        return {
            'model_id': self.model_id,
            'architecture': 'consistent_hash'
        }
    
    def __call__(self, x: Any) -> str:
        """Make model callable."""
        return self.forward(x)


def create_test_model(model_type: str = "deterministic", **kwargs) -> Any:
    """
    Factory function to create test models.
    
    Args:
        model_type: Type of test model to create
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Test model instance
    """
    if model_type == "deterministic":
        return DeterministicMockModel(**kwargs)
    elif model_type == "linear":
        return LinearTestModel(**kwargs)
    elif model_type == "hash":
        return ConsistentHashModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")