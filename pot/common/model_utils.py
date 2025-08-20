"""
Common model utilities shared across the PoT framework.
Provides base classes and utilities for model manipulation and testing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from abc import ABC, abstractmethod
import hashlib
import json
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Common configuration for models"""
    model_id: str = "base_model"
    output_dim: int = 10
    hidden_dim: int = 64
    num_layers: int = 2
    activation: str = "relu"
    dropout: float = 0.0
    device: str = "cpu"
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary"""
        return cls(**data)


class ModelStateMixin:
    """
    Mixin providing state_dict functionality for non-PyTorch models.
    Used by mock models and custom implementations.
    """
    
    def __init__(self):
        """Initialize state tracking"""
        self._state = {}
        self._metadata = {}
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get model state dictionary.
        Consolidates common state_dict logic.
        
        Returns:
            Dictionary containing model state
        """
        state = {
            'state': dict(self._state),
            'metadata': dict(self._metadata)
        }
        
        # Add model-specific attributes if they exist
        if hasattr(self, 'model_id'):
            state['model_id'] = self.model_id
        if hasattr(self, 'output_dim'):
            state['output_dim'] = self.output_dim
        if hasattr(self, 'config'):
            state['config'] = self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config
        
        # Add PyTorch model state if this is a nn.Module
        if isinstance(self, nn.Module):
            state['pytorch_state'] = super().state_dict()
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load model state from dictionary.
        
        Args:
            state_dict: State dictionary to load
        """
        self._state = state_dict.get('state', {})
        self._metadata = state_dict.get('metadata', {})
        
        # Load model-specific attributes
        if 'model_id' in state_dict:
            self.model_id = state_dict['model_id']
        if 'output_dim' in state_dict:
            self.output_dim = state_dict['output_dim']
        if 'config' in state_dict:
            if hasattr(self, 'config'):
                if hasattr(self.config.__class__, 'from_dict'):
                    self.config = self.config.__class__.from_dict(state_dict['config'])
                else:
                    self.config = state_dict['config']
        
        # Load PyTorch state if present
        if isinstance(self, nn.Module) and 'pytorch_state' in state_dict:
            super().load_state_dict(state_dict['pytorch_state'])
    
    def get_fingerprint(self) -> str:
        """
        Get a fingerprint hash of the model state.
        
        Returns:
            SHA256 hash of model state
        """
        state_str = json.dumps(self.state_dict(), sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()


class BaseModelWrapper(nn.Module, ModelStateMixin):
    """
    Base wrapper for PyTorch models with common functionality.
    Consolidates forward() implementations and state management.
    """
    
    def __init__(self, 
                 base_model: Optional[nn.Module] = None,
                 config: Optional[ModelConfig] = None):
        """
        Initialize model wrapper.
        
        Args:
            base_model: Optional base model to wrap
            config: Model configuration
        """
        nn.Module.__init__(self)
        ModelStateMixin.__init__(self)
        
        self.config = config or ModelConfig()
        self.base_model = base_model
        self.device = torch.device(self.config.device)
        
        if base_model:
            self.base_model = base_model.to(self.device)
    
    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Consolidated forward pass implementation.
        
        Args:
            x: Input tensor or numpy array
            
        Returns:
            Output tensor
        """
        # Convert numpy to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Move to device
        x = x.to(self.device)
        
        # Use base model if available
        if self.base_model is not None:
            return self.base_model(x)
        
        # Default implementation for mock models
        batch_size = x.shape[0] if x.dim() > 0 else 1
        return torch.randn(batch_size, self.config.output_dim, device=self.device)
    
    def get_activations(self, x: torch.Tensor, layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Get intermediate activations from specified layers.
        
        Args:
            x: Input tensor
            layer_names: Names of layers to extract (None = all)
            
        Returns:
            Dictionary mapping layer names to activations
        """
        activations = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks
        if self.base_model:
            for name, module in self.base_model.named_modules():
                if layer_names is None or name in layer_names:
                    hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        _ = self.forward(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations


def create_mock_model(model_type: str = "simple",
                     output_dim: int = 10,
                     **kwargs) -> BaseModelWrapper:
    """
    Factory function to create mock models for testing.
    
    Args:
        model_type: Type of mock model ("simple", "stateful", "deterministic")
        output_dim: Output dimension
        **kwargs: Additional model parameters
        
    Returns:
        Mock model instance
    """
    config = ModelConfig(
        model_id=f"mock_{model_type}",
        output_dim=output_dim,
        **kwargs
    )
    
    if model_type == "simple":
        return SimpleForwardModel(config)
    elif model_type == "stateful":
        return StatefulMockModel(config)
    elif model_type == "deterministic":
        return DeterministicMockModel(config)
    else:
        return BaseModelWrapper(config=config)


class SimpleForwardModel(BaseModelWrapper):
    """
    Simple mock model with basic forward pass.
    Extracted from multiple test files.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize simple model"""
        super().__init__(config=config)
        self.call_count = 0
    
    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Simple forward pass with call counting.
        
        Args:
            x: Input tensor
            
        Returns:
            Random output tensor
        """
        self.call_count += 1
        self._state['last_input_shape'] = x.shape if hasattr(x, 'shape') else len(x)
        return super().forward(x)


class StatefulMockModel(BaseModelWrapper):
    """
    Mock model that maintains internal state.
    Used for testing stateful verification.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize stateful model"""
        super().__init__(config=config)
        self.internal_state = torch.zeros(config.hidden_dim if config else 64)
        self.call_history = []
    
    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Forward pass with state updates.
        
        Args:
            x: Input tensor
            
        Returns:
            Output influenced by internal state
        """
        x = super().forward(x)
        
        # Update internal state
        self.internal_state += torch.randn_like(self.internal_state) * 0.01
        self.call_history.append({
            'input_shape': x.shape,
            'state_norm': self.internal_state.norm().item()
        })
        
        # Influence output with state
        state_influence = self.internal_state.mean().item()
        return x + state_influence
    
    def reset_state(self):
        """Reset internal state"""
        self.internal_state.zero_()
        self.call_history.clear()


class DeterministicMockModel(BaseModelWrapper):
    """
    Deterministic mock model for reproducible testing.
    Always produces the same output for the same input.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize deterministic model"""
        super().__init__(config=config)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
    
    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Deterministic forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Deterministic output based on input
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        x = x.to(self.device)
        
        # Generate deterministic output based on input
        batch_size = x.shape[0] if x.dim() > 0 else 1
        
        # Use input sum as seed for reproducibility
        seed = int(x.sum().item() * 1000) % (2**32)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        
        return torch.randn(
            batch_size, 
            self.config.output_dim,
            device=self.device,
            generator=generator
        )


def wrap_model(model: nn.Module,
               wrapper_type: str = "monitoring",
               **kwargs) -> nn.Module:
    """
    Wrap a model with additional functionality.
    
    Args:
        model: Model to wrap
        wrapper_type: Type of wrapper ("monitoring", "defensive", "logging")
        **kwargs: Wrapper-specific parameters
        
    Returns:
        Wrapped model
    """
    if wrapper_type == "monitoring":
        return MonitoringWrapper(model, **kwargs)
    elif wrapper_type == "defensive":
        return DefensiveWrapper(model, **kwargs)
    elif wrapper_type == "logging":
        return LoggingWrapper(model, **kwargs)
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}")


class MonitoringWrapper(BaseModelWrapper):
    """
    Wrapper that monitors model behavior.
    Tracks inputs, outputs, and performance metrics.
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 track_gradients: bool = False,
                 track_activations: bool = False):
        """
        Initialize monitoring wrapper.
        
        Args:
            base_model: Model to monitor
            track_gradients: Whether to track gradients
            track_activations: Whether to track activations
        """
        super().__init__(base_model=base_model)
        self.track_gradients = track_gradients
        self.track_activations = track_activations
        self.metrics = {
            'forward_calls': 0,
            'input_stats': [],
            'output_stats': [],
            'gradient_norms': [],
            'activation_stats': []
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with monitoring.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        self.metrics['forward_calls'] += 1
        
        # Track input statistics
        self.metrics['input_stats'].append({
            'mean': x.mean().item(),
            'std': x.std().item(),
            'min': x.min().item(),
            'max': x.max().item()
        })
        
        # Get activations if requested
        if self.track_activations:
            activations = self.get_activations(x)
            self.metrics['activation_stats'].append({
                name: {'mean': act.mean().item(), 'std': act.std().item()}
                for name, act in activations.items()
            })
        
        # Forward pass
        output = super().forward(x)
        
        # Track output statistics  
        self.metrics['output_stats'].append({
            'mean': output.mean().item(),
            'std': output.std().item(),
            'min': output.min().item(),
            'max': output.max().item()
        })
        
        # Track gradients if requested
        if self.track_gradients and output.requires_grad:
            def grad_hook(grad):
                self.metrics['gradient_norms'].append(grad.norm().item())
                return grad
            output.register_hook(grad_hook)
        
        return output
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics"""
        return dict(self.metrics)
    
    def reset_metrics(self):
        """Reset collected metrics"""
        self.metrics = {
            'forward_calls': 0,
            'input_stats': [],
            'output_stats': [],
            'gradient_norms': [],
            'activation_stats': []
        }


class DefensiveWrapper(BaseModelWrapper):
    """
    Wrapper that adds defensive mechanisms against attacks.
    """
    
    def __init__(self,
                 base_model: nn.Module,
                 input_bounds: Optional[Tuple[float, float]] = None,
                 output_bounds: Optional[Tuple[float, float]] = None,
                 detect_adversarial: bool = True):
        """
        Initialize defensive wrapper.
        
        Args:
            base_model: Model to protect
            input_bounds: Valid input range (min, max)
            output_bounds: Valid output range (min, max)
            detect_adversarial: Whether to detect adversarial inputs
        """
        super().__init__(base_model=base_model)
        self.input_bounds = input_bounds or (0.0, 1.0)
        self.output_bounds = output_bounds
        self.detect_adversarial = detect_adversarial
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with defensive checks.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor (possibly sanitized)
        """
        # Clamp inputs to valid range
        x = torch.clamp(x, self.input_bounds[0], self.input_bounds[1])
        
        # Detect potential adversarial inputs
        if self.detect_adversarial:
            if self._is_adversarial(x):
                # Log warning and apply additional defense
                self._metadata['adversarial_detected'] = True
                x = self._sanitize_input(x)
        
        # Forward pass
        output = super().forward(x)
        
        # Clamp outputs if bounds specified
        if self.output_bounds:
            output = torch.clamp(output, self.output_bounds[0], self.output_bounds[1])
        
        return output
    
    def _is_adversarial(self, x: torch.Tensor) -> bool:
        """
        Simple adversarial detection based on input statistics.
        
        Args:
            x: Input tensor
            
        Returns:
            True if input appears adversarial
        """
        # Check for unusual input statistics
        input_std = x.std().item()
        input_range = (x.max() - x.min()).item()
        
        # Heuristic thresholds
        if input_std < 0.01 or input_std > 10.0:
            return True
        if input_range < 0.01 or input_range > 100.0:
            return True
        
        return False
    
    def _sanitize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sanitize potentially adversarial input.
        
        Args:
            x: Input tensor
            
        Returns:
            Sanitized input
        """
        # Add small random noise to break adversarial patterns
        noise = torch.randn_like(x) * 0.01
        return x + noise


class LoggingWrapper(BaseModelWrapper):
    """
    Wrapper that logs all model interactions.
    """
    
    def __init__(self,
                 base_model: nn.Module,
                 log_inputs: bool = True,
                 log_outputs: bool = True,
                 max_log_size: int = 1000):
        """
        Initialize logging wrapper.
        
        Args:
            base_model: Model to log
            log_inputs: Whether to log inputs
            log_outputs: Whether to log outputs
            max_log_size: Maximum number of entries to keep
        """
        super().__init__(base_model=base_model)
        self.log_inputs = log_inputs
        self.log_outputs = log_outputs
        self.max_log_size = max_log_size
        self.logs = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with logging.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        import time
        
        log_entry = {
            'timestamp': time.time(),
            'call_number': len(self.logs) + 1
        }
        
        if self.log_inputs:
            log_entry['input'] = {
                'shape': list(x.shape),
                'dtype': str(x.dtype),
                'device': str(x.device),
                'summary': {
                    'mean': x.mean().item(),
                    'std': x.std().item(),
                    'min': x.min().item(),
                    'max': x.max().item()
                }
            }
        
        # Forward pass
        start_time = time.perf_counter()
        output = super().forward(x)
        inference_time = time.perf_counter() - start_time
        
        log_entry['inference_time'] = inference_time
        
        if self.log_outputs:
            log_entry['output'] = {
                'shape': list(output.shape),
                'dtype': str(output.dtype),
                'summary': {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item()
                }
            }
        
        # Add to logs (with size limit)
        self.logs.append(log_entry)
        if len(self.logs) > self.max_log_size:
            self.logs.pop(0)
        
        return output
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get collected logs"""
        return list(self.logs)
    
    def clear_logs(self):
        """Clear collected logs"""
        self.logs.clear()


def get_model_signature(model: nn.Module) -> Dict[str, Any]:
    """
    Get a signature/summary of a model's architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing model signature
    """
    signature = {
        'class_name': model.__class__.__name__,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'num_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'layers': []
    }
    
    # Collect layer information
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            layer_info = {
                'name': name,
                'type': module.__class__.__name__,
                'params': sum(p.numel() for p in module.parameters())
            }
            
            # Add layer-specific info
            if hasattr(module, 'in_features'):
                layer_info['in_features'] = module.in_features
            if hasattr(module, 'out_features'):
                layer_info['out_features'] = module.out_features
            if hasattr(module, 'kernel_size'):
                layer_info['kernel_size'] = module.kernel_size
            
            signature['layers'].append(layer_info)
    
    # Add state dict hash
    if hasattr(model, 'get_fingerprint'):
        signature['fingerprint'] = model.get_fingerprint()
    else:
        state_str = str(model.state_dict())
        signature['fingerprint'] = hashlib.sha256(state_str.encode()).hexdigest()[:16]
    
    return signature


def compare_models(model1: nn.Module, model2: nn.Module, tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Compare two models for structural and weight differences.
    
    Args:
        model1: First model
        model2: Second model  
        tolerance: Tolerance for weight comparison
        
    Returns:
        Comparison results
    """
    results = {
        'structurally_identical': True,
        'weights_identical': True,
        'differences': []
    }
    
    # Compare signatures
    sig1 = get_model_signature(model1)
    sig2 = get_model_signature(model2)
    
    if sig1['class_name'] != sig2['class_name']:
        results['structurally_identical'] = False
        results['differences'].append(f"Different classes: {sig1['class_name']} vs {sig2['class_name']}")
    
    if sig1['num_parameters'] != sig2['num_parameters']:
        results['structurally_identical'] = False
        results['differences'].append(f"Different parameter counts: {sig1['num_parameters']} vs {sig2['num_parameters']}")
    
    # Compare state dicts
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    
    for key in state1.keys():
        if key not in state2:
            results['weights_identical'] = False
            results['differences'].append(f"Missing key in model2: {key}")
        else:
            if not torch.allclose(state1[key], state2[key], atol=tolerance):
                results['weights_identical'] = False
                diff = (state1[key] - state2[key]).abs().max().item()
                results['differences'].append(f"Weight difference in {key}: max_diff={diff}")
    
    for key in state2.keys():
        if key not in state1:
            results['weights_identical'] = False
            results['differences'].append(f"Extra key in model2: {key}")
    
    results['identical'] = results['structurally_identical'] and results['weights_identical']
    
    return results