"""
Hook management utilities for vision verification.
This module consolidates all hook-related functionality to avoid duplication.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import defaultdict
import warnings


class HookManager:
    """
    Centralized hook management for model inspection.
    Consolidates duplicate hook implementations from lines 615 and 1611.
    """
    
    def __init__(self):
        """Initialize hook manager"""
        self.hooks = []
        self.activations = defaultdict(list)
        self.gradients = defaultdict(list)
        self.enabled = True
        
    def register_forward_hook(self, 
                            module: nn.Module,
                            name: Optional[str] = None) -> Any:
        """
        Register a forward hook on a module.
        
        Args:
            module: Module to hook
            name: Optional name for the hook
            
        Returns:
            Hook handle
        """
        if name is None:
            name = f"{module.__class__.__name__}_{id(module)}"
        
        def hook(module, input, output):
            """Forward hook function - consolidates duplicate from lines 615, 1611"""
            if self.enabled:
                # Store activation
                if isinstance(output, tuple):
                    output = output[0]
                
                if isinstance(output, torch.Tensor):
                    self.activations[name].append(output.detach().cpu())
                else:
                    self.activations[name].append(output)
        
        handle = module.register_forward_hook(hook)
        self.hooks.append((handle, name))
        return handle
    
    def register_backward_hook(self,
                              module: nn.Module,
                              name: Optional[str] = None) -> Any:
        """
        Register a backward hook on a module.
        
        Args:
            module: Module to hook
            name: Optional name for the hook
            
        Returns:
            Hook handle
        """
        if name is None:
            name = f"{module.__class__.__name__}_{id(module)}"
        
        def hook(module, grad_input, grad_output):
            """Backward hook function"""
            if self.enabled:
                # Store gradient
                if isinstance(grad_output, tuple):
                    grad_output = grad_output[0]
                
                if isinstance(grad_output, torch.Tensor):
                    self.gradients[name].append(grad_output.detach().cpu())
                else:
                    self.gradients[name].append(grad_output)
        
        handle = module.register_backward_hook(hook)
        self.hooks.append((handle, name))
        return handle
    
    def register_hooks_recursive(self,
                                model: nn.Module,
                                forward: bool = True,
                                backward: bool = False,
                                filter_fn: Optional[Callable] = None):
        """
        Recursively register hooks on all modules.
        
        Args:
            model: Model to hook
            forward: Register forward hooks
            backward: Register backward hooks
            filter_fn: Optional filter function for modules
        """
        for name, module in model.named_modules():
            # Skip if filter function rejects
            if filter_fn and not filter_fn(name, module):
                continue
            
            # Register hooks
            if forward:
                self.register_forward_hook(module, name)
            if backward:
                self.register_backward_hook(module, name)
    
    def remove_all_hooks(self):
        """Remove all registered hooks"""
        for handle, _ in self.hooks:
            handle.remove()
        self.hooks.clear()
    
    def clear_activations(self):
        """Clear stored activations"""
        self.activations.clear()
        self.gradients.clear()
    
    def get_activation(self, name: str) -> Optional[List[torch.Tensor]]:
        """Get activations for a specific layer"""
        return self.activations.get(name)
    
    def get_gradient(self, name: str) -> Optional[List[torch.Tensor]]:
        """Get gradients for a specific layer"""
        return self.gradients.get(name)
    
    def get_all_activations(self) -> Dict[str, List[torch.Tensor]]:
        """Get all stored activations"""
        return dict(self.activations)
    
    def get_all_gradients(self) -> Dict[str, List[torch.Tensor]]:
        """Get all stored gradients"""
        return dict(self.gradients)
    
    def disable(self):
        """Temporarily disable hooks"""
        self.enabled = False
    
    def enable(self):
        """Re-enable hooks"""
        self.enabled = True
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup hooks"""
        self.remove_all_hooks()
        self.clear_activations()
        return False


class ActivationRecorder:
    """
    Specialized hook for recording activations during verification.
    """
    
    def __init__(self, layers_to_record: Optional[List[str]] = None):
        """
        Initialize activation recorder.
        
        Args:
            layers_to_record: Specific layers to record (None = all)
        """
        self.layers_to_record = layers_to_record
        self.recordings = {}
        self.hook_manager = HookManager()
    
    def attach_to_model(self, model: nn.Module):
        """Attach recorder to model"""
        def filter_fn(name, module):
            if self.layers_to_record is None:
                return True
            return any(layer in name for layer in self.layers_to_record)
        
        self.hook_manager.register_hooks_recursive(
            model,
            forward=True,
            backward=False,
            filter_fn=filter_fn
        )
    
    def detach(self):
        """Detach recorder from model"""
        self.hook_manager.remove_all_hooks()
    
    def get_recordings(self) -> Dict[str, torch.Tensor]:
        """Get recorded activations"""
        recordings = {}
        for name, acts in self.hook_manager.get_all_activations().items():
            if acts:
                # Concatenate along batch dimension
                recordings[name] = torch.cat(acts, dim=0)
        return recordings
    
    def clear(self):
        """Clear recordings"""
        self.hook_manager.clear_activations()
        self.recordings.clear()


class GradientMonitor:
    """
    Monitor gradients during backward pass.
    """
    
    def __init__(self, alert_threshold: float = 10.0):
        """
        Initialize gradient monitor.
        
        Args:
            alert_threshold: Threshold for gradient magnitude alerts
        """
        self.alert_threshold = alert_threshold
        self.hook_manager = HookManager()
        self.alerts = []
    
    def attach_to_model(self, model: nn.Module):
        """Attach monitor to model"""
        self.hook_manager.register_hooks_recursive(
            model,
            forward=False,
            backward=True
        )
    
    def check_gradients(self) -> List[str]:
        """Check for gradient issues"""
        alerts = []
        
        for name, grads in self.hook_manager.get_all_gradients().items():
            for grad in grads:
                if grad is None:
                    alerts.append(f"None gradient in {name}")
                    continue
                
                # Check for NaN
                if torch.isnan(grad).any():
                    alerts.append(f"NaN gradient in {name}")
                
                # Check for Inf
                if torch.isinf(grad).any():
                    alerts.append(f"Inf gradient in {name}")
                
                # Check magnitude
                grad_norm = torch.norm(grad)
                if grad_norm > self.alert_threshold:
                    alerts.append(f"Large gradient in {name}: {grad_norm:.2f}")
                
                # Check for vanishing gradients
                if grad_norm < 1e-7:
                    alerts.append(f"Vanishing gradient in {name}: {grad_norm:.2e}")
        
        self.alerts.extend(alerts)
        return alerts
    
    def get_gradient_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about gradients"""
        stats = {}
        
        for name, grads in self.hook_manager.get_all_gradients().items():
            if grads and grads[0] is not None:
                grad_tensor = torch.cat([g.flatten() for g in grads])
                stats[name] = {
                    'mean': float(grad_tensor.mean()),
                    'std': float(grad_tensor.std()),
                    'min': float(grad_tensor.min()),
                    'max': float(grad_tensor.max()),
                    'norm': float(torch.norm(grad_tensor))
                }
        
        return stats
    
    def detach(self):
        """Detach monitor"""
        self.hook_manager.remove_all_hooks()
    
    def clear(self):
        """Clear recordings and alerts"""
        self.hook_manager.clear_activations()
        self.alerts.clear()


def create_hook_filter(layer_types: Optional[List[type]] = None,
                       layer_names: Optional[List[str]] = None,
                       exclude_types: Optional[List[type]] = None,
                       exclude_names: Optional[List[str]] = None) -> Callable:
    """
    Create a filter function for selective hooking.
    
    Args:
        layer_types: Types of layers to include
        layer_names: Names of layers to include
        exclude_types: Types of layers to exclude
        exclude_names: Names of layers to exclude
        
    Returns:
        Filter function
    """
    def filter_fn(name: str, module: nn.Module) -> bool:
        # Check exclusions first
        if exclude_types:
            if any(isinstance(module, t) for t in exclude_types):
                return False
        
        if exclude_names:
            if any(ex in name for ex in exclude_names):
                return False
        
        # Check inclusions
        include = True
        
        if layer_types:
            include = include and any(isinstance(module, t) for t in layer_types)
        
        if layer_names:
            include = include and any(ln in name for ln in layer_names)
        
        return include
    
    return filter_fn