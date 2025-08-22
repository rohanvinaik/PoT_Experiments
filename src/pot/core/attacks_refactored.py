"""
Refactored attack implementations with proper class hierarchy.
This module fixes the context window thrashing by organizing attack types into separate classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass


@dataclass
class AttackConfig:
    """Configuration for attacks"""
    epsilon: float = 0.01
    steps: int = 10
    step_size: float = 0.003
    targeted: bool = False
    device: str = "cpu"


class BaseAttackModel(nn.Module, ABC):
    """
    Base class for all attack models.
    This consolidates common forward() logic to avoid duplication.
    """
    
    def __init__(self, base_model: nn.Module):
        """
        Initialize base attack model.
        
        Args:
            base_model: Original model to attack/wrap
        """
        super().__init__()
        self.base = base_model
        
        # Freeze base model parameters by default
        for param in self.base.parameters():
            param.requires_grad = False
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attack model.
        Must be implemented by subclasses.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def get_base_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get output from base model without gradients"""
        with torch.no_grad():
            return self.base(x)


class WrappedModel(BaseAttackModel):
    """
    Model that wraps a base model with conditional routing.
    Refactored from line 176 of original attacks.py.
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 routes: Dict[Callable[[Any], bool], nn.Module]):
        """
        Initialize wrapped model with routing logic.
        
        Args:
            base_model: Base model to wrap
            routes: Dictionary mapping predicates to alternative models
        """
        super().__init__(base_model)
        self.routes = routes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditional routing.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        outputs = []
        
        for i in range(x.size(0)):
            sample = x[i:i+1]
            used = False
            
            # Check routes
            for pred, alt_model in self.routes.items():
                try:
                    if pred(sample):
                        outputs.append(alt_model(sample))
                        used = True
                        break
                except Exception:
                    continue
            
            # Use base model if no route matched
            if not used:
                outputs.append(self.base(sample))
        
        return torch.cat(outputs, dim=0)


class WrapperModel(BaseAttackModel):
    """
    Model that applies wrapper layers on top of base model output.
    Refactored from line 710 of original attacks.py.
    """
    
    def __init__(self,
                 base_model: nn.Module,
                 wrapper_layers: List[nn.Module]):
        """
        Initialize wrapper model.
        
        Args:
            base_model: Base model to wrap
            wrapper_layers: List of wrapper layers to apply
        """
        super().__init__(base_model)
        self.wrappers = nn.ModuleList(wrapper_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through base model and wrapper layers.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Get base output without gradients
        output = self.get_base_output(x)
        
        # Apply wrapper layers sequentially
        for wrapper in self.wrappers:
            output = wrapper(output)
        
        return output


class AdversarialAttack(ABC):
    """Abstract base class for adversarial attacks"""
    
    def __init__(self, config: Optional[AttackConfig] = None):
        """
        Initialize adversarial attack.
        
        Args:
            config: Attack configuration
        """
        self.config = config or AttackConfig()
        self.device = torch.device(self.config.device)
    
    @abstractmethod
    def generate(self,
                model: nn.Module,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate adversarial examples.
        
        Args:
            model: Target model
            x: Input samples
            y: Optional target labels
            
        Returns:
            Adversarial examples
        """
        pass


class FGSMAttack(AdversarialAttack):
    """Fast Gradient Sign Method attack"""
    
    def generate(self,
                model: nn.Module,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate FGSM adversarial examples.
        
        Args:
            model: Target model
            x: Input samples
            y: Target labels (if None, uses model predictions)
            
        Returns:
            Adversarial examples
        """
        x = x.to(self.device)
        x.requires_grad = True
        
        # Get model output
        output = model(x)
        
        # Get labels if not provided
        if y is None:
            y = output.argmax(dim=1)
        else:
            y = y.to(self.device)
        
        # Calculate loss
        loss = F.cross_entropy(output, y)
        
        # Calculate gradients
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        x_adv = x + self.config.epsilon * x.grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach()


class PGDAttack(AdversarialAttack):
    """Projected Gradient Descent attack"""
    
    def generate(self,
                model: nn.Module,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate PGD adversarial examples.
        
        Args:
            model: Target model
            x: Input samples
            y: Target labels
            
        Returns:
            Adversarial examples
        """
        x = x.to(self.device)
        x_adv = x.clone().detach()
        
        # Get labels if not provided
        if y is None:
            with torch.no_grad():
                output = model(x)
                y = output.argmax(dim=1)
        else:
            y = y.to(self.device)
        
        # PGD iterations
        for _ in range(self.config.steps):
            x_adv.requires_grad = True
            
            output = model(x_adv)
            loss = F.cross_entropy(output, y)
            
            model.zero_grad()
            loss.backward()
            
            # Update adversarial examples
            x_adv = x_adv + self.config.step_size * x_adv.grad.sign()
            
            # Project back to epsilon ball
            delta = torch.clamp(x_adv - x, -self.config.epsilon, self.config.epsilon)
            x_adv = torch.clamp(x + delta, 0, 1).detach()
        
        return x_adv


class ModelManipulationAttack:
    """Base class for model manipulation attacks"""
    
    def __init__(self):
        """Initialize model manipulation attack"""
        self.modified_model = None
    
    def apply(self, model: nn.Module) -> nn.Module:
        """
        Apply manipulation to model.
        
        Args:
            model: Original model
            
        Returns:
            Manipulated model
        """
        raise NotImplementedError("Subclasses must implement apply()")


class FineTuningAttack(ModelManipulationAttack):
    """Attack by fine-tuning the model"""
    
    def __init__(self,
                 poison_data: torch.Tensor,
                 poison_labels: torch.Tensor,
                 epochs: int = 10,
                 lr: float = 0.001):
        """
        Initialize fine-tuning attack.
        
        Args:
            poison_data: Poisoned training data
            poison_labels: Labels for poisoned data
            epochs: Number of fine-tuning epochs
            lr: Learning rate
        """
        super().__init__()
        self.poison_data = poison_data
        self.poison_labels = poison_labels
        self.epochs = epochs
        self.lr = lr
    
    def apply(self, model: nn.Module) -> nn.Module:
        """
        Apply fine-tuning attack.
        
        Args:
            model: Original model
            
        Returns:
            Fine-tuned model
        """
        # Clone model
        import copy
        attacked_model = copy.deepcopy(model)
        attacked_model.train()
        
        # Setup optimizer
        optimizer = torch.optim.Adam(attacked_model.parameters(), lr=self.lr)
        
        # Fine-tune on poisoned data
        for epoch in range(self.epochs):
            output = attacked_model(self.poison_data)
            loss = F.cross_entropy(output, self.poison_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        attacked_model.eval()
        return attacked_model


class WeightPerturbationAttack(ModelManipulationAttack):
    """Attack by perturbing model weights"""
    
    def __init__(self, perturbation_scale: float = 0.01):
        """
        Initialize weight perturbation attack.
        
        Args:
            perturbation_scale: Scale of weight perturbations
        """
        super().__init__()
        self.perturbation_scale = perturbation_scale
    
    def apply(self, model: nn.Module) -> nn.Module:
        """
        Apply weight perturbation attack.
        
        Args:
            model: Original model
            
        Returns:
            Model with perturbed weights
        """
        import copy
        attacked_model = copy.deepcopy(model)
        
        # Perturb weights
        with torch.no_grad():
            for param in attacked_model.parameters():
                noise = torch.randn_like(param) * self.perturbation_scale
                param.add_(noise)
        
        return attacked_model


class AttackOrchestrator:
    """Orchestrates multiple attack strategies"""
    
    def __init__(self):
        """Initialize attack orchestrator"""
        self.attacks = {
            'fgsm': FGSMAttack(),
            'pgd': PGDAttack(),
            'fine_tuning': FineTuningAttack,  # Class, not instance
            'weight_perturbation': WeightPerturbationAttack()
        }
    
    def execute_attack(self,
                       attack_type: str,
                       model: nn.Module,
                       data: torch.Tensor,
                       **kwargs) -> Union[torch.Tensor, nn.Module]:
        """
        Execute specified attack.
        
        Args:
            attack_type: Type of attack to execute
            model: Target model
            data: Input data
            **kwargs: Additional attack parameters
            
        Returns:
            Adversarial examples or attacked model
        """
        if attack_type not in self.attacks:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        attack = self.attacks[attack_type]
        
        if isinstance(attack, AdversarialAttack):
            # Generate adversarial examples
            return attack.generate(model, data, **kwargs)
        elif isinstance(attack, ModelManipulationAttack):
            # Apply model manipulation
            return attack.apply(model)
        elif attack_type == 'fine_tuning':
            # Special case for fine-tuning (needs instantiation)
            if 'poison_data' not in kwargs or 'poison_labels' not in kwargs:
                raise ValueError("Fine-tuning attack requires poison_data and poison_labels")
            
            ft_attack = attack(kwargs['poison_data'], kwargs['poison_labels'])
            return ft_attack.apply(model)
        else:
            raise RuntimeError(f"Cannot execute attack: {attack_type}")


# Convenience functions
def create_wrapped_model(base_model: nn.Module,
                        routes: Optional[Dict] = None,
                        wrapper_layers: Optional[List] = None) -> BaseAttackModel:
    """
    Create a wrapped model based on provided configuration.
    
    Args:
        base_model: Base model to wrap
        routes: Routing configuration for WrappedModel
        wrapper_layers: Wrapper layers for WrapperModel
        
    Returns:
        Wrapped model instance
    """
    if routes is not None:
        return WrappedModel(base_model, routes)
    elif wrapper_layers is not None:
        return WrapperModel(base_model, wrapper_layers)
    else:
        raise ValueError("Either routes or wrapper_layers must be provided")


def create_adversarial_examples(model: nn.Module,
                               data: torch.Tensor,
                               attack_type: str = "fgsm",
                               config: Optional[AttackConfig] = None) -> torch.Tensor:
    """
    Generate adversarial examples using specified attack.
    
    Args:
        model: Target model
        data: Input data
        attack_type: Type of attack ("fgsm" or "pgd")
        config: Attack configuration
        
    Returns:
        Adversarial examples
    """
    if attack_type == "fgsm":
        attack = FGSMAttack(config)
    elif attack_type == "pgd":
        attack = PGDAttack(config)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    return attack.generate(model, data)