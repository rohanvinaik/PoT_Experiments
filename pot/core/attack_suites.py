"""
Parameterized Attack Suites for Proof-of-Training Verification

This module provides standardized attack configurations and adaptive attack generation
for comprehensive evaluation of PoT defense mechanisms.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
import random
import copy
from abc import ABC, abstractmethod
import time

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


@dataclass
class AttackConfig:
    """Base configuration for all attacks."""
    name: str
    attack_type: str
    budget: Dict[str, Any]  # queries, compute_time, epochs, etc.
    strength: str  # 'weak', 'moderate', 'strong', 'adaptive'
    success_metrics: Dict[str, float]  # target thresholds
    parameters: Dict[str, Any] = field(default_factory=dict)  # attack-specific params
    metadata: Dict[str, Any] = field(default_factory=dict)  # additional info
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_strengths = ['weak', 'moderate', 'strong', 'adaptive']
        if self.strength not in valid_strengths:
            raise ValueError(f"Strength must be one of {valid_strengths}")
        
        # Ensure required budget fields
        if 'queries' not in self.budget:
            self.budget['queries'] = 10000  # Default
        if 'compute_time' not in self.budget:
            self.budget['compute_time'] = 300.0  # 5 minutes default
    
    def scale_budget(self, factor: float) -> 'AttackConfig':
        """Create a new config with scaled budget."""
        new_config = copy.deepcopy(self)
        for key, value in new_config.budget.items():
            if isinstance(value, (int, float)):
                new_config.budget[key] = int(value * factor) if isinstance(value, int) else value * factor
        return new_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'attack_type': self.attack_type,
            'budget': self.budget,
            'strength': self.strength,
            'success_metrics': self.success_metrics,
            'parameters': self.parameters,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttackConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AttackResult:
    """Results from executing an attack configuration."""
    config: AttackConfig
    success: bool
    metrics: Dict[str, float]
    execution_time: float
    resources_used: Dict[str, Any]
    error_message: Optional[str] = None
    detailed_results: Dict[str, Any] = field(default_factory=dict)
    
    def meets_success_criteria(self) -> bool:
        """Check if attack meets its success criteria."""
        for metric_name, target_value in self.config.success_metrics.items():
            if metric_name not in self.metrics:
                return False
            
            actual_value = self.metrics[metric_name]
            
            # Different metrics have different success conditions
            if metric_name in ['fidelity', 'accuracy', 'precision', 'recall']:
                # Higher is better
                if actual_value < target_value:
                    return False
            elif metric_name in ['accuracy_drop', 'loss', 'error_rate']:
                # Lower is better
                if actual_value > target_value:
                    return False
            else:
                # Default: higher is better
                if actual_value < target_value:
                    return False
        
        return True


class StandardAttackSuite:
    """Collection of standard attacks with varying strengths."""
    
    @staticmethod
    def get_distillation_configs() -> List[AttackConfig]:
        """Get distillation attack configurations."""
        return [
            AttackConfig(
                name="weak_distillation",
                attack_type="distillation",
                budget={"queries": 1000, "epochs": 5, "compute_time": 60.0},
                strength="weak",
                success_metrics={"fidelity": 0.7, "accuracy": 0.8},
                parameters={
                    "temperature": 3.0,
                    "alpha": 0.7,
                    "learning_rate": 0.001,
                    "optimizer": "adam"
                },
                metadata={"description": "Basic distillation with limited budget"}
            ),
            AttackConfig(
                name="moderate_distillation",
                attack_type="distillation",
                budget={"queries": 5000, "epochs": 15, "compute_time": 180.0},
                strength="moderate",
                success_metrics={"fidelity": 0.85, "accuracy": 0.9},
                parameters={
                    "temperature": 4.0,
                    "alpha": 0.8,
                    "learning_rate": 0.0005,
                    "optimizer": "adam",
                    "patience": 3
                },
                metadata={"description": "Standard distillation with moderate resources"}
            ),
            AttackConfig(
                name="strong_distillation",
                attack_type="distillation", 
                budget={"queries": 10000, "epochs": 50, "compute_time": 600.0},
                strength="strong",
                success_metrics={"fidelity": 0.95, "accuracy": 0.95},
                parameters={
                    "temperature": 5.0,
                    "alpha": 0.9,
                    "learning_rate": 0.0001,
                    "optimizer": "adam",
                    "patience": 5,
                    "lr_scheduler": True
                },
                metadata={"description": "High-resource distillation for maximum fidelity"}
            ),
            AttackConfig(
                name="multi_stage_distillation",
                attack_type="distillation",
                budget={"queries": 15000, "epochs": 30, "compute_time": 900.0},
                strength="strong",
                success_metrics={"fidelity": 0.98, "accuracy": 0.97},
                parameters={
                    "temperature": [3.0, 4.0, 6.0],  # Multi-stage temperatures
                    "alpha": [0.6, 0.8, 0.9],  # Progressive weighting
                    "learning_rate": [0.001, 0.0005, 0.0001],  # Decreasing LR
                    "stages": 3,
                    "optimizer": "adam"
                },
                metadata={"description": "Multi-stage progressive distillation"}
            )
        ]
    
    @staticmethod
    def get_compression_configs() -> List[AttackConfig]:
        """Get compression attack configurations."""
        return [
            AttackConfig(
                name="mild_pruning",
                attack_type="pruning",
                budget={"compression_ratio": 0.3, "compute_time": 120.0},
                strength="weak",
                success_metrics={"accuracy_drop": 0.05, "compression_ratio": 0.25},
                parameters={
                    "pruning_method": "l1_unstructured",
                    "fine_tune_epochs": 5,
                    "learning_rate": 0.0001
                },
                metadata={"description": "Light pruning with minimal accuracy loss"}
            ),
            AttackConfig(
                name="moderate_pruning",
                attack_type="pruning",
                budget={"compression_ratio": 0.5, "compute_time": 180.0},
                strength="moderate",
                success_metrics={"accuracy_drop": 0.1, "compression_ratio": 0.45},
                parameters={
                    "pruning_method": "l1_unstructured",
                    "fine_tune_epochs": 10,
                    "learning_rate": 0.0001,
                    "gradual_pruning": True
                },
                metadata={"description": "Moderate pruning with gradual compression"}
            ),
            AttackConfig(
                name="aggressive_pruning",
                attack_type="pruning",
                budget={"compression_ratio": 0.8, "compute_time": 300.0},
                strength="strong",
                success_metrics={"accuracy_drop": 0.2, "compression_ratio": 0.75},
                parameters={
                    "pruning_method": "l1_unstructured", 
                    "fine_tune_epochs": 20,
                    "learning_rate": 0.00005,
                    "gradual_pruning": True,
                    "recovery_training": True
                },
                metadata={"description": "Aggressive pruning with recovery training"}
            ),
            AttackConfig(
                name="basic_quantization",
                attack_type="quantization",
                budget={"bits": 8, "compute_time": 60.0},
                strength="weak",
                success_metrics={"accuracy_drop": 0.03, "compression_ratio": 0.7},
                parameters={
                    "quantization_type": "dynamic",
                    "backend": "fbgemm"
                },
                metadata={"description": "8-bit dynamic quantization"}
            ),
            AttackConfig(
                name="aggressive_quantization",
                attack_type="quantization",
                budget={"bits": 4, "compute_time": 120.0},
                strength="strong",
                success_metrics={"accuracy_drop": 0.15, "compression_ratio": 0.85},
                parameters={
                    "quantization_type": "static",
                    "backend": "fbgemm",
                    "calibration_data": True,
                    "fine_tune_epochs": 5
                },
                metadata={"description": "4-bit static quantization with calibration"}
            ),
            AttackConfig(
                name="hybrid_compression",
                attack_type="compression",
                budget={"compression_ratio": 0.9, "compute_time": 400.0},
                strength="strong",
                success_metrics={"accuracy_drop": 0.12, "compression_ratio": 0.85},
                parameters={
                    "pruning_ratio": 0.6,
                    "quantization_bits": 6,
                    "sequential": True,  # Prune then quantize
                    "fine_tune_epochs": 15,
                    "learning_rate": 0.00005
                },
                metadata={"description": "Combined pruning and quantization"}
            )
        ]
    
    @staticmethod
    def get_wrapper_configs() -> List[AttackConfig]:
        """Get wrapper attack configurations."""
        return [
            AttackConfig(
                name="simple_wrapper",
                attack_type="wrapper",
                budget={"queries": 5000, "epochs": 10, "compute_time": 180.0},
                strength="weak",
                success_metrics={"wrapper_adaptation": 0.5, "final_loss": 1.0},
                parameters={
                    "wrapper_layers": 2,
                    "hidden_size": 64,
                    "learning_rate": 0.001,
                    "optimizer": "adam"
                },
                metadata={"description": "Simple wrapper with basic adaptation"}
            ),
            AttackConfig(
                name="adaptive_wrapper",
                attack_type="wrapper",
                budget={"queries": 10000, "epochs": 20, "compute_time": 360.0},
                strength="moderate",
                success_metrics={"wrapper_adaptation": 0.8, "final_loss": 0.5},
                parameters={
                    "wrapper_layers": 3,
                    "hidden_size": 128,
                    "learning_rate": 0.0005,
                    "optimizer": "adam",
                    "adaptive_lr": True,
                    "dropout": 0.1
                },
                metadata={"description": "Adaptive wrapper with regularization"}
            ),
            AttackConfig(
                name="sophisticated_wrapper",
                attack_type="wrapper",
                budget={"queries": 15000, "epochs": 30, "compute_time": 600.0},
                strength="strong",
                success_metrics={"wrapper_adaptation": 0.9, "final_loss": 0.3},
                parameters={
                    "wrapper_layers": 4,
                    "hidden_size": 256,
                    "learning_rate": 0.0001,
                    "optimizer": "adamw",
                    "adaptive_lr": True,
                    "dropout": 0.15,
                    "residual_connections": True,
                    "attention_mechanism": True
                },
                metadata={"description": "Sophisticated wrapper with attention"}
            )
        ]
    
    @staticmethod
    def get_extraction_configs() -> List[AttackConfig]:
        """Get model extraction attack configurations."""
        return [
            AttackConfig(
                name="basic_extraction",
                attack_type="extraction",
                budget={"queries": 5000, "epochs": 15, "compute_time": 300.0},
                strength="weak",
                success_metrics={"extraction_fidelity": 0.6, "accuracy": 0.7},
                parameters={
                    "surrogate_architecture": "simple",
                    "learning_rate": 0.001,
                    "optimizer": "sgd",
                    "query_strategy": "random"
                },
                metadata={"description": "Basic model extraction with random queries"}
            ),
            AttackConfig(
                name="smart_extraction",
                attack_type="extraction",
                budget={"queries": 10000, "epochs": 25, "compute_time": 600.0},
                strength="moderate",
                success_metrics={"extraction_fidelity": 0.8, "accuracy": 0.85},
                parameters={
                    "surrogate_architecture": "adaptive",
                    "learning_rate": 0.0005,
                    "optimizer": "adam",
                    "query_strategy": "uncertainty_sampling",
                    "active_learning": True
                },
                metadata={"description": "Smart extraction with active learning"}
            ),
            AttackConfig(
                name="advanced_extraction",
                attack_type="extraction",
                budget={"queries": 20000, "epochs": 40, "compute_time": 1200.0},
                strength="strong",
                success_metrics={"extraction_fidelity": 0.9, "accuracy": 0.9},
                parameters={
                    "surrogate_architecture": "ensemble",
                    "learning_rate": 0.0001,
                    "optimizer": "adamw",
                    "query_strategy": "adversarial",
                    "active_learning": True,
                    "ensemble_size": 3,
                    "knowledge_distillation": True
                },
                metadata={"description": "Advanced extraction with adversarial queries"}
            )
        ]
    
    @staticmethod
    def get_vision_specific_configs() -> List[AttackConfig]:
        """Get vision-specific attack configurations."""
        return [
            # Adversarial Patch Attacks
            AttackConfig(
                name="patch_weak_small",
                attack_type="adversarial_patch",
                budget={"queries": 1000, "compute_time": 120.0},
                strength="weak",
                success_metrics={"attack_success_rate": 0.2},
                parameters={
                    "patch_size": (8, 8),
                    "patch_location": "random",
                    "iterations": 200,
                    "learning_rate": 0.01,
                    "target_class": None,  # Untargeted
                    "success_threshold": 0.1
                }
            ),
            AttackConfig(
                name="patch_moderate_targeted",
                attack_type="adversarial_patch",
                budget={"queries": 2000, "compute_time": 300.0},
                strength="moderate",
                success_metrics={"attack_success_rate": 0.4, "targeted_success_rate": 0.3},
                parameters={
                    "patch_size": (16, 16),
                    "patch_location": "center",
                    "iterations": 500,
                    "learning_rate": 0.02,
                    "target_class": 7,
                    "success_threshold": 0.2
                }
            ),
            AttackConfig(
                name="patch_strong_large",
                attack_type="adversarial_patch",
                budget={"queries": 5000, "compute_time": 600.0},
                strength="strong",
                success_metrics={"attack_success_rate": 0.6, "targeted_success_rate": 0.5},
                parameters={
                    "patch_size": (32, 32),
                    "patch_location": "corner",
                    "iterations": 1000,
                    "learning_rate": 0.03,
                    "target_class": 0,
                    "success_threshold": 0.3
                }
            ),
            
            # Universal Perturbation Attacks
            AttackConfig(
                name="universal_weak_small_epsilon",
                attack_type="universal_perturbation",
                budget={"queries": 2000, "compute_time": 200.0},
                strength="weak",
                success_metrics={"fooling_rate": 0.4},
                parameters={
                    "epsilon": 0.05,
                    "iterations": 200,
                    "xi": 5.0,
                    "target_fooling_rate": 0.3,
                    "success_threshold": 0.2
                }
            ),
            AttackConfig(
                name="universal_moderate_standard",
                attack_type="universal_perturbation",
                budget={"queries": 5000, "compute_time": 400.0},
                strength="moderate",
                success_metrics={"fooling_rate": 0.6},
                parameters={
                    "epsilon": 0.1,
                    "iterations": 500,
                    "xi": 10.0,
                    "target_fooling_rate": 0.5,
                    "success_threshold": 0.4
                }
            ),
            AttackConfig(
                name="universal_strong_large_epsilon",
                attack_type="universal_perturbation",
                budget={"queries": 10000, "compute_time": 800.0},
                strength="strong",
                success_metrics={"fooling_rate": 0.8},
                parameters={
                    "epsilon": 0.2,
                    "iterations": 1000,
                    "xi": 15.0,
                    "target_fooling_rate": 0.7,
                    "success_threshold": 0.6
                }
            ),
            
            # Model Extraction Attacks (Vision-specific)
            AttackConfig(
                name="vision_extraction_weak_prediction",
                attack_type="model_extraction",
                budget={"queries": 3000, "compute_time": 300.0},
                strength="weak",
                success_metrics={"agreement_rate": 0.6, "fidelity": 0.6},
                parameters={
                    "method": "prediction",
                    "query_budget": 3000,
                    "architecture": "simple",
                    "num_classes": 10,
                    "epochs": 20,
                    "success_threshold": 0.5
                }
            ),
            AttackConfig(
                name="vision_extraction_moderate_resnet",
                attack_type="model_extraction",
                budget={"queries": 7000, "compute_time": 600.0},
                strength="moderate",
                success_metrics={"agreement_rate": 0.75, "fidelity": 0.75},
                parameters={
                    "method": "prediction",
                    "query_budget": 7000,
                    "architecture": "resnet18",
                    "num_classes": 10,
                    "epochs": 40,
                    "success_threshold": 0.7
                }
            ),
            AttackConfig(
                name="vision_extraction_strong_jacobian",
                attack_type="model_extraction", 
                budget={"queries": 15000, "compute_time": 1200.0},
                strength="strong",
                success_metrics={"agreement_rate": 0.85, "fidelity": 0.85},
                parameters={
                    "method": "jacobian",
                    "query_budget": 15000,
                    "architecture": "resnet18",
                    "num_classes": 10,
                    "epochs": 50,
                    "success_threshold": 0.8
                }
            ),
            
            # Backdoor Attacks
            AttackConfig(
                name="backdoor_weak_small_trigger",
                attack_type="backdoor",
                budget={"queries": 1000, "compute_time": 180.0, "epochs": 5},
                strength="weak",
                success_metrics={"backdoor_success_rate": 0.7, "clean_accuracy": 0.8},
                parameters={
                    "trigger_size": (3, 3),
                    "trigger_location": "bottom_right",
                    "pattern_type": "solid",
                    "target_class": 0,
                    "poisoning_rate": 0.05,
                    "epochs": 5,
                    "learning_rate": 0.001,
                    "success_threshold": 0.6
                }
            ),
            AttackConfig(
                name="backdoor_moderate_checkerboard",
                attack_type="backdoor",
                budget={"queries": 2000, "compute_time": 360.0, "epochs": 10},
                strength="moderate",
                success_metrics={"backdoor_success_rate": 0.85, "clean_accuracy": 0.85},
                parameters={
                    "trigger_size": (4, 4),
                    "trigger_location": "top_left",
                    "pattern_type": "checkerboard",
                    "target_class": 1,
                    "poisoning_rate": 0.1,
                    "epochs": 10,
                    "learning_rate": 0.001,
                    "success_threshold": 0.8
                }
            ),
            AttackConfig(
                name="backdoor_strong_complex_trigger",
                attack_type="backdoor",
                budget={"queries": 5000, "compute_time": 600.0, "epochs": 20},
                strength="strong",
                success_metrics={"backdoor_success_rate": 0.95, "clean_accuracy": 0.9},
                parameters={
                    "trigger_size": (6, 6),
                    "trigger_location": "random",
                    "pattern_type": "custom",
                    "target_class": 9,
                    "poisoning_rate": 0.15,
                    "epochs": 20,
                    "learning_rate": 0.0005,
                    "success_threshold": 0.9
                }
            )
        ]
    
    @staticmethod
    def get_all_configs() -> Dict[str, List[AttackConfig]]:
        """Get all standard attack configurations."""
        return {
            "distillation": StandardAttackSuite.get_distillation_configs(),
            "compression": StandardAttackSuite.get_compression_configs(),
            "wrapper": StandardAttackSuite.get_wrapper_configs(),
            "extraction": StandardAttackSuite.get_extraction_configs(),
            "vision_specific": StandardAttackSuite.get_vision_specific_configs()
        }


class AdaptiveAttackSuite:
    """Attacks that adapt based on defense responses."""
    
    def __init__(self, defense_observations: List[Dict] = None):
        """
        Initialize with observations of defense behavior.
        
        Args:
            defense_observations: List of defense response observations
        """
        self.defense_observations = defense_observations or []
        self.adaptation_history = []
        self.weakness_patterns = {}
        
        if self.defense_observations:
            self._analyze_defense_patterns()
    
    def add_defense_observation(self, observation: Dict):
        """Add a new defense observation."""
        self.defense_observations.append(observation)
        self._update_weakness_patterns(observation)
    
    def _analyze_defense_patterns(self):
        """Analyze defense observations to identify patterns."""
        if not self.defense_observations:
            return
        
        # Aggregate detection rates by attack type
        detection_rates = {}
        success_rates = {}
        
        for obs in self.defense_observations:
            attack_type = obs.get('attack_type', 'unknown')
            detected = obs.get('detected', False)
            successful = obs.get('attack_success', False)
            
            if attack_type not in detection_rates:
                detection_rates[attack_type] = []
                success_rates[attack_type] = []
            
            detection_rates[attack_type].append(detected)
            success_rates[attack_type].append(successful)
        
        # Compute weakness scores
        for attack_type in detection_rates:
            det_rate = np.mean(detection_rates[attack_type])
            succ_rate = np.mean(success_rates[attack_type])
            
            # Lower detection rate + higher success rate = higher weakness
            weakness_score = (1 - det_rate) * succ_rate
            
            self.weakness_patterns[attack_type] = {
                'detection_rate': det_rate,
                'success_rate': succ_rate,
                'weakness_score': weakness_score,
                'sample_count': len(detection_rates[attack_type])
            }
    
    def _update_weakness_patterns(self, observation: Dict):
        """Update weakness patterns with new observation."""
        attack_type = observation.get('attack_type', 'unknown')
        
        if attack_type in self.weakness_patterns:
            pattern = self.weakness_patterns[attack_type]
            n = pattern['sample_count']
            
            # Update running averages
            detected = observation.get('detected', False)
            successful = observation.get('attack_success', False)
            
            pattern['detection_rate'] = (pattern['detection_rate'] * n + detected) / (n + 1)
            pattern['success_rate'] = (pattern['success_rate'] * n + successful) / (n + 1)
            pattern['weakness_score'] = (1 - pattern['detection_rate']) * pattern['success_rate']
            pattern['sample_count'] = n + 1
        else:
            # New attack type
            detected = observation.get('detected', False)
            successful = observation.get('attack_success', False)
            
            self.weakness_patterns[attack_type] = {
                'detection_rate': float(detected),
                'success_rate': float(successful),
                'weakness_score': (1 - float(detected)) * float(successful),
                'sample_count': 1
            }
    
    def generate_adaptive_config(self, 
                                attack_type: str,
                                base_config: AttackConfig = None,
                                defense_weaknesses: Dict = None) -> AttackConfig:
        """
        Generate attack config targeting observed weaknesses.
        
        Args:
            attack_type: Type of attack to generate
            base_config: Base configuration to adapt
            defense_weaknesses: Specific weaknesses to target
            
        Returns:
            Adapted attack configuration
        """
        # Use provided weaknesses or inferred patterns
        weaknesses = defense_weaknesses or self.weakness_patterns.get(attack_type, {})
        
        # Get base config if not provided
        if base_config is None:
            all_configs = StandardAttackSuite.get_all_configs()
            if attack_type in all_configs and all_configs[attack_type]:
                base_config = all_configs[attack_type][1]  # Use moderate strength as base
            else:
                raise ValueError(f"No base configuration available for {attack_type}")
        
        # Create adaptive config
        adaptive_config = copy.deepcopy(base_config)
        adaptive_config.name = f"adaptive_{attack_type}_{int(time.time())}"
        adaptive_config.strength = "adaptive"
        
        # Adapt based on observed weaknesses
        weakness_score = weaknesses.get('weakness_score', 0.5)
        detection_rate = weaknesses.get('detection_rate', 0.5)
        
        # If defense is weak against this attack type, use lighter resources
        if weakness_score > 0.7:
            # Defense is vulnerable, use efficient attack
            adaptive_config.budget = self._scale_budget(adaptive_config.budget, 0.7)
            adaptive_config.success_metrics = self._relax_metrics(adaptive_config.success_metrics, 0.9)
        elif weakness_score < 0.3:
            # Defense is strong, use more resources
            adaptive_config.budget = self._scale_budget(adaptive_config.budget, 1.5)
            adaptive_config.success_metrics = self._tighten_metrics(adaptive_config.success_metrics, 1.1)
        
        # Adapt parameters based on attack type
        if attack_type == "distillation":
            self._adapt_distillation_params(adaptive_config, weaknesses)
        elif attack_type == "compression":
            self._adapt_compression_params(adaptive_config, weaknesses)
        elif attack_type == "wrapper":
            self._adapt_wrapper_params(adaptive_config, weaknesses)
        
        adaptive_config.metadata['adaptation_reason'] = f"Targeting weakness_score={weakness_score:.3f}"
        adaptive_config.metadata['base_config'] = base_config.name
        
        return adaptive_config
    
    def _scale_budget(self, budget: Dict[str, Any], factor: float) -> Dict[str, Any]:
        """Scale budget parameters."""
        scaled = budget.copy()
        for key, value in scaled.items():
            if isinstance(value, (int, float)):
                if isinstance(value, int):
                    scaled[key] = max(1, int(value * factor))
                else:
                    scaled[key] = value * factor
        return scaled
    
    def _relax_metrics(self, metrics: Dict[str, float], factor: float) -> Dict[str, float]:
        """Relax success metrics (lower targets)."""
        relaxed = {}
        for key, value in metrics.items():
            if key in ['accuracy_drop', 'loss', 'error_rate']:
                # Higher is worse, so increase threshold
                relaxed[key] = value * factor
            else:
                # Lower is worse, so decrease threshold
                relaxed[key] = value / factor
        return relaxed
    
    def _tighten_metrics(self, metrics: Dict[str, float], factor: float) -> Dict[str, float]:
        """Tighten success metrics (higher targets)."""
        tightened = {}
        for key, value in metrics.items():
            if key in ['accuracy_drop', 'loss', 'error_rate']:
                # Higher is worse, so decrease threshold
                tightened[key] = value / factor
            else:
                # Lower is worse, so increase threshold
                tightened[key] = min(1.0, value * factor)
        return tightened
    
    def _adapt_distillation_params(self, config: AttackConfig, weaknesses: Dict):
        """Adapt distillation-specific parameters."""
        params = config.parameters
        
        # If detection rate is high, use more subtle distillation
        detection_rate = weaknesses.get('detection_rate', 0.5)
        
        if detection_rate > 0.7:
            # High detection, use subtler approach
            params['temperature'] = max(2.0, params.get('temperature', 3.0) - 1.0)
            params['alpha'] = min(0.9, params.get('alpha', 0.7) + 0.1)
            params['learning_rate'] = params.get('learning_rate', 0.001) * 0.5
        elif detection_rate < 0.3:
            # Low detection, can be more aggressive
            params['temperature'] = min(8.0, params.get('temperature', 3.0) + 2.0)
            params['alpha'] = max(0.5, params.get('alpha', 0.7) - 0.1)
            params['learning_rate'] = params.get('learning_rate', 0.001) * 2.0
    
    def _adapt_compression_params(self, config: AttackConfig, weaknesses: Dict):
        """Adapt compression-specific parameters."""
        params = config.parameters
        detection_rate = weaknesses.get('detection_rate', 0.5)
        
        if detection_rate > 0.7:
            # High detection, use lighter compression
            if 'compression_ratio' in config.budget:
                config.budget['compression_ratio'] *= 0.8
            params['fine_tune_epochs'] = params.get('fine_tune_epochs', 5) * 2
        elif detection_rate < 0.3:
            # Low detection, can compress more aggressively
            if 'compression_ratio' in config.budget:
                config.budget['compression_ratio'] = min(0.95, config.budget['compression_ratio'] * 1.2)
    
    def _adapt_wrapper_params(self, config: AttackConfig, weaknesses: Dict):
        """Adapt wrapper-specific parameters."""
        params = config.parameters
        detection_rate = weaknesses.get('detection_rate', 0.5)
        
        if detection_rate > 0.7:
            # High detection, use simpler wrapper
            params['wrapper_layers'] = max(1, params.get('wrapper_layers', 2) - 1)
            params['hidden_size'] = max(32, params.get('hidden_size', 64) // 2)
            params['dropout'] = params.get('dropout', 0.1) + 0.1
        elif detection_rate < 0.3:
            # Low detection, can use more complex wrapper
            params['wrapper_layers'] = min(6, params.get('wrapper_layers', 2) + 1)
            params['hidden_size'] = min(512, params.get('hidden_size', 64) * 2)
    
    def evolutionary_attack(self, 
                          base_attack_type: str,
                          population_size: int = 20,
                          generations: int = 10,
                          mutation_rate: float = 0.1) -> List[AttackConfig]:
        """
        Evolve attack parameters using genetic algorithm.
        
        Args:
            base_attack_type: Base attack type to evolve
            population_size: Size of population per generation
            generations: Number of generations to evolve
            mutation_rate: Probability of parameter mutation
            
        Returns:
            List of evolved attack configurations
        """
        # Initialize population with random variations
        all_configs = StandardAttackSuite.get_all_configs()
        if base_attack_type not in all_configs:
            raise ValueError(f"Unknown attack type: {base_attack_type}")
        
        base_configs = all_configs[base_attack_type]
        population = []
        
        # Create initial population
        for i in range(population_size):
            base_config = random.choice(base_configs)
            individual = self._mutate_config(base_config, mutation_rate * 2)  # Higher initial diversity
            individual.name = f"evolved_{base_attack_type}_gen0_ind{i}"
            population.append(individual)
        
        evolution_history = [population.copy()]
        
        # Evolve over generations
        for gen in range(generations):
            # Evaluate fitness (placeholder - would need actual attack execution)
            fitness_scores = [self._estimate_fitness(config) for config in population]
            
            # Select parents (tournament selection)
            new_population = []
            for _ in range(population_size):
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover and mutation
                child = self._crossover_configs(parent1, parent2)
                child = self._mutate_config(child, mutation_rate)
                child.name = f"evolved_{base_attack_type}_gen{gen+1}_ind{len(new_population)}"
                
                new_population.append(child)
            
            population = new_population
            evolution_history.append(population.copy())
        
        # Return final generation sorted by estimated fitness
        final_fitness = [self._estimate_fitness(config) for config in population]
        sorted_indices = sorted(range(len(final_fitness)), key=lambda i: final_fitness[i], reverse=True)
        sorted_population = [population[i] for i in sorted_indices]
        
        return sorted_population
    
    def _estimate_fitness(self, config: AttackConfig) -> float:
        """
        Estimate fitness of a configuration without execution.
        
        This is a placeholder that estimates based on configuration parameters.
        In practice, this would involve actually running the attack.
        """
        # Base fitness from attack strength
        strength_scores = {'weak': 0.3, 'moderate': 0.6, 'strong': 0.9, 'adaptive': 0.8}
        fitness = strength_scores.get(config.strength, 0.5)
        
        # Bonus for higher budget utilization
        budget_score = 0.0
        if 'queries' in config.budget:
            budget_score += min(1.0, config.budget['queries'] / 10000) * 0.2
        if 'epochs' in config.budget:
            budget_score += min(1.0, config.budget['epochs'] / 50) * 0.2
        
        fitness += budget_score
        
        # Penalty for unrealistic parameters
        if config.attack_type == "distillation":
            temp = config.parameters.get('temperature', 3.0)
            # Handle both single values and lists (for multi-stage)
            if isinstance(temp, list):
                temp = temp[0] if temp else 3.0
            if temp < 1.0 or temp > 10.0:
                fitness -= 0.3
        
        return max(0.0, min(1.0, fitness))
    
    def _tournament_selection(self, population: List[AttackConfig], 
                            fitness_scores: List[float], 
                            tournament_size: int = 3) -> AttackConfig:
        """Select parent using tournament selection."""
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover_configs(self, parent1: AttackConfig, parent2: AttackConfig) -> AttackConfig:
        """Create child configuration from two parents."""
        child = copy.deepcopy(parent1)
        
        # Crossover budget parameters
        for key in child.budget:
            if key in parent2.budget and random.random() < 0.5:
                child.budget[key] = parent2.budget[key]
        
        # Crossover attack parameters
        for key in child.parameters:
            if key in parent2.parameters and random.random() < 0.5:
                child.parameters[key] = parent2.parameters[key]
        
        # Crossover success metrics
        for key in child.success_metrics:
            if key in parent2.success_metrics and random.random() < 0.5:
                child.success_metrics[key] = parent2.success_metrics[key]
        
        return child
    
    def _mutate_config(self, config: AttackConfig, mutation_rate: float) -> AttackConfig:
        """Apply random mutations to configuration."""
        mutated = copy.deepcopy(config)
        
        # Mutate budget parameters
        for key, value in mutated.budget.items():
            if random.random() < mutation_rate and isinstance(value, (int, float)):
                if isinstance(value, int):
                    # ±20% variation for integers
                    delta = int(value * 0.2 * (random.random() - 0.5) * 2)
                    mutated.budget[key] = max(1, value + delta)
                else:
                    # ±20% variation for floats
                    delta = value * 0.2 * (random.random() - 0.5) * 2
                    mutated.budget[key] = max(0.01, value + delta)
        
        # Mutate attack parameters
        for key, value in mutated.parameters.items():
            if random.random() < mutation_rate:
                if isinstance(value, (int, float)):
                    if isinstance(value, int):
                        delta = max(1, int(value * 0.15 * (random.random() - 0.5) * 2))
                        mutated.parameters[key] = max(1, value + delta)
                    else:
                        delta = value * 0.15 * (random.random() - 0.5) * 2
                        mutated.parameters[key] = max(0.001, value + delta)
                elif isinstance(value, bool):
                    mutated.parameters[key] = not value
        
        return mutated


class ComprehensiveAttackSuite:
    """Combines all attack types with systematic parameter exploration."""
    
    def __init__(self):
        self.standard_suite = StandardAttackSuite()
        self.adaptive_suite = None  # Initialize when needed
    
    def get_all_standard_configs(self) -> List[AttackConfig]:
        """Get all standard attack configurations."""
        all_configs = []
        config_dict = self.standard_suite.get_all_configs()
        
        for attack_type, configs in config_dict.items():
            all_configs.extend(configs)
        
        return all_configs
    
    def get_configs_by_strength(self, strength: str) -> List[AttackConfig]:
        """Get configurations filtered by strength level."""
        all_configs = self.get_all_standard_configs()
        return [config for config in all_configs if config.strength == strength]
    
    def get_configs_by_type(self, attack_type: str) -> List[AttackConfig]:
        """Get configurations filtered by attack type."""
        all_configs = self.get_all_standard_configs()
        return [config for config in all_configs if config.attack_type == attack_type]
    
    def generate_parameter_sweep(self, 
                               base_config: AttackConfig,
                               parameter_ranges: Dict[str, List]) -> List[AttackConfig]:
        """
        Generate systematic parameter sweep around base configuration.
        
        Args:
            base_config: Base configuration to vary
            parameter_ranges: Dict mapping parameter names to value lists
            
        Returns:
            List of configurations with varied parameters
        """
        configs = []
        
        # Generate all combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        from itertools import product
        for value_combination in product(*param_values):
            config = copy.deepcopy(base_config)
            config.name = f"{base_config.name}_sweep_{len(configs)}"
            
            # Update parameters
            for param_name, param_value in zip(param_names, value_combination):
                if param_name in config.parameters:
                    config.parameters[param_name] = param_value
                elif param_name in config.budget:
                    config.budget[param_name] = param_value
                else:
                    config.parameters[param_name] = param_value
            
            configs.append(config)
        
        return configs
    
    def get_benchmark_suite(self, 
                          suite_type: str = "standard",
                          max_configs_per_type: int = None) -> List[AttackConfig]:
        """
        Get benchmark suite for evaluation.
        
        Args:
            suite_type: Type of suite ('standard', 'comprehensive', 'quick')
            max_configs_per_type: Maximum configurations per attack type
            
        Returns:
            List of benchmark configurations
        """
        if suite_type == "quick":
            # One config per attack type and strength
            configs = []
            config_dict = self.standard_suite.get_all_configs()
            
            for attack_type, type_configs in config_dict.items():
                # Get one config per strength level
                for strength in ['weak', 'moderate', 'strong']:
                    strength_configs = [c for c in type_configs if c.strength == strength]
                    if strength_configs:
                        configs.append(strength_configs[0])
            
            return configs
        
        elif suite_type == "standard":
            # All standard configurations
            configs = self.get_all_standard_configs()
            
            if max_configs_per_type:
                # Limit per attack type
                limited_configs = []
                config_dict = {}
                
                for config in configs:
                    if config.attack_type not in config_dict:
                        config_dict[config.attack_type] = []
                    config_dict[config.attack_type].append(config)
                
                for attack_type, type_configs in config_dict.items():
                    limited_configs.extend(type_configs[:max_configs_per_type])
                
                return limited_configs
            
            return configs
        
        elif suite_type == "comprehensive":
            # All standard configs plus parameter sweeps
            configs = self.get_all_standard_configs()
            
            # Add parameter sweeps for key configurations
            for attack_type in ['distillation', 'compression']:
                type_configs = self.get_configs_by_type(attack_type)
                if len(type_configs) > 1:
                    base_config = type_configs[1]  # Use moderate strength
                elif type_configs:
                    base_config = type_configs[0]  # Use first available
                else:
                    continue  # Skip if no configs available
                    
                if attack_type == "distillation":
                    parameter_ranges = {
                        'temperature': [2.0, 4.0, 6.0],
                        'alpha': [0.6, 0.8, 0.9],
                        'learning_rate': [0.001, 0.0005, 0.0001]
                    }
                else:  # compression
                    parameter_ranges = {
                        'compression_ratio': [0.3, 0.5, 0.7],
                        'fine_tune_epochs': [5, 10, 15]
                    }
                
                sweep_configs = self.generate_parameter_sweep(base_config, parameter_ranges)
                configs.extend(sweep_configs[:9])  # Limit sweep size
            
            return configs
        
        else:
            raise ValueError(f"Unknown suite type: {suite_type}")


# Benchmark suite registry
BENCHMARK_SUITES = {
    "standard": StandardAttackSuite(),
    "adaptive": AdaptiveAttackSuite,
    "comprehensive": ComprehensiveAttackSuite()
}


def get_benchmark_suite(suite_name: str) -> Any:
    """Get benchmark suite by name."""
    if suite_name not in BENCHMARK_SUITES:
        raise ValueError(f"Unknown benchmark suite: {suite_name}. Available: {list(BENCHMARK_SUITES.keys())}")
    
    suite = BENCHMARK_SUITES[suite_name]
    
    # Instantiate if it's a class
    if isinstance(suite, type):
        return suite()
    else:
        return suite


def create_custom_config(name: str,
                        attack_type: str,
                        budget: Dict[str, Any],
                        strength: str = "moderate",
                        success_metrics: Dict[str, float] = None,
                        parameters: Dict[str, Any] = None) -> AttackConfig:
    """
    Create a custom attack configuration.
    
    Args:
        name: Configuration name
        attack_type: Type of attack
        budget: Resource budget
        strength: Attack strength level
        success_metrics: Success criteria
        parameters: Attack-specific parameters
        
    Returns:
        Custom attack configuration
    """
    if success_metrics is None:
        # Default success metrics based on attack type
        if attack_type == "distillation":
            success_metrics = {"fidelity": 0.8, "accuracy": 0.85}
        elif attack_type in ["pruning", "quantization", "compression"]:
            success_metrics = {"accuracy_drop": 0.1, "compression_ratio": 0.5}
        elif attack_type == "wrapper":
            success_metrics = {"wrapper_adaptation": 0.7, "final_loss": 0.5}
        elif attack_type == "extraction":
            success_metrics = {"extraction_fidelity": 0.7, "accuracy": 0.8}
        else:
            success_metrics = {"success_rate": 0.7}
    
    return AttackConfig(
        name=name,
        attack_type=attack_type,
        budget=budget,
        strength=strength,
        success_metrics=success_metrics,
        parameters=parameters or {},
        metadata={"custom": True}
    )


class AttackExecutor:
    """Framework for executing attack configurations."""
    
    def __init__(self, target_model, data_loader=None, device: str = 'cpu'):
        """
        Initialize attack executor.
        
        Args:
            target_model: Model to attack
            data_loader: Data loader for attacks that need training data
            device: Device to use for computation
        """
        self.target_model = target_model
        self.data_loader = data_loader
        self.device = device
        self.execution_history = []
    
    def execute_attack(self, config: AttackConfig) -> AttackResult:
        """
        Execute a single attack configuration.
        
        Args:
            config: Attack configuration to execute
            
        Returns:
            Attack execution result
        """
        start_time = time.time()
        
        try:
            # Import attack functions
            from .attacks import (
                distillation_loop, fine_tune_wrapper, compression_attack_enhanced,
                evaluate_attack_metrics
            )
            
            # Execute based on attack type
            if config.attack_type == "distillation":
                result = self._execute_distillation(config, distillation_loop)
            elif config.attack_type in ["pruning", "quantization", "compression"]:
                result = self._execute_compression(config, compression_attack_enhanced)
            elif config.attack_type == "wrapper":
                result = self._execute_wrapper(config, fine_tune_wrapper)
            elif config.attack_type == "extraction":
                result = self._execute_extraction(config)
            else:
                raise ValueError(f"Unknown attack type: {config.attack_type}")
            
            execution_time = time.time() - start_time
            
            # Create attack result
            attack_result = AttackResult(
                config=config,
                success=result.get('success', False),
                metrics=result.get('metrics', {}),
                execution_time=execution_time,
                resources_used=result.get('resources_used', {}),
                detailed_results=result.get('detailed_results', {})
            )
            
            # Check success criteria
            attack_result.success = attack_result.meets_success_criteria()
            
            self.execution_history.append(attack_result)
            return attack_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            error_result = AttackResult(
                config=config,
                success=False,
                metrics={},
                execution_time=execution_time,
                resources_used={},
                error_message=str(e)
            )
            
            self.execution_history.append(error_result)
            return error_result
    
    def _execute_distillation(self, config: AttackConfig, distillation_func) -> Dict[str, Any]:
        """Execute distillation attack."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for distillation attacks")
        
        # Create student model (simplified - would need proper architecture)
        if hasattr(self.target_model, 'state_dict'):
            student_model = copy.deepcopy(self.target_model)
            # Reinitialize weights
            for param in student_model.parameters():
                if param.requires_grad:
                    nn.init.normal_(param, mean=0.0, std=0.02)
        else:
            raise ValueError("Target model must be a PyTorch model for distillation")
        
        # Execute distillation
        metrics = distillation_func(
            teacher_model=self.target_model,
            student_model=student_model,
            train_loader=self.data_loader,
            temperature=config.parameters.get('temperature', 3.0),
            alpha=config.parameters.get('alpha', 0.7),
            epochs=config.budget.get('epochs', 10),
            learning_rate=config.parameters.get('learning_rate', 0.001),
            device=self.device
        )
        
        return {
            'success': metrics.get('attack_success', False),
            'metrics': {
                'fidelity': metrics.get('final_fidelity', 0.0),
                'accuracy': 1.0 - metrics.get('accuracy_drop', 1.0),  # Estimate
                'convergence_epoch': metrics.get('convergence_epoch', -1),
                'training_time': metrics.get('total_training_time', 0.0)
            },
            'resources_used': {
                'epochs': metrics.get('epochs_trained', 0),
                'time': metrics.get('total_training_time', 0.0)
            },
            'detailed_results': metrics
        }
    
    def _execute_compression(self, config: AttackConfig, compression_func) -> Dict[str, Any]:
        """Execute compression attack."""
        compression_type = config.attack_type
        if config.attack_type == "compression":
            compression_type = "both"  # Both pruning and quantization
        
        compressed_model, metrics = compression_func(
            model=copy.deepcopy(self.target_model) if hasattr(self.target_model, 'state_dict') else self.target_model,
            compression_type=compression_type,
            compression_ratio=config.budget.get('compression_ratio', 0.5),
            fine_tune_epochs=config.parameters.get('fine_tune_epochs', 5),
            learning_rate=config.parameters.get('learning_rate', 0.0001),
            train_loader=self.data_loader,
            device=self.device
        )
        
        return {
            'success': metrics.get('compression_success', False),
            'metrics': {
                'compression_ratio': metrics.get('actual_ratio', 0.0),
                'accuracy_drop': 0.1,  # Estimate - would need actual evaluation
                'compression_time': metrics.get('compression_time', 0.0),
                'fine_tune_success': metrics.get('fine_tune_success', False)
            },
            'resources_used': {
                'compression_time': metrics.get('compression_time', 0.0),
                'fine_tune_time': metrics.get('fine_tune_time', 0.0),
                'total_time': metrics.get('total_time', 0.0)
            },
            'detailed_results': metrics
        }
    
    def _execute_wrapper(self, config: AttackConfig, wrapper_func) -> Dict[str, Any]:
        """Execute wrapper attack."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for wrapper attacks")
        
        # Create wrapper layers
        num_layers = config.parameters.get('wrapper_layers', 2)
        hidden_size = config.parameters.get('hidden_size', 64)
        
        # Simple wrapper layers (would need to be adapted to model architecture)
        wrapper_layers = []
        for i in range(num_layers):
            if i == 0:
                # First layer - needs to match model output size
                input_size = 10  # Placeholder
                wrapper_layers.append(nn.Linear(input_size, hidden_size))
            elif i == num_layers - 1:
                # Last layer - output layer
                wrapper_layers.append(nn.Linear(hidden_size, 10))  # Placeholder
            else:
                # Hidden layers
                wrapper_layers.append(nn.Linear(hidden_size, hidden_size))
            
            if i < num_layers - 1:
                wrapper_layers.append(nn.ReLU())
                if config.parameters.get('dropout', 0.0) > 0:
                    wrapper_layers.append(nn.Dropout(config.parameters['dropout']))
        
        wrapper_model, metrics = wrapper_func(
            base_model=self.target_model,
            wrapper_layers=wrapper_layers,
            train_loader=self.data_loader,
            attack_budget=config.budget,
            optimization_config={
                'device': self.device,
                'learning_rate': config.parameters.get('learning_rate', 0.001),
                'optimizer': config.parameters.get('optimizer', 'adam')
            }
        )
        
        return {
            'success': metrics.get('attack_success', False),
            'metrics': {
                'wrapper_adaptation': metrics.get('total_adaptation', 0.0),
                'final_loss': metrics.get('final_loss', float('inf')),
                'training_time': metrics.get('training_time', 0.0),
                'queries_used': metrics.get('queries_used', 0)
            },
            'resources_used': {
                'queries': metrics.get('queries_used', 0),
                'epochs': metrics.get('epochs_completed', 0),
                'time': metrics.get('training_time', 0.0)
            },
            'detailed_results': metrics
        }
    
    def _execute_extraction(self, config: AttackConfig) -> Dict[str, Any]:
        """Execute model extraction attack."""
        from .attacks import extraction_attack
        
        extracted_model = extraction_attack(
            model=self.target_model,
            data_loader=self.data_loader,
            query_budget=config.budget.get('queries', 10000),
            epochs=config.budget.get('epochs', 10),
            lr=config.parameters.get('learning_rate', 0.001),
            device=self.device
        )
        
        # Evaluate extraction quality (simplified)
        extraction_fidelity = 0.8  # Placeholder - would need actual evaluation
        
        return {
            'success': extraction_fidelity > config.success_metrics.get('extraction_fidelity', 0.7),
            'metrics': {
                'extraction_fidelity': extraction_fidelity,
                'accuracy': 0.85,  # Placeholder
                'queries_used': config.budget.get('queries', 10000)
            },
            'resources_used': {
                'queries': config.budget.get('queries', 10000),
                'epochs': config.budget.get('epochs', 10),
                'time': 300.0  # Placeholder
            },
            'detailed_results': {'extracted_model': extracted_model}
        }
    
    def execute_suite(self, configs: List[AttackConfig]) -> List[AttackResult]:
        """
        Execute a suite of attack configurations.
        
        Args:
            configs: List of attack configurations
            
        Returns:
            List of attack results
        """
        results = []
        
        for i, config in enumerate(configs):
            print(f"Executing attack {i+1}/{len(configs)}: {config.name}")
            result = self.execute_attack(config)
            results.append(result)
            
            print(f"  Success: {result.success}")
            print(f"  Execution time: {result.execution_time:.2f}s")
            if result.error_message:
                print(f"  Error: {result.error_message}")
            print()
        
        return results
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executed attacks."""
        if not self.execution_history:
            return {"total_attacks": 0}
        
        total_attacks = len(self.execution_history)
        successful_attacks = sum(1 for result in self.execution_history if result.success)
        total_time = sum(result.execution_time for result in self.execution_history)
        
        # Group by attack type
        type_stats = {}
        for result in self.execution_history:
            attack_type = result.config.attack_type
            if attack_type not in type_stats:
                type_stats[attack_type] = {'total': 0, 'successful': 0, 'avg_time': 0.0}
            
            type_stats[attack_type]['total'] += 1
            if result.success:
                type_stats[attack_type]['successful'] += 1
            type_stats[attack_type]['avg_time'] += result.execution_time
        
        # Compute averages
        for stats in type_stats.values():
            if stats['total'] > 0:
                stats['success_rate'] = stats['successful'] / stats['total']
                stats['avg_time'] /= stats['total']
        
        return {
            'total_attacks': total_attacks,
            'successful_attacks': successful_attacks,
            'success_rate': successful_attacks / total_attacks,
            'total_execution_time': total_time,
            'average_execution_time': total_time / total_attacks,
            'attack_type_stats': type_stats
        }


class AttackSuiteEvaluator:
    """Evaluate and compare attack suite performance."""
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_suite_performance(self, 
                                 attack_results: List[AttackResult],
                                 defense_results: List[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate performance of an attack suite.
        
        Args:
            attack_results: Results from executing attack suite
            defense_results: Optional defense detection results
            
        Returns:
            Comprehensive evaluation metrics
        """
        if not attack_results:
            return {"error": "No attack results provided"}
        
        # Basic statistics
        total_attacks = len(attack_results)
        successful_attacks = sum(1 for result in attack_results if result.success)
        success_rate = successful_attacks / total_attacks
        
        # Resource utilization
        total_time = sum(result.execution_time for result in attack_results)
        avg_time = total_time / total_attacks
        
        # Attack type breakdown
        type_breakdown = {}
        strength_breakdown = {}
        
        for result in attack_results:
            attack_type = result.config.attack_type
            strength = result.config.strength
            
            # Type breakdown
            if attack_type not in type_breakdown:
                type_breakdown[attack_type] = {'total': 0, 'successful': 0, 'avg_time': 0.0}
            type_breakdown[attack_type]['total'] += 1
            if result.success:
                type_breakdown[attack_type]['successful'] += 1
            type_breakdown[attack_type]['avg_time'] += result.execution_time
            
            # Strength breakdown
            if strength not in strength_breakdown:
                strength_breakdown[strength] = {'total': 0, 'successful': 0, 'avg_time': 0.0}
            strength_breakdown[strength]['total'] += 1
            if result.success:
                strength_breakdown[strength]['successful'] += 1
            strength_breakdown[strength]['avg_time'] += result.execution_time
        
        # Compute success rates and averages
        for stats in type_breakdown.values():
            if stats['total'] > 0:
                stats['success_rate'] = stats['successful'] / stats['total']
                stats['avg_time'] /= stats['total']
        
        for stats in strength_breakdown.values():
            if stats['total'] > 0:
                stats['success_rate'] = stats['successful'] / stats['total']
                stats['avg_time'] /= stats['total']
        
        # Defense evasion analysis
        evasion_analysis = {}
        if defense_results:
            detected_attacks = sum(1 for dr in defense_results if dr.get('detected', False))
            evasion_rate = 1 - (detected_attacks / len(defense_results))
            
            evasion_analysis = {
                'total_defense_evaluations': len(defense_results),
                'detected_attacks': detected_attacks,
                'evasion_rate': evasion_rate,
                'detection_rate': detected_attacks / len(defense_results)
            }
        
        # Efficiency metrics
        efficiency_metrics = {
            'attacks_per_hour': 3600 / avg_time if avg_time > 0 else 0,
            'successful_attacks_per_hour': 3600 * success_rate / avg_time if avg_time > 0 else 0,
            'time_per_successful_attack': total_time / successful_attacks if successful_attacks > 0 else float('inf')
        }
        
        evaluation = {
            'summary': {
                'total_attacks': total_attacks,
                'successful_attacks': successful_attacks,
                'success_rate': success_rate,
                'total_execution_time': total_time,
                'average_execution_time': avg_time
            },
            'breakdown': {
                'by_attack_type': type_breakdown,
                'by_strength': strength_breakdown
            },
            'evasion_analysis': evasion_analysis,
            'efficiency_metrics': efficiency_metrics,
            'failed_attacks': [
                {
                    'name': result.config.name,
                    'type': result.config.attack_type,
                    'error': result.error_message
                }
                for result in attack_results 
                if not result.success and result.error_message
            ]
        }
        
        self.evaluation_history.append(evaluation)
        return evaluation
    
    def compare_suites(self, 
                      suite_evaluations: List[Dict[str, Any]],
                      suite_names: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple attack suite evaluations.
        
        Args:
            suite_evaluations: List of suite evaluation results
            suite_names: Optional names for the suites
            
        Returns:
            Comparative analysis
        """
        if not suite_evaluations:
            return {"error": "No suite evaluations provided"}
        
        if suite_names is None:
            suite_names = [f"Suite_{i+1}" for i in range(len(suite_evaluations))]
        
        comparison = {
            'suite_names': suite_names,
            'metrics_comparison': {},
            'rankings': {},
            'recommendations': []
        }
        
        # Extract key metrics for comparison
        metrics = ['success_rate', 'average_execution_time', 'attacks_per_hour']
        
        for metric in metrics:
            comparison['metrics_comparison'][metric] = []
            
            for i, evaluation in enumerate(suite_evaluations):
                if metric in evaluation['summary']:
                    value = evaluation['summary'][metric]
                elif metric in evaluation.get('efficiency_metrics', {}):
                    value = evaluation['efficiency_metrics'][metric]
                else:
                    value = 0.0
                
                comparison['metrics_comparison'][metric].append({
                    'suite': suite_names[i],
                    'value': value
                })
        
        # Rank suites by different criteria
        for metric in metrics:
            values = [(item['suite'], item['value']) for item in comparison['metrics_comparison'][metric]]
            
            # Sort appropriately (higher is better for success_rate and attacks_per_hour, lower for time)
            reverse = metric != 'average_execution_time'
            ranked = sorted(values, key=lambda x: x[1], reverse=reverse)
            
            comparison['rankings'][metric] = [
                {'rank': i+1, 'suite': suite, 'value': value}
                for i, (suite, value) in enumerate(ranked)
            ]
        
        # Generate recommendations
        best_success_rate = comparison['rankings']['success_rate'][0]
        best_efficiency = comparison['rankings']['attacks_per_hour'][0]
        
        if best_success_rate['suite'] == best_efficiency['suite']:
            comparison['recommendations'].append(
                f"{best_success_rate['suite']} offers the best balance of success rate and efficiency"
            )
        else:
            comparison['recommendations'].append(
                f"{best_success_rate['suite']} has the highest success rate ({best_success_rate['value']:.2%})"
            )
            comparison['recommendations'].append(
                f"{best_efficiency['suite']} has the highest efficiency ({best_efficiency['value']:.1f} attacks/hour)"
            )
        
        return comparison