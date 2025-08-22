"""
Comprehensive Defense Mechanisms for Proof-of-Training Verification

This module implements active defense systems including adaptive verification,
input filtering, and randomized defenses to protect against sophisticated attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import copy
from collections import deque
import warnings

try:
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    stats = None
    StandardScaler = None
    IsolationForest = None
    OneClassSVM = None

logger = logging.getLogger(__name__)


@dataclass
class DefenseConfig:
    """Configuration for defense mechanisms."""
    defense_type: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    adaptation_rate: float = 0.1
    memory_size: int = 1000
    update_frequency: int = 10


@dataclass
class AttackObservation:
    """Record of observed attack attempt."""
    timestamp: float
    attack_type: str
    attack_strength: float
    detection_confidence: float
    input_characteristics: Dict[str, float]
    defense_response: Dict[str, Any]
    success: bool


class BaseVerifier(ABC):
    """Base class for PoT verifiers."""
    
    @abstractmethod
    def verify(self, input_data: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
        """Perform verification and return results."""
        pass
    
    @abstractmethod
    def compute_confidence(self, verification_result: Dict[str, Any]) -> float:
        """Compute confidence score for verification result."""
        pass


class MockBaseVerifier(BaseVerifier):
    """Mock verifier for testing and demonstration."""
    
    def __init__(self, base_confidence: float = 0.8):
        self.base_confidence = base_confidence
        
    def verify(self, input_data: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
        """Mock verification with random results."""
        # Simulate verification process
        time.sleep(0.001)  # Small delay to simulate processing
        
        # Simple heuristic: larger inputs are more suspicious
        input_magnitude = torch.norm(input_data).item()
        suspicion_score = min(input_magnitude / 10.0, 1.0)
        
        confidence = self.base_confidence * (1 - suspicion_score * 0.3)
        is_legitimate = confidence > 0.5
        
        return {
            'verified': is_legitimate,
            'confidence': confidence,
            'suspicion_score': suspicion_score,
            'input_magnitude': input_magnitude,
            'processing_time': 0.001
        }
        
    def compute_confidence(self, verification_result: Dict[str, Any]) -> float:
        """Extract confidence from verification result."""
        return verification_result.get('confidence', 0.0)


class AdaptiveVerifier:
    """
    Adaptive verification system that learns from attack patterns and adjusts defenses.
    
    This class implements dynamic threshold adjustment, attack pattern recognition,
    and multi-layered defense strategies that evolve based on observed threats.
    """
    
    def __init__(self, 
                 base_verifier: BaseVerifier,
                 config: Optional[DefenseConfig] = None):
        """
        Initialize adaptive verifier.
        
        Args:
            base_verifier: Base verification system to enhance
            config: Configuration for adaptive behavior
        """
        self.base_verifier = base_verifier
        self.config = config or DefenseConfig(
            defense_type='adaptive_verification',
            parameters={
                'adaptation_strength': 0.2,
                'pattern_memory': 500,
                'threshold_bounds': (0.1, 0.9)
            }
        )
        
        # Attack tracking and adaptation
        self.attack_history: deque = deque(maxlen=self.config.memory_size)
        self.attack_patterns: Dict[str, Dict] = {}
        self.adaptation_counter = 0
        
        # Dynamic thresholds
        self.thresholds = {
            'confidence_threshold': 0.7,
            'suspicion_threshold': 0.3,
            'anomaly_threshold': 0.5,
            'adaptation_threshold': 0.1
        }
        self.thresholds.update(self.config.thresholds)
        
        # Defense layers
        self.defense_layers: List[Callable] = []
        self.layer_weights: List[float] = []
        
        # Statistics tracking
        self.verification_stats = {
            'total_verifications': 0,
            'adaptations_made': 0,
            'attacks_detected': 0,
            'false_positives': 0,
            'average_confidence': 0.0
        }
        
    def verify(self, 
               input_data: torch.Tensor, 
               model: nn.Module,
               metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform adaptive verification with dynamic defense adjustment.
        
        Args:
            input_data: Input to verify
            model: Model being verified
            metadata: Additional context information
            
        Returns:
            Enhanced verification results with adaptive insights
        """
        start_time = time.time()
        
        # Apply defense layers first
        filtered_input = input_data
        defense_responses = []
        
        for i, (defense_layer, weight) in enumerate(zip(self.defense_layers, self.layer_weights)):
            try:
                layer_result = defense_layer(filtered_input, model)
                defense_responses.append({
                    'layer_id': i,
                    'weight': weight,
                    'result': layer_result
                })
                
                # Apply filtering if the layer modifies input
                if isinstance(layer_result, torch.Tensor):
                    filtered_input = layer_result
                    
            except Exception as e:
                logger.warning(f"Defense layer {i} failed: {e}")
                defense_responses.append({
                    'layer_id': i,
                    'weight': weight,
                    'result': None,
                    'error': str(e)
                })
        
        # Perform base verification
        base_result = self.base_verifier.verify(filtered_input, model)
        
        # Compute adaptive confidence adjustment
        confidence_adjustment = self._compute_confidence_adjustment(
            input_data, base_result, defense_responses
        )
        
        # Apply threshold adaptation
        adapted_confidence = base_result.get('confidence', 0.0) + confidence_adjustment
        adapted_confidence = np.clip(adapted_confidence, 0.0, 1.0)
        
        # Detect potential attacks
        attack_indicators = self._analyze_attack_indicators(
            input_data, base_result, defense_responses
        )
        
        # Create comprehensive result
        adaptive_result = {
            'verified': adapted_confidence > self.thresholds['confidence_threshold'],
            'original_confidence': base_result.get('confidence', 0.0),
            'adapted_confidence': adapted_confidence,
            'confidence_adjustment': confidence_adjustment,
            'attack_indicators': attack_indicators,
            'defense_responses': defense_responses,
            'thresholds_used': self.thresholds.copy(),
            'processing_time': time.time() - start_time,
            'adaptation_metadata': {
                'adaptations_made': self.verification_stats['adaptations_made'],
                'defense_layers_active': len(self.defense_layers),
                'pattern_memory_size': len(self.attack_history)
            }
        }
        
        # Update verification statistics
        self._update_verification_stats(adaptive_result)
        
        # Check if adaptation is needed
        if self._should_adapt(adaptive_result):
            self._trigger_adaptation(input_data, adaptive_result, metadata)
        
        return adaptive_result
        
    def adapt_to_attack(self, 
                       attack_type: str,
                       attack_observations: Dict[str, Any]) -> None:
        """
        Adapt verification strategy based on observed attack patterns.
        
        Args:
            attack_type: Type of attack observed
            attack_observations: Detailed observations about the attack
        """
        logger.info(f"Adapting to {attack_type} attack")
        
        # Record attack observation
        observation = AttackObservation(
            timestamp=time.time(),
            attack_type=attack_type,
            attack_strength=attack_observations.get('strength', 0.5),
            detection_confidence=attack_observations.get('detection_confidence', 0.5),
            input_characteristics=attack_observations.get('characteristics', {}),
            defense_response=attack_observations.get('defense_response', {}),
            success=attack_observations.get('success', False)
        )
        
        self.attack_history.append(observation)
        
        # Update attack patterns
        if attack_type not in self.attack_patterns:
            self.attack_patterns[attack_type] = {
                'count': 0,
                'success_rate': 0.0,
                'average_strength': 0.0,
                'characteristics_stats': {},
                'last_seen': 0.0,
                'adaptation_responses': []
            }
            
        pattern = self.attack_patterns[attack_type]
        pattern['count'] += 1
        pattern['last_seen'] = observation.timestamp
        
        # Update success rate (exponential moving average)
        alpha = 0.1
        pattern['success_rate'] = (
            alpha * float(observation.success) + 
            (1 - alpha) * pattern['success_rate']
        )
        
        # Update average strength
        pattern['average_strength'] = (
            alpha * observation.attack_strength + 
            (1 - alpha) * pattern['average_strength']
        )
        
        # Determine adaptation response
        adaptation_response = self._determine_adaptation_response(attack_type, observation)
        pattern['adaptation_responses'].append(adaptation_response)
        
        # Apply adaptations
        for adaptation in adaptation_response:
            self._apply_adaptation(adaptation)
            
        self.verification_stats['adaptations_made'] += 1
        logger.info(f"Applied {len(adaptation_response)} adaptations for {attack_type}")
        
    def update_thresholds(self, 
                         attack_samples: torch.Tensor,
                         legitimate_samples: Optional[torch.Tensor] = None) -> None:
        """
        Dynamically update verification thresholds based on new data.
        
        Args:
            attack_samples: Known attack samples for threshold calibration
            legitimate_samples: Known legitimate samples for comparison
        """
        logger.info("Updating verification thresholds")
        
        # Analyze attack sample characteristics
        attack_stats = self._compute_sample_statistics(attack_samples)
        
        if legitimate_samples is not None:
            legit_stats = self._compute_sample_statistics(legitimate_samples)
            
            # Compute optimal separation thresholds
            separation_analysis = self._analyze_threshold_separation(
                attack_stats, legit_stats
            )
            
            # Update thresholds based on separation analysis
            for threshold_name, current_value in self.thresholds.items():
                if threshold_name in separation_analysis:
                    new_value = separation_analysis[threshold_name]
                    
                    # Apply adaptation rate damping
                    adaptation_rate = self.config.adaptation_rate
                    updated_value = (
                        adaptation_rate * new_value + 
                        (1 - adaptation_rate) * current_value
                    )
                    
                    # Respect threshold bounds
                    bounds = self.config.parameters.get('threshold_bounds', (0.1, 0.9))
                    updated_value = np.clip(updated_value, bounds[0], bounds[1])
                    
                    logger.info(f"Threshold {threshold_name}: {current_value:.3f} -> {updated_value:.3f}")
                    self.thresholds[threshold_name] = updated_value
        else:
            # Update thresholds based on attack samples only
            self._update_thresholds_from_attacks(attack_stats)
            
    def add_defense_layer(self, defense_type: str, **kwargs) -> None:
        """
        Add new defense mechanism in response to attacks.
        
        Args:
            defense_type: Type of defense to add
            **kwargs: Defense-specific parameters
        """
        logger.info(f"Adding {defense_type} defense layer")
        
        if defense_type == 'input_filtering':
            defense_layer = self._create_input_filter(**kwargs)
        elif defense_type == 'noise_injection':
            defense_layer = self._create_noise_injector(**kwargs)
        elif defense_type == 'gradient_masking':
            defense_layer = self._create_gradient_masker(**kwargs)
        elif defense_type == 'ensemble_verification':
            defense_layer = self._create_ensemble_verifier(**kwargs)
        else:
            logger.warning(f"Unknown defense type: {defense_type}")
            return
            
        # Add layer with appropriate weight
        weight = kwargs.get('weight', 1.0)
        self.defense_layers.append(defense_layer)
        self.layer_weights.append(weight)
        
        logger.info(f"Added {defense_type} layer with weight {weight}")
        
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation history and current state."""
        return {
            'verification_stats': self.verification_stats.copy(),
            'current_thresholds': self.thresholds.copy(),
            'attack_patterns': {
                attack_type: {
                    'count': pattern['count'],
                    'success_rate': pattern['success_rate'],
                    'average_strength': pattern['average_strength'],
                    'last_seen': pattern['last_seen']
                }
                for attack_type, pattern in self.attack_patterns.items()
            },
            'defense_layers': len(self.defense_layers),
            'memory_usage': len(self.attack_history),
            'adaptation_config': {
                'adaptation_rate': self.config.adaptation_rate,
                'memory_size': self.config.memory_size,
                'update_frequency': self.config.update_frequency
            }
        }
        
    def _compute_confidence_adjustment(self, 
                                     input_data: torch.Tensor,
                                     base_result: Dict[str, Any],
                                     defense_responses: List[Dict]) -> float:
        """Compute confidence adjustment based on defense layer outputs."""
        adjustment = 0.0
        
        # Factor in defense layer responses
        for response in defense_responses:
            if response['result'] is not None:
                weight = response['weight']
                
                if isinstance(response['result'], dict):
                    # Extract confidence adjustment from layer result
                    layer_adjustment = response['result'].get('confidence_delta', 0.0)
                    adjustment += weight * layer_adjustment
                    
        # Factor in historical attack patterns
        input_risk = self._assess_input_risk(input_data)
        adjustment -= input_risk * 0.1  # Reduce confidence for risky inputs
        
        return np.clip(adjustment, -0.5, 0.5)  # Limit adjustment magnitude
        
    def _analyze_attack_indicators(self, 
                                 input_data: torch.Tensor,
                                 base_result: Dict[str, Any],
                                 defense_responses: List[Dict]) -> Dict[str, float]:
        """Analyze various indicators that suggest an attack."""
        indicators = {}
        
        # Statistical anomaly indicators
        input_stats = self._compute_sample_statistics(input_data.unsqueeze(0))
        indicators['input_magnitude_anomaly'] = min(input_stats['mean_magnitude'] / 5.0, 1.0)
        indicators['input_variance_anomaly'] = min(input_stats['std_magnitude'] / 2.0, 1.0)
        
        # Confidence drop indicator
        base_confidence = base_result.get('confidence', 0.0)
        historical_avg = self.verification_stats.get('average_confidence', 0.7)
        if historical_avg > 0:
            confidence_drop = max(0, historical_avg - base_confidence) / historical_avg
            indicators['confidence_drop'] = confidence_drop
        
        # Defense layer alerts
        layer_alerts = 0
        for response in defense_responses:
            if isinstance(response['result'], dict):
                if response['result'].get('alert', False):
                    layer_alerts += response['weight']
        indicators['defense_layer_alerts'] = min(layer_alerts, 1.0)
        
        # Pattern matching against known attacks
        pattern_match_score = 0.0
        for attack_type, pattern in self.attack_patterns.items():
            if pattern['count'] > 0:
                # Simple pattern matching based on input characteristics
                characteristics_match = self._match_input_characteristics(
                    input_data, pattern.get('characteristics_stats', {})
                )
                pattern_match_score = max(pattern_match_score, characteristics_match)
        
        indicators['known_pattern_match'] = pattern_match_score
        
        return indicators
        
    def _should_adapt(self, verification_result: Dict[str, Any]) -> bool:
        """Determine if adaptation should be triggered."""
        # Adapt if many attack indicators are present
        attack_indicators = verification_result.get('attack_indicators', {})
        total_indicator_score = sum(attack_indicators.values())
        
        if total_indicator_score > self.thresholds['adaptation_threshold']:
            return True
            
        # Adapt periodically based on update frequency
        self.adaptation_counter += 1
        if self.adaptation_counter >= self.config.update_frequency:
            self.adaptation_counter = 0
            return True
            
        return False
        
    def _trigger_adaptation(self, 
                          input_data: torch.Tensor,
                          verification_result: Dict[str, Any],
                          metadata: Optional[Dict]) -> None:
        """Trigger adaptation process based on current observations."""
        # Create synthetic attack observation
        attack_indicators = verification_result.get('attack_indicators', {})
        
        # Determine most likely attack type
        attack_type = 'unknown'
        max_indicator = 0.0
        for indicator_name, score in attack_indicators.items():
            if score > max_indicator:
                max_indicator = score
                if 'pattern_match' in indicator_name:
                    attack_type = 'pattern_based'
                elif 'magnitude' in indicator_name:
                    attack_type = 'perturbation'
                elif 'confidence' in indicator_name:
                    attack_type = 'evasion'
                    
        # Create observation record
        attack_observations = {
            'strength': max_indicator,
            'detection_confidence': sum(attack_indicators.values()) / len(attack_indicators),
            'characteristics': self._extract_input_characteristics(input_data),
            'defense_response': verification_result.get('defense_responses', []),
            'success': not verification_result.get('verified', True)
        }
        
        # Trigger adaptation
        self.adapt_to_attack(attack_type, attack_observations)
        
    def _determine_adaptation_response(self, 
                                     attack_type: str,
                                     observation: AttackObservation) -> List[Dict[str, Any]]:
        """Determine appropriate adaptations for observed attack."""
        adaptations = []
        
        # Threshold adjustments
        if observation.attack_strength > 0.7:
            adaptations.append({
                'type': 'threshold_adjustment',
                'target': 'confidence_threshold',
                'adjustment': 0.05,  # Increase threshold
                'reason': f'High strength {attack_type} attack'
            })
            
        if observation.success:
            adaptations.append({
                'type': 'threshold_adjustment', 
                'target': 'suspicion_threshold',
                'adjustment': -0.02,  # Lower suspicion threshold
                'reason': f'Successful {attack_type} attack'
            })
            
        # Defense layer additions
        pattern = self.attack_patterns.get(attack_type, {})
        if pattern.get('count', 0) > 3 and pattern.get('success_rate', 0) > 0.3:
            if attack_type == 'perturbation':
                adaptations.append({
                    'type': 'add_defense_layer',
                    'defense_type': 'input_filtering',
                    'parameters': {'noise_threshold': 0.1},
                    'reason': f'Repeated {attack_type} attacks'
                })
            elif attack_type == 'evasion':
                adaptations.append({
                    'type': 'add_defense_layer',
                    'defense_type': 'ensemble_verification',
                    'parameters': {'ensemble_size': 3},
                    'reason': f'Repeated {attack_type} attacks'
                })
                
        return adaptations
        
    def _apply_adaptation(self, adaptation: Dict[str, Any]) -> None:
        """Apply a specific adaptation."""
        adaptation_type = adaptation['type']
        
        if adaptation_type == 'threshold_adjustment':
            target = adaptation['target']
            adjustment = adaptation['adjustment']
            
            if target in self.thresholds:
                old_value = self.thresholds[target]
                new_value = old_value + adjustment
                
                # Respect bounds
                bounds = self.config.parameters.get('threshold_bounds', (0.1, 0.9))
                new_value = np.clip(new_value, bounds[0], bounds[1])
                
                self.thresholds[target] = new_value
                logger.info(f"Adjusted {target}: {old_value:.3f} -> {new_value:.3f}")
                
        elif adaptation_type == 'add_defense_layer':
            defense_type = adaptation['defense_type']
            parameters = adaptation.get('parameters', {})
            
            self.add_defense_layer(defense_type, **parameters)
            
    def _compute_sample_statistics(self, samples: torch.Tensor) -> Dict[str, float]:
        """Compute statistical characteristics of input samples."""
        with torch.no_grad():
            flattened = samples.flatten(1)  # Keep batch dimension
            
            stats = {
                'mean_magnitude': torch.mean(torch.norm(flattened, dim=1)).item(),
                'std_magnitude': torch.std(torch.norm(flattened, dim=1)).item(),
                'mean_value': torch.mean(flattened).item(),
                'std_value': torch.std(flattened).item(),
                'min_value': torch.min(flattened).item(),
                'max_value': torch.max(flattened).item(),
                'l1_norm': torch.mean(torch.sum(torch.abs(flattened), dim=1)).item(),
                'l2_norm': torch.mean(torch.norm(flattened, dim=1)).item(),
                'linf_norm': torch.mean(torch.norm(flattened, p=float('inf'), dim=1)).item()
            }
            
        return stats
        
    def _analyze_threshold_separation(self, 
                                    attack_stats: Dict[str, float],
                                    legit_stats: Dict[str, float]) -> Dict[str, float]:
        """Analyze optimal threshold separation between attack and legitimate samples."""
        separations = {}
        
        for stat_name in attack_stats:
            if stat_name in legit_stats:
                attack_val = attack_stats[stat_name]
                legit_val = legit_stats[stat_name]
                
                # Compute optimal threshold as midpoint with bias toward legitimate
                if stat_name in ['mean_magnitude', 'l1_norm', 'l2_norm', 'linf_norm']:
                    # For magnitude-based stats, higher values suggest attacks
                    optimal_threshold = (attack_val + legit_val) / 2.0
                    # Bias slightly toward legitimate samples
                    optimal_threshold = optimal_threshold * 0.9 + legit_val * 0.1
                else:
                    # For other stats, use midpoint
                    optimal_threshold = (attack_val + legit_val) / 2.0
                    
                separations[f'{stat_name}_threshold'] = optimal_threshold
                
        return separations
        
    def _update_thresholds_from_attacks(self, attack_stats: Dict[str, float]) -> None:
        """Update thresholds based on attack samples only."""
        # Conservative updates when we only have attack samples
        for stat_name, value in attack_stats.items():
            threshold_name = f'{stat_name}_threshold'
            
            if threshold_name not in self.thresholds:
                # Initialize threshold conservatively
                self.thresholds[threshold_name] = value * 0.5
            else:
                # Move threshold away from attack characteristics
                current = self.thresholds[threshold_name]
                # Small adjustment away from attack value
                adjustment = (value - current) * -0.05
                new_threshold = current + adjustment
                
                bounds = self.config.parameters.get('threshold_bounds', (0.1, 0.9))
                new_threshold = np.clip(new_threshold, bounds[0], bounds[1])
                
                self.thresholds[threshold_name] = new_threshold
                
    def _assess_input_risk(self, input_data: torch.Tensor) -> float:
        """Assess risk level of input based on historical patterns."""
        risk_score = 0.0
        
        # Check against known attack patterns
        for attack_type, pattern in self.attack_patterns.items():
            if pattern['count'] > 0:
                characteristics_match = self._match_input_characteristics(
                    input_data, pattern.get('characteristics_stats', {})
                )
                pattern_risk = characteristics_match * pattern['success_rate']
                risk_score = max(risk_score, pattern_risk)
                
        return min(risk_score, 1.0)
        
    def _match_input_characteristics(self, 
                                   input_data: torch.Tensor,
                                   pattern_stats: Dict[str, float]) -> float:
        """Match input characteristics against known pattern statistics."""
        if not pattern_stats:
            return 0.0
            
        input_stats = self._compute_sample_statistics(input_data.unsqueeze(0))
        
        match_score = 0.0
        count = 0
        
        for stat_name, pattern_value in pattern_stats.items():
            if stat_name in input_stats:
                input_value = input_stats[stat_name]
                
                # Compute similarity (inverse of normalized difference)
                if pattern_value != 0:
                    diff = abs(input_value - pattern_value) / abs(pattern_value)
                    similarity = max(0, 1 - diff)
                else:
                    similarity = 1.0 if input_value == 0 else 0.0
                    
                match_score += similarity
                count += 1
                
        return match_score / max(count, 1)
        
    def _extract_input_characteristics(self, input_data: torch.Tensor) -> Dict[str, float]:
        """Extract characteristics from input for pattern learning."""
        return self._compute_sample_statistics(input_data.unsqueeze(0))
        
    def _update_verification_stats(self, result: Dict[str, Any]) -> None:
        """Update running verification statistics."""
        self.verification_stats['total_verifications'] += 1
        
        # Update average confidence
        confidence = result.get('adapted_confidence', 0.0)
        total = self.verification_stats['total_verifications']
        current_avg = self.verification_stats['average_confidence']
        
        # Exponential moving average
        alpha = 1.0 / min(total, 100)  # Adapt based on sample size
        new_avg = alpha * confidence + (1 - alpha) * current_avg
        self.verification_stats['average_confidence'] = new_avg
        
        # Count attack detections
        attack_indicators = result.get('attack_indicators', {})
        if sum(attack_indicators.values()) > self.thresholds['adaptation_threshold']:
            self.verification_stats['attacks_detected'] += 1
            
    def _create_input_filter(self, **kwargs) -> Callable:
        """Create input filtering defense layer."""
        noise_threshold = kwargs.get('noise_threshold', 0.1)
        
        def filter_layer(input_data: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
            # Simple noise filtering
            input_stats = self._compute_sample_statistics(input_data.unsqueeze(0))
            
            alert = input_stats['std_value'] > noise_threshold
            
            if alert:
                # Apply simple denoising
                filtered_input = input_data * 0.95  # Slight dampening
            else:
                filtered_input = input_data
                
            return {
                'filtered_input': filtered_input,
                'alert': alert,
                'confidence_delta': -0.1 if alert else 0.0,
                'noise_level': input_stats['std_value']
            }
            
        return filter_layer
        
    def _create_noise_injector(self, **kwargs) -> Callable:
        """Create noise injection defense layer."""
        noise_level = kwargs.get('noise_level', 0.01)
        
        def noise_layer(input_data: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
            # Add small amount of random noise
            noise = torch.randn_like(input_data) * noise_level
            noisy_input = input_data + noise
            
            return {
                'filtered_input': noisy_input,
                'alert': False,
                'confidence_delta': 0.02,  # Slight confidence boost
                'noise_added': noise_level
            }
            
        return noise_layer
        
    def _create_gradient_masker(self, **kwargs) -> Callable:
        """Create gradient masking defense layer."""
        def gradient_mask_layer(input_data: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
            # This would implement gradient masking in a real scenario
            # For now, just return the input unchanged
            return {
                'filtered_input': input_data,
                'alert': False,
                'confidence_delta': 0.0,
                'masking_applied': True
            }
            
        return gradient_mask_layer
        
    def _create_ensemble_verifier(self, **kwargs) -> Callable:
        """Create ensemble verification defense layer."""
        ensemble_size = kwargs.get('ensemble_size', 3)
        
        def ensemble_layer(input_data: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
            # Simulate ensemble verification
            confidences = []
            
            for i in range(ensemble_size):
                # Add small random perturbations for ensemble diversity
                perturbed_input = input_data + torch.randn_like(input_data) * 0.001
                
                # Mock ensemble member verification
                result = self.base_verifier.verify(perturbed_input, model)
                confidences.append(result.get('confidence', 0.0))
                
            ensemble_confidence = np.mean(confidences)
            ensemble_std = np.std(confidences)
            
            # High variance suggests potential attack
            alert = ensemble_std > 0.1
            
            return {
                'filtered_input': input_data,
                'alert': alert,
                'confidence_delta': 0.05 if not alert else -0.05,
                'ensemble_confidence': ensemble_confidence,
                'ensemble_variance': ensemble_std
            }
            
        return ensemble_layer


class InputFilter:
    """
    Advanced input filtering system for adversarial detection and sanitization.
    
    This class implements multiple detection methods and sanitization techniques
    to identify and neutralize adversarial inputs before verification.
    """
    
    def __init__(self, config: Optional[DefenseConfig] = None):
        """
        Initialize input filter.
        
        Args:
            config: Configuration for filtering behavior
        """
        self.config = config or DefenseConfig(
            defense_type='input_filtering',
            parameters={
                'detection_methods': ['statistical', 'isolation_forest', 'one_class_svm'],
                'sanitization_methods': ['gaussian_blur', 'median_filter', 'total_variation'],
                'detection_threshold': 0.5,
                'sanitization_strength': 0.1
            }
        )
        
        # Detection models
        self.detection_models = {}
        self.detection_stats = {}
        self.calibrated = False
        
        # Sanitization parameters
        self.sanitization_params = self.config.parameters.get('sanitization_methods', [])
        
        # Statistics tracking
        self.filter_stats = {
            'total_processed': 0,
            'adversarial_detected': 0,
            'inputs_sanitized': 0,
            'false_positive_rate': 0.0,
            'detection_accuracy': 0.0
        }
        
    def calibrate(self, 
                 clean_samples: torch.Tensor,
                 adversarial_samples: Optional[torch.Tensor] = None) -> None:
        """
        Calibrate detection models with clean and adversarial samples.
        
        Args:
            clean_samples: Known clean inputs for calibration
            adversarial_samples: Known adversarial inputs (if available)
        """
        logger.info("Calibrating input filter detection models")
        
        if not HAS_SKLEARN:
            logger.warning("Scikit-learn not available, using statistical methods only")
            self._calibrate_statistical_only(clean_samples, adversarial_samples)
            return
            
        # Extract features from clean samples
        clean_features = self._extract_features(clean_samples)
        
        # Calibrate statistical detector
        self._calibrate_statistical_detector(clean_features, adversarial_samples)
        
        # Calibrate isolation forest
        self._calibrate_isolation_forest(clean_features)
        
        # Calibrate one-class SVM if we have enough samples
        if len(clean_features) >= 50:
            self._calibrate_one_class_svm(clean_features)
            
        # If we have adversarial samples, evaluate detection performance
        if adversarial_samples is not None:
            self._evaluate_detection_performance(clean_samples, adversarial_samples)
            
        self.calibrated = True
        logger.info("Input filter calibration completed")
        
    def detect_adversarial(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        Detect adversarial inputs using multiple detection methods.
        
        Args:
            input_data: Input tensor to analyze
            
        Returns:
            Detection results with confidence scores and method details
        """
        if not self.calibrated:
            logger.warning("Filter not calibrated, using basic detection")
            return self._basic_detection(input_data)
            
        # Extract features
        features = self._extract_features(input_data.unsqueeze(0))
        
        detection_results = {}
        detection_scores = []
        
        # Statistical detection
        if 'statistical' in self.config.parameters.get('detection_methods', []):
            stat_result = self._statistical_detection(features[0])
            detection_results['statistical'] = stat_result
            detection_scores.append(stat_result['score'])
            
        # Isolation forest detection
        if 'isolation_forest' in self.detection_models:
            iso_result = self._isolation_forest_detection(features[0])
            detection_results['isolation_forest'] = iso_result
            detection_scores.append(iso_result['score'])
            
        # One-class SVM detection
        if 'one_class_svm' in self.detection_models:
            svm_result = self._one_class_svm_detection(features[0])
            detection_results['one_class_svm'] = svm_result
            detection_scores.append(svm_result['score'])
            
        # Combine detection scores
        if detection_scores:
            combined_score = np.mean(detection_scores)
            max_score = np.max(detection_scores)
            consensus = np.std(detection_scores)  # Low std means consensus
        else:
            combined_score = 0.0
            max_score = 0.0
            consensus = 1.0
            
        threshold = self.config.parameters.get('detection_threshold', 0.5)
        is_adversarial = combined_score > threshold
        
        # Update statistics
        self.filter_stats['total_processed'] += 1
        if is_adversarial:
            self.filter_stats['adversarial_detected'] += 1
            
        return {
            'is_adversarial': is_adversarial,
            'combined_score': combined_score,
            'max_score': max_score,
            'consensus': 1.0 - consensus,  # High consensus = low std
            'detection_results': detection_results,
            'threshold_used': threshold,
            'confidence': max_score if is_adversarial else (1.0 - max_score)
        }
        
    def sanitize_input(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        Remove potential adversarial perturbations from input.
        
        Args:
            input_data: Input tensor to sanitize
            
        Returns:
            Sanitized input and sanitization metadata
        """
        sanitized = input_data.clone()
        sanitization_applied = []
        
        sanitization_strength = self.config.parameters.get('sanitization_strength', 0.1)
        
        # Apply different sanitization methods
        if 'gaussian_blur' in self.sanitization_params:
            sanitized = self._apply_gaussian_blur(sanitized, strength=sanitization_strength)
            sanitization_applied.append('gaussian_blur')
            
        if 'median_filter' in self.sanitization_params:
            sanitized = self._apply_median_filter(sanitized, strength=sanitization_strength)
            sanitization_applied.append('median_filter')
            
        if 'total_variation' in self.sanitization_params:
            sanitized = self._apply_total_variation_denoising(sanitized, strength=sanitization_strength)
            sanitization_applied.append('total_variation')
            
        if 'quantization' in self.sanitization_params:
            sanitized = self._apply_quantization(sanitized, strength=sanitization_strength)
            sanitization_applied.append('quantization')
            
        # Compute sanitization metrics
        l2_change = torch.norm(sanitized - input_data).item()
        linf_change = torch.norm(sanitized - input_data, p=float('inf')).item()
        
        # Update statistics
        if sanitization_applied:
            self.filter_stats['inputs_sanitized'] += 1
            
        return {
            'sanitized_input': sanitized,
            'original_input': input_data,
            'methods_applied': sanitization_applied,
            'l2_change': l2_change,
            'linf_change': linf_change,
            'sanitization_strength': sanitization_strength
        }
        
    def validate_input_distribution(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        Check if input follows expected distribution patterns.
        
        Args:
            input_data: Input tensor to validate
            
        Returns:
            Distribution validation results
        """
        # Extract statistical properties
        flattened = input_data.flatten()
        
        # Basic distribution tests
        validation_results = {
            'mean': torch.mean(flattened).item(),
            'std': torch.std(flattened).item(),
            'min': torch.min(flattened).item(),
            'max': torch.max(flattened).item(),
            'skewness': self._compute_skewness(flattened),
            'kurtosis': self._compute_kurtosis(flattened)
        }
        
        # Check against expected ranges (assuming normalized inputs)
        expected_ranges = {
            'mean': (-0.5, 0.5),
            'std': (0.0, 2.0),
            'min': (-3.0, 1.0),
            'max': (0.0, 4.0),
            'skewness': (-2.0, 2.0),
            'kurtosis': (1.0, 10.0)
        }
        
        violations = []
        for metric, value in validation_results.items():
            if metric in expected_ranges:
                min_val, max_val = expected_ranges[metric]
                if not (min_val <= value <= max_val):
                    violations.append({
                        'metric': metric,
                        'value': value,
                        'expected_range': (min_val, max_val),
                        'severity': abs(value - np.mean([min_val, max_val])) / (max_val - min_val)
                    })
                    
        # Overall validation score
        total_severity = sum(v['severity'] for v in violations)
        validation_score = max(0.0, 1.0 - total_severity / len(expected_ranges))
        
        is_valid = len(violations) == 0 or validation_score > 0.7
        
        return {
            'is_valid': is_valid,
            'validation_score': validation_score,
            'distribution_stats': validation_results,
            'violations': violations,
            'num_violations': len(violations)
        }
        
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get comprehensive filtering statistics."""
        stats = self.filter_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['detection_rate'] = stats['adversarial_detected'] / stats['total_processed']
            stats['sanitization_rate'] = stats['inputs_sanitized'] / stats['total_processed']
        else:
            stats['detection_rate'] = 0.0
            stats['sanitization_rate'] = 0.0
            
        stats['calibrated'] = self.calibrated
        stats['detection_models'] = list(self.detection_models.keys())
        stats['config'] = self.config.parameters
        
        return stats
        
    def _extract_features(self, input_batch: torch.Tensor) -> np.ndarray:
        """Extract relevant features from input batch for detection."""
        batch_size = input_batch.size(0)
        features_list = []
        
        for i in range(batch_size):
            input_tensor = input_batch[i]
            flattened = input_tensor.flatten()
            
            # Statistical features
            features = [
                torch.mean(flattened).item(),
                torch.std(flattened).item(),
                torch.min(flattened).item(),
                torch.max(flattened).item(),
                torch.norm(flattened, p=1).item(),
                torch.norm(flattened, p=2).item(),
                torch.norm(flattened, p=float('inf')).item(),
                self._compute_skewness(flattened),
                self._compute_kurtosis(flattened)
            ]
            
            # High-frequency component analysis (simple approximation)
            if len(input_tensor.shape) >= 2:
                # Compute gradients as proxy for high-frequency content
                grad_x = torch.diff(input_tensor, dim=-1)
                grad_y = torch.diff(input_tensor, dim=-2)
                
                features.extend([
                    torch.mean(torch.abs(grad_x)).item(),
                    torch.std(grad_x.flatten()).item(),
                    torch.mean(torch.abs(grad_y)).item(),
                    torch.std(grad_y.flatten()).item()
                ])
                
            features_list.append(features)
            
        return np.array(features_list)
        
    def _calibrate_statistical_only(self, 
                                  clean_samples: torch.Tensor,
                                  adversarial_samples: Optional[torch.Tensor]) -> None:
        """Calibrate using statistical methods only when sklearn is unavailable."""
        clean_features = self._extract_features(clean_samples)
        
        # Compute statistics for clean samples
        self.detection_stats['clean_mean'] = np.mean(clean_features, axis=0)
        self.detection_stats['clean_std'] = np.std(clean_features, axis=0)
        self.detection_stats['clean_min'] = np.min(clean_features, axis=0)
        self.detection_stats['clean_max'] = np.max(clean_features, axis=0)
        
        if adversarial_samples is not None:
            adv_features = self._extract_features(adversarial_samples)
            self.detection_stats['adv_mean'] = np.mean(adv_features, axis=0)
            self.detection_stats['adv_std'] = np.std(adv_features, axis=0)
            
    def _calibrate_statistical_detector(self, 
                                      clean_features: np.ndarray,
                                      adversarial_samples: Optional[torch.Tensor]) -> None:
        """Calibrate statistical anomaly detector."""
        self.detection_stats['clean_mean'] = np.mean(clean_features, axis=0)
        self.detection_stats['clean_std'] = np.std(clean_features, axis=0)
        self.detection_stats['clean_cov'] = np.cov(clean_features.T)
        
        if adversarial_samples is not None:
            adv_features = self._extract_features(adversarial_samples)
            self.detection_stats['adv_mean'] = np.mean(adv_features, axis=0)
            self.detection_stats['adv_std'] = np.std(adv_features, axis=0)
            
    def _calibrate_isolation_forest(self, clean_features: np.ndarray) -> None:
        """Calibrate isolation forest detector."""
        if not HAS_SKLEARN:
            return
            
        try:
            iso_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            iso_forest.fit(clean_features)
            self.detection_models['isolation_forest'] = iso_forest
            logger.info("Isolation forest calibrated successfully")
        except Exception as e:
            logger.warning(f"Failed to calibrate isolation forest: {e}")
            
    def _calibrate_one_class_svm(self, clean_features: np.ndarray) -> None:
        """Calibrate one-class SVM detector."""
        if not HAS_SKLEARN:
            return
            
        try:
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(clean_features)
            
            svm = OneClassSVM(
                kernel='rbf',
                gamma='scale',
                nu=0.1
            )
            svm.fit(scaled_features)
            
            self.detection_models['one_class_svm'] = svm
            self.detection_models['svm_scaler'] = scaler
            logger.info("One-class SVM calibrated successfully")
        except Exception as e:
            logger.warning(f"Failed to calibrate one-class SVM: {e}")
            
    def _basic_detection(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """Basic detection when not calibrated."""
        # Simple statistical checks
        flattened = input_data.flatten()
        
        # Check for extreme values
        mean_val = torch.mean(flattened).item()
        std_val = torch.std(flattened).item()
        max_val = torch.max(torch.abs(flattened)).item()
        
        # Simple heuristics
        extreme_mean = abs(mean_val) > 2.0
        extreme_std = std_val > 3.0
        extreme_max = max_val > 5.0
        
        anomaly_score = sum([extreme_mean, extreme_std, extreme_max]) / 3.0
        
        return {
            'is_adversarial': anomaly_score > 0.5,
            'combined_score': anomaly_score,
            'max_score': anomaly_score,
            'consensus': 1.0,
            'detection_results': {
                'basic': {
                    'score': anomaly_score,
                    'extreme_mean': extreme_mean,
                    'extreme_std': extreme_std,
                    'extreme_max': extreme_max
                }
            },
            'threshold_used': 0.5,
            'confidence': anomaly_score
        }
        
    def _statistical_detection(self, features: np.ndarray) -> Dict[str, Any]:
        """Statistical anomaly detection."""
        if 'clean_mean' not in self.detection_stats:
            return {'score': 0.0, 'method': 'statistical', 'error': 'Not calibrated'}
            
        clean_mean = self.detection_stats['clean_mean']
        clean_std = self.detection_stats['clean_std']
        
        # Compute z-scores
        z_scores = np.abs((features - clean_mean) / (clean_std + 1e-8))
        
        # Anomaly score based on max z-score
        max_z_score = np.max(z_scores)
        anomaly_score = min(max_z_score / 3.0, 1.0)  # Normalize by 3-sigma rule
        
        return {
            'score': anomaly_score,
            'method': 'statistical',
            'max_z_score': max_z_score,
            'mean_z_score': np.mean(z_scores)
        }
        
    def _isolation_forest_detection(self, features: np.ndarray) -> Dict[str, Any]:
        """Isolation forest based detection."""
        if 'isolation_forest' not in self.detection_models:
            return {'score': 0.0, 'method': 'isolation_forest', 'error': 'Model not available'}
            
        try:
            iso_forest = self.detection_models['isolation_forest']
            
            # Get anomaly score
            anomaly_score = iso_forest.decision_function([features])[0]
            
            # Convert to 0-1 range (more negative = more anomalous)
            normalized_score = max(0.0, -anomaly_score)
            
            return {
                'score': min(normalized_score, 1.0),
                'method': 'isolation_forest',
                'raw_score': anomaly_score
            }
        except Exception as e:
            return {'score': 0.0, 'method': 'isolation_forest', 'error': str(e)}
            
    def _one_class_svm_detection(self, features: np.ndarray) -> Dict[str, Any]:
        """One-class SVM based detection."""
        if 'one_class_svm' not in self.detection_models:
            return {'score': 0.0, 'method': 'one_class_svm', 'error': 'Model not available'}
            
        try:
            svm = self.detection_models['one_class_svm']
            scaler = self.detection_models['svm_scaler']
            
            # Scale features
            scaled_features = scaler.transform([features])
            
            # Get decision function value
            decision_value = svm.decision_function(scaled_features)[0]
            
            # Convert to anomaly score (negative values indicate anomalies)
            anomaly_score = max(0.0, -decision_value)
            
            return {
                'score': min(anomaly_score, 1.0),
                'method': 'one_class_svm',
                'decision_value': decision_value
            }
        except Exception as e:
            return {'score': 0.0, 'method': 'one_class_svm', 'error': str(e)}
            
    def _evaluate_detection_performance(self, 
                                      clean_samples: torch.Tensor,
                                      adversarial_samples: torch.Tensor) -> None:
        """Evaluate detection performance on labeled data."""
        # Test on clean samples
        clean_results = []
        for i in range(min(len(clean_samples), 100)):  # Limit for efficiency
            result = self.detect_adversarial(clean_samples[i])
            clean_results.append(result['is_adversarial'])
            
        # Test on adversarial samples
        adv_results = []
        for i in range(min(len(adversarial_samples), 100)):
            result = self.detect_adversarial(adversarial_samples[i])
            adv_results.append(result['is_adversarial'])
            
        # Compute metrics
        true_negatives = sum(1 - r for r in clean_results)
        false_positives = sum(clean_results)
        true_positives = sum(adv_results)
        false_negatives = sum(1 - r for r in adv_results)
        
        total_clean = len(clean_results)
        total_adv = len(adv_results)
        
        self.filter_stats['false_positive_rate'] = false_positives / max(total_clean, 1)
        self.filter_stats['detection_accuracy'] = (true_positives + true_negatives) / max(total_clean + total_adv, 1)
        
        logger.info(f"Detection performance: FPR={self.filter_stats['false_positive_rate']:.3f}, "
                   f"Accuracy={self.filter_stats['detection_accuracy']:.3f}")
        
    def _apply_gaussian_blur(self, input_tensor: torch.Tensor, strength: float) -> torch.Tensor:
        """Apply Gaussian blur for smoothing."""
        # Simple approximation of Gaussian blur using convolution
        if len(input_tensor.shape) < 3:
            return input_tensor
            
        # Create simple blur kernel
        kernel_size = max(3, int(strength * 10))
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # For simplicity, apply averaging filter
        with torch.no_grad():
            # Pad input
            pad_size = kernel_size // 2
            padded = F.pad(input_tensor, [pad_size] * 4, mode='reflect')
            
            # Apply averaging
            kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
            
            if len(input_tensor.shape) == 3:
                channels = input_tensor.size(0)
                kernel = kernel.repeat(channels, 1, 1, 1)
                padded = padded.unsqueeze(0)
                blurred = F.conv2d(padded, kernel, groups=channels, padding=0)
                return blurred.squeeze(0)
            else:
                return F.conv2d(padded.unsqueeze(0).unsqueeze(0), kernel, padding=0).squeeze()
        
    def _apply_median_filter(self, input_tensor: torch.Tensor, strength: float) -> torch.Tensor:
        """Apply median filtering."""
        # Simple approximation using percentile
        if len(input_tensor.shape) < 2:
            return input_tensor
            
        kernel_size = max(3, int(strength * 8))
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # For simplicity, return slightly smoothed version
        smoothed = input_tensor * 0.8 + torch.mean(input_tensor) * 0.2
        return smoothed
        
    def _apply_total_variation_denoising(self, input_tensor: torch.Tensor, strength: float) -> torch.Tensor:
        """Apply total variation denoising."""
        if len(input_tensor.shape) < 2:
            return input_tensor
            
        # Simple TV denoising approximation
        lambda_tv = strength * 0.1
        
        with torch.no_grad():
            # Compute gradients
            if len(input_tensor.shape) == 3:
                grad_x = torch.diff(input_tensor, dim=2, prepend=input_tensor[:, :, :1])
                grad_y = torch.diff(input_tensor, dim=1, prepend=input_tensor[:, :1, :])
            else:
                grad_x = torch.diff(input_tensor, dim=1, prepend=input_tensor[:, :1])
                grad_y = torch.diff(input_tensor, dim=0, prepend=input_tensor[:1, :])
            
            # Apply TV regularization (simplified)
            tv_loss = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))
            denoised = input_tensor - lambda_tv * torch.sign(grad_x + grad_y)
            
        return denoised
        
    def _apply_quantization(self, input_tensor: torch.Tensor, strength: float) -> torch.Tensor:
        """Apply quantization to reduce precision."""
        # Quantize to reduce precision
        num_levels = max(16, int(256 * (1 - strength)))
        
        # Quantize and dequantize
        quantized = torch.round(input_tensor * num_levels) / num_levels
        return quantized
        
    def _compute_skewness(self, tensor: torch.Tensor) -> float:
        """Compute skewness of tensor values."""
        with torch.no_grad():
            mean = torch.mean(tensor)
            std = torch.std(tensor)
            
            if std == 0:
                return 0.0
                
            normalized = (tensor - mean) / std
            skewness = torch.mean(normalized ** 3)
            
        return skewness.item()
        
    def _compute_kurtosis(self, tensor: torch.Tensor) -> float:
        """Compute kurtosis of tensor values."""
        with torch.no_grad():
            mean = torch.mean(tensor)
            std = torch.std(tensor)
            
            if std == 0:
                return 0.0
                
            normalized = (tensor - mean) / std
            kurtosis = torch.mean(normalized ** 4)
            
        return kurtosis.item()


class RandomizedDefense:
    """
    Randomized defense mechanisms including smoothing and stochastic verification.
    
    This class implements probabilistic defenses that use randomness to make
    attacks more difficult and improve robustness.
    """
    
    def __init__(self, config: Optional[DefenseConfig] = None):
        """
        Initialize randomized defense system.
        
        Args:
            config: Configuration for randomized defenses
        """
        self.config = config or DefenseConfig(
            defense_type='randomized_defense',
            parameters={
                'smoothing_noise_levels': [0.05, 0.1, 0.15],
                'verification_trials': [5, 10, 20],
                'consensus_threshold': 0.7,
                'confidence_threshold': 0.8
            }
        )
        
        # Defense statistics
        self.defense_stats = {
            'smoothing_applications': 0,
            'stochastic_verifications': 0,
            'consensus_achieved': 0,
            'average_noise_level': 0.0,
            'average_trials': 0.0
        }
        
    def random_smoothing(self, 
                        model: nn.Module,
                        input_data: torch.Tensor,
                        noise_level: float = 0.1,
                        n_samples: int = 100) -> Dict[str, Any]:
        """
        Apply randomized smoothing defense.
        
        Args:
            model: Model to apply smoothing to
            input_data: Input data to smooth
            noise_level: Standard deviation of Gaussian noise
            n_samples: Number of noise samples for smoothing
            
        Returns:
            Smoothed prediction results and confidence metrics
        """
        if not isinstance(input_data, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
            
        model.eval()
        batch_size = input_data.size(0) if len(input_data.shape) > 3 else 1
        
        if len(input_data.shape) == 3:  # Single image
            input_data = input_data.unsqueeze(0)
            
        smoothed_results = []
        
        with torch.no_grad():
            for batch_idx in range(batch_size):
                single_input = input_data[batch_idx:batch_idx+1]
                
                # Generate noisy samples
                predictions = []
                confidence_scores = []
                
                for _ in range(n_samples):
                    # Add Gaussian noise
                    noise = torch.randn_like(single_input) * noise_level
                    noisy_input = single_input + noise
                    
                    # Clamp to valid range (assuming [0, 1])
                    noisy_input = torch.clamp(noisy_input, 0, 1)
                    
                    try:
                        # Get model prediction
                        output = model(noisy_input)
                        
                        if len(output.shape) > 1:
                            # Classification case
                            probs = F.softmax(output, dim=1)
                            pred_class = torch.argmax(probs, dim=1).item()
                            confidence = torch.max(probs, dim=1)[0].item()
                        else:
                            # Regression case
                            pred_class = output.item()
                            confidence = 1.0 / (1.0 + torch.abs(output).item())  # Simple confidence
                            
                        predictions.append(pred_class)
                        confidence_scores.append(confidence)
                        
                    except Exception as e:
                        logger.warning(f"Error in smoothing sample: {e}")
                        continue
                        
                # Analyze predictions
                if predictions:
                    smoothed_result = self._analyze_smoothed_predictions(
                        predictions, confidence_scores, noise_level, n_samples
                    )
                else:
                    smoothed_result = {
                        'prediction': None,
                        'confidence': 0.0,
                        'consensus': 0.0,
                        'error': 'No valid predictions obtained'
                    }
                    
                smoothed_results.append(smoothed_result)
                
        # Update statistics
        self.defense_stats['smoothing_applications'] += 1
        self.defense_stats['average_noise_level'] = (
            (self.defense_stats['average_noise_level'] * (self.defense_stats['smoothing_applications'] - 1) + 
             noise_level) / self.defense_stats['smoothing_applications']
        )
        
        return {
            'smoothed_results': smoothed_results,
            'noise_level': noise_level,
            'n_samples': n_samples,
            'batch_size': batch_size,
            'overall_confidence': np.mean([r.get('confidence', 0.0) for r in smoothed_results]),
            'overall_consensus': np.mean([r.get('consensus', 0.0) for r in smoothed_results])
        }
        
    def stochastic_verification(self, 
                              verifier: BaseVerifier,
                              input_data: torch.Tensor,
                              model: nn.Module,
                              n_trials: int = 10) -> Dict[str, Any]:
        """
        Randomized verification with majority voting.
        
        Args:
            verifier: Base verifier to use
            input_data: Input to verify
            model: Model being verified
            n_trials: Number of verification trials
            
        Returns:
            Stochastic verification results with consensus analysis
        """
        verification_results = []
        verification_confidences = []
        verification_decisions = []
        
        for trial in range(n_trials):
            # Add small random perturbation for diversity
            noise_level = 0.001  # Very small noise
            noise = torch.randn_like(input_data) * noise_level
            perturbed_input = input_data + noise
            
            try:
                # Perform verification
                result = verifier.verify(perturbed_input, model)
                
                verification_results.append(result)
                
                # Extract decision and confidence
                decision = result.get('verified', False)
                confidence = verifier.compute_confidence(result)
                
                verification_decisions.append(decision)
                verification_confidences.append(confidence)
                
            except Exception as e:
                logger.warning(f"Verification trial {trial} failed: {e}")
                verification_decisions.append(False)
                verification_confidences.append(0.0)
                
        # Analyze consensus
        if verification_decisions:
            consensus_analysis = self._analyze_verification_consensus(
                verification_decisions, verification_confidences, n_trials
            )
        else:
            consensus_analysis = {
                'final_decision': False,
                'consensus_strength': 0.0,
                'confidence': 0.0,
                'error': 'No successful verifications'
            }
            
        # Update statistics
        self.defense_stats['stochastic_verifications'] += 1
        self.defense_stats['average_trials'] = (
            (self.defense_stats['average_trials'] * (self.defense_stats['stochastic_verifications'] - 1) + 
             n_trials) / self.defense_stats['stochastic_verifications']
        )
        
        if consensus_analysis['consensus_strength'] > self.config.parameters.get('consensus_threshold', 0.7):
            self.defense_stats['consensus_achieved'] += 1
            
        return {
            'final_decision': consensus_analysis['final_decision'],
            'consensus_strength': consensus_analysis['consensus_strength'],
            'overall_confidence': consensus_analysis['confidence'],
            'n_trials': n_trials,
            'successful_trials': len(verification_decisions),
            'individual_results': verification_results,
            'decision_distribution': {
                'verified': sum(verification_decisions),
                'rejected': len(verification_decisions) - sum(verification_decisions)
            },
            'confidence_stats': {
                'mean': np.mean(verification_confidences) if verification_confidences else 0.0,
                'std': np.std(verification_confidences) if verification_confidences else 0.0,
                'min': np.min(verification_confidences) if verification_confidences else 0.0,
                'max': np.max(verification_confidences) if verification_confidences else 0.0
            }
        }
        
    def adaptive_smoothing(self, 
                          model: nn.Module,
                          input_data: torch.Tensor,
                          threat_level: float = 0.5) -> Dict[str, Any]:
        """
        Apply adaptive smoothing based on perceived threat level.
        
        Args:
            model: Model to apply smoothing to
            input_data: Input data
            threat_level: Perceived threat level (0-1)
            
        Returns:
            Adaptive smoothing results
        """
        # Adjust parameters based on threat level
        noise_levels = self.config.parameters.get('smoothing_noise_levels', [0.05, 0.1, 0.15])
        base_samples = 50
        
        # Higher threat = more noise and samples
        noise_level = noise_levels[min(int(threat_level * len(noise_levels)), len(noise_levels) - 1)]
        n_samples = int(base_samples * (1 + threat_level))
        
        logger.info(f"Adaptive smoothing: threat={threat_level:.2f}, noise={noise_level:.3f}, samples={n_samples}")
        
        # Apply smoothing
        smoothing_result = self.random_smoothing(
            model=model,
            input_data=input_data,
            noise_level=noise_level,
            n_samples=n_samples
        )
        
        # Add adaptation metadata
        smoothing_result['adaptation_metadata'] = {
            'threat_level': threat_level,
            'adapted_noise_level': noise_level,
            'adapted_samples': n_samples,
            'adaptation_reason': 'threat_level_based'
        }
        
        return smoothing_result
        
    def ensemble_randomized_verification(self, 
                                       verifiers: List[BaseVerifier],
                                       input_data: torch.Tensor,
                                       model: nn.Module) -> Dict[str, Any]:
        """
        Combine multiple verifiers with randomized trials.
        
        Args:
            verifiers: List of verifiers to ensemble
            input_data: Input to verify
            model: Model being verified
            
        Returns:
            Ensemble verification results
        """
        if not verifiers:
            raise ValueError("At least one verifier must be provided")
            
        ensemble_results = {}
        verifier_decisions = []
        verifier_confidences = []
        
        for i, verifier in enumerate(verifiers):
            verifier_name = f"verifier_{i}"
            
            # Run stochastic verification for each verifier
            trials = self.config.parameters.get('verification_trials', [5, 10, 20])
            n_trials = trials[min(i, len(trials) - 1)]
            
            stochastic_result = self.stochastic_verification(
                verifier=verifier,
                input_data=input_data,
                model=model,
                n_trials=n_trials
            )
            
            ensemble_results[verifier_name] = stochastic_result
            
            verifier_decisions.append(stochastic_result['final_decision'])
            verifier_confidences.append(stochastic_result['overall_confidence'])
            
        # Ensemble decision making
        if verifier_decisions:
            # Majority vote
            positive_votes = sum(verifier_decisions)
            total_votes = len(verifier_decisions)
            
            # Weighted by confidence
            weighted_decision = sum(
                decision * confidence 
                for decision, confidence in zip(verifier_decisions, verifier_confidences)
            ) / sum(verifier_confidences) if sum(verifier_confidences) > 0 else 0.0
            
            ensemble_decision = positive_votes > total_votes / 2
            ensemble_confidence = np.mean(verifier_confidences)
            consensus_strength = max(positive_votes, total_votes - positive_votes) / total_votes
            
        else:
            ensemble_decision = False
            ensemble_confidence = 0.0
            consensus_strength = 0.0
            weighted_decision = 0.0
            
        return {
            'ensemble_decision': ensemble_decision,
            'ensemble_confidence': ensemble_confidence,
            'consensus_strength': consensus_strength,
            'weighted_decision': weighted_decision,
            'individual_verifiers': ensemble_results,
            'voting_summary': {
                'total_verifiers': len(verifiers),
                'positive_votes': sum(verifier_decisions),
                'negative_votes': len(verifier_decisions) - sum(verifier_decisions),
                'confidence_range': (min(verifier_confidences), max(verifier_confidences)) if verifier_confidences else (0, 0)
            }
        }
        
    def get_defense_statistics(self) -> Dict[str, Any]:
        """Get comprehensive randomized defense statistics."""
        stats = self.defense_stats.copy()
        
        # Compute derived statistics
        if stats['stochastic_verifications'] > 0:
            stats['consensus_rate'] = stats['consensus_achieved'] / stats['stochastic_verifications']
        else:
            stats['consensus_rate'] = 0.0
            
        stats['config'] = self.config.parameters
        
        return stats
        
    def _analyze_smoothed_predictions(self, 
                                    predictions: List,
                                    confidence_scores: List[float],
                                    noise_level: float,
                                    n_samples: int) -> Dict[str, Any]:
        """Analyze smoothed prediction results."""
        if not predictions:
            return {'prediction': None, 'confidence': 0.0, 'consensus': 0.0}
            
        # For classification
        if isinstance(predictions[0], (int, np.integer)):
            # Count votes for each class
            from collections import Counter
            vote_counts = Counter(predictions)
            
            # Most voted class
            predicted_class = vote_counts.most_common(1)[0][0]
            vote_fraction = vote_counts[predicted_class] / len(predictions)
            
            # Confidence based on voting consensus and individual confidences
            voting_confidence = vote_fraction
            avg_confidence = np.mean(confidence_scores)
            combined_confidence = 0.7 * voting_confidence + 0.3 * avg_confidence
            
            return {
                'prediction': predicted_class,
                'confidence': combined_confidence,
                'consensus': vote_fraction,
                'vote_distribution': dict(vote_counts),
                'noise_level': noise_level,
                'n_samples': n_samples
            }
            
        # For regression
        else:
            # Use mean and std of predictions
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            # Confidence inversely related to variance
            confidence = 1.0 / (1.0 + std_pred)
            
            # Consensus based on coefficient of variation
            if abs(mean_pred) > 1e-8:
                cv = std_pred / abs(mean_pred)
                consensus = max(0.0, 1.0 - cv)
            else:
                consensus = 1.0 if std_pred < 0.1 else 0.0
                
            return {
                'prediction': mean_pred,
                'confidence': confidence,
                'consensus': consensus,
                'prediction_std': std_pred,
                'prediction_range': (min(predictions), max(predictions)),
                'noise_level': noise_level,
                'n_samples': n_samples
            }
            
    def _analyze_verification_consensus(self, 
                                      decisions: List[bool],
                                      confidences: List[float],
                                      n_trials: int) -> Dict[str, Any]:
        """Analyze consensus from verification trials."""
        if not decisions:
            return {
                'final_decision': False,
                'consensus_strength': 0.0,
                'confidence': 0.0
            }
            
        # Count positive decisions
        positive_count = sum(decisions)
        negative_count = len(decisions) - positive_count
        
        # Majority vote
        final_decision = positive_count > negative_count
        
        # Consensus strength (how decisive the vote was)
        consensus_strength = max(positive_count, negative_count) / len(decisions)
        
        # Overall confidence (average of individual confidences)
        overall_confidence = np.mean(confidences)
        
        # Adjust confidence based on consensus
        # High consensus increases confidence, low consensus decreases it
        consensus_adjusted_confidence = overall_confidence * consensus_strength
        
        return {
            'final_decision': final_decision,
            'consensus_strength': consensus_strength,
            'confidence': consensus_adjusted_confidence,
            'raw_confidence': overall_confidence,
            'vote_counts': {
                'positive': positive_count,
                'negative': negative_count,
                'total': len(decisions)
            }
        }


# Integrated defense system
class IntegratedDefenseSystem:
    """
    Comprehensive defense system integrating all defense mechanisms.
    
    This class orchestrates adaptive verification, input filtering, and
    randomized defenses for comprehensive protection.
    """
    
    def __init__(self, 
                 base_verifier: BaseVerifier,
                 defense_configs: Optional[Dict[str, DefenseConfig]] = None):
        """
        Initialize integrated defense system.
        
        Args:
            base_verifier: Base verification system
            defense_configs: Configurations for each defense component
        """
        self.base_verifier = base_verifier
        
        # Initialize defense components
        adaptive_config = defense_configs.get('adaptive') if defense_configs else None
        filter_config = defense_configs.get('filter') if defense_configs else None
        random_config = defense_configs.get('random') if defense_configs else None
        
        self.adaptive_verifier = AdaptiveVerifier(base_verifier, adaptive_config)
        self.input_filter = InputFilter(filter_config)
        self.randomized_defense = RandomizedDefense(random_config)
        
        # System configuration
        self.defense_pipeline = ['filter', 'randomize', 'adaptive']
        self.threat_assessment_enabled = True
        
        # System statistics
        self.system_stats = {
            'total_inputs_processed': 0,
            'threats_detected': 0,
            'successful_defenses': 0,
            'pipeline_failures': 0
        }
        
    def comprehensive_defense(self, 
                            input_data: torch.Tensor,
                            model: nn.Module,
                            threat_level: Optional[float] = None) -> Dict[str, Any]:
        """
        Apply comprehensive defense pipeline.
        
        Args:
            input_data: Input to defend against
            model: Model being protected
            threat_level: Optional threat level assessment
            
        Returns:
            Comprehensive defense results
        """
        start_time = time.time()
        pipeline_results = {}
        
        # Assess threat level if not provided
        if threat_level is None and self.threat_assessment_enabled:
            threat_level = self._assess_threat_level(input_data)
        else:
            threat_level = threat_level or 0.5
            
        pipeline_results['threat_assessment'] = {
            'threat_level': threat_level,
            'assessment_method': 'integrated' if self.threat_assessment_enabled else 'provided'
        }
        
        current_input = input_data
        defense_confidence = 1.0
        
        try:
            # Stage 1: Input Filtering
            if 'filter' in self.defense_pipeline:
                filter_result = self._apply_input_filtering(current_input, threat_level)
                pipeline_results['input_filtering'] = filter_result
                
                # Use sanitized input if filtering detected threats
                if filter_result['detection']['is_adversarial']:
                    current_input = filter_result['sanitization']['sanitized_input']
                    defense_confidence *= 0.8  # Reduce confidence due to detected threat
                    
            # Stage 2: Randomized Defense
            if 'randomize' in self.defense_pipeline:
                random_result = self._apply_randomized_defense(current_input, model, threat_level)
                pipeline_results['randomized_defense'] = random_result
                
                defense_confidence *= random_result.get('overall_confidence', 1.0)
                
            # Stage 3: Adaptive Verification
            if 'adaptive' in self.defense_pipeline:
                adaptive_result = self._apply_adaptive_verification(current_input, model, pipeline_results)
                pipeline_results['adaptive_verification'] = adaptive_result
                
                defense_confidence *= adaptive_result.get('adapted_confidence', 1.0)
                
            # Determine final decision
            final_decision = self._make_final_decision(pipeline_results, defense_confidence, threat_level)
            
            # Update system statistics
            self._update_system_stats(final_decision, threat_level)
            
            return {
                'final_decision': final_decision,
                'defense_confidence': defense_confidence,
                'threat_level': threat_level,
                'pipeline_results': pipeline_results,
                'processing_time': time.time() - start_time,
                'defense_summary': self._generate_defense_summary(pipeline_results, final_decision)
            }
            
        except Exception as e:
            logger.error(f"Defense pipeline failed: {e}")
            self.system_stats['pipeline_failures'] += 1
            
            return {
                'final_decision': {'verified': False, 'confidence': 0.0},
                'defense_confidence': 0.0,
                'threat_level': threat_level,
                'pipeline_results': pipeline_results,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
            
    def train_defenses(self, 
                      clean_samples: torch.Tensor,
                      adversarial_samples: Optional[torch.Tensor] = None) -> None:
        """
        Train/calibrate all defense components.
        
        Args:
            clean_samples: Clean samples for calibration
            adversarial_samples: Known adversarial samples (optional)
        """
        logger.info("Training integrated defense system")
        
        # Calibrate input filter
        self.input_filter.calibrate(clean_samples, adversarial_samples)
        
        # Update adaptive verifier thresholds
        if adversarial_samples is not None:
            self.adaptive_verifier.update_thresholds(adversarial_samples, clean_samples)
            
        logger.info("Defense system training completed")
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'system_stats': self.system_stats.copy(),
            'adaptive_verifier_status': self.adaptive_verifier.get_adaptation_summary(),
            'input_filter_status': self.input_filter.get_filter_statistics(),
            'randomized_defense_status': self.randomized_defense.get_defense_statistics(),
            'pipeline_configuration': {
                'defense_pipeline': self.defense_pipeline,
                'threat_assessment_enabled': self.threat_assessment_enabled
            }
        }
        
    def _assess_threat_level(self, input_data: torch.Tensor) -> float:
        """Assess threat level of input."""
        # Simple threat assessment based on input characteristics
        input_stats = self._compute_input_statistics(input_data)
        
        # Normalize threat indicators
        magnitude_threat = min(input_stats['magnitude'] / 5.0, 1.0)
        variance_threat = min(input_stats['variance'] / 2.0, 1.0)
        extreme_value_threat = min(input_stats['max_abs'] / 3.0, 1.0)
        
        # Combine threat indicators
        threat_level = (magnitude_threat + variance_threat + extreme_value_threat) / 3.0
        
        return np.clip(threat_level, 0.0, 1.0)
        
    def _apply_input_filtering(self, input_data: torch.Tensor, threat_level: float) -> Dict[str, Any]:
        """Apply input filtering stage."""
        # Detect adversarial content
        detection_result = self.input_filter.detect_adversarial(input_data)
        
        # Apply sanitization if needed
        sanitization_result = None
        if detection_result['is_adversarial'] or threat_level > 0.6:
            sanitization_result = self.input_filter.sanitize_input(input_data)
            
        # Validate input distribution
        validation_result = self.input_filter.validate_input_distribution(input_data)
        
        return {
            'detection': detection_result,
            'sanitization': sanitization_result,
            'validation': validation_result,
            'filtering_applied': sanitization_result is not None
        }
        
    def _apply_randomized_defense(self, 
                                input_data: torch.Tensor, 
                                model: nn.Module,
                                threat_level: float) -> Dict[str, Any]:
        """Apply randomized defense stage."""
        # Use adaptive smoothing based on threat level
        smoothing_result = self.randomized_defense.adaptive_smoothing(
            model=model,
            input_data=input_data,
            threat_level=threat_level
        )
        
        return smoothing_result
        
    def _apply_adaptive_verification(self, 
                                   input_data: torch.Tensor,
                                   model: nn.Module,
                                   pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptive verification stage."""
        # Create metadata from pipeline results
        metadata = {
            'threat_level': pipeline_results.get('threat_assessment', {}).get('threat_level', 0.5),
            'filtering_applied': pipeline_results.get('input_filtering', {}).get('filtering_applied', False),
            'randomization_applied': 'randomized_defense' in pipeline_results
        }
        
        # Perform adaptive verification
        verification_result = self.adaptive_verifier.verify(
            input_data=input_data,
            model=model,
            metadata=metadata
        )
        
        return verification_result
        
    def _make_final_decision(self, 
                           pipeline_results: Dict[str, Any],
                           defense_confidence: float,
                           threat_level: float) -> Dict[str, Any]:
        """Make final verification decision based on all pipeline results."""
        # Extract key decisions
        filter_decision = pipeline_results.get('input_filtering', {}).get('detection', {}).get('is_adversarial', False)
        adaptive_decision = pipeline_results.get('adaptive_verification', {}).get('verified', False)
        random_confidence = pipeline_results.get('randomized_defense', {}).get('overall_confidence', 1.0)
        
        # Conservative decision making
        if filter_decision:  # Input filter detected adversarial content
            final_verified = False
            decision_confidence = 1.0 - defense_confidence
            decision_reason = "adversarial_content_detected"
        elif not adaptive_decision:  # Adaptive verifier rejected
            final_verified = False
            decision_confidence = pipeline_results.get('adaptive_verification', {}).get('adapted_confidence', 0.0)
            decision_reason = "adaptive_verification_failed"
        elif random_confidence < 0.5:  # Low randomized defense confidence
            final_verified = False
            decision_confidence = random_confidence
            decision_reason = "low_randomized_confidence"
        else:  # All checks passed
            final_verified = True
            decision_confidence = defense_confidence
            decision_reason = "all_defenses_passed"
            
        return {
            'verified': final_verified,
            'confidence': decision_confidence,
            'decision_reason': decision_reason,
            'threat_level': threat_level
        }
        
    def _update_system_stats(self, final_decision: Dict[str, Any], threat_level: float) -> None:
        """Update system statistics."""
        self.system_stats['total_inputs_processed'] += 1
        
        if threat_level > 0.6:
            self.system_stats['threats_detected'] += 1
            
        if not final_decision['verified']:
            self.system_stats['successful_defenses'] += 1
            
    def _generate_defense_summary(self, 
                                pipeline_results: Dict[str, Any],
                                final_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable defense summary."""
        summary = {
            'defense_activated': [],
            'threats_found': [],
            'confidence_factors': []
        }
        
        # Check each defense stage
        if 'input_filtering' in pipeline_results:
            filter_result = pipeline_results['input_filtering']
            if filter_result.get('filtering_applied', False):
                summary['defense_activated'].append('input_sanitization')
            if filter_result.get('detection', {}).get('is_adversarial', False):
                summary['threats_found'].append('adversarial_input_detected')
                
        if 'randomized_defense' in pipeline_results:
            summary['defense_activated'].append('randomized_smoothing')
            
        if 'adaptive_verification' in pipeline_results:
            adaptive_result = pipeline_results['adaptive_verification']
            if adaptive_result.get('thresholds_used', {}):
                summary['defense_activated'].append('adaptive_thresholds')
                
        # Final assessment
        summary['final_status'] = 'VERIFIED' if final_decision['verified'] else 'REJECTED'
        summary['overall_confidence'] = final_decision['confidence']
        summary['decision_basis'] = final_decision.get('decision_reason', 'unknown')
        
        return summary
        
    def _compute_input_statistics(self, input_data: torch.Tensor) -> Dict[str, float]:
        """Compute basic input statistics for threat assessment."""
        flattened = input_data.flatten()
        
        return {
            'magnitude': torch.norm(flattened).item(),
            'variance': torch.var(flattened).item(),
            'mean': torch.mean(flattened).item(),
            'max_abs': torch.max(torch.abs(flattened)).item(),
            'std': torch.std(flattened).item()
        }


# Helper function for creating defense configurations
    def calculate_false_positive_rate(self, results: List[Dict]) -> float:
        """Calculate false positive rate.
        
        Args:
            results: List of detection results
            
        Returns:
            False positive rate
        """
        if not results:
            return 0.0
            
        clean_flagged = sum(1 for r in results if r.get('clean', False) and r.get('flagged', False))
        total_clean = sum(1 for r in results if r.get('clean', False))
        
        if total_clean == 0:
            return 0.0
            
        return clean_flagged / total_clean
    
    def calculate_detection_accuracy(self, results: List[Dict]) -> float:
        """Calculate detection accuracy.
        
        Args:
            results: List of detection results
            
        Returns:
            Detection accuracy
        """
        if not results:
            return 0.0
            
        correct = sum(1 for r in results if r.get('is_attack', False) == r.get('detected', False))
        total = len(results)
        
        return correct / total if total > 0 else 0.0
    
    def _ensemble_vote(self, results: List[Dict]) -> Dict[str, Any]:
        """Perform ensemble voting on verification results.
        
        Args:
            results: List of verification results
            
        Returns:
            Consensus result
        """
        if not results:
            return {'verified': False, 'confidence': 0.0}
            
        # Count votes
        verified_votes = sum(1 for r in results if r.get('verified', False))
        total_votes = len(results)
        
        # Average confidence
        confidences = [r.get('confidence', 0.0) for r in results]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'verified': verified_votes > total_votes / 2,
            'confidence': avg_confidence,
            'vote_ratio': verified_votes / total_votes if total_votes > 0 else 0.0
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the defense system.
        
        Returns:
            Performance metrics dictionary
        """
        return {
            'total_defenses': self.system_stats.get('total_inputs_processed', 0),
            'threats_detected': self.system_stats.get('threats_detected', 0),
            'successful_defenses': self.system_stats.get('successful_defenses', 0),
            'pipeline_failures': self.system_stats.get('pipeline_failures', 0),
            'detection_rate': (self.system_stats.get('threats_detected', 0) / 
                             max(1, self.system_stats.get('total_inputs_processed', 0))),
            'average_latency': 0.0  # Placeholder - would track actual latencies
        }


def create_defense_config(defense_type: str, **kwargs) -> DefenseConfig:
    """
    Create defense configuration with sensible defaults.
    
    Args:
        defense_type: Type of defense ('adaptive', 'filter', 'randomized')
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured DefenseConfig object
    """
    default_configs = {
        'adaptive': {
            'adaptation_rate': 0.1,
            'memory_size': 1000,
            'update_frequency': 10,
            'parameters': {
                'adaptation_strength': 0.2,
                'pattern_memory': 500,
                'threshold_bounds': (0.1, 0.9)
            }
        },
        'filter': {
            'parameters': {
                'detection_methods': ['statistical', 'isolation_forest'],
                'sanitization_methods': ['gaussian_blur', 'quantization'],
                'detection_threshold': 0.5,
                'sanitization_strength': 0.1
            }
        },
        'randomized': {
            'parameters': {
                'smoothing_noise_levels': [0.05, 0.1, 0.15],
                'verification_trials': [5, 10, 20],
                'consensus_threshold': 0.7,
                'confidence_threshold': 0.8
            }
        }
    }
    
    if defense_type not in default_configs:
        raise ValueError(f"Unknown defense type: {defense_type}")
        
    config = default_configs[defense_type].copy()
    config.update(kwargs)
    
    return DefenseConfig(defense_type=defense_type, **config)


# Export all classes and functions
__all__ = [
    'DefenseConfig',
    'AttackObservation', 
    'BaseVerifier',
    'MockBaseVerifier',
    'AdaptiveVerifier',
    'InputFilter',
    'RandomizedDefense',
    'IntegratedDefenseSystem',
    'create_defense_config'
]