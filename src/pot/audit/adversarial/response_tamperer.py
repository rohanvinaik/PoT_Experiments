"""
Response Tamperer

Implements response tampering attacks for PoT systems.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class TamperingResult:
    """Result of response tampering attempt"""
    original_response: Any
    tampered_response: Any
    tampering_method: str
    preserves_format: bool
    detection_probability: float
    semantic_distance: float


class ResponseTamperer:
    """
    Tampers with model responses to evade detection or manipulate verification.
    
    Features:
    - Format-preserving tampering
    - Semantic manipulation
    - Statistical property preservation
    - Gradual drift attacks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the response tamperer"""
        self.config = config or {}
        self.tampering_strength = self.config.get('strength', 0.1)
        
    def tamper_response(
        self,
        response: Any,
        method: str = 'adaptive',
        constraints: Optional[Dict[str, Any]] = None
    ) -> TamperingResult:
        """
        Tamper with a response using specified method.
        
        Args:
            response: Original response
            method: Tampering method
            constraints: Constraints to preserve
            
        Returns:
            TamperingResult
        """
        methods = {
            'noise': self._add_noise,
            'semantic': self._semantic_tampering,
            'statistical': self._statistical_tampering,
            'format_preserving': self._format_preserving_tampering,
            'gradual_drift': self._gradual_drift,
            'adaptive': self._adaptive_tampering
        }
        
        tamper_fn = methods.get(method, self._adaptive_tampering)
        return tamper_fn(response, constraints)
    
    def _add_noise(self, response: Any, constraints: Optional[Dict[str, Any]]) -> TamperingResult:
        """Add noise to response"""
        if isinstance(response, (int, float)):
            noise = np.random.normal(0, self.tampering_strength)
            tampered = response + noise
            
            # Apply constraints
            if constraints:
                if 'min_value' in constraints:
                    tampered = max(tampered, constraints['min_value'])
                if 'max_value' in constraints:
                    tampered = max(tampered, constraints['max_value'])
            
            return TamperingResult(
                original_response=response,
                tampered_response=tampered,
                tampering_method='noise',
                preserves_format=True,
                detection_probability=0.3,
                semantic_distance=abs(noise)
            )
        
        elif isinstance(response, str):
            # Add character-level noise
            tampered = self._add_string_noise(response)
            return TamperingResult(
                original_response=response,
                tampered_response=tampered,
                tampering_method='noise',
                preserves_format=True,
                detection_probability=0.5,
                semantic_distance=self._string_distance(response, tampered)
            )
        
        elif isinstance(response, list):
            # Add noise to each element
            tampered = []
            for item in response:
                if isinstance(item, (int, float)):
                    tampered.append(item + np.random.normal(0, self.tampering_strength))
                else:
                    tampered.append(item)
            
            return TamperingResult(
                original_response=response,
                tampered_response=tampered,
                tampering_method='noise',
                preserves_format=True,
                detection_probability=0.3,
                semantic_distance=np.mean([abs(a - b) for a, b in zip(response, tampered) 
                                          if isinstance(a, (int, float))])
            )
        
        return TamperingResult(
            original_response=response,
            tampered_response=response,
            tampering_method='noise',
            preserves_format=True,
            detection_probability=0.0,
            semantic_distance=0.0
        )
    
    def _semantic_tampering(self, response: Any, constraints: Optional[Dict[str, Any]]) -> TamperingResult:
        """Tamper with semantic meaning while preserving format"""
        if isinstance(response, str):
            # Synonym replacement
            replacements = {
                'yes': 'affirmative',
                'no': 'negative',
                'true': 'correct',
                'false': 'incorrect',
                'good': 'acceptable',
                'bad': 'unacceptable'
            }
            
            tampered = response.lower()
            for orig, repl in replacements.items():
                if orig in tampered:
                    tampered = tampered.replace(orig, repl)
                    break
            
            return TamperingResult(
                original_response=response,
                tampered_response=tampered,
                tampering_method='semantic',
                preserves_format=True,
                detection_probability=0.2,
                semantic_distance=0.5 if tampered != response else 0.0
            )
        
        elif isinstance(response, dict):
            # Tamper with dict values
            tampered = {}
            for key, value in response.items():
                if isinstance(value, str):
                    result = self._semantic_tampering(value, constraints)
                    tampered[key] = result.tampered_response
                else:
                    tampered[key] = value
            
            return TamperingResult(
                original_response=response,
                tampered_response=tampered,
                tampering_method='semantic',
                preserves_format=True,
                detection_probability=0.3,
                semantic_distance=0.3
            )
        
        return TamperingResult(
            original_response=response,
            tampered_response=response,
            tampering_method='semantic',
            preserves_format=True,
            detection_probability=0.0,
            semantic_distance=0.0
        )
    
    def _statistical_tampering(self, response: Any, constraints: Optional[Dict[str, Any]]) -> TamperingResult:
        """Preserve statistical properties while tampering"""
        if isinstance(response, list) and all(isinstance(x, (int, float)) for x in response):
            # Preserve mean and std
            original_mean = np.mean(response)
            original_std = np.std(response)
            
            # Generate new values with same statistics
            tampered = np.random.normal(original_mean, original_std, len(response))
            
            # Sort to preserve ordering properties if needed
            if constraints and constraints.get('preserve_order'):
                tampered = np.sort(tampered)
            
            tampered = tampered.tolist()
            
            return TamperingResult(
                original_response=response,
                tampered_response=tampered,
                tampering_method='statistical',
                preserves_format=True,
                detection_probability=0.1,  # Hard to detect
                semantic_distance=np.mean(np.abs(np.array(response) - np.array(tampered)))
            )
        
        return self._add_noise(response, constraints)
    
    def _format_preserving_tampering(self, response: Any, constraints: Optional[Dict[str, Any]]) -> TamperingResult:
        """Tamper while strictly preserving format"""
        if isinstance(response, str):
            # Preserve length and character types
            tampered = []
            for char in response:
                if char.isdigit():
                    tampered.append(str(np.random.randint(0, 10)))
                elif char.isalpha():
                    if char.isupper():
                        tampered.append(chr(np.random.randint(65, 91)))
                    else:
                        tampered.append(chr(np.random.randint(97, 123)))
                else:
                    tampered.append(char)
            
            tampered = ''.join(tampered)
            
            return TamperingResult(
                original_response=response,
                tampered_response=tampered,
                tampering_method='format_preserving',
                preserves_format=True,
                detection_probability=0.4,
                semantic_distance=self._string_distance(response, tampered)
            )
        
        elif isinstance(response, dict):
            # Preserve structure
            tampered = {}
            for key, value in response.items():
                if isinstance(value, (int, float)):
                    # Slight modification
                    tampered[key] = value * (1 + np.random.uniform(-0.1, 0.1))
                elif isinstance(value, str):
                    # Preserve length
                    tampered[key] = ''.join(np.random.choice(list(value), len(value)))
                else:
                    tampered[key] = value
            
            return TamperingResult(
                original_response=response,
                tampered_response=tampered,
                tampering_method='format_preserving',
                preserves_format=True,
                detection_probability=0.3,
                semantic_distance=0.4
            )
        
        return TamperingResult(
            original_response=response,
            tampered_response=response,
            tampering_method='format_preserving',
            preserves_format=True,
            detection_probability=0.0,
            semantic_distance=0.0
        )
    
    def _gradual_drift(self, response: Any, constraints: Optional[Dict[str, Any]]) -> TamperingResult:
        """Implement gradual drift attack"""
        # Store history for drift
        if not hasattr(self, 'drift_history'):
            self.drift_history = []
        
        self.drift_history.append(response)
        
        if isinstance(response, (int, float)):
            # Calculate drift based on history
            drift_rate = 0.01 * len(self.drift_history)
            tampered = response * (1 + drift_rate)
            
            return TamperingResult(
                original_response=response,
                tampered_response=tampered,
                tampering_method='gradual_drift',
                preserves_format=True,
                detection_probability=0.05 * len(self.drift_history),  # Increases over time
                semantic_distance=abs(response - tampered)
            )
        
        return self._add_noise(response, constraints)
    
    def _adaptive_tampering(self, response: Any, constraints: Optional[Dict[str, Any]]) -> TamperingResult:
        """Adaptively choose tampering method"""
        # Analyze response type and choose appropriate method
        if isinstance(response, (int, float)):
            methods = ['noise', 'gradual_drift']
        elif isinstance(response, str):
            methods = ['semantic', 'format_preserving']
        elif isinstance(response, list):
            methods = ['statistical', 'noise']
        elif isinstance(response, dict):
            methods = ['format_preserving', 'semantic']
        else:
            methods = ['noise']
        
        # Try methods and pick one with lowest detection probability
        best_result = None
        best_score = float('inf')
        
        for method in methods:
            try:
                result = self.tamper_response(response, method, constraints)
                # Score based on detection probability and impact
                score = result.detection_probability - result.semantic_distance * 0.5
                
                if score < best_score:
                    best_score = score
                    best_result = result
            except Exception as e:
                logger.debug(f"Method {method} failed: {e}")
        
        if best_result:
            return best_result
        
        return TamperingResult(
            original_response=response,
            tampered_response=response,
            tampering_method='adaptive',
            preserves_format=True,
            detection_probability=0.0,
            semantic_distance=0.0
        )
    
    def _add_string_noise(self, text: str) -> str:
        """Add noise to string"""
        if not text:
            return text
        
        noise_types = ['swap', 'delete', 'insert', 'replace']
        noise_type = np.random.choice(noise_types)
        
        text_list = list(text)
        
        if noise_type == 'swap' and len(text_list) > 1:
            # Swap two adjacent characters
            idx = np.random.randint(0, len(text_list) - 1)
            text_list[idx], text_list[idx + 1] = text_list[idx + 1], text_list[idx]
        
        elif noise_type == 'delete' and len(text_list) > 1:
            # Delete a character
            idx = np.random.randint(0, len(text_list))
            del text_list[idx]
        
        elif noise_type == 'insert':
            # Insert a character
            idx = np.random.randint(0, len(text_list) + 1)
            char = chr(np.random.randint(97, 123))  # Random lowercase letter
            text_list.insert(idx, char)
        
        elif noise_type == 'replace' and text_list:
            # Replace a character
            idx = np.random.randint(0, len(text_list))
            text_list[idx] = chr(np.random.randint(97, 123))
        
        return ''.join(text_list)
    
    def _string_distance(self, s1: str, s2: str) -> float:
        """Calculate normalized string distance"""
        if not s1 and not s2:
            return 0.0
        if not s1 or not s2:
            return 1.0
        
        # Levenshtein distance (simplified)
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            new_distances = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(1 + min(distances[i1], distances[i1 + 1], new_distances[-1]))
            distances = new_distances
        
        return distances[-1] / max(len(s1), len(s2))