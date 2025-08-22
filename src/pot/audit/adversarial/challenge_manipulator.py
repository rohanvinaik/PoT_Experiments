"""
Challenge Manipulator

Implements challenge manipulation attacks for PoT verification systems.
"""

import numpy as np
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass 
class ManipulationResult:
    """Result of challenge manipulation attempt"""
    original_challenge: str
    manipulated_challenge: str
    manipulation_type: str
    success: bool
    detection_evaded: bool
    impact: float


class ChallengeManipulator:
    """
    Manipulates challenges to bypass verification or extract information.
    
    Features:
    - Challenge prediction attacks
    - Replay attacks with modifications
    - Semantic-preserving manipulations
    - Collision generation attempts
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the challenge manipulator"""
        self.config = config or {}
        self.manipulation_history = []
        
    def manipulate_challenge(
        self,
        challenge: str,
        strategy: str = 'adaptive'
    ) -> ManipulationResult:
        """
        Manipulate a challenge using specified strategy.
        
        Args:
            challenge: Original challenge
            strategy: Manipulation strategy
            
        Returns:
            ManipulationResult
        """
        strategies = {
            'bit_flip': self._bit_flip_manipulation,
            'semantic': self._semantic_manipulation,
            'replay': self._replay_manipulation,
            'prediction': self._prediction_manipulation,
            'collision': self._collision_manipulation,
            'adaptive': self._adaptive_manipulation
        }
        
        manip_fn = strategies.get(strategy, self._adaptive_manipulation)
        return manip_fn(challenge)
    
    def _bit_flip_manipulation(self, challenge: str) -> ManipulationResult:
        """Flip bits in challenge to test integrity"""
        # Convert to bytes and flip random bits
        challenge_bytes = bytearray(challenge.encode())
        num_flips = min(3, len(challenge_bytes) // 10)
        
        for _ in range(num_flips):
            pos = np.random.randint(0, len(challenge_bytes))
            bit_pos = np.random.randint(0, 8)
            challenge_bytes[pos] ^= (1 << bit_pos)
        
        manipulated = challenge_bytes.decode('utf-8', errors='ignore')
        
        return ManipulationResult(
            original_challenge=challenge,
            manipulated_challenge=manipulated,
            manipulation_type='bit_flip',
            success=manipulated != challenge,
            detection_evaded=False,  # Usually detected
            impact=self._calculate_hamming_distance(challenge, manipulated) / len(challenge)
        )
    
    def _semantic_manipulation(self, challenge: str) -> ManipulationResult:
        """Preserve semantic meaning while changing representation"""
        manipulations = []
        
        # Try different encodings
        manipulations.append(challenge.upper())
        manipulations.append(challenge.lower())
        manipulations.append(' '.join(challenge.split()))  # Normalize whitespace
        
        # Add Unicode variations
        unicode_map = {'a': 'а', 'e': 'е', 'o': 'о'}  # Cyrillic lookalikes
        manipulated = challenge
        for char, replacement in unicode_map.items():
            if char in manipulated:
                manipulated = manipulated.replace(char, replacement, 1)
                break
        
        manipulations.append(manipulated)
        
        # Choose most different but semantically similar
        best_manip = max(manipulations, key=lambda x: self._semantic_similarity(challenge, x))
        
        return ManipulationResult(
            original_challenge=challenge,
            manipulated_challenge=best_manip,
            manipulation_type='semantic',
            success=best_manip != challenge,
            detection_evaded=True,  # Often evades simple checks
            impact=0.3  # Moderate impact
        )
    
    def _replay_manipulation(self, challenge: str) -> ManipulationResult:
        """Replay previous challenge with slight modifications"""
        if self.manipulation_history:
            # Use previous challenge as base
            prev_challenge = np.random.choice(self.manipulation_history)
            
            # Combine with current challenge
            manipulated = self._combine_challenges(prev_challenge, challenge)
        else:
            # First replay - just store
            manipulated = challenge
        
        self.manipulation_history.append(challenge)
        
        return ManipulationResult(
            original_challenge=challenge,
            manipulated_challenge=manipulated,
            manipulation_type='replay',
            success=manipulated != challenge,
            detection_evaded=len(self.manipulation_history) > 5,
            impact=0.5 if manipulated != challenge else 0.0
        )
    
    def _prediction_manipulation(self, challenge: str) -> ManipulationResult:
        """Attempt to predict next challenge"""
        # Analyze patterns in history
        if len(self.manipulation_history) >= 3:
            # Simple pattern detection
            pattern = self._detect_pattern(self.manipulation_history[-3:])
            predicted = self._apply_pattern(challenge, pattern)
        else:
            # Not enough history
            predicted = challenge + '_predicted'
        
        return ManipulationResult(
            original_challenge=challenge,
            manipulated_challenge=predicted,
            manipulation_type='prediction',
            success=False,  # Prediction attacks rarely succeed directly
            detection_evaded=False,
            impact=0.1
        )
    
    def _collision_manipulation(self, challenge: str) -> ManipulationResult:
        """Attempt to find collision (same hash, different input)"""
        # Simplified collision attempt (real collision finding is computationally hard)
        original_hash = hashlib.sha256(challenge.encode()).hexdigest()[:8]
        
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts:
            candidate = challenge + f'_{attempts}'
            candidate_hash = hashlib.sha256(candidate.encode()).hexdigest()[:8]
            
            if candidate_hash == original_hash and candidate != challenge:
                return ManipulationResult(
                    original_challenge=challenge,
                    manipulated_challenge=candidate,
                    manipulation_type='collision',
                    success=True,
                    detection_evaded=True,
                    impact=0.9  # High impact if successful
                )
            
            attempts += 1
        
        return ManipulationResult(
            original_challenge=challenge,
            manipulated_challenge=challenge,
            manipulation_type='collision',
            success=False,
            detection_evaded=False,
            impact=0.0
        )
    
    def _adaptive_manipulation(self, challenge: str) -> ManipulationResult:
        """Adaptively choose best manipulation strategy"""
        # Try multiple strategies and pick best
        strategies = ['bit_flip', 'semantic', 'replay', 'collision']
        results = []
        
        for strategy in strategies:
            if strategy == 'adaptive':
                continue
            try:
                result = self.manipulate_challenge(challenge, strategy)
                results.append(result)
            except Exception as e:
                logger.debug(f"Strategy {strategy} failed: {e}")
        
        if results:
            # Choose strategy with highest impact that evades detection
            best = max(results, key=lambda r: r.impact * (2.0 if r.detection_evaded else 1.0))
            return best
        
        return ManipulationResult(
            original_challenge=challenge,
            manipulated_challenge=challenge,
            manipulation_type='adaptive',
            success=False,
            detection_evaded=False,
            impact=0.0
        )
    
    def _calculate_hamming_distance(self, s1: str, s2: str) -> int:
        """Calculate Hamming distance between strings"""
        if len(s1) != len(s2):
            return max(len(s1), len(s2))
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    def _semantic_similarity(self, s1: str, s2: str) -> float:
        """Calculate semantic similarity (simplified)"""
        # Simple character overlap metric
        set1 = set(s1.lower())
        set2 = set(s2.lower())
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)
    
    def _combine_challenges(self, prev: str, curr: str) -> str:
        """Combine two challenges"""
        # Take parts from each
        mid = len(prev) // 2
        return prev[:mid] + curr[mid:]
    
    def _detect_pattern(self, history: List[str]) -> Dict[str, Any]:
        """Detect patterns in challenge history"""
        pattern = {
            'length_trend': np.diff([len(h) for h in history]).mean(),
            'common_prefix': self._common_prefix(history),
            'common_suffix': self._common_suffix(history)
        }
        return pattern
    
    def _apply_pattern(self, challenge: str, pattern: Dict[str, Any]) -> str:
        """Apply detected pattern to predict next challenge"""
        predicted = challenge
        
        if pattern['common_prefix']:
            predicted = pattern['common_prefix'] + predicted
        if pattern['common_suffix']:
            predicted = predicted + pattern['common_suffix']
        
        # Adjust length based on trend
        if pattern['length_trend'] > 0:
            predicted += '_ext'
        elif pattern['length_trend'] < 0 and len(predicted) > 5:
            predicted = predicted[:-3]
        
        return predicted
    
    def _common_prefix(self, strings: List[str]) -> str:
        """Find common prefix of strings"""
        if not strings:
            return ""
        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix
    
    def _common_suffix(self, strings: List[str]) -> str:
        """Find common suffix of strings"""
        if not strings:
            return ""
        reversed_strings = [s[::-1] for s in strings]
        suffix = self._common_prefix(reversed_strings)
        return suffix[::-1]