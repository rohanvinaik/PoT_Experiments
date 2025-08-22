"""
Information Leakage Detector

Identifies and quantifies information leakage in model responses and audit trails.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import stats
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class LeakageReport:
    """Report of detected information leakage"""
    leakage_score: float
    mutual_information: float
    entropy_loss: float
    statistical_distance: float
    leakage_types: List[str]
    evidence: Dict[str, Any]
    recommendations: List[str]


class LeakageDetector:
    """
    Detects and quantifies information leakage in PoT verification.
    
    Features:
    - Mutual information analysis
    - Statistical distance measurement
    - Pattern recognition for common leakage types
    - Differential privacy analysis
    - Side-channel leakage detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the leakage detector.
        
        Args:
            config: Configuration for leakage detection parameters
        """
        self.config = config or {}
        self.sensitivity_threshold = self.config.get('sensitivity_threshold', 0.1)
        self.epsilon = self.config.get('epsilon', 1.0)  # Differential privacy parameter
        self.detection_methods = self.config.get('methods', ['mutual_info', 'statistical', 'pattern'])
        self.leakage_patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, Any]:
        """Initialize known leakage patterns"""
        return {
            'weight_exposure': {
                'keywords': ['weight', 'parameter', 'coefficient'],
                'pattern': r'\d+\.\d{6,}',  # High precision floats
                'severity': 'high'
            },
            'architecture_leak': {
                'keywords': ['layer', 'neuron', 'activation', 'dropout'],
                'pattern': r'(conv|dense|lstm|gru|attention)\d+',
                'severity': 'medium'
            },
            'training_data': {
                'keywords': ['example', 'sample', 'data point'],
                'pattern': r'(?:train|test|val)_\d+',
                'severity': 'critical'
            },
            'hyperparameter': {
                'keywords': ['learning_rate', 'batch_size', 'epochs'],
                'pattern': r'(lr|bs|ep)=\d+',
                'severity': 'low'
            }
        }
    
    def analyze(
        self, 
        responses: List[str],
        challenges: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LeakageReport:
        """
        Analyze responses for information leakage.
        
        Args:
            responses: Model responses to analyze
            challenges: Original challenges (if available)
            metadata: Additional metadata for context
            
        Returns:
            LeakageReport with detailed findings
        """
        leakage_types = []
        evidence = {}
        
        # Calculate information-theoretic metrics
        mutual_info = self._calculate_mutual_information(responses, challenges)
        entropy_loss = self._calculate_entropy_loss(responses)
        statistical_dist = self._calculate_statistical_distance(responses)
        
        # Detect specific leakage patterns
        pattern_leaks = self._detect_pattern_leakage(responses)
        leakage_types.extend(pattern_leaks)
        
        # Check for side-channel leakage
        timing_leakage = self._detect_timing_leakage(responses, metadata)
        if timing_leakage:
            leakage_types.append('timing_channel')
            evidence['timing'] = timing_leakage
        
        # Calculate differential privacy violation
        dp_violation = self._check_differential_privacy(responses)
        if dp_violation:
            leakage_types.append('differential_privacy_violation')
            evidence['dp_violation'] = dp_violation
        
        # Calculate overall leakage score
        leakage_score = self._calculate_leakage_score(
            mutual_info, entropy_loss, statistical_dist, len(leakage_types)
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(leakage_types, leakage_score)
        
        return LeakageReport(
            leakage_score=leakage_score,
            mutual_information=mutual_info,
            entropy_loss=entropy_loss,
            statistical_distance=statistical_dist,
            leakage_types=leakage_types,
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _calculate_mutual_information(
        self, 
        responses: List[str],
        challenges: Optional[List[str]]
    ) -> float:
        """
        Calculate mutual information between challenges and responses.
        
        Args:
            responses: Model responses
            challenges: Input challenges
            
        Returns:
            Mutual information in bits
        """
        if not challenges or len(challenges) != len(responses):
            return 0.0
        
        # Convert to discrete representations
        response_tokens = [self._tokenize(r) for r in responses]
        challenge_tokens = [self._tokenize(c) for c in challenges]
        
        # Calculate joint and marginal distributions
        joint_dist = Counter(zip(challenge_tokens, response_tokens))
        challenge_dist = Counter(challenge_tokens)
        response_dist = Counter(response_tokens)
        
        total = sum(joint_dist.values())
        mutual_info = 0.0
        
        for (c, r), count in joint_dist.items():
            p_joint = count / total
            p_c = challenge_dist[c] / total
            p_r = response_dist[r] / total
            
            if p_joint > 0:
                mutual_info += p_joint * np.log2(p_joint / (p_c * p_r))
        
        return mutual_info
    
    def _tokenize(self, text: str) -> str:
        """Simple tokenization for MI calculation"""
        # Use hash for discrete representation
        return hashlib.md5(text.encode()).hexdigest()[:8]
    
    def _calculate_entropy_loss(self, responses: List[str]) -> float:
        """
        Calculate entropy loss in responses.
        
        Args:
            responses: Model responses
            
        Returns:
            Entropy loss value
        """
        if not responses:
            return 0.0
        
        # Calculate response entropy
        response_counts = Counter(responses)
        total = len(responses)
        entropy = 0.0
        
        for count in response_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Maximum possible entropy
        max_entropy = np.log2(total)
        
        # Entropy loss (normalized)
        if max_entropy > 0:
            entropy_loss = 1.0 - (entropy / max_entropy)
        else:
            entropy_loss = 0.0
        
        return entropy_loss
    
    def _calculate_statistical_distance(self, responses: List[str]) -> float:
        """
        Calculate statistical distance from expected distribution.
        
        Args:
            responses: Model responses
            
        Returns:
            Statistical distance measure
        """
        if len(responses) < 2:
            return 0.0
        
        # Convert responses to numerical features
        features = []
        for response in responses:
            feature_vec = [
                len(response),
                response.count(' '),
                len(set(response.split())),
                sum(1 for c in response if c.isdigit())
            ]
            features.append(feature_vec)
        
        features = np.array(features)
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                dist = np.linalg.norm(features[i] - features[j])
                distances.append(dist)
        
        if distances:
            # Normalize by expected distance
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            if std_dist > 0:
                statistical_distance = mean_dist / std_dist
            else:
                statistical_distance = 0.0
        else:
            statistical_distance = 0.0
        
        return min(statistical_distance, 1.0)
    
    def _detect_pattern_leakage(self, responses: List[str]) -> List[str]:
        """
        Detect specific leakage patterns in responses.
        
        Args:
            responses: Model responses to analyze
            
        Returns:
            List of detected leakage pattern types
        """
        import re
        detected = []
        
        combined_text = ' '.join(responses).lower()
        
        for pattern_name, pattern_info in self.leakage_patterns.items():
            # Check keywords
            keyword_found = any(kw in combined_text for kw in pattern_info['keywords'])
            
            # Check regex pattern
            pattern_found = bool(re.search(pattern_info['pattern'], combined_text))
            
            if keyword_found or pattern_found:
                detected.append(pattern_name)
        
        return detected
    
    def _detect_timing_leakage(
        self, 
        responses: List[str],
        metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect timing-based side channel leakage.
        
        Args:
            responses: Model responses
            metadata: Metadata including timing information
            
        Returns:
            Timing leakage evidence if detected
        """
        if not metadata or 'timings' not in metadata:
            return None
        
        timings = metadata['timings']
        if len(timings) != len(responses):
            return None
        
        # Group by response similarity
        response_groups = {}
        for i, response in enumerate(responses):
            key = len(response)  # Simple grouping by length
            if key not in response_groups:
                response_groups[key] = []
            response_groups[key].append(timings[i])
        
        # Check for timing correlation
        if len(response_groups) > 1:
            groups = list(response_groups.values())
            # Perform ANOVA test
            f_stat, p_value = stats.f_oneway(*groups)
            
            if p_value < 0.05:  # Significant timing difference
                return {
                    'correlation_detected': True,
                    'p_value': p_value,
                    'f_statistic': f_stat,
                    'group_means': {k: np.mean(v) for k, v in response_groups.items()}
                }
        
        return None
    
    def _check_differential_privacy(self, responses: List[str]) -> Optional[Dict[str, Any]]:
        """
        Check for differential privacy violations.
        
        Args:
            responses: Model responses
            
        Returns:
            DP violation evidence if detected
        """
        if len(responses) < 10:
            return None
        
        # Simulate neighboring datasets by removing one response
        violations = []
        
        for i in range(min(5, len(responses))):  # Check first 5 removals
            neighbor_responses = responses[:i] + responses[i+1:]
            
            # Calculate sensitivity
            original_stats = self._calculate_response_statistics(responses)
            neighbor_stats = self._calculate_response_statistics(neighbor_responses)
            
            sensitivity = abs(original_stats['mean_length'] - neighbor_stats['mean_length'])
            
            # Check if sensitivity exceeds DP bound
            dp_bound = 2.0 / (self.epsilon * len(responses))
            
            if sensitivity > dp_bound:
                violations.append({
                    'removed_index': i,
                    'sensitivity': sensitivity,
                    'dp_bound': dp_bound,
                    'violation_ratio': sensitivity / dp_bound
                })
        
        if violations:
            return {
                'violations': violations,
                'max_violation': max(v['violation_ratio'] for v in violations),
                'epsilon': self.epsilon
            }
        
        return None
    
    def _calculate_response_statistics(self, responses: List[str]) -> Dict[str, float]:
        """Calculate basic statistics for responses"""
        lengths = [len(r) for r in responses]
        return {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'unique_ratio': len(set(responses)) / len(responses)
        }
    
    def _calculate_leakage_score(
        self,
        mutual_info: float,
        entropy_loss: float,
        statistical_dist: float,
        pattern_count: int
    ) -> float:
        """
        Calculate overall leakage score.
        
        Args:
            mutual_info: Mutual information measure
            entropy_loss: Entropy loss measure
            statistical_dist: Statistical distance measure
            pattern_count: Number of detected patterns
            
        Returns:
            Leakage score between 0.0 and 1.0
        """
        # Normalize mutual information (cap at 10 bits)
        mi_score = min(mutual_info / 10.0, 1.0) * 0.3
        
        # Entropy loss contribution
        entropy_score = entropy_loss * 0.3
        
        # Statistical distance contribution
        stat_score = statistical_dist * 0.2
        
        # Pattern detection contribution
        pattern_score = min(pattern_count * 0.1, 0.2)
        
        total_score = mi_score + entropy_score + stat_score + pattern_score
        
        return min(total_score, 1.0)
    
    def _generate_recommendations(
        self,
        leakage_types: List[str],
        leakage_score: float
    ) -> List[str]:
        """
        Generate recommendations based on detected leakage.
        
        Args:
            leakage_types: Types of leakage detected
            leakage_score: Overall leakage score
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if leakage_score > 0.7:
            recommendations.append("Critical: Implement immediate leakage mitigation")
        elif leakage_score > 0.4:
            recommendations.append("Warning: Review and strengthen privacy controls")
        
        if 'weight_exposure' in leakage_types:
            recommendations.append("Implement weight obfuscation or quantization")
        
        if 'architecture_leak' in leakage_types:
            recommendations.append("Use generic model descriptions in responses")
        
        if 'training_data' in leakage_types:
            recommendations.append("Critical: Remove all training data references")
            recommendations.append("Implement differential privacy training")
        
        if 'timing_channel' in leakage_types:
            recommendations.append("Add random delays to prevent timing attacks")
        
        if 'differential_privacy_violation' in leakage_types:
            recommendations.append("Increase epsilon parameter or add noise to outputs")
        
        if not recommendations:
            recommendations.append("Maintain current privacy controls")
        
        return recommendations
    
    def measure_channel_capacity(
        self,
        responses: List[str],
        challenges: List[str]
    ) -> float:
        """
        Measure the channel capacity for information leakage.
        
        Args:
            responses: Model responses
            challenges: Input challenges
            
        Returns:
            Channel capacity in bits
        """
        if len(responses) != len(challenges):
            return 0.0
        
        # Build conditional probability matrix
        unique_challenges = list(set(challenges))
        unique_responses = list(set(responses))
        
        prob_matrix = np.zeros((len(unique_challenges), len(unique_responses)))
        
        for c_idx, challenge in enumerate(unique_challenges):
            challenge_responses = [r for c, r in zip(challenges, responses) if c == challenge]
            for r_idx, response in enumerate(unique_responses):
                prob_matrix[c_idx, r_idx] = challenge_responses.count(response) / len(challenge_responses) if challenge_responses else 0
        
        # Calculate channel capacity (maximum mutual information)
        # Using iterative algorithm
        input_dist = np.ones(len(unique_challenges)) / len(unique_challenges)
        
        for _ in range(100):  # Iterate to convergence
            output_dist = input_dist @ prob_matrix
            if np.sum(output_dist) > 0:
                scales = prob_matrix / output_dist[np.newaxis, :]
                scales[np.isnan(scales)] = 0
                geometric_mean = np.exp(input_dist @ np.log(np.maximum(scales, 1e-10)))
                input_dist = input_dist * geometric_mean
                input_dist /= np.sum(input_dist)
        
        # Calculate capacity
        capacity = 0.0
        for i in range(len(unique_challenges)):
            for j in range(len(unique_responses)):
                if prob_matrix[i, j] > 0 and output_dist[j] > 0:
                    capacity += input_dist[i] * prob_matrix[i, j] * np.log2(
                        prob_matrix[i, j] / output_dist[j]
                    )
        
        return capacity