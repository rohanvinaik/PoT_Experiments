"""
Statistical Attack Detector

Detects adversarial attacks using statistical methods.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of attack detection"""
    attack_detected: bool
    confidence: float
    attack_type: Optional[str]
    evidence: Dict[str, Any]
    recommended_action: str


class StatisticalAttackDetector:
    """
    Detects adversarial attacks using statistical analysis.
    
    Features:
    - Anomaly detection using statistical tests
    - Distribution shift detection
    - Sequential analysis
    - Adaptive thresholds
    - Multi-modal detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the attack detector"""
        self.config = config or {}
        self.sensitivity = self.config.get('sensitivity', 0.95)
        self.window_size = self.config.get('window_size', 100)
        self.history = deque(maxlen=self.window_size)
        self.baseline_stats = None
        self.detection_thresholds = self._initialize_thresholds()
        
    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize detection thresholds"""
        return {
            'distribution_shift': 0.05,  # p-value threshold
            'anomaly_score': 3.0,  # z-score threshold
            'entropy_change': 0.3,  # relative entropy threshold
            'correlation_break': 0.5,  # correlation threshold
            'frequency_anomaly': 0.1  # frequency deviation threshold
        }
    
    def establish_baseline(self, normal_data: List[Any]) -> None:
        """
        Establish baseline statistics from normal data.
        
        Args:
            normal_data: Normal/benign data samples
        """
        if not normal_data:
            logger.warning("No data provided for baseline")
            return
        
        # Extract features from data
        features = [self._extract_features(d) for d in normal_data]
        features = np.array(features)
        
        self.baseline_stats = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'median': np.median(features, axis=0),
            'quantiles': np.percentile(features, [25, 75], axis=0),
            'covariance': np.cov(features.T) if features.shape[1] > 1 else None
        }
        
        logger.info("Baseline statistics established")
    
    def detect_attack(self, data: Any) -> DetectionResult:
        """
        Detect if data represents an attack.
        
        Args:
            data: Data to analyze
            
        Returns:
            DetectionResult
        """
        features = self._extract_features(data)
        self.history.append(features)
        
        # Run multiple detection methods
        detections = {
            'anomaly': self._detect_anomaly(features),
            'distribution': self._detect_distribution_shift(),
            'sequential': self._detect_sequential_attack(),
            'entropy': self._detect_entropy_attack(data),
            'correlation': self._detect_correlation_attack()
        }
        
        # Aggregate detection results
        attack_scores = {}
        evidence = {}
        
        for method, result in detections.items():
            if result:
                attack_scores[method] = result['confidence']
                evidence[method] = result['evidence']
        
        # Overall detection decision
        if attack_scores:
            max_confidence = max(attack_scores.values())
            attack_type = max(attack_scores, key=attack_scores.get)
            attack_detected = max_confidence > (1 - self.sensitivity)
        else:
            max_confidence = 0.0
            attack_type = None
            attack_detected = False
        
        # Recommend action
        if attack_detected:
            if max_confidence > 0.9:
                action = "Block immediately and investigate"
            elif max_confidence > 0.7:
                action = "Flag for review and monitor closely"
            else:
                action = "Monitor and collect more evidence"
        else:
            action = "Continue normal operation"
        
        return DetectionResult(
            attack_detected=attack_detected,
            confidence=max_confidence,
            attack_type=attack_type,
            evidence=evidence,
            recommended_action=action
        )
    
    def _extract_features(self, data: Any) -> np.ndarray:
        """Extract statistical features from data"""
        features = []
        
        if isinstance(data, (int, float)):
            features = [data]
        
        elif isinstance(data, str):
            features = [
                len(data),
                len(set(data)),  # Unique characters
                data.count(' '),  # Spaces
                sum(1 for c in data if c.isdigit()),  # Digits
                sum(1 for c in data if c.isupper())  # Uppercase
            ]
        
        elif isinstance(data, list):
            if all(isinstance(x, (int, float)) for x in data):
                features = [
                    np.mean(data),
                    np.std(data),
                    np.min(data),
                    np.max(data),
                    len(data)
                ]
            else:
                features = [len(data), len(set(map(str, data)))]
        
        elif isinstance(data, dict):
            features = [
                len(data),
                sum(1 for v in data.values() if v is not None)
            ]
        
        else:
            # Default features
            features = [hash(str(data)) % 1000]
        
        return np.array(features, dtype=float)
    
    def _detect_anomaly(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect anomalies using z-score"""
        if self.baseline_stats is None:
            return None
        
        mean = self.baseline_stats['mean']
        std = self.baseline_stats['std']
        
        # Avoid division by zero
        std = np.where(std == 0, 1e-10, std)
        
        # Calculate z-scores
        z_scores = np.abs((features - mean) / std)
        max_z = np.max(z_scores)
        
        if max_z > self.detection_thresholds['anomaly_score']:
            return {
                'confidence': min(max_z / 10.0, 1.0),
                'evidence': {
                    'max_z_score': float(max_z),
                    'anomalous_features': np.where(z_scores > self.detection_thresholds['anomaly_score'])[0].tolist()
                }
            }
        
        return None
    
    def _detect_distribution_shift(self) -> Optional[Dict[str, Any]]:
        """Detect distribution shift using statistical tests"""
        if len(self.history) < 20 or self.baseline_stats is None:
            return None
        
        recent_data = np.array(list(self.history)[-20:])
        baseline_mean = self.baseline_stats['mean']
        
        # Perform multivariate Hotelling's T-squared test
        if recent_data.shape[1] > 1:
            mean_diff = np.mean(recent_data, axis=0) - baseline_mean
            cov = self.baseline_stats.get('covariance')
            
            if cov is not None and np.linalg.det(cov) != 0:
                cov_inv = np.linalg.inv(cov)
                t_squared = len(recent_data) * mean_diff @ cov_inv @ mean_diff.T
                
                # Convert to F-statistic
                p = recent_data.shape[1]
                n = len(recent_data)
                f_stat = (n - p) / (p * (n - 1)) * t_squared
                p_value = 1 - stats.f.cdf(f_stat, p, n - p)
                
                if p_value < self.detection_thresholds['distribution_shift']:
                    return {
                        'confidence': 1 - p_value,
                        'evidence': {
                            'p_value': p_value,
                            'f_statistic': f_stat,
                            'mean_shift': mean_diff.tolist()
                        }
                    }
        else:
            # Univariate t-test
            t_stat, p_value = stats.ttest_1samp(recent_data[:, 0], baseline_mean[0])
            
            if p_value < self.detection_thresholds['distribution_shift']:
                return {
                    'confidence': 1 - p_value,
                    'evidence': {
                        'p_value': p_value,
                        't_statistic': t_stat
                    }
                }
        
        return None
    
    def _detect_sequential_attack(self) -> Optional[Dict[str, Any]]:
        """Detect sequential pattern attacks"""
        if len(self.history) < 10:
            return None
        
        recent = np.array(list(self.history)[-10:])
        
        # Check for suspicious patterns
        patterns_detected = []
        
        # Monotonic increase/decrease
        if recent.shape[1] > 0:
            for i in range(recent.shape[1]):
                col = recent[:, i]
                if np.all(np.diff(col) > 0) or np.all(np.diff(col) < 0):
                    patterns_detected.append(f"monotonic_feature_{i}")
        
        # Periodicity detection
        if len(recent) > 4:
            for period in [2, 3, 4]:
                if len(recent) >= period * 2:
                    periodic = True
                    for i in range(period, len(recent)):
                        if not np.allclose(recent[i], recent[i - period], rtol=0.1):
                            periodic = False
                            break
                    if periodic:
                        patterns_detected.append(f"period_{period}")
        
        if patterns_detected:
            return {
                'confidence': min(len(patterns_detected) * 0.3, 1.0),
                'evidence': {
                    'patterns': patterns_detected,
                    'sequence_length': len(recent)
                }
            }
        
        return None
    
    def _detect_entropy_attack(self, data: Any) -> Optional[Dict[str, Any]]:
        """Detect attacks based on entropy changes"""
        if not isinstance(data, (str, list)):
            return None
        
        # Calculate entropy
        if isinstance(data, str):
            entropy = self._calculate_entropy(list(data))
        else:
            entropy = self._calculate_entropy(data)
        
        # Compare with expected entropy
        if self.baseline_stats and 'entropy' in self.baseline_stats:
            baseline_entropy = self.baseline_stats['entropy']
            entropy_change = abs(entropy - baseline_entropy) / baseline_entropy
            
            if entropy_change > self.detection_thresholds['entropy_change']:
                return {
                    'confidence': min(entropy_change * 2, 1.0),
                    'evidence': {
                        'current_entropy': entropy,
                        'baseline_entropy': baseline_entropy,
                        'relative_change': entropy_change
                    }
                }
        
        # Check for abnormally low or high entropy
        if entropy < 0.1 or entropy > 0.95:
            return {
                'confidence': 0.7,
                'evidence': {
                    'entropy': entropy,
                    'abnormal': 'low' if entropy < 0.1 else 'high'
                }
            }
        
        return None
    
    def _detect_correlation_attack(self) -> Optional[Dict[str, Any]]:
        """Detect attacks that break expected correlations"""
        if len(self.history) < 20 or self.baseline_stats is None:
            return None
        
        if self.baseline_stats.get('covariance') is None:
            return None
        
        recent = np.array(list(self.history)[-20:])
        
        if recent.shape[1] < 2:
            return None
        
        # Calculate correlation matrix
        recent_corr = np.corrcoef(recent.T)
        baseline_cov = self.baseline_stats['covariance']
        baseline_std = np.sqrt(np.diag(baseline_cov))
        baseline_corr = baseline_cov / np.outer(baseline_std, baseline_std)
        
        # Find broken correlations
        broken = []
        for i in range(recent_corr.shape[0]):
            for j in range(i + 1, recent_corr.shape[1]):
                diff = abs(recent_corr[i, j] - baseline_corr[i, j])
                if diff > self.detection_thresholds['correlation_break']:
                    broken.append({
                        'features': (i, j),
                        'baseline': baseline_corr[i, j],
                        'current': recent_corr[i, j],
                        'difference': diff
                    })
        
        if broken:
            max_diff = max(b['difference'] for b in broken)
            return {
                'confidence': min(max_diff, 1.0),
                'evidence': {
                    'broken_correlations': broken,
                    'num_broken': len(broken)
                }
            }
        
        return None
    
    def _calculate_entropy(self, data: List[Any]) -> float:
        """Calculate Shannon entropy"""
        if not data:
            return 0.0
        
        # Count occurrences
        counts = {}
        for item in data:
            counts[item] = counts.get(item, 0) + 1
        
        # Calculate probabilities
        total = len(data)
        entropy = 0.0
        
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(counts))
        if max_entropy > 0:
            return entropy / max_entropy
        
        return 0.0
    
    def update_thresholds(self, false_positive_rate: float) -> None:
        """
        Update detection thresholds based on observed false positive rate.
        
        Args:
            false_positive_rate: Observed false positive rate
        """
        if false_positive_rate > 0.1:
            # Too many false positives, reduce sensitivity
            for key in self.detection_thresholds:
                if key == 'distribution_shift':
                    self.detection_thresholds[key] = min(0.1, self.detection_thresholds[key] * 1.5)
                else:
                    self.detection_thresholds[key] *= 1.2
        elif false_positive_rate < 0.01:
            # Too few detections, increase sensitivity
            for key in self.detection_thresholds:
                if key == 'distribution_shift':
                    self.detection_thresholds[key] = max(0.001, self.detection_thresholds[key] * 0.8)
                else:
                    self.detection_thresholds[key] *= 0.9
    
    def reset(self) -> None:
        """Reset detector state"""
        self.history.clear()
        self.baseline_stats = None
        self.detection_thresholds = self._initialize_thresholds()