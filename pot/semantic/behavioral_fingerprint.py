"""
Behavioral fingerprinting module for continuous model behavior monitoring.
Captures patterns of model outputs over time and creates behavioral fingerprints
for anomaly detection and drift monitoring.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Deque
from collections import deque
from dataclasses import dataclass, field
import time
import logging
import hashlib
from scipy import stats
from sklearn.decomposition import PCA
import warnings

from .match import SemanticMatcher
from .library import ConceptLibrary
from .utils import normalize_embeddings, compute_embedding_statistics

logger = logging.getLogger(__name__)


@dataclass
class BehaviorSnapshot:
    """Represents a single behavioral observation."""
    output: torch.Tensor
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age(self) -> float:
        """Age of snapshot in seconds since capture."""
        return time.time() - self.timestamp


@dataclass
class FingerprintHistory:
    """Stores historical fingerprints for drift detection."""
    fingerprints: List[torch.Tensor] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    max_history: int = 1000
    
    def add(self, fingerprint: torch.Tensor, timestamp: float = None, 
            metadata: Dict[str, Any] = None) -> None:
        """Add a fingerprint to history."""
        if timestamp is None:
            timestamp = time.time()
        
        self.fingerprints.append(fingerprint.clone())
        self.timestamps.append(timestamp)
        self.metadata.append(metadata or {})
        
        # Trim history if needed
        if len(self.fingerprints) > self.max_history:
            self.fingerprints = self.fingerprints[-self.max_history:]
            self.timestamps = self.timestamps[-self.max_history:]
            self.metadata = self.metadata[-self.max_history:]
    
    def get_recent(self, n: int = 10) -> List[torch.Tensor]:
        """Get n most recent fingerprints."""
        return self.fingerprints[-n:] if self.fingerprints else []
    
    def clear(self) -> None:
        """Clear history."""
        self.fingerprints.clear()
        self.timestamps.clear()
        self.metadata.clear()


class BehavioralFingerprint:
    """
    Captures and analyzes patterns of model outputs over time to create
    behavioral fingerprints for anomaly detection and monitoring.
    """
    
    def __init__(self, window_size: int = 100, fingerprint_dim: int = 256,
                 decay_factor: float = 0.95, use_pca: bool = True,
                 semantic_matcher: Optional[SemanticMatcher] = None):
        """
        Initialize the behavioral fingerprinting system.
        
        Args:
            window_size: Size of sliding window for behavioral observations
            fingerprint_dim: Dimensionality of fingerprint vectors
            decay_factor: Exponential decay factor for aging observations (0-1)
            use_pca: Whether to use PCA for dimensionality reduction
            semantic_matcher: Optional SemanticMatcher for enhanced analysis
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if fingerprint_dim <= 0:
            raise ValueError("fingerprint_dim must be positive")
        if not 0 < decay_factor <= 1:
            raise ValueError("decay_factor must be in (0, 1]")
        
        self.window_size = window_size
        self.fingerprint_dim = fingerprint_dim
        self.decay_factor = decay_factor
        self.use_pca = use_pca
        self.semantic_matcher = semantic_matcher
        
        # Sliding window of behavioral observations
        self.observation_window: Deque[BehaviorSnapshot] = deque(maxlen=window_size)
        
        # Fingerprint history for drift detection
        self.history = FingerprintHistory()
        
        # PCA for dimensionality reduction
        self.pca = None
        if use_pca:
            self.pca = PCA(n_components=min(fingerprint_dim, window_size))
        
        # Statistics for normalization
        self.running_mean = None
        self.running_std = None
        self.n_updates = 0
        
        # Reference fingerprint for anomaly detection
        self.reference_fingerprint = None
        self.reference_threshold = 0.9
        
        # Alert callbacks
        self.alert_callbacks = []
        
        logger.info(f"Initialized BehavioralFingerprint with window_size={window_size}, "
                   f"fingerprint_dim={fingerprint_dim}, decay_factor={decay_factor}")
    
    def update(self, output: torch.Tensor, timestamp: Optional[float] = None,
               metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update behavioral observations with new model output.
        
        Args:
            output: Model output tensor
            timestamp: Optional timestamp (uses current time if None)
            metadata: Optional metadata about the output
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Ensure output is a tensor
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output)
        
        # Flatten output if needed
        if output.dim() > 1:
            output = output.flatten()
        
        # Create snapshot
        snapshot = BehaviorSnapshot(
            output=output.clone().detach(),
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        # Add to observation window
        self.observation_window.append(snapshot)
        
        # Update running statistics
        self._update_statistics(output)
        
        self.n_updates += 1
        
        # Log significant updates
        if self.n_updates % 100 == 0:
            logger.debug(f"Behavioral fingerprint updated: {self.n_updates} total updates")
    
    def compute_fingerprint(self, normalize: bool = True) -> torch.Tensor:
        """
        Compute current behavioral fingerprint from observations.
        
        Args:
            normalize: Whether to normalize the fingerprint
            
        Returns:
            Fingerprint vector of shape (fingerprint_dim,)
        """
        if len(self.observation_window) == 0:
            # Return zero fingerprint if no observations
            return torch.zeros(self.fingerprint_dim)
        
        # Extract outputs and compute weights based on age
        outputs = []
        weights = []
        current_time = time.time()
        
        for snapshot in self.observation_window:
            age = current_time - snapshot.timestamp
            weight = self.decay_factor ** (age / 3600)  # Decay per hour
            
            outputs.append(snapshot.output)
            weights.append(weight)
        
        # Stack outputs into matrix
        output_matrix = torch.stack(outputs)  # (n_observations, output_dim)
        weights_tensor = torch.tensor(weights, dtype=output_matrix.dtype)
        
        # Apply weights
        weighted_outputs = output_matrix * weights_tensor.unsqueeze(1)
        
        # Compute statistical features
        features = self._compute_statistical_features(weighted_outputs, weights_tensor)
        
        # Reduce dimensionality if needed
        if features.shape[0] > self.fingerprint_dim:
            features = self._reduce_dimension(features)
        elif features.shape[0] < self.fingerprint_dim:
            # Pad with zeros if needed
            padding = torch.zeros(self.fingerprint_dim - features.shape[0])
            features = torch.cat([features, padding])
        
        # Normalize if requested
        if normalize:
            features = normalize_embeddings(features, method='l2')
        
        # Add to history
        self.history.add(features, current_time)
        
        return features
    
    def compare_fingerprints(self, fp1: torch.Tensor, fp2: torch.Tensor,
                            method: str = 'cosine') -> float:
        """
        Compare two behavioral fingerprints.
        
        Args:
            fp1: First fingerprint vector
            fp2: Second fingerprint vector
            method: Comparison method ('cosine', 'euclidean', 'correlation')
            
        Returns:
            Similarity score in [0, 1] where 1 is identical
        """
        # Ensure same dimensionality
        if fp1.shape != fp2.shape:
            raise ValueError(f"Fingerprint dimensions mismatch: {fp1.shape} vs {fp2.shape}")
        
        # Handle zero fingerprints
        if torch.allclose(fp1, torch.zeros_like(fp1)) or torch.allclose(fp2, torch.zeros_like(fp2)):
            return 0.0
        
        if method == 'cosine':
            # Cosine similarity
            fp1_norm = fp1 / (torch.norm(fp1) + 1e-8)
            fp2_norm = fp2 / (torch.norm(fp2) + 1e-8)
            similarity = torch.dot(fp1_norm, fp2_norm).item()
            # Map from [-1, 1] to [0, 1]
            similarity = (similarity + 1.0) / 2.0
            
        elif method == 'euclidean':
            # Euclidean distance converted to similarity
            distance = torch.norm(fp1 - fp2).item()
            max_distance = torch.norm(fp1).item() + torch.norm(fp2).item()
            similarity = 1.0 - (distance / (max_distance + 1e-8))
            
        elif method == 'correlation':
            # Pearson correlation
            fp1_np = fp1.detach().cpu().numpy()
            fp2_np = fp2.detach().cpu().numpy()
            
            if np.std(fp1_np) > 0 and np.std(fp2_np) > 0:
                correlation = np.corrcoef(fp1_np, fp2_np)[0, 1]
                # Map from [-1, 1] to [0, 1]
                similarity = (correlation + 1.0) / 2.0
            else:
                similarity = 0.5  # Neutral if no variance
            
        else:
            raise ValueError(f"Unknown comparison method: {method}")
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def detect_anomaly(self, current_fp: Optional[torch.Tensor] = None,
                       threshold: Optional[float] = None) -> Tuple[bool, float]:
        """
        Detect anomalous behavior based on fingerprint deviation.
        
        Args:
            current_fp: Current fingerprint (computes if None)
            threshold: Anomaly threshold (uses default if None)
            
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        if current_fp is None:
            current_fp = self.compute_fingerprint()
        
        if threshold is None:
            threshold = self.reference_threshold
        
        # Need reference fingerprint for comparison
        if self.reference_fingerprint is None:
            # Use first fingerprint as reference or compute from history
            if self.history.fingerprints:
                self.reference_fingerprint = torch.mean(
                    torch.stack(self.history.get_recent(10)), dim=0
                )
            else:
                logger.warning("No reference fingerprint available for anomaly detection")
                return False, 0.0
        
        # Compare with reference
        similarity = self.compare_fingerprints(current_fp, self.reference_fingerprint)
        anomaly_score = 1.0 - similarity
        
        is_anomaly = similarity < threshold
        
        # Check with semantic matcher if available
        if self.semantic_matcher and is_anomaly:
            # Additional semantic check
            semantic_matches = self.semantic_matcher.match_to_library(current_fp)
            if semantic_matches:
                best_match = max(semantic_matches.values())
                # Reduce anomaly score if semantically similar to known concepts
                anomaly_score *= (1.0 - best_match * 0.5)
                is_anomaly = anomaly_score > (1.0 - threshold)
        
        # Trigger alerts if anomaly detected
        if is_anomaly:
            self._trigger_alerts(anomaly_score, current_fp)
        
        return is_anomaly, float(anomaly_score)
    
    def detect_drift(self, window: int = 50, method: str = 'ks') -> Tuple[bool, float]:
        """
        Detect behavioral drift using statistical tests.
        
        Args:
            window: Window size for drift detection
            method: Detection method ('ks' for Kolmogorov-Smirnov, 'wasserstein')
            
        Returns:
            Tuple of (has_drift, drift_score)
        """
        if len(self.history.fingerprints) < window * 2:
            return False, 0.0
        
        # Get recent and reference windows
        recent_fps = self.history.fingerprints[-window:]
        reference_fps = self.history.fingerprints[-window*2:-window]
        
        # Convert to matrices
        recent_matrix = torch.stack(recent_fps).detach().cpu().numpy()
        reference_matrix = torch.stack(reference_fps).detach().cpu().numpy()
        
        if method == 'ks':
            # Kolmogorov-Smirnov test per dimension
            p_values = []
            for dim in range(recent_matrix.shape[1]):
                _, p_value = stats.ks_2samp(
                    reference_matrix[:, dim],
                    recent_matrix[:, dim]
                )
                p_values.append(p_value)
            
            # Combine p-values (Fisher's method)
            combined_stat = -2 * np.sum(np.log(np.array(p_values) + 1e-10))
            combined_p = 1 - stats.chi2.cdf(combined_stat, 2 * len(p_values))
            
            drift_score = 1.0 - combined_p
            has_drift = combined_p < 0.05
            
        elif method == 'wasserstein':
            # Wasserstein distance
            from scipy.stats import wasserstein_distance
            
            distances = []
            for dim in range(recent_matrix.shape[1]):
                dist = wasserstein_distance(
                    reference_matrix[:, dim],
                    recent_matrix[:, dim]
                )
                distances.append(dist)
            
            drift_score = np.mean(distances)
            has_drift = drift_score > 0.1  # Threshold
            
        else:
            raise ValueError(f"Unknown drift detection method: {method}")
        
        if has_drift:
            logger.warning(f"Behavioral drift detected: score={drift_score:.3f}")
        
        return has_drift, float(drift_score)
    
    def set_reference(self, fingerprint: Optional[torch.Tensor] = None,
                     threshold: float = 0.9) -> None:
        """
        Set reference fingerprint for anomaly detection.
        
        Args:
            fingerprint: Reference fingerprint (computes from history if None)
            threshold: Anomaly detection threshold
        """
        if fingerprint is None:
            if self.history.fingerprints:
                # Use mean of recent fingerprints
                fingerprint = torch.mean(
                    torch.stack(self.history.get_recent(20)), dim=0
                )
            else:
                # Compute from current observations
                fingerprint = self.compute_fingerprint()
        
        self.reference_fingerprint = fingerprint.clone()
        self.reference_threshold = threshold
        
        logger.info(f"Set reference fingerprint with threshold={threshold}")
    
    def register_alert_callback(self, callback: callable) -> None:
        """
        Register a callback for anomaly alerts.
        
        Args:
            callback: Function to call on anomaly detection
                     Should accept (anomaly_score, fingerprint, metadata)
        """
        self.alert_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics about behavioral fingerprinting.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'n_observations': len(self.observation_window),
            'n_updates': self.n_updates,
            'history_size': len(self.history.fingerprints),
            'window_size': self.window_size,
            'fingerprint_dim': self.fingerprint_dim,
        }
        
        if self.observation_window:
            ages = [s.age for s in self.observation_window]
            stats['observation_ages'] = {
                'min': min(ages),
                'max': max(ages),
                'mean': np.mean(ages)
            }
        
        if self.history.fingerprints:
            # Compute fingerprint diversity
            recent_fps = torch.stack(self.history.get_recent(10))
            pairwise_sims = []
            for i in range(len(recent_fps)):
                for j in range(i + 1, len(recent_fps)):
                    sim = self.compare_fingerprints(recent_fps[i], recent_fps[j])
                    pairwise_sims.append(sim)
            
            if pairwise_sims:
                stats['fingerprint_diversity'] = 1.0 - np.mean(pairwise_sims)
        
        # Check for drift
        has_drift, drift_score = self.detect_drift()
        stats['has_drift'] = has_drift
        stats['drift_score'] = drift_score
        
        return stats
    
    def reset(self) -> None:
        """Reset all observations and history."""
        self.observation_window.clear()
        self.history.clear()
        self.running_mean = None
        self.running_std = None
        self.n_updates = 0
        self.reference_fingerprint = None
        
        logger.info("Behavioral fingerprint system reset")
    
    def save_state(self, path: str) -> None:
        """
        Save fingerprint state to disk.
        
        Args:
            path: Path to save state
        """
        state = {
            'window_size': self.window_size,
            'fingerprint_dim': self.fingerprint_dim,
            'decay_factor': self.decay_factor,
            'n_updates': self.n_updates,
            'running_mean': self.running_mean,
            'running_std': self.running_std,
            'reference_fingerprint': self.reference_fingerprint,
            'reference_threshold': self.reference_threshold,
            'history': {
                'fingerprints': self.history.fingerprints,
                'timestamps': self.history.timestamps,
                'metadata': self.history.metadata
            }
        }
        
        torch.save(state, path)
        logger.info(f"Saved behavioral fingerprint state to {path}")
    
    def load_state(self, path: str) -> None:
        """
        Load fingerprint state from disk.
        
        Args:
            path: Path to load state from
        """
        state = torch.load(path, map_location='cpu')
        
        self.window_size = state['window_size']
        self.fingerprint_dim = state['fingerprint_dim']
        self.decay_factor = state['decay_factor']
        self.n_updates = state['n_updates']
        self.running_mean = state['running_mean']
        self.running_std = state['running_std']
        self.reference_fingerprint = state['reference_fingerprint']
        self.reference_threshold = state['reference_threshold']
        
        # Restore history
        self.history.fingerprints = state['history']['fingerprints']
        self.history.timestamps = state['history']['timestamps']
        self.history.metadata = state['history']['metadata']
        
        logger.info(f"Loaded behavioral fingerprint state from {path}")
    
    # Private methods
    
    def _update_statistics(self, output: torch.Tensor) -> None:
        """Update running statistics for normalization."""
        if self.running_mean is None:
            self.running_mean = output.clone()
            self.running_std = torch.zeros_like(output)
        else:
            # Exponential moving average
            alpha = 0.01
            self.running_mean = (1 - alpha) * self.running_mean + alpha * output
            
            variance = (output - self.running_mean) ** 2
            self.running_std = torch.sqrt(
                (1 - alpha) * self.running_std ** 2 + alpha * variance
            )
    
    def _compute_statistical_features(self, outputs: torch.Tensor, 
                                     weights: torch.Tensor) -> torch.Tensor:
        """
        Compute statistical features from weighted outputs.
        
        Args:
            outputs: Output matrix (n_observations, output_dim)
            weights: Weight vector (n_observations,)
            
        Returns:
            Feature vector
        """
        # Normalize weights
        weights = weights / (weights.sum() + 1e-8)
        
        # Weighted mean
        weighted_mean = torch.sum(outputs * weights.unsqueeze(1), dim=0)
        
        # Weighted standard deviation
        centered = outputs - weighted_mean
        weighted_var = torch.sum(centered ** 2 * weights.unsqueeze(1), dim=0)
        weighted_std = torch.sqrt(weighted_var + 1e-8)
        
        # Additional features
        features = [
            weighted_mean,
            weighted_std,
            torch.min(outputs, dim=0)[0],
            torch.max(outputs, dim=0)[0],
            torch.median(outputs, dim=0)[0]
        ]
        
        # Percentiles
        for p in [25, 75]:
            percentile = torch.quantile(outputs, p / 100.0, dim=0)
            features.append(percentile)
        
        # Concatenate all features
        feature_vector = torch.cat([f.flatten() for f in features])
        
        return feature_vector
    
    def _reduce_dimension(self, features: torch.Tensor) -> torch.Tensor:
        """
        Reduce feature dimensionality using PCA or other methods.
        
        Args:
            features: High-dimensional feature vector
            
        Returns:
            Reduced feature vector of size (fingerprint_dim,)
        """
        if self.use_pca and self.pca is not None:
            # Use PCA
            features_np = features.detach().cpu().numpy()
            
            # Fit PCA if needed
            if not hasattr(self.pca, 'components_'):
                # Need more samples to fit PCA
                if len(self.history.fingerprints) >= 10:
                    history_matrix = torch.stack(self.history.fingerprints[:10])
                    history_np = history_matrix.detach().cpu().numpy()
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.pca.fit(history_np)
                else:
                    # Use random projection as fallback
                    projection = torch.randn(features.shape[0], self.fingerprint_dim)
                    projection = projection / torch.norm(projection, dim=0)
                    return torch.matmul(features, projection)
            
            # Transform using PCA
            try:
                reduced = self.pca.transform(features_np.reshape(1, -1))[0]
                return torch.tensor(reduced, dtype=features.dtype)
            except:
                # Fallback to random projection
                projection = torch.randn(features.shape[0], self.fingerprint_dim)
                projection = projection / torch.norm(projection, dim=0)
                return torch.matmul(features, projection)
        else:
            # Simple linear projection
            if features.shape[0] > self.fingerprint_dim:
                # Use first fingerprint_dim dimensions
                return features[:self.fingerprint_dim]
            else:
                return features
    
    def _trigger_alerts(self, anomaly_score: float, fingerprint: torch.Tensor) -> None:
        """Trigger registered alert callbacks."""
        metadata = {
            'timestamp': time.time(),
            'n_observations': len(self.observation_window),
            'anomaly_score': anomaly_score
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(anomaly_score, fingerprint, metadata)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")


class ContinuousMonitor:
    """
    Continuous monitoring system that combines behavioral fingerprinting
    with semantic matching for real-time anomaly detection.
    """
    
    def __init__(self, fingerprint: BehavioralFingerprint,
                 semantic_matcher: Optional[SemanticMatcher] = None,
                 alert_threshold: float = 0.9,
                 check_interval: int = 10):
        """
        Initialize continuous monitor.
        
        Args:
            fingerprint: BehavioralFingerprint instance
            semantic_matcher: Optional SemanticMatcher for semantic analysis
            alert_threshold: Threshold for anomaly alerts
            check_interval: Check for anomalies every N updates
        """
        self.fingerprint = fingerprint
        self.semantic_matcher = semantic_matcher
        self.alert_threshold = alert_threshold
        self.check_interval = check_interval
        
        # Link semantic matcher to fingerprint
        if semantic_matcher:
            fingerprint.semantic_matcher = semantic_matcher
        
        # Monitoring state
        self.n_checks = 0
        self.anomaly_history = []
        self.drift_history = []
        
        # Register alert callback
        fingerprint.register_alert_callback(self._on_anomaly)
        
        logger.info(f"Initialized ContinuousMonitor with threshold={alert_threshold}")
    
    def process_output(self, output: torch.Tensor, 
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process model output and check for anomalies.
        
        Args:
            output: Model output to process
            metadata: Optional metadata
            
        Returns:
            Monitoring results dictionary
        """
        # Update fingerprint
        self.fingerprint.update(output, metadata=metadata)
        
        results = {
            'processed': True,
            'n_updates': self.fingerprint.n_updates
        }
        
        # Check for anomalies periodically
        if self.fingerprint.n_updates % self.check_interval == 0:
            self.n_checks += 1
            
            # Compute current fingerprint
            current_fp = self.fingerprint.compute_fingerprint()
            
            # Check for anomalies
            is_anomaly, anomaly_score = self.fingerprint.detect_anomaly(
                current_fp, self.alert_threshold
            )
            
            results['anomaly_check'] = {
                'is_anomaly': is_anomaly,
                'score': anomaly_score
            }
            
            if is_anomaly:
                self.anomaly_history.append({
                    'timestamp': time.time(),
                    'score': anomaly_score,
                    'fingerprint': current_fp
                })
            
            # Check for drift periodically
            if self.n_checks % 10 == 0:
                has_drift, drift_score = self.fingerprint.detect_drift()
                results['drift_check'] = {
                    'has_drift': has_drift,
                    'score': drift_score
                }
                
                if has_drift:
                    self.drift_history.append({
                        'timestamp': time.time(),
                        'score': drift_score
                    })
            
            # Semantic analysis if available
            if self.semantic_matcher:
                semantic_matches = self.semantic_matcher.match_to_library(current_fp)
                if semantic_matches:
                    results['semantic_matches'] = dict(list(semantic_matches.items())[:5])
        
        return results
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring results."""
        stats = self.fingerprint.get_statistics()
        
        summary = {
            'fingerprint_stats': stats,
            'n_checks': self.n_checks,
            'n_anomalies': len(self.anomaly_history),
            'n_drifts': len(self.drift_history),
            'recent_anomalies': self.anomaly_history[-5:] if self.anomaly_history else [],
            'recent_drifts': self.drift_history[-5:] if self.drift_history else []
        }
        
        if self.anomaly_history:
            anomaly_scores = [a['score'] for a in self.anomaly_history]
            summary['anomaly_stats'] = {
                'mean_score': np.mean(anomaly_scores),
                'max_score': np.max(anomaly_scores),
                'frequency': len(self.anomaly_history) / max(1, self.n_checks)
            }
        
        return summary
    
    def _on_anomaly(self, score: float, fingerprint: torch.Tensor, 
                   metadata: Dict[str, Any]) -> None:
        """Handle anomaly detection alert."""
        logger.warning(f"Anomaly detected: score={score:.3f}, metadata={metadata}")


def create_behavioral_monitor(window_size: int = 100,
                             fingerprint_dim: int = 256,
                             semantic_library: Optional[ConceptLibrary] = None,
                             **kwargs) -> ContinuousMonitor:
    """
    Create a behavioral monitoring system with optional semantic integration.
    
    Args:
        window_size: Observation window size
        fingerprint_dim: Fingerprint dimensionality
        semantic_library: Optional ConceptLibrary for semantic analysis
        **kwargs: Additional arguments for BehavioralFingerprint
        
    Returns:
        Configured ContinuousMonitor instance
    """
    # Create fingerprint system
    fingerprint = BehavioralFingerprint(
        window_size=window_size,
        fingerprint_dim=fingerprint_dim,
        **kwargs
    )
    
    # Create semantic matcher if library provided
    semantic_matcher = None
    if semantic_library:
        semantic_matcher = SemanticMatcher(
            library=semantic_library,
            threshold=0.7
        )
    
    # Create monitor
    monitor = ContinuousMonitor(
        fingerprint=fingerprint,
        semantic_matcher=semantic_matcher
    )
    
    return monitor