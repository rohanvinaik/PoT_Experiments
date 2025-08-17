"""
Wrapper Attack Detection for Proof-of-Training
Based on paper Section 5: Security Analysis and Adversary Models
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import time
from scipy import stats
from sklearn.ensemble import IsolationForest
import hashlib
import warnings
from collections import defaultdict


__all__ = ["detect_wrapper"]

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


@dataclass
class WrapperDetectionResult:
    """Result of wrapper attack detection"""

    is_wrapper: bool
    confidence: float
    anomaly_score: float
    timing_anomaly: bool
    consistency_score: float
    metadata: Dict[str, Any]


class WrapperAttackDetector:
    """
    Detect wrapper attacks where adversary wraps genuine model
    From paper Section 5.1: "A wrapper attack involves an adversary who has
    access to the genuine model f* and creates a wrapper that selectively
    routes challenges to f* while using a different model for regular inputs"
    """

    def __init__(
        self,
        sensitivity: float = 0.95,
        perfect_match_threshold: float = 1e-6,
        quality_drop_threshold: float = 0.5,
    ):
        """
        Initialize wrapper detector

        Args:
            sensitivity: Detection sensitivity (1 - false negative rate)
            perfect_match_threshold: Maximum absolute difference for
                numerical responses to be considered perfect
            quality_drop_threshold: Threshold for perturbed response quality
                below which behavior is flagged as suspicious
        """
        self.sensitivity = sensitivity
        self.perfect_match_threshold = perfect_match_threshold
        self.quality_drop_threshold = quality_drop_threshold
        self.timing_history = []
        self.response_history = []
        self.anomaly_detector = IsolationForest(contamination=1 - sensitivity)

    def detect_timing_anomaly(self, response_times: List[float]) -> Tuple[bool, float]:
        """
        Detect timing anomalies that may indicate wrapper routing

        Wrapper attacks often show bimodal timing distribution:
        - Fast responses for non-challenge inputs (local model)
        - Slower responses for challenges (routing to genuine model)
        """
        if len(response_times) < 10:
            return False, 0.0

        times = np.array(response_times)

        # Test for bimodality using Hartigan's dip test approximation
        sorted_times = np.sort(times)
        n = len(sorted_times)

        # Compute empirical CDF
        ecdf = np.arange(1, n + 1) / n

        # Find maximum deviation from uniform (unimodal approximation)
        uniform_cdf = (sorted_times - sorted_times[0]) / (sorted_times[-1] - sorted_times[0])
        max_deviation = np.max(np.abs(ecdf - uniform_cdf))

        # Threshold for anomaly (calibrated empirically)
        anomaly_threshold = 0.15
        is_anomaly = max_deviation > anomaly_threshold

        # Also check for suspiciously consistent timing
        cv = np.std(times) / np.mean(times)  # Coefficient of variation
        if cv < 0.05:  # Very low variation is suspicious
            is_anomaly = True
            max_deviation = 1.0

        return is_anomaly, float(max_deviation)

    def detect_response_inconsistency(
        self, challenge_responses: List[np.ndarray], regular_responses: List[np.ndarray]
    ) -> float:
        """
        Detect inconsistencies between challenge and regular responses

        Wrapper models may show different statistical properties between
        challenge responses (from genuine model) and regular responses
        """
        if not challenge_responses or not regular_responses:
            return 0.0

        # Convert to arrays
        chal_resp = np.array(challenge_responses)
        reg_resp = np.array(regular_responses)

        # Compute statistical properties
        chal_mean = np.mean(chal_resp, axis=0)
        chal_std = np.std(chal_resp, axis=0)
        reg_mean = np.mean(reg_resp, axis=0)
        reg_std = np.std(reg_resp, axis=0)

        # KL divergence approximation (assuming normal)
        kl_div = 0.0
        for i in range(min(len(chal_mean), len(reg_mean))):
            if chal_std[i] > 0 and reg_std[i] > 0:
                # KL divergence between two normals
                term1 = np.log(reg_std[i] / chal_std[i])
                term2 = (chal_std[i] ** 2 + (chal_mean[i] - reg_mean[i]) ** 2) / (
                    2 * reg_std[i] ** 2
                )
                kl_div += term1 + term2 - 0.5

        # Normalize
        kl_div = kl_div / max(len(chal_mean), 1)

        # High KL divergence indicates inconsistency
        return min(1.0, kl_div / 10.0)  # Normalize to [0, 1]

    def detect_challenge_memorization(
        self, challenges: List[Dict[str, Any]], responses: List[Any], model_eval_func: Callable
    ) -> float:
        """
        Detect if model has memorized specific challenges

        Tests by creating slight perturbations of challenges
        """
        if len(challenges) < 5:
            return 0.0

        memorization_scores = []

        for i, challenge in enumerate(challenges[:10]):  # Test subset
            original_response = responses[i]

            # Create perturbed versions
            perturbed_responses = []
            for _ in range(3):
                # Perturb challenge slightly
                perturbed = self._perturb_challenge(challenge)
                perturbed_response = model_eval_func(perturbed)
                perturbed_responses.append(perturbed_response)

            # Check if responses to perturbations are very different
            # while original is perfect
            if self._is_perfect_response(original_response):
                # Check if perturbed responses are much worse
                perturbed_quality = np.mean(
                    [self._response_quality(r) for r in perturbed_responses]
                )

                if perturbed_quality < self.quality_drop_threshold:
                    memorization_scores.append(1.0)
                else:
                    memorization_scores.append(0.0)
            else:
                memorization_scores.append(0.0)

        return np.mean(memorization_scores) if memorization_scores else 0.0

    def _perturb_challenge(self, challenge: Dict[str, Any]) -> Dict[str, Any]:
        """Create slight perturbation of challenge"""
        perturbed = challenge.copy()

        # Add small noise to numerical values
        for key, value in perturbed.items():
            if isinstance(value, (int, float)):
                noise = np.random.normal(0, 0.01)
                perturbed[key] = value + noise
            elif isinstance(value, str) and len(value) > 0:
                # Slightly modify strings
                chars = list(value)
                if len(chars) > 5:
                    # Swap two characters
                    i, j = np.random.choice(len(chars), 2, replace=False)
                    chars[i], chars[j] = chars[j], chars[i]
                    perturbed[key] = "".join(chars)

        return perturbed

    def _is_perfect_response(self, response: Any) -> bool:
        """Check if response exactly matches its reference."""

        if isinstance(response, dict):
            output = response.get("output")
            reference = response.get("reference")

            if isinstance(output, (int, float)) and isinstance(reference, (int, float)):
                return abs(output - reference) <= self.perfect_match_threshold

            return output == reference

        return False

    def _response_quality(self, response: Any) -> float:
        """Return normalized quality score in [0, 1]."""

        if isinstance(response, dict):
            output = response.get("output")
            reference = response.get("reference")

            if isinstance(output, (int, float)) and isinstance(reference, (int, float)):
                denom = max(abs(reference), self.perfect_match_threshold)
                error = abs(output - reference) / denom
                return max(0.0, 1.0 - error)

            return 1.0 if output == reference else 0.0

        return 0.0

    def detect_routing_patterns(
        self,
        request_sequence: List[Dict[str, Any]],
        response_sequence: List[Any],
        timing_sequence: List[float],
    ) -> float:
        """
        Detect routing patterns indicative of wrapper behavior

        Looks for:
        - Sudden timing changes for specific request types
        - Correlation between request properties and timing
        - Suspicious patterns in response ordering
        """
        if len(request_sequence) < 20:
            return 0.0

        routing_score = 0.0

        # Classify requests as challenge vs regular
        is_challenge = []
        for req in request_sequence:
            # Simple heuristic: challenges have specific structure
            is_chal = "challenge_id" in req or "template" in req
            is_challenge.append(is_chal)

        # Check if timing correlates with request type
        challenge_times = [t for t, c in zip(timing_sequence, is_challenge) if c]
        regular_times = [t for t, c in zip(timing_sequence, is_challenge) if not c]

        if challenge_times and regular_times:
            # T-test for different distributions
            t_stat, p_value = stats.ttest_ind(challenge_times, regular_times)

            if p_value < 0.05:  # Significant difference
                routing_score += 0.5

            # Check if challenge times are consistently higher
            if np.mean(challenge_times) > 1.5 * np.mean(regular_times):
                routing_score += 0.5

        return min(1.0, routing_score)

    def comprehensive_detection(
        self,
        challenge_responses: List[Any],
        regular_responses: List[Any],
        timing_data: List[float],
        request_sequence: List[Dict[str, Any]] = None,
    ) -> WrapperDetectionResult:
        """
        Comprehensive wrapper attack detection combining multiple signals

        Args:
            challenge_responses: Responses to verification challenges
            regular_responses: Responses to regular inputs
            timing_data: Response times for all requests
            request_sequence: Optional sequence of requests

        Returns:
            WrapperDetectionResult with detection outcome
        """
        # Timing anomaly detection
        timing_anomaly, timing_score = self.detect_timing_anomaly(timing_data)

        # Response inconsistency
        if challenge_responses and regular_responses:
            # Convert responses to numerical features if needed
            chal_features = self._extract_features(challenge_responses)
            reg_features = self._extract_features(regular_responses)
            inconsistency = self.detect_response_inconsistency(chal_features, reg_features)
        else:
            inconsistency = 0.0

        # Routing pattern detection
        if request_sequence:
            routing_score = self.detect_routing_patterns(
                request_sequence, challenge_responses + regular_responses, timing_data
            )
        else:
            routing_score = 0.0

        # Combine signals
        anomaly_score = timing_score * 0.4 + inconsistency * 0.4 + routing_score * 0.2

        # Make decision
        is_wrapper = anomaly_score > (1 - self.sensitivity)

        # Confidence based on signal strength
        confidence = min(1.0, anomaly_score * 1.2) if is_wrapper else 1.0 - anomaly_score

        metadata = {
            "timing_score": float(timing_score),
            "inconsistency_score": float(inconsistency),
            "routing_score": float(routing_score),
            "n_challenges": len(challenge_responses),
            "n_regular": len(regular_responses),
            "mean_timing": float(np.mean(timing_data)) if timing_data else 0.0,
            "timing_std": float(np.std(timing_data)) if timing_data else 0.0,
        }

        return WrapperDetectionResult(
            is_wrapper=is_wrapper,
            confidence=confidence,
            anomaly_score=anomaly_score,
            timing_anomaly=timing_anomaly,
            consistency_score=1.0 - inconsistency,
            metadata=metadata,
        )

    def _extract_features(self, responses: List[Any]) -> List[np.ndarray]:
        """Extract numerical features from responses"""
        features = []

        for response in responses:
            if isinstance(response, np.ndarray):
                features.append(response.flatten())
            elif isinstance(response, (list, tuple)):
                features.append(np.array(response))
            elif isinstance(response, str):
                # Hash string to get consistent numerical representation
                hash_val = int(hashlib.md5(response.encode()).hexdigest()[:8], 16)
                features.append(np.array([hash_val / 2**32]))
            else:
                # Default feature
                features.append(np.array([0.0]))

        # Pad to same length
        if features:
            max_len = max(len(f) for f in features)
            padded = []
            for f in features:
                if len(f) < max_len:
                    padded.append(np.pad(f, (0, max_len - len(f))))
                else:
                    padded.append(f)
            return padded

        return []


# Statistical Test Functions
def kolmogorov_smirnov_test(sample: np.ndarray, 
                           reference: np.ndarray) -> Tuple[float, float]:
    """
    KS test for distribution comparison.
    
    Args:
        sample: Sample data to test
        reference: Reference distribution data
        
    Returns:
        Tuple of (test_statistic, p_value)
    """
    try:
        ks_stat, p_value = stats.ks_2samp(sample, reference)
        return float(ks_stat), float(p_value)
    except Exception:
        return 0.0, 1.0


def anderson_darling_test(sample: np.ndarray,
                        reference: np.ndarray) -> Tuple[float, float]:
    """
    Anderson-Darling test for distribution comparison.
    
    Args:
        sample: Sample data to test
        reference: Reference distribution data
        
    Returns:
        Tuple of (test_statistic, p_value)
    """
    try:
        # Combine samples and sort
        combined = np.concatenate([sample, reference])
        combined_sorted = np.sort(combined)
        
        # Compute empirical CDFs
        n1, n2 = len(sample), len(reference)
        n = n1 + n2
        
        # Get ranks
        sample_ranks = np.searchsorted(combined_sorted, sample, side='right')
        reference_ranks = np.searchsorted(combined_sorted, reference, side='right')
        
        # Compute AD statistic
        h = np.zeros(n)
        h[sample_ranks - 1] = 1.0 / n1
        h[reference_ranks - 1] -= 1.0 / n2
        
        h_cumsum = np.cumsum(h)
        ad_stat = n * np.sum((h_cumsum[:-1] ** 2) / (np.arange(1, n) * (n - np.arange(1, n)) / n))
        
        # Approximate p-value (simplified)
        p_value = np.exp(-1.2337 * ad_stat * (1 + 4.0/n - 25.0/(n*n)))
        p_value = max(0.0, min(1.0, p_value))
        
        return float(ad_stat), float(p_value)
    except Exception:
        return 0.0, 1.0


def wasserstein_distance(sample: np.ndarray,
                        reference: np.ndarray) -> float:
    """
    Earth mover's distance between distributions.
    
    Args:
        sample: Sample data
        reference: Reference distribution data
        
    Returns:
        Wasserstein distance
    """
    try:
        distance = stats.wasserstein_distance(sample, reference)
        return float(distance)
    except Exception:
        return 0.0


class WrapperDetector:
    """
    Comprehensive wrapper detection system with baseline statistics and adaptive thresholds.
    """
    
    def __init__(self, baseline_stats: Dict[str, Any],
                 detection_thresholds: Dict[str, float] = None):
        """
        Initialize with baseline statistics from legitimate model.
        
        Args:
            baseline_stats: Reference timing/behavior statistics
            detection_thresholds: Configurable detection thresholds
        """
        self.baseline_stats = baseline_stats
        self.thresholds = detection_thresholds or self._default_thresholds()
        
        # Extract baseline timing distribution
        self.baseline_times = np.array(baseline_stats.get('response_times', []))
        self.baseline_responses = baseline_stats.get('responses', [])
        
        # Precompute baseline ECDF for efficiency
        if len(self.baseline_times) > 0:
            self.baseline_times_sorted = np.sort(self.baseline_times)
            self.baseline_ecdf_values = np.arange(1, len(self.baseline_times) + 1) / len(self.baseline_times)
        else:
            self.baseline_times_sorted = np.array([])
            self.baseline_ecdf_values = np.array([])
    
    def _default_thresholds(self) -> Dict[str, float]:
        """Default detection thresholds."""
        return {
            'timing_anomaly': 0.15,
            'behavioral_drift': 0.20,
            'ks_test_threshold': 0.05,
            'ad_test_threshold': 0.05,
            'wasserstein_threshold': 0.1,
            'ecdf_deviation_threshold': 0.15,
            'bimodal_threshold': 0.3,
            'confidence_threshold': 0.7
        }
    
    def detect_wrapper(self, 
                       response_times: List[float],
                       responses: List[torch.Tensor] if HAS_TORCH else List[np.ndarray],
                       metadata: Dict = None) -> Dict[str, Any]:
        """
        Main API for wrapper detection.
        
        Args:
            response_times: List of response times
            responses: List of model responses
            metadata: Optional metadata
            
        Returns:
            {
                'is_wrapped': bool,
                'confidence': float,
                'evidence': {
                    'timing_anomaly': float,
                    'ecdf_deviation': float,
                    'behavioral_drift': float,
                    'statistical_tests': dict
                }
            }
        """
        # Convert inputs to numpy arrays
        times = np.array(response_times)
        
        # Compute individual scores
        timing_score = self.compute_timing_score(times)
        behavioral_score = self.compute_behavioral_score(responses)
        
        # Statistical tests
        statistical_tests = self._run_statistical_tests(times, responses)
        
        # ECDF deviation
        ecdf_deviation = self._compute_ecdf_deviation(times)
        
        # Combine evidence
        evidence = {
            'timing_anomaly': timing_score,
            'ecdf_deviation': ecdf_deviation,
            'behavioral_drift': behavioral_score,
            'statistical_tests': statistical_tests
        }
        
        # Overall detection decision
        is_wrapped, confidence = self._make_detection_decision(evidence)
        
        return {
            'is_wrapped': is_wrapped,
            'confidence': confidence,
            'evidence': evidence,
            'metadata': metadata or {}
        }
    
    def compute_timing_score(self, times: List[float]) -> float:
        """
        Analyze timing patterns for wrapper signatures.
        
        Args:
            times: Response times to analyze
            
        Returns:
            Timing anomaly score (0-1, higher = more suspicious)
        """
        if len(times) < 5:
            return 0.0
        
        times_array = np.array(times)
        
        # 1. Check for bimodal distribution
        bimodal_score = self._detect_bimodal_timing(times_array)
        
        # 2. Check coefficient of variation (too consistent is suspicious)
        cv = np.std(times_array) / (np.mean(times_array) + 1e-8)
        consistency_score = 1.0 if cv < 0.05 else 0.0
        
        # 3. Check for outliers indicating routing delays
        q75, q25 = np.percentile(times_array, [75, 25])
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr
        outlier_ratio = np.sum(times_array > outlier_threshold) / len(times_array)
        outlier_score = min(1.0, outlier_ratio * 5)  # Scale to [0,1]
        
        # 4. Compare with baseline if available
        baseline_score = 0.0
        if len(self.baseline_times) > 0:
            ks_stat, ks_p = kolmogorov_smirnov_test(times_array, self.baseline_times)
            baseline_score = 1.0 if ks_p < self.thresholds['ks_test_threshold'] else 0.0
        
        # Combine scores
        overall_score = (
            bimodal_score * 0.4 +
            consistency_score * 0.2 +
            outlier_score * 0.2 +
            baseline_score * 0.2
        )
        
        return min(1.0, overall_score)
    
    def _detect_bimodal_timing(self, times: np.ndarray) -> float:
        """Detect bimodal distribution in timing data."""
        if len(times) < 10:
            return 0.0
        
        # Use Hartigan's dip test approximation
        sorted_times = np.sort(times)
        n = len(sorted_times)
        
        # Compute uniform distribution for comparison
        uniform_spacing = (sorted_times[-1] - sorted_times[0]) / (n - 1)
        expected_times = sorted_times[0] + np.arange(n) * uniform_spacing
        
        # Compare actual vs expected using ECDF
        actual_ecdf = np.arange(1, n + 1) / n
        expected_ecdf = (sorted_times - sorted_times[0]) / (sorted_times[-1] - sorted_times[0])
        
        max_deviation = np.max(np.abs(actual_ecdf - expected_ecdf))
        
        return 1.0 if max_deviation > self.thresholds['bimodal_threshold'] else 0.0
    
    def compute_behavioral_score(self, responses: List) -> float:
        """
        Analyze response patterns for wrapper artifacts.
        
        Args:
            responses: List of model responses
            
        Returns:
            Behavioral anomaly score (0-1, higher = more suspicious)
        """
        if len(responses) < 5:
            return 0.0
        
        # Convert responses to numerical features
        features = self._extract_response_features(responses)
        
        if len(features) == 0:
            return 0.0
        
        features_array = np.array(features)
        
        # 1. Check output variance consistency
        variance_score = self._check_output_variance(features_array)
        
        # 2. Check for clustering in response space
        clustering_score = self._detect_response_clustering(features_array)
        
        # 3. Compare with baseline responses if available
        baseline_score = 0.0
        if self.baseline_responses:
            baseline_features = self._extract_response_features(self.baseline_responses)
            if len(baseline_features) > 0:
                baseline_array = np.array(baseline_features)
                baseline_score = self._compare_response_distributions(features_array, baseline_array)
        
        # Combine scores
        overall_score = (
            variance_score * 0.4 +
            clustering_score * 0.3 +
            baseline_score * 0.3
        )
        
        return min(1.0, overall_score)
    
    def _extract_response_features(self, responses: List) -> List[np.ndarray]:
        """Extract numerical features from responses."""
        features = []
        
        for response in responses:
            if HAS_TORCH and torch.is_tensor(response):
                features.append(response.detach().cpu().numpy().flatten())
            elif isinstance(response, np.ndarray):
                features.append(response.flatten())
            elif isinstance(response, (list, tuple)):
                features.append(np.array(response).flatten())
            elif isinstance(response, (int, float)):
                features.append(np.array([response]))
            elif isinstance(response, str):
                # Hash string for consistent numerical representation
                hash_val = int(hashlib.md5(response.encode()).hexdigest()[:8], 16)
                features.append(np.array([hash_val / 2**32]))
            else:
                features.append(np.array([0.0]))
        
        # Normalize feature lengths
        if features:
            max_len = max(len(f) for f in features)
            normalized_features = []
            for f in features:
                if len(f) < max_len:
                    padded = np.pad(f, (0, max_len - len(f)), mode='constant')
                    normalized_features.append(padded)
                else:
                    normalized_features.append(f[:max_len])
            return normalized_features
        
        return []
    
    def _check_output_variance(self, features: np.ndarray) -> float:
        """Check if output variance is suspiciously low or high."""
        if features.shape[0] < 3:
            return 0.0
        
        # Compute variance across samples for each feature dimension
        variances = np.var(features, axis=0)
        mean_variance = np.mean(variances)
        
        # Very low variance suggests memorization/caching
        # Very high variance suggests random responses
        if mean_variance < 1e-6:
            return 1.0  # Suspiciously low variance
        elif mean_variance > 1000:
            return 0.8  # Suspiciously high variance
        else:
            return 0.0
    
    def _detect_response_clustering(self, features: np.ndarray) -> float:
        """Detect if responses cluster in suspicious ways."""
        if features.shape[0] < 10:
            return 0.0
        
        try:
            from sklearn.cluster import KMeans
            
            # Try k=2 clustering to detect bimodal responses
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            # Check cluster balance
            cluster_sizes = np.bincount(labels)
            min_cluster_size = np.min(cluster_sizes)
            max_cluster_size = np.max(cluster_sizes)
            
            # Suspicious if one cluster is very small (< 10% of data)
            if min_cluster_size / len(features) < 0.1:
                return 0.8
            
            # Suspicious if clusters are very separated
            centroids = kmeans.cluster_centers_
            centroid_distance = np.linalg.norm(centroids[0] - centroids[1])
            
            # Normalize by average intra-cluster distance
            intra_cluster_distances = []
            for i in range(2):
                cluster_points = features[labels == i]
                if len(cluster_points) > 1:
                    centroid = centroids[i]
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    intra_cluster_distances.extend(distances)
            
            if intra_cluster_distances:
                avg_intra_distance = np.mean(intra_cluster_distances)
                separation_ratio = centroid_distance / (avg_intra_distance + 1e-8)
                
                if separation_ratio > 5.0:  # Very well separated clusters
                    return 0.6
            
            return 0.0
            
        except ImportError:
            # Fallback without sklearn
            return 0.0
    
    def _compare_response_distributions(self, current: np.ndarray, baseline: np.ndarray) -> float:
        """Compare current responses with baseline distribution."""
        # Flatten for distribution comparison
        current_flat = current.flatten()
        baseline_flat = baseline.flatten()
        
        # KS test
        ks_stat, ks_p = kolmogorov_smirnov_test(current_flat, baseline_flat)
        
        # Wasserstein distance
        w_distance = wasserstein_distance(current_flat, baseline_flat)
        
        # Anderson-Darling test
        ad_stat, ad_p = anderson_darling_test(current_flat, baseline_flat)
        
        # Combine test results
        ks_score = 1.0 if ks_p < self.thresholds['ks_test_threshold'] else 0.0
        ad_score = 1.0 if ad_p < self.thresholds['ad_test_threshold'] else 0.0
        w_score = 1.0 if w_distance > self.thresholds['wasserstein_threshold'] else 0.0
        
        return (ks_score + ad_score + w_score) / 3.0
    
    def _compute_ecdf_deviation(self, times: np.ndarray) -> float:
        """Compute ECDF deviation from baseline."""
        if len(self.baseline_times_sorted) == 0 or len(times) == 0:
            return 0.0
        
        # Compute ECDF for current times
        times_sorted = np.sort(times)
        current_ecdf = np.arange(1, len(times) + 1) / len(times)
        
        # Interpolate baseline ECDF at current time points
        baseline_ecdf_interp = np.interp(times_sorted, self.baseline_times_sorted, self.baseline_ecdf_values)
        
        # Compute maximum deviation
        max_deviation = np.max(np.abs(current_ecdf - baseline_ecdf_interp))
        
        return max_deviation
    
    def _run_statistical_tests(self, times: np.ndarray, responses: List) -> Dict[str, Dict[str, float]]:
        """Run comprehensive statistical tests."""
        tests = {}
        
        # Timing tests against baseline
        if len(self.baseline_times) > 0:
            ks_stat, ks_p = kolmogorov_smirnov_test(times, self.baseline_times)
            ad_stat, ad_p = anderson_darling_test(times, self.baseline_times)
            w_dist = wasserstein_distance(times, self.baseline_times)
            
            tests['timing'] = {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'ad_statistic': ad_stat,
                'ad_p_value': ad_p,
                'wasserstein_distance': w_dist
            }
        
        # Response tests against baseline
        if self.baseline_responses:
            response_features = self._extract_response_features(responses)
            baseline_features = self._extract_response_features(self.baseline_responses)
            
            if response_features and baseline_features:
                current_flat = np.array(response_features).flatten()
                baseline_flat = np.array(baseline_features).flatten()
                
                ks_stat, ks_p = kolmogorov_smirnov_test(current_flat, baseline_flat)
                ad_stat, ad_p = anderson_darling_test(current_flat, baseline_flat)
                w_dist = wasserstein_distance(current_flat, baseline_flat)
                
                tests['responses'] = {
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                    'ad_statistic': ad_stat,
                    'ad_p_value': ad_p,
                    'wasserstein_distance': w_dist
                }
        
        return tests
    
    def _make_detection_decision(self, evidence: Dict[str, Any]) -> Tuple[bool, float]:
        """Make final detection decision based on evidence."""
        # Weighted combination of evidence
        weights = {
            'timing_anomaly': 0.3,
            'ecdf_deviation': 0.2,
            'behavioral_drift': 0.3,
            'statistical_significance': 0.2
        }
        
        # Compute statistical significance score
        stat_tests = evidence.get('statistical_tests', {})
        stat_score = 0.0
        
        for test_name, test_results in stat_tests.items():
            ks_significant = test_results.get('ks_p_value', 1.0) < self.thresholds['ks_test_threshold']
            ad_significant = test_results.get('ad_p_value', 1.0) < self.thresholds['ad_test_threshold']
            w_significant = test_results.get('wasserstein_distance', 0.0) > self.thresholds['wasserstein_threshold']
            
            test_score = (int(ks_significant) + int(ad_significant) + int(w_significant)) / 3.0
            stat_score = max(stat_score, test_score)
        
        # Compute overall score
        overall_score = (
            evidence['timing_anomaly'] * weights['timing_anomaly'] +
            evidence['ecdf_deviation'] * weights['ecdf_deviation'] +
            evidence['behavioral_drift'] * weights['behavioral_drift'] +
            stat_score * weights['statistical_significance']
        )
        
        # Make decision
        is_wrapped = overall_score > self.thresholds['confidence_threshold']
        confidence = min(1.0, overall_score * 1.2) if is_wrapped else 1.0 - overall_score
        
        return is_wrapped, confidence


class AdaptiveThresholdManager:
    """
    Adaptive threshold management for wrapper detection.
    """
    
    def __init__(self, initial_thresholds: Dict[str, float] = None):
        """Initialize with optional initial thresholds."""
        self.thresholds = initial_thresholds or {}
        self.performance_history = defaultdict(list)
        self.labeled_data = []
        
    def update_thresholds(self, new_data: Dict[str, Any], 
                         labels: np.ndarray) -> None:
        """
        Update detection thresholds based on new labeled data.
        
        Args:
            new_data: Dictionary with detection scores for each sample
            labels: Binary labels (1 = wrapper, 0 = legitimate)
        """
        self.labeled_data.append((new_data, labels))
        
        # Keep only recent data (last 1000 samples)
        if len(self.labeled_data) > 1000:
            self.labeled_data = self.labeled_data[-1000:]
        
        # Recompute optimal thresholds
        self._optimize_thresholds()
    
    def _optimize_thresholds(self):
        """Optimize thresholds using labeled data."""
        if len(self.labeled_data) < 10:
            return
        
        # Combine all labeled data
        all_scores = defaultdict(list)
        all_labels = []
        
        for data, labels in self.labeled_data:
            for key, scores in data.items():
                if isinstance(scores, (list, np.ndarray)):
                    all_scores[key].extend(scores)
                else:
                    all_scores[key].append(scores)
            all_labels.extend(labels)
        
        all_labels = np.array(all_labels)
        
        # Optimize threshold for each score type
        for score_name, scores in all_scores.items():
            scores_array = np.array(scores)
            
            if len(scores_array) == len(all_labels):
                optimal_threshold = self._find_optimal_threshold(scores_array, all_labels)
                self.thresholds[score_name] = optimal_threshold
    
    def _find_optimal_threshold(self, scores: np.ndarray, labels: np.ndarray, 
                               target_far: float = 0.05) -> float:
        """Find optimal threshold for given target false acceptance rate."""
        if len(np.unique(labels)) < 2:
            return 0.5  # Default if no class diversity
        
        # Try different thresholds
        thresholds = np.linspace(np.min(scores), np.max(scores), 100)
        best_threshold = 0.5
        best_score = -np.inf
        
        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            
            # Compute metrics
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            tn = np.sum((predictions == 0) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            if tp + fn == 0:  # No positive samples
                continue
            if tn + fp == 0:  # No negative samples
                continue
            
            tpr = tp / (tp + fn)  # True positive rate
            fpr = fp / (fp + tn)  # False positive rate
            
            # Objective: maximize TPR while keeping FPR <= target_far
            if fpr <= target_far:
                score = tpr
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        
        return best_threshold
    
    def get_optimal_threshold(self, score_name: str, target_far: float = 0.01) -> float:
        """
        Get optimal threshold for target false acceptance rate.
        
        Args:
            score_name: Name of the score to get threshold for
            target_far: Target false acceptance rate
            
        Returns:
            Optimal threshold value
        """
        if score_name in self.thresholds:
            return self.thresholds[score_name]
        
        # Return conservative default
        return 0.7
    
    def get_all_thresholds(self) -> Dict[str, float]:
        """Get all current thresholds."""
        return self.thresholds.copy()


class AdversarySimulator:
    """
    Simulate various adversary strategies for testing
    Based on adversary models from paper Section 5.2
    """

    def __init__(self, genuine_model: Callable, adversary_type: str = "wrapper"):
        """
        Initialize adversary simulator

        Args:
            genuine_model: The genuine model f*
            adversary_type: Type of adversary to simulate
        """
        self.genuine_model = genuine_model
        self.adversary_type = adversary_type
        self.routing_delay = 0.1  # Additional delay for routing

    def wrapper_adversary(self, input_data: Any, is_challenge: bool = False) -> Tuple[Any, float]:
        """
        Simulate wrapper adversary behavior

        Routes challenges to genuine model, uses fake model otherwise
        """
        start_time = time.time()

        if is_challenge:
            # Route to genuine model (with added network delay)
            time.sleep(self.routing_delay)
            response = self.genuine_model(input_data)
        else:
            # Use fake model (fast local response)
            response = self._fake_model(input_data)

        elapsed = time.time() - start_time
        return response, elapsed

    def extraction_adversary(self, input_data: Any) -> Tuple[Any, float]:
        """
        Simulate model extraction adversary

        Has imperfect copy of genuine model
        """
        start_time = time.time()

        # Use extracted model (imperfect approximation)
        genuine_response = self.genuine_model(input_data)

        # Add noise to simulate imperfect extraction
        if isinstance(genuine_response, np.ndarray):
            noise = np.random.normal(0, 0.05, genuine_response.shape)
            response = genuine_response + noise
        else:
            response = genuine_response

        elapsed = time.time() - start_time
        return response, elapsed

    def adaptive_adversary(self, input_data: Any, detection_risk: float = 0.5) -> Tuple[Any, float]:
        """
        Simulate adaptive adversary that adjusts behavior

        Tries to avoid detection by varying strategies
        """
        start_time = time.time()

        # Randomly decide strategy based on detection risk
        if np.random.random() < detection_risk:
            # Use genuine model to reduce detection risk
            response = self.genuine_model(input_data)
            # Add realistic delay variation
            time.sleep(np.random.uniform(0.05, 0.15))
        else:
            # Use fake model for efficiency
            response = self._fake_model(input_data)

        elapsed = time.time() - start_time
        return response, elapsed

    def _fake_model(self, input_data: Any) -> Any:
        """Simple fake model for non-challenge inputs"""
        # Return random response of appropriate type
        if isinstance(input_data, np.ndarray):
            return np.random.randn(*input_data.shape)
        elif isinstance(input_data, str):
            return "Fake response to: " + input_data[:20]
        else:
            return None

    def generate_attack_sequence(
        self, n_requests: int = 100, challenge_ratio: float = 0.1
    ) -> Dict[str, List]:
        """
        Generate sequence of adversarial interactions

        Returns:
            Dictionary with requests, responses, timings, and labels
        """
        requests = []
        responses = []
        timings = []
        is_challenge_flags = []

        for i in range(n_requests):
            # Randomly determine if this is a challenge
            is_challenge = np.random.random() < challenge_ratio

            # Generate appropriate request
            if is_challenge:
                request = {"challenge_id": f"chal_{i}", "data": np.random.randn(10)}
            else:
                request = {"query": f"regular_{i}", "data": np.random.randn(10)}

            # Get adversarial response based on type
            if self.adversary_type == "wrapper":
                response, timing = self.wrapper_adversary(request["data"], is_challenge)
            elif self.adversary_type == "extraction":
                response, timing = self.extraction_adversary(request["data"])
            elif self.adversary_type == "adaptive":
                response, timing = self.adaptive_adversary(request["data"])
            else:
                response, timing = self.genuine_model(request["data"]), 0.05

            requests.append(request)
            responses.append(response)
            timings.append(timing)
            is_challenge_flags.append(is_challenge)

        return {
            "requests": requests,
            "responses": responses,
            "timings": timings,
            "is_challenge": is_challenge_flags,
            "adversary_type": self.adversary_type,
        }
