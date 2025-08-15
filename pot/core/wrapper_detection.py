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
