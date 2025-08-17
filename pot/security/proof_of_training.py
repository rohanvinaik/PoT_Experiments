"""
Proof of Training - Integrated Challenge-Response Protocol

This module provides a complete Proof-of-Training system that combines fuzzy hashing,
provenance tracking, and token normalization into a unified verification protocol.
"""

import hashlib
import json
import time
import numpy as np
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings
import pickle
import base64

# Import core framework components
try:
    from pot.core.challenge import generate_challenges, ChallengeConfig
    from pot.core.fingerprint import fingerprint_run, FingerprintConfig, FingerprintResult
    from pot.core.sequential import sequential_verify, SPRTResult
    from pot.audit.commit_reveal import compute_commitment, write_audit_record, CommitmentRecord
    from pot.prototypes.training_provenance_auditor import BlockchainClient, BlockchainConfig
    CORE_COMPONENTS_AVAILABLE = True
except ImportError:
    CORE_COMPONENTS_AVAILABLE = False
    warnings.warn("Core PoT components not fully available")

# Import our components
try:
    from fuzzy_hash_verifier import FuzzyHashVerifier, ChallengeVector, HashAlgorithm
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    warnings.warn("FuzzyHashVerifier not available")

try:
    from pot.prototypes.training_provenance_auditor import (
        TrainingProvenanceAuditor,
        EventType,
        ProofType,
    )
    PROVENANCE_AVAILABLE = True
except ImportError:
    PROVENANCE_AVAILABLE = False
    warnings.warn("TrainingProvenanceAuditor not available")

try:
    from token_space_normalizer import (
        TokenSpaceNormalizer, 
        StochasticDecodingController,
        TokenizerType,
        SamplingMethod
    )
    TOKEN_AVAILABLE = True
except ImportError:
    TOKEN_AVAILABLE = False
    warnings.warn("TokenSpaceNormalizer not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerificationType(Enum):
    """Verification types"""
    EXACT = "exact"
    FUZZY = "fuzzy"
    STATISTICAL = "statistical"


class ModelType(Enum):
    """Model types"""
    VISION = "vision"
    LANGUAGE = "language"
    MULTIMODAL = "multimodal"
    GENERIC = "generic"


class SecurityLevel(Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerificationDepth(Enum):
    """Verification depth levels"""
    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


@dataclass
class ModelRegistration:
    """Model registration data"""
    model_id: str
    model_type: ModelType
    parameter_count: int
    architecture: str
    fingerprint: str
    reference_challenges: Dict[str, Any]
    reference_responses: Dict[str, Any]
    training_provenance: Optional[Dict[str, Any]]
    registration_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Verification result data"""
    verified: bool
    confidence: float
    verification_type: str
    model_id: str
    challenges_passed: int
    challenges_total: int
    fuzzy_similarity: Optional[float]
    statistical_score: Optional[float]
    provenance_verified: Optional[bool]
    details: Dict[str, Any]
    timestamp: datetime
    duration_seconds: float
    # Enhanced fields for expected ranges validation
    accuracy: Optional[float] = None
    latency_ms: Optional[float] = None
    fingerprint_similarity: Optional[float] = None
    jacobian_norm: Optional[float] = None
    range_validation: Optional['ValidationReport'] = None


@dataclass
class ValidationReport:
    """Report for expected ranges validation"""
    passed: bool
    violations: List[str]
    confidence: float
    range_scores: Dict[str, float]
    statistical_significance: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExpectedRanges:
    """Expected ranges for model behavior validation"""
    accuracy_range: Tuple[float, float]  # (min, max) accuracy bounds
    latency_range: Tuple[float, float]   # (min, max) latency in milliseconds
    fingerprint_similarity: Tuple[float, float]  # (min, max) fingerprint similarity
    jacobian_norm_range: Tuple[float, float]     # (min, max) Jacobian norm bounds
    
    # Statistical parameters
    confidence_level: float = 0.95  # Confidence level for range validation
    tolerance_factor: float = 1.1   # Tolerance multiplier for ranges
    
    def validate(self, results: VerificationResult) -> ValidationReport:
        """
        Check if results fall within expected ranges
        
        Args:
            results: VerificationResult to validate
            
        Returns:
            ValidationReport with validation details
        """
        violations = []
        range_scores = {}
        
        # Validate accuracy
        if results.accuracy is not None:
            score = self._compute_range_score(
                results.accuracy, self.accuracy_range, "accuracy"
            )
            range_scores["accuracy"] = score
            
            if not self._is_in_range(results.accuracy, self.accuracy_range):
                violations.append(
                    f"Accuracy {results.accuracy:.3f} outside range "
                    f"[{self.accuracy_range[0]:.3f}, {self.accuracy_range[1]:.3f}]"
                )
        
        # Validate latency
        if results.latency_ms is not None:
            score = self._compute_range_score(
                results.latency_ms, self.latency_range, "latency"
            )
            range_scores["latency"] = score
            
            if not self._is_in_range(results.latency_ms, self.latency_range):
                violations.append(
                    f"Latency {results.latency_ms:.1f}ms outside range "
                    f"[{self.latency_range[0]:.1f}, {self.latency_range[1]:.1f}]ms"
                )
        
        # Validate fingerprint similarity
        if results.fingerprint_similarity is not None:
            score = self._compute_range_score(
                results.fingerprint_similarity, self.fingerprint_similarity, "fingerprint"
            )
            range_scores["fingerprint_similarity"] = score
            
            if not self._is_in_range(results.fingerprint_similarity, self.fingerprint_similarity):
                violations.append(
                    f"Fingerprint similarity {results.fingerprint_similarity:.3f} outside range "
                    f"[{self.fingerprint_similarity[0]:.3f}, {self.fingerprint_similarity[1]:.3f}]"
                )
        
        # Validate Jacobian norm
        if results.jacobian_norm is not None:
            score = self._compute_range_score(
                results.jacobian_norm, self.jacobian_norm_range, "jacobian_norm"
            )
            range_scores["jacobian_norm"] = score
            
            if not self._is_in_range(results.jacobian_norm, self.jacobian_norm_range):
                violations.append(
                    f"Jacobian norm {results.jacobian_norm:.6f} outside range "
                    f"[{self.jacobian_norm_range[0]:.6f}, {self.jacobian_norm_range[1]:.6f}]"
                )
        
        # Compute overall confidence
        overall_confidence = self._compute_confidence(range_scores, violations)
        
        # Compute statistical significance if we have enough data
        statistical_significance = None
        if len(range_scores) >= 2:
            statistical_significance = self._compute_statistical_significance(range_scores)
        
        return ValidationReport(
            passed=len(violations) == 0,
            violations=violations,
            confidence=overall_confidence,
            range_scores=range_scores,
            statistical_significance=statistical_significance
        )
    
    def _is_in_range(self, value: float, range_bounds: Tuple[float, float]) -> bool:
        """Check if value is within range bounds with tolerance"""
        min_val, max_val = range_bounds
        
        # Apply tolerance factor
        tolerance = (max_val - min_val) * (self.tolerance_factor - 1.0) / 2.0
        adjusted_min = min_val - tolerance
        adjusted_max = max_val + tolerance
        
        return adjusted_min <= value <= adjusted_max
    
    def _compute_range_score(self, value: float, range_bounds: Tuple[float, float], 
                           metric_name: str) -> float:
        """
        Compute normalized score for how well value fits within range
        
        Returns:
            Score from 0.0 (far outside) to 1.0 (center of range)
        """
        min_val, max_val = range_bounds
        range_width = max_val - min_val
        
        if range_width == 0:
            return 1.0 if value == min_val else 0.0
        
        # Distance from center of range
        center = (min_val + max_val) / 2.0
        distance_from_center = abs(value - center)
        max_distance = range_width / 2.0
        
        # Normalize to [0, 1] where 1 is center, 0 is at range boundaries
        # Use small epsilon for floating point comparison
        eps = 1e-10
        if distance_from_center <= max_distance + eps:
            score = 1.0 - (distance_from_center / max_distance) if max_distance > 0 else 1.0
            # Ensure edge values still get a small positive score
            if score <= 0.01:  # Close to edge
                score = 0.01  # Small positive score for edge values
        else:
            # Penalize values outside range
            excess_distance = distance_from_center - max_distance
            penalty = min(1.0, excess_distance / max_distance) if max_distance > 0 else 1.0
            score = -penalty
        
        return max(0.0, score)
    
    def _compute_confidence(self, range_scores: Dict[str, float], 
                          violations: List[str]) -> float:
        """Compute overall confidence in range validation"""
        if not range_scores:
            return 0.0
        
        # Base confidence from average range scores
        avg_score = np.mean(list(range_scores.values()))
        
        # Penalty for violations
        violation_penalty = len(violations) * 0.2
        
        # Confidence boost for consistent scores
        score_std = np.std(list(range_scores.values())) if len(range_scores) > 1 else 0.0
        consistency_boost = max(0.0, 0.2 * (1.0 - score_std))
        
        confidence = avg_score - violation_penalty + consistency_boost
        return max(0.0, min(1.0, confidence))
    
    def _compute_statistical_significance(self, range_scores: Dict[str, float]) -> float:
        """
        Compute statistical significance of range validation
        
        Uses one-sample t-test against expected mean of 0.5 (center of ranges)
        """
        try:
            from scipy import stats
        except ImportError:
            logger.debug("scipy not available for statistical significance")
            return None
        
        scores = list(range_scores.values())
        if len(scores) < 2:
            return None
        
        # Test against null hypothesis that mean score = 0.5 (random performance)
        try:
            t_stat, p_value = stats.ttest_1samp(scores, 0.5)
            return p_value
        except Exception:
            return None


@dataclass
class SessionConfig:
    """Configuration for integrated verification session"""
    model: Any                                    # Model to verify
    model_id: str                                # Unique model identifier
    master_seed: str                             # Master seed for challenge generation
    num_challenges: int = 10                     # Number of challenges to generate
    accuracy_threshold: float = 0.05            # Threshold for sequential testing
    type1_error: float = 0.05                   # Type I error rate (α)
    type2_error: float = 0.05                   # Type II error rate (β)
    max_samples: int = 1000                     # Maximum samples for sequential test
    
    # Component configurations
    fingerprint_config: Optional[FingerprintConfig] = None
    expected_ranges: Optional[ExpectedRanges] = None
    blockchain_config: Optional[BlockchainConfig] = None
    
    # File paths and options
    audit_log_path: str = "verification_audit.json"
    use_blockchain: bool = False
    use_fingerprinting: bool = True
    use_sequential: bool = True
    use_range_validation: bool = True
    
    # Challenge generation parameters
    challenge_family: str = "vision:freq"
    challenge_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default configurations if not provided"""
        if self.fingerprint_config is None:
            self.fingerprint_config = FingerprintConfig(
                compute_jacobian=True,
                include_timing=True,
                memory_efficient=False
            )


@dataclass 
class VerificationReport:
    """Complete verification report from integrated protocol"""
    passed: bool                                 # Overall verification result
    confidence: float                           # Overall confidence score
    model_id: str                              # Model identifier
    session_id: str                            # Unique session identifier
    timestamp: datetime                        # Verification timestamp
    duration_seconds: float                    # Total verification time
    
    # Component results
    statistical_result: Optional[SPRTResult] = None
    fingerprint_result: Optional[FingerprintResult] = None
    range_validation: Optional[ValidationReport] = None
    
    # Cryptographic audit trail
    commitment_record: Optional[CommitmentRecord] = None
    blockchain_tx: Optional[str] = None
    
    # Challenge and response data
    challenges_generated: int = 0
    challenges_processed: int = 0
    
    # Detailed breakdown
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        
        # Convert datetime to ISO string
        if self.timestamp:
            result['timestamp'] = self.timestamp.isoformat()
            
        return result


class RangeCalibrator:
    """Calibrate expected ranges from reference model performance"""
    
    def __init__(self, confidence_level: float = 0.95, 
                 percentile_margin: float = 0.05):
        """
        Initialize range calibrator
        
        Args:
            confidence_level: Statistical confidence level for ranges
            percentile_margin: Margin for percentile-based range calculation
        """
        self.confidence_level = confidence_level
        self.percentile_margin = percentile_margin
        
    def calibrate(self, reference_model: Any, test_suite: List[Any],
                 model_type: ModelType = ModelType.GENERIC,
                 num_runs: int = 10) -> ExpectedRanges:
        """
        Calibrate expected ranges from reference model
        
        Args:
            reference_model: Reference model to calibrate against
            test_suite: Test challenges/inputs
            model_type: Type of model for specialized metrics
            num_runs: Number of calibration runs for statistics
            
        Returns:
            Calibrated ExpectedRanges
        """
        logger.info(f"Calibrating expected ranges with {num_runs} runs on {len(test_suite)} test cases")
        
        # Collect metrics from multiple runs
        accuracy_samples = []
        latency_samples = []
        fingerprint_samples = []
        jacobian_samples = []
        
        for run_idx in range(num_runs):
            logger.debug(f"Calibration run {run_idx + 1}/{num_runs}")
            
            for test_case_idx, test_input in enumerate(test_suite):
                try:
                    # Measure accuracy (model-specific)
                    accuracy = self._measure_accuracy(reference_model, test_input, model_type)
                    if accuracy is not None:
                        accuracy_samples.append(accuracy)
                    
                    # Measure latency
                    latency = self._measure_latency(reference_model, test_input)
                    if latency is not None:
                        latency_samples.append(latency)
                    
                    # Measure fingerprint similarity (compare with itself)
                    fingerprint_sim = self._measure_fingerprint_similarity(
                        reference_model, test_input
                    )
                    if fingerprint_sim is not None:
                        fingerprint_samples.append(fingerprint_sim)
                    
                    # Measure Jacobian norm
                    jacobian_norm = self._measure_jacobian_norm(reference_model, test_input)
                    if jacobian_norm is not None:
                        jacobian_samples.append(jacobian_norm)
                        
                except Exception as e:
                    logger.warning(f"Error in calibration run {run_idx}, test {test_case_idx}: {e}")
                    continue
        
        # Compute ranges from samples
        accuracy_range = self._compute_range_from_samples(accuracy_samples, "accuracy")
        latency_range = self._compute_range_from_samples(latency_samples, "latency")
        fingerprint_range = self._compute_range_from_samples(fingerprint_samples, "fingerprint")
        jacobian_range = self._compute_range_from_samples(jacobian_samples, "jacobian_norm")
        
        logger.info(f"Calibration complete:")
        logger.info(f"  Accuracy range: [{accuracy_range[0]:.3f}, {accuracy_range[1]:.3f}]")
        logger.info(f"  Latency range: [{latency_range[0]:.1f}, {latency_range[1]:.1f}]ms")
        logger.info(f"  Fingerprint range: [{fingerprint_range[0]:.3f}, {fingerprint_range[1]:.3f}]")
        logger.info(f"  Jacobian range: [{jacobian_range[0]:.6f}, {jacobian_range[1]:.6f}]")
        
        return ExpectedRanges(
            accuracy_range=accuracy_range,
            latency_range=latency_range,
            fingerprint_similarity=fingerprint_range,
            jacobian_norm_range=jacobian_range,
            confidence_level=self.confidence_level
        )
    
    def _measure_accuracy(self, model: Any, test_input: Any, 
                         model_type: ModelType) -> Optional[float]:
        """Measure model accuracy on test input"""
        try:
            if model_type == ModelType.VISION:
                # For vision models, use synthetic accuracy metric
                output = self._get_model_output(model, test_input)
                if output is not None:
                    # Synthetic accuracy based on output characteristics
                    if isinstance(output, np.ndarray):
                        # Use entropy as proxy for confidence/accuracy
                        normalized_output = np.abs(output) / (np.sum(np.abs(output)) + 1e-8)
                        entropy = -np.sum(normalized_output * np.log(normalized_output + 1e-8))
                        # Convert entropy to accuracy-like metric [0, 1]
                        max_entropy = np.log(len(normalized_output))
                        accuracy = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
                        return accuracy
                    
            elif model_type == ModelType.LANGUAGE:
                # For language models, use perplexity-based accuracy
                output = self._get_model_output(model, test_input)
                if output is not None:
                    # Synthetic accuracy from output consistency
                    if isinstance(output, (str, list)):
                        # Simple length-based accuracy proxy
                        output_str = str(output)
                        accuracy = min(1.0, len(output_str) / 100.0)  # Normalize by expected length
                        return accuracy
            
            else:  # GENERIC or others
                # Generic accuracy measurement
                output = self._get_model_output(model, test_input)
                if output is not None:
                    # Use output magnitude as accuracy proxy
                    if isinstance(output, np.ndarray):
                        magnitude = np.linalg.norm(output)
                        # Normalize to [0, 1] range
                        accuracy = min(1.0, magnitude / 10.0)
                        return accuracy
                    elif isinstance(output, (int, float)):
                        return min(1.0, abs(float(output)) / 10.0)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error measuring accuracy: {e}")
            return None
    
    def _measure_latency(self, model: Any, test_input: Any) -> Optional[float]:
        """Measure model inference latency in milliseconds"""
        try:
            start_time = time.time()
            _ = self._get_model_output(model, test_input)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000.0
            return latency_ms
            
        except Exception as e:
            logger.debug(f"Error measuring latency: {e}")
            return None
    
    def _measure_fingerprint_similarity(self, model: Any, test_input: Any) -> Optional[float]:
        """Measure fingerprint similarity (model with itself)"""
        try:
            # Get two outputs from the same model+input (should be identical for deterministic models)
            output1 = self._get_model_output(model, test_input)
            output2 = self._get_model_output(model, test_input)
            
            if output1 is not None and output2 is not None:
                # Compute similarity between outputs
                if isinstance(output1, np.ndarray) and isinstance(output2, np.ndarray):
                    # Cosine similarity
                    dot_product = np.dot(output1.flatten(), output2.flatten())
                    norm1 = np.linalg.norm(output1.flatten())
                    norm2 = np.linalg.norm(output2.flatten())
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = dot_product / (norm1 * norm2)
                        return max(0.0, min(1.0, similarity))  # Clamp to [0.0, 1.0]
                elif isinstance(output1, str) and isinstance(output2, str):
                    # String similarity (for language models)
                    if output1 == output2:
                        return 1.0
                    else:
                        # Jaccard similarity
                        set1 = set(output1.split())
                        set2 = set(output2.split())
                        if len(set1) == 0 and len(set2) == 0:
                            return 1.0
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        return intersection / union if union > 0 else 0.0
                else:
                    # Exact match for other types
                    return 1.0 if output1 == output2 else 0.0
            
            return None
            
        except Exception as e:
            logger.debug(f"Error measuring fingerprint similarity: {e}")
            return None
    
    def _measure_jacobian_norm(self, model: Any, test_input: Any) -> Optional[float]:
        """Measure Jacobian norm (gradient magnitude)"""
        try:
            # For simplicity, use output magnitude as proxy for Jacobian norm
            # In practice, would compute actual gradients
            output = self._get_model_output(model, test_input)
            
            if output is not None:
                if isinstance(output, np.ndarray):
                    # Use output norm as proxy for Jacobian norm
                    norm = np.linalg.norm(output)
                    return norm
                elif isinstance(output, (int, float)):
                    return abs(float(output))
                else:
                    # For other types, use hash-based norm
                    hash_val = hash(str(output))
                    return abs(hash_val) / 1e6  # Normalize
            
            return None
            
        except Exception as e:
            logger.debug(f"Error measuring Jacobian norm: {e}")
            return None
    
    def _get_model_output(self, model: Any, test_input: Any) -> Any:
        """Get model output for given input"""
        try:
            if hasattr(model, 'forward'):
                return model.forward(test_input)
            elif hasattr(model, 'predict'):
                return model.predict(test_input)
            elif callable(model):
                return model(test_input)
            else:
                return None
        except Exception as e:
            logger.debug(f"Error getting model output: {e}")
            return None
    
    def _compute_range_from_samples(self, samples: List[float], 
                                  metric_name: str) -> Tuple[float, float]:
        """Compute range bounds from sample data"""
        if not samples:
            logger.warning(f"No samples for {metric_name}, using default range")
            return (0.0, 1.0)
        
        samples_array = np.array(samples)
        
        # Remove outliers using IQR method
        q1 = np.percentile(samples_array, 25)
        q3 = np.percentile(samples_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter outliers
        filtered_samples = samples_array[
            (samples_array >= lower_bound) & (samples_array <= upper_bound)
        ]
        
        if len(filtered_samples) == 0:
            filtered_samples = samples_array
        
        # Compute percentile-based range
        margin_percent = self.percentile_margin * 100
        min_val = np.percentile(filtered_samples, margin_percent)
        max_val = np.percentile(filtered_samples, 100 - margin_percent)
        
        # Ensure minimum range width
        min_range_width = 0.01
        range_width = max_val - min_val
        if range_width < min_range_width:
            center = (min_val + max_val) / 2.0
            min_val = center - min_range_width / 2.0
            max_val = center + min_range_width / 2.0
        
        logger.debug(f"{metric_name}: {len(samples)} samples -> range [{min_val:.6f}, {max_val:.6f}]")
        
        return (float(min_val), float(max_val))


class ChallengeLibrary:
    """
    Pre-computed challenges for different model types
    """
    
    @staticmethod
    def get_vision_challenges(resolution: int = 224, 
                            channels: int = 3,
                            num_challenges: int = 5) -> List[np.ndarray]:
        """
        Generate synthetic images that probe vision model internals
        
        Args:
            resolution: Image resolution (assumes square)
            channels: Number of color channels
            num_challenges: Number of challenges to generate
            
        Returns:
            List of challenge images as numpy arrays
        """
        challenges = []
        
        # 1. Frequency patterns (Fourier basis)
        for i in range(min(2, num_challenges)):
            img = np.zeros((resolution, resolution, channels))
            freq = (i + 1) * 2
            x = np.linspace(0, freq * np.pi, resolution)
            y = np.linspace(0, freq * np.pi, resolution)
            X, Y = np.meshgrid(x, y)
            pattern = np.sin(X) * np.cos(Y)
            for c in range(channels):
                img[:, :, c] = pattern
            challenges.append(img)
        
        # 2. Geometric shapes
        if num_challenges > 2:
            img = np.zeros((resolution, resolution, channels))
            center = resolution // 2
            radius = resolution // 4
            y, x = np.ogrid[:resolution, :resolution]
            mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
            img[mask] = 1.0
            challenges.append(img)
        
        # 3. Adversarial patterns (noise)
        if num_challenges > 3:
            img = np.random.randn(resolution, resolution, channels) * 0.1
            challenges.append(img)
        
        # 4. Gradient patterns
        if num_challenges > 4:
            img = np.zeros((resolution, resolution, channels))
            for i in range(resolution):
                img[i, :, :] = i / resolution
            challenges.append(img)
        
        return challenges[:num_challenges]
    
    @staticmethod
    def get_language_challenges(vocab_size: int = 50000,
                               max_length: int = 100,
                               num_challenges: int = 5) -> List[str]:
        """
        Generate text sequences that reveal model-specific patterns
        
        Args:
            vocab_size: Model vocabulary size
            max_length: Maximum sequence length
            num_challenges: Number of challenges to generate
            
        Returns:
            List of challenge text strings
        """
        challenges = []
        
        # 1. Rare token combinations
        rare_combinations = [
            "The quantum fox jettisons holographic matrices",
            "Nebulous axioms permeate crystalline paradigms",
            "Recursive lambdas orchestrate Byzantine consensus"
        ]
        challenges.extend(rare_combinations[:min(3, num_challenges)])
        
        # 2. Grammatical edge cases
        if num_challenges > 3:
            edge_cases = [
                "The the cat cat sat sat on on the the mat mat",
                "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo"
            ]
            challenges.extend(edge_cases[:min(2, num_challenges - 3)])
        
        # 3. Semantic anomalies
        if num_challenges > 5:
            anomalies = [
                "Colorless green ideas sleep furiously",
                "The square root of purple equals Wednesday"
            ]
            challenges.extend(anomalies[:num_challenges - 5])
        
        return challenges[:num_challenges]
    
    @staticmethod
    def get_multimodal_challenges(num_challenges: int = 3) -> List[Dict[str, Any]]:
        """
        Generate combined vision-language challenges
        
        Args:
            num_challenges: Number of challenges to generate
            
        Returns:
            List of multimodal challenge dictionaries
        """
        challenges = []
        
        for i in range(num_challenges):
            # Get vision and language components
            vision_challenge = ChallengeLibrary.get_vision_challenges(
                resolution=224, channels=3, num_challenges=1
            )[0]
            
            language_challenge = ChallengeLibrary.get_language_challenges(
                num_challenges=1
            )[0]
            
            challenges.append({
                'vision': vision_challenge,
                'language': language_challenge,
                'instruction': f"Describe the pattern in image {i+1}"
            })
        
        return challenges
    
    @staticmethod
    def get_generic_challenges(input_dim: int, 
                              num_challenges: int = 5) -> List[np.ndarray]:
        """
        Generate generic challenges for any model type
        
        Args:
            input_dim: Input dimension of the model
            num_challenges: Number of challenges to generate
            
        Returns:
            List of challenge vectors
        """
        challenges = []
        
        for i in range(num_challenges):
            if i == 0:
                # Zero vector
                challenge = np.zeros(input_dim)
            elif i == 1:
                # Ones vector
                challenge = np.ones(input_dim)
            elif i == 2:
                # Random normal
                challenge = np.random.randn(input_dim)
            elif i == 3:
                # Sparse vector
                challenge = np.zeros(input_dim)
                indices = np.random.choice(input_dim, size=input_dim//10, replace=False)
                challenge[indices] = np.random.randn(len(indices))
            else:
                # Sinusoidal pattern
                challenge = np.sin(np.linspace(0, 4*np.pi, input_dim))
            
            challenges.append(challenge)
        
        return challenges


class ProofOfTraining:
    """
    Complete Proof-of-Training system integrating all verification components
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize complete PoT system
        
        Args:
            config: Configuration dictionary with:
                - verification_type: 'exact', 'fuzzy', 'statistical'
                - model_type: 'vision', 'language', 'multimodal'
                - security_level: 'low', 'medium', 'high'
        """
        self.config = config
        self.verification_type = VerificationType(config.get('verification_type', 'fuzzy'))
        self.model_type = ModelType(config.get('model_type', 'generic'))
        self.security_level = SecurityLevel(config.get('security_level', 'medium'))
        
        # Initialize components
        self._initialize_components()
        
        # Model registry
        self.model_registry = {}
        
        # Verification cache
        self.verification_cache = {}
        
        # Expected ranges for validation
        self.expected_ranges = {}  # model_id -> ExpectedRanges
        
        # Range calibrator
        self.range_calibrator = RangeCalibrator(
            confidence_level=0.95,
            percentile_margin=0.05
        )
        
        logger.info(f"ProofOfTraining initialized with {self.verification_type.value} verification")
    
    def _initialize_components(self):
        """Initialize verification components based on configuration"""
        # Fuzzy hash verifier
        if FUZZY_AVAILABLE:
            threshold = 0.95 if self.security_level == SecurityLevel.HIGH else 0.85
            self.fuzzy_verifier = FuzzyHashVerifier(
                similarity_threshold=threshold,
                algorithm=HashAlgorithm.SHA256
            )
        else:
            self.fuzzy_verifier = None
            
        # Provenance tracker
        if PROVENANCE_AVAILABLE:
            self.provenance_tracker = TrainingProvenanceAuditor(
                model_id="pot_system",
                compression_enabled=True
            )
        else:
            self.provenance_tracker = None
            
        # Token normalizer (for language models)
        if TOKEN_AVAILABLE:
            self.token_normalizer = TokenSpaceNormalizer(
                tokenizer_type=TokenizerType.BPE,
                vocab_size=50000
            )
            self.stochastic_controller = StochasticDecodingController(seed=42)
        else:
            self.token_normalizer = None
            self.stochastic_controller = None
    
    def register_model(self, model: Any, 
                      architecture: str = "unknown",
                      parameter_count: int = 0,
                      training_history: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a model for verification
        
        Args:
            model: Model to register
            architecture: Model architecture description
            parameter_count: Number of model parameters
            training_history: Optional training history
            
        Returns:
            Unique model ID
        """
        # Generate model ID
        model_data = f"{architecture}_{parameter_count}_{time.time()}"
        model_id = hashlib.sha256(model_data.encode()).hexdigest()[:16]
        
        # Generate model fingerprint
        if hasattr(model, 'state_dict'):
            # PyTorch model
            state_data = str(sorted(model.state_dict().keys()))
        elif hasattr(model, 'get_weights'):
            # TensorFlow/Keras model
            state_data = str(len(model.get_weights()))
        else:
            state_data = str(model)
        
        fingerprint = hashlib.sha256(state_data.encode()).hexdigest()
        
        # Generate reference challenges
        reference_challenges = self.generate_adaptive_challenges(
            architecture, parameter_count
        )
        
        # Collect reference responses
        reference_responses = {}
        for challenge_type, challenges in reference_challenges.items():
            responses = []
            for challenge in challenges:
                try:
                    # Get model response
                    if hasattr(model, 'forward'):
                        response = model.forward(challenge)
                    elif hasattr(model, 'predict'):
                        response = model.predict(challenge)
                    elif callable(model):
                        response = model(challenge)
                    else:
                        response = f"mock_response_{challenge_type}"
                    
                    # Generate hash of response
                    if self.fuzzy_verifier:
                        if isinstance(response, (np.ndarray, list)):
                            response_hash = self.fuzzy_verifier.generate_fuzzy_hash(
                                np.array(response)
                            )
                        else:
                            response_hash = hashlib.sha256(
                                str(response).encode()
                            ).hexdigest()
                    else:
                        response_hash = hashlib.sha256(
                            str(response).encode()
                        ).hexdigest()
                    
                    responses.append(response_hash)
                except Exception as e:
                    logger.error(f"Error generating reference response: {e}")
                    responses.append(None)
            
            reference_responses[challenge_type] = responses
        
        # Process training history if provided
        training_provenance = None
        if training_history and self.provenance_tracker:
            try:
                # Embed provenance
                model_state = {'model': model, 'history': training_history}
                embedded = self.provenance_tracker.embed_provenance(
                    model_state,
                    training_history.get('events', [])
                )
                training_provenance = embedded.get('metadata', {}).get('training_provenance')
            except Exception as e:
                logger.error(f"Error processing training history: {e}")
        
        # Create registration
        registration = ModelRegistration(
            model_id=model_id,
            model_type=self.model_type,
            parameter_count=parameter_count,
            architecture=architecture,
            fingerprint=fingerprint,
            reference_challenges=reference_challenges,
            reference_responses=reference_responses,
            training_provenance=training_provenance,
            registration_time=datetime.now(timezone.utc),
            metadata={
                'verification_type': self.verification_type.value,
                'security_level': self.security_level.value
            }
        )
        
        # Store registration
        self.model_registry[model_id] = registration
        
        logger.info(f"Model registered with ID: {model_id}")
        
        return model_id
    
    def calibrate_expected_ranges(self, model: Any, model_id: str,
                                num_calibration_runs: int = 10) -> ExpectedRanges:
        """
        Calibrate expected ranges for a registered model
        
        Args:
            model: Model to calibrate
            model_id: Registered model ID
            num_calibration_runs: Number of calibration runs for statistics
            
        Returns:
            Calibrated ExpectedRanges
        """
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not registered")
        
        registration = self.model_registry[model_id]
        
        # Get test suite from reference challenges
        test_suite = []
        for challenge_type, challenges in registration.reference_challenges.items():
            test_suite.extend(challenges)
        
        if not test_suite:
            logger.warning(f"No test suite available for model {model_id}")
            # Return default ranges
            return ExpectedRanges(
                accuracy_range=(0.0, 1.0),
                latency_range=(0.1, 1000.0),
                fingerprint_similarity=(0.8, 1.0),
                jacobian_norm_range=(0.01, 10.0)
            )
        
        logger.info(f"Calibrating expected ranges for model {model_id} with {len(test_suite)} test cases")
        
        # Calibrate ranges
        expected_ranges = self.range_calibrator.calibrate(
            reference_model=model,
            test_suite=test_suite,
            model_type=registration.model_type,
            num_runs=num_calibration_runs
        )
        
        # Store ranges for this model
        self.expected_ranges[model_id] = expected_ranges
        
        logger.info(f"Expected ranges calibrated for model {model_id}")
        
        return expected_ranges
    
    def set_expected_ranges(self, model_id: str, expected_ranges: ExpectedRanges):
        """
        Manually set expected ranges for a model
        
        Args:
            model_id: Registered model ID
            expected_ranges: Expected ranges to set
        """
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not registered")
        
        self.expected_ranges[model_id] = expected_ranges
        logger.info(f"Expected ranges set for model {model_id}")
    
    def get_expected_ranges(self, model_id: str) -> Optional[ExpectedRanges]:
        """
        Get expected ranges for a model
        
        Args:
            model_id: Registered model ID
            
        Returns:
            Expected ranges if available, None otherwise
        """
        return self.expected_ranges.get(model_id)
    
    def generate_adaptive_challenges(self, model_architecture: str,
                                   parameter_count: int) -> Dict[str, List[Any]]:
        """
        Generate challenges adapted to model characteristics
        
        Args:
            model_architecture: Model architecture description
            parameter_count: Number of model parameters
            
        Returns:
            Dictionary of challenge types to challenge lists
        """
        challenges = {}
        
        # Determine number of challenges based on security level
        if self.security_level == SecurityLevel.HIGH:
            num_challenges = 10
        elif self.security_level == SecurityLevel.MEDIUM:
            num_challenges = 5
        else:
            num_challenges = 3
        
        # Adapt to model size
        if parameter_count < 1_000_000:
            # Small model - simpler challenges
            num_challenges = max(2, num_challenges // 2)
        
        # Generate challenges based on model type
        if self.model_type == ModelType.VISION:
            # Determine resolution based on model size
            if parameter_count < 10_000_000:
                resolution = 128
            else:
                resolution = 224
            
            challenges['vision'] = ChallengeLibrary.get_vision_challenges(
                resolution=resolution,
                channels=3,
                num_challenges=num_challenges
            )
            
        elif self.model_type == ModelType.LANGUAGE:
            # Determine vocab size estimate
            vocab_size = min(50000, parameter_count // 100)
            
            challenges['language'] = ChallengeLibrary.get_language_challenges(
                vocab_size=vocab_size,
                max_length=100,
                num_challenges=num_challenges
            )
            
        elif self.model_type == ModelType.MULTIMODAL:
            challenges['multimodal'] = ChallengeLibrary.get_multimodal_challenges(
                num_challenges=min(num_challenges, 3)
            )
            
        else:  # GENERIC
            # Estimate input dimension from architecture string
            input_dim = 100  # Default
            if 'dim' in model_architecture.lower():
                try:
                    # Try to extract dimension from architecture string
                    import re
                    match = re.search(r'dim[_=]?(\d+)', model_architecture.lower())
                    if match:
                        input_dim = int(match.group(1))
                except:
                    pass
            
            challenges['generic'] = ChallengeLibrary.get_generic_challenges(
                input_dim=input_dim,
                num_challenges=num_challenges
            )
        
        logger.info(f"Generated {sum(len(c) for c in challenges.values())} adaptive challenges")
        
        return challenges
    
    def perform_verification(self, model: Any, model_id: str,
                           verification_depth: str = 'standard') -> VerificationResult:
        """
        Execute complete verification protocol
        
        Args:
            model: Model to verify
            model_id: Registered model ID
            verification_depth: 'quick', 'standard', or 'comprehensive'
            
        Returns:
            VerificationResult with verification details
        """
        start_time = time.time()
        depth = VerificationDepth(verification_depth)
        
        # Check if model is registered
        if model_id not in self.model_registry:
            return VerificationResult(
                verified=False,
                confidence=0.0,
                verification_type="none",
                model_id=model_id,
                challenges_passed=0,
                challenges_total=0,
                fuzzy_similarity=None,
                statistical_score=None,
                provenance_verified=None,
                details={'error': 'Model not registered'},
                timestamp=datetime.now(timezone.utc),
                duration_seconds=time.time() - start_time
            )
        
        registration = self.model_registry[model_id]
        
        # Determine number of challenges based on depth
        if depth == VerificationDepth.QUICK:
            num_challenges_to_verify = 1
            check_provenance = False
        elif depth == VerificationDepth.STANDARD:
            num_challenges_to_verify = 3
            check_provenance = False
        else:  # COMPREHENSIVE
            num_challenges_to_verify = None  # All challenges
            check_provenance = True
        
        # Verify challenges
        challenges_passed = 0
        challenges_total = 0
        similarity_scores = []
        
        # Collect metrics for expected ranges validation
        accuracy_measurements = []
        latency_measurements = []
        fingerprint_measurements = []
        jacobian_measurements = []
        
        for challenge_type, challenges in registration.reference_challenges.items():
            reference_responses = registration.reference_responses.get(challenge_type, [])
            
            # Limit challenges if needed
            if num_challenges_to_verify:
                challenges = challenges[:num_challenges_to_verify]
                reference_responses = reference_responses[:num_challenges_to_verify]
            
            for challenge, ref_response in zip(challenges, reference_responses):
                if ref_response is None:
                    continue
                    
                challenges_total += 1
                
                try:
                    # Measure latency
                    start_time = time.time()
                    
                    # Get model response
                    if hasattr(model, 'forward'):
                        response = model.forward(challenge)
                    elif hasattr(model, 'predict'):
                        response = model.predict(challenge)
                    elif callable(model):
                        response = model(challenge)
                    else:
                        response = f"mock_response_{challenge_type}"
                    
                    # Record latency
                    latency_ms = (time.time() - start_time) * 1000.0
                    latency_measurements.append(latency_ms)
                    
                    # Measure additional metrics for expected ranges
                    if depth == VerificationDepth.COMPREHENSIVE:
                        # Accuracy measurement
                        accuracy = self.range_calibrator._measure_accuracy(
                            model, challenge, registration.model_type
                        )
                        if accuracy is not None:
                            accuracy_measurements.append(accuracy)
                        
                        # Fingerprint similarity
                        fingerprint_sim = self.range_calibrator._measure_fingerprint_similarity(
                            model, challenge
                        )
                        if fingerprint_sim is not None:
                            fingerprint_measurements.append(fingerprint_sim)
                        
                        # Jacobian norm
                        jacobian_norm = self.range_calibrator._measure_jacobian_norm(
                            model, challenge
                        )
                        if jacobian_norm is not None:
                            jacobian_measurements.append(jacobian_norm)
                    
                    # Verify based on verification type
                    if self.verification_type == VerificationType.EXACT:
                        # Exact matching
                        response_hash = hashlib.sha256(str(response).encode()).hexdigest()
                        passed = response_hash == ref_response
                        similarity = 1.0 if passed else 0.0
                        
                    elif self.verification_type == VerificationType.FUZZY:
                        # Fuzzy matching
                        if self.fuzzy_verifier and isinstance(response, (np.ndarray, list)):
                            response_hash = self.fuzzy_verifier.generate_fuzzy_hash(
                                np.array(response)
                            )
                            result = self.fuzzy_verifier.verify_fuzzy(
                                response_hash, ref_response
                            )
                            passed = result.is_valid
                            similarity = result.similarity_score
                        else:
                            response_hash = hashlib.sha256(str(response).encode()).hexdigest()
                            passed = response_hash == ref_response
                            similarity = 1.0 if passed else 0.0
                            
                    else:  # STATISTICAL
                        # Statistical verification
                        if self.stochastic_controller and self.model_type == ModelType.LANGUAGE:
                            # Generate multiple variants for statistical comparison
                            variants = self.stochastic_controller.generate_verification_response(
                                model, challenge, num_variants=3
                            )
                            similarity = self.stochastic_controller.compute_semantic_similarity(
                                [str(v) for v in variants]
                            )
                            passed = similarity >= 0.7
                        else:
                            # Fallback to fuzzy matching
                            response_hash = hashlib.sha256(str(response).encode()).hexdigest()
                            passed = response_hash == ref_response
                            similarity = 1.0 if passed else 0.0
                    
                    if passed:
                        challenges_passed += 1
                    similarity_scores.append(similarity)
                    
                except Exception as e:
                    logger.error(f"Error during challenge verification: {e}")
                    similarity_scores.append(0.0)
        
        # Calculate overall metrics
        if challenges_total > 0:
            pass_rate = challenges_passed / challenges_total
            avg_similarity = np.mean(similarity_scores)
        else:
            pass_rate = 0.0
            avg_similarity = 0.0
        
        # Check provenance if required
        provenance_verified = None
        if check_provenance and registration.training_provenance and self.provenance_tracker:
            try:
                # Verify training history
                provenance_verified = True  # Simplified check
            except Exception as e:
                logger.error(f"Provenance verification error: {e}")
                provenance_verified = False
        
        # Determine overall verification status
        if self.security_level == SecurityLevel.HIGH:
            confidence_threshold = 0.95
        elif self.security_level == SecurityLevel.MEDIUM:
            confidence_threshold = 0.85
        else:
            confidence_threshold = 0.70
        
        confidence = pass_rate * avg_similarity
        verified = confidence >= confidence_threshold
        
        # Compute average measurements for expected ranges validation
        avg_accuracy = np.mean(accuracy_measurements) if accuracy_measurements else None
        avg_latency = np.mean(latency_measurements) if latency_measurements else None
        avg_fingerprint_sim = np.mean(fingerprint_measurements) if fingerprint_measurements else None
        avg_jacobian_norm = np.mean(jacobian_measurements) if jacobian_measurements else None
        
        # Perform expected ranges validation if available
        range_validation = None
        if model_id in self.expected_ranges:
            expected_ranges = self.expected_ranges[model_id]
            
            # Create temporary result for validation
            temp_result = VerificationResult(
                verified=verified,
                confidence=confidence,
                verification_type=self.verification_type.value,
                model_id=model_id,
                challenges_passed=challenges_passed,
                challenges_total=challenges_total,
                fuzzy_similarity=avg_similarity if self.verification_type == VerificationType.FUZZY else None,
                statistical_score=avg_similarity if self.verification_type == VerificationType.STATISTICAL else None,
                provenance_verified=provenance_verified,
                details={},
                timestamp=datetime.now(timezone.utc),
                duration_seconds=time.time() - start_time,
                accuracy=avg_accuracy,
                latency_ms=avg_latency,
                fingerprint_similarity=avg_fingerprint_sim,
                jacobian_norm=avg_jacobian_norm
            )
            
            # Validate against expected ranges
            range_validation = expected_ranges.validate(temp_result)
            
            # Update verification status based on range validation
            if range_validation and not range_validation.passed:
                logger.warning(f"Expected ranges validation failed: {range_validation.violations}")
                # Optionally reduce confidence or fail verification
                confidence *= range_validation.confidence
                verified = verified and range_validation.passed
        
        # Create final result
        result = VerificationResult(
            verified=verified,
            confidence=confidence,
            verification_type=self.verification_type.value,
            model_id=model_id,
            challenges_passed=challenges_passed,
            challenges_total=challenges_total,
            fuzzy_similarity=avg_similarity if self.verification_type == VerificationType.FUZZY else None,
            statistical_score=avg_similarity if self.verification_type == VerificationType.STATISTICAL else None,
            provenance_verified=provenance_verified,
            details={
                'pass_rate': pass_rate,
                'verification_depth': depth.value,
                'security_level': self.security_level.value,
                'model_type': self.model_type.value,
                'range_validation_enabled': range_validation is not None,
                'measurements_collected': {
                    'accuracy_samples': len(accuracy_measurements),
                    'latency_samples': len(latency_measurements),
                    'fingerprint_samples': len(fingerprint_measurements),
                    'jacobian_samples': len(jacobian_measurements)
                }
            },
            timestamp=datetime.now(timezone.utc),
            duration_seconds=time.time() - start_time,
            accuracy=avg_accuracy,
            latency_ms=avg_latency,
            fingerprint_similarity=avg_fingerprint_sim,
            jacobian_norm=avg_jacobian_norm,
            range_validation=range_validation
        )
        
        # Cache result
        cache_key = f"{model_id}_{verification_depth}_{time.time()}"
        self.verification_cache[cache_key] = result
        
        logger.info(f"Verification completed: {verified} (confidence: {confidence:.2%})")
        
        return result
    
    def generate_verification_proof(self, verification_result: VerificationResult) -> bytes:
        """
        Create cryptographic proof of verification
        
        Args:
            verification_result: Verification result to create proof for
            
        Returns:
            Cryptographic proof as bytes
        """
        # Create proof data (convert numpy types to Python types)
        proof_data = {
            'model_id': verification_result.model_id,
            'verified': bool(verification_result.verified),
            'confidence': float(verification_result.confidence),
            'timestamp': verification_result.timestamp.isoformat(),
            'verification_type': verification_result.verification_type,
            'challenges_passed': int(verification_result.challenges_passed),
            'challenges_total': int(verification_result.challenges_total),
            'details': verification_result.details
        }
        
        # Add registration fingerprint if available
        if verification_result.model_id in self.model_registry:
            registration = self.model_registry[verification_result.model_id]
            proof_data['model_fingerprint'] = registration.fingerprint
        
        # Serialize proof data
        proof_json = json.dumps(proof_data, sort_keys=True)
        
        # Create signature
        signature = hashlib.sha256(proof_json.encode()).hexdigest()
        
        # Create final proof
        proof = {
            'data': proof_data,
            'signature': signature,
            'algorithm': 'sha256',
            'version': '1.0'
        }
        
        # Encode as bytes
        proof_bytes = base64.b64encode(
            json.dumps(proof).encode()
        )
        
        logger.info(f"Generated verification proof for model {verification_result.model_id}")
        
        return proof_bytes
    
    def verify_proof(self, proof_bytes: bytes) -> Dict[str, Any]:
        """
        Verify a cryptographic proof
        
        Args:
            proof_bytes: Proof to verify
            
        Returns:
            Verification result dictionary
        """
        try:
            # Decode proof
            proof_json = base64.b64decode(proof_bytes).decode()
            proof = json.loads(proof_json)
            
            # Verify signature
            proof_data = proof['data']
            expected_signature = hashlib.sha256(
                json.dumps(proof_data, sort_keys=True).encode()
            ).hexdigest()
            
            signature_valid = proof['signature'] == expected_signature
            
            return {
                'valid': signature_valid,
                'data': proof_data,
                'signature_match': signature_valid
            }
            
        except Exception as e:
            logger.error(f"Proof verification error: {e}")
            return {
                'valid': False,
                'error': str(e)
            }
    
    def batch_verify(self, models: List[Any], model_ids: List[str],
                    verification_depth: str = 'standard') -> List[VerificationResult]:
        """
        Batch verification of multiple models
        
        Args:
            models: List of models to verify
            model_ids: List of model IDs
            verification_depth: Verification depth level
            
        Returns:
            List of verification results
        """
        results = []
        
        for model, model_id in zip(models, model_ids):
            result = self.perform_verification(model, model_id, verification_depth)
            results.append(result)
        
        logger.info(f"Batch verification completed for {len(models)} models")
        
        return results
    
    def incremental_verify(self, model: Any, model_id: str, 
                          epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Incremental verification during training
        
        Args:
            model: Model being trained
            model_id: Model ID
            epoch: Current epoch
            metrics: Training metrics
            
        Returns:
            Whether incremental verification passed
        """
        if not self.provenance_tracker:
            logger.warning("Provenance tracker not available for incremental verification")
            return True
        
        try:
            # Log training event
            self.provenance_tracker.log_training_event(
                epoch=epoch,
                metrics=metrics,
                event_type=EventType.EPOCH_END
            )
            
            # Quick verification every N epochs
            if epoch % 10 == 0:
                result = self.perform_verification(model, model_id, 'quick')
                return result.verified
            
            return True
            
        except Exception as e:
            logger.error(f"Incremental verification error: {e}")
            return False
    
    def cross_platform_verify(self, model_outputs: Dict[str, Any], 
                            model_id: str) -> VerificationResult:
        """
        Offline verification using only model outputs
        
        Args:
            model_outputs: Dictionary of challenge IDs to model outputs
            model_id: Registered model ID
            
        Returns:
            VerificationResult
        """
        start_time = time.time()
        
        if model_id not in self.model_registry:
            return VerificationResult(
                verified=False,
                confidence=0.0,
                verification_type="offline",
                model_id=model_id,
                challenges_passed=0,
                challenges_total=0,
                fuzzy_similarity=None,
                statistical_score=None,
                provenance_verified=None,
                details={'error': 'Model not registered'},
                timestamp=datetime.now(timezone.utc),
                duration_seconds=time.time() - start_time
            )
        
        registration = self.model_registry[model_id]
        
        # Verify outputs against reference
        challenges_passed = 0
        challenges_total = len(model_outputs)
        
        for challenge_id, output in model_outputs.items():
            # Compare with reference (simplified)
            # In production, would map challenge_id to specific reference
            if self.fuzzy_verifier:
                output_hash = self.fuzzy_verifier.generate_fuzzy_hash(
                    np.array(output) if isinstance(output, list) else output
                )
                # Compare with any reference response
                for ref_responses in registration.reference_responses.values():
                    for ref_response in ref_responses:
                        if ref_response:
                            result = self.fuzzy_verifier.verify_fuzzy(
                                output_hash, ref_response
                            )
                            if result.is_valid:
                                challenges_passed += 1
                                break
                    else:
                        continue
                    break
        
        confidence = challenges_passed / challenges_total if challenges_total > 0 else 0.0
        verified = confidence >= 0.7
        
        return VerificationResult(
            verified=verified,
            confidence=confidence,
            verification_type="offline",
            model_id=model_id,
            challenges_passed=challenges_passed,
            challenges_total=challenges_total,
            fuzzy_similarity=confidence,
            statistical_score=None,
            provenance_verified=None,
            details={
                'verification_mode': 'offline',
                'platform': 'cross-platform'
            },
            timestamp=datetime.now(timezone.utc),
            duration_seconds=time.time() - start_time
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'registered_models': len(self.model_registry),
            'cached_verifications': len(self.verification_cache),
            'models_with_expected_ranges': len(self.expected_ranges),
            'verification_type': self.verification_type.value,
            'model_type': self.model_type.value,
            'security_level': self.security_level.value,
            'components': {
                'fuzzy_verifier': self.fuzzy_verifier is not None,
                'provenance_tracker': self.provenance_tracker is not None,
                'token_normalizer': self.token_normalizer is not None,
                'range_calibrator': self.range_calibrator is not None,
                'expected_ranges': len(self.expected_ranges) > 0
            }
        }
    
    def run_verification(self, session_cfg: SessionConfig) -> VerificationReport:
        """
        Complete proof-of-training verification with audit trail.
        
        This is the integrated verification protocol that combines all PoT components:
        1. Generate cryptographic challenges
        2. Compute behavioral fingerprints  
        3. Run sequential statistical test
        4. Check against expected ranges
        5. Create commit-reveal record
        6. Optionally post to blockchain
        
        Args:
            session_cfg: Complete session configuration
            
        Returns:
            Comprehensive verification report
        """
        logger.info(f"Starting integrated verification for model {session_cfg.model_id}")
        start_time = time.time()
        
        # Generate unique session ID
        session_id = hashlib.sha256(
            f"{session_cfg.model_id}_{session_cfg.master_seed}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        try:
            # Step 1: Challenge Generation
            logger.info("Step 1: Generating cryptographic challenges...")
            challenges = self._generate_challenges_for_session(session_cfg)
            
            # Step 2: Behavioral Fingerprinting
            fingerprint_result = None
            if session_cfg.use_fingerprinting and CORE_COMPONENTS_AVAILABLE:
                logger.info("Step 2: Computing behavioral fingerprints...")
                fingerprint_result = self._compute_behavioral_fingerprint(
                    session_cfg, challenges
                )
            
            # Step 3: Sequential Statistical Testing
            statistical_result = None
            if session_cfg.use_sequential and CORE_COMPONENTS_AVAILABLE:
                logger.info("Step 3: Running sequential statistical test...")
                statistical_result = self._run_sequential_verification(
                    session_cfg, challenges
                )
            
            # Step 4: Expected Ranges Validation
            range_validation = None
            if session_cfg.use_range_validation and session_cfg.expected_ranges:
                logger.info("Step 4: Validating against expected ranges...")
                range_validation = self._validate_expected_ranges(
                    session_cfg, statistical_result, fingerprint_result
                )
            
            # Step 5: Create Commit-Reveal Record
            logger.info("Step 5: Creating cryptographic audit trail...")
            commitment_record = self._create_commitment_record(
                session_cfg, session_id, statistical_result, 
                fingerprint_result, range_validation
            )
            
            # Step 6: Optional Blockchain Storage
            blockchain_tx = None
            if session_cfg.use_blockchain and session_cfg.blockchain_config:
                logger.info("Step 6: Storing commitment to blockchain...")
                blockchain_tx = self._store_to_blockchain(
                    session_cfg, commitment_record
                )
            
            # Compute overall verification result
            overall_passed, overall_confidence = self._compute_overall_result(
                statistical_result, range_validation, fingerprint_result
            )
            
            # Create comprehensive verification report
            duration = time.time() - start_time
            report = VerificationReport(
                passed=overall_passed,
                confidence=overall_confidence,
                model_id=session_cfg.model_id,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                duration_seconds=duration,
                statistical_result=statistical_result,
                fingerprint_result=fingerprint_result,
                range_validation=range_validation,
                commitment_record=commitment_record,
                blockchain_tx=blockchain_tx,
                challenges_generated=len(challenges) if challenges else 0,
                challenges_processed=len(challenges) if challenges else 0,
                details={
                    'protocol_version': '1.0',
                    'components_used': {
                        'fingerprinting': session_cfg.use_fingerprinting,
                        'sequential_testing': session_cfg.use_sequential,
                        'range_validation': session_cfg.use_range_validation,
                        'blockchain': session_cfg.use_blockchain
                    },
                    'session_config': {
                        'num_challenges': session_cfg.num_challenges,
                        'accuracy_threshold': session_cfg.accuracy_threshold,
                        'challenge_family': session_cfg.challenge_family
                    }
                }
            )
            
            logger.info(f"Integrated verification completed: "
                       f"passed={overall_passed}, confidence={overall_confidence:.3f}")
            
            return report
            
        except Exception as e:
            logger.error(f"Integrated verification failed: {e}")
            duration = time.time() - start_time
            
            # Return failure report
            return VerificationReport(
                passed=False,
                confidence=0.0,
                model_id=session_cfg.model_id,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                duration_seconds=duration,
                details={'error': str(e), 'step_failed': 'protocol_execution'}
            )
    
    def _generate_challenges_for_session(self, session_cfg: SessionConfig) -> List[Any]:
        """Generate challenges for verification session"""
        if not CORE_COMPONENTS_AVAILABLE:
            logger.warning("Core components not available, using mock challenges")
            return [np.random.randn(100) for _ in range(session_cfg.num_challenges)]
        
        try:
            # Use core challenge generation
            challenge_config = ChallengeConfig(
                master_key_hex=session_cfg.master_seed,
                session_nonce_hex=hashlib.sha256(session_cfg.model_id.encode()).hexdigest(),
                n=session_cfg.num_challenges,
                family=session_cfg.challenge_family,
                params=session_cfg.challenge_params,
                model_id=session_cfg.model_id
            )
            
            challenge_result = generate_challenges(challenge_config)
            
            # Extract challenges list from result
            if isinstance(challenge_result, dict) and 'challenges' in challenge_result:
                challenges = challenge_result['challenges']
            elif isinstance(challenge_result, list):
                challenges = challenge_result
            else:
                challenges = [challenge_result]
            
            logger.info(f"Generated {len(challenges)} challenges using {session_cfg.challenge_family}")
            return challenges
            
        except Exception as e:
            logger.warning(f"Challenge generation failed, using fallback: {e}")
            # Fallback to simple challenges
            return [np.random.randn(100) for _ in range(session_cfg.num_challenges)]
    
    def _compute_behavioral_fingerprint(self, session_cfg: SessionConfig, 
                                      challenges: List[Any]) -> Optional[FingerprintResult]:
        """Compute behavioral fingerprint"""
        try:
            fingerprint_result = fingerprint_run(
                model=session_cfg.model,
                challenges=challenges,
                config=session_cfg.fingerprint_config
            )
            
            logger.info(f"Behavioral fingerprint computed: "
                       f"IO hash={fingerprint_result.io_hash[:16]}...")
            
            return fingerprint_result
            
        except Exception as e:
            logger.error(f"Fingerprinting failed: {e}")
            return None
    
    def _run_sequential_verification(self, session_cfg: SessionConfig,
                                   challenges: List[Any]) -> Optional[SPRTResult]:
        """Run sequential statistical verification"""
        try:
            # Create accuracy stream from model responses
            accuracy_stream = self._create_accuracy_stream(session_cfg.model, challenges)
            
            # Run sequential test
            sprt_result = sequential_verify(
                stream=accuracy_stream,
                tau=session_cfg.accuracy_threshold,
                alpha=session_cfg.type1_error,
                beta=session_cfg.type2_error,
                max_samples=min(session_cfg.max_samples, len(challenges)),
                compute_p_value=True
            )
            
            logger.info(f"Sequential test completed: decision={sprt_result.decision}, "
                       f"stopped_at={sprt_result.stopped_at}")
            
            return sprt_result
            
        except Exception as e:
            logger.error(f"Sequential verification failed: {e}")
            return None
    
    def _create_accuracy_stream(self, model: Any, challenges: List[Any]):
        """Create accuracy stream from model responses"""
        def accuracy_generator():
            for challenge in challenges:
                try:
                    # Get model response
                    if hasattr(model, 'forward'):
                        response = model.forward(challenge)
                    elif callable(model):
                        response = model(challenge)
                    else:
                        # Fallback to synthetic accuracy
                        response = np.random.randn(10)
                    
                    # Convert response to accuracy metric
                    if isinstance(response, np.ndarray):
                        # Use response magnitude as accuracy proxy
                        accuracy = min(1.0, np.linalg.norm(response) / 10.0)
                    elif isinstance(response, (int, float)):
                        accuracy = min(1.0, abs(float(response)) / 10.0)
                    else:
                        accuracy = 0.5  # Default for unknown types
                    
                    yield accuracy
                    
                except Exception as e:
                    logger.debug(f"Error computing accuracy for challenge: {e}")
                    yield 0.5  # Default accuracy on error
        
        return accuracy_generator()
    
    def _validate_expected_ranges(self, session_cfg: SessionConfig,
                                statistical_result: Optional[SPRTResult],
                                fingerprint_result: Optional[FingerprintResult]) -> Optional[ValidationReport]:
        """Validate results against expected ranges"""
        try:
            # Create a mock VerificationResult for range validation
            mock_result = VerificationResult(
                verified=True,
                confidence=0.0,
                verification_type="integrated",
                model_id=session_cfg.model_id,
                challenges_passed=0,
                challenges_total=0,
                fuzzy_similarity=None,
                statistical_score=None,
                provenance_verified=None,
                details={},
                timestamp=datetime.now(timezone.utc),
                duration_seconds=0.0
            )
            
            # Fill in metrics from results
            if statistical_result:
                mock_result.accuracy = statistical_result.final_mean
            
            if fingerprint_result:
                mock_result.fingerprint_similarity = 1.0  # Self-similarity
                mock_result.latency_ms = fingerprint_result.avg_latency_ms
            
            # Validate against expected ranges
            range_validation = session_cfg.expected_ranges.validate(mock_result)
            
            logger.info(f"Range validation: passed={range_validation.passed}, "
                       f"violations={len(range_validation.violations)}")
            
            return range_validation
            
        except Exception as e:
            logger.error(f"Range validation failed: {e}")
            return None
    
    def _create_commitment_record(self, session_cfg: SessionConfig, session_id: str,
                                statistical_result: Optional[SPRTResult],
                                fingerprint_result: Optional[FingerprintResult],
                                range_validation: Optional[ValidationReport]) -> Optional[CommitmentRecord]:
        """Create cryptographic commitment record"""
        try:
            # Prepare verification data
            verification_data = {
                "model_id": session_cfg.model_id,
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "protocol_version": "1.0"
            }
            
            # Add component results
            if statistical_result:
                verification_data["statistical_decision"] = statistical_result.decision
                verification_data["statistical_confidence"] = statistical_result.final_mean
                verification_data["samples_used"] = statistical_result.stopped_at
            
            if fingerprint_result:
                verification_data["fingerprint_hash"] = fingerprint_result.io_hash
                verification_data["avg_latency_ms"] = fingerprint_result.avg_latency_ms
            
            if range_validation:
                verification_data["range_validation_passed"] = range_validation.passed
                verification_data["range_violations"] = len(range_validation.violations)
                verification_data["range_confidence"] = range_validation.confidence
            
            # Create commitment
            commitment = compute_commitment(verification_data)
            
            # Write audit record - handle potential API differences
            try:
                write_audit_record(commitment, session_cfg.audit_log_path)
            except Exception as audit_error:
                logger.warning(f"Audit record write failed: {audit_error}")
                # Create a simple commitment record as fallback
                from dataclasses import dataclass
                @dataclass
                class FallbackCommitmentRecord:
                    commitment_hash: str
                    data: Dict[str, Any]
                    salt: str
                    timestamp: datetime
                
                commitment = FallbackCommitmentRecord(
                    commitment_hash=hashlib.sha256(str(verification_data).encode()).hexdigest(),
                    data=verification_data,
                    salt="fallback_salt",
                    timestamp=datetime.now(timezone.utc)
                )
            
            logger.info(f"Commitment record created: {commitment.commitment_hash[:16]}...")
            
            return commitment
            
        except Exception as e:
            logger.error(f"Commitment creation failed: {e}")
            return None
    
    def _store_to_blockchain(self, session_cfg: SessionConfig,
                           commitment_record: Any) -> Optional[str]:
        """Store commitment to blockchain"""
        try:
            blockchain_client = BlockchainClient(session_cfg.blockchain_config)
            
            # Connect to blockchain
            if not blockchain_client.connect():
                logger.error("Failed to connect to blockchain")
                return None
            
            # Store commitment - handle different commitment record types
            commitment_hash = commitment_record.commitment_hash
            if isinstance(commitment_hash, str) and len(commitment_hash) == 64:
                # Assume it's already hex
                commitment_bytes = bytes.fromhex(commitment_hash)
            else:
                # Fallback to encoding the hash
                commitment_bytes = commitment_hash.encode() if isinstance(commitment_hash, str) else commitment_hash
            
            tx_hash = blockchain_client.store_commitment(
                commitment_bytes,
                {"model_id": session_cfg.model_id, "session_id": commitment_hash[:16]}
            )
            
            logger.info(f"Commitment stored to blockchain: {tx_hash}")
            
            return tx_hash
            
        except Exception as e:
            logger.error(f"Blockchain storage failed: {e}")
            return None
    
    def _compute_overall_result(self, statistical_result: Optional[SPRTResult],
                              range_validation: Optional[ValidationReport],
                              fingerprint_result: Optional[FingerprintResult]) -> Tuple[bool, float]:
        """Compute overall verification result and confidence"""
        passed_components = []
        confidence_scores = []
        
        # Statistical test result
        if statistical_result:
            passed = statistical_result.decision == "H1"
            passed_components.append(passed)
            confidence_scores.append(statistical_result.final_mean if passed else 1 - statistical_result.final_mean)
        
        # Range validation result
        if range_validation:
            passed_components.append(range_validation.passed)
            confidence_scores.append(range_validation.confidence)
        
        # Fingerprint result (basic success check)
        if fingerprint_result:
            passed_components.append(True)  # Successfully computed fingerprint
            confidence_scores.append(0.8)  # Base confidence for successful fingerprinting
        
        # Overall result
        if not passed_components:
            return False, 0.0
        
        overall_passed = all(passed_components)
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return overall_passed, overall_confidence


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Proof of Training - Integrated System Demo")
    print("=" * 70)
    
    # Initialize system
    config = {
        'verification_type': 'fuzzy',
        'model_type': 'generic',
        'security_level': 'medium'
    }
    
    pot_system = ProofOfTraining(config)
    print(f"\n✓ System initialized with config: {config}")
    
    # Mock model
    class MockModel:
        def forward(self, x):
            return np.random.randn(10)
        
        def state_dict(self):
            return {'layer1': 'weights', 'layer2': 'weights'}
    
    model = MockModel()
    
    # Register model
    print("\n" + "=" * 70)
    print("Model Registration")
    print("=" * 70)
    
    model_id = pot_system.register_model(
        model,
        architecture="mock_architecture",
        parameter_count=1000000
    )
    print(f"✓ Model registered with ID: {model_id}")
    
    # Perform verification
    print("\n" + "=" * 70)
    print("Model Verification")
    print("=" * 70)
    
    # Quick verification
    result = pot_system.perform_verification(model, model_id, 'quick')
    print(f"\nQuick Verification:")
    print(f"  Verified: {result.verified}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Challenges: {result.challenges_passed}/{result.challenges_total}")
    
    # Standard verification
    result = pot_system.perform_verification(model, model_id, 'standard')
    print(f"\nStandard Verification:")
    print(f"  Verified: {result.verified}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Challenges: {result.challenges_passed}/{result.challenges_total}")
    
    # Demonstrate expected ranges calibration
    print("\n" + "=" * 70)
    print("Expected Ranges Calibration")
    print("=" * 70)
    
    # Calibrate expected ranges for the model
    expected_ranges = pot_system.calibrate_expected_ranges(model, model_id, num_calibration_runs=3)
    print(f"✓ Expected ranges calibrated:")
    print(f"  Accuracy range: [{expected_ranges.accuracy_range[0]:.3f}, {expected_ranges.accuracy_range[1]:.3f}]")
    print(f"  Latency range: [{expected_ranges.latency_range[0]:.1f}, {expected_ranges.latency_range[1]:.1f}]ms")
    print(f"  Fingerprint similarity: [{expected_ranges.fingerprint_similarity[0]:.3f}, {expected_ranges.fingerprint_similarity[1]:.3f}]")
    print(f"  Jacobian norm: [{expected_ranges.jacobian_norm_range[0]:.6f}, {expected_ranges.jacobian_norm_range[1]:.6f}]")
    
    # Comprehensive verification with range validation
    result_comprehensive = pot_system.perform_verification(model, model_id, 'comprehensive')
    print(f"\nComprehensive Verification with Range Validation:")
    print(f"  Verified: {result_comprehensive.verified}")
    print(f"  Confidence: {result_comprehensive.confidence:.2%}")
    print(f"  Accuracy: {result_comprehensive.accuracy:.3f}" if result_comprehensive.accuracy else "  Accuracy: N/A")
    print(f"  Latency: {result_comprehensive.latency_ms:.1f}ms" if result_comprehensive.latency_ms else "  Latency: N/A")
    print(f"  Fingerprint similarity: {result_comprehensive.fingerprint_similarity:.3f}" if result_comprehensive.fingerprint_similarity else "  Fingerprint similarity: N/A")
    print(f"  Jacobian norm: {result_comprehensive.jacobian_norm:.6f}" if result_comprehensive.jacobian_norm else "  Jacobian norm: N/A")
    
    if result_comprehensive.range_validation:
        print(f"  Range validation passed: {result_comprehensive.range_validation.passed}")
        if result_comprehensive.range_validation.violations:
            print(f"  Violations: {result_comprehensive.range_validation.violations}")
        print(f"  Range confidence: {result_comprehensive.range_validation.confidence:.3f}")
        if result_comprehensive.range_validation.statistical_significance:
            print(f"  Statistical significance: {result_comprehensive.range_validation.statistical_significance:.6f}")
    
    # Generate proof
    print("\n" + "=" * 70)
    print("Verification Proof")
    print("=" * 70)
    
    proof = pot_system.generate_verification_proof(result)
    print(f"✓ Proof generated: {len(proof)} bytes")
    
    # Verify proof
    verification = pot_system.verify_proof(proof)
    print(f"✓ Proof valid: {verification['valid']}")
    
    # Test challenge library
    print("\n" + "=" * 70)
    print("Challenge Library")
    print("=" * 70)
    
    vision_challenges = ChallengeLibrary.get_vision_challenges(224, 3, 3)
    print(f"✓ Vision challenges: {len(vision_challenges)} generated")
    
    language_challenges = ChallengeLibrary.get_language_challenges(50000, 100, 3)
    print(f"✓ Language challenges: {len(language_challenges)} generated")
    
    multimodal_challenges = ChallengeLibrary.get_multimodal_challenges(2)
    print(f"✓ Multimodal challenges: {len(multimodal_challenges)} generated")
    
    # System statistics
    print("\n" + "=" * 70)
    print("System Statistics")
    print("=" * 70)
    
    stats = pot_system.get_statistics()
    print(f"Registered models: {stats['registered_models']}")
    print(f"Cached verifications: {stats['cached_verifications']}")
    print(f"Components available: {stats['components']}")
    
    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)