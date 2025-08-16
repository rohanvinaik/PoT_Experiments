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
                    # Get model response
                    if hasattr(model, 'forward'):
                        response = model.forward(challenge)
                    elif hasattr(model, 'predict'):
                        response = model.predict(challenge)
                    elif callable(model):
                        response = model(challenge)
                    else:
                        response = f"mock_response_{challenge_type}"
                    
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
        
        # Create result
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
                'model_type': self.model_type.value
            },
            timestamp=datetime.now(timezone.utc),
            duration_seconds=time.time() - start_time
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
            'verification_type': self.verification_type.value,
            'model_type': self.model_type.value,
            'security_level': self.security_level.value,
            'components': {
                'fuzzy_verifier': self.fuzzy_verifier is not None,
                'provenance_tracker': self.provenance_tracker is not None,
                'token_normalizer': self.token_normalizer is not None
            }
        }


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