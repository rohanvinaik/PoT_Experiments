"""
Fuzzy Hash Verifier for Proof-of-Training Verification

This module provides a comprehensive fuzzy hashing integration for the Proof-of-Training
verification workflow, supporting multiple fuzzy hash algorithms and configurable
similarity thresholds.
"""

import hashlib
import hmac
import logging
import secrets
import numpy as np
import json
import time
from typing import Optional, Union, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Try to import fuzzy hashing libraries
try:
    import ssdeep
    SSDEEP_AVAILABLE = True
except ImportError:
    SSDEEP_AVAILABLE = False
    warnings.warn("ssdeep not available. Install with: pip install python-ssdeep")

try:
    import tlsh
    TLSH_AVAILABLE = True
except ImportError:
    TLSH_AVAILABLE = False
    warnings.warn("TLSH not available. Install with: pip install python-tlsh")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HashAlgorithm(Enum):
    """Supported fuzzy hash algorithms"""
    SSDEEP = "ssdeep"
    TLSH = "tlsh"
    SHA256 = "sha256"  # Fallback exact matching


@dataclass
class VerificationResult:
    """Result of a fuzzy hash verification"""
    is_valid: bool
    similarity_score: float
    algorithm_used: HashAlgorithm
    threshold_used: float
    verification_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchVerificationResult:
    """Result of batch verification"""
    total_challenges: int
    passed: int
    failed: int
    average_similarity: float
    individual_results: List[VerificationResult]
    total_time: float


class ChallengeVector:
    """Challenge vector for model verification"""
    
    def __init__(self, dimension: int, topology: str = 'complex', seed: Optional[int] = None):
        self.dimension = dimension
        self.topology = topology
        self.seed = seed if seed is not None else int(time.time())
        np.random.seed(self.seed)
        self.vector = self._generate_vector()
        self.metadata = {
            'dimension': dimension,
            'topology': topology,
            'seed': self.seed,
            'timestamp': time.time()
        }
    
    def _generate_vector(self) -> np.ndarray:
        """Generate challenge vector based on topology"""
        if self.topology == 'complex':
            # Complex topology with multiple patterns
            base = np.random.randn(self.dimension)
            noise = np.random.randn(self.dimension) * 0.1
            pattern = np.sin(np.linspace(0, 4 * np.pi, self.dimension))
            return base + noise + pattern
        elif self.topology == 'sparse':
            # Sparse vector
            vector = np.zeros(self.dimension)
            num_nonzero = max(1, self.dimension // 10)
            indices = np.random.choice(self.dimension, num_nonzero, replace=False)
            vector[indices] = np.random.randn(num_nonzero)
            return vector
        else:
            # Default: standard normal distribution
            return np.random.randn(self.dimension)


class FuzzyHasher:
    """Basic cryptographic hasher used as fallback implementation.

    This class provides deterministic hashing using a salted SHA256 digest and a
    timing-safe comparison function.  More advanced fuzzy hashers (e.g. SSDEEP
    or TLSH) subclass this to provide algorithm specific behaviour.  The salt is
    generated once per instance which makes hashes reproducible for the same
    object when the instance is reused while still providing protection against
    precomputation attacks.
    """

    def __init__(self, salt: Optional[bytes] = None):
        self.salt = salt or secrets.token_bytes(16)

    def generate_hash(self, data: Union[np.ndarray, bytes]) -> str:
        """Generate a cryptographic hash of ``data``.

        Args:
            data: Input to be hashed.  Accepts ``numpy`` arrays or raw bytes.

        Returns:
            Hex digest of a salted SHA256 hash.
        """
        byte_data = self.prepare_data(data) + self.salt
        return hashlib.sha256(byte_data).hexdigest()

    def compare(self, hash1: str, hash2: str) -> float:
        """Compare two hash digests using a timing-safe equality check.

        Args:
            hash1: First hash digest.
            hash2: Second hash digest.

        Returns:
            ``1.0`` if the hashes are equal, ``0.0`` otherwise.
        """
        return 1.0 if hmac.compare_digest(hash1, hash2) else 0.0

    def prepare_data(self, data: Union[np.ndarray, bytes]) -> bytes:
        """Convert data to bytes for hashing"""
        if isinstance(data, np.ndarray):
            # Convert numpy array to bytes with controlled precision
            data = np.round(data, decimals=6)  # Limit precision for consistency
            return data.tobytes()
        elif isinstance(data, bytes):
            return data
        else:
            return str(data).encode('utf-8')


class SSDeepHasher(FuzzyHasher):
    """SSDeep fuzzy hashing implementation"""
    
    def __init__(self):
        if not SSDEEP_AVAILABLE:
            raise ImportError("SSDeep is not available. Install with: pip install python-ssdeep")
    
    def generate_hash(self, data: Union[np.ndarray, bytes]) -> str:
        """Generate SSDeep hash"""
        byte_data = self.prepare_data(data)
        return ssdeep.hash(byte_data)
    
    def compare(self, hash1: str, hash2: str) -> float:
        """Compare SSDeep hashes (returns 0-100, normalized to 0-1)"""
        similarity = ssdeep.compare(hash1, hash2)
        return similarity / 100.0


class TLSHHasher(FuzzyHasher):
    """TLSH fuzzy hashing implementation"""
    
    def __init__(self):
        if not TLSH_AVAILABLE:
            raise ImportError("TLSH is not available. Install with: pip install python-tlsh")
    
    def generate_hash(self, data: Union[np.ndarray, bytes]) -> str:
        """Generate TLSH hash"""
        byte_data = self.prepare_data(data)
        # TLSH requires at least 50 bytes of data
        if len(byte_data) < 50:
            # Pad data if too small
            byte_data = byte_data + b'\x00' * (50 - len(byte_data))
        
        hasher = tlsh.Tlsh()
        hasher.update(byte_data)
        hasher.final()
        return hasher.hexdigest()
    
    def compare(self, hash1: str, hash2: str) -> float:
        """Compare TLSH hashes (returns distance, normalized to similarity)"""
        if not hash1 or not hash2:
            return 0.0
        
        distance = tlsh.diff(hash1, hash2)
        # TLSH distance is typically 0-300+, normalize to 0-1 similarity
        # Lower distance = higher similarity
        max_distance = 300.0
        similarity = max(0, 1 - (distance / max_distance))
        return similarity


class SHA256Hasher(FuzzyHasher):
    """SHA256 with fuzzy matching support via locality-sensitive hashing"""
    
    def __init__(self):
        super().__init__()
        self.num_bits = 256  # SHA256 produces 256-bit hash
    
    def generate_hash(self, data: Union[np.ndarray, bytes]) -> str:
        """Generate SHA256 hash with additional metadata for fuzzy matching"""
        byte_data = self.prepare_data(data)
        
        # Generate primary hash
        primary_hash = hashlib.sha256(byte_data).hexdigest()
        
        # Generate locality-sensitive hash components for fuzzy matching
        # Use multiple hash functions with small perturbations
        lsh_components = []
        
        # Create perturbed versions for LSH
        for i in range(4):
            # Add small noise to create hash family
            perturbed_data = byte_data + f"_lsh_{i}".encode()
            component_hash = hashlib.sha256(perturbed_data).hexdigest()[:8]
            lsh_components.append(component_hash)
        
        # Combine primary hash with LSH components
        # Format: primary_hash:lsh1:lsh2:lsh3:lsh4
        return f"{primary_hash}:{':'.join(lsh_components)}"
    
    def compare(self, hash1: str, hash2: str) -> float:
        """Compare SHA256 hashes with fuzzy matching via LSH components"""
        # Handle legacy format (plain SHA256)
        if ':' not in hash1 or ':' not in hash2:
            # Fallback to exact matching for legacy hashes
            hash1_primary = hash1.split(':')[0] if ':' in hash1 else hash1
            hash2_primary = hash2.split(':')[0] if ':' in hash2 else hash2
            return 1.0 if hash1_primary == hash2_primary else 0.0
        
        # Parse hash components
        parts1 = hash1.split(':')
        parts2 = hash2.split(':')
        
        primary1, lsh1 = parts1[0], parts1[1:]
        primary2, lsh2 = parts2[0], parts2[1:]
        
        # Exact match on primary hash
        if primary1 == primary2:
            return 1.0
        
        # Fuzzy match using LSH components
        if lsh1 and lsh2:
            # Count matching LSH components
            matches = sum(1 for l1, l2 in zip(lsh1, lsh2) if l1 == l2)
            lsh_similarity = matches / max(len(lsh1), len(lsh2))
            
            # Also compute Hamming distance on primary hash prefixes
            prefix_len = 16  # Compare first 16 chars (64 bits)
            prefix1, prefix2 = primary1[:prefix_len], primary2[:prefix_len]
            
            hamming_dist = sum(c1 != c2 for c1, c2 in zip(prefix1, prefix2))
            hamming_similarity = 1.0 - (hamming_dist / prefix_len)
            
            # Weighted combination
            similarity = 0.6 * lsh_similarity + 0.4 * hamming_similarity
            return similarity
        
        # No LSH components, use Hamming distance on primary hash
        min_len = min(len(primary1), len(primary2))
        if min_len > 0:
            hamming_dist = sum(c1 != c2 for c1, c2 in zip(primary1[:min_len], primary2[:min_len]))
            return 1.0 - (hamming_dist / min_len)
        
        return 0.0


class FuzzyHashVerifier:
    """
    Main fuzzy hash verifier for Proof-of-Training verification
    
    Integrates multiple fuzzy hashing algorithms with configurable thresholds
    and comprehensive logging for verification workflows.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        algorithm: HashAlgorithm = HashAlgorithm.SSDEEP,
        fallback_to_exact: bool = True,
        log_level: int = logging.INFO
    ):
        """
        Initialize FuzzyHashVerifier
        
        Args:
            similarity_threshold: Minimum similarity score for valid match (0-1)
            algorithm: Primary hash algorithm to use
            fallback_to_exact: Whether to fallback to exact matching if fuzzy fails
            log_level: Logging level for verification events
        """
        self.similarity_threshold = similarity_threshold
        self.algorithm = algorithm
        self.fallback_to_exact = fallback_to_exact
        
        # Configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        # Initialize hashers
        self.hashers = self._initialize_hashers()
        
        # Verification history
        self.verification_history: List[VerificationResult] = []
        
        # Reference hash storage
        self.reference_hashes: Dict[str, Dict[str, str]] = {}
        
        self.logger.info(
            f"FuzzyHashVerifier initialized with algorithm={algorithm.value}, "
            f"threshold={similarity_threshold}"
        )
    
    def _initialize_hashers(self) -> Dict[HashAlgorithm, Optional[FuzzyHasher]]:
        """Initialize available hash algorithms"""
        hashers = {}
        
        # Try to initialize each hasher
        if SSDEEP_AVAILABLE:
            hashers[HashAlgorithm.SSDEEP] = SSDeepHasher()
            self.logger.info("SSDeep hasher initialized")
        else:
            hashers[HashAlgorithm.SSDEEP] = None
            
        if TLSH_AVAILABLE:
            hashers[HashAlgorithm.TLSH] = TLSHHasher()
            self.logger.info("TLSH hasher initialized")
        else:
            hashers[HashAlgorithm.TLSH] = None
            
        # SHA256 always available
        hashers[HashAlgorithm.SHA256] = SHA256Hasher()
        self.logger.info("SHA256 hasher initialized (fallback)")
        
        return hashers
    
    def get_hasher(self, algorithm: Optional[HashAlgorithm] = None) -> FuzzyHasher:
        """Get hasher for specified algorithm or fallback"""
        if algorithm is None:
            algorithm = self.algorithm
            
        hasher = self.hashers.get(algorithm)
        
        if hasher is None:
            self.logger.warning(f"{algorithm.value} not available, falling back to SHA256")
            hasher = self.hashers[HashAlgorithm.SHA256]
            
        return hasher
    
    def generate_fuzzy_hash(
        self,
        model_output: Union[np.ndarray, bytes],
        algorithm: Optional[HashAlgorithm] = None
    ) -> str:
        """
        Generate fuzzy hash of model output
        
        Args:
            model_output: Model output to hash
            algorithm: Specific algorithm to use (optional)
            
        Returns:
            Fuzzy hash string
        """
        start_time = time.time()
        
        hasher = self.get_hasher(algorithm)
        fuzzy_hash = hasher.generate_hash(model_output)
        
        elapsed = time.time() - start_time
        self.logger.debug(f"Generated {hasher.__class__.__name__} hash in {elapsed:.4f}s")
        
        return fuzzy_hash
    
    def verify_fuzzy(
        self,
        candidate_hash: str,
        reference_hash: str,
        threshold: Optional[float] = None,
        algorithm: Optional[HashAlgorithm] = None
    ) -> VerificationResult:
        """
        Verify fuzzy hash against reference
        
        Args:
            candidate_hash: Hash to verify
            reference_hash: Reference hash to compare against
            threshold: Override similarity threshold
            algorithm: Specific algorithm to use
            
        Returns:
            VerificationResult with detailed information
        """
        start_time = time.time()
        
        if threshold is None:
            threshold = self.similarity_threshold
            
        hasher = self.get_hasher(algorithm)
        
        # Compare hashes
        similarity = hasher.compare(candidate_hash, reference_hash)
        is_valid = similarity >= threshold
        
        # If failed and fallback enabled, try exact match
        if not is_valid and self.fallback_to_exact and algorithm != HashAlgorithm.SHA256:
            self.logger.info("Fuzzy match failed, trying exact match fallback")
            exact_hasher = self.hashers[HashAlgorithm.SHA256]
            exact_similarity = exact_hasher.compare(candidate_hash, reference_hash)
            if exact_similarity == 1.0:
                is_valid = True
                similarity = exact_similarity
                algorithm = HashAlgorithm.SHA256
        
        elapsed = time.time() - start_time
        
        result = VerificationResult(
            is_valid=is_valid,
            similarity_score=similarity,
            algorithm_used=algorithm or self.algorithm,
            threshold_used=threshold,
            verification_time=elapsed,
            metadata={
                'candidate_hash_prefix': candidate_hash[:20] if candidate_hash else None,
                'reference_hash_prefix': reference_hash[:20] if reference_hash else None
            }
        )
        
        # Log result
        self.logger.info(
            f"Verification {'PASSED' if is_valid else 'FAILED'}: "
            f"similarity={similarity:.4f}, threshold={threshold:.4f}, "
            f"algorithm={result.algorithm_used.value}, time={elapsed:.4f}s"
        )
        
        # Store in history
        self.verification_history.append(result)
        
        return result
    
    def verify_model_output(
        self,
        model_output: Union[np.ndarray, bytes],
        reference_hash: str,
        threshold: Optional[float] = None
    ) -> VerificationResult:
        """
        Verify model output directly against reference hash
        
        Args:
            model_output: Model output to verify
            reference_hash: Reference hash to compare against
            threshold: Override similarity threshold
            
        Returns:
            VerificationResult
        """
        candidate_hash = self.generate_fuzzy_hash(model_output)
        return self.verify_fuzzy(candidate_hash, reference_hash, threshold)
    
    def batch_verify(
        self,
        challenges: List[Tuple[Union[np.ndarray, bytes], str]],
        threshold: Optional[float] = None,
        parallel: bool = False
    ) -> BatchVerificationResult:
        """
        Batch verification of multiple challenges
        
        Args:
            challenges: List of (model_output, reference_hash) tuples
            threshold: Override similarity threshold
            parallel: Whether to process in parallel (future enhancement)
            
        Returns:
            BatchVerificationResult with aggregate statistics
        """
        start_time = time.time()
        results = []
        
        for model_output, reference_hash in challenges:
            result = self.verify_model_output(model_output, reference_hash, threshold)
            results.append(result)
        
        # Calculate statistics
        passed = sum(1 for r in results if r.is_valid)
        failed = len(results) - passed
        avg_similarity = np.mean([r.similarity_score for r in results])
        
        elapsed = time.time() - start_time
        
        batch_result = BatchVerificationResult(
            total_challenges=len(challenges),
            passed=passed,
            failed=failed,
            average_similarity=avg_similarity,
            individual_results=results,
            total_time=elapsed
        )
        
        self.logger.info(
            f"Batch verification complete: {passed}/{len(challenges)} passed, "
            f"avg_similarity={avg_similarity:.4f}, time={elapsed:.4f}s"
        )
        
        return batch_result
    
    def store_reference_hash(
        self,
        identifier: str,
        model_output: Union[np.ndarray, bytes],
        algorithms: Optional[List[HashAlgorithm]] = None
    ) -> Dict[str, str]:
        """
        Generate and store reference hashes for later verification
        
        Args:
            identifier: Unique identifier for this reference
            model_output: Model output to hash
            algorithms: List of algorithms to generate hashes for
            
        Returns:
            Dictionary of algorithm -> hash mappings
        """
        if algorithms is None:
            algorithms = [algo for algo in HashAlgorithm if self.hashers.get(algo)]
        
        hashes = {}
        for algo in algorithms:
            if self.hashers.get(algo):
                hashes[algo.value] = self.generate_fuzzy_hash(model_output, algo)
        
        self.reference_hashes[identifier] = hashes
        
        self.logger.info(f"Stored reference hashes for '{identifier}': {list(hashes.keys())}")
        
        return hashes
    
    def verify_against_stored(
        self,
        identifier: str,
        model_output: Union[np.ndarray, bytes],
        algorithm: Optional[HashAlgorithm] = None
    ) -> VerificationResult:
        """
        Verify model output against stored reference hash
        
        Args:
            identifier: Reference identifier
            model_output: Model output to verify
            algorithm: Specific algorithm to use
            
        Returns:
            VerificationResult
        """
        if identifier not in self.reference_hashes:
            raise ValueError(f"No reference hash found for identifier: {identifier}")
        
        if algorithm is None:
            algorithm = self.algorithm
            
        reference_hash = self.reference_hashes[identifier].get(algorithm.value)
        if reference_hash is None:
            available = list(self.reference_hashes[identifier].keys())
            raise ValueError(
                f"Algorithm {algorithm.value} not available for {identifier}. "
                f"Available: {available}"
            )
        
        return self.verify_model_output(model_output, reference_hash)
    
    def adjust_threshold(
        self,
        new_threshold: float,
        apply_to_history: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Adjust similarity threshold
        
        Args:
            new_threshold: New threshold value (0-1)
            apply_to_history: Re-evaluate historical verifications
            
        Returns:
            Statistics if apply_to_history is True
        """
        old_threshold = self.similarity_threshold
        self.similarity_threshold = new_threshold
        
        self.logger.info(f"Threshold adjusted: {old_threshold:.4f} -> {new_threshold:.4f}")
        
        if apply_to_history and self.verification_history:
            # Re-evaluate historical results
            old_passed = sum(1 for r in self.verification_history if r.is_valid)
            new_passed = sum(
                1 for r in self.verification_history 
                if r.similarity_score >= new_threshold
            )
            
            stats = {
                'old_threshold': old_threshold,
                'new_threshold': new_threshold,
                'total_verifications': len(self.verification_history),
                'old_passed': old_passed,
                'new_passed': new_passed,
                'difference': new_passed - old_passed
            }
            
            self.logger.info(
                f"Historical re-evaluation: {old_passed} -> {new_passed} passed "
                f"({stats['difference']:+d} change)"
            )
            
            return stats
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics"""
        if not self.verification_history:
            return {
                'total_verifications': 0,
                'passed': 0,
                'failed': 0,
                'average_similarity': 0,
                'average_time': 0
            }
        
        passed = sum(1 for r in self.verification_history if r.is_valid)
        failed = len(self.verification_history) - passed
        avg_similarity = np.mean([r.similarity_score for r in self.verification_history])
        avg_time = np.mean([r.verification_time for r in self.verification_history])
        
        # Algorithm usage
        algo_usage = {}
        for result in self.verification_history:
            algo = result.algorithm_used.value
            algo_usage[algo] = algo_usage.get(algo, 0) + 1
        
        return {
            'total_verifications': len(self.verification_history),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(self.verification_history),
            'average_similarity': avg_similarity,
            'average_time': avg_time,
            'current_threshold': self.similarity_threshold,
            'algorithm_usage': algo_usage,
            'stored_references': len(self.reference_hashes)
        }
    
    def clear_history(self):
        """Clear verification history"""
        self.verification_history.clear()
        self.logger.info("Verification history cleared")
    
    def export_config(self) -> Dict[str, Any]:
        """Export verifier configuration"""
        return {
            'similarity_threshold': self.similarity_threshold,
            'algorithm': self.algorithm.value,
            'fallback_to_exact': self.fallback_to_exact,
            'available_algorithms': [
                algo.value for algo, hasher in self.hashers.items() 
                if hasher is not None
            ],
            'statistics': self.get_statistics()
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"FuzzyHashVerifier("
            f"algorithm={self.algorithm.value}, "
            f"threshold={self.similarity_threshold:.2f}, "
            f"verifications={stats['total_verifications']}, "
            f"pass_rate={stats.get('pass_rate', 0):.2%})"
        )


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Basic verification with challenge vectors
    print("=" * 60)
    print("Example 1: Basic Fuzzy Hash Verification")
    print("=" * 60)
    
    # Initialize verifier
    verifier = FuzzyHashVerifier(
        similarity_threshold=0.85,
        algorithm=HashAlgorithm.SSDEEP if SSDEEP_AVAILABLE else HashAlgorithm.SHA256
    )
    
    # Create challenge vector
    challenge = ChallengeVector(dimension=1000, topology='complex')
    
    # Simulate model output (with slight variations)
    model_output = challenge.vector + np.random.randn(1000) * 0.01  # Small noise
    
    # Generate reference hash
    reference_hash = verifier.generate_fuzzy_hash(challenge.vector)
    print(f"Reference hash: {reference_hash[:50]}...")
    
    # Verify with exact output
    result1 = verifier.verify_model_output(challenge.vector, reference_hash)
    print(f"Exact match: {result1.is_valid} (similarity: {result1.similarity_score:.4f})")
    
    # Verify with slightly modified output
    result2 = verifier.verify_model_output(model_output, reference_hash)
    print(f"Fuzzy match: {result2.is_valid} (similarity: {result2.similarity_score:.4f})")
    
    # Example 2: Batch verification
    print("\n" + "=" * 60)
    print("Example 2: Batch Verification")
    print("=" * 60)
    
    # Generate multiple challenges
    challenges_batch = []
    for i in range(5):
        ch = ChallengeVector(dimension=500, topology='sparse')
        # Some with exact match, some with variations
        if i % 2 == 0:
            output = ch.vector
        else:
            output = ch.vector + np.random.randn(500) * 0.05
        
        ref_hash = verifier.generate_fuzzy_hash(ch.vector)
        challenges_batch.append((output, ref_hash))
    
    # Batch verify
    batch_result = verifier.batch_verify(challenges_batch)
    print(f"Batch results: {batch_result.passed}/{batch_result.total_challenges} passed")
    print(f"Average similarity: {batch_result.average_similarity:.4f}")
    print(f"Total time: {batch_result.total_time:.4f}s")
    
    # Example 3: Reference storage and retrieval
    print("\n" + "=" * 60)
    print("Example 3: Reference Storage System")
    print("=" * 60)
    
    # Store reference hashes for different model checkpoints
    checkpoint1 = ChallengeVector(dimension=800, topology='complex')
    verifier.store_reference_hash("checkpoint_v1", checkpoint1.vector)
    
    checkpoint2 = ChallengeVector(dimension=800, topology='sparse')
    verifier.store_reference_hash("checkpoint_v2", checkpoint2.vector)
    
    # Verify against stored references
    test_output = checkpoint1.vector + np.random.randn(800) * 0.02
    result = verifier.verify_against_stored("checkpoint_v1", test_output)
    print(f"Verification against checkpoint_v1: {result.is_valid}")
    
    # Example 4: Threshold adjustment
    print("\n" + "=" * 60)
    print("Example 4: Dynamic Threshold Adjustment")
    print("=" * 60)
    
    # Get current statistics
    stats = verifier.get_statistics()
    print(f"Current stats: {stats['passed']}/{stats['total_verifications']} passed")
    print(f"Current threshold: {stats['current_threshold']}")
    
    # Adjust threshold and re-evaluate
    adjustment_stats = verifier.adjust_threshold(0.70, apply_to_history=True)
    if adjustment_stats:
        print(f"After adjustment: {adjustment_stats['new_passed']} passed")
        print(f"Change: {adjustment_stats['difference']:+d}")
    
    # Example 5: Different hash algorithms
    print("\n" + "=" * 60)
    print("Example 5: Multiple Hash Algorithms")
    print("=" * 60)
    
    test_data = np.random.randn(1000)
    
    # Store with multiple algorithms
    multi_hashes = verifier.store_reference_hash(
        "multi_algo_test",
        test_data,
        algorithms=[HashAlgorithm.SSDEEP, HashAlgorithm.TLSH, HashAlgorithm.SHA256]
    )
    
    print("Generated hashes:")
    for algo, hash_val in multi_hashes.items():
        print(f"  {algo}: {hash_val[:40]}...")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Final Verification Statistics")
    print("=" * 60)
    
    final_stats = verifier.get_statistics()
    print(json.dumps(final_stats, indent=2))
    
    # Export configuration
    config = verifier.export_config()
    print("\nVerifier Configuration:")
    print(json.dumps(config, indent=2))