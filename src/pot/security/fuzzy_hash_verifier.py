import logging
import hashlib
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HashResult:
    algorithm: str
    digest: str
    is_fuzzy: bool

class SHA256Hasher:
    """SHA-256 exact hash as fallback"""
    
    @staticmethod
    def generate_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def compare(h1: str, h2: str) -> float:
        """Exact comparison - 1.0 if identical, 0.0 otherwise"""
        return 1.0 if h1 == h2 else 0.0
    
    @staticmethod
    def prefix_similarity(h1: str, h2: str, prefix_len: int = 16) -> float:
        """Compare prefixes for cross-algorithm comparison"""
        p1 = h1[:prefix_len] if len(h1) >= prefix_len else h1
        p2 = h2[:prefix_len] if len(h2) >= prefix_len else h2
        
        if p1 == p2:
            return 1.0
        
        # Count matching chars
        matches = sum(1 for a, b in zip(p1, p2) if a == b)
        return matches / max(len(p1), len(p2))

class TLSHHasher:
    """TLSH fuzzy hasher"""
    
    def __init__(self):
        try:
            import tlsh
            self.tlsh = tlsh
            self.available = True
        except ImportError:
            self.available = False
            logger.info("TLSH not available - install with: pip install python-tlsh")
    
    def generate_hash(self, data: bytes) -> str:
        if not self.available:
            raise RuntimeError("TLSH not available")
        
        h = self.tlsh.hash(data)
        if not h:
            # TLSH needs minimum data size
            raise ValueError("Insufficient data for TLSH (needs 50+ bytes)")
        return h
    
    def compare(self, h1: str, h2: str) -> float:
        if not self.available:
            return 0.0
        
        # TLSH diff returns 0 for identical, higher for more different
        # Convert to similarity score 0-1
        diff = self.tlsh.diff(h1, h2)
        
        # Map diff to similarity (rough approximation)
        # diff=0 -> sim=1.0, diff=100 -> sim=0.5, diff=300+ -> sim=0.0
        if diff == 0:
            return 1.0
        elif diff < 100:
            return 1.0 - (diff / 200.0)
        elif diff < 300:
            return 0.5 - ((diff - 100) / 400.0)
        else:
            return 0.0

class SSDeepHasher:
    """ssdeep fuzzy hasher"""
    
    def __init__(self):
        try:
            import ssdeep
            self.ssdeep = ssdeep
            self.available = True
        except ImportError:
            self.available = False
            logger.info("ssdeep not available - install with: pip install ssdeep")
    
    def generate_hash(self, data: bytes) -> str:
        if not self.available:
            raise RuntimeError("ssdeep not available")
        return self.ssdeep.hash(data)
    
    def compare(self, h1: str, h2: str) -> float:
        if not self.available:
            return 0.0
        
        # ssdeep returns 0-100 similarity score
        return self.ssdeep.compare(h1, h2) / 100.0

class FuzzyHashVerifier:
    """Fuzzy hash verifier with TLSH/ssdeep preference"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize available hashers
        self.hashers = {}
        
        # Try TLSH first
        try:
            tlsh_hasher = TLSHHasher()
            if tlsh_hasher.available:
                self.hashers["tlsh"] = tlsh_hasher
                self.logger.info("TLSH hasher available")
        except Exception as e:
            self.logger.info(f"TLSH not available: {e}")
        
        # Try ssdeep
        try:
            ssdeep_hasher = SSDeepHasher()
            if ssdeep_hasher.available:
                self.hashers["ssdeep"] = ssdeep_hasher
                self.logger.info("ssdeep hasher available")
        except Exception as e:
            self.logger.info(f"ssdeep not available: {e}")
        
        # SHA-256 always available as fallback
        self.hashers["sha256"] = SHA256Hasher()
        
        # Preference order
        self.preferred_order = ["tlsh", "ssdeep", "sha256"]
    
    def get_hasher(self, preferred: Optional[str] = None):
        """Get hasher with preference"""
        if preferred and preferred in self.hashers:
            return preferred, self.hashers[preferred]
        
        # Use first available in preference order
        for algo in self.preferred_order:
            if algo in self.hashers:
                return algo, self.hashers[algo]
        
        # Shouldn't reach here, but fallback to SHA-256
        return "sha256", self.hashers["sha256"]
    
    def generate_fuzzy_hash(self, data: bytes, algorithm: Optional[str] = None) -> Dict[str, Any]:
        """Generate hash with algorithm labeling"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        algo, hasher = self.get_hasher(algorithm)
        
        try:
            digest = hasher.generate_hash(data)
            is_fuzzy = algo in ["tlsh", "ssdeep"]
            
            result = {
                "algorithm": algo if is_fuzzy else f"{algo} (exact)",
                "digest": digest,
                "is_fuzzy": is_fuzzy
            }
            
            if not is_fuzzy:
                self.logger.warning(f"Using {algo} as fallback - not a fuzzy hash")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Hash generation failed with {algo}: {e}")
            
            # Try fallback
            if algo != "sha256":
                self.logger.info("Falling back to SHA-256")
                return self.generate_fuzzy_hash(data, "sha256")
            raise
    
    def compare(self, h1: Dict[str, Any], h2: Dict[str, Any]) -> float:
        """Compare hashes with cross-algorithm support"""
        a1, d1 = h1.get("algorithm", "").replace(" (exact)", ""), h1["digest"]
        a2, d2 = h2.get("algorithm", "").replace(" (exact)", ""), h2["digest"]
        
        if a1 == a2 and a1 in self.hashers:
            # Same algorithm - use native comparison
            return self.hashers[a1].compare(d1, d2)
        else:
            # Cross-algorithm - use prefix comparison
            self.logger.warning(f"Cross-algorithm comparison: {a1} vs {a2}")
            return SHA256Hasher.prefix_similarity(d1, d2, prefix_len=16)
    
    def verify_similarity(self, h1: Dict[str, Any], h2: Dict[str, Any], threshold: float = 0.85) -> bool:
        """Verify if similarity meets threshold"""
        similarity = self.compare(h1, h2)
        
        # Log warning if using exact hash
        if not h1.get("is_fuzzy", False) or not h2.get("is_fuzzy", False):
            self.logger.warning("Using exact hash comparison - not fuzzy")
        
        return similarity >= threshold


# Legacy compatibility for existing code
from typing import Union, List, Tuple
import numpy as np
import time
from dataclasses import field
from enum import Enum


class HashAlgorithm(Enum):
    """Supported hash algorithms"""
    SSDEEP = "ssdeep"  # True fuzzy hashing
    TLSH = "tlsh"      # True fuzzy hashing (locality sensitive)
    SHA256 = "sha256"  # Exact hash (not fuzzy)


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


# Legacy wrapper class for backward compatibility
class LegacyFuzzyHashVerifier:
    """Legacy wrapper maintaining original API"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        algorithm: HashAlgorithm = HashAlgorithm.TLSH,
        fallback_to_exact: bool = True,
        log_level: int = logging.INFO
    ):
        self.similarity_threshold = similarity_threshold
        self.algorithm = algorithm
        self.fallback_to_exact = fallback_to_exact
        
        # Use new implementation
        self.verifier = FuzzyHashVerifier()
        
        # Map algorithm preference
        if algorithm == HashAlgorithm.TLSH:
            self.preferred_algo = "tlsh"
        elif algorithm == HashAlgorithm.SSDEEP:
            self.preferred_algo = "ssdeep"
        else:
            self.preferred_algo = "sha256"
        
        # Verification history
        self.verification_history: List[VerificationResult] = []
        
        logger.info(
            f"FuzzyHashVerifier initialized with algorithm={algorithm.value}, "
            f"threshold={similarity_threshold}"
        )
    
    def generate_fuzzy_hash(self, model_output: Union[np.ndarray, bytes]) -> str:
        """Generate fuzzy hash using preferred algorithm"""
        if isinstance(model_output, np.ndarray):
            model_output = model_output.tobytes()
        
        result = self.verifier.generate_fuzzy_hash(model_output, self.preferred_algo)
        return result["digest"]
    
    def verify_fuzzy(
        self,
        candidate_hash: str,
        reference_hash: str,
        threshold: Optional[float] = None
    ) -> VerificationResult:
        """Verify fuzzy hash with legacy API"""
        start_time = time.time()
        
        if threshold is None:
            threshold = self.similarity_threshold
        
        # Create hash objects for comparison
        h1 = {"algorithm": self.preferred_algo, "digest": candidate_hash, "is_fuzzy": self.preferred_algo != "sha256"}
        h2 = {"algorithm": self.preferred_algo, "digest": reference_hash, "is_fuzzy": self.preferred_algo != "sha256"}
        
        similarity = self.verifier.compare(h1, h2)
        is_valid = similarity >= threshold
        
        elapsed = time.time() - start_time
        
        result = VerificationResult(
            is_valid=is_valid,
            similarity_score=similarity,
            algorithm_used=self.algorithm,
            threshold_used=threshold,
            verification_time=elapsed,
            metadata={
                'candidate_hash_prefix': candidate_hash[:20] if candidate_hash else None,
                'reference_hash_prefix': reference_hash[:20] if reference_hash else None
            }
        )
        
        self.verification_history.append(result)
        return result
    
    def verify_model_output(
        self,
        model_output: Union[np.ndarray, bytes],
        reference_hash: str,
        threshold: Optional[float] = None
    ) -> VerificationResult:
        """Verify model output directly against reference hash"""
        candidate_hash = self.generate_fuzzy_hash(model_output)
        return self.verify_fuzzy(candidate_hash, reference_hash, threshold)
    
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
        
        return {
            'total_verifications': len(self.verification_history),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(self.verification_history),
            'average_similarity': avg_similarity,
            'average_time': avg_time,
            'current_threshold': self.similarity_threshold,
            'algorithm_used': self.algorithm.value
        }


# Example usage and testing
if __name__ == "__main__":
    # Test new implementation
    print("=" * 60)
    print("Testing New Fuzzy Hash Implementation")
    print("=" * 60)
    
    verifier = FuzzyHashVerifier()
    
    # Test data
    test_data = b"This is test data for fuzzy hashing verification"
    
    # Generate hash with preferred algorithm
    hash_result = verifier.generate_fuzzy_hash(test_data)
    print(f"Generated hash: {hash_result}")
    print(f"Algorithm: {hash_result['algorithm']}")
    print(f"Is fuzzy: {hash_result['is_fuzzy']}")
    print(f"Digest: {hash_result['digest'][:50]}...")
    
    # Test similarity
    hash_result2 = verifier.generate_fuzzy_hash(test_data + b" with modification")
    
    similarity = verifier.compare(hash_result, hash_result2)
    print(f"Similarity: {similarity:.4f}")
    
    is_similar = verifier.verify_similarity(hash_result, hash_result2, threshold=0.5)
    print(f"Similar (threshold=0.5): {is_similar}")
    
    # Test exact match
    hash_result3 = verifier.generate_fuzzy_hash(test_data)
    exact_similarity = verifier.compare(hash_result, hash_result3)
    print(f"Exact match similarity: {exact_similarity:.4f}")
    
    print("\nTesting legacy compatibility...")
    
    # Test legacy wrapper
    legacy_verifier = LegacyFuzzyHashVerifier()
    
    test_vector = np.random.randn(1000)
    legacy_hash = legacy_verifier.generate_fuzzy_hash(test_vector)
    print(f"Legacy hash: {legacy_hash[:50]}...")
    
    # Verify with modification
    modified_vector = test_vector + np.random.randn(1000) * 0.01
    result = legacy_verifier.verify_model_output(modified_vector, legacy_hash, threshold=0.8)
    print(f"Legacy verification: {result.is_valid} (similarity: {result.similarity_score:.4f})")
    
    stats = legacy_verifier.get_statistics()
    print(f"Legacy stats: {stats}")