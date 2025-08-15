"""
Fuzzy hashing for language model verification
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

try:
    import ssdeep   # pip install ssdeep
    HAS_SSDEEP = True
except ImportError:
    ssdeep = None
    HAS_SSDEEP = False

try:
    import tlsh     # pip install python-tlsh
    HAS_TLSH = True
except ImportError:
    tlsh = None
    HAS_TLSH = False


@dataclass
class FuzzyHashConfig:
    """Configuration for fuzzy hashing"""
    algo: str = "ssdeep"   # "ssdeep"|"tlsh"|"sha256"
    threshold: float = 0.85  # Similarity threshold for acceptance
    
    def __post_init__(self):
        if self.algo == "ssdeep" and not HAS_SSDEEP:
            print(f"Warning: ssdeep not available, falling back to sha256")
            self.algo = "sha256"
        if self.algo == "tlsh" and not HAS_TLSH:
            print(f"Warning: tlsh not available, falling back to sha256")
            self.algo = "sha256"


class FuzzyHashVerifier:
    """Fuzzy hash verifier for approximate matching"""
    
    def __init__(self, cfg: FuzzyHashConfig = None):
        self.cfg = cfg or FuzzyHashConfig()
    
    def hash_bytes(self, b: bytes) -> str:
        """
        Generate fuzzy hash from bytes
        
        Args:
            b: Bytes to hash
            
        Returns:
            Hash string
        """
        if self.cfg.algo == "ssdeep" and HAS_SSDEEP:
            return ssdeep.hash(b)
        elif self.cfg.algo == "tlsh" and HAS_TLSH:
            h = tlsh.Tlsh()
            h.update(b)
            h.final()
            return h.hexdigest()
        else:
            # Fallback to exact hash
            import hashlib
            return hashlib.sha256(b).hexdigest()
    
    def hash_text(self, text: str, canonicalize: bool = True) -> str:
        """
        Generate fuzzy hash from text
        
        Args:
            text: Text to hash
            canonicalize: Whether to canonicalize text first
            
        Returns:
            Hash string
        """
        if canonicalize:
            # Basic text canonicalization
            text = text.lower()
            # Collapse whitespace
            import re
            text = re.sub(r'\s+', ' ', text).strip()
        
        return self.hash_bytes(text.encode('utf-8'))
    
    def similarity(self, h1: str, h2: str) -> float:
        """
        Compute similarity between two hashes
        
        Args:
            h1: First hash
            h2: Second hash
            
        Returns:
            Similarity score in [0, 1]
        """
        if self.cfg.algo == "ssdeep" and HAS_SSDEEP:
            return ssdeep.compare(h1, h2) / 100.0
        elif self.cfg.algo == "tlsh" and HAS_TLSH:
            # TLSH returns distance, convert to similarity
            d = tlsh.diff(h1, h2)
            # Map distance to similarity (lower distance = higher similarity)
            # Use exponential decay
            return 1.0 / (1.0 + d / 100.0)
        else:
            # Exact match for SHA256
            return 1.0 if h1 == h2 else 0.0
    
    def verify(self, b1: bytes, b2: bytes) -> Dict[str, Any]:
        """
        Verify similarity between two byte sequences
        
        Args:
            b1: First byte sequence
            b2: Second byte sequence
            
        Returns:
            Dict with similarity score and acceptance decision
        """
        h1 = self.hash_bytes(b1)
        h2 = self.hash_bytes(b2)
        sim = self.similarity(h1, h2)
        
        return {
            "sim": sim,
            "accept": sim >= self.cfg.threshold,
            "h1": h1[:32] + "..." if len(h1) > 35 else h1,  # Truncate for display
            "h2": h2[:32] + "..." if len(h2) > 35 else h2,
            "algo": self.cfg.algo,
            "threshold": self.cfg.threshold
        }
    
    def verify_text(self, text1: str, text2: str, canonicalize: bool = True) -> Dict[str, Any]:
        """
        Verify similarity between two text strings
        
        Args:
            text1: First text
            text2: Second text
            canonicalize: Whether to canonicalize text first
            
        Returns:
            Dict with similarity score and acceptance decision
        """
        h1 = self.hash_text(text1, canonicalize)
        h2 = self.hash_text(text2, canonicalize)
        sim = self.similarity(h1, h2)
        
        return {
            "sim": sim,
            "accept": sim >= self.cfg.threshold,
            "h1": h1[:32] + "..." if len(h1) > 35 else h1,
            "h2": h2[:32] + "..." if len(h2) > 35 else h2,
            "algo": self.cfg.algo,
            "threshold": self.cfg.threshold,
            "canonicalized": canonicalize
        }
    
    def batch_verify(self, texts1: list, texts2: list, canonicalize: bool = True) -> Dict[str, Any]:
        """
        Batch verify multiple text pairs
        
        Args:
            texts1: List of first texts
            texts2: List of second texts
            canonicalize: Whether to canonicalize text first
            
        Returns:
            Dict with aggregate and per-pair results
        """
        if len(texts1) != len(texts2):
            raise ValueError("Text lists must have same length")
        
        results = []
        for t1, t2 in zip(texts1, texts2):
            res = self.verify_text(t1, t2, canonicalize)
            results.append(res)
        
        sims = [r["sim"] for r in results]
        accepts = [r["accept"] for r in results]
        
        return {
            "mean_sim": sum(sims) / len(sims) if sims else 0.0,
            "accept_rate": sum(accepts) / len(accepts) if accepts else 0.0,
            "all_accept": all(accepts),
            "any_accept": any(accepts),
            "n_pairs": len(results),
            "results": results
        }


def integrate_with_lm_verification(outputs1: list, outputs2: list, 
                                  fuzzy_config: Optional[FuzzyHashConfig] = None) -> Dict[str, Any]:
    """
    Integrate fuzzy hashing with LM verification pipeline
    
    Args:
        outputs1: First set of model outputs
        outputs2: Second set of model outputs
        fuzzy_config: Fuzzy hash configuration
        
    Returns:
        Verification results including fuzzy similarity
    """
    # Initialize verifier
    verifier = FuzzyHashVerifier(fuzzy_config)
    
    # Convert outputs to strings if needed
    texts1 = [str(o) for o in outputs1]
    texts2 = [str(o) for o in outputs2]
    
    # Perform batch verification
    fuzzy_results = verifier.batch_verify(texts1, texts2, canonicalize=True)
    
    # Also compute exact match rate for comparison
    exact_matches = sum(1 for t1, t2 in zip(texts1, texts2) if t1 == t2)
    exact_rate = exact_matches / len(texts1) if texts1 else 0.0
    
    return {
        "fuzzy": fuzzy_results,
        "exact_match_rate": exact_rate,
        "n_outputs": len(texts1),
        "recommendation": "accept" if fuzzy_results["all_accept"] else "reject",
        "confidence": fuzzy_results["mean_sim"]
    }