"""
Fuzzy hashing for language models with n-gram approach
Based on Definition 1 from paper Section 3.1

Handles tokenization variability by computing hashes over token n-grams
"""

import hashlib
import xxhash
from typing import List, Set, Dict, Tuple, Optional
from collections import Counter
import numpy as np

try:
    import ssdeep
    SSDEEP_AVAILABLE = True
except ImportError:
    SSDEEP_AVAILABLE = False

try:
    import tlsh
    TLSH_AVAILABLE = True
except ImportError:
    TLSH_AVAILABLE = False


class NGramFuzzyHasher:
    """
    Token-Level Fuzzy Hash implementation from paper Definition 1
    
    For token sequence s = [t_1, ..., t_k]:
    H_fuzzy(s) = {h(n-gram) : n-gram ∈ s, n ∈ {2,3,4}}
    
    This handles tokenization variability in language models
    """
    
    def __init__(self, n_values: List[int] = [2, 3, 4], 
                 hash_func: str = 'xxhash'):
        """
        Initialize n-gram fuzzy hasher
        
        Args:
            n_values: List of n-gram sizes to use (default: [2,3,4] from paper)
            hash_func: Hash function to use ('xxhash', 'sha256', 'md5')
        """
        self.n_values = n_values
        self.hash_func = hash_func
        
    def _hash_ngram(self, ngram: Tuple[int, ...]) -> str:
        """Hash a single n-gram"""
        # Convert tuple of token IDs to bytes
        ngram_bytes = b''.join(t.to_bytes(4, 'big') for t in ngram)
        
        if self.hash_func == 'xxhash':
            return xxhash.xxh3_64_hexdigest(ngram_bytes)
        elif self.hash_func == 'sha256':
            return hashlib.sha256(ngram_bytes).hexdigest()[:16]
        elif self.hash_func == 'md5':
            return hashlib.md5(ngram_bytes).hexdigest()[:16]
        else:
            raise ValueError(f"Unknown hash function: {self.hash_func}")
    
    def extract_ngrams(self, tokens: List[int], n: int) -> List[Tuple[int, ...]]:
        """Extract all n-grams from token sequence"""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def compute_fuzzy_hash(self, tokens: List[int]) -> Set[str]:
        """
        Compute fuzzy hash for token sequence
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Set of n-gram hashes
        """
        hashes = set()
        
        for n in self.n_values:
            ngrams = self.extract_ngrams(tokens, n)
            for ngram in ngrams:
                hashes.add(self._hash_ngram(ngram))
        
        return hashes
    
    def jaccard_similarity(self, hash1: Set[str], hash2: Set[str]) -> float:
        """
        Compute Jaccard similarity between two fuzzy hashes
        
        J(H1, H2) = |H1 ∩ H2| / |H1 ∪ H2|
        """
        if not hash1 and not hash2:
            return 1.0
        if not hash1 or not hash2:
            return 0.0
        
        intersection = len(hash1 & hash2)
        union = len(hash1 | hash2)
        
        return intersection / union if union > 0 else 0.0
    
    def containment_similarity(self, hash1: Set[str], hash2: Set[str]) -> float:
        """
        Compute containment similarity (asymmetric)
        
        C(H1, H2) = |H1 ∩ H2| / |H1|
        
        Useful for checking if one output contains another
        """
        if not hash1:
            return 0.0 if hash2 else 1.0
        
        intersection = len(hash1 & hash2)
        return intersection / len(hash1)
    
    def weighted_similarity(self, tokens1: List[int], tokens2: List[int]) -> float:
        """
        Compute weighted similarity considering n-gram frequencies
        """
        # Count n-gram frequencies
        freq1 = Counter()
        freq2 = Counter()
        
        for n in self.n_values:
            for ngram in self.extract_ngrams(tokens1, n):
                freq1[self._hash_ngram(ngram)] += 1
            for ngram in self.extract_ngrams(tokens2, n):
                freq2[self._hash_ngram(ngram)] += 1
        
        # Compute weighted Jaccard
        all_hashes = set(freq1.keys()) | set(freq2.keys())
        if not all_hashes:
            return 1.0
        
        intersection_weight = sum(min(freq1[h], freq2[h]) for h in all_hashes)
        union_weight = sum(max(freq1[h], freq2[h]) for h in all_hashes)
        
        return intersection_weight / union_weight if union_weight > 0 else 0.0


class TokenSpaceNormalizer:
    """
    Handle tokenization variability in language models
    Implements fuzzy matching from paper Section 3.1
    """
    
    def __init__(self, tokenizer=None):
        """
        Initialize normalizer
        
        Args:
            tokenizer: HuggingFace tokenizer instance
        """
        self.tokenizer = tokenizer
        self.fuzzy_hasher = NGramFuzzyHasher()
        
    def normalize_tokens(self, tokens: List[int]) -> List[int]:
        """
        Normalize token sequence for comparison
        
        - Remove padding tokens
        - Handle special tokens consistently
        - Optionally merge split tokens
        """
        if not self.tokenizer:
            # Basic normalization without tokenizer
            return [t for t in tokens if t != 0]  # Remove padding (assumed 0)
        
        # Remove padding and special tokens
        normalized = []
        for token_id in tokens:
            if token_id == self.tokenizer.pad_token_id:
                continue
            if token_id in [self.tokenizer.bos_token_id, 
                           self.tokenizer.eos_token_id]:
                continue  # Optionally keep these
            normalized.append(token_id)
        
        return normalized
    
    def compute_distance(self, tokens1: List[int], tokens2: List[int], 
                        method: str = 'fuzzy') -> float:
        """
        Compute distance between token sequences
        
        Args:
            tokens1: First token sequence
            tokens2: Second token sequence  
            method: 'fuzzy', 'exact', 'weighted'
            
        Returns:
            Distance in [0, 1] where 0 = identical
        """
        # Normalize both sequences
        norm1 = self.normalize_tokens(tokens1)
        norm2 = self.normalize_tokens(tokens2)
        
        if method == 'exact':
            # Exact match
            return 0.0 if norm1 == norm2 else 1.0
            
        elif method == 'fuzzy':
            # Fuzzy hash based similarity
            hash1 = self.fuzzy_hasher.compute_fuzzy_hash(norm1)
            hash2 = self.fuzzy_hasher.compute_fuzzy_hash(norm2)
            similarity = self.fuzzy_hasher.jaccard_similarity(hash1, hash2)
            return 1.0 - similarity
            
        elif method == 'weighted':
            # Weighted n-gram similarity
            similarity = self.fuzzy_hasher.weighted_similarity(norm1, norm2)
            return 1.0 - similarity
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def is_equivalent(self, tokens1: List[int], tokens2: List[int], 
                      threshold: float = 0.1) -> bool:
        """
        Check if two token sequences are equivalent under fuzzy matching
        
        Args:
            tokens1: First token sequence
            tokens2: Second token sequence
            threshold: Maximum distance for equivalence
            
        Returns:
            True if sequences are equivalent
        """
        distance = self.compute_distance(tokens1, tokens2, method='fuzzy')
        return distance <= threshold


class AdvancedFuzzyHasher:
    """
    Advanced fuzzy hashing using ssdeep and TLSH libraries
    Provides stronger locality-sensitive hashing
    """
    
    def __init__(self):
        self.ssdeep_available = SSDEEP_AVAILABLE
        self.tlsh_available = TLSH_AVAILABLE
        
        if not self.ssdeep_available and not self.tlsh_available:
            raise ImportError("Neither ssdeep nor tlsh is available. "
                            "Install with: pip install ssdeep py-tlsh")
    
    def tokens_to_bytes(self, tokens: List[int]) -> bytes:
        """Convert token list to bytes for hashing"""
        # Pack tokens as 4-byte integers
        return b''.join(t.to_bytes(4, 'big') for t in tokens)
    
    def compute_ssdeep(self, tokens: List[int]) -> Optional[str]:
        """Compute ssdeep fuzzy hash"""
        if not self.ssdeep_available:
            return None
        
        data = self.tokens_to_bytes(tokens)
        return ssdeep.hash(data)
    
    def compute_tlsh(self, tokens: List[int]) -> Optional[str]:
        """Compute TLSH fuzzy hash"""
        if not self.tlsh_available:
            return None
        
        data = self.tokens_to_bytes(tokens)
        
        # TLSH requires minimum data size
        if len(data) < 50:
            # Pad with zeros if too small
            data = data + b'\x00' * (50 - len(data))
        
        h = tlsh.hash(data)
        return h if h else None
    
    def compare_ssdeep(self, hash1: str, hash2: str) -> float:
        """Compare two ssdeep hashes (0-100 similarity score)"""
        if not self.ssdeep_available:
            return 0.0
        
        score = ssdeep.compare(hash1, hash2)
        return score / 100.0  # Normalize to [0, 1]
    
    def compare_tlsh(self, hash1: str, hash2: str) -> float:
        """Compare two TLSH hashes (lower distance = more similar)"""
        if not self.tlsh_available:
            return 1.0
        
        # TLSH diff returns distance (0 = identical, higher = more different)
        # We normalize to similarity score
        distance = tlsh.diff(hash1, hash2)
        
        # Convert distance to similarity (rough approximation)
        # TLSH distances can be 0-400+, we cap at 200 for normalization
        max_distance = 200
        similarity = max(0, 1 - (distance / max_distance))
        
        return similarity
    
    def compute_combined_similarity(self, tokens1: List[int], 
                                   tokens2: List[int]) -> Dict[str, float]:
        """
        Compute similarity using all available methods
        
        Returns:
            Dictionary with similarity scores for each method
        """
        results = {}
        
        if self.ssdeep_available:
            hash1 = self.compute_ssdeep(tokens1)
            hash2 = self.compute_ssdeep(tokens2)
            if hash1 and hash2:
                results['ssdeep'] = self.compare_ssdeep(hash1, hash2)
        
        if self.tlsh_available:
            hash1 = self.compute_tlsh(tokens1)
            hash2 = self.compute_tlsh(tokens2)
            if hash1 and hash2:
                results['tlsh'] = self.compare_tlsh(hash1, hash2)
        
        # Also compute n-gram based similarity
        hasher = NGramFuzzyHasher()
        h1 = hasher.compute_fuzzy_hash(tokens1)
        h2 = hasher.compute_fuzzy_hash(tokens2)
        results['ngram'] = hasher.jaccard_similarity(h1, h2)
        
        return results