"""
Fuzzy Hashing Components for Language Model Verification
Implements similarity-preserving hash functions for efficient text comparison
"""

import hashlib
import struct
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from collections import defaultdict
import numpy as np


class FuzzyHasher:
    """
    Fuzzy hasher for similarity-preserving hashing.
    Supports multiple hash algorithms for different use cases.
    """
    
    def __init__(self, hash_type: str = 'simhash'):
        """
        Initialize fuzzy hasher for similarity-preserving hashing.
        
        Args:
            hash_type: Type of hash - 'ssdeep', 'tlsh', 'simhash', 'minhash'
        """
        self.hash_type = hash_type
        
        # Initialize parameters based on hash type
        if hash_type == 'minhash':
            self.num_hashes = 128
            self._init_minhash_functions()
        elif hash_type == 'simhash':
            self.hash_bits = 64
        elif hash_type == 'ssdeep':
            self.block_size_min = 3
            self.spamsum_length = 64
        elif hash_type == 'tlsh':
            self.bucket_count = 256
            self.checksum_length = 1
    
    def compute_hash(self, text: str) -> Union[str, int, List[int]]:
        """
        Compute fuzzy hash of text.
        
        Args:
            text: Input text
            
        Returns:
            Hash value (format depends on hash_type)
        """
        if not text:
            return self._empty_hash()
        
        if self.hash_type == 'ssdeep':
            return self._ssdeep_hash(text)
        elif self.hash_type == 'tlsh':
            return self._tlsh_hash(text)
        elif self.hash_type == 'simhash':
            return self._simhash(text)
        elif self.hash_type == 'minhash':
            return self._minhash(text)
        else:
            raise ValueError(f"Unknown hash type: {self.hash_type}")
    
    def _empty_hash(self) -> Union[str, int, List[int]]:
        """Return empty hash for the current hash type."""
        if self.hash_type == 'minhash':
            return [0] * self.num_hashes
        elif self.hash_type == 'simhash':
            return 0
        else:
            return ''
    
    def _ssdeep_hash(self, text: str) -> str:
        """
        Context-triggered piecewise hashing (simplified SSDeep).
        
        Args:
            text: Input text
            
        Returns:
            SSDeep-style hash string
        """
        # Compute block size based on text length
        block_size = self._compute_block_size(len(text))
        
        # Get rolling hash chunks
        chunks = self._rolling_hash_chunks(text, block_size)
        
        # Encode chunks
        hash_string = self._encode_chunks(chunks)
        
        # Format: block_size:hash1:hash2
        # We'll use a simplified version with just one hash
        return f"{block_size}:{hash_string}"
    
    def _compute_block_size(self, text_length: int) -> int:
        """
        Compute appropriate block size for SSDeep.
        
        Args:
            text_length: Length of input text
            
        Returns:
            Block size
        """
        block_size = self.block_size_min
        
        while block_size * self.spamsum_length < text_length:
            block_size *= 2
        
        return block_size
    
    def _rolling_hash_chunks(self, text: str, block_size: int) -> List[int]:
        """
        Generate chunks using rolling hash with context triggers.
        
        Args:
            text: Input text
            block_size: Size of blocks
            
        Returns:
            List of chunk hashes
        """
        chunks = []
        rolling_hash = 0
        trigger = block_size
        
        for i, char in enumerate(text):
            # Update rolling hash
            rolling_hash = (rolling_hash * 7 + ord(char)) % (2**32)
            
            # Check for context trigger
            if rolling_hash % trigger == trigger - 1:
                # Store chunk hash
                chunk_data = text[max(0, i - block_size):i + 1]
                chunk_hash = hash(chunk_data) % 64
                chunks.append(chunk_hash)
        
        # Ensure we have at least some chunks
        if not chunks:
            for i in range(0, len(text), block_size):
                chunk = text[i:i + block_size]
                if chunk:
                    chunks.append(hash(chunk) % 64)
        
        return chunks[:self.spamsum_length]
    
    def _encode_chunks(self, chunks: List[int]) -> str:
        """
        Encode chunk hashes as base64-like string.
        
        Args:
            chunks: List of chunk hashes
            
        Returns:
            Encoded string
        """
        # Base64-like alphabet for encoding
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        
        encoded = ''.join(alphabet[c % 64] for c in chunks)
        return encoded
    
    def _tlsh_hash(self, text: str) -> str:
        """
        Compute TLSH (Trend Micro Locality Sensitive Hash).
        Simplified implementation.
        
        Args:
            text: Input text
            
        Returns:
            TLSH hash string
        """
        # Extract trigrams
        trigrams = self._get_trigrams(text)
        
        # Build histogram
        buckets = [0] * self.bucket_count
        for trigram in trigrams:
            bucket_idx = hash(trigram) % self.bucket_count
            buckets[bucket_idx] = min(buckets[bucket_idx] + 1, 255)
        
        # Compute quartiles
        sorted_buckets = sorted(buckets)
        q1 = sorted_buckets[len(sorted_buckets) // 4]
        q2 = sorted_buckets[len(sorted_buckets) // 2]
        q3 = sorted_buckets[3 * len(sorted_buckets) // 4]
        
        # Generate hash body
        hash_body = []
        for i in range(0, self.bucket_count, 4):
            nibble = 0
            for j in range(4):
                if i + j < self.bucket_count:
                    bucket_val = buckets[i + j]
                    if bucket_val <= q1:
                        bit_val = 0
                    elif bucket_val <= q2:
                        bit_val = 1
                    elif bucket_val <= q3:
                        bit_val = 2
                    else:
                        bit_val = 3
                    nibble |= (bit_val << (2 * j))
            hash_body.append(nibble)
        
        # Convert to hex string
        hex_string = ''.join(f'{b:02x}' for b in hash_body)
        
        # Add checksum and length
        checksum = sum(buckets) % 256
        length_byte = min(len(text) // 256, 255)
        
        return f"{checksum:02x}{length_byte:02x}{hex_string}"
    
    def _get_trigrams(self, text: str) -> List[str]:
        """Extract trigrams from text."""
        trigrams = []
        for i in range(len(text) - 2):
            trigrams.append(text[i:i + 3])
        return trigrams
    
    def _simhash(self, text: str, hash_bits: int = 64) -> int:
        """
        Compute SimHash for near-duplicate detection.
        
        Args:
            text: Input text
            hash_bits: Number of bits in hash
            
        Returns:
            SimHash value as integer
        """
        if hash_bits != 64:
            self.hash_bits = hash_bits
        
        # Extract features (words or shingles)
        features = self._extract_features(text)
        
        if not features:
            return 0
        
        # Initialize hash vector
        v = [0] * self.hash_bits
        
        for feature in features:
            # Get feature hash
            feature_hash = int(hashlib.md5(feature.encode()).hexdigest(), 16)
            
            # Update vector based on hash bits
            for i in range(self.hash_bits):
                bit = (feature_hash >> i) & 1
                if bit:
                    v[i] += 1
                else:
                    v[i] -= 1
        
        # Generate final hash
        simhash = 0
        for i in range(self.hash_bits):
            if v[i] > 0:
                simhash |= (1 << i)
        
        return simhash
    
    def _extract_features(self, text: str, k: int = 2) -> List[str]:
        """
        Extract features from text for hashing.
        
        Args:
            text: Input text
            k: Size of shingles
            
        Returns:
            List of features
        """
        # Use words as features for simplicity
        words = text.lower().split()
        
        if len(words) <= k:
            return words
        
        # Create k-shingles from words
        features = []
        for i in range(len(words) - k + 1):
            shingle = ' '.join(words[i:i + k])
            features.append(shingle)
        
        return features
    
    def _init_minhash_functions(self):
        """Initialize hash functions for MinHash."""
        self.hash_functions = []
        
        # Generate hash function parameters
        # Using universal hashing: h(x) = (a*x + b) mod p
        self.prime = 2**31 - 1  # Large prime
        
        for i in range(self.num_hashes):
            # Generate random parameters
            np.random.seed(i)
            a = np.random.randint(1, self.prime)
            b = np.random.randint(0, self.prime)
            self.hash_functions.append((a, b))
    
    def _generate_hash_functions(self, num_hashes: int) -> List[callable]:
        """
        Generate hash functions for MinHash.
        
        Args:
            num_hashes: Number of hash functions
            
        Returns:
            List of hash functions
        """
        functions = []
        
        for a, b in self.hash_functions[:num_hashes]:
            def hash_func(x, a=a, b=b):
                # Convert to integer if string
                if isinstance(x, str):
                    x = int(hashlib.md5(x.encode()).hexdigest()[:8], 16)
                return (a * x + b) % self.prime
            
            functions.append(hash_func)
        
        return functions
    
    def _minhash(self, text: str, num_hashes: Optional[int] = None) -> List[int]:
        """
        Compute MinHash signature.
        
        Args:
            text: Input text
            num_hashes: Number of hash functions to use
            
        Returns:
            MinHash signature
        """
        if num_hashes is None:
            num_hashes = self.num_hashes
        
        # Get shingles
        shingles = self._get_shingles(text, k=3)
        
        if not shingles:
            return [0] * num_hashes
        
        # Generate hash functions
        hash_funcs = self._generate_hash_functions(num_hashes)
        
        # Compute minimum hash for each function
        signature = []
        for hash_func in hash_funcs:
            min_hash = float('inf')
            for shingle in shingles:
                h = hash_func(shingle)
                min_hash = min(min_hash, h)
            signature.append(int(min_hash))
        
        return signature
    
    def _get_shingles(self, text: str, k: int = 3) -> Set[str]:
        """
        Get k-shingles from text.
        
        Args:
            text: Input text
            k: Shingle size
            
        Returns:
            Set of shingles
        """
        if len(text) < k:
            return {text}
        
        shingles = set()
        for i in range(len(text) - k + 1):
            shingles.add(text[i:i + k])
        
        return shingles
    
    def compare_hashes(self, hash1: Any, hash2: Any) -> float:
        """
        Compare two fuzzy hashes and return similarity score.
        
        Args:
            hash1: First hash
            hash2: Second hash
            
        Returns:
            Similarity score in [0, 1]
        """
        if hash1 is None or hash2 is None:
            return 0.0
        
        if self.hash_type == 'ssdeep':
            return self._ssdeep_compare(hash1, hash2)
        elif self.hash_type == 'tlsh':
            return self._tlsh_compare(hash1, hash2)
        elif self.hash_type == 'simhash':
            return self._hamming_similarity(hash1, hash2)
        elif self.hash_type == 'minhash':
            return self._jaccard_similarity(hash1, hash2)
        else:
            return 0.0
    
    def _ssdeep_compare(self, hash1: str, hash2: str) -> float:
        """
        Compare SSDeep hashes.
        
        Args:
            hash1: First SSDeep hash
            hash2: Second SSDeep hash
            
        Returns:
            Similarity score
        """
        # Parse hashes
        parts1 = hash1.split(':')
        parts2 = hash2.split(':')
        
        if len(parts1) < 2 or len(parts2) < 2:
            return 0.0
        
        block_size1 = int(parts1[0])
        block_size2 = int(parts2[0])
        
        # Block sizes should be similar
        if abs(block_size1 - block_size2) > max(block_size1, block_size2) // 2:
            return 0.0
        
        # Compare hash strings
        hash_str1 = parts1[1]
        hash_str2 = parts2[1]
        
        # Use Longest Common Substring
        return self._longest_common_substring_ratio(hash_str1, hash_str2)
    
    def _tlsh_compare(self, hash1: str, hash2: str) -> float:
        """
        Compare TLSH hashes.
        
        Args:
            hash1: First TLSH hash
            hash2: Second TLSH hash
            
        Returns:
            Similarity score
        """
        if len(hash1) != len(hash2):
            return 0.0
        
        # Compute Hamming distance
        distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        
        # Convert to similarity
        max_distance = len(hash1)
        similarity = 1.0 - (distance / max_distance)
        
        return similarity
    
    def _hamming_similarity(self, hash1: int, hash2: int) -> float:
        """
        Compute Hamming similarity between SimHash values.
        
        Args:
            hash1: First SimHash
            hash2: Second SimHash
            
        Returns:
            Similarity score
        """
        # XOR to find different bits
        xor = hash1 ^ hash2
        
        # Count different bits (Hamming distance)
        distance = bin(xor).count('1')
        
        # Convert to similarity
        similarity = 1.0 - (distance / self.hash_bits)
        
        return similarity
    
    def _jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """
        Estimate Jaccard similarity from MinHash signatures.
        
        Args:
            sig1: First MinHash signature
            sig2: Second MinHash signature
            
        Returns:
            Estimated Jaccard similarity
        """
        if len(sig1) != len(sig2):
            return 0.0
        
        if not sig1:
            return 1.0
        
        # Count matching hash values
        matches = sum(h1 == h2 for h1, h2 in zip(sig1, sig2))
        
        # Estimate Jaccard similarity
        similarity = matches / len(sig1)
        
        return similarity
    
    def _longest_common_substring_ratio(self, s1: str, s2: str) -> float:
        """
        Compute ratio based on longest common substring.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity ratio
        """
        if not s1 or not s2:
            return 0.0
        
        # Dynamic programming for LCS
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    max_length = max(max_length, dp[i][j])
        
        # Compute ratio
        avg_length = (len(s1) + len(s2)) / 2
        ratio = max_length / avg_length if avg_length > 0 else 0.0
        
        return min(ratio * 2, 1.0)  # Scale up and cap at 1.0


class LocalitySensitiveHash:
    """
    Locality-Sensitive Hashing for efficient similarity search.
    """
    
    def __init__(self, num_bands: int = 20, rows_per_band: int = 5):
        """
        Initialize LSH for approximate nearest neighbor search.
        
        Args:
            num_bands: Number of bands for hashing
            rows_per_band: Number of rows (hash values) per band
        """
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.num_hashes = num_bands * rows_per_band
        self.buckets = defaultdict(list)
        self.signatures = {}  # Store signatures for retrieval
    
    def index(self, doc_id: str, minhash_signature: List[int]) -> None:
        """
        Index document using LSH.
        
        Args:
            doc_id: Document identifier
            minhash_signature: MinHash signature of document
        """
        if len(minhash_signature) != self.num_hashes:
            raise ValueError(f"Signature must have {self.num_hashes} hash values")
        
        # Store signature
        self.signatures[doc_id] = minhash_signature
        
        # Hash into bands
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(minhash_signature[start:end])
            
            # Hash band to bucket
            bucket_id = hash(band)
            self.buckets[bucket_id].append(doc_id)
    
    def query(self, minhash_signature: List[int], 
             return_similarities: bool = False) -> Union[Set[str], List[Tuple[str, float]]]:
        """
        Find similar documents.
        
        Args:
            minhash_signature: MinHash signature to query
            return_similarities: If True, return (doc_id, similarity) pairs
            
        Returns:
            Set of candidate document IDs or list of (doc_id, similarity) pairs
        """
        if len(minhash_signature) != self.num_hashes:
            raise ValueError(f"Signature must have {self.num_hashes} hash values")
        
        candidates = set()
        
        # Check each band
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(minhash_signature[start:end])
            
            # Look up bucket
            bucket_id = hash(band)
            if bucket_id in self.buckets:
                candidates.update(self.buckets[bucket_id])
        
        if not return_similarities:
            return candidates
        
        # Compute actual similarities
        results = []
        for doc_id in candidates:
            stored_sig = self.signatures[doc_id]
            similarity = self._estimate_jaccard(minhash_signature, stored_sig)
            results.append((doc_id, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _estimate_jaccard(self, sig1: List[int], sig2: List[int]) -> float:
        """
        Estimate Jaccard similarity from signatures.
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            Estimated Jaccard similarity
        """
        if len(sig1) != len(sig2):
            return 0.0
        
        matches = sum(h1 == h2 for h1, h2 in zip(sig1, sig2))
        return matches / len(sig1)
    
    def remove(self, doc_id: str) -> bool:
        """
        Remove document from index.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if document was removed, False if not found
        """
        if doc_id not in self.signatures:
            return False
        
        signature = self.signatures[doc_id]
        
        # Remove from all buckets
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(signature[start:end])
            
            bucket_id = hash(band)
            if bucket_id in self.buckets and doc_id in self.buckets[bucket_id]:
                self.buckets[bucket_id].remove(doc_id)
                
                # Clean up empty buckets
                if not self.buckets[bucket_id]:
                    del self.buckets[bucket_id]
        
        # Remove signature
        del self.signatures[doc_id]
        
        return True
    
    def clear(self):
        """Clear all indexed documents."""
        self.buckets.clear()
        self.signatures.clear()
    
    def get_signature(self, doc_id: str) -> Optional[List[int]]:
        """
        Get stored signature for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            MinHash signature or None if not found
        """
        return self.signatures.get(doc_id)
    
    def size(self) -> int:
        """Get number of indexed documents."""
        return len(self.signatures)
    
    def get_collision_probability(self, similarity: float) -> float:
        """
        Calculate probability of collision for given similarity.
        
        Args:
            similarity: Jaccard similarity
            
        Returns:
            Probability of documents being candidates
        """
        # Probability = 1 - (1 - s^r)^b
        # where s = similarity, r = rows_per_band, b = num_bands
        prob_not_in_band = 1 - (similarity ** self.rows_per_band)
        prob_not_candidate = prob_not_in_band ** self.num_bands
        prob_candidate = 1 - prob_not_candidate
        
        return prob_candidate


# Utility functions
def compute_text_hashes(text: str, 
                       hash_types: List[str] = None) -> Dict[str, Any]:
    """
    Compute multiple hash types for a text.
    
    Args:
        text: Input text
        hash_types: List of hash types to compute (default: all)
        
    Returns:
        Dictionary of hash type to hash value
    """
    if hash_types is None:
        hash_types = ['simhash', 'minhash', 'ssdeep']
    
    results = {}
    
    for hash_type in hash_types:
        hasher = FuzzyHasher(hash_type=hash_type)
        hash_value = hasher.compute_hash(text)
        results[hash_type] = hash_value
    
    return results


def find_near_duplicates(texts: List[str],
                        threshold: float = 0.8,
                        hash_type: str = 'minhash',
                        use_lsh: bool = True) -> List[Tuple[int, int, float]]:
    """
    Find near-duplicate texts using fuzzy hashing.
    
    Args:
        texts: List of texts
        threshold: Similarity threshold
        hash_type: Type of hash to use
        use_lsh: Whether to use LSH for efficiency
        
    Returns:
        List of (index1, index2, similarity) tuples for near-duplicates
    """
    if not texts:
        return []
    
    hasher = FuzzyHasher(hash_type=hash_type)
    
    # Compute hashes
    hashes = []
    for text in texts:
        hash_value = hasher.compute_hash(text)
        hashes.append(hash_value)
    
    duplicates = []
    
    if use_lsh and hash_type == 'minhash':
        # Use LSH for efficient search
        # Match the number of hashes from MinHash (128 = 16 * 8)
        lsh = LocalitySensitiveHash(num_bands=16, rows_per_band=8)
        
        # Index all documents
        for i, signature in enumerate(hashes):
            lsh.index(str(i), signature)
        
        # Query each document
        for i, signature in enumerate(hashes):
            candidates = lsh.query(signature, return_similarities=True)
            
            for doc_id_str, similarity in candidates:
                j = int(doc_id_str)
                if j > i and similarity >= threshold:
                    duplicates.append((i, j, similarity))
    else:
        # Brute force comparison
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                similarity = hasher.compare_hashes(hashes[i], hashes[j])
                if similarity >= threshold:
                    duplicates.append((i, j, similarity))
    
    return duplicates


def cluster_by_similarity(texts: List[str],
                         threshold: float = 0.7,
                         hash_type: str = 'minhash') -> List[Set[int]]:
    """
    Cluster texts by similarity using fuzzy hashing.
    
    Args:
        texts: List of texts
        threshold: Similarity threshold for clustering
        hash_type: Type of hash to use
        
    Returns:
        List of clusters (each cluster is a set of text indices)
    """
    # Find near duplicates
    duplicates = find_near_duplicates(texts, threshold, hash_type)
    
    # Build clusters using union-find
    parent = list(range(len(texts)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union similar texts
    for i, j, _ in duplicates:
        union(i, j)
    
    # Group by cluster
    clusters = defaultdict(set)
    for i in range(len(texts)):
        root = find(i)
        clusters[root].add(i)
    
    return list(clusters.values())