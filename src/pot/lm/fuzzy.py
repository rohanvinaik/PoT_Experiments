"""
Fuzzy Matching Components for Language Model Verification
Implements various fuzzy string matching algorithms for robust text comparison
"""

import re
import unicodedata
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter
import numpy as np


class FuzzyMatcher:
    """
    Fuzzy string matcher for approximate text comparison.
    Implements multiple matching algorithms for different use cases.
    """
    
    def __init__(self, threshold: float = 0.8):
        """
        Initialize fuzzy matcher for approximate string matching.
        
        Args:
            threshold: Minimum similarity score for considering a match
        """
        self.threshold = threshold
        self._cache = {}  # Cache for computed similarities
    
    def fuzzy_match(self, 
                   text1: str, 
                   text2: str,
                   method: str = 'token_set_ratio') -> float:
        """
        Compute fuzzy match score between texts.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            method: Matching algorithm - 'ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio'
            
        Returns:
            Similarity score in range [0, 1]
        """
        # Check cache
        cache_key = (text1, text2, method)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Preprocess texts
        text1 = self._preprocess(text1)
        text2 = self._preprocess(text2)
        
        # Compute similarity based on method
        if method == 'ratio':
            score = self._simple_ratio(text1, text2)
        elif method == 'partial_ratio':
            score = self._partial_ratio(text1, text2)
        elif method == 'token_sort_ratio':
            score = self._token_sort_ratio(text1, text2)
        elif method == 'token_set_ratio':
            score = self._token_set_ratio(text1, text2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Cache result
        self._cache[cache_key] = score
        return score
    
    def _preprocess(self, text: str) -> str:
        """
        Preprocess text for fuzzy matching.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Remove non-alphanumeric characters except spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _simple_ratio(self, s1: str, s2: str) -> float:
        """
        Basic Levenshtein ratio.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity ratio
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        distance = self._levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1 - (distance / max_len) if max_len > 0 else 1.0
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Compute Levenshtein distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        # Use dynamic programming
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                
                current_row.append(min(insertions, deletions, substitutions))
            
            previous_row = current_row
        
        return previous_row[-1]
    
    def _partial_ratio(self, s1: str, s2: str) -> float:
        """
        Find best partial match ratio.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Best partial match ratio
        """
        if not s1 or not s2:
            return 0.0
        
        # Ensure s1 is the shorter string
        shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
        
        if len(shorter) == 0:
            return 1.0 if len(longer) == 0 else 0.0
        
        best_ratio = 0
        
        # Slide shorter string across longer string
        for i in range(len(longer) - len(shorter) + 1):
            substring = longer[i:i + len(shorter)]
            ratio = self._simple_ratio(shorter, substring)
            best_ratio = max(best_ratio, ratio)
            
            # Early termination if perfect match found
            if best_ratio >= 0.99:
                break
        
        return best_ratio
    
    def _token_sort_ratio(self, s1: str, s2: str) -> float:
        """
        Sort tokens before comparison.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Token sort ratio
        """
        # Tokenize and sort
        tokens1 = sorted(s1.split())
        tokens2 = sorted(s2.split())
        
        # Rejoin and compare
        sorted1 = ' '.join(tokens1)
        sorted2 = ' '.join(tokens2)
        
        return self._simple_ratio(sorted1, sorted2)
    
    def _token_set_ratio(self, s1: str, s2: str) -> float:
        """
        Token set ratio - handles duplicates and order.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Token set ratio
        """
        # Get token sets
        tokens1 = set(s1.split())
        tokens2 = set(s2.split())
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        # Compute intersection and differences
        intersection = tokens1 & tokens2
        diff1 = tokens1 - tokens2
        diff2 = tokens2 - tokens1
        
        # Sort tokens for consistent comparison
        sorted_intersection = ' '.join(sorted(intersection))
        sorted_diff1 = ' '.join(sorted(diff1))
        sorted_diff2 = ' '.join(sorted(diff2))
        
        # Combine for comparison
        combined1 = sorted_intersection
        combined2 = sorted_intersection
        
        if sorted_diff1:
            combined1 = f"{combined1} {sorted_diff1}".strip()
        if sorted_diff2:
            combined2 = f"{combined2} {sorted_diff2}".strip()
        
        # Compare different combinations
        ratios = [
            self._simple_ratio(sorted_intersection, combined1),
            self._simple_ratio(sorted_intersection, combined2),
            self._simple_ratio(combined1, combined2)
        ]
        
        return max(ratios)
    
    def find_best_match(self, 
                       query: str, 
                       candidates: List[str],
                       n_best: int = 1,
                       method: str = 'token_set_ratio') -> List[Tuple[str, float]]:
        """
        Find best matching candidates.
        
        Args:
            query: Query string to match
            candidates: List of candidate strings
            n_best: Number of best matches to return
            method: Matching method to use
            
        Returns:
            List of (candidate, score) tuples sorted by score
        """
        if not candidates:
            return []
        
        scores = []
        for candidate in candidates:
            score = self.fuzzy_match(query, candidate, method=method)
            scores.append((candidate, score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n_best matches that meet threshold
        result = []
        for candidate, score in scores[:n_best]:
            if score >= self.threshold:
                result.append((candidate, score))
        
        return result
    
    def group_similar(self, 
                     texts: List[str],
                     method: str = 'token_set_ratio') -> List[List[str]]:
        """
        Group similar texts together.
        
        Args:
            texts: List of texts to group
            method: Matching method to use
            
        Returns:
            List of groups (each group is a list of similar texts)
        """
        if not texts:
            return []
        
        # Build similarity matrix
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    score = self.fuzzy_match(texts[i], texts[j], method=method)
                    similarity_matrix[i][j] = score
                    similarity_matrix[j][i] = score
        
        # Group using simple clustering
        groups = []
        used = set()
        
        for i in range(n):
            if i in used:
                continue
            
            # Start new group
            group = [texts[i]]
            used.add(i)
            
            # Add similar texts to group
            for j in range(i + 1, n):
                if j not in used and similarity_matrix[i][j] >= self.threshold:
                    group.append(texts[j])
                    used.add(j)
            
            groups.append(group)
        
        return groups


class NGramMatcher:
    """
    N-gram based fuzzy matcher for more granular text comparison.
    """
    
    def __init__(self, n: int = 3):
        """
        Initialize n-gram matcher.
        
        Args:
            n: Size of n-grams
        """
        self.n = n
    
    def get_ngrams(self, text: str, n: Optional[int] = None) -> List[str]:
        """
        Extract n-grams from text.
        
        Args:
            text: Input text
            n: N-gram size (uses self.n if not specified)
            
        Returns:
            List of n-grams
        """
        if n is None:
            n = self.n
        
        if len(text) < n:
            return [text]
        
        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i + n])
        
        return ngrams
    
    def ngram_similarity(self, text1: str, text2: str) -> float:
        """
        Compute n-gram based similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score
        """
        if not text1 or not text2:
            return 0.0 if (text1 or text2) else 1.0
        
        # Get n-grams
        ngrams1 = set(self.get_ngrams(text1))
        ngrams2 = set(self.get_ngrams(text2))
        
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    def weighted_ngram_similarity(self, text1: str, text2: str,
                                 weights: Optional[Dict[int, float]] = None) -> float:
        """
        Compute weighted n-gram similarity using multiple n-gram sizes.
        
        Args:
            text1: First text
            text2: Second text
            weights: Weights for different n-gram sizes (default: {2: 0.3, 3: 0.5, 4: 0.2})
            
        Returns:
            Weighted similarity score
        """
        if weights is None:
            weights = {2: 0.3, 3: 0.5, 4: 0.2}
        
        total_score = 0.0
        total_weight = 0.0
        
        for n, weight in weights.items():
            # Get n-grams for this size
            ngrams1 = set(self.get_ngrams(text1, n))
            ngrams2 = set(self.get_ngrams(text2, n))
            
            # Skip if either text is too short
            if not ngrams1 or not ngrams2:
                continue
            
            # Compute similarity for this n-gram size
            intersection = len(ngrams1 & ngrams2)
            union = len(ngrams1 | ngrams2)
            similarity = intersection / union if union > 0 else 0.0
            
            total_score += similarity * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


class PhoneticMatcher:
    """
    Phonetic matching for handling similar-sounding text.
    """
    
    def __init__(self):
        """Initialize phonetic matcher."""
        self._init_soundex_map()
        self._init_metaphone_rules()
    
    def _init_soundex_map(self):
        """Initialize Soundex character mapping."""
        self.soundex_map = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2',
            'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }
    
    def _init_metaphone_rules(self):
        """Initialize basic Metaphone rules."""
        # Simplified rules for demonstration
        self.metaphone_rules = {
            'PH': 'F',
            'GH': '',
            'WR': 'R',
            'KN': 'N',
            'PS': 'S',
            'WH': 'W',
            'CK': 'K',
            'SCH': 'SK',
            'CH': 'X',
            'SH': 'X',
            'TH': '0',
            'TCH': 'CH'
        }
    
    def soundex(self, word: str, length: int = 4) -> str:
        """
        Compute Soundex code for a word.
        
        Args:
            word: Input word
            length: Length of Soundex code
            
        Returns:
            Soundex code
        """
        if not word:
            return ''
        
        word = word.upper()
        
        # Keep first letter
        soundex = word[0]
        
        # Map remaining letters
        for char in word[1:]:
            if char in self.soundex_map:
                code = self.soundex_map[char]
                # Avoid consecutive duplicates
                if not soundex or soundex[-1] != code:
                    soundex += code
        
        # Remove vowels (except first letter)
        soundex = soundex[0] + ''.join(c for c in soundex[1:] if c in '123456')
        
        # Pad with zeros or truncate
        if len(soundex) < length:
            soundex += '0' * (length - len(soundex))
        else:
            soundex = soundex[:length]
        
        return soundex
    
    def metaphone(self, word: str, max_length: int = 10) -> str:
        """
        Compute basic Metaphone encoding.
        
        Args:
            word: Input word
            max_length: Maximum length of encoding
            
        Returns:
            Metaphone encoding
        """
        if not word:
            return ''
        
        word = word.upper()
        result = []
        i = 0
        
        while i < len(word) and len(result) < max_length:
            # Check for multi-character rules
            matched = False
            for pattern, replacement in self.metaphone_rules.items():
                if word[i:i+len(pattern)] == pattern:
                    if replacement:
                        result.append(replacement)
                    i += len(pattern)
                    matched = True
                    break
            
            if not matched:
                # Single character processing
                char = word[i]
                if char in 'AEIOU':
                    # Keep vowel only if it's the first character
                    if i == 0:
                        result.append(char)
                elif char.isalpha():
                    result.append(char)
                i += 1
        
        return ''.join(result[:max_length])
    
    def phonetic_similarity(self, word1: str, word2: str,
                          method: str = 'soundex') -> float:
        """
        Compute phonetic similarity between words.
        
        Args:
            word1: First word
            word2: Second word
            method: 'soundex' or 'metaphone'
            
        Returns:
            Similarity score
        """
        if method == 'soundex':
            code1 = self.soundex(word1)
            code2 = self.soundex(word2)
        elif method == 'metaphone':
            code1 = self.metaphone(word1)
            code2 = self.metaphone(word2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if not code1 or not code2:
            return 0.0
        
        # Compute similarity of phonetic codes
        if code1 == code2:
            return 1.0
        
        # Use Levenshtein ratio for partial similarity
        matcher = FuzzyMatcher()
        return matcher._simple_ratio(code1, code2)


class SemanticMatcher:
    """
    Semantic similarity matching using word embeddings.
    """
    
    def __init__(self, embedding_dim: int = 100):
        """
        Initialize semantic matcher.
        
        Args:
            embedding_dim: Dimension of word embeddings
        """
        self.embedding_dim = embedding_dim
        self.word_vectors = {}  # Cache for word vectors
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """
        Get word embedding vector.
        
        Args:
            word: Input word
            
        Returns:
            Word embedding vector
        """
        if word in self.word_vectors:
            return self.word_vectors[word]
        
        # Simulate word embedding (in practice, use pre-trained embeddings)
        # Use hash of word as seed for reproducibility
        np.random.seed(hash(word) % (2**32))
        vector = np.random.randn(self.embedding_dim)
        
        # Normalize
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        
        self.word_vectors[word] = vector
        return vector
    
    def text_to_vector(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector.
        
        Args:
            text: Input text
            
        Returns:
            Text embedding (average of word embeddings)
        """
        words = text.lower().split()
        if not words:
            return np.zeros(self.embedding_dim)
        
        vectors = []
        for word in words:
            vectors.append(self.get_word_vector(word))
        
        # Average word vectors
        return np.mean(vectors, axis=0)
    
    def semantic_similarity(self, text1: str, text2: str,
                          metric: str = 'cosine') -> float:
        """
        Compute semantic similarity between texts.
        
        Args:
            text1: First text
            text2: Second text
            metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            Similarity score
        """
        vec1 = self.text_to_vector(text1)
        vec2 = self.text_to_vector(text2)
        
        if metric == 'cosine':
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            # Convert to [0, 1] range
            return (similarity + 1) / 2
        
        elif metric == 'euclidean':
            # Euclidean distance converted to similarity
            distance = np.linalg.norm(vec1 - vec2)
            # Convert to similarity (assuming max distance is 2 for normalized vectors)
            return max(0, 1 - distance / 2)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")


# Utility functions
def batch_fuzzy_match(queries: List[str], 
                     candidates: List[str],
                     matcher: Optional[FuzzyMatcher] = None,
                     method: str = 'token_set_ratio',
                     threshold: float = 0.8) -> Dict[str, List[Tuple[str, float]]]:
    """
    Perform batch fuzzy matching.
    
    Args:
        queries: List of query strings
        candidates: List of candidate strings
        matcher: FuzzyMatcher instance (creates new if None)
        method: Matching method
        threshold: Minimum similarity threshold
        
    Returns:
        Dictionary mapping each query to its matches
    """
    if matcher is None:
        matcher = FuzzyMatcher(threshold=threshold)
    
    results = {}
    for query in queries:
        matches = matcher.find_best_match(query, candidates, n_best=5, method=method)
        results[query] = matches
    
    return results


def deduplicate_fuzzy(texts: List[str],
                     threshold: float = 0.9,
                     method: str = 'token_set_ratio') -> List[str]:
    """
    Remove fuzzy duplicates from a list of texts.
    
    Args:
        texts: List of texts
        threshold: Similarity threshold for considering duplicates
        method: Matching method
        
    Returns:
        List of unique texts
    """
    if not texts:
        return []
    
    matcher = FuzzyMatcher(threshold=threshold)
    unique = []
    
    for text in texts:
        is_duplicate = False
        
        for unique_text in unique:
            similarity = matcher.fuzzy_match(text, unique_text, method=method)
            if similarity >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(text)
    
    return unique