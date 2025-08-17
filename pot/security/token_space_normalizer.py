"""
Token Space Normalizer for Proof-of-Training
Handles normalization of token sequences across different tokenizers and models
"""

import re
import unicodedata
from typing import Any, List, Tuple, Dict, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum


class TokenizerType(Enum):
    """Supported tokenizer types"""
    BERT = "bert"
    GPT2 = "gpt2"
    T5 = "t5"
    ROBERTA = "roberta"
    CUSTOM = "custom"


@dataclass
class AlignmentResult:
    """Result of sequence alignment"""
    alignment: List[Tuple[int, int]]  # List of aligned token pairs (idx1, idx2)
    score: float  # Alignment quality score
    method: str  # Alignment method used
    metadata: Dict[str, Any]  # Additional metadata


class StochasticDecodingController:
    """
    Controls stochastic decoding for language models to ensure reproducibility
    and enable fuzzy matching in token space.
    """
    
    def __init__(self, 
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 seed: Optional[int] = None):
        """
        Initialize stochastic decoding controller.
        
        Args:
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            seed: Random seed for reproducibility
        """
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature if self.temperature > 0 else logits
    
    def top_k_filtering(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        if self.top_k > 0:
            values, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
            min_value = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_value, 
                                torch.full_like(logits, float('-inf')), 
                                logits)
        return logits
    
    def top_p_filtering(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample from logits with configured strategy.
        
        Args:
            logits: Model output logits
            
        Returns:
            Sampled token indices
        """
        # Apply temperature
        logits = self.apply_temperature(logits)
        
        # Apply filtering
        logits = self.top_k_filtering(logits)
        logits = self.top_p_filtering(logits)
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    def greedy_decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy decoding (argmax)."""
        return torch.argmax(logits, dim=-1)
    
    def get_config(self) -> Dict[str, Any]:
        """Get controller configuration."""
        return {
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'seed': self.seed
        }


class TokenSpaceNormalizer:
    """
    Normalizes token sequences for robust comparison across different tokenizers.
    Supports multiple normalization modes for different use cases.
    """
    
    def __init__(self, tokenizer: Any, mode: str = 'canonical'):
        """
        Initialize normalizer with tokenizer and mode.
        
        Args:
            tokenizer: Tokenizer instance (HuggingFace or similar)
            mode: Normalization mode - 'canonical', 'string', 'byte', 'semantic'
        """
        self.tokenizer = tokenizer
        self.mode = mode
        
        # Get vocabulary
        if hasattr(tokenizer, 'get_vocab'):
            self.vocab = tokenizer.get_vocab()
        elif hasattr(tokenizer, 'vocab'):
            self.vocab = tokenizer.vocab
        else:
            self.vocab = {}
        
        # Create inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Cache for normalized tokens
        self._normalization_cache = {}
        
        # Special token handling
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize special token handling"""
        self.special_tokens = set()
        
        if hasattr(self.tokenizer, 'special_tokens_map'):
            special_tokens_map = self.tokenizer.special_tokens_map
            # Handle both dict and Mock objects
            if hasattr(special_tokens_map, 'items'):
                for token_type, token in special_tokens_map.items():
                    if isinstance(token, str):
                        token_id = self.vocab.get(token)
                        if token_id is not None:
                            self.special_tokens.add(token_id)
        
        # Common special tokens
        for token in ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '<s>', '</s>', '<pad>', '<unk>']:
            token_id = self.vocab.get(token)
            if token_id is not None:
                self.special_tokens.add(token_id)
    
    def normalize(self, tokens: List[int]) -> Union[List[int], str, bytes, torch.Tensor]:
        """
        Normalize tokens based on configured mode.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Normalized representation based on mode
        """
        if self.mode == 'canonical':
            return self.normalize_canonical(tokens)
        elif self.mode == 'string':
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            return self.normalize_string(text)
        elif self.mode == 'byte':
            return self.normalize_byte(tokens)
        elif self.mode == 'semantic':
            return self.normalize_semantic(tokens)
        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")
    
    def normalize_canonical(self, tokens: List[int]) -> List[int]:
        """
        Canonical normalization: map tokens to canonical forms.
        
        Handles:
        - Case variations (Hello, hello, HELLO → canonical)
        - Unicode normalization (é, è → normalized form)
        - Whitespace standardization
        - Punctuation normalization
        
        Args:
            tokens: List of token IDs
            
        Returns:
            List of canonically normalized token IDs
        """
        normalized = []
        
        for token_id in tokens:
            # Check cache
            if token_id in self._normalization_cache:
                normalized.extend(self._normalization_cache[token_id])
                continue
            
            # Skip special tokens
            if token_id in self.special_tokens:
                normalized.append(token_id)
                self._normalization_cache[token_id] = [token_id]
                continue
            
            # Get token string
            token_str = self.inverse_vocab.get(token_id, '')
            if not token_str:
                normalized.append(token_id)
                continue
            
            # Apply normalization pipeline
            normalized_str = token_str
            
            # 1. Unicode normalization (NFKC - compatibility decomposition)
            normalized_str = unicodedata.normalize('NFKC', normalized_str)
            
            # 2. Case normalization for content tokens
            if self._is_content_token(normalized_str):
                # Preserve first character case for sentence boundaries
                if len(normalized_str) > 0 and not self._is_sentence_start(tokens, normalized):
                    normalized_str = normalized_str.lower()
            
            # 3. Whitespace normalization
            normalized_str = self._normalize_whitespace(normalized_str)
            
            # 4. Punctuation normalization
            normalized_str = self._normalize_punctuation(normalized_str)
            
            # Re-tokenize normalized string
            if normalized_str:
                new_tokens = self.tokenizer.encode(normalized_str, add_special_tokens=False)
                normalized.extend(new_tokens)
                self._normalization_cache[token_id] = new_tokens
            
        return normalized
    
    def normalize_string(self, text: str) -> str:
        """
        String-level normalization before tokenization.
        
        Args:
            text: Input text string
            
        Returns:
            Normalized text string
        """
        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        
        # Normalize quotes and apostrophes
        text = re.sub(r'[''`´]', "'", text)
        text = re.sub(r'[""„"«»]', '"', text)
        
        # Normalize dashes
        text = re.sub(r'[–—]', '-', text)
        
        # Normalize ellipsis
        text = re.sub(r'\.{3,}', '...', text)
        
        # Normalize whitespace (including non-breaking spaces)
        text = re.sub(r'[\xa0\t]+', ' ', text)
        text = ' '.join(text.split())
        
        # Remove control characters except newlines
        text = ''.join(ch for ch in text if ch == '\n' or not unicodedata.category(ch).startswith('C'))
        
        return text
    
    def normalize_byte(self, tokens: List[int]) -> bytes:
        """
        Byte-level normalization for exact comparison.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            UTF-8 byte representation
        """
        # Decode tokens to text
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        
        # Apply string normalization first
        text = self.normalize_string(text)
        
        # Convert to bytes with error handling
        return text.encode('utf-8', errors='ignore')
    
    def normalize_semantic(self, tokens: List[int]) -> torch.Tensor:
        """
        Semantic normalization using token embeddings.
        Falls back to one-hot encoding if no embedding model available.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Tensor representation of tokens
        """
        # For now, return one-hot encoding as fallback
        # In practice, this would use an embedding model
        vocab_size = len(self.vocab) if self.vocab else max(tokens) + 1
        
        embeddings = []
        for token_id in tokens:
            one_hot = torch.zeros(vocab_size)
            if token_id < vocab_size:
                one_hot[token_id] = 1.0
            embeddings.append(one_hot)
        
        return torch.stack(embeddings) if embeddings else torch.empty(0, vocab_size)
    
    def _is_content_token(self, token_str: str) -> bool:
        """Check if token is a content word (not punctuation or special)"""
        if not token_str:
            return False
        
        # Check if token is primarily alphanumeric
        alpha_count = sum(c.isalnum() for c in token_str)
        return alpha_count > len(token_str) / 2
    
    def _is_sentence_start(self, tokens: List[int], normalized: List[int]) -> bool:
        """Check if current position is likely a sentence start"""
        if not normalized:
            return True
        
        # Check if previous token was sentence-ending punctuation
        if normalized:
            prev_token_str = self.inverse_vocab.get(normalized[-1], '')
            if prev_token_str in {'.', '!', '?', ':', ';'}:
                return True
        
        return False
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace variations"""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Normalize leading/trailing whitespace in tokens
        if text.startswith(' ') and len(text.strip()) > 0:
            text = ' ' + text.strip()
        elif text.endswith(' ') and len(text.strip()) > 0:
            text = text.strip() + ' '
        
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation variations"""
        # Normalize similar punctuation marks
        replacements = {
            '…': '...',
            '‚': ',',
            '‛': "'",
            '‟': '"',
            '′': "'",
            '″': '"',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def compute_distance(self, tokens1: List[int], tokens2: List[int],
                        method: str = 'jaccard') -> float:
        """
        Compute distance between two token sequences.
        
        Args:
            tokens1: First token sequence
            tokens2: Second token sequence
            method: Distance metric ('jaccard', 'levenshtein', 'cosine')
            
        Returns:
            Distance score (0 = identical, 1 = completely different)
        """
        # Normalize both sequences
        norm1 = self.normalize(tokens1)
        norm2 = self.normalize(tokens2)
        
        if method == 'jaccard':
            set1 = set(norm1) if isinstance(norm1, list) else {norm1}
            set2 = set(norm2) if isinstance(norm2, list) else {norm2}
            
            if not set1 and not set2:
                return 0.0
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            return 1.0 - (intersection / union if union > 0 else 0.0)
        
        elif method == 'levenshtein':
            # Simple Levenshtein distance
            if isinstance(norm1, list) and isinstance(norm2, list):
                return self._levenshtein_distance(norm1, norm2) / max(len(norm1), len(norm2), 1)
            else:
                return 1.0
        
        elif method == 'cosine':
            # Cosine distance for semantic normalization
            if isinstance(norm1, torch.Tensor) and isinstance(norm2, torch.Tensor):
                cos_sim = F.cosine_similarity(norm1.mean(0).unsqueeze(0), 
                                             norm2.mean(0).unsqueeze(0))
                return 1.0 - min(1.0, max(0.0, cos_sim.item()))
            else:
                return 1.0
        
        else:
            raise ValueError(f"Unknown distance method: {method}")
    
    def _levenshtein_distance(self, seq1: List[int], seq2: List[int]) -> int:
        """Compute Levenshtein distance between sequences"""
        n, m = len(seq1), len(seq2)
        
        if n == 0:
            return m
        if m == 0:
            return n
        
        # Create distance matrix
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        # Initialize first row and column
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        
        # Fill matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if seq1[i-1] == seq2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
        
        return dp[n][m]


class TokenAligner:
    """
    Aligns token sequences for detailed comparison.
    Supports multiple alignment algorithms.
    """
    
    def __init__(self, normalizer: Optional[TokenSpaceNormalizer] = None):
        """
        Initialize aligner with optional normalizer.
        
        Args:
            normalizer: TokenSpaceNormalizer instance for pre-processing
        """
        self.normalizer = normalizer
    
    def align_sequences(self, 
                       seq1: List[int], 
                       seq2: List[int],
                       method: str = 'dynamic_programming') -> AlignmentResult:
        """
        Align two token sequences for comparison.
        
        Args:
            seq1: First token sequence
            seq2: Second token sequence
            method: Alignment method ('dynamic_programming', 'needleman_wunsch', 'semantic')
            
        Returns:
            AlignmentResult with alignment pairs and score
        """
        # Optionally normalize sequences first
        if self.normalizer:
            seq1 = self.normalizer.normalize(seq1)
            seq2 = self.normalizer.normalize(seq2)
            
            # Convert back to list if needed
            if not isinstance(seq1, list):
                seq1 = [seq1]
            if not isinstance(seq2, list):
                seq2 = [seq2]
        
        # Apply alignment method
        if method == 'dynamic_programming':
            alignment = self._dp_alignment(seq1, seq2)
        elif method == 'needleman_wunsch':
            alignment = self._needleman_wunsch(seq1, seq2)
        elif method == 'semantic':
            alignment = self._semantic_alignment(seq1, seq2)
        else:
            raise ValueError(f"Unknown alignment method: {method}")
        
        # Compute alignment score
        score = self.compute_alignment_score(alignment, seq1, seq2)
        
        # Gather metadata
        metadata = {
            'seq1_length': len(seq1),
            'seq2_length': len(seq2),
            'num_aligned': len(alignment),
            'num_gaps': max(len(seq1), len(seq2)) - len(alignment)
        }
        
        return AlignmentResult(
            alignment=alignment,
            score=score,
            method=method,
            metadata=metadata
        )
    
    def _dp_alignment(self, seq1: List[int], seq2: List[int]) -> List[Tuple[int, int]]:
        """
        Dynamic programming sequence alignment (Longest Common Subsequence).
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            List of aligned token pairs (idx1, idx2)
        """
        n, m = len(seq1), len(seq2)
        
        # DP table for LCS
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # Backtrack to find alignment
        alignment = []
        i, j = n, m
        
        while i > 0 and j > 0:
            if seq1[i-1] == seq2[j-1]:
                alignment.append((i-1, j-1))
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1
        
        return list(reversed(alignment))
    
    def _needleman_wunsch(self, seq1: List[int], seq2: List[int],
                         match_score: float = 1.0,
                         mismatch_score: float = -1.0,
                         gap_score: float = -1.0) -> List[Tuple[int, int]]:
        """
        Needleman-Wunsch global alignment algorithm.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            match_score: Score for matching tokens
            mismatch_score: Score for mismatched tokens
            gap_score: Score for gaps
            
        Returns:
            List of aligned token pairs (idx1, idx2)
        """
        n, m = len(seq1), len(seq2)
        
        # Initialize scoring matrix
        score_matrix = [[0.0] * (m + 1) for _ in range(n + 1)]
        
        # Initialize first row and column with gap penalties
        for i in range(1, n + 1):
            score_matrix[i][0] = i * gap_score
        for j in range(1, m + 1):
            score_matrix[0][j] = j * gap_score
        
        # Fill scoring matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Match/mismatch score
                if seq1[i-1] == seq2[j-1]:
                    diag_score = score_matrix[i-1][j-1] + match_score
                else:
                    diag_score = score_matrix[i-1][j-1] + mismatch_score
                
                # Gap scores
                up_score = score_matrix[i-1][j] + gap_score
                left_score = score_matrix[i][j-1] + gap_score
                
                # Take maximum
                score_matrix[i][j] = max(diag_score, up_score, left_score)
        
        # Backtrack to find alignment
        alignment = []
        i, j = n, m
        
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                current_score = score_matrix[i][j]
                diag_score = score_matrix[i-1][j-1]
                
                # Check if we came from diagonal (match/mismatch)
                expected_score = diag_score + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score)
                
                if abs(current_score - expected_score) < 1e-6:
                    alignment.append((i-1, j-1))
                    i -= 1
                    j -= 1
                elif i > 0 and abs(current_score - (score_matrix[i-1][j] + gap_score)) < 1e-6:
                    i -= 1
                else:
                    j -= 1
            elif i > 0:
                i -= 1
            else:
                j -= 1
        
        return list(reversed(alignment))
    
    def _semantic_alignment(self, seq1: List[int], seq2: List[int],
                           threshold: float = 0.7) -> List[Tuple[int, int]]:
        """
        Semantic alignment based on token similarity.
        Requires embeddings or falls back to exact matching.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            threshold: Similarity threshold for alignment
            
        Returns:
            List of aligned token pairs (idx1, idx2)
        """
        # For now, fall back to exact matching
        # In practice, this would use semantic similarity
        alignment = []
        
        # Create similarity matrix
        for i, token1 in enumerate(seq1):
            for j, token2 in enumerate(seq2):
                if token1 == token2:  # Exact match for now
                    # Check if this position is not already aligned
                    if not any(a[0] == i or a[1] == j for a in alignment):
                        alignment.append((i, j))
                        break
        
        # Sort by first index
        alignment.sort(key=lambda x: x[0])
        
        return alignment
    
    def compute_alignment_score(self, 
                               alignment: List[Tuple[int, int]],
                               seq1: List[int],
                               seq2: List[int]) -> float:
        """
        Compute quality score for alignment.
        
        Args:
            alignment: List of aligned pairs
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Alignment score (0.0 = no alignment, 1.0 = perfect alignment)
        """
        # Total possible matches
        total = max(len(seq1), len(seq2))
        
        if total == 0:
            return 1.0  # Both sequences empty, perfect alignment
        
        if not alignment:
            return 0.0
        
        # Number of matches
        matches = len(alignment)
        
        # Basic score
        basic_score = matches / total
        
        # Penalize for gaps in alignment
        if matches > 1:
            # Check for consecutive alignments
            consecutive = 0
            for i in range(1, len(alignment)):
                if (alignment[i][0] == alignment[i-1][0] + 1 and 
                    alignment[i][1] == alignment[i-1][1] + 1):
                    consecutive += 1
            
            # Bonus for consecutive alignments
            consecutivity_bonus = consecutive / (matches - 1) * 0.2
            basic_score = min(1.0, basic_score + consecutivity_bonus)
        
        return basic_score
    
    def visualize_alignment(self, 
                           alignment: List[Tuple[int, int]],
                           seq1: List[int],
                           seq2: List[int],
                           tokenizer: Optional[Any] = None) -> str:
        """
        Create a visual representation of the alignment.
        
        Args:
            alignment: List of aligned pairs
            seq1: First sequence
            seq2: Second sequence
            tokenizer: Optional tokenizer for decoding tokens
            
        Returns:
            String visualization of alignment
        """
        lines = []
        
        # Convert tokens to strings if tokenizer provided
        if tokenizer:
            str1 = [tokenizer.decode([t]) if hasattr(tokenizer, 'decode') else str(t) for t in seq1]
            str2 = [tokenizer.decode([t]) if hasattr(tokenizer, 'decode') else str(t) for t in seq2]
        else:
            str1 = [str(t) for t in seq1]
            str2 = [str(t) for t in seq2]
        
        # Create alignment map
        align_map = {i: j for i, j in alignment}
        
        # Build visualization
        top_line = []
        middle_line = []
        bottom_line = []
        
        i, j = 0, 0
        
        while i < len(seq1) or j < len(seq2):
            if i in align_map and align_map[i] == j:
                # Aligned tokens
                token1 = str1[i] if i < len(str1) else ''
                token2 = str2[j] if j < len(str2) else ''
                max_len = max(len(token1), len(token2))
                
                top_line.append(token1.ljust(max_len))
                middle_line.append('|'.ljust(max_len))
                bottom_line.append(token2.ljust(max_len))
                
                i += 1
                j += 1
            elif i < len(seq1) and (i not in align_map or align_map[i] > j):
                # Gap in seq2
                token1 = str1[i]
                
                top_line.append(token1)
                middle_line.append(' ' * len(token1))
                bottom_line.append('-' * len(token1))
                
                i += 1
            else:
                # Gap in seq1
                token2 = str2[j] if j < len(str2) else ''
                
                top_line.append('-' * len(token2))
                middle_line.append(' ' * len(token2))
                bottom_line.append(token2)
                
                j += 1
        
        # Join with spaces
        lines.append(' '.join(top_line))
        lines.append(' '.join(middle_line))
        lines.append(' '.join(bottom_line))
        
        return '\n'.join(lines)


class SemanticNormalizer:
    """
    Semantic normalization using embeddings and clustering.
    Maps tokens to semantic space for robust comparison.
    """
    
    def __init__(self, embedding_model: Optional[Any] = None,
                 num_clusters: int = 1000,
                 embedding_dim: int = 768):
        """
        Initialize semantic normalizer.
        
        Args:
            embedding_model: Model for generating token embeddings
            num_clusters: Number of semantic clusters for quantization
            embedding_dim: Dimension of embeddings
        """
        self.embedding_model = embedding_model
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        
        # Initialize cluster centers (would be learned in practice)
        self.cluster_centers = None
        self._init_clusters()
    
    def _init_clusters(self):
        """Initialize cluster centers for vector quantization"""
        # Random initialization (would be learned from data in practice)
        self.cluster_centers = torch.randn(self.num_clusters, self.embedding_dim)
        self.cluster_centers = F.normalize(self.cluster_centers, p=2, dim=1)
    
    def normalize_semantic(self, tokens: List[int]) -> torch.Tensor:
        """
        Map tokens to semantic embedding space.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Normalized semantic representation
        """
        if self.embedding_model:
            # Get embeddings from model
            embeddings = self.embedding_model.get_token_embeddings(tokens)
        else:
            # Fallback: random embeddings based on token ID
            embeddings = []
            for token_id in tokens:
                # Use token ID as seed for reproducible random embedding
                torch.manual_seed(token_id)
                emb = torch.randn(self.embedding_dim)
                embeddings.append(emb)
            
            embeddings = torch.stack(embeddings) if embeddings else torch.empty(0, self.embedding_dim)
        
        # Apply semantic quantization
        normalized = self._semantic_quantization(embeddings)
        
        return normalized
    
    def _semantic_quantization(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantize embeddings to semantic clusters.
        
        Args:
            embeddings: Token embeddings [seq_len, embed_dim]
            
        Returns:
            Quantized embeddings
        """
        if embeddings.shape[0] == 0:
            return embeddings
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Find nearest cluster for each embedding
        quantized = []
        
        for emb in embeddings:
            # Compute distances to all clusters
            distances = torch.cdist(emb.unsqueeze(0), self.cluster_centers)
            
            # Find nearest cluster
            nearest_idx = distances.argmin()
            
            # Use cluster center as quantized embedding
            quantized.append(self.cluster_centers[nearest_idx])
        
        return torch.stack(quantized) if quantized else embeddings
    
    def compute_semantic_similarity(self, 
                                   tokens1: List[int],
                                   tokens2: List[int],
                                   method: str = 'cosine') -> float:
        """
        Compute semantic similarity between token sequences.
        
        Args:
            tokens1: First token sequence
            tokens2: Second token sequence
            method: Similarity method ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            Similarity score (0.0 = different, 1.0 = identical)
        """
        # Get semantic representations
        emb1 = self.normalize_semantic(tokens1)
        emb2 = self.normalize_semantic(tokens2)
        
        if emb1.shape[0] == 0 or emb2.shape[0] == 0:
            return 0.0
        
        # Aggregate embeddings (mean pooling)
        emb1_pooled = emb1.mean(dim=0)
        emb2_pooled = emb2.mean(dim=0)
        
        if method == 'cosine':
            similarity = F.cosine_similarity(emb1_pooled.unsqueeze(0),
                                            emb2_pooled.unsqueeze(0))
            return min(1.0, max(0.0, similarity.item()))
        
        elif method == 'euclidean':
            distance = torch.dist(emb1_pooled, emb2_pooled, p=2)
            # Convert distance to similarity (assuming max distance ~2 for normalized vectors)
            similarity = 1.0 - min(distance.item() / 2.0, 1.0)
            return similarity
        
        elif method == 'manhattan':
            distance = torch.dist(emb1_pooled, emb2_pooled, p=1)
            # Convert distance to similarity
            max_distance = self.embedding_dim * 2  # Max L1 distance for normalized vectors
            similarity = 1.0 - min(distance.item() / max_distance, 1.0)
            return similarity
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def find_semantic_clusters(self, token_sequences: List[List[int]],
                              n_clusters: int = 10) -> Dict[int, List[int]]:
        """
        Find semantic clusters in a collection of token sequences.
        
        Args:
            token_sequences: List of token sequences
            n_clusters: Number of clusters to find
            
        Returns:
            Dictionary mapping cluster ID to sequence indices
        """
        if not token_sequences:
            return {}
        
        # Get embeddings for all sequences
        sequence_embeddings = []
        for tokens in token_sequences:
            emb = self.normalize_semantic(tokens)
            if emb.shape[0] > 0:
                # Mean pooling
                sequence_embeddings.append(emb.mean(dim=0))
            else:
                sequence_embeddings.append(torch.zeros(self.embedding_dim))
        
        if not sequence_embeddings:
            return {}
        
        embeddings_matrix = torch.stack(sequence_embeddings)
        
        # Simple k-means clustering
        clusters = defaultdict(list)
        
        # Initialize cluster centers randomly
        n_sequences = len(sequence_embeddings)
        n_clusters = min(n_clusters, n_sequences)
        
        indices = torch.randperm(n_sequences)[:n_clusters]
        centers = embeddings_matrix[indices]
        
        # Iterate until convergence (simplified)
        for _ in range(10):  # Fixed iterations for simplicity
            # Assign sequences to nearest cluster
            new_clusters = defaultdict(list)
            
            for i, emb in enumerate(embeddings_matrix):
                distances = torch.cdist(emb.unsqueeze(0), centers)
                nearest = distances.argmin().item()
                new_clusters[nearest].append(i)
            
            # Update centers
            for cluster_id, member_indices in new_clusters.items():
                if member_indices:
                    member_embeddings = embeddings_matrix[member_indices]
                    centers[cluster_id] = member_embeddings.mean(dim=0)
            
            clusters = new_clusters
        
        return dict(clusters)


# Utility functions
def create_normalizer(tokenizer: Any, mode: str = 'canonical') -> TokenSpaceNormalizer:
    """
    Factory function to create a normalizer.
    
    Args:
        tokenizer: Tokenizer instance
        mode: Normalization mode
        
    Returns:
        Configured TokenSpaceNormalizer
    """
    return TokenSpaceNormalizer(tokenizer, mode)


def align_and_compare(tokens1: List[int], 
                     tokens2: List[int],
                     tokenizer: Any,
                     normalize: bool = True,
                     alignment_method: str = 'dynamic_programming') -> Dict[str, Any]:
    """
    Complete pipeline for aligning and comparing token sequences.
    
    Args:
        tokens1: First token sequence
        tokens2: Second token sequence
        tokenizer: Tokenizer for normalization
        normalize: Whether to normalize before alignment
        alignment_method: Method for alignment
        
    Returns:
        Dictionary with alignment results and metrics
    """
    # Create normalizer and aligner
    normalizer = TokenSpaceNormalizer(tokenizer) if normalize else None
    aligner = TokenAligner(normalizer)
    
    # Perform alignment
    alignment_result = aligner.align_sequences(tokens1, tokens2, method=alignment_method)
    
    # Compute additional metrics
    if normalizer:
        jaccard_distance = normalizer.compute_distance(tokens1, tokens2, method='jaccard')
        levenshtein_distance = normalizer.compute_distance(tokens1, tokens2, method='levenshtein')
    else:
        # Basic metrics without normalization
        set1, set2 = set(tokens1), set(tokens2)
        jaccard_distance = 1.0 - len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0
        levenshtein_distance = None
    
    return {
        'alignment': alignment_result.alignment,
        'alignment_score': alignment_result.score,
        'alignment_method': alignment_result.method,
        'jaccard_distance': jaccard_distance,
        'levenshtein_distance': levenshtein_distance,
        'metadata': alignment_result.metadata
    }