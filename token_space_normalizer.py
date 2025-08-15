"""
Token Space Normalizer and Stochastic Decoding Controller

This module provides comprehensive tokenization normalization and controlled
stochastic decoding for consistent model verification across different
tokenization schemes and generation strategies.
"""

import hashlib
import json
import logging
import unicodedata
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
import time
from functools import lru_cache
import warnings

# Try to import torch and transformers
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some features will be limited.")

try:
    from transformers import (
        AutoTokenizer,
        GPT2Tokenizer,
        BertTokenizer,
        T5Tokenizer,
        PreTrainedTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. Using mock tokenizers.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenizerType(Enum):
    """Supported tokenizer types"""
    BPE = "bpe"  # Byte Pair Encoding (GPT-2, GPT-3)
    WORDPIECE = "wordpiece"  # WordPiece (BERT)
    SENTENCEPIECE = "sentencepiece"  # SentencePiece (T5, XLNet)
    CHARACTER = "character"  # Character-level
    WHITESPACE = "whitespace"  # Simple whitespace
    CUSTOM = "custom"


class SamplingMethod(Enum):
    """Sampling methods for generation"""
    ARGMAX = "argmax"  # Greedy decoding
    TEMPERATURE = "temperature"  # Temperature sampling
    TOP_K = "top_k"  # Top-k sampling
    TOP_P = "top_p"  # Nucleus sampling
    BEAM_SEARCH = "beam_search"  # Beam search
    TYPICAL = "typical"  # Typical sampling


@dataclass
class TokenizationResult:
    """Result of tokenization with metadata"""
    tokens: List[Union[int, str]]
    text: str
    tokenizer_type: TokenizerType
    vocab_size: int
    special_tokens: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AlignmentResult:
    """Result of token alignment between tokenizers"""
    text: str
    tokenizer_a_tokens: List[Any]
    tokenizer_b_tokens: List[Any]
    alignment_map: List[Tuple[int, int]]
    similarity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockTokenizer:
    """Mock tokenizer for when transformers is not available"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.vocab = {f"token_{i}": i for i in range(vocab_size)}
        self.inv_vocab = {i: f"token_{i}" for i in range(vocab_size)}
    
    def encode(self, text: str) -> List[int]:
        """Simple encoding based on character codes"""
        return [ord(c) % self.vocab_size for c in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """Simple decoding"""
        return ''.join(chr(tid % 128) for tid in token_ids)
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.split()


class TokenSpaceNormalizer:
    """
    Normalizes model outputs across different tokenization schemes
    for consistent verification.
    """
    
    def __init__(self, tokenizer_type: TokenizerType = TokenizerType.BPE,
                 vocab_size: int = 50000,
                 cache_size: int = 1000):
        """
        Initialize TokenSpaceNormalizer
        
        Args:
            tokenizer_type: Type of tokenizer to use
            vocab_size: Size of vocabulary
            cache_size: Size of LRU cache for tokenization results
        """
        self.tokenizer_type = tokenizer_type
        self.vocab_size = vocab_size
        self.cache_size = cache_size
        
        # Initialize tokenizers
        self.tokenizers = self._initialize_tokenizers()
        
        # Caching
        self._cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Unicode normalization form
        self.unicode_form = 'NFKC'  # Compatibility decomposition
        
        # Special token patterns
        self.special_token_patterns = {
            'pad': re.compile(r'\[PAD\]|<pad>|<|pad|>'),
            'unk': re.compile(r'\[UNK\]|<unk>|<|unk|>'),
            'bos': re.compile(r'\[CLS\]|<s>|<|startoftext|>'),
            'eos': re.compile(r'\[SEP\]|</s>|<|endoftext|>'),
            'mask': re.compile(r'\[MASK\]|<mask>')
        }
        
        # Subword regularization parameters
        self.subword_regularization_alpha = 0.1
        
        logger.info(f"TokenSpaceNormalizer initialized with {tokenizer_type.value}, vocab_size={vocab_size}")
    
    def _initialize_tokenizers(self) -> Dict[TokenizerType, Any]:
        """Initialize available tokenizers"""
        tokenizers = {}
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # BPE (GPT-2 style)
                tokenizers[TokenizerType.BPE] = GPT2Tokenizer.from_pretrained('gpt2')
            except:
                tokenizers[TokenizerType.BPE] = MockTokenizer(self.vocab_size)
            
            try:
                # WordPiece (BERT style)
                tokenizers[TokenizerType.WORDPIECE] = BertTokenizer.from_pretrained('bert-base-uncased')
            except:
                tokenizers[TokenizerType.WORDPIECE] = MockTokenizer(self.vocab_size)
            
            try:
                # SentencePiece (T5 style)
                tokenizers[TokenizerType.SENTENCEPIECE] = T5Tokenizer.from_pretrained('t5-small')
            except:
                tokenizers[TokenizerType.SENTENCEPIECE] = MockTokenizer(self.vocab_size)
        else:
            # Use mock tokenizers
            for ttype in TokenizerType:
                tokenizers[ttype] = MockTokenizer(self.vocab_size)
        
        # Always available tokenizers
        tokenizers[TokenizerType.CHARACTER] = self._character_tokenizer
        tokenizers[TokenizerType.WHITESPACE] = self._whitespace_tokenizer
        
        return tokenizers
    
    def _character_tokenizer(self, text: str, encode: bool = True) -> Union[List[int], List[str]]:
        """Character-level tokenizer"""
        if encode:
            return [ord(c) for c in text]
        else:
            return list(text)
    
    def _whitespace_tokenizer(self, text: str, encode: bool = True) -> Union[List[int], List[str]]:
        """Whitespace tokenizer"""
        tokens = text.split()
        if encode:
            # Simple hash-based encoding
            return [hash(token) % self.vocab_size for token in tokens]
        else:
            return tokens
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text using Unicode normalization
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Unicode normalization
        text = unicodedata.normalize(self.unicode_form, text)
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @lru_cache(maxsize=256)
    def normalize_token_sequence(self, tokens: tuple,
                                target_space: str = 'canonical') -> List[Any]:
        """
        Convert tokens to canonical representation
        
        Args:
            tokens: Token sequence (as tuple for caching)
            target_space: Target representation space
            
        Returns:
            Normalized token sequence
        """
        tokens = list(tokens)  # Convert back from tuple
        
        if target_space == 'canonical':
            # Convert to canonical integer representation
            normalized = []
            for token in tokens:
                if isinstance(token, str):
                    # Hash string tokens to integers
                    normalized.append(hash(token) % self.vocab_size)
                elif isinstance(token, int):
                    # Keep integers, ensure within vocab size
                    normalized.append(token % self.vocab_size)
                else:
                    # Handle other types
                    normalized.append(hash(str(token)) % self.vocab_size)
            
            return normalized
        
        elif target_space == 'string':
            # Convert to string representation
            return [str(token) for token in tokens]
        
        elif target_space == 'byte':
            # Convert to byte representation
            byte_tokens = []
            for token in tokens:
                if isinstance(token, str):
                    byte_tokens.extend(token.encode('utf-8'))
                elif isinstance(token, int):
                    byte_tokens.append(token % 256)
                else:
                    byte_tokens.extend(str(token).encode('utf-8'))
            
            return byte_tokens
        
        else:
            raise ValueError(f"Unknown target space: {target_space}")
    
    def align_token_boundaries(self, text: str, 
                              tokenizer_a: Any,
                              tokenizer_b: Any) -> AlignmentResult:
        """
        Align tokens from different tokenizers for comparison
        
        Args:
            text: Input text
            tokenizer_a: First tokenizer
            tokenizer_b: Second tokenizer
            
        Returns:
            AlignmentResult with mapping
        """
        # Normalize text first
        text = self.normalize_text(text)
        
        # Tokenize with both tokenizers
        if hasattr(tokenizer_a, 'tokenize'):
            tokens_a = tokenizer_a.tokenize(text)
        else:
            tokens_a = tokenizer_a(text, encode=False)
        
        if hasattr(tokenizer_b, 'tokenize'):
            tokens_b = tokenizer_b.tokenize(text)
        else:
            tokens_b = tokenizer_b(text, encode=False)
        
        # Build character position maps
        char_to_token_a = self._build_char_to_token_map(text, tokens_a)
        char_to_token_b = self._build_char_to_token_map(text, tokens_b)
        
        # Create alignment map
        alignment_map = []
        token_pairs = set()
        
        for char_idx in range(len(text)):
            if char_idx in char_to_token_a and char_idx in char_to_token_b:
                pair = (char_to_token_a[char_idx], char_to_token_b[char_idx])
                if pair not in token_pairs:
                    alignment_map.append(pair)
                    token_pairs.add(pair)
        
        # Calculate similarity score
        similarity = self._calculate_alignment_similarity(tokens_a, tokens_b, alignment_map)
        
        return AlignmentResult(
            text=text,
            tokenizer_a_tokens=tokens_a,
            tokenizer_b_tokens=tokens_b,
            alignment_map=alignment_map,
            similarity_score=similarity,
            metadata={
                'tokenizer_a_count': len(tokens_a),
                'tokenizer_b_count': len(tokens_b),
                'alignment_count': len(alignment_map)
            }
        )
    
    def _build_char_to_token_map(self, text: str, tokens: List[str]) -> Dict[int, int]:
        """Build mapping from character positions to token indices"""
        char_to_token = {}
        char_idx = 0
        
        for token_idx, token in enumerate(tokens):
            # Handle special tokens
            if token.startswith('[') and token.endswith(']'):
                continue
            if token.startswith('<') and token.endswith('>'):
                continue
            
            # Remove special prefixes (e.g., "##" for BERT)
            clean_token = token.replace('##', '').replace('▁', ' ')
            
            # Find token in text
            token_start = text.find(clean_token, char_idx)
            if token_start != -1:
                for i in range(token_start, token_start + len(clean_token)):
                    char_to_token[i] = token_idx
                char_idx = token_start + len(clean_token)
        
        return char_to_token
    
    def _calculate_alignment_similarity(self, tokens_a: List[str],
                                       tokens_b: List[str],
                                       alignment_map: List[Tuple[int, int]]) -> float:
        """Calculate similarity score for token alignment"""
        if not alignment_map:
            return 0.0
        
        # Calculate coverage
        covered_a = len(set(a for a, _ in alignment_map))
        covered_b = len(set(b for _, b in alignment_map))
        
        coverage_a = covered_a / len(tokens_a) if tokens_a else 0
        coverage_b = covered_b / len(tokens_b) if tokens_b else 0
        
        # Average coverage as similarity
        return (coverage_a + coverage_b) / 2
    
    def compute_token_invariant_hash(self, text: str,
                                    normalize_unicode: bool = True,
                                    normalize_case: bool = False) -> str:
        """
        Generate hash that's invariant to tokenization differences
        
        Args:
            text: Input text
            normalize_unicode: Whether to normalize Unicode
            normalize_case: Whether to normalize case
            
        Returns:
            Invariant hash string
        """
        # Normalize text
        if normalize_unicode:
            text = self.normalize_text(text)
        
        if normalize_case:
            text = text.lower()
        
        # Remove all special tokens
        for pattern in self.special_token_patterns.values():
            text = pattern.sub('', text)
        
        # Remove all whitespace for true invariance
        invariant_text = ''.join(text.split())
        
        # Generate hash
        hash_obj = hashlib.sha256(invariant_text.encode('utf-8'))
        
        # Add structural information
        structure_info = {
            'char_count': len(invariant_text),
            'word_count': len(text.split()),
            'unique_chars': len(set(invariant_text))
        }
        
        hash_obj.update(json.dumps(structure_info, sort_keys=True).encode())
        
        return hash_obj.hexdigest()
    
    def tokenize_with_subword_regularization(self, text: str,
                                            alpha: Optional[float] = None,
                                            num_samples: int = 1) -> List[List[Any]]:
        """
        Apply subword regularization for robustness
        
        Args:
            text: Input text
            alpha: Regularization strength
            num_samples: Number of samples to generate
            
        Returns:
            List of tokenization variants
        """
        if alpha is None:
            alpha = self.subword_regularization_alpha
        
        variants = []
        
        for _ in range(num_samples):
            # Get base tokenization
            if self.tokenizer_type in self.tokenizers:
                tokenizer = self.tokenizers[self.tokenizer_type]
                
                if hasattr(tokenizer, 'encode'):
                    tokens = tokenizer.encode(text)
                else:
                    tokens = tokenizer(text, encode=True)
                
                # Apply regularization
                if alpha > 0 and len(tokens) > 1:
                    # Randomly merge or split tokens
                    regularized = []
                    i = 0
                    while i < len(tokens):
                        if np.random.random() < alpha and i < len(tokens) - 1:
                            # Merge tokens
                            if isinstance(tokens[i], int):
                                merged = (tokens[i] + tokens[i + 1]) % self.vocab_size
                            else:
                                merged = str(tokens[i]) + str(tokens[i + 1])
                            regularized.append(merged)
                            i += 2
                        else:
                            regularized.append(tokens[i])
                            i += 1
                    
                    variants.append(regularized)
                else:
                    variants.append(tokens)
            else:
                # Fallback to character tokenization
                variants.append(self._character_tokenizer(text))
        
        return variants
    
    def handle_unknown_tokens(self, tokens: List[Any],
                            fallback_strategy: str = 'hash') -> List[Any]:
        """
        Handle unknown tokens with fallback strategies
        
        Args:
            tokens: Token sequence
            fallback_strategy: Strategy for handling unknown tokens
            
        Returns:
            Processed token sequence
        """
        processed = []
        
        for token in tokens:
            if self._is_unknown_token(token):
                if fallback_strategy == 'hash':
                    # Hash unknown token to known range
                    processed.append(hash(str(token)) % self.vocab_size)
                elif fallback_strategy == 'skip':
                    # Skip unknown tokens
                    continue
                elif fallback_strategy == 'replace':
                    # Replace with UNK token
                    processed.append(0)  # Assuming 0 is UNK
                else:
                    processed.append(token)
            else:
                processed.append(token)
        
        return processed
    
    def _is_unknown_token(self, token: Any) -> bool:
        """Check if token is unknown"""
        if isinstance(token, str):
            return token in ['[UNK]', '<unk>', 'UNK']
        elif isinstance(token, int):
            return token >= self.vocab_size
        return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self._cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'max_cache_size': self.cache_size
        }
    
    def clear_cache(self):
        """Clear tokenization cache"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Tokenization cache cleared")


class StochasticDecodingController:
    """
    Controls stochastic decoding for reproducible and verifiable generation
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize StochasticDecodingController
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed if seed is not None else int(time.time())
        self.rng = np.random.RandomState(self.seed)
        
        if TORCH_AVAILABLE:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        
        # Default generation parameters
        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.95
        self.typical_p = 0.95
        self.repetition_penalty = 1.0
        
        # Deterministic mode flag
        self.deterministic_mode = False
        
        # Generation history for verification
        self.generation_history = []
        
        logger.info(f"StochasticDecodingController initialized with seed={self.seed}")
    
    def set_deterministic_mode(self, temperature: float = 0.0,
                              top_k: int = 1,
                              top_p: float = 1.0):
        """
        Configure fully deterministic generation
        
        Args:
            temperature: Temperature for softmax (0 = deterministic)
            top_k: Number of top tokens to consider
            top_p: Cumulative probability for nucleus sampling
        """
        self.deterministic_mode = True
        self.temperature = max(1e-10, temperature)  # Avoid division by zero
        self.top_k = top_k
        self.top_p = top_p
        
        logger.info(f"Deterministic mode set: temp={temperature}, top_k={top_k}, top_p={top_p}")
    
    def controlled_sampling(self, logits: Union[np.ndarray, 'torch.Tensor'],
                          method: SamplingMethod = SamplingMethod.ARGMAX,
                          **kwargs) -> Union[int, List[int]]:
        """
        Sample with controlled randomness
        
        Args:
            logits: Model output logits
            method: Sampling method to use
            **kwargs: Additional sampling parameters
            
        Returns:
            Sampled token ID(s)
        """
        if not TORCH_AVAILABLE and not isinstance(logits, np.ndarray):
            raise ValueError("PyTorch not available, please provide numpy array")
        
        # Convert to tensor if needed
        if isinstance(logits, np.ndarray):
            if TORCH_AVAILABLE:
                logits = torch.from_numpy(logits).float()
            else:
                # Use numpy implementation
                return self._numpy_sampling(logits, method, **kwargs)
        
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Apply repetition penalty if specified
        if 'token_history' in kwargs and self.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, kwargs['token_history'])
        
        if method == SamplingMethod.ARGMAX:
            return torch.argmax(logits, dim=-1).item()
        
        elif method == SamplingMethod.TEMPERATURE:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).item()
        
        elif method == SamplingMethod.TOP_K:
            return self._top_k_sampling(logits, k=kwargs.get('k', self.top_k))
        
        elif method == SamplingMethod.TOP_P:
            return self._top_p_sampling(logits, p=kwargs.get('p', self.top_p))
        
        elif method == SamplingMethod.TYPICAL:
            return self._typical_sampling(logits, p=kwargs.get('p', self.typical_p))
        
        elif method == SamplingMethod.BEAM_SEARCH:
            return self._beam_search(logits, beam_size=kwargs.get('beam_size', 5))
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def _numpy_sampling(self, logits: np.ndarray, method: SamplingMethod, **kwargs) -> int:
        """Numpy implementation of sampling methods"""
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        if method == SamplingMethod.ARGMAX:
            return np.argmax(logits)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        if method == SamplingMethod.TEMPERATURE:
            return self.rng.choice(len(probs), p=probs)
        
        elif method == SamplingMethod.TOP_K:
            k = kwargs.get('k', self.top_k)
            top_k_idx = np.argsort(logits)[-k:]
            top_k_probs = probs[top_k_idx]
            top_k_probs = top_k_probs / top_k_probs.sum()
            return top_k_idx[self.rng.choice(len(top_k_idx), p=top_k_probs)]
        
        elif method == SamplingMethod.TOP_P:
            p = kwargs.get('p', self.top_p)
            sorted_idx = np.argsort(logits)[::-1]
            sorted_probs = probs[sorted_idx]
            cumsum = np.cumsum(sorted_probs)
            mask = cumsum <= p
            if mask.sum() == 0:
                mask[0] = True
            nucleus = sorted_idx[mask]
            nucleus_probs = probs[nucleus]
            nucleus_probs = nucleus_probs / nucleus_probs.sum()
            return nucleus[self.rng.choice(len(nucleus), p=nucleus_probs)]
        
        return np.argmax(logits)
    
    def _top_k_sampling(self, logits: 'torch.Tensor', k: int) -> int:
        """Top-k sampling"""
        top_k_values, top_k_indices = torch.topk(logits, k=min(k, logits.size(-1)))
        top_k_probs = F.softmax(top_k_values, dim=-1)
        sampled_idx = torch.multinomial(top_k_probs, num_samples=1)
        return top_k_indices[sampled_idx].item()
    
    def _top_p_sampling(self, logits: 'torch.Tensor', p: float) -> int:
        """Nucleus (top-p) sampling"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
    
    def _typical_sampling(self, logits: 'torch.Tensor', p: float) -> int:
        """Typical sampling"""
        # Calculate entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # Calculate surprisal
        surprisal = -torch.log(probs + 1e-10)
        
        # Calculate typical score
        typical_score = torch.abs(surprisal - entropy.unsqueeze(-1))
        
        # Sort by typical score
        sorted_scores, sorted_indices = torch.sort(typical_score)
        cumulative_probs = torch.cumsum(probs[sorted_indices], dim=-1)
        
        # Find cutoff
        cutoff_idx = torch.searchsorted(cumulative_probs, p)
        selected_indices = sorted_indices[:cutoff_idx + 1]
        
        selected_probs = probs[selected_indices]
        selected_probs = selected_probs / selected_probs.sum()
        
        sampled_idx = torch.multinomial(selected_probs, num_samples=1)
        return selected_indices[sampled_idx].item()
    
    def _beam_search(self, logits: 'torch.Tensor', beam_size: int) -> List[int]:
        """Simple beam search (returns top sequence)"""
        # For single step, just return top-k
        top_k_values, top_k_indices = torch.topk(logits, k=min(beam_size, logits.size(-1)))
        return top_k_indices.tolist()
    
    def _apply_repetition_penalty(self, logits: 'torch.Tensor',
                                 token_history: List[int]) -> 'torch.Tensor':
        """Apply repetition penalty to logits"""
        for token_id in set(token_history):
            if token_id < logits.size(-1):
                logits[token_id] /= self.repetition_penalty
        return logits
    
    def generate_verification_response(self, model: Any,
                                      challenge: Any,
                                      num_variants: int = 5,
                                      max_length: int = 100) -> List[str]:
        """
        Generate multiple controlled variants for robustness testing
        
        Args:
            model: Model to generate with
            challenge: Input challenge
            num_variants: Number of variants to generate
            max_length: Maximum generation length
            
        Returns:
            List of generated responses
        """
        responses = []
        
        # Store original parameters
        orig_temp = self.temperature
        orig_top_k = self.top_k
        orig_top_p = self.top_p
        
        # Generate variants with different settings
        settings = [
            {'temperature': 0.0, 'top_k': 1},  # Deterministic
            {'temperature': 0.5, 'top_k': 10},  # Low randomness
            {'temperature': 0.8, 'top_k': 40},  # Medium randomness
            {'temperature': 1.0, 'top_p': 0.9},  # Nucleus sampling
            {'temperature': 0.7, 'top_p': 0.95},  # Typical settings
        ]
        
        for i in range(min(num_variants, len(settings))):
            setting = settings[i]
            
            # Apply settings
            self.temperature = setting.get('temperature', self.temperature)
            self.top_k = setting.get('top_k', self.top_k)
            self.top_p = setting.get('top_p', self.top_p)
            
            # Generate response
            response = self._generate_single_response(model, challenge, max_length)
            responses.append(response)
            
            # Store in history
            self.generation_history.append({
                'challenge': str(challenge),
                'response': response,
                'settings': setting,
                'timestamp': time.time()
            })
        
        # Restore original parameters
        self.temperature = orig_temp
        self.top_k = orig_top_k
        self.top_p = orig_top_p
        
        return responses
    
    def _generate_single_response(self, model: Any, challenge: Any,
                                 max_length: int) -> str:
        """Generate a single response"""
        # This is a placeholder - actual implementation would depend on model type
        if hasattr(model, 'generate'):
            # Hugging Face model
            if TORCH_AVAILABLE:
                with torch.no_grad():
                    outputs = model.generate(
                        challenge,
                        max_length=max_length,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                        do_sample=self.temperature > 0
                    )
                return outputs
        
        # Fallback: return mock response
        return f"Response to {challenge} with temp={self.temperature}"
    
    def compute_semantic_similarity(self, responses: List[str],
                                  method: str = 'jaccard') -> float:
        """
        Verify semantic consistency across variants
        
        Args:
            responses: List of response strings
            method: Similarity metric to use
            
        Returns:
            Average pairwise similarity score
        """
        if len(responses) < 2:
            return 1.0
        
        similarities = []
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                if method == 'jaccard':
                    sim = self._jaccard_similarity(responses[i], responses[j])
                elif method == 'levenshtein':
                    sim = self._levenshtein_similarity(responses[i], responses[j])
                elif method == 'cosine':
                    sim = self._cosine_similarity(responses[i], responses[j])
                else:
                    sim = self._token_overlap(responses[i], responses[j])
                
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between texts"""
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _levenshtein_similarity(self, text1: str, text2: str) -> float:
        """Calculate normalized Levenshtein similarity"""
        if text1 == text2:
            return 1.0
        
        len1, len2 = len(text1), len(text2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Create distance matrix
        dist = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            dist[i][0] = i
        for j in range(len2 + 1):
            dist[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if text1[i-1] == text2[j-1]:
                    dist[i][j] = dist[i-1][j-1]
                else:
                    dist[i][j] = 1 + min(dist[i-1][j], dist[i][j-1], dist[i-1][j-1])
        
        # Normalize
        max_len = max(len1, len2)
        return 1.0 - (dist[len1][len2] / max_len)
    
    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between texts"""
        # Simple bag-of-words cosine similarity
        words1 = text1.split()
        words2 = text2.split()
        
        # Create vocabulary
        vocab = list(set(words1 + words2))
        
        # Create vectors
        vec1 = [words1.count(word) for word in vocab]
        vec2 = [words2.count(word) for word in vocab]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _token_overlap(self, text1: str, text2: str) -> float:
        """Calculate token overlap ratio"""
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())
        
        if not tokens1 and not tokens2:
            return 1.0
        
        overlap = len(tokens1.intersection(tokens2))
        total = len(tokens1) + len(tokens2)
        
        return (2 * overlap) / total if total > 0 else 0.0
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        if not self.generation_history:
            return {
                'total_generations': 0,
                'deterministic_ratio': 0,
                'average_similarity': 0
            }
        
        deterministic_count = sum(
            1 for g in self.generation_history
            if g['settings'].get('temperature', 1.0) == 0.0
        )
        
        # Calculate average similarity of recent generations
        recent_responses = [g['response'] for g in self.generation_history[-10:]]
        avg_similarity = self.compute_semantic_similarity(recent_responses) if len(recent_responses) > 1 else 1.0
        
        return {
            'total_generations': len(self.generation_history),
            'deterministic_ratio': deterministic_count / len(self.generation_history),
            'average_similarity': avg_similarity,
            'seed': self.seed,
            'current_settings': {
                'temperature': self.temperature,
                'top_k': self.top_k,
                'top_p': self.top_p,
                'deterministic_mode': self.deterministic_mode
            }
        }
    
    def reset_seed(self, seed: int):
        """Reset random seed for reproducibility"""
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        logger.info(f"Random seed reset to {seed}")


class IntegratedVerificationSystem:
    """
    Integrates TokenSpaceNormalizer and StochasticDecodingController
    with challenge-response verification system
    """
    
    def __init__(self, tokenizer_type: TokenizerType = TokenizerType.BPE,
                 vocab_size: int = 50000,
                 seed: Optional[int] = None):
        """
        Initialize integrated verification system
        
        Args:
            tokenizer_type: Type of tokenizer
            vocab_size: Vocabulary size
            seed: Random seed
        """
        self.normalizer = TokenSpaceNormalizer(tokenizer_type, vocab_size)
        self.controller = StochasticDecodingController(seed)
        
        # Verification thresholds
        self.min_similarity_threshold = 0.85
        self.max_variance_threshold = 0.15
        
        # Verification history
        self.verification_results = []
        
        logger.info("IntegratedVerificationSystem initialized")
    
    def verify_model_response(self, model: Any, challenge: Any,
                             num_variants: int = 3) -> Dict[str, Any]:
        """
        Verify model response with tokenization normalization and controlled generation
        
        Args:
            model: Model to verify
            challenge: Challenge input
            num_variants: Number of variants to generate
            
        Returns:
            Verification result dictionary
        """
        # Generate response variants
        responses = self.controller.generate_verification_response(
            model, challenge, num_variants
        )
        
        # Normalize responses
        normalized_hashes = []
        for response in responses:
            hash_val = self.normalizer.compute_token_invariant_hash(response)
            normalized_hashes.append(hash_val)
        
        # Check consistency
        unique_hashes = len(set(normalized_hashes))
        hash_consistency = unique_hashes / len(normalized_hashes)
        
        # Compute semantic similarity
        semantic_similarity = self.controller.compute_semantic_similarity(responses)
        
        # Determine verification result
        is_verified = (
            semantic_similarity >= self.min_similarity_threshold and
            hash_consistency >= (1 - self.max_variance_threshold)
        )
        
        result = {
            'verified': is_verified,
            'num_variants': num_variants,
            'unique_hashes': unique_hashes,
            'hash_consistency': hash_consistency,
            'semantic_similarity': semantic_similarity,
            'responses': responses,
            'normalized_hashes': normalized_hashes,
            'timestamp': time.time()
        }
        
        self.verification_results.append(result)
        
        return result
    
    def cross_tokenizer_verification(self, text: str,
                                    tokenizer_types: List[TokenizerType]) -> Dict[str, Any]:
        """
        Verify text across multiple tokenizer types
        
        Args:
            text: Text to verify
            tokenizer_types: List of tokenizer types to test
            
        Returns:
            Cross-tokenizer verification results
        """
        results = {}
        hashes = []
        
        for ttype in tokenizer_types:
            if ttype in self.normalizer.tokenizers:
                tokenizer = self.normalizer.tokenizers[ttype]
                
                # Tokenize
                if hasattr(tokenizer, 'encode'):
                    tokens = tokenizer.encode(text)
                else:
                    tokens = tokenizer(text, encode=True)
                
                # Normalize
                normalized = self.normalizer.normalize_token_sequence(
                    tuple(tokens), 'canonical'
                )
                
                # Compute hash
                hash_val = self.normalizer.compute_token_invariant_hash(text)
                
                results[ttype.value] = {
                    'token_count': len(tokens),
                    'normalized_tokens': normalized[:10],  # First 10 for display
                    'hash': hash_val
                }
                
                hashes.append(hash_val)
        
        # Check consistency across tokenizers
        unique_hashes = len(set(hashes))
        consistency = 1.0 if unique_hashes == 1 else 1.0 / unique_hashes
        
        return {
            'tokenizer_results': results,
            'unique_hashes': unique_hashes,
            'consistency_score': consistency,
            'all_hashes_match': unique_hashes == 1
        }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Token Space Normalizer and Stochastic Decoding Controller")
    print("=" * 70)
    
    # Initialize components
    normalizer = TokenSpaceNormalizer(TokenizerType.BPE, vocab_size=50000)
    controller = StochasticDecodingController(seed=42)
    
    # Test text
    test_text = "The quick brown fox jumps over the lazy dog."
    
    print(f"\nTest text: {test_text}")
    
    # Test normalization
    print("\n" + "=" * 70)
    print("Text Normalization")
    print("=" * 70)
    
    normalized = normalizer.normalize_text(test_text)
    print(f"Normalized: {normalized}")
    
    invariant_hash = normalizer.compute_token_invariant_hash(test_text)
    print(f"Invariant hash: {invariant_hash[:40]}...")
    
    # Test tokenization with regularization
    print("\n" + "=" * 70)
    print("Subword Regularization")
    print("=" * 70)
    
    variants = normalizer.tokenize_with_subword_regularization(test_text, alpha=0.2, num_samples=3)
    for i, variant in enumerate(variants):
        print(f"Variant {i+1}: {variant[:10]}... (length: {len(variant)})")
    
    # Test controlled sampling
    print("\n" + "=" * 70)
    print("Controlled Sampling")
    print("=" * 70)
    
    # Simulate logits
    logits = np.random.randn(50)
    
    # Deterministic sampling
    controller.set_deterministic_mode(temperature=0.0)
    deterministic_token = controller.controlled_sampling(logits, SamplingMethod.ARGMAX)
    print(f"Deterministic token: {deterministic_token}")
    
    # Stochastic sampling
    controller.temperature = 0.8
    controller.top_k = 10
    stochastic_tokens = [
        controller.controlled_sampling(logits, SamplingMethod.TOP_K)
        for _ in range(5)
    ]
    print(f"Stochastic tokens (top-k): {stochastic_tokens}")
    
    # Test semantic similarity
    print("\n" + "=" * 70)
    print("Semantic Similarity")
    print("=" * 70)
    
    responses = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown fox jumped over a lazy dog.",
        "The fast brown fox leaps over the sleepy dog.",
        "Something completely different."
    ]
    
    similarity = controller.compute_semantic_similarity(responses[:3])
    print(f"Similarity (similar texts): {similarity:.4f}")
    
    similarity_with_different = controller.compute_semantic_similarity(responses)
    print(f"Similarity (with different): {similarity_with_different:.4f}")
    
    # Test integrated verification
    print("\n" + "=" * 70)
    print("Integrated Verification")
    print("=" * 70)
    
    verifier = IntegratedVerificationSystem(TokenizerType.BPE, vocab_size=50000, seed=42)
    
    # Cross-tokenizer verification
    cross_results = verifier.cross_tokenizer_verification(
        test_text,
        [TokenizerType.BPE, TokenizerType.WORDPIECE, TokenizerType.CHARACTER]
    )
    
    print(f"Cross-tokenizer consistency: {cross_results['consistency_score']:.4f}")
    print(f"All hashes match: {cross_results['all_hashes_match']}")
    
    for tokenizer, result in cross_results['tokenizer_results'].items():
        print(f"  {tokenizer}: {result['token_count']} tokens, hash: {result['hash'][:20]}...")
    
    # Statistics
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)
    
    cache_stats = normalizer.get_cache_stats()
    print(f"Cache stats: {cache_stats}")
    
    gen_stats = controller.get_generation_stats()
    print(f"Generation stats: {gen_stats}")
    
    print("\n" + "=" * 70)
    print("Complete")
    print("=" * 70)