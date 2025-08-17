"""
Tests for Token Space Normalizer
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any
from dataclasses import dataclass

# Import the classes to test

# Add parent directory to path for pot imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pot.security.token_space_normalizer import (
    TokenSpaceNormalizer,
    TokenAligner,
    SemanticNormalizer,
    AlignmentResult,
    create_normalizer,
    align_and_compare
)


class TestTokenNormalizer(unittest.TestCase):
    """Test suite for token normalization"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.get_vocab.return_value = {
            '[PAD]': 0,
            '[CLS]': 101,
            '[SEP]': 102,
            '[MASK]': 103,
            'hello': 7592,
            'world': 2088,
            'Hello': 7593,
            'World': 2089,
            'test': 3231,
            'this': 2023,
            'is': 2003,
            'a': 1037,
            '##ing': 2075,
            '.': 1012,
            ',': 1010
        }
        
        self.mock_tokenizer.encode.side_effect = self._mock_encode
        self.mock_tokenizer.decode.side_effect = self._mock_decode
        self.mock_tokenizer.special_tokens_map = {
            'pad_token': '[PAD]',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'mask_token': '[MASK]'
        }
        
        # Create normalizer
        self.normalizer = TokenSpaceNormalizer(self.mock_tokenizer, mode='canonical')
    
    def _mock_encode(self, text, add_special_tokens=True):
        """Mock tokenizer encode function"""
        vocab = self.mock_tokenizer.get_vocab()
        tokens = []
        
        if add_special_tokens:
            tokens.append(vocab['[CLS]'])
        
        # Simple word-level tokenization
        for word in text.lower().split():
            if word in vocab:
                tokens.append(vocab[word])
            else:
                tokens.append(vocab.get('[UNK]', 100))
        
        if add_special_tokens:
            tokens.append(vocab['[SEP]'])
        
        return tokens
    
    def _mock_decode(self, token_ids, skip_special_tokens=False):
        """Mock tokenizer decode function"""
        inverse_vocab = {v: k for k, v in self.mock_tokenizer.get_vocab().items()}
        
        words = []
        for token_id in token_ids:
            if token_id in inverse_vocab:
                token = inverse_vocab[token_id]
                if skip_special_tokens and token in ['[PAD]', '[CLS]', '[SEP]', '[MASK]']:
                    continue
                words.append(token)
        
        return ' '.join(words)
    
    def test_canonical_normalization(self):
        """Test canonical normalization mode"""
        tokens = [7593, 2089]  # Hello World
        normalized = self.normalizer.normalize_canonical(tokens)
        
        # Should normalize case
        self.assertIsInstance(normalized, list)
        # Normalized tokens should be different (lowercased)
        self.assertNotEqual(normalized, tokens)
    
    def test_string_normalization(self):
        """Test string-level normalization"""
        text = "Hello   world!!!"
        normalized = self.normalizer.normalize_string(text)
        
        # Should normalize whitespace and punctuation
        self.assertNotIn('   ', normalized)
        self.assertEqual(normalized.count('!'), 3)  # Preserves punctuation count
    
    def test_byte_normalization(self):
        """Test byte-level normalization"""
        tokens = [7592, 2088]  # hello world
        
        self.normalizer.mode = 'byte'
        normalized = self.normalizer.normalize(tokens)
        
        # Should return bytes
        self.assertIsInstance(normalized, bytes)
    
    def test_semantic_normalization(self):
        """Test semantic normalization"""
        tokens = [7592, 2088]  # hello world
        
        self.normalizer.mode = 'semantic'
        normalized = self.normalizer.normalize(tokens)
        
        # Should return tensor
        self.assertIsInstance(normalized, torch.Tensor)
        self.assertEqual(normalized.shape[0], len(tokens))
    
    def test_special_token_handling(self):
        """Test that special tokens are preserved"""
        tokens = [101, 7592, 2088, 102]  # [CLS] hello world [SEP]
        normalized = self.normalizer.normalize_canonical(tokens)
        
        # Special tokens should be preserved
        self.assertEqual(normalized[0], 101)  # [CLS]
        self.assertEqual(normalized[-1], 102)  # [SEP]
    
    def test_compute_distance_jaccard(self):
        """Test Jaccard distance computation"""
        tokens1 = [7592, 2088, 3231]  # hello world test
        tokens2 = [7592, 2088, 2023]  # hello world this
        
        distance = self.normalizer.compute_distance(tokens1, tokens2, method='jaccard')
        
        # 2 common tokens out of 4 unique -> Jaccard = 2/4 = 0.5, distance = 0.5
        self.assertAlmostEqual(distance, 0.5, places=5)
    
    def test_compute_distance_levenshtein(self):
        """Test Levenshtein distance computation"""
        tokens1 = [7592, 2088]
        tokens2 = [7592, 3231]
        
        distance = self.normalizer.compute_distance(tokens1, tokens2, method='levenshtein')
        
        # One substitution in length 2 -> distance = 1/2 = 0.5
        self.assertAlmostEqual(distance, 0.5, places=5)
    
    def test_unicode_normalization(self):
        """Test Unicode normalization in canonical mode"""
        # Mock tokenizer for unicode
        self.mock_tokenizer.get_vocab.return_value['café'] = 5000
        self.mock_tokenizer.get_vocab.return_value['cafe'] = 5001
        
        # Test with unicode character
        inverse_vocab = {v: k for k, v in self.mock_tokenizer.get_vocab().items()}
        self.normalizer.inverse_vocab = inverse_vocab
        
        tokens = [5000]  # café with accent
        normalized = self.normalizer.normalize_canonical(tokens)
        
        # Should normalize unicode
        self.assertIsNotNone(normalized)
    
    def test_whitespace_normalization(self):
        """Test whitespace normalization"""
        text = "  hello\t\tworld  \n  test  "
        normalized = self.normalizer.normalize_string(text)
        
        # Should normalize all whitespace
        self.assertNotIn('\t', normalized)
        self.assertNotIn('\n', normalized)
        self.assertNotIn('  ', normalized)
    
    def test_cache_functionality(self):
        """Test that normalization cache works"""
        tokens = [7592, 2088]
        
        # First call
        result1 = self.normalizer.normalize_canonical(tokens)
        
        # Second call (should use cache)
        result2 = self.normalizer.normalize_canonical(tokens)
        
        # Results should be identical
        self.assertEqual(result1, result2)


class TestTokenAligner(unittest.TestCase):
    """Test suite for token alignment"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.aligner = TokenAligner()
        
        # Test sequences
        self.seq1 = [1, 2, 3, 4, 5]
        self.seq2 = [1, 2, 4, 5, 6]
    
    def test_dp_alignment(self):
        """Test dynamic programming alignment"""
        result = self.aligner.align_sequences(
            self.seq1,
            self.seq2,
            method='dynamic_programming'
        )
        
        # Check result structure
        self.assertIsInstance(result, AlignmentResult)
        self.assertIsInstance(result.alignment, list)
        self.assertIsInstance(result.score, float)
        
        # Check alignment quality
        self.assertGreater(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        
        # Should find common subsequence [1, 2, 4, 5]
        self.assertEqual(len(result.alignment), 4)
    
    def test_needleman_wunsch_alignment(self):
        """Test Needleman-Wunsch alignment"""
        result = self.aligner.align_sequences(
            self.seq1,
            self.seq2,
            method='needleman_wunsch'
        )
        
        # Check result
        self.assertIsInstance(result, AlignmentResult)
        self.assertGreater(result.score, 0.0)
        
        # Check metadata
        self.assertEqual(result.metadata['seq1_length'], len(self.seq1))
        self.assertEqual(result.metadata['seq2_length'], len(self.seq2))
    
    def test_semantic_alignment(self):
        """Test semantic alignment (falls back to exact matching)"""
        result = self.aligner.align_sequences(
            self.seq1,
            self.seq2,
            method='semantic'
        )
        
        # Should find exact matches
        self.assertIsInstance(result, AlignmentResult)
        self.assertGreater(len(result.alignment), 0)
    
    def test_identical_sequence_alignment(self):
        """Test alignment of identical sequences"""
        seq = [1, 2, 3, 4, 5]
        result = self.aligner.align_sequences(seq, seq)
        
        # Perfect alignment
        self.assertEqual(result.score, 1.0)
        self.assertEqual(len(result.alignment), len(seq))
        
        # All positions should be aligned
        for i, (idx1, idx2) in enumerate(result.alignment):
            self.assertEqual(idx1, i)
            self.assertEqual(idx2, i)
    
    def test_empty_sequence_alignment(self):
        """Test alignment with empty sequences"""
        result = self.aligner.align_sequences([], [1, 2, 3])
        
        self.assertEqual(result.score, 0.0)
        self.assertEqual(len(result.alignment), 0)
        
        # Both empty
        result = self.aligner.align_sequences([], [])
        self.assertEqual(result.score, 1.0)  # Empty sequences are identical
    
    def test_alignment_score_computation(self):
        """Test alignment score computation"""
        # Perfect alignment
        alignment = [(0, 0), (1, 1), (2, 2)]
        seq1 = [1, 2, 3]
        seq2 = [1, 2, 3]
        
        score = self.aligner.compute_alignment_score(alignment, seq1, seq2)
        self.assertEqual(score, 1.0)
        
        # Partial alignment
        alignment = [(0, 0), (2, 2)]
        score = self.aligner.compute_alignment_score(alignment, seq1, seq2)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)
    
    def test_consecutive_alignment_bonus(self):
        """Test that consecutive alignments get bonus score"""
        # Consecutive alignment
        alignment1 = [(0, 0), (1, 1), (2, 2)]
        seq = [1, 2, 3, 4]
        score1 = self.aligner.compute_alignment_score(alignment1, seq, seq)
        
        # Non-consecutive alignment
        alignment2 = [(0, 0), (2, 2), (3, 3)]
        score2 = self.aligner.compute_alignment_score(alignment2, seq, seq)
        
        # Consecutive should score higher
        self.assertGreaterEqual(score1, score2)
    
    def test_visualize_alignment(self):
        """Test alignment visualization"""
        alignment = [(0, 0), (1, 1), (3, 2)]
        seq1 = [1, 2, 3, 4]
        seq2 = [1, 2, 4, 5]
        
        visualization = self.aligner.visualize_alignment(
            alignment, seq1, seq2
        )
        
        # Should return string visualization
        self.assertIsInstance(visualization, str)
        self.assertIn('|', visualization)  # Should contain alignment markers
        self.assertIn('-', visualization)  # Should contain gap markers


class TestSemanticNormalizer(unittest.TestCase):
    """Test suite for semantic normalization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.normalizer = SemanticNormalizer(
            embedding_model=None,
            num_clusters=100,
            embedding_dim=768
        )
    
    def test_initialization(self):
        """Test semantic normalizer initialization"""
        self.assertEqual(self.normalizer.num_clusters, 100)
        self.assertEqual(self.normalizer.embedding_dim, 768)
        self.assertIsNotNone(self.normalizer.cluster_centers)
        self.assertEqual(self.normalizer.cluster_centers.shape, (100, 768))
    
    def test_normalize_semantic(self):
        """Test semantic normalization of tokens"""
        tokens = [1, 2, 3, 4, 5]
        normalized = self.normalizer.normalize_semantic(tokens)
        
        # Should return tensor
        self.assertIsInstance(normalized, torch.Tensor)
        self.assertEqual(normalized.shape[0], len(tokens))
        self.assertEqual(normalized.shape[1], self.normalizer.embedding_dim)
    
    def test_semantic_quantization(self):
        """Test semantic quantization"""
        # Create random embeddings
        embeddings = torch.randn(5, 768)
        quantized = self.normalizer._semantic_quantization(embeddings)
        
        # Should return quantized embeddings
        self.assertEqual(quantized.shape, embeddings.shape)
        
        # Each quantized embedding should be a cluster center
        for emb in quantized:
            # Check if it's one of the cluster centers
            distances = torch.cdist(emb.unsqueeze(0), self.normalizer.cluster_centers)
            min_distance = distances.min()
            self.assertAlmostEqual(min_distance.item(), 0.0, places=5)
    
    def test_compute_semantic_similarity_cosine(self):
        """Test cosine similarity computation"""
        tokens1 = [1, 2, 3]
        tokens2 = [1, 2, 3]
        
        similarity = self.normalizer.compute_semantic_similarity(
            tokens1, tokens2, method='cosine'
        )
        
        # Same tokens should have high similarity
        self.assertGreater(similarity, 0.5)
        self.assertLessEqual(similarity, 1.0)
    
    def test_compute_semantic_similarity_euclidean(self):
        """Test Euclidean distance-based similarity"""
        tokens1 = [1, 2, 3]
        tokens2 = [4, 5, 6]
        
        similarity = self.normalizer.compute_semantic_similarity(
            tokens1, tokens2, method='euclidean'
        )
        
        # Should return valid similarity
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_compute_semantic_similarity_manhattan(self):
        """Test Manhattan distance-based similarity"""
        tokens1 = [1]
        tokens2 = [2]
        
        similarity = self.normalizer.compute_semantic_similarity(
            tokens1, tokens2, method='manhattan'
        )
        
        # Should return valid similarity
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_find_semantic_clusters(self):
        """Test semantic clustering"""
        token_sequences = [
            [1, 2, 3],
            [1, 2, 4],
            [5, 6, 7],
            [5, 6, 8],
            [9, 10, 11]
        ]
        
        clusters = self.normalizer.find_semantic_clusters(
            token_sequences,
            n_clusters=3
        )
        
        # Should return dictionary of clusters
        self.assertIsInstance(clusters, dict)
        
        # Should have at most 3 clusters
        self.assertLessEqual(len(clusters), 3)
        
        # Each sequence should be in exactly one cluster
        all_indices = []
        for indices in clusters.values():
            all_indices.extend(indices)
        self.assertEqual(len(all_indices), len(token_sequences))
        self.assertEqual(len(set(all_indices)), len(token_sequences))
    
    def test_empty_sequence_handling(self):
        """Test handling of empty sequences"""
        # Empty token list
        normalized = self.normalizer.normalize_semantic([])
        self.assertEqual(normalized.shape[0], 0)
        
        # Empty sequence similarity
        similarity = self.normalizer.compute_semantic_similarity([], [1, 2, 3])
        self.assertEqual(similarity, 0.0)
        
        # Empty clustering
        clusters = self.normalizer.find_semantic_clusters([])
        self.assertEqual(len(clusters), 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.get_vocab.return_value = {
            'hello': 1,
            'world': 2,
            'test': 3
        }
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        self.mock_tokenizer.decode.return_value = "hello world test"
    
    def test_create_normalizer(self):
        """Test normalizer factory function"""
        normalizer = create_normalizer(self.mock_tokenizer, mode='canonical')
        
        self.assertIsInstance(normalizer, TokenSpaceNormalizer)
        self.assertEqual(normalizer.mode, 'canonical')
    
    def test_align_and_compare(self):
        """Test complete alignment and comparison pipeline"""
        tokens1 = [1, 2, 3, 4]
        tokens2 = [1, 2, 4, 5]
        
        result = align_and_compare(
            tokens1,
            tokens2,
            self.mock_tokenizer,
            normalize=True,
            alignment_method='dynamic_programming'
        )
        
        # Check result structure
        self.assertIn('alignment', result)
        self.assertIn('alignment_score', result)
        self.assertIn('alignment_method', result)
        self.assertIn('jaccard_distance', result)
        self.assertIn('metadata', result)
        
        # Check values
        self.assertIsInstance(result['alignment'], list)
        self.assertGreaterEqual(result['alignment_score'], 0.0)
        self.assertLessEqual(result['alignment_score'], 1.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.get_vocab.return_value = {}
        self.normalizer = TokenSpaceNormalizer(self.mock_tokenizer)
        self.aligner = TokenAligner()
    
    def test_invalid_normalization_mode(self):
        """Test invalid normalization mode"""
        self.normalizer.mode = 'invalid'
        
        with self.assertRaises(ValueError):
            self.normalizer.normalize([1, 2, 3])
    
    def test_invalid_distance_method(self):
        """Test invalid distance computation method"""
        with self.assertRaises(ValueError):
            self.normalizer.compute_distance([1], [2], method='invalid')
    
    def test_invalid_alignment_method(self):
        """Test invalid alignment method"""
        with self.assertRaises(ValueError):
            self.aligner.align_sequences([1], [2], method='invalid')
    
    def test_invalid_similarity_method(self):
        """Test invalid similarity method"""
        semantic_norm = SemanticNormalizer()
        
        with self.assertRaises(ValueError):
            semantic_norm.compute_semantic_similarity(
                [1], [2], method='invalid'
            )
    
    def test_very_long_sequences(self):
        """Test with very long token sequences"""
        long_seq1 = list(range(1000))
        long_seq2 = list(range(500, 1500))
        
        # Should handle long sequences
        result = self.aligner.align_sequences(long_seq1, long_seq2)
        self.assertIsInstance(result, AlignmentResult)
        self.assertGreater(len(result.alignment), 0)
    
    def test_unicode_in_tokens(self):
        """Test handling of unicode in token strings"""
        # Create tokenizer with unicode
        mock_tokenizer = Mock()
        mock_tokenizer.get_vocab.return_value = {
            '世界': 1000,
            '🌍': 1001,
            'café': 1002
        }
        
        normalizer = TokenSpaceNormalizer(mock_tokenizer)
        
        # Should handle unicode tokens
        tokens = [1000, 1001, 1002]
        normalized = normalizer.normalize_canonical(tokens)
        self.assertIsNotNone(normalized)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full system"""
    
    def test_full_pipeline(self):
        """Test complete normalization and alignment pipeline"""
        # Create mock tokenizer
        mock_tokenizer = Mock()
        vocab = {
            '[CLS]': 101,
            '[SEP]': 102,
            'the': 1,
            'quick': 2,
            'brown': 3,
            'fox': 4,
            'jumps': 5,
            'lazy': 6,
            'dog': 7
        }
        mock_tokenizer.get_vocab.return_value = vocab
        mock_tokenizer.special_tokens_map = {
            'cls_token': '[CLS]',
            'sep_token': '[SEP]'
        }
        
        # Create components
        normalizer = TokenSpaceNormalizer(mock_tokenizer, mode='canonical')
        aligner = TokenAligner(normalizer)
        
        # Test sequences
        seq1 = [101, 1, 2, 3, 4, 102]  # [CLS] the quick brown fox [SEP]
        seq2 = [101, 1, 6, 7, 102]     # [CLS] the lazy dog [SEP]
        
        # Perform alignment
        result = aligner.align_sequences(seq1, seq2)
        
        # Check result
        self.assertIsInstance(result, AlignmentResult)
        self.assertGreater(result.score, 0.0)
        
        # Should align [CLS], 'the', and [SEP]
        self.assertGreaterEqual(len(result.alignment), 3)
    
    def test_semantic_pipeline(self):
        """Test semantic normalization pipeline"""
        # Create semantic normalizer
        semantic_norm = SemanticNormalizer(
            embedding_model=None,
            num_clusters=50,
            embedding_dim=256
        )
        
        # Test token sequences
        sequences = [
            [1, 2, 3, 4],
            [1, 2, 5, 6],
            [7, 8, 9, 10],
            [7, 8, 11, 12]
        ]
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                sim = semantic_norm.compute_semantic_similarity(
                    sequences[i],
                    sequences[j],
                    method='cosine'
                )
                similarities.append((i, j, sim))
        
        # Check that we computed all pairs
        expected_pairs = len(sequences) * (len(sequences) - 1) // 2
        self.assertEqual(len(similarities), expected_pairs)
        
        # All similarities should be valid
        for i, j, sim in similarities:
            self.assertGreaterEqual(sim, -1.0)
            self.assertLessEqual(sim, 1.0)
        
        # Find clusters
        clusters = semantic_norm.find_semantic_clusters(sequences, n_clusters=2)
        
        # Should produce clusters
        self.assertGreater(len(clusters), 0)
        self.assertLessEqual(len(clusters), 2)


if __name__ == '__main__':
    unittest.main()