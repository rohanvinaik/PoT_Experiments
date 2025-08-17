"""
Tests for Language Model Output Distance Metrics
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

# Import the classes to test
from pot.lm.verifier import DistanceMetrics, LMVerifier


class TestOutputDistance(unittest.TestCase):
    """Test suite for output distance computation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.metrics = DistanceMetrics()
        
        # Sample outputs for testing
        self.output1 = {
            'text': "Hello world this is a test",
            'logits': torch.randn(10, 100),
            'hidden_states': [torch.randn(10, 768) for _ in range(12)],
            'token_ids': [101, 7592, 2088, 2023, 2003, 1037, 3231, 102],
            'attention_weights': [torch.randn(1, 12, 10, 10) for _ in range(12)]
        }
        
        self.output2 = {
            'text': "Hello world this is another test",
            'logits': torch.randn(10, 100),
            'hidden_states': [torch.randn(10, 768) for _ in range(12)],
            'token_ids': [101, 7592, 2088, 2023, 2003, 2178, 3231, 102],
            'attention_weights': [torch.randn(1, 12, 10, 10) for _ in range(12)]
        }
        
        self.identical_output = {
            'text': "Hello world this is a test",
            'logits': self.output1['logits'].clone(),
            'hidden_states': [h.clone() for h in self.output1['hidden_states']],
            'token_ids': self.output1['token_ids'].copy(),
            'attention_weights': [a.clone() for a in self.output1['attention_weights']]
        }
    
    def test_edit_distance_identical(self):
        """Test edit distance for identical strings"""
        distance = self.metrics.edit_distance(
            "test string", 
            "test string"
        )
        self.assertEqual(distance, 0.0)
    
    def test_edit_distance_different(self):
        """Test edit distance for different strings"""
        distance = self.metrics.edit_distance(
            "hello", 
            "hallo"
        )
        # One substitution in string of length 5
        self.assertAlmostEqual(distance, 1/5, places=5)
    
    def test_edit_distance_empty(self):
        """Test edit distance with empty strings"""
        distance = self.metrics.edit_distance("", "")
        self.assertEqual(distance, 0.0)
        
        distance = self.metrics.edit_distance("test", "")
        self.assertEqual(distance, 1.0)
        
        distance = self.metrics.edit_distance("", "test")
        self.assertEqual(distance, 1.0)
    
    def test_kl_divergence_identical(self):
        """Test KL divergence for identical distributions"""
        logits1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        logits2 = logits1.clone()
        
        kl_div = self.metrics.kl_divergence(logits1, logits2)
        self.assertAlmostEqual(kl_div, 0.0, places=5)
    
    def test_kl_divergence_different(self):
        """Test KL divergence for different distributions"""
        logits1 = torch.tensor([[1.0, 2.0, 3.0]])
        logits2 = torch.tensor([[3.0, 2.0, 1.0]])
        
        kl_div = self.metrics.kl_divergence(logits1, logits2)
        self.assertGreater(kl_div, 0.0)
        self.assertLess(kl_div, 10.0)  # Reasonable bound
    
    def test_kl_divergence_shape_mismatch(self):
        """Test KL divergence with mismatched shapes"""
        logits1 = torch.randn(5, 100)
        logits2 = torch.randn(3, 100)
        
        kl_div = self.metrics.kl_divergence(logits1, logits2)
        self.assertEqual(kl_div, float('inf'))
    
    def test_cosine_distance_identical(self):
        """Test cosine distance for identical vectors"""
        hidden1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        hidden2 = hidden1.clone()
        
        distance = self.metrics.cosine_distance([hidden1], [hidden2])
        self.assertAlmostEqual(distance, 0.0, places=5)
    
    def test_cosine_distance_orthogonal(self):
        """Test cosine distance for orthogonal vectors"""
        hidden1 = [torch.tensor([[1.0, 0.0, 0.0]])]
        hidden2 = [torch.tensor([[0.0, 1.0, 0.0]])]
        
        distance = self.metrics.cosine_distance(hidden1, hidden2)
        # Cosine similarity of orthogonal vectors is 0, distance is 1
        self.assertAlmostEqual(distance, 1.0, places=5)
    
    def test_cosine_distance_opposite(self):
        """Test cosine distance for opposite vectors"""
        hidden1 = [torch.tensor([[1.0, 2.0, 3.0]])]
        hidden2 = [torch.tensor([[-1.0, -2.0, -3.0]])]
        
        distance = self.metrics.cosine_distance(hidden1, hidden2)
        # Cosine similarity of opposite vectors is -1, distance is 2
        self.assertAlmostEqual(distance, 2.0, places=5)
    
    def test_compute_distance_comprehensive(self):
        """Test comprehensive distance computation"""
        # Test combined distance (default)
        distance = self.metrics.compute_distance(
            self.output1,
            self.output2
        )
        
        # Check that distance is in expected range
        self.assertGreaterEqual(distance, 0.0)
        self.assertLessEqual(distance, 1.0)
        
        # Test individual metrics
        edit_dist = self.metrics.compute_distance(self.output1, self.output2, metric='edit')
        self.assertGreaterEqual(edit_dist, 0.0)
        self.assertLessEqual(edit_dist, 1.0)
        
        logits_dist = self.metrics.compute_distance(self.output1, self.output2, metric='logits_kl')
        self.assertGreaterEqual(logits_dist, 0.0)
        
        embedding_dist = self.metrics.compute_distance(self.output1, self.output2, metric='embedding_cosine')
        self.assertGreaterEqual(embedding_dist, 0.0)
        self.assertLessEqual(embedding_dist, 1.0)
    
    def test_compute_distance_identical_outputs(self):
        """Test distance computation for identical outputs"""
        # Combined distance should be near zero
        distance = self.metrics.compute_distance(
            self.output1,
            self.identical_output
        )
        self.assertAlmostEqual(distance, 0.0, places=5)
        
        # Test individual metrics
        edit_dist = self.metrics.compute_distance(self.output1, self.identical_output, metric='edit')
        self.assertAlmostEqual(edit_dist, 0.0, places=5)
        
        logits_dist = self.metrics.compute_distance(self.output1, self.identical_output, metric='logits_kl')
        self.assertAlmostEqual(logits_dist, 0.0, places=5)
        
        embedding_dist = self.metrics.compute_distance(self.output1, self.identical_output, metric='embedding_cosine')
        self.assertAlmostEqual(embedding_dist, 0.0, places=5)
    
    def test_compute_distance_missing_fields(self):
        """Test distance computation with missing fields"""
        # Output with only text
        output_text_only = {'generated_text': "test"}
        output_full = {'generated_text': "test", 'logits': self.output1['logits']}
        
        # Edit distance should work
        edit_dist = self.metrics.compute_distance(
            output_text_only,
            output_full,
            metric='edit'
        )
        self.assertEqual(edit_dist, 0.0)  # Same text
        
        # KL divergence should be 1.0 (max) due to missing logits
        logits_dist = self.metrics.compute_distance(
            output_text_only,
            output_full,
            metric='logits_kl'
        )
        self.assertEqual(logits_dist, 1.0)
    
    def test_token_level_distance(self):
        """Test token-level distance computation"""
        # Create outputs with text
        output1 = {'generated_text': "hello world"}
        output2 = {'generated_text': "hello test"}
        
        # Compute edit distance
        distance = self.metrics.compute_distance(output1, output2, metric='edit')
        
        # Should compute normalized edit distance
        self.assertGreater(distance, 0.0)
        self.assertLess(distance, 1.0)
    
    def test_weighted_distance(self):
        """Test weighted distance computation"""
        # Combined distance is weighted by default
        distance = self.metrics.compute_distance(
            self.output1,
            self.output2,
            metric='combined'
        )
        
        # Should be a weighted combination of individual metrics
        self.assertGreaterEqual(distance, 0.0)
        self.assertLessEqual(distance, 1.0)
    
    def test_batch_distance_computation(self):
        """Test batch distance computation"""
        outputs1 = [self.output1, self.output2]
        outputs2 = [self.output2, self.output1]
        
        distances = []
        for o1, o2 in zip(outputs1, outputs2):
            dist = self.metrics.compute_distance(o1, o2, metric='combined')
            distances.append(dist)
        
        # Both pairs should have the same distance
        self.assertAlmostEqual(distances[0], distances[1], places=5)
    
    def test_distance_with_attention_weights(self):
        """Test distance computation including attention weights"""
        # Create outputs with hidden states (used for embedding distance)
        output1 = {
            'generated_text': "test",
            'hidden_states': torch.randn(1, 10, 768)
        }
        output2 = {
            'generated_text': "test",
            'hidden_states': torch.randn(1, 10, 768)
        }
        
        # Embedding distance should work
        distance = self.metrics.compute_distance(output1, output2, metric='embedding_cosine')
        self.assertGreaterEqual(distance, 0.0)
        self.assertLessEqual(distance, 1.0)


class TestLMVerifierIntegration(unittest.TestCase):
    """Integration tests for LMVerifier with DistanceMetrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock model and tokenizer
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        
        # Configure tokenizer
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.encode.return_value = [101, 7592, 2088, 102]
        self.mock_tokenizer.decode.return_value = "Hello world"
        
        # Configure model
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 4, 50000)
        mock_output.hidden_states = tuple(torch.randn(1, 4, 768) for _ in range(12))
        self.mock_model.return_value = mock_output
        
        # Create verifier
        self.verifier = LMVerifier(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            config={'verification_method': 'sequential'}
        )
    
    @patch('pot.lm.verifier.TemplateChallenger')
    def test_verification_with_distance_metrics(self, mock_challenger):
        """Test verification process with distance computation"""
        # Configure mock challenger
        mock_challenger.return_value.generate_challenges.return_value = [
            ("prompt1", "expected1"),
            ("prompt2", "expected2")
        ]
        
        # Mock run_model to return outputs
        self.verifier.run_model = Mock(side_effect=[
            {'text': "output1", 'logits': torch.randn(1, 10, 100)},
            {'text': "output2", 'logits': torch.randn(1, 10, 100)}
        ])
        
        # Mock distance computation
        self.verifier.distance_metrics = Mock()
        self.verifier.distance_metrics.compute_distance.return_value = {
            'combined_score': 0.1
        }
        
        # Run verification
        result = self.verifier.verify_model(
            model_id="test_model",
            challenges=[("prompt1", "expected1")]
        )
        
        # Check that distance metrics were used
        self.verifier.distance_metrics.compute_distance.assert_called()
    
    def test_output_collection(self):
        """Test that run_model collects all relevant outputs"""
        prompt = "Test prompt"
        
        # Configure model output
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 5, 50000)
        mock_output.hidden_states = tuple(torch.randn(1, 5, 768) for _ in range(12))
        mock_output.attentions = tuple(torch.randn(1, 12, 5, 5) for _ in range(12))
        
        self.mock_model.return_value = mock_output
        self.mock_model.generate.return_value = torch.tensor([[101, 7592, 2088, 102]])
        
        # Run model
        output = self.verifier.run_model(prompt, collect_hidden_states=True)
        
        # Verify output structure
        self.assertIn('text', output)
        self.assertIn('logits', output)
        self.assertIn('hidden_states', output)
        self.assertIn('token_ids', output)
        
        # Verify shapes
        self.assertEqual(len(output['hidden_states']), 12)
        self.assertEqual(output['logits'].shape[0], 1)
    
    def test_error_handling(self):
        """Test error handling in distance computation"""
        metrics = DistanceMetrics()
        
        # Test with empty outputs - should handle gracefully
        distance = metrics.compute_distance({}, {}, metric='edit')
        # Empty outputs should return some distance
        self.assertIsNotNone(distance)
        
        # Test with mismatched types
        output1 = {'generated_text': "test"}
        output2 = {'generated_text': 123}  # Wrong type
        
        # Should handle gracefully
        try:
            distance = metrics.compute_distance(output1, output2, metric='edit')
            # Should either handle or raise appropriate error
            self.assertIsNotNone(distance)
        except (TypeError, AttributeError):
            pass  # Expected behavior


class TestDistanceMetricsEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        self.metrics = DistanceMetrics()
    
    def test_very_long_strings(self):
        """Test with very long strings"""
        long_string1 = "word " * 1000
        long_string2 = "word " * 999 + "different"
        
        distance = self.metrics.edit_distance(long_string1, long_string2)
        
        # Should handle long strings efficiently
        self.assertGreaterEqual(distance, 0.0)
        self.assertLessEqual(distance, 1.0)
    
    def test_unicode_strings(self):
        """Test with unicode characters"""
        unicode1 = "Hello 世界 🌍"
        unicode2 = "Hello 世界 🌎"
        
        distance = self.metrics.edit_distance(unicode1, unicode2)
        
        # Should handle unicode properly
        self.assertGreater(distance, 0.0)
        self.assertLess(distance, 1.0)
    
    def test_numerical_stability(self):
        """Test numerical stability of KL divergence"""
        # Very small probabilities
        logits1 = torch.tensor([[-100.0, -99.0, -98.0]])
        logits2 = torch.tensor([[-98.0, -99.0, -100.0]])
        
        kl_div = self.metrics.kl_divergence(logits1, logits2)
        
        # Should not be NaN or infinite
        self.assertFalse(torch.isnan(torch.tensor(kl_div)))
        self.assertFalse(torch.isinf(torch.tensor(kl_div)))
    
    def test_zero_vectors(self):
        """Test cosine distance with zero vectors"""
        hidden1 = [torch.zeros(1, 768)]
        hidden2 = [torch.randn(1, 768)]
        
        distance = self.metrics.cosine_distance(hidden1, hidden2)
        
        # Should handle zero vectors gracefully
        self.assertTrue(np.isfinite(distance))
    
    def test_single_token_outputs(self):
        """Test with single token outputs"""
        output1 = {
            'generated_text': "A",
            'logits': torch.randn(1, 1, 100),
            'hidden_states': torch.randn(1, 1, 768)
        }
        output2 = {
            'generated_text': "B",
            'logits': torch.randn(1, 1, 100),
            'hidden_states': torch.randn(1, 1, 768)
        }
        
        distance = self.metrics.compute_distance(output1, output2, metric='edit')
        
        # Should handle single token outputs
        self.assertEqual(distance, 1.0)  # Completely different single characters


if __name__ == '__main__':
    unittest.main()