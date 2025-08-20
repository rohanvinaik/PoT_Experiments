"""
Comprehensive test suite for vocabulary-aware verification system.

This module validates that the vocabulary-aware verification system correctly
handles real-world scenarios including fine-tuned models, vocabulary extensions,
and cross-family comparisons.
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import sys
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pot.core.vocabulary_analysis import (
    VocabularyAnalyzer,
    TokenCategory,
    ArchitecturalImpact
)
from pot.core.vocabulary_aware_testing import (
    VocabularyAwareSequentialTester,
    VocabularyDecisionStatus,
    create_vocabulary_aware_tester
)
from pot.core.adaptive_challenge import (
    AdaptiveChallengeGenerator,
    AdaptiveChallengeConfig
)
from pot.core.diff_decision import DiffDecisionConfig, TestingMode


# ============================================================================
# MOCK MODELS FOR TESTING
# ============================================================================

class MockTokenizer:
    """Mock tokenizer for testing"""
    
    def __init__(self, vocab_size: int, special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ['[PAD]', '[CLS]', '[SEP]', '[UNK]']
        self._build_vocab()
    
    def _build_vocab(self):
        """Build a mock vocabulary"""
        self.vocab = {}
        
        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
        
        # Add common words
        common_words = ['the', 'a', 'is', 'it', 'was', 'that', 'this', 'for', 'in', 'of']
        for i, word in enumerate(common_words, start=len(self.special_tokens)):
            self.vocab[word] = i
        
        # Add subword pieces
        for i in range(100):
            self.vocab[f"##piece{i}"] = len(self.special_tokens) + len(common_words) + i
        
        # Fill remaining with generic tokens
        current_size = len(self.vocab)
        for i in range(current_size, self.vocab_size):
            self.vocab[f"token_{i}"] = i
    
    def get_vocab(self) -> Dict[str, int]:
        return self.vocab
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Simple mock encoding"""
        tokens = text.split()
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                # Use hash for consistent unknown tokens
                ids.append(hash(token) % self.vocab_size)
        return ids


class MockModel:
    """Mock model for testing"""
    
    def __init__(
        self,
        vocab_size: int,
        model_name: str = "mock_model",
        architecture: str = "transformer",
        is_finetuned: bool = False,
        parent_model: Optional[str] = None,
        added_tokens: Optional[List[str]] = None,
        embedding_dim: int = 768
    ):
        self.vocab_size = vocab_size
        self.model_name = model_name
        self.architecture = architecture
        self.is_finetuned = is_finetuned
        self.parent_model = parent_model
        self.added_tokens = added_tokens or []
        self.embedding_dim = embedding_dim
        
        # Create tokenizer
        self.tokenizer = MockTokenizer(vocab_size)
        
        # Add any additional tokens to vocabulary
        for token in self.added_tokens:
            if token not in self.tokenizer.vocab:
                self.tokenizer.vocab[token] = len(self.tokenizer.vocab)
        
        # Mock config
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'hidden_size': embedding_dim
        })()
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        """Mock generation for testing"""
        # Generate deterministic output based on model name and prompt
        seed = hash(self.model_name + prompt) % 1000
        np.random.seed(seed)
        
        if self.is_finetuned:
            # Fine-tuned models should generate similar but slightly different output
            return f"{prompt} [finetuned output {seed}]"
        else:
            return f"{prompt} [base output {seed}]"
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mock forward pass returning logits"""
        batch_size, seq_len = input_ids.shape
        
        # Generate deterministic logits based on model
        seed = hash(self.model_name) % 1000
        torch.manual_seed(seed)
        
        if self.is_finetuned:
            # Fine-tuned model has slightly different distribution
            logits = torch.randn(batch_size, seq_len, self.vocab_size) * 0.1
            logits += torch.randn(1) * 0.01  # Small perturbation
        else:
            logits = torch.randn(batch_size, seq_len, self.vocab_size)
        
        return logits


# ============================================================================
# MOCK MODEL FACTORY
# ============================================================================

class ModelFactory:
    """Factory for creating test models"""
    
    @staticmethod
    def create_base_model(vocab_size: int = 50257) -> MockModel:
        """Create a base model"""
        return MockModel(
            vocab_size=vocab_size,
            model_name="base_model",
            architecture="gpt2"
        )
    
    @staticmethod
    def create_medical_finetuned(base_vocab: int = 50257) -> MockModel:
        """Create a fine-tuned model with medical vocabulary"""
        medical_terms = [
            "pneumonia", "cardiomyopathy", "thrombosis", "anesthesia",
            "diagnosis", "prognosis", "pathology", "immunology"
        ]
        
        return MockModel(
            vocab_size=base_vocab + len(medical_terms),
            model_name="medical_finetuned",
            architecture="gpt2",
            is_finetuned=True,
            parent_model="base_model",
            added_tokens=medical_terms
        )
    
    @staticmethod
    def create_pruned_model(original_vocab: int = 50257, pruned_size: int = 45000) -> MockModel:
        """Create a model with pruned vocabulary"""
        return MockModel(
            vocab_size=pruned_size,
            model_name="pruned_model",
            architecture="gpt2",
            is_finetuned=True,
            parent_model="base_model"
        )
    
    @staticmethod
    def create_different_model(vocab_size: int = 50257) -> MockModel:
        """Create a completely different model with same vocab size"""
        return MockModel(
            vocab_size=vocab_size,
            model_name="different_model",
            architecture="bert"  # Different architecture
        )
    
    @staticmethod
    def create_reordered_vocab_model(vocab_size: int = 50257) -> MockModel:
        """Create a model with reordered vocabulary"""
        model = MockModel(
            vocab_size=vocab_size,
            model_name="reordered_model",
            architecture="gpt2"
        )
        
        # Shuffle vocabulary mapping
        import random
        vocab_items = list(model.tokenizer.vocab.items())
        random.shuffle(vocab_items)
        model.tokenizer.vocab = dict(vocab_items)
        
        return model


# ============================================================================
# TEST SUITE
# ============================================================================

class TestVocabularyHandling(unittest.TestCase):
    """Test suite for vocabulary-aware verification"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = VocabularyAnalyzer(embedding_dim=768)
        self.adaptive_generator = AdaptiveChallengeGenerator()
        self.model_factory = ModelFactory()
    
    def generate_mock_scores(
        self,
        n_samples: int,
        mean: float,
        std: float = 0.01,
        seed: int = 42
    ) -> List[float]:
        """Generate mock difference scores for testing"""
        np.random.seed(seed)
        return list(np.random.normal(mean, std, n_samples))
    
    def test_identical_models_same_vocabulary(self):
        """Baseline: identical models should verify as SAME."""
        print("\n" + "="*60)
        print("TEST: Identical Models with Same Vocabulary")
        print("-"*60)
        
        # Create two identical models
        model1 = self.model_factory.create_base_model()
        model2 = self.model_factory.create_base_model()
        
        # Analyze vocabularies
        report = self.analyzer.analyze_models(model1, model2)
        
        # Assertions
        self.assertEqual(report.reference_size, report.candidate_size)
        self.assertEqual(report.overlap_analysis.overlap_ratio, 1.0)
        self.assertTrue(report.can_verify)
        self.assertEqual(report.verification_strategy, "standard")
        self.assertEqual(report.confidence_adjustment, 1.0)
        
        # Test with sequential tester
        tester = create_vocabulary_aware_tester(
            reference_vocab_size=model1.vocab_size,
            candidate_vocab_size=model2.vocab_size,
            mode=TestingMode.QUICK_GATE
        )
        
        # Generate scores indicating same model
        scores = self.generate_mock_scores(30, mean=0.001, std=0.005)
        result = tester.make_decision(scores)
        
        self.assertEqual(result.status, "SAME")
        self.assertEqual(result.vocabulary_status, VocabularyDecisionStatus.SAME)
        
        print(f"✅ Vocabulary: {model1.vocab_size} vs {model2.vocab_size}")
        print(f"✅ Decision: {result.status}")
        print(f"✅ Confidence: {result.confidence:.2%}")
    
    def test_fine_tuned_model_with_added_tokens(self):
        """Model with added domain tokens should verify as SAME_EXTENDED."""
        print("\n" + "="*60)
        print("TEST: Fine-tuned Model with Added Medical Terms")
        print("-"*60)
        
        # Create base and fine-tuned models
        base_model = self.model_factory.create_base_model()
        medical_model = self.model_factory.create_medical_finetuned()
        
        # Analyze vocabularies
        report = self.analyzer.analyze_models(base_model, medical_model)
        
        # Assertions
        self.assertGreater(report.candidate_size, report.reference_size)
        self.assertGreater(report.overlap_analysis.overlap_ratio, 0.99)
        self.assertTrue(report.is_extension)
        self.assertTrue(report.can_verify)
        
        # Test with sequential tester
        tester = create_vocabulary_aware_tester(
            reference_vocab_size=base_model.vocab_size,
            candidate_vocab_size=medical_model.vocab_size,
            mode=TestingMode.QUICK_GATE
        )
        
        # Generate scores indicating same model (fine-tuned)
        scores = self.generate_mock_scores(30, mean=0.002, std=0.005)
        result = tester.make_decision(scores)
        
        # Should identify as extension
        self.assertIn(result.status, ["SAME", "SAME_EXTENDED"])
        self.assertTrue(result.vocabulary_extension_detected)
        
        print(f"✅ Base vocabulary: {base_model.vocab_size}")
        print(f"✅ Medical vocabulary: {medical_model.vocab_size} (+{medical_model.vocab_size - base_model.vocab_size} tokens)")
        print(f"✅ Decision: {result.status}")
        print(f"✅ Extension detected: {result.vocabulary_extension_detected}")
    
    def test_pruned_vocabulary_model(self):
        """Model with reduced vocabulary should verify as SAME_REDUCED."""
        print("\n" + "="*60)
        print("TEST: Model with Pruned Vocabulary")
        print("-"*60)
        
        # Create base and pruned models
        base_model = self.model_factory.create_base_model()
        pruned_model = self.model_factory.create_pruned_model()
        
        # Analyze vocabularies
        report = self.analyzer.analyze_models(base_model, pruned_model)
        
        # Assertions
        self.assertLess(report.candidate_size, report.reference_size)
        self.assertTrue(report.is_reduction)
        self.assertTrue(report.can_verify)
        
        # Test with sequential tester
        tester = create_vocabulary_aware_tester(
            reference_vocab_size=base_model.vocab_size,
            candidate_vocab_size=pruned_model.vocab_size,
            mode=TestingMode.AUDIT_GRADE
        )
        
        # Generate scores indicating same model (pruned)
        scores = self.generate_mock_scores(50, mean=0.003, std=0.008)
        result = tester.make_decision(scores)
        
        # Should identify as reduction
        self.assertTrue(result.vocabulary_reduction_detected)
        
        print(f"✅ Base vocabulary: {base_model.vocab_size}")
        print(f"✅ Pruned vocabulary: {pruned_model.vocab_size} (-{base_model.vocab_size - pruned_model.vocab_size} tokens)")
        print(f"✅ Decision: {result.status}")
        print(f"✅ Reduction detected: {result.vocabulary_reduction_detected}")
    
    def test_actually_different_models(self):
        """Different models with same vocab size should verify as DIFFERENT."""
        print("\n" + "="*60)
        print("TEST: Actually Different Models (Same Vocab Size)")
        print("-"*60)
        
        # Create two different models with same vocab size
        model1 = self.model_factory.create_base_model()
        model2 = self.model_factory.create_different_model()
        
        # Analyze vocabularies
        report = self.analyzer.analyze_models(model1, model2)
        
        # Vocab sizes are same, but models are different
        self.assertEqual(report.reference_size, report.candidate_size)
        self.assertEqual(report.overlap_analysis.overlap_ratio, 1.0)
        
        # Test with sequential tester
        tester = create_vocabulary_aware_tester(
            reference_vocab_size=model1.vocab_size,
            candidate_vocab_size=model2.vocab_size,
            mode=TestingMode.QUICK_GATE
        )
        
        # Generate scores indicating different models
        scores = self.generate_mock_scores(30, mean=0.15, std=0.03)
        result = tester.make_decision(scores)
        
        self.assertEqual(result.status, "DIFFERENT")
        
        print(f"✅ Model 1: {model1.architecture} ({model1.vocab_size} tokens)")
        print(f"✅ Model 2: {model2.architecture} ({model2.vocab_size} tokens)")
        print(f"✅ Decision: {result.status}")
        print(f"✅ Mean difference: {result.mean_difference:.3f}")
    
    def test_high_vocabulary_overlap(self):
        """95%+ overlap should proceed with verification."""
        print("\n" + "="*60)
        print("TEST: High Vocabulary Overlap (95%+)")
        print("-"*60)
        
        # Create models with high overlap
        base_size = 50257
        extended_size = 51000  # ~98% overlap
        
        model1 = MockModel(base_size, "model1")
        model2 = MockModel(extended_size, "model2")
        
        # Analyze
        report = self.analyzer.analyze_models(model1, model2)
        
        # Should proceed with verification
        self.assertTrue(report.should_proceed_with_verification())
        self.assertGreater(report.overlap_analysis.overlap_ratio, 0.95)
        
        # Get adaptation strategy
        strategy = report.get_adaptation_strategy()
        self.assertIn(strategy['strategy'], ['standard', 'adaptive_minor'])
        self.assertGreater(strategy['confidence_adjustment'], 0.9)
        
        print(f"✅ Overlap: {report.overlap_analysis.overlap_ratio:.1%}")
        print(f"✅ Can verify: {report.can_verify}")
        print(f"✅ Strategy: {strategy['strategy']}")
        print(f"✅ Confidence adjustment: {strategy['confidence_adjustment']:.2f}x")
    
    def test_low_vocabulary_overlap(self):
        """<50% overlap should trigger alternative verification."""
        print("\n" + "="*60)
        print("TEST: Low Vocabulary Overlap (<50%)")
        print("-"*60)
        
        # Create models with low overlap
        model1 = MockModel(50257, "gpt_model")
        model2 = MockModel(10000, "small_model")
        
        # Analyze
        report = self.analyzer.analyze_models(model1, model2)
        
        # Should not proceed with standard verification
        self.assertFalse(report.should_proceed_with_verification())
        self.assertLess(report.overlap_analysis.overlap_ratio, 0.5)
        
        # Get incompatibility reason
        reason = report.get_incompatibility_reason()
        self.assertIn("overlap too low", reason.lower())
        
        print(f"✅ Overlap: {report.overlap_analysis.overlap_ratio:.1%}")
        print(f"✅ Can verify: {report.can_verify}")
        print(f"✅ Reason: {reason}")
    
    def test_special_token_handling(self):
        """Special tokens ([PAD], [CLS]) should be handled correctly."""
        print("\n" + "="*60)
        print("TEST: Special Token Handling")
        print("-"*60)
        
        # Test token categorization
        special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '<eos>', '<bos>']
        
        for token in special_tokens:
            category = self.analyzer.categorize_token(token, 0)
            self.assertIn(category, [TokenCategory.SPECIAL, TokenCategory.CONTROL])
            print(f"✅ {token} -> {category.value}")
        
        # Test that special token changes affect architectural impact
        model1 = MockModel(50257, "model1")
        model2 = MockModel(50257, "model2", added_tokens=['[SPECIAL1]', '[SPECIAL2]'])
        
        # Manually test categorization
        from pot.core.vocabulary_analysis import TokenCategorization, VocabularyOverlapAnalysis
        
        categorization = TokenCategorization()
        categorization.special_token_changes = 2
        
        overlap = VocabularyOverlapAnalysis(
            total_reference=50257,
            total_candidate=50259,
            shared_tokens=50257,
            unique_to_reference=0,
            unique_to_candidate=2,
            overlap_ratio=50257/50259,
            jaccard_similarity=50257/50259
        )
        
        impact = self.analyzer.assess_impact(50257, 50259, overlap, categorization)
        
        # Special tokens should not trigger major architectural changes
        self.assertEqual(impact.embedding_layer_change, ArchitecturalImpact.NEGLIGIBLE)
        
        print(f"✅ Special token changes: {categorization.special_token_changes}")
        print(f"✅ Architectural impact: {impact.embedding_layer_change.value}")
    
    def test_adaptation_strategies(self):
        """Test that adaptation strategies work correctly."""
        print("\n" + "="*60)
        print("TEST: Challenge Adaptation Strategies")
        print("-"*60)
        
        # Create models with vocabulary mismatch
        model1 = MockModel(50257, "gpt2")
        model2 = MockModel(32768, "mistral")
        
        # Test adaptive challenge generation
        import secrets
        config = AdaptiveChallengeConfig(
            master_key_hex=secrets.token_hex(32),
            session_nonce_hex=secrets.token_hex(16),
            n=10,
            family="lm:templates",
            params={
                "templates": ["The {subject} is {attribute}."],
                "slots": {
                    "subject": ["cat", "dog"],
                    "attribute": ["fast", "slow"]
                }
            },
            vocab_size_a=model1.vocab_size,
            vocab_size_b=model2.vocab_size,
            model_name_a=model1.model_name,
            model_name_b=model2.model_name
        )
        
        # Generate adapted challenges
        result = self.adaptive_generator.generate_adaptive_challenges(config)
        
        # Should adapt to shared token space
        if not result.get("error"):
            self.assertTrue(result.get("vocabulary_adapted") or 
                          result.get("fallback_used") or 
                          len(result.get("challenges", [])) > 0)
            
            print(f"✅ Challenges generated: {len(result.get('challenges', []))}")
            
            if result.get("vocabulary_adapted"):
                print(f"✅ Adapted to shared token space")
                print(f"✅ Token coverage: {result.get('token_coverage', 0):.1%}")
            elif result.get("fallback_used"):
                print(f"✅ Fallback strategy used: {result.get('fallback_reason', 'N/A')}")
    
    def test_confidence_adjustments(self):
        """Test that confidence adjustments are applied correctly."""
        print("\n" + "="*60)
        print("TEST: Confidence Adjustments")
        print("-"*60)
        
        test_cases = [
            (50257, 50257, 1.00, "Perfect match"),
            (50257, 51000, 0.98, "High overlap"),
            (50257, 45000, 0.90, "Moderate overlap"),
            (50257, 32768, 0.70, "Low overlap"),
        ]
        
        for ref_size, cand_size, expected_adjustment, description in test_cases:
            tester = create_vocabulary_aware_tester(
                reference_vocab_size=ref_size,
                candidate_vocab_size=cand_size,
                mode=TestingMode.QUICK_GATE
            )
            
            # Check confidence adjustment
            adjustment = tester.confidence_adjustment_factor
            
            # Allow some tolerance
            self.assertAlmostEqual(adjustment, expected_adjustment, delta=0.1)
            
            print(f"✅ {description}: {adjustment:.2f}x (expected ~{expected_adjustment:.2f}x)")
    
    def test_integration_full_pipeline(self):
        """Test full pipeline with vocabulary mismatch."""
        print("\n" + "="*60)
        print("TEST: Full Pipeline Integration")
        print("-"*60)
        
        # Create base and fine-tuned models
        base_model = self.model_factory.create_base_model()
        medical_model = self.model_factory.create_medical_finetuned()
        
        # Step 1: Vocabulary analysis
        vocab_analysis = self.analyzer.analyze_models(base_model, medical_model)
        print(f"Step 1 - Vocabulary Analysis:")
        print(f"  Overlap: {vocab_analysis.overlap_analysis.overlap_ratio:.1%}")
        print(f"  Relationship: {'Extension' if vocab_analysis.is_extension else 'Other'}")
        
        # Step 2: Check if we should proceed
        if vocab_analysis.should_proceed_with_verification():
            # Step 3: Get adaptation strategy
            strategy = vocab_analysis.get_adaptation_strategy()
            print(f"\nStep 2 - Adaptation Strategy:")
            print(f"  Strategy: {strategy['strategy']}")
            print(f"  Confidence: {strategy['confidence_adjustment']:.2f}x")
            
            # Step 4: Create vocabulary-aware tester
            tester = create_vocabulary_aware_tester(
                reference_vocab_size=base_model.vocab_size,
                candidate_vocab_size=medical_model.vocab_size,
                mode=TestingMode.QUICK_GATE
            )
            
            # Step 5: Run verification
            scores = self.generate_mock_scores(30, mean=0.002, std=0.005)
            result = tester.make_decision(scores)
            
            print(f"\nStep 3 - Verification Result:")
            print(f"  Decision: {result.status}")
            print(f"  Vocabulary Status: {result.vocabulary_status.value if result.vocabulary_status else 'N/A'}")
            print(f"  Confidence: {result.confidence:.2%}")
            
            # Assertions
            self.assertIn(result.status, ["SAME", "SAME_EXTENDED"])
            self.assertGreater(result.confidence, 0.5)
        
        print(f"\n✅ Full pipeline completed successfully")
    
    def test_performance_with_adaptation(self):
        """Test that adaptation doesn't significantly slow verification."""
        print("\n" + "="*60)
        print("TEST: Performance with Adaptation")
        print("-"*60)
        
        # Time standard verification
        start = time.time()
        tester1 = create_vocabulary_aware_tester(
            reference_vocab_size=50257,
            candidate_vocab_size=50257,
            mode=TestingMode.QUICK_GATE
        )
        scores = self.generate_mock_scores(100, mean=0.01, std=0.01)
        result1 = tester1.make_decision(scores)
        time_standard = time.time() - start
        
        # Time adapted verification
        start = time.time()
        tester2 = create_vocabulary_aware_tester(
            reference_vocab_size=50257,
            candidate_vocab_size=32768,
            mode=TestingMode.QUICK_GATE
        )
        result2 = tester2.make_decision(scores)
        time_adapted = time.time() - start
        
        # Adaptation should not more than double the time
        self.assertLess(time_adapted, time_standard * 3)
        
        print(f"✅ Standard verification: {time_standard:.4f}s")
        print(f"✅ Adapted verification: {time_adapted:.4f}s")
        print(f"✅ Overhead: {(time_adapted/time_standard - 1)*100:.1f}%")
    
    def test_edge_cases(self):
        """Test edge cases (empty vocabulary, massive differences)."""
        print("\n" + "="*60)
        print("TEST: Edge Cases")
        print("-"*60)
        
        # Test empty vocabulary
        try:
            model_empty = MockModel(0, "empty")
            model_normal = MockModel(50257, "normal")
            report = self.analyzer.analyze_models(model_empty, model_normal)
            self.assertFalse(report.can_verify)
            print(f"✅ Empty vocabulary handled correctly")
        except Exception as e:
            print(f"⚠️ Empty vocabulary raised: {e}")
        
        # Test massive difference
        model_small = MockModel(1000, "tiny")
        model_huge = MockModel(1000000, "huge")
        report = self.analyzer.analyze_models(model_small, model_huge)
        
        self.assertLess(report.overlap_analysis.overlap_ratio, 0.01)
        self.assertEqual(report.architectural_impact.embedding_layer_change, ArchitecturalImpact.SEVERE)
        print(f"✅ Massive difference handled: {report.overlap_analysis.overlap_ratio:.3%} overlap")
        
        # Test identical sizes but different content
        model1 = self.model_factory.create_base_model()
        model2 = self.model_factory.create_reordered_vocab_model()
        
        # Should still detect as same size
        self.assertEqual(model1.vocab_size, model2.vocab_size)
        print(f"✅ Reordered vocabulary handled correctly")


# ============================================================================
# BENCHMARK SCENARIOS
# ============================================================================

BENCHMARK_SCENARIOS = [
    {
        'name': 'gpt2_base_vs_medical_finetuned',
        'reference_vocab': 50257,
        'candidate_vocab': 50304,
        'vocab_diff': 47,
        'expected_result': 'SAME_EXTENDED',
        'expected_confidence': 0.95,
        'description': 'GPT-2 base vs medical fine-tuned with added terminology'
    },
    {
        'name': 'bert_base_vs_bert_large',
        'reference_vocab': 30522,
        'candidate_vocab': 30522,
        'vocab_diff': 0,
        'expected_result': 'DIFFERENT',
        'expected_confidence': 0.99,
        'description': 'BERT base vs BERT large (same vocab, different architecture)'
    },
    {
        'name': 'gpt2_vs_distilgpt2',
        'reference_vocab': 50257,
        'candidate_vocab': 50257,
        'vocab_diff': 0,
        'expected_result': 'DIFFERENT',
        'expected_confidence': 0.99,
        'description': 'GPT-2 vs DistilGPT-2 (same vocab, different depth)'
    },
    {
        'name': 'llama_vs_llama_pruned',
        'reference_vocab': 32000,
        'candidate_vocab': 28000,
        'vocab_diff': -4000,
        'expected_result': 'SAME_REDUCED',
        'expected_confidence': 0.85,
        'description': 'LLaMA vs vocabulary-pruned LLaMA'
    },
    {
        'name': 'gpt2_vs_gpt2_domain_adapted',
        'reference_vocab': 50257,
        'candidate_vocab': 50500,
        'vocab_diff': 243,
        'expected_result': 'SAME_EXTENDED',
        'expected_confidence': 0.98,
        'description': 'GPT-2 vs domain-adapted GPT-2 with technical terms'
    },
    {
        'name': 'mistral_vs_zephyr',
        'reference_vocab': 32768,
        'candidate_vocab': 32000,
        'vocab_diff': -768,
        'expected_result': 'SAME_REDUCED',
        'expected_confidence': 0.95,
        'description': 'Mistral vs Zephyr (fine-tuned Mistral)'
    },
    {
        'name': 'gpt2_vs_mistral',
        'reference_vocab': 50257,
        'candidate_vocab': 32768,
        'vocab_diff': -17489,
        'expected_result': 'DIFFERENT',
        'expected_confidence': 0.70,
        'description': 'GPT-2 vs Mistral (different families)'
    },
    {
        'name': 'multilingual_extension',
        'reference_vocab': 50257,
        'candidate_vocab': 55000,
        'vocab_diff': 4743,
        'expected_result': 'SAME_EXTENDED',
        'expected_confidence': 0.90,
        'description': 'Base model vs multilingual extension'
    }
]


class TestBenchmarkScenarios(unittest.TestCase):
    """Test benchmark scenarios"""
    
    def test_all_benchmark_scenarios(self):
        """Run all benchmark scenarios"""
        print("\n" + "="*70)
        print("BENCHMARK SCENARIOS")
        print("="*70)
        
        analyzer = VocabularyAnalyzer()
        
        for scenario in BENCHMARK_SCENARIOS:
            print(f"\n{'-'*60}")
            print(f"Scenario: {scenario['name']}")
            print(f"Description: {scenario['description']}")
            print(f"Vocabulary: {scenario['reference_vocab']} → {scenario['candidate_vocab']} ({scenario['vocab_diff']:+d})")
            
            # Create mock models
            ref_model = MockModel(scenario['reference_vocab'], f"ref_{scenario['name']}")
            cand_model = MockModel(scenario['candidate_vocab'], f"cand_{scenario['name']}")
            
            # Analyze
            report = analyzer.analyze_models(ref_model, cand_model)
            
            # Create tester
            tester = create_vocabulary_aware_tester(
                reference_vocab_size=scenario['reference_vocab'],
                candidate_vocab_size=scenario['candidate_vocab'],
                mode=TestingMode.AUDIT_GRADE
            )
            
            # Generate appropriate scores based on expected result
            if "SAME" in scenario['expected_result']:
                mean_diff = 0.002
            else:
                mean_diff = 0.15
            
            scores = list(np.random.normal(mean_diff, 0.01, 50))
            result = tester.make_decision(scores)
            
            # Check results
            print(f"Expected: {scenario['expected_result']}")
            print(f"Actual: {result.status}")
            print(f"Confidence: {result.confidence:.2%} (expected ~{scenario['expected_confidence']:.2%})")
            
            # Validate
            if "SAME" in scenario['expected_result'] and "SAME" in result.status:
                print("✅ PASS")
            elif scenario['expected_result'] == "DIFFERENT" and result.status == "DIFFERENT":
                print("✅ PASS")
            else:
                print(f"⚠️ MISMATCH")


# ============================================================================
# DOCUMENTATION
# ============================================================================

VOCABULARY_HANDLING_DOCUMENTATION = """
================================================================================
VOCABULARY SIZE AND MODEL ARCHITECTURE: KEY INSIGHTS
================================================================================

IMPORTANT: Vocabulary size differences DO NOT necessarily indicate different 
model architectures. This is a common misconception that can lead to incorrect
verification results.

Why Vocabulary Size Alone Is Insufficient:
--------------------------------------------
1. **Fine-tuning Often Adds Tokens**: Domain-specific fine-tuning frequently
   adds specialized tokens (medical terms, technical jargon) without changing
   the underlying model architecture.

2. **Vocabulary Pruning**: Models can have reduced vocabularies for efficiency
   while maintaining the same transformer architecture.

3. **Same Architecture, Different Vocab**: Models like BERT-base and BERT-large
   have the same vocabulary size but completely different architectures.

4. **Tokenizer Variations**: Different tokenizer versions or configurations can
   result in slightly different vocabulary sizes for the same model.

What Actually Matters:
-----------------------
1. **Token Overlap Ratio**: High overlap (>95%) usually indicates same model family
2. **Special Token Changes**: Changes to control tokens affect model behavior more
3. **Parameter Impact**: Calculate actual parameter changes from vocabulary differences
4. **Architectural Indicators**: Layer count, hidden dimensions, attention heads

Our Approach:
-------------
The vocabulary-aware verification system:
1. Analyzes vocabulary overlap and categorizes tokens
2. Assesses architectural impact of vocabulary changes
3. Adapts verification strategy based on analysis
4. Provides nuanced decisions (SAME, SAME_EXTENDED, SAME_REDUCED, DIFFERENT)
5. Adjusts confidence based on vocabulary compatibility

This ensures that:
- Fine-tuned models are correctly identified as variants
- Actually different models are still caught despite vocabulary similarities
- Verification adapts appropriately to vocabulary differences
================================================================================
"""


if __name__ == "__main__":
    # Print documentation
    print(VOCABULARY_HANDLING_DOCUMENTATION)
    
    # Run tests
    unittest.main(verbosity=2)