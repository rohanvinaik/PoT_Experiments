#!/usr/bin/env python3
"""
Test script for vocabulary-aware statistical verification.

Demonstrates how the enhanced statistical testing framework accounts for
vocabulary differences and adjusts confidence accordingly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from typing import List, Dict, Any

from pot.core.vocabulary_aware_testing import (
    VocabularyAwareSequentialTester,
    VocabularyDecisionStatus,
    create_vocabulary_aware_tester
)
from pot.core.diff_decision import DiffDecisionConfig, TestingMode


def generate_synthetic_scores(
    n_samples: int,
    mean_diff: float,
    std_dev: float = 0.01,
    seed: int = 42
) -> List[float]:
    """Generate synthetic difference scores for testing."""
    np.random.seed(seed)
    return list(np.random.normal(mean_diff, std_dev, n_samples))


def test_vocabulary_extensions():
    """Test detection of vocabulary extensions."""
    
    print("=" * 70)
    print("Testing Vocabulary Extension Detection")
    print("=" * 70)
    
    # Test case: GPT-2 vs GPT-2 with extended vocabulary
    reference_vocab = 50257
    candidate_vocab = 50260  # 3 tokens added
    
    # Create tester
    config = DiffDecisionConfig(mode=TestingMode.QUICK_GATE)
    tester = VocabularyAwareSequentialTester(
        config=config,
        vocab_overlap_ratio=50257/50260,
        reference_vocab_size=reference_vocab,
        candidate_vocab_size=candidate_vocab,
        shared_token_count=50257
    )
    
    print(f"\nReference vocabulary: {reference_vocab:,} tokens")
    print(f"Candidate vocabulary: {candidate_vocab:,} tokens")
    print(f"Overlap ratio: {tester.vocab_overlap_ratio:.4%}")
    print(f"Extension detected: {tester.vocabulary_extension_detected}")
    
    # Generate samples with small differences (same model)
    samples = generate_synthetic_scores(30, mean_diff=0.002, std_dev=0.005)
    
    # Make decision
    result = tester.make_decision(samples)
    
    print(f"\n📊 Statistical Results:")
    print(f"  Mean difference: {result.mean_difference:.6f}")
    print(f"  CI: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
    print(f"  Base decision: {result.status}")
    print(f"  Vocabulary status: {result.vocabulary_status.value if result.vocabulary_status else 'N/A'}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Confidence adjustment: {result.confidence_adjustment_factor:.2f}x")
    
    if result.vocabulary_extension_detected:
        print(f"\n✅ Correctly identified as vocabulary extension")
        print(f"   Added tokens: {result.candidate_unique_tokens}")
    
    # Print interpretation
    if 'vocabulary_interpretation' in result.details:
        print(f"\n📝 Interpretation: {result.details['vocabulary_interpretation']}")


def test_vocabulary_reduction():
    """Test detection of vocabulary reduction/pruning."""
    
    print("\n" + "=" * 70)
    print("Testing Vocabulary Reduction Detection")
    print("=" * 70)
    
    # Test case: Model with pruned vocabulary
    reference_vocab = 50257
    candidate_vocab = 40000  # Significant reduction
    
    # Create tester
    config = DiffDecisionConfig(mode=TestingMode.AUDIT_GRADE)
    tester = VocabularyAwareSequentialTester(
        config=config,
        vocab_overlap_ratio=40000/50257,
        reference_vocab_size=reference_vocab,
        candidate_vocab_size=candidate_vocab,
        shared_token_count=40000
    )
    
    print(f"\nReference vocabulary: {reference_vocab:,} tokens")
    print(f"Candidate vocabulary: {candidate_vocab:,} tokens")
    print(f"Overlap ratio: {tester.vocab_overlap_ratio:.4%}")
    print(f"Reduction detected: {tester.vocabulary_reduction_detected}")
    
    # Generate samples with small differences
    samples = generate_synthetic_scores(50, mean_diff=0.008, std_dev=0.01)
    
    # Make decision
    result = tester.make_decision(samples)
    
    print(f"\n📊 Statistical Results:")
    print(f"  Mean difference: {result.mean_difference:.6f}")
    print(f"  CI: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
    print(f"  Base decision: {result.status}")
    print(f"  Vocabulary status: {result.vocabulary_status.value if result.vocabulary_status else 'N/A'}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Removed tokens: {result.reference_unique_tokens:,}")
    
    # Show adjusted thresholds
    print(f"\n🔧 Adjusted Thresholds:")
    print(f"  gamma (equivalence band): {tester.config.gamma:.4f}")
    print(f"  delta_star (min effect): {tester.config.delta_star:.4f}")
    print(f"  n_min: {tester.config.n_min}")


def test_cross_family_comparison():
    """Test comparison between different model families."""
    
    print("\n" + "=" * 70)
    print("Testing Cross-Family Model Comparison")
    print("=" * 70)
    
    # Test case: GPT-2 vs Mistral (different families)
    reference_vocab = 50257  # GPT-2
    candidate_vocab = 32768  # Mistral
    
    # Create tester with factory function
    tester = create_vocabulary_aware_tester(
        reference_vocab_size=reference_vocab,
        candidate_vocab_size=candidate_vocab,
        mode=TestingMode.AUDIT_GRADE
    )
    
    print(f"\nGPT-2 vocabulary: {reference_vocab:,} tokens")
    print(f"Mistral vocabulary: {candidate_vocab:,} tokens")
    print(f"Overlap ratio: {tester.vocab_overlap_ratio:.4%}")
    print(f"Confidence adjustment: {tester.confidence_adjustment_factor:.2f}x")
    
    # Generate samples with larger differences (different models)
    samples = generate_synthetic_scores(100, mean_diff=0.15, std_dev=0.03)
    
    # Make decision
    result = tester.make_decision(samples)
    
    print(f"\n📊 Statistical Results:")
    print(f"  Mean difference: {result.mean_difference:.6f}")
    print(f"  Effect size: {result.effect_size:.4f}")
    print(f"  Decision: {result.status}")
    print(f"  Vocabulary status: {result.vocabulary_status.value if result.vocabulary_status else 'N/A'}")
    print(f"  Adjusted confidence: {result.confidence:.2%}")
    
    if 'vocabulary_confirmation' in result.details:
        print(f"\n✅ {result.details['vocabulary_confirmation']}")


def test_confidence_adjustments():
    """Test confidence adjustment for various overlap ratios."""
    
    print("\n" + "=" * 70)
    print("Testing Confidence Adjustments")
    print("=" * 70)
    
    test_cases = [
        (50257, 50257, "Identical vocabularies"),
        (50257, 51200, "High overlap (Phi-2)"),
        (32768, 32000, "Good overlap (Mistral/Zephyr)"),
        (50257, 40000, "Moderate overlap"),
        (50257, 32768, "Low overlap (cross-family)"),
        (50257, 20000, "Very low overlap")
    ]
    
    config = DiffDecisionConfig(mode=TestingMode.QUICK_GATE)
    
    print("\n" + "-" * 60)
    print(f"{'Scenario':<25} {'Overlap':<10} {'Adjustment':<12} {'Impact':<15}")
    print("-" * 60)
    
    for ref_size, cand_size, description in test_cases:
        shared = min(ref_size, cand_size)
        overlap = shared / max(ref_size, cand_size)
        
        tester = VocabularyAwareSequentialTester(
            config=config,
            vocab_overlap_ratio=overlap,
            reference_vocab_size=ref_size,
            candidate_vocab_size=cand_size,
            shared_token_count=shared
        )
        
        adjustment = tester.confidence_adjustment_factor
        guidelines = tester._get_interpretation_guidelines()
        
        print(f"{description:<25} {overlap:>8.1%}  {adjustment:>10.2f}x  {guidelines['confidence_impact']:<15}")


def test_comprehensive_report():
    """Test comprehensive reporting with vocabulary analysis."""
    
    print("\n" + "=" * 70)
    print("Testing Comprehensive Reporting")
    print("=" * 70)
    
    # Create tester with moderate vocabulary difference
    config = DiffDecisionConfig(mode=TestingMode.AUDIT_GRADE)
    tester = VocabularyAwareSequentialTester(
        config=config,
        vocab_overlap_ratio=0.85,
        reference_vocab_size=50257,
        candidate_vocab_size=45000,
        shared_token_count=42000
    )
    
    # Generate samples
    samples = generate_synthetic_scores(75, mean_diff=0.05, std_dev=0.02)
    
    # Make decision
    result = tester.make_decision(samples)
    
    # Generate report
    report = tester.generate_report()
    
    print("\n📋 Full Report:")
    print("-" * 60)
    
    # Configuration section
    print("Configuration:")
    print(f"  Mode: {report['config']['mode']}")
    print(f"  Confidence level: {report['config']['confidence']:.1%}")
    print(f"  Samples: {report['sampling']['n_current']}/{report['sampling']['n_max']}")
    
    # Statistical results
    print(f"\nStatistical Analysis:")
    print(f"  Mean: {report['results']['mean']:.6f}")
    print(f"  CI: [{report['results']['ci_lower']:.6f}, {report['results']['ci_upper']:.6f}]")
    print(f"  Effect size: {report['results']['effect_size']:.4f}")
    print(f"  Decision: {report['decision']['status']}")
    
    # Vocabulary analysis
    vocab_analysis = report['vocabulary_analysis']
    print(f"\nVocabulary Analysis:")
    print(f"  Overlap ratio: {vocab_analysis['overlap_ratio']:.1%}")
    print(f"  Shared tokens: {vocab_analysis['shared_tokens']:,}")
    print(f"  Unique to reference: {vocab_analysis['unique_to_reference']:,}")
    print(f"  Unique to candidate: {vocab_analysis['unique_to_candidate']:,}")
    print(f"  Relationship: {vocab_analysis['relationship']}")
    print(f"  Confidence adjustment: {vocab_analysis['confidence_adjustment_factor']:.2f}x")
    
    # Interpretation
    interp = vocab_analysis['interpretation']
    print(f"\nInterpretation:")
    print(f"  Overlap level: {interp['overlap_level']}")
    print(f"  Impact on confidence: {interp['confidence_impact']}")
    print(f"  Recommendation: {interp['recommendation']}")
    
    # Get summary
    print("\n" + "=" * 60)
    print(tester.get_vocabulary_summary())


def test_decision_categories():
    """Test all decision categories with different scenarios."""
    
    print("\n" + "=" * 70)
    print("Testing Decision Categories")
    print("=" * 70)
    
    scenarios = [
        # (ref_vocab, cand_vocab, mean_diff, std, expected_status, description)
        (50257, 50257, 0.001, 0.003, "SAME", "Identical models"),
        (50257, 50260, 0.002, 0.005, "SAME_EXTENDED", "Model with vocabulary extension"),
        (50257, 45000, 0.003, 0.008, "SAME_REDUCED", "Model with vocabulary pruning"),
        (50257, 50300, 0.008, 0.01, "SAME_ADAPTED", "Model with minor vocab changes"),
        (50257, 32768, 0.20, 0.05, "DIFFERENT", "Different model families"),
    ]
    
    for ref_vocab, cand_vocab, mean, std, expected, description in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {description}")
        print(f"Vocabularies: {ref_vocab:,} vs {cand_vocab:,}")
        print("-" * 60)
        
        # Create tester
        config = DiffDecisionConfig(mode=TestingMode.QUICK_GATE)
        shared = min(ref_vocab, cand_vocab)
        overlap = shared / max(ref_vocab, cand_vocab)
        
        tester = VocabularyAwareSequentialTester(
            config=config,
            vocab_overlap_ratio=overlap,
            reference_vocab_size=ref_vocab,
            candidate_vocab_size=cand_vocab,
            shared_token_count=shared
        )
        
        # Generate samples
        samples = generate_synthetic_scores(50, mean, std)
        
        # Make decision
        result = tester.make_decision(samples)
        
        print(f"Expected: {expected}")
        print(f"Actual: {result.status}")
        
        if result.vocabulary_status:
            print(f"Vocabulary status: {result.vocabulary_status.value}")
        
        print(f"Confidence: {result.confidence:.2%}")
        
        # Check if matches expected
        status_match = expected in result.status or (
            expected == "SAME_EXTENDED" and result.vocabulary_status == VocabularyDecisionStatus.SAME_EXTENDED
        )
        
        if status_match:
            print("✅ Correct classification")
        else:
            print(f"❌ Mismatch (got {result.status})")
        
        # Show details if available
        for key in ['vocabulary_interpretation', 'vocabulary_warning', 'vocabulary_confirmation']:
            if key in result.details:
                print(f"\n📝 {result.details[key]}")


def save_test_results(results: Dict[str, Any], filename: str = "vocabulary_aware_test_results.json"):
    """Save test results to file."""
    output_dir = Path("experimental_results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / filename
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")


if __name__ == "__main__":
    print("🚀 Vocabulary-Aware Statistical Testing Framework")
    print("=" * 70)
    
    # Run all tests
    test_vocabulary_extensions()
    test_vocabulary_reduction()
    test_cross_family_comparison()
    test_confidence_adjustments()
    test_decision_categories()
    test_comprehensive_report()
    
    # Save summary results
    summary = {
        "test_name": "Vocabulary-Aware Statistical Testing",
        "timestamp": str(Path(__file__).stat().st_mtime),
        "tests_run": [
            "vocabulary_extensions",
            "vocabulary_reduction",
            "cross_family_comparison",
            "confidence_adjustments",
            "decision_categories",
            "comprehensive_report"
        ],
        "status": "completed"
    }
    
    save_test_results(summary)
    
    print("\n" + "=" * 70)
    print("✅ All Vocabulary-Aware Tests Complete")
    print("=" * 70)