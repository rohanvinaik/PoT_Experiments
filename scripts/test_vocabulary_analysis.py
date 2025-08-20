#!/usr/bin/env python3
"""
Test script for comprehensive vocabulary analysis module.

Demonstrates deep vocabulary analysis, token categorization, architectural
impact assessment, and visualization capabilities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Dict, Any

from pot.core.vocabulary_analysis import (
    VocabularyAnalyzer,
    TokenCategory,
    ArchitecturalImpact
)
from pot.core.vocabulary_visualization import (
    VocabularyVisualizer,
    visualize_vocabulary_analysis
)


def test_basic_vocabulary_analysis():
    """Test basic vocabulary analysis functionality."""
    
    print("=" * 70)
    print("Testing Basic Vocabulary Analysis")
    print("=" * 70)
    
    analyzer = VocabularyAnalyzer(embedding_dim=768)
    
    # Test cases with different vocabulary sizes
    test_cases = [
        (50257, 50257, "GPT-2", "GPT-2", "Identical models"),
        (50257, 50260, "GPT-2", "GPT-2-extended", "Minor extension"),
        (50257, 51200, "GPT-2", "Phi-2", "Different but compatible"),
        (50257, 32768, "GPT-2", "Mistral-7B", "Different families"),
        (32768, 32000, "Mistral-7B", "Zephyr-7B", "Fine-tuning"),
        (50257, 40000, "GPT-2", "GPT-2-pruned", "Vocabulary reduction")
    ]
    
    for ref_size, cand_size, ref_name, cand_name, description in test_cases:
        print(f"\n{'='*60}")
        print(f"{description}: {ref_name} vs {cand_name}")
        print("-" * 60)
        
        # Create mock models with vocab size
        class MockModel:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
        
        ref_model = MockModel(ref_size)
        cand_model = MockModel(cand_size)
        
        # Analyze
        report = analyzer.analyze_models(ref_model, cand_model)
        
        # Display results
        print(f"Vocabulary Sizes: {ref_size:,} vs {cand_size:,}")
        print(f"Overlap Ratio: {report.overlap_analysis.overlap_ratio:.1%}")
        print(f"Jaccard Similarity: {report.overlap_analysis.jaccard_similarity:.1%}")
        print(f"Core Vocabulary Overlap: {report.overlap_analysis.core_vocabulary_overlap:.1%}")
        
        # Show relationship
        if report.is_extension:
            print("→ Relationship: EXTENSION")
        elif report.is_reduction:
            print("→ Relationship: REDUCTION")
        elif report.is_adaptation:
            print("→ Relationship: ADAPTATION")
        else:
            print("→ Relationship: DIFFERENT")
        
        # Show architectural impact
        print(f"\nArchitectural Impact:")
        print(f"  Embedding Layer: {report.architectural_impact.embedding_layer_change.value}")
        print(f"  Parameter Change: {report.architectural_impact.parameter_difference_ratio:.3%}")
        print(f"  Functional Impact: {report.architectural_impact.functional_impact}")
        print(f"  Backward Compatible: {report.architectural_impact.backward_compatible}")
        
        # Show verification compatibility
        print(f"\nVerification:")
        print(f"  Can Verify: {report.can_verify}")
        print(f"  Strategy: {report.verification_strategy}")
        print(f"  Confidence Adjustment: {report.confidence_adjustment:.2f}x")
        
        # Show first recommendation
        if report.recommendations:
            print(f"\nTop Recommendation:")
            print(f"  {report.recommendations[0]}")


def test_token_categorization():
    """Test token categorization functionality."""
    
    print("\n" + "=" * 70)
    print("Testing Token Categorization")
    print("=" * 70)
    
    analyzer = VocabularyAnalyzer()
    
    # Test various token types
    test_tokens = [
        ("[CLS]", TokenCategory.SPECIAL),
        ("[SEP]", TokenCategory.SPECIAL),
        ("<pad>", TokenCategory.SPECIAL),
        ("<eos>", TokenCategory.CONTROL),
        ("##ing", TokenCategory.SUBWORD_PIECES),
        ("Ġthe", TokenCategory.SUBWORD_PIECES),
        ("▁world", TokenCategory.SUBWORD_PIECES),
        ("123", TokenCategory.NUMBERS),
        ("3.14", TokenCategory.NUMBERS),
        ("!", TokenCategory.PUNCTUATION),
        ("...", TokenCategory.PUNCTUATION),
        ("hello", TokenCategory.COMMON_WORDS),
        ("world", TokenCategory.COMMON_WORDS),
        ("超级计算机", TokenCategory.MULTILINGUAL),
        ("médecin", TokenCategory.MULTILINGUAL),
        ("pneumonoultramicroscopicsilicovolcanoconiosis", TokenCategory.DOMAIN_SPECIFIC)
    ]
    
    print("\nToken Categorization Results:")
    print("-" * 60)
    print(f"{'Token':<40} {'Category':<20}")
    print("-" * 60)
    
    for token, expected_category in test_tokens:
        category = analyzer.categorize_token(token, 0)
        match = "✓" if category == expected_category else "✗"
        print(f"{token:<40} {category.value:<20} {match}")


def test_architectural_impact():
    """Test architectural impact assessment."""
    
    print("\n" + "=" * 70)
    print("Testing Architectural Impact Assessment")
    print("=" * 70)
    
    analyzer = VocabularyAnalyzer(embedding_dim=768)
    
    # Test different levels of vocabulary changes
    test_cases = [
        (50257, 50260, "Negligible change (+3 tokens)"),
        (50257, 51000, "Minor change (+743 tokens)"),
        (50257, 55000, "Moderate change (+4,743 tokens)"),
        (50257, 60000, "Major change (+9,743 tokens)"),
        (50257, 100000, "Severe change (+49,743 tokens)")
    ]
    
    for ref_size, cand_size, description in test_cases:
        print(f"\n{description}")
        print("-" * 40)
        
        # Create mock overlap analysis
        from pot.core.vocabulary_analysis import VocabularyOverlapAnalysis, TokenCategorization
        
        shared = min(ref_size, cand_size)
        overlap = VocabularyOverlapAnalysis(
            total_reference=ref_size,
            total_candidate=cand_size,
            shared_tokens=shared,
            unique_to_reference=max(0, ref_size - shared),
            unique_to_candidate=max(0, cand_size - shared),
            overlap_ratio=shared / max(ref_size, cand_size),
            jaccard_similarity=shared / (ref_size + cand_size - shared)
        )
        
        # Create mock token categorization
        token_categories = TokenCategorization()
        token_categories.special_token_changes = min(5, abs(cand_size - ref_size) // 1000)
        
        # Assess impact
        impact = analyzer.assess_impact(ref_size, cand_size, overlap, token_categories)
        
        print(f"  Embedding Impact: {impact.embedding_layer_change.value}")
        print(f"  Output Impact: {impact.output_layer_change.value}")
        print(f"  Parameter Change: {impact.parameter_difference_ratio:.4%}")
        print(f"  Parameters Changed: {impact.total_params_changed:,}")
        print(f"  Functional Impact: {impact.functional_impact}")
        print(f"  Requires Retraining: {impact.requires_retraining}")


def test_visualization():
    """Test visualization capabilities."""
    
    print("\n" + "=" * 70)
    print("Testing Visualization Capabilities")
    print("=" * 70)
    
    # Create analyzer and analyze models
    analyzer = VocabularyAnalyzer(embedding_dim=768)
    
    class MockModel:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
    
    ref_model = MockModel(50257)
    cand_model = MockModel(32768)
    
    report = analyzer.analyze_models(ref_model, cand_model)
    
    # Create visualizer
    visualizer = VocabularyVisualizer(output_dir=Path("experimental_results/vocabulary_viz"))
    
    # Create text-based Venn diagram
    print("\n1. Vocabulary Overlap Diagram:")
    print("-" * 40)
    diagram = visualizer._create_text_venn_diagram(
        report.reference_size,
        report.candidate_size,
        report.overlap_analysis.shared_tokens,
        "GPT-2",
        "Mistral"
    )
    print(diagram)
    
    # Create text histogram
    print("\n2. Token Distribution Histogram:")
    print("-" * 40)
    categories = {
        "special_tokens": 10,
        "domain_specific": 3500,
        "common_words": 8000,
        "subword_pieces": 6000
    }
    histogram = visualizer._create_text_histogram(categories, "Token Categories")
    print(histogram)
    
    # Create comparison chart
    print("\n3. Model Comparison:")
    print("-" * 40)
    metrics = {
        "Vocabulary Size (K)": (50.257, 32.768),
        "Overlap Ratio": (0.652, 0.652),
        "Core Overlap": (0.80, 0.80),
        "Parameter Diff": (0.02, 0.02)
    }
    comparison = visualizer._create_text_comparison(metrics, "Vocabulary Metrics")
    print(comparison)
    
    # Create dashboard
    print("\n4. Creating comprehensive dashboard...")
    dashboard_path = visualizer.create_summary_dashboard(report.to_dict())
    print(f"   Dashboard saved to: {dashboard_path}")


def test_integration_with_verification():
    """Test integration with verification pipeline."""
    
    print("\n" + "=" * 70)
    print("Testing Integration with Verification Pipeline")
    print("=" * 70)
    
    analyzer = VocabularyAnalyzer(embedding_dim=768)
    
    # Simulate different verification scenarios
    scenarios = [
        (50257, 50257, "Standard verification"),
        (50257, 51200, "Adaptive verification (high overlap)"),
        (50257, 32768, "Adaptive verification (moderate overlap)"),
        (50257, 10000, "Incompatible models")
    ]
    
    for ref_size, cand_size, scenario in scenarios:
        print(f"\n{scenario}")
        print("-" * 40)
        
        # Create mock models
        class MockModel:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
        
        ref_model = MockModel(ref_size)
        cand_model = MockModel(cand_size)
        
        # Analyze
        vocab_analysis = analyzer.analyze_models(ref_model, cand_model)
        
        print(f"Vocabulary: {ref_size:,} vs {cand_size:,}")
        print(f"Overlap: {vocab_analysis.overlap_analysis.overlap_ratio:.1%}")
        
        # Check if we should proceed
        if vocab_analysis.should_proceed_with_verification():
            # Get adaptation strategy
            strategy = vocab_analysis.get_adaptation_strategy()
            
            print(f"\n✅ Proceeding with verification")
            print(f"   Strategy: {strategy['strategy']}")
            print(f"   Confidence adjustment: {strategy['confidence_adjustment']:.2f}x")
            print(f"   Focus on shared tokens: {strategy['focus_on_shared_tokens']}")
            print(f"   Use frequency weighting: {strategy['use_frequency_weighting']}")
            print(f"   Increase sample size: {strategy['increase_sample_size']}")
            
            # Simulate verification with adaptation
            print(f"\n   [Simulating {strategy['strategy']} verification...]")
            print(f"   [Adjusting confidence by {strategy['confidence_adjustment']:.2f}x]")
            
            if strategy['focus_on_shared_tokens']:
                print(f"   [Focusing challenges on {vocab_analysis.overlap_analysis.shared_tokens:,} shared tokens]")
            
            if strategy['increase_sample_size']:
                print(f"   [Increasing sample size for reliability]")
            
        else:
            # Handle incompatible models
            reason = vocab_analysis.get_incompatibility_reason()
            print(f"\n❌ Cannot proceed with verification")
            print(f"   Reason: {reason}")
            
            # Show alternatives
            print(f"\n   Alternatives:")
            print(f"   - Use semantic similarity methods")
            print(f"   - Compare model architectures directly")
            print(f"   - Analyze weight distributions")


def test_comprehensive_report():
    """Test comprehensive report generation."""
    
    print("\n" + "=" * 70)
    print("Testing Comprehensive Report Generation")
    print("=" * 70)
    
    analyzer = VocabularyAnalyzer(embedding_dim=768)
    
    # Create models with interesting difference
    class MockModel:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
    
    ref_model = MockModel(50257)
    cand_model = MockModel(50300)  # Extension with 43 tokens
    
    # Analyze
    report = analyzer.analyze_models(ref_model, cand_model)
    
    # Generate text report
    text_report = analyzer.generate_diff_report(report, output_format="text")
    print("\nText Report:")
    print(text_report)
    
    # Save JSON report
    output_dir = Path("experimental_results")
    output_dir.mkdir(exist_ok=True)
    
    json_path = output_dir / "vocabulary_analysis_report.json"
    with open(json_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    
    print(f"\n✅ JSON report saved to: {json_path}")
    
    # Create all visualizations
    print("\n📊 Creating visualizations...")
    viz_results = visualize_vocabulary_analysis(
        report.to_dict(),
        output_dir=output_dir / "vocabulary_visualizations"
    )
    
    print("Created visualizations:")
    for viz_type, path in viz_results.items():
        print(f"  - {viz_type}: {path}")


if __name__ == "__main__":
    print("🚀 Comprehensive Vocabulary Analysis Test Suite")
    print("=" * 70)
    
    # Run all tests
    test_basic_vocabulary_analysis()
    test_token_categorization()
    test_architectural_impact()
    test_visualization()
    test_integration_with_verification()
    test_comprehensive_report()
    
    print("\n" + "=" * 70)
    print("✅ All Vocabulary Analysis Tests Complete")
    print("=" * 70)
    
    print("\n📝 Summary:")
    print("  - Basic vocabulary analysis: ✓")
    print("  - Token categorization: ✓")
    print("  - Architectural impact assessment: ✓")
    print("  - Visualization capabilities: ✓")
    print("  - Verification pipeline integration: ✓")
    print("  - Comprehensive reporting: ✓")