"""
Comprehensive Vocabulary Analysis Module

This module provides deep vocabulary analysis for model comparison, distinguishing
between meaningful architectural changes and minor vocabulary adjustments.
"""

import re
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import Counter, defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class TokenCategory(Enum):
    """Categories for token classification"""
    SPECIAL = "special_tokens"          # [CLS], [SEP], <pad>, etc.
    CONTROL = "control_tokens"          # <eos>, <bos>, <unk>
    DOMAIN_SPECIFIC = "domain_specific" # Technical terms, names
    COMMON_WORDS = "common_words"       # Regular vocabulary
    SUBWORD_PIECES = "subword_pieces"   # Token fragments (##ing, Ġthe)
    NUMBERS = "numbers"                 # Numeric tokens
    PUNCTUATION = "punctuation"         # Punctuation marks
    MULTILINGUAL = "multilingual"       # Non-English tokens
    UNKNOWN = "unknown"                 # Unclassified


class ArchitecturalImpact(Enum):
    """Levels of architectural impact from vocabulary changes"""
    NEGLIGIBLE = "negligible"   # <1% parameter change
    MINOR = "minor"             # 1-5% parameter change
    MODERATE = "moderate"       # 5-10% parameter change
    MAJOR = "major"             # 10-20% parameter change
    SEVERE = "severe"           # >20% parameter change


@dataclass
class TokenAnalysis:
    """Analysis of a single token or token group"""
    token_id: int
    token_str: str
    category: TokenCategory
    frequency_rank: Optional[int] = None
    is_special: bool = False
    is_added: bool = False
    is_removed: bool = False
    linguistic_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VocabularyOverlapAnalysis:
    """Detailed vocabulary overlap analysis"""
    total_reference: int
    total_candidate: int
    shared_tokens: int
    unique_to_reference: int
    unique_to_candidate: int
    overlap_ratio: float
    jaccard_similarity: float
    
    # Token sets
    shared_token_ids: Set[int] = field(default_factory=set)
    reference_only_ids: Set[int] = field(default_factory=set)
    candidate_only_ids: Set[int] = field(default_factory=set)
    
    # Advanced metrics
    core_vocabulary_overlap: float = 0.0  # Overlap in first 10K tokens
    extended_vocabulary_overlap: float = 0.0  # Overlap beyond 10K
    special_token_overlap: float = 0.0  # Overlap in special tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'total_reference': self.total_reference,
            'total_candidate': self.total_candidate,
            'shared_tokens': self.shared_tokens,
            'unique_to_reference': self.unique_to_reference,
            'unique_to_candidate': self.unique_to_candidate,
            'overlap_ratio': self.overlap_ratio,
            'jaccard_similarity': self.jaccard_similarity,
            'core_vocabulary_overlap': self.core_vocabulary_overlap,
            'extended_vocabulary_overlap': self.extended_vocabulary_overlap,
            'special_token_overlap': self.special_token_overlap
        }


@dataclass
class TokenCategorization:
    """Categorization of vocabulary differences"""
    categories: Dict[TokenCategory, List[TokenAnalysis]] = field(default_factory=dict)
    category_counts: Dict[TokenCategory, int] = field(default_factory=dict)
    
    # Summary statistics
    total_analyzed: int = 0
    special_token_changes: int = 0
    domain_specific_changes: int = 0
    subword_changes: int = 0
    
    def add_token(self, analysis: TokenAnalysis):
        """Add a token analysis to the categorization"""
        if analysis.category not in self.categories:
            self.categories[analysis.category] = []
        self.categories[analysis.category].append(analysis)
        
        if analysis.category not in self.category_counts:
            self.category_counts[analysis.category] = 0
        self.category_counts[analysis.category] += 1
        
        self.total_analyzed += 1
        
        # Update specific counters
        if analysis.category == TokenCategory.SPECIAL:
            self.special_token_changes += 1
        elif analysis.category == TokenCategory.DOMAIN_SPECIFIC:
            self.domain_specific_changes += 1
        elif analysis.category == TokenCategory.SUBWORD_PIECES:
            self.subword_changes += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get categorization summary"""
        return {
            'total_analyzed': self.total_analyzed,
            'category_distribution': {
                cat.value: count for cat, count in self.category_counts.items()
            },
            'special_token_changes': self.special_token_changes,
            'domain_specific_changes': self.domain_specific_changes,
            'subword_changes': self.subword_changes
        }


@dataclass
class ArchitecturalImpactAssessment:
    """Assessment of architectural impact from vocabulary changes"""
    embedding_layer_change: ArchitecturalImpact
    output_layer_change: ArchitecturalImpact
    core_transformer_affected: bool
    parameter_difference_ratio: float
    functional_impact: str
    
    # Detailed metrics
    embedding_params_changed: int = 0
    output_params_changed: int = 0
    total_params_changed: int = 0
    
    # Impact analysis
    requires_retraining: bool = False
    backward_compatible: bool = True
    can_share_weights: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'embedding_layer_change': self.embedding_layer_change.value,
            'output_layer_change': self.output_layer_change.value,
            'core_transformer_affected': self.core_transformer_affected,
            'parameter_difference_ratio': self.parameter_difference_ratio,
            'functional_impact': self.functional_impact,
            'embedding_params_changed': self.embedding_params_changed,
            'output_params_changed': self.output_params_changed,
            'total_params_changed': self.total_params_changed,
            'requires_retraining': self.requires_retraining,
            'backward_compatible': self.backward_compatible,
            'can_share_weights': self.can_share_weights
        }


@dataclass
class VocabularyAnalysisReport:
    """Comprehensive vocabulary analysis report"""
    reference_size: int
    candidate_size: int
    overlap_analysis: VocabularyOverlapAnalysis
    token_categories: TokenCategorization
    architectural_impact: ArchitecturalImpactAssessment
    recommendations: List[str]
    
    # Additional analysis
    vocabulary_family: Optional[str] = None
    is_extension: bool = False
    is_reduction: bool = False
    is_adaptation: bool = False
    
    # Verification compatibility
    can_verify: bool = True
    verification_strategy: str = "standard"
    confidence_adjustment: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'reference_size': self.reference_size,
            'candidate_size': self.candidate_size,
            'overlap_analysis': self.overlap_analysis.to_dict(),
            'token_categories': self.token_categories.get_summary(),
            'architectural_impact': self.architectural_impact.to_dict(),
            'recommendations': self.recommendations,
            'vocabulary_family': self.vocabulary_family,
            'is_extension': self.is_extension,
            'is_reduction': self.is_reduction,
            'is_adaptation': self.is_adaptation,
            'can_verify': self.can_verify,
            'verification_strategy': self.verification_strategy,
            'confidence_adjustment': self.confidence_adjustment
        }
    
    def should_proceed_with_verification(self) -> bool:
        """Determine if verification should proceed"""
        return self.can_verify and self.overlap_analysis.overlap_ratio >= 0.5
    
    def get_adaptation_strategy(self) -> Dict[str, Any]:
        """Get recommended adaptation strategy"""
        return {
            'strategy': self.verification_strategy,
            'confidence_adjustment': self.confidence_adjustment,
            'focus_on_shared_tokens': self.overlap_analysis.overlap_ratio < 0.9,
            'use_frequency_weighting': self.token_categories.domain_specific_changes > 100,
            'increase_sample_size': self.overlap_analysis.overlap_ratio < 0.7
        }
    
    def get_incompatibility_reason(self) -> str:
        """Get reason for incompatibility if verification cannot proceed"""
        if self.overlap_analysis.overlap_ratio < 0.5:
            return f"Vocabulary overlap too low ({self.overlap_analysis.overlap_ratio:.1%})"
        elif self.architectural_impact.core_transformer_affected:
            return "Core transformer architecture affected by vocabulary changes"
        elif self.architectural_impact.parameter_difference_ratio > 0.2:
            return f"Parameter difference too large ({self.architectural_impact.parameter_difference_ratio:.1%})"
        else:
            return "Models incompatible for direct comparison"


class VocabularyAnalyzer:
    """
    Comprehensive vocabulary analyzer for model comparison.
    
    Provides deep insights about vocabulary differences, categorizes tokens,
    assesses architectural impact, and generates recommendations.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        analyze_token_content: bool = True,
        frequency_data: Optional[Dict[int, float]] = None
    ):
        """
        Initialize vocabulary analyzer.
        
        Args:
            embedding_dim: Dimension of embeddings (for parameter calculation)
            analyze_token_content: Whether to analyze token strings
            frequency_data: Optional token frequency data
        """
        self.embedding_dim = embedding_dim
        self.analyze_token_content = analyze_token_content
        self.frequency_data = frequency_data or {}
        
        # Pattern matchers for token categorization
        self.special_token_pattern = re.compile(r'^\[.*\]$|^<.*>$|^##.*')
        self.subword_pattern = re.compile(r'^##|^Ġ|^▁')
        self.number_pattern = re.compile(r'^-?\d+\.?\d*$')
        self.punctuation_pattern = re.compile(r'^[^\w\s]+$')
        
        # Known special tokens across frameworks
        self.known_special_tokens = {
            '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
            '<s>', '</s>', '<pad>', '<unk>', '<bos>', '<eos>',
            '<|endoftext|>', '<|startoftext|>', '<|pad|>'
        }
    
    def analyze_models(
        self,
        reference_model: Any,
        candidate_model: Any,
        reference_tokenizer: Optional[Any] = None,
        candidate_tokenizer: Optional[Any] = None
    ) -> VocabularyAnalysisReport:
        """
        Perform comprehensive vocabulary analysis between two models.
        
        Args:
            reference_model: Reference model
            candidate_model: Candidate model
            reference_tokenizer: Optional reference tokenizer
            candidate_tokenizer: Optional candidate tokenizer
        
        Returns:
            VocabularyAnalysisReport with comprehensive analysis
        """
        # Get vocabulary sizes
        ref_size = self.get_vocab_size(reference_model, reference_tokenizer)
        cand_size = self.get_vocab_size(candidate_model, candidate_tokenizer)
        
        logger.info(f"Analyzing vocabularies: {ref_size} (reference) vs {cand_size} (candidate)")
        
        # Analyze overlap
        overlap_analysis = self.analyze_overlap(ref_size, cand_size)
        
        # Get token mappings if tokenizers provided
        if reference_tokenizer and candidate_tokenizer:
            ref_vocab = self.get_vocabulary(reference_tokenizer)
            cand_vocab = self.get_vocabulary(candidate_tokenizer)
            
            # Categorize differences
            token_categories = self.categorize_differences(
                ref_vocab, cand_vocab, overlap_analysis
            )
        else:
            # Basic categorization without tokenizers
            token_categories = self.basic_categorize_differences(overlap_analysis)
        
        # Assess architectural impact
        architectural_impact = self.assess_impact(
            ref_size, cand_size, overlap_analysis, token_categories
        )
        
        # Determine relationship
        is_extension = cand_size > ref_size and overlap_analysis.overlap_ratio > 0.95
        is_reduction = cand_size < ref_size and overlap_analysis.overlap_ratio > 0.95
        is_adaptation = abs(cand_size - ref_size) < 1000 and overlap_analysis.overlap_ratio > 0.90
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            overlap_analysis, token_categories, architectural_impact,
            is_extension, is_reduction, is_adaptation
        )
        
        # Determine verification strategy
        verification_strategy, confidence_adjustment = self.determine_verification_strategy(
            overlap_analysis, architectural_impact
        )
        
        # Detect vocabulary family
        vocabulary_family = self.detect_vocabulary_family(ref_size, cand_size)
        
        return VocabularyAnalysisReport(
            reference_size=ref_size,
            candidate_size=cand_size,
            overlap_analysis=overlap_analysis,
            token_categories=token_categories,
            architectural_impact=architectural_impact,
            recommendations=recommendations,
            vocabulary_family=vocabulary_family,
            is_extension=is_extension,
            is_reduction=is_reduction,
            is_adaptation=is_adaptation,
            can_verify=architectural_impact.functional_impact != "incompatible",
            verification_strategy=verification_strategy,
            confidence_adjustment=confidence_adjustment
        )
    
    def get_vocab_size(self, model: Any, tokenizer: Optional[Any] = None) -> int:
        """Get vocabulary size from model or tokenizer"""
        if tokenizer and hasattr(tokenizer, 'vocab_size'):
            return tokenizer.vocab_size
        elif tokenizer and hasattr(tokenizer, 'get_vocab'):
            return len(tokenizer.get_vocab())
        elif hasattr(model, 'vocab_size'):
            return model.vocab_size
        elif hasattr(model, 'get_vocab_size'):
            return model.get_vocab_size()
        elif hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
            return model.config.vocab_size
        else:
            logger.warning("Could not determine vocabulary size, using default")
            return 50257  # Default GPT-2 size
    
    def get_vocabulary(self, tokenizer: Any) -> Dict[str, int]:
        """Get vocabulary dictionary from tokenizer"""
        if hasattr(tokenizer, 'get_vocab'):
            return tokenizer.get_vocab()
        elif hasattr(tokenizer, 'vocab'):
            return tokenizer.vocab
        else:
            return {}
    
    def analyze_overlap(self, ref_size: int, cand_size: int) -> VocabularyOverlapAnalysis:
        """Analyze vocabulary overlap between models"""
        shared = min(ref_size, cand_size)
        unique_ref = max(0, ref_size - shared)
        unique_cand = max(0, cand_size - shared)
        
        # Calculate metrics
        total = max(ref_size, cand_size)
        overlap_ratio = shared / total if total > 0 else 0
        
        # Jaccard similarity
        union = ref_size + cand_size - shared
        jaccard = shared / union if union > 0 else 0
        
        # Core vs extended vocabulary overlap
        core_size = 10000
        core_shared = min(shared, core_size)
        core_total = min(max(ref_size, cand_size), core_size)
        core_overlap = core_shared / core_total if core_total > 0 else 0
        
        extended_shared = max(0, shared - core_size)
        extended_total = max(0, max(ref_size, cand_size) - core_size)
        extended_overlap = extended_shared / extended_total if extended_total > 0 else 0
        
        return VocabularyOverlapAnalysis(
            total_reference=ref_size,
            total_candidate=cand_size,
            shared_tokens=shared,
            unique_to_reference=unique_ref,
            unique_to_candidate=unique_cand,
            overlap_ratio=overlap_ratio,
            jaccard_similarity=jaccard,
            core_vocabulary_overlap=core_overlap,
            extended_vocabulary_overlap=extended_overlap,
            special_token_overlap=1.0  # Assume special tokens are shared
        )
    
    def categorize_token(self, token_str: str, token_id: int) -> TokenCategory:
        """Categorize a single token"""
        if not token_str:
            return TokenCategory.UNKNOWN
        
        # Check for special tokens
        if token_str in self.known_special_tokens or self.special_token_pattern.match(token_str):
            return TokenCategory.SPECIAL
        
        # Check for control tokens
        if token_str in ['<s>', '</s>', '<bos>', '<eos>', '<pad>', '<unk>']:
            return TokenCategory.CONTROL
        
        # Check for subword pieces
        if self.subword_pattern.match(token_str):
            return TokenCategory.SUBWORD_PIECES
        
        # Check for numbers
        if self.number_pattern.match(token_str):
            return TokenCategory.NUMBERS
        
        # Check for punctuation
        if self.punctuation_pattern.match(token_str):
            return TokenCategory.PUNCTUATION
        
        # Check for non-ASCII (potentially multilingual)
        if not all(ord(c) < 128 for c in token_str):
            return TokenCategory.MULTILINGUAL
        
        # Check token length for classification
        if len(token_str) > 15:
            return TokenCategory.DOMAIN_SPECIFIC
        
        # Default to common words
        return TokenCategory.COMMON_WORDS
    
    def categorize_differences(
        self,
        ref_vocab: Dict[str, int],
        cand_vocab: Dict[str, int],
        overlap_analysis: VocabularyOverlapAnalysis
    ) -> TokenCategorization:
        """Categorize vocabulary differences by token type"""
        categorization = TokenCategorization()
        
        # Find added tokens (in candidate but not reference)
        added_tokens = set(cand_vocab.keys()) - set(ref_vocab.keys())
        for token_str in added_tokens:
            token_id = cand_vocab[token_str]
            category = self.categorize_token(token_str, token_id)
            
            analysis = TokenAnalysis(
                token_id=token_id,
                token_str=token_str,
                category=category,
                is_added=True,
                frequency_rank=self.get_frequency_rank(token_id)
            )
            categorization.add_token(analysis)
        
        # Find removed tokens (in reference but not candidate)
        removed_tokens = set(ref_vocab.keys()) - set(cand_vocab.keys())
        for token_str in removed_tokens:
            token_id = ref_vocab[token_str]
            category = self.categorize_token(token_str, token_id)
            
            analysis = TokenAnalysis(
                token_id=token_id,
                token_str=token_str,
                category=category,
                is_removed=True,
                frequency_rank=self.get_frequency_rank(token_id)
            )
            categorization.add_token(analysis)
        
        return categorization
    
    def basic_categorize_differences(
        self,
        overlap_analysis: VocabularyOverlapAnalysis
    ) -> TokenCategorization:
        """Basic categorization without tokenizer access"""
        categorization = TokenCategorization()
        
        # Estimate category distribution based on vocabulary sizes
        total_diff = overlap_analysis.unique_to_reference + overlap_analysis.unique_to_candidate
        
        if total_diff > 0:
            # Typical distribution estimates
            categorization.special_token_changes = min(10, total_diff)
            categorization.domain_specific_changes = int(total_diff * 0.2)
            categorization.subword_changes = int(total_diff * 0.5)
            categorization.total_analyzed = total_diff
        
        return categorization
    
    def assess_impact(
        self,
        ref_size: int,
        cand_size: int,
        overlap_analysis: VocabularyOverlapAnalysis,
        token_categories: TokenCategorization
    ) -> ArchitecturalImpactAssessment:
        """Assess architectural impact of vocabulary changes"""
        # Calculate parameter changes
        vocab_diff = abs(cand_size - ref_size)
        embedding_params_changed = vocab_diff * self.embedding_dim
        output_params_changed = vocab_diff * self.embedding_dim  # Assuming tied weights
        total_params_changed = embedding_params_changed + output_params_changed
        
        # Estimate total model parameters (rough estimate)
        # Assuming BERT-base size as baseline
        estimated_total_params = 110_000_000
        param_diff_ratio = total_params_changed / estimated_total_params
        
        # Determine impact levels
        if param_diff_ratio < 0.001:
            embedding_impact = ArchitecturalImpact.NEGLIGIBLE
            output_impact = ArchitecturalImpact.NEGLIGIBLE
        elif param_diff_ratio < 0.01:
            embedding_impact = ArchitecturalImpact.MINOR
            output_impact = ArchitecturalImpact.MINOR
        elif param_diff_ratio < 0.05:
            embedding_impact = ArchitecturalImpact.MODERATE
            output_impact = ArchitecturalImpact.MODERATE
        elif param_diff_ratio < 0.1:
            embedding_impact = ArchitecturalImpact.MAJOR
            output_impact = ArchitecturalImpact.MAJOR
        else:
            embedding_impact = ArchitecturalImpact.SEVERE
            output_impact = ArchitecturalImpact.SEVERE
        
        # Core transformer affected only if special tokens changed significantly
        core_affected = token_categories.special_token_changes > 5
        
        # Determine functional impact
        if overlap_analysis.overlap_ratio > 0.95:
            functional_impact = "negligible"
        elif overlap_analysis.overlap_ratio > 0.85:
            functional_impact = "minor"
        elif overlap_analysis.overlap_ratio > 0.70:
            functional_impact = "moderate"
        elif overlap_analysis.overlap_ratio > 0.50:
            functional_impact = "significant"
        else:
            functional_impact = "incompatible"
        
        # Compatibility assessment
        requires_retraining = param_diff_ratio > 0.01 or core_affected
        backward_compatible = overlap_analysis.overlap_ratio > 0.95 and not core_affected
        can_share_weights = overlap_analysis.overlap_ratio > 0.90
        
        return ArchitecturalImpactAssessment(
            embedding_layer_change=embedding_impact,
            output_layer_change=output_impact,
            core_transformer_affected=core_affected,
            parameter_difference_ratio=param_diff_ratio,
            functional_impact=functional_impact,
            embedding_params_changed=embedding_params_changed,
            output_params_changed=output_params_changed,
            total_params_changed=total_params_changed,
            requires_retraining=requires_retraining,
            backward_compatible=backward_compatible,
            can_share_weights=can_share_weights
        )
    
    def generate_recommendations(
        self,
        overlap_analysis: VocabularyOverlapAnalysis,
        token_categories: TokenCategorization,
        architectural_impact: ArchitecturalImpactAssessment,
        is_extension: bool,
        is_reduction: bool,
        is_adaptation: bool
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Overlap-based recommendations
        if overlap_analysis.overlap_ratio > 0.95:
            if is_extension:
                recommendations.append(
                    "Models appear to be same base with vocabulary extension. "
                    "Consider using adaptive verification with focus on shared tokens."
                )
            elif is_reduction:
                recommendations.append(
                    "Models appear to be same base with vocabulary pruning. "
                    "Verify using only the reduced vocabulary space."
                )
            else:
                recommendations.append(
                    "High vocabulary overlap suggests same model family. "
                    "Standard verification should be reliable."
                )
        elif overlap_analysis.overlap_ratio > 0.80:
            recommendations.append(
                "Moderate vocabulary overlap. Use adaptive challenge generation "
                "and increase sample size for reliable verification."
            )
        elif overlap_analysis.overlap_ratio > 0.60:
            recommendations.append(
                "Limited vocabulary overlap suggests different model families. "
                "Consider semantic verification methods instead of token-level comparison."
            )
        else:
            recommendations.append(
                "Very low vocabulary overlap. Models likely use different tokenizers. "
                "Direct comparison may not be meaningful."
            )
        
        # Token category recommendations
        if token_categories.special_token_changes > 0:
            recommendations.append(
                f"Detected {token_categories.special_token_changes} special token changes. "
                "This may affect model behavior significantly."
            )
        
        if token_categories.domain_specific_changes > 100:
            recommendations.append(
                "Large number of domain-specific token changes detected. "
                "Models may be specialized for different domains."
            )
        
        # Architecture recommendations
        if architectural_impact.requires_retraining:
            recommendations.append(
                "Vocabulary changes require model retraining for optimal performance."
            )
        
        if not architectural_impact.backward_compatible:
            recommendations.append(
                "Models are not backward compatible due to vocabulary changes."
            )
        
        if architectural_impact.can_share_weights:
            recommendations.append(
                "Core transformer weights can potentially be shared between models."
            )
        
        # Verification strategy recommendations
        if overlap_analysis.core_vocabulary_overlap > 0.95:
            recommendations.append(
                "Core vocabulary (first 10K tokens) has high overlap. "
                "Focus verification on core vocabulary for best results."
            )
        
        if overlap_analysis.extended_vocabulary_overlap < 0.50:
            recommendations.append(
                "Extended vocabulary has low overlap. "
                "Avoid using rare tokens in verification challenges."
            )
        
        return recommendations
    
    def determine_verification_strategy(
        self,
        overlap_analysis: VocabularyOverlapAnalysis,
        architectural_impact: ArchitecturalImpactAssessment
    ) -> Tuple[str, float]:
        """Determine verification strategy and confidence adjustment"""
        if overlap_analysis.overlap_ratio > 0.95:
            return "standard", 1.0
        elif overlap_analysis.overlap_ratio > 0.85:
            return "adaptive_minor", 0.95
        elif overlap_analysis.overlap_ratio > 0.70:
            return "adaptive_moderate", 0.85
        elif overlap_analysis.overlap_ratio > 0.60:
            return "adaptive_major", 0.70
        else:
            return "incompatible", 0.50
    
    def detect_vocabulary_family(self, ref_size: int, cand_size: int) -> Optional[str]:
        """Detect vocabulary family based on sizes"""
        # Common vocabulary sizes for different model families
        families = {
            "gpt": [50257, 50258, 50259, 50260],
            "bert": [30522, 30523, 28996],
            "llama": [32000, 32001, 32002],
            "mistral": [32768, 32000],
            "t5": [32100, 32128],
            "roberta": [50265],
            "phi": [51200, 51201]
        }
        
        for family, sizes in families.items():
            if ref_size in sizes or cand_size in sizes:
                return family
        
        return None
    
    def get_frequency_rank(self, token_id: int) -> Optional[int]:
        """Get frequency rank for a token"""
        if self.frequency_data and token_id in self.frequency_data:
            # Sort tokens by frequency and find rank
            sorted_tokens = sorted(
                self.frequency_data.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for rank, (tid, _) in enumerate(sorted_tokens):
                if tid == token_id:
                    return rank + 1
        return None
    
    def generate_diff_report(
        self,
        report: VocabularyAnalysisReport,
        output_format: str = "text"
    ) -> str:
        """Generate a detailed diff report"""
        if output_format == "json":
            return json.dumps(report.to_dict(), indent=2)
        
        # Generate text report
        lines = [
            "=" * 70,
            "VOCABULARY ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Reference Model: {report.reference_size:,} tokens",
            f"Candidate Model: {report.candidate_size:,} tokens",
            f"Size Difference: {abs(report.candidate_size - report.reference_size):,} tokens",
            "",
            "OVERLAP ANALYSIS",
            "-" * 40,
            f"Shared Tokens: {report.overlap_analysis.shared_tokens:,}",
            f"Overlap Ratio: {report.overlap_analysis.overlap_ratio:.1%}",
            f"Jaccard Similarity: {report.overlap_analysis.jaccard_similarity:.1%}",
            f"Core Vocabulary Overlap: {report.overlap_analysis.core_vocabulary_overlap:.1%}",
            f"Extended Vocabulary Overlap: {report.overlap_analysis.extended_vocabulary_overlap:.1%}",
            "",
            "TOKEN CATEGORIZATION",
            "-" * 40
        ]
        
        # Add category breakdown
        summary = report.token_categories.get_summary()
        if summary['category_distribution']:
            for category, count in summary['category_distribution'].items():
                lines.append(f"  {category}: {count}")
        
        lines.extend([
            "",
            "ARCHITECTURAL IMPACT",
            "-" * 40,
            f"Embedding Layer Change: {report.architectural_impact.embedding_layer_change.value}",
            f"Output Layer Change: {report.architectural_impact.output_layer_change.value}",
            f"Core Transformer Affected: {report.architectural_impact.core_transformer_affected}",
            f"Parameter Difference: {report.architectural_impact.parameter_difference_ratio:.2%}",
            f"Functional Impact: {report.architectural_impact.functional_impact}",
            ""
        ])
        
        # Add relationship
        if report.is_extension:
            lines.append("RELATIONSHIP: Vocabulary Extension")
        elif report.is_reduction:
            lines.append("RELATIONSHIP: Vocabulary Reduction")
        elif report.is_adaptation:
            lines.append("RELATIONSHIP: Vocabulary Adaptation")
        else:
            lines.append("RELATIONSHIP: Different Vocabularies")
        
        lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 40
        ])
        
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")
        
        lines.extend([
            "",
            "VERIFICATION COMPATIBILITY",
            "-" * 40,
            f"Can Verify: {report.can_verify}",
            f"Strategy: {report.verification_strategy}",
            f"Confidence Adjustment: {report.confidence_adjustment:.2f}x",
            "",
            "=" * 70
        ])
        
        return "\n".join(lines)