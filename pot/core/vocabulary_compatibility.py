"""
Vocabulary Compatibility Analysis System

This module provides intelligent handling of models with different vocabulary sizes,
recognizing that vocabulary differences often represent minor fine-tuning rather than
fundamental architecture changes.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class VocabularyMismatchBehavior(Enum):
    """Defines how to handle vocabulary mismatches"""
    WARN = "warn"       # Log warning and continue with overlapping tokens
    ADAPT = "adapt"     # Adapt verification strategy for mismatch
    FAIL = "fail"       # Fail verification (legacy behavior)


@dataclass
class VocabularyCompatibilityReport:
    """Report on vocabulary compatibility between two models"""
    vocab_size_a: int
    vocab_size_b: int
    shared_tokens: int
    overlap_ratio: float
    size_difference: int
    size_difference_ratio: float
    is_subset: bool  # True if one vocab is subset of other
    is_extension: bool  # True if larger vocab appears to be extension
    recommended_strategy: str
    warnings: list
    metadata: Dict[str, Any]
    
    def __str__(self) -> str:
        return (
            f"Vocabulary Compatibility Report:\n"
            f"  Model A vocab size: {self.vocab_size_a:,}\n"
            f"  Model B vocab size: {self.vocab_size_b:,}\n"
            f"  Shared tokens: {self.shared_tokens:,}\n"
            f"  Overlap ratio: {self.overlap_ratio:.2%}\n"
            f"  Size difference: {self.size_difference:,} tokens ({self.size_difference_ratio:.2%})\n"
            f"  Is subset: {self.is_subset}\n"
            f"  Is extension: {self.is_extension}\n"
            f"  Recommended strategy: {self.recommended_strategy}\n"
            f"  Warnings: {len(self.warnings)} issue(s)"
        )
    
    @property
    def is_compatible(self) -> bool:
        """Check if vocabularies are compatible for verification"""
        return self.overlap_ratio >= 0.95
    
    @property
    def is_highly_compatible(self) -> bool:
        """Check if vocabularies are highly compatible (>99% overlap)"""
        return self.overlap_ratio >= 0.99


class VocabularyCompatibilityAnalyzer:
    """
    Analyzes vocabulary compatibility between models with different sizes.
    
    This analyzer recognizes that models with slightly different vocabulary sizes
    often share 99%+ of their architecture, with differences typically due to:
    - Domain-specific token additions
    - Special tokens for fine-tuning
    - Tokenizer updates/improvements
    - Multi-lingual extensions
    """
    
    def __init__(
        self,
        min_overlap_ratio: float = 0.95,
        mismatch_behavior: VocabularyMismatchBehavior = VocabularyMismatchBehavior.WARN,
        allow_extended_vocabularies: bool = True,
        extension_threshold: int = 1000
    ):
        """
        Initialize the analyzer.
        
        Args:
            min_overlap_ratio: Minimum overlap ratio to consider compatible (default 0.95)
            mismatch_behavior: How to handle mismatches (warn/adapt/fail)
            allow_extended_vocabularies: Whether to allow vocabulary extensions
            extension_threshold: Max tokens difference to consider as extension (default 1000)
        """
        self.min_overlap_ratio = min_overlap_ratio
        self.mismatch_behavior = mismatch_behavior
        self.allow_extended_vocabularies = allow_extended_vocabularies
        self.extension_threshold = extension_threshold
    
    def analyze_vocabulary_overlap(
        self,
        vocab_size_a: int,
        vocab_size_b: int,
        model_name_a: Optional[str] = None,
        model_name_b: Optional[str] = None
    ) -> VocabularyCompatibilityReport:
        """
        Analyze vocabulary overlap between two models.
        
        Args:
            vocab_size_a: Vocabulary size of first model
            vocab_size_b: Vocabulary size of second model
            model_name_a: Optional name of first model
            model_name_b: Optional name of second model
            
        Returns:
            Detailed compatibility report
        """
        shared_tokens = min(vocab_size_a, vocab_size_b)
        max_vocab = max(vocab_size_a, vocab_size_b)
        size_difference = abs(vocab_size_a - vocab_size_b)
        
        # Calculate overlap ratio (shared tokens / larger vocabulary)
        overlap_ratio = shared_tokens / max_vocab if max_vocab > 0 else 1.0
        
        # Calculate size difference ratio
        size_difference_ratio = size_difference / max_vocab if max_vocab > 0 else 0.0
        
        # Check if one vocabulary is a subset/extension of the other
        is_subset = (vocab_size_a == shared_tokens) or (vocab_size_b == shared_tokens)
        is_extension = is_subset and size_difference <= self.extension_threshold
        
        # Determine recommended strategy
        strategy = self._determine_strategy(
            overlap_ratio, size_difference, is_extension
        )
        
        # Generate warnings
        warnings = self._generate_warnings(
            vocab_size_a, vocab_size_b, overlap_ratio, size_difference
        )
        
        # Collect metadata
        metadata = {
            "model_name_a": model_name_a,
            "model_name_b": model_name_b,
            "min_overlap_for_compatibility": self.min_overlap_ratio,
            "behavior_on_mismatch": self.mismatch_behavior.value,
            "common_vocab_families": self._identify_vocab_family(vocab_size_a, vocab_size_b)
        }
        
        report = VocabularyCompatibilityReport(
            vocab_size_a=vocab_size_a,
            vocab_size_b=vocab_size_b,
            shared_tokens=shared_tokens,
            overlap_ratio=overlap_ratio,
            size_difference=size_difference,
            size_difference_ratio=size_difference_ratio,
            is_subset=is_subset,
            is_extension=is_extension,
            recommended_strategy=strategy,
            warnings=warnings,
            metadata=metadata
        )
        
        # Log the analysis
        self._log_analysis(report)
        
        return report
    
    def determine_shared_token_space(
        self,
        vocab_size_a: int,
        vocab_size_b: int
    ) -> Tuple[int, int]:
        """
        Determine the shared token space indices.
        
        Args:
            vocab_size_a: Vocabulary size of first model
            vocab_size_b: Vocabulary size of second model
            
        Returns:
            Tuple of (start_idx, end_idx) for shared token space
        """
        # Most tokenizers share the same initial tokens
        # Extensions typically happen at the end
        shared_end = min(vocab_size_a, vocab_size_b)
        return (0, shared_end)
    
    def calculate_overlap_percentage(
        self,
        vocab_size_a: int,
        vocab_size_b: int
    ) -> float:
        """
        Calculate the percentage of vocabulary overlap.
        
        Args:
            vocab_size_a: Vocabulary size of first model
            vocab_size_b: Vocabulary size of second model
            
        Returns:
            Overlap percentage (0.0 to 1.0)
        """
        if vocab_size_a == 0 or vocab_size_b == 0:
            return 0.0
        
        shared = min(vocab_size_a, vocab_size_b)
        max_size = max(vocab_size_a, vocab_size_b)
        return shared / max_size
    
    def suggest_verification_strategy(
        self,
        report: VocabularyCompatibilityReport
    ) -> Dict[str, Any]:
        """
        Suggest a verification strategy based on compatibility analysis.
        
        Args:
            report: Vocabulary compatibility report
            
        Returns:
            Dictionary with strategy recommendations
        """
        strategy = {
            "can_proceed": False,
            "method": None,
            "confidence_adjustment": 1.0,
            "notes": []
        }
        
        if report.overlap_ratio >= 0.99:
            # Nearly identical vocabularies
            strategy["can_proceed"] = True
            strategy["method"] = "standard_verification"
            strategy["notes"].append("Vocabularies are nearly identical (>99% overlap)")
            
        elif report.overlap_ratio >= 0.95:
            # High overlap, likely compatible
            strategy["can_proceed"] = True
            strategy["method"] = "shared_token_verification"
            strategy["confidence_adjustment"] = 0.98
            strategy["notes"].append(f"High vocabulary overlap ({report.overlap_ratio:.1%})")
            strategy["notes"].append(f"Verification limited to {report.shared_tokens:,} shared tokens")
            
        elif report.is_extension and self.allow_extended_vocabularies:
            # One vocabulary extends the other
            strategy["can_proceed"] = True
            strategy["method"] = "base_vocabulary_verification"
            strategy["confidence_adjustment"] = 0.95
            strategy["notes"].append("Detected vocabulary extension (likely fine-tuning)")
            strategy["notes"].append(f"Added {report.size_difference:,} specialized tokens")
            
        elif report.overlap_ratio >= 0.90:
            # Moderate overlap, proceed with caution
            strategy["can_proceed"] = True
            strategy["method"] = "limited_verification"
            strategy["confidence_adjustment"] = 0.90
            strategy["notes"].append("Moderate vocabulary overlap - results may be less reliable")
            strategy["notes"].append("Consider these as different model families")
            
        else:
            # Low overlap, likely incompatible
            strategy["can_proceed"] = False
            strategy["method"] = "incompatible"
            strategy["notes"].append(f"Low vocabulary overlap ({report.overlap_ratio:.1%})")
            strategy["notes"].append("Models appear to use different tokenizers/architectures")
        
        # Apply behavior policy
        if self.mismatch_behavior == VocabularyMismatchBehavior.FAIL and report.size_difference > 0:
            strategy["can_proceed"] = False
            strategy["notes"].append("Failing due to strict vocabulary matching policy")
        elif self.mismatch_behavior == VocabularyMismatchBehavior.ADAPT:
            strategy["notes"].append("Adapting verification strategy for vocabulary mismatch")
        
        return strategy
    
    def _determine_strategy(
        self,
        overlap_ratio: float,
        size_difference: int,
        is_extension: bool
    ) -> str:
        """Determine the recommended verification strategy"""
        if overlap_ratio >= 0.99:
            return "standard_verification"
        elif overlap_ratio >= 0.95:
            return "shared_token_verification"
        elif is_extension and self.allow_extended_vocabularies:
            return "base_vocabulary_verification"
        elif overlap_ratio >= 0.90:
            return "limited_verification"
        else:
            return "incompatible_architectures"
    
    def _generate_warnings(
        self,
        vocab_size_a: int,
        vocab_size_b: int,
        overlap_ratio: float,
        size_difference: int
    ) -> list:
        """Generate warnings based on vocabulary analysis"""
        warnings = []
        
        if size_difference > 0:
            warnings.append(
                f"Vocabulary size mismatch: {vocab_size_a:,} vs {vocab_size_b:,} "
                f"(difference: {size_difference:,} tokens)"
            )
        
        if overlap_ratio < self.min_overlap_ratio:
            warnings.append(
                f"Vocabulary overlap ({overlap_ratio:.1%}) below minimum threshold "
                f"({self.min_overlap_ratio:.1%})"
            )
        
        if size_difference > self.extension_threshold:
            warnings.append(
                f"Large vocabulary difference ({size_difference:,} tokens) suggests "
                "different model families or major architectural changes"
            )
        
        if 0.90 <= overlap_ratio < 0.95:
            warnings.append(
                "Moderate vocabulary overlap - verification results should be "
                "interpreted with caution"
            )
        
        return warnings
    
    def _identify_vocab_family(
        self,
        vocab_size_a: int,
        vocab_size_b: int
    ) -> str:
        """Identify common vocabulary families based on sizes"""
        sizes = {vocab_size_a, vocab_size_b}
        
        # Common vocabulary sizes and their families
        if sizes.issubset({50257, 50258, 50259, 50260}):
            return "GPT-2/GPT-3 family"
        elif sizes.issubset({32000, 32001, 32002, 32768}):
            return "LLaMA/Mistral family"
        elif sizes.issubset({30000, 30522, 30523}):
            return "BERT family"
        elif sizes.issubset({51200, 51201}):
            return "Phi family"
        elif all(49000 <= s <= 52000 for s in sizes):
            return "Extended GPT family"
        else:
            return "Mixed or unknown families"
    
    def _log_analysis(self, report: VocabularyCompatibilityReport):
        """Log the vocabulary analysis results"""
        if report.is_highly_compatible:
            logger.info(
                f"Vocabulary sizes highly compatible: {report.vocab_size_a:,} vs "
                f"{report.vocab_size_b:,} ({report.overlap_ratio:.1%} overlap)"
            )
        elif report.is_compatible:
            logger.info(
                f"Vocabulary sizes compatible: {report.vocab_size_a:,} vs "
                f"{report.vocab_size_b:,} ({report.overlap_ratio:.1%} overlap). "
                f"Proceeding with {report.shared_tokens:,} shared tokens."
            )
        else:
            logger.warning(
                f"Vocabulary compatibility issues detected: {report.vocab_size_a:,} vs "
                f"{report.vocab_size_b:,} ({report.overlap_ratio:.1%} overlap)"
            )
        
        for warning in report.warnings:
            logger.warning(warning)