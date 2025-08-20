"""
Vocabulary-Aware Statistical Testing Framework

This module extends the enhanced statistical testing framework to account for
vocabulary differences between models, adjusting confidence calculations and
providing nuanced decision categories.
"""

import math
import logging
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from .diff_decision import (
    EnhancedSequentialTester,
    DiffDecisionConfig,
    TestingMode
)
from .vocabulary_compatibility import (
    VocabularyCompatibilityAnalyzer,
    VocabularyCompatibilityReport
)

logger = logging.getLogger(__name__)


class VocabularyDecisionStatus(Enum):
    """Extended decision categories for vocabulary-aware testing"""
    SAME = "SAME"                          # Identical models
    SAME_EXTENDED = "SAME_EXTENDED"        # Same base model with vocabulary additions
    SAME_REDUCED = "SAME_REDUCED"          # Same base model with vocabulary pruning  
    SAME_ADAPTED = "SAME_ADAPTED"          # Same model with minor vocabulary changes
    DIFFERENT = "DIFFERENT"                # Actually different models
    UNDECIDED = "UNDECIDED"                # Need more samples
    ERROR = "ERROR"                        # Error in decision process


@dataclass
class VocabularyAwareDecisionResult:
    """Extended decision result with vocabulary analysis"""
    status: str
    confidence: float
    samples_used: int
    mean_difference: float
    ci_lower: float
    ci_upper: float
    ci_half_width: float
    effect_size: float
    relative_margin_error: float
    details: Dict[str, Any]
    suggestions: List[str]
    vocabulary_status: Optional[VocabularyDecisionStatus] = None
    vocabulary_overlap_ratio: float = 1.0
    shared_token_count: int = 0
    reference_unique_tokens: int = 0
    candidate_unique_tokens: int = 0
    vocabulary_extension_detected: bool = False
    vocabulary_reduction_detected: bool = False
    confidence_adjustment_factor: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'status': self.status,
            'confidence': self.confidence,
            'samples_used': self.samples_used,
            'mean_difference': self.mean_difference,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'ci_half_width': self.ci_half_width,
            'effect_size': self.effect_size,
            'relative_margin_error': self.relative_margin_error,
            'details': self.details,
            'suggestions': self.suggestions,
            'vocabulary_status': self.vocabulary_status.value if self.vocabulary_status else None,
            'vocabulary_overlap_ratio': self.vocabulary_overlap_ratio,
            'shared_token_count': self.shared_token_count,
            'reference_unique_tokens': self.reference_unique_tokens,
            'candidate_unique_tokens': self.candidate_unique_tokens,
            'vocabulary_extension_detected': self.vocabulary_extension_detected,
            'vocabulary_reduction_detected': self.vocabulary_reduction_detected,
            'confidence_adjustment_factor': self.confidence_adjustment_factor
        }


class VocabularyAwareSequentialTester(EnhancedSequentialTester):
    """
    Statistical tester that accounts for vocabulary differences between models.
    
    This class extends the enhanced sequential tester to:
    1. Adjust confidence based on vocabulary overlap
    2. Distinguish between model differences and vocabulary extensions
    3. Provide detailed vocabulary-aware diagnostics
    """
    
    def __init__(
        self,
        config: DiffDecisionConfig,
        vocab_overlap_ratio: float = 1.0,
        reference_vocab_size: int = 0,
        candidate_vocab_size: int = 0,
        shared_token_count: int = 0,
        adjust_confidence_for_vocabulary: bool = True,
        vocabulary_analyzer: Optional[VocabularyCompatibilityAnalyzer] = None
    ):
        """
        Initialize vocabulary-aware sequential tester.
        
        Args:
            config: Base configuration for statistical testing
            vocab_overlap_ratio: Ratio of shared vocabulary (0.0 to 1.0)
            reference_vocab_size: Size of reference model vocabulary
            candidate_vocab_size: Size of candidate model vocabulary
            shared_token_count: Number of shared tokens between models
            adjust_confidence_for_vocabulary: Whether to adjust confidence for vocab differences
            vocabulary_analyzer: Optional analyzer for detailed vocabulary analysis
        """
        super().__init__(config)
        
        self.vocab_overlap_ratio = vocab_overlap_ratio
        self.reference_vocab_size = reference_vocab_size
        self.candidate_vocab_size = candidate_vocab_size
        self.shared_token_count = shared_token_count
        self.adjust_confidence_for_vocabulary = adjust_confidence_for_vocabulary
        self.vocabulary_analyzer = vocabulary_analyzer or VocabularyCompatibilityAnalyzer()
        
        # Calculate unique tokens
        self.reference_unique_tokens = max(0, reference_vocab_size - shared_token_count)
        self.candidate_unique_tokens = max(0, candidate_vocab_size - shared_token_count)
        
        # Determine vocabulary relationship
        self.vocabulary_extension_detected = candidate_vocab_size > reference_vocab_size
        self.vocabulary_reduction_detected = candidate_vocab_size < reference_vocab_size
        
        # Calculate confidence adjustment factor based on vocabulary overlap
        self.confidence_adjustment_factor = self._calculate_confidence_adjustment()
        
        # Adjust decision thresholds based on vocabulary differences
        if self.adjust_confidence_for_vocabulary:
            self._adjust_thresholds_for_vocabulary()
    
    def _calculate_confidence_adjustment(self) -> float:
        """
        Calculate confidence adjustment factor based on vocabulary overlap.
        
        Returns adjustment factor between 0.5 and 1.0:
        - 1.0 for perfect overlap (100%)
        - 0.98 for high overlap (95-99%)
        - 0.95 for good overlap (90-95%)
        - 0.90 for moderate overlap (80-90%)
        - 0.80 for low overlap (70-80%)
        - 0.70 for poor overlap (60-70%)
        - 0.50 for very poor overlap (<60%)
        """
        if self.vocab_overlap_ratio >= 0.99:
            return 1.0
        elif self.vocab_overlap_ratio >= 0.95:
            return 0.98
        elif self.vocab_overlap_ratio >= 0.90:
            return 0.95
        elif self.vocab_overlap_ratio >= 0.80:
            return 0.90
        elif self.vocab_overlap_ratio >= 0.70:
            return 0.80
        elif self.vocab_overlap_ratio >= 0.60:
            return 0.70
        else:
            return 0.50
    
    def _adjust_thresholds_for_vocabulary(self):
        """
        Adjust decision thresholds based on vocabulary differences.
        
        When vocabularies differ significantly:
        - Increase gamma (equivalence band) for SAME decisions
        - Decrease delta_star (minimum effect size) for DIFFERENT decisions
        - Increase required sample sizes
        """
        if self.vocab_overlap_ratio < 1.0:
            # Adjust equivalence band based on overlap
            overlap_factor = self.vocab_overlap_ratio
            
            # Make SAME decision harder with lower overlap
            self.config.gamma *= (1.0 + (1.0 - overlap_factor) * 0.5)
            
            # Make DIFFERENT decision easier with lower overlap
            self.config.delta_star *= (0.8 + overlap_factor * 0.2)
            
            # Increase minimum samples for lower overlap
            if self.vocab_overlap_ratio < 0.90:
                self.config.n_min = int(self.config.n_min * (1.0 + (1.0 - overlap_factor)))
            
            logger.info(f"Adjusted thresholds for {self.vocab_overlap_ratio:.1%} vocabulary overlap:")
            logger.info(f"  gamma: {self.config.gamma:.4f}")
            logger.info(f"  delta_star: {self.config.delta_star:.4f}")
            logger.info(f"  n_min: {self.config.n_min}")
    
    def make_decision(self, samples: List[float]) -> VocabularyAwareDecisionResult:
        """
        Make vocabulary-aware decision based on samples.
        
        Args:
            samples: List of difference scores
        
        Returns:
            VocabularyAwareDecisionResult with detailed analysis
        """
        # Feed samples to tester
        for sample in samples:
            self.update(sample)
        
        # Get decision from parent class
        should_stop, decision_info = self.should_stop()
        
        # Extract base statistics
        (ci_lo, ci_hi), half_width = self.compute_ci()
        mean = self.mean
        n = self.n
        
        # Calculate additional metrics
        sigma = math.sqrt(self.variance) if self.variance > 0 else 1e-10
        effect_size = abs(mean) / max(sigma, 1e-10)
        relative_margin_error = half_width / max(abs(mean), self.config.min_effect_floor)
        
        # Determine status and details
        if decision_info:
            status = decision_info.get("decision", "UNDECIDED")
            details = decision_info.copy()
            suggestions = decision_info.get("suggestions", [])
        else:
            status = "UNDECIDED"
            details = {"reason": "Need more samples"}
            suggestions = [f"Current n={n}, need at least {self.config.n_min}"]
        
        # Base confidence (will be adjusted for vocabulary)
        confidence = self.config.confidence
        
        # Create vocabulary-aware result
        result = VocabularyAwareDecisionResult(
            status=status,
            confidence=confidence,
            samples_used=n,
            mean_difference=mean,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            ci_half_width=half_width,
            effect_size=effect_size,
            relative_margin_error=relative_margin_error,
            details=details,
            suggestions=suggestions if isinstance(suggestions, list) else [],
            vocabulary_overlap_ratio=self.vocab_overlap_ratio,
            shared_token_count=self.shared_token_count,
            reference_unique_tokens=self.reference_unique_tokens,
            candidate_unique_tokens=self.candidate_unique_tokens,
            vocabulary_extension_detected=self.vocabulary_extension_detected,
            vocabulary_reduction_detected=self.vocabulary_reduction_detected,
            confidence_adjustment_factor=self.confidence_adjustment_factor
        )
        
        # Adjust decision based on vocabulary analysis
        if self.adjust_confidence_for_vocabulary and self.vocab_overlap_ratio < 1.0:
            result = self._adjust_for_vocabulary_mismatch(result)
        
        # Adjust confidence based on vocabulary overlap
        result.confidence *= self.confidence_adjustment_factor
        
        return result
    
    def _adjust_for_vocabulary_mismatch(
        self, 
        decision: VocabularyAwareDecisionResult
    ) -> VocabularyAwareDecisionResult:
        """
        Adjust decision based on vocabulary mismatch analysis.
        
        Args:
            decision: Initial decision result
        
        Returns:
            Adjusted decision with vocabulary-aware status
        """
        # High overlap with DIFFERENT decision suggests vocabulary extension
        if decision.status == "DIFFERENT" and self.vocab_overlap_ratio > 0.95:
            if self.vocabulary_extension_detected:
                decision.vocabulary_status = VocabularyDecisionStatus.SAME_EXTENDED
                decision.status = "SAME_EXTENDED"
                decision.details['vocabulary_interpretation'] = (
                    f"High vocabulary overlap ({self.vocab_overlap_ratio:.1%}) with "
                    f"{self.candidate_unique_tokens:,} additional tokens suggests "
                    "vocabulary extension rather than different model"
                )
            elif self.vocabulary_reduction_detected:
                decision.vocabulary_status = VocabularyDecisionStatus.SAME_REDUCED
                decision.status = "SAME_REDUCED"
                decision.details['vocabulary_interpretation'] = (
                    f"High vocabulary overlap ({self.vocab_overlap_ratio:.1%}) with "
                    f"{self.reference_unique_tokens:,} tokens removed suggests "
                    "vocabulary pruning rather than different model"
                )
            else:
                decision.vocabulary_status = VocabularyDecisionStatus.SAME_ADAPTED
                decision.status = "SAME_ADAPTED"
                decision.details['vocabulary_interpretation'] = (
                    f"High vocabulary overlap ({self.vocab_overlap_ratio:.1%}) with "
                    "minor vocabulary changes suggests adapted model"
                )
            
            # Add suggestion for further verification
            decision.suggestions.append(
                "Consider additional semantic tests to verify model similarity"
            )
        
        # Low overlap with SAME decision needs warning
        elif decision.status == "SAME" and self.vocab_overlap_ratio < 0.80:
            decision.details['vocabulary_warning'] = (
                f"Low vocabulary overlap ({self.vocab_overlap_ratio:.1%}) "
                "but statistical similarity detected. Results may be unreliable."
            )
            decision.suggestions.append(
                "Increase sample size or use semantic verification methods"
            )
            decision.vocabulary_status = VocabularyDecisionStatus.SAME
        
        # Standard SAME decision with high overlap
        elif decision.status == "SAME":
            decision.vocabulary_status = VocabularyDecisionStatus.SAME
        
        # Standard DIFFERENT decision with low overlap
        elif decision.status == "DIFFERENT" and self.vocab_overlap_ratio < 0.80:
            decision.vocabulary_status = VocabularyDecisionStatus.DIFFERENT
            decision.details['vocabulary_confirmation'] = (
                f"Low vocabulary overlap ({self.vocab_overlap_ratio:.1%}) "
                "confirms models are different"
            )
        
        # UNDECIDED remains undecided
        elif decision.status == "UNDECIDED":
            decision.vocabulary_status = VocabularyDecisionStatus.UNDECIDED
            
            # Add vocabulary-specific suggestions
            if self.vocab_overlap_ratio < 0.90:
                decision.suggestions.append(
                    f"Low vocabulary overlap ({self.vocab_overlap_ratio:.1%}) "
                    "may require more samples for reliable decision"
                )
        
        return decision
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report including vocabulary analysis.
        
        Returns:
            Dictionary with full statistical and vocabulary analysis
        """
        # Build base report manually since parent doesn't have this method
        (ci_lo, ci_hi), half_width = self.compute_ci()
        
        report = {
            'config': {
                'mode': self.config.mode.value if hasattr(self.config.mode, 'value') else str(self.config.mode),
                'confidence': self.config.confidence,
                'gamma': self.config.gamma,
                'delta_star': self.config.delta_star,
                'epsilon_diff': self.config.epsilon_diff
            },
            'sampling': {
                'n_current': self.n,
                'n_min': self.config.n_min,
                'n_max': self.config.n_max,
                'positions_per_prompt': self.config.positions_per_prompt
            },
            'results': {
                'mean': self.mean,
                'variance': self.variance,
                'ci_lower': ci_lo,
                'ci_upper': ci_hi,
                'half_width': half_width,
                'effect_size': abs(self.mean) / max(math.sqrt(self.variance), 1e-10) if self.variance > 0 else 0
            },
            'decision': {
                'status': 'UNDECIDED'  # Will be updated
            }
        }
        
        # Add vocabulary analysis section
        report['vocabulary_analysis'] = {
            'overlap_ratio': self.vocab_overlap_ratio,
            'shared_tokens': self.shared_token_count,
            'reference_vocab_size': self.reference_vocab_size,
            'candidate_vocab_size': self.candidate_vocab_size,
            'unique_to_reference': self.reference_unique_tokens,
            'unique_to_candidate': self.candidate_unique_tokens,
            'verification_adapted': self.vocab_overlap_ratio < 1.0,
            'confidence_adjustment_factor': self.confidence_adjustment_factor
        }
        
        # Add vocabulary relationship
        if self.vocabulary_extension_detected:
            report['vocabulary_analysis']['relationship'] = 'extended'
            report['vocabulary_analysis']['tokens_added'] = self.candidate_unique_tokens
        elif self.vocabulary_reduction_detected:
            report['vocabulary_analysis']['relationship'] = 'reduced'
            report['vocabulary_analysis']['tokens_removed'] = self.reference_unique_tokens
        else:
            report['vocabulary_analysis']['relationship'] = 'identical' if self.vocab_overlap_ratio == 1.0 else 'adapted'
        
        # Add threshold adjustments if applied
        if self.adjust_confidence_for_vocabulary and self.vocab_overlap_ratio < 1.0:
            report['vocabulary_analysis']['threshold_adjustments'] = {
                'gamma_adjusted': self.config.gamma,
                'delta_star_adjusted': self.config.delta_star,
                'n_min_adjusted': self.config.n_min
            }
        
        # Add interpretation guidelines
        report['vocabulary_analysis']['interpretation'] = self._get_interpretation_guidelines()
        
        return report
    
    def _get_interpretation_guidelines(self) -> Dict[str, str]:
        """
        Get interpretation guidelines based on vocabulary overlap.
        
        Returns:
            Dictionary with interpretation guidelines
        """
        guidelines = {}
        
        if self.vocab_overlap_ratio >= 0.99:
            guidelines['overlap_level'] = 'excellent'
            guidelines['confidence_impact'] = 'none'
            guidelines['recommendation'] = 'Standard verification sufficient'
        elif self.vocab_overlap_ratio >= 0.95:
            guidelines['overlap_level'] = 'high'
            guidelines['confidence_impact'] = 'minimal'
            guidelines['recommendation'] = 'Likely same model family or fine-tuning'
        elif self.vocab_overlap_ratio >= 0.90:
            guidelines['overlap_level'] = 'good'
            guidelines['confidence_impact'] = 'minor'
            guidelines['recommendation'] = 'Consider vocabulary extension/adaptation'
        elif self.vocab_overlap_ratio >= 0.80:
            guidelines['overlap_level'] = 'moderate'
            guidelines['confidence_impact'] = 'moderate'
            guidelines['recommendation'] = 'Increase sample size for reliability'
        elif self.vocab_overlap_ratio >= 0.70:
            guidelines['overlap_level'] = 'low'
            guidelines['confidence_impact'] = 'significant'
            guidelines['recommendation'] = 'Different model families likely'
        else:
            guidelines['overlap_level'] = 'poor'
            guidelines['confidence_impact'] = 'severe'
            guidelines['recommendation'] = 'Models likely incompatible for direct comparison'
        
        return guidelines
    
    def get_vocabulary_summary(self) -> str:
        """
        Get human-readable vocabulary analysis summary.
        
        Returns:
            Formatted string with vocabulary analysis
        """
        lines = [
            "Vocabulary Analysis Summary",
            "=" * 40,
            f"Overlap Ratio: {self.vocab_overlap_ratio:.1%}",
            f"Shared Tokens: {self.shared_token_count:,}",
            f"Reference Vocabulary: {self.reference_vocab_size:,} tokens",
            f"Candidate Vocabulary: {self.candidate_vocab_size:,} tokens"
        ]
        
        if self.vocabulary_extension_detected:
            lines.append(f"Extension Detected: +{self.candidate_unique_tokens:,} tokens")
        elif self.vocabulary_reduction_detected:
            lines.append(f"Reduction Detected: -{self.reference_unique_tokens:,} tokens")
        
        lines.extend([
            "",
            f"Confidence Adjustment: {self.confidence_adjustment_factor:.2f}x",
            f"Verification Adapted: {'Yes' if self.adjust_confidence_for_vocabulary else 'No'}"
        ])
        
        # Add interpretation
        guidelines = self._get_interpretation_guidelines()
        lines.extend([
            "",
            "Interpretation:",
            f"  Overlap Level: {guidelines['overlap_level']}",
            f"  Confidence Impact: {guidelines['confidence_impact']}",
            f"  Recommendation: {guidelines['recommendation']}"
        ])
        
        return "\n".join(lines)


def create_vocabulary_aware_tester(
    reference_vocab_size: int,
    candidate_vocab_size: int,
    config: Optional[DiffDecisionConfig] = None,
    mode: TestingMode = TestingMode.AUDIT_GRADE
) -> VocabularyAwareSequentialTester:
    """
    Factory function to create a vocabulary-aware tester.
    
    Args:
        reference_vocab_size: Size of reference model vocabulary
        candidate_vocab_size: Size of candidate model vocabulary
        config: Optional configuration (will create default if None)
        mode: Testing mode (QUICK_GATE or AUDIT_GRADE)
    
    Returns:
        Configured VocabularyAwareSequentialTester
    """
    # Create default config if not provided
    if config is None:
        config = DiffDecisionConfig(mode=mode)
    
    # Calculate vocabulary overlap
    shared_tokens = min(reference_vocab_size, candidate_vocab_size)
    max_vocab = max(reference_vocab_size, candidate_vocab_size)
    overlap_ratio = shared_tokens / max_vocab if max_vocab > 0 else 1.0
    
    # Create tester
    tester = VocabularyAwareSequentialTester(
        config=config,
        vocab_overlap_ratio=overlap_ratio,
        reference_vocab_size=reference_vocab_size,
        candidate_vocab_size=candidate_vocab_size,
        shared_token_count=shared_tokens,
        adjust_confidence_for_vocabulary=True
    )
    
    return tester