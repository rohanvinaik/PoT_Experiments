"""
Enhanced Statistical Difference Decision Framework with Variance-Based Relationship Inference

This module extends the basic diff_decision framework with sophisticated variance analysis
to identify structural relationships between models (same architecture/different scale,
fine-tuning, distillation, etc.) rather than returning UNDECIDED.
"""

import math
import logging
import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Literal
from dataclasses import dataclass
from enum import Enum

from .diff_decision import (
    DiffDecisionConfig, 
    TestingMode,
    EnhancedSequentialTester as BaseTester
)

logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED DECISION TYPES
# ============================================================================

class ModelRelationship(Enum):
    """Detailed model relationship categories based on statistical signatures"""
    
    # Identity
    IDENTICAL = "IDENTICAL"  # Same model, same weights
    
    # Same architecture relationships
    SAME_ARCHITECTURE_DIFFERENT_SCALE = "SAME_ARCH_DIFF_SCALE"  # e.g., GPT-Neo-125M vs 1.3B
    SAME_ARCHITECTURE_QUANTIZED = "SAME_ARCH_QUANTIZED"  # Same model, different precision
    SAME_ARCHITECTURE_FINE_TUNED = "SAME_ARCH_FINE_TUNED"  # Fine-tuned variant
    
    # Different architecture relationships  
    DISTILLED = "DISTILLED"  # Student-teacher relationship
    DIFFERENT_ARCHITECTURE = "DIFFERENT_ARCH"  # Completely different models
    
    # Edge cases
    NEAR_CLONE = "NEAR_CLONE"  # Almost identical but not quite
    INCONCLUSIVE = "INCONCLUSIVE"  # Cannot determine (replaces UNDECIDED for true errors)

@dataclass
class VarianceSignature:
    """Statistical signature for variance-based relationship inference"""
    
    mean_effect: float  # Mean difference
    variance: float  # Variance of differences
    cv: float  # Coefficient of variation (std/mean)
    variance_ratio: float  # Ratio to expected variance
    n_samples: int
    ci_width: float  # Confidence interval width
    
    # Derived metrics
    normalized_variance: float = 0.0  # Variance normalized by effect size
    stability_score: float = 0.0  # How stable the measurements are
    
    def __post_init__(self):
        """Compute derived metrics"""
        if abs(self.mean_effect) > 1e-10:
            self.normalized_variance = self.variance / (self.mean_effect ** 2)
        else:
            self.normalized_variance = float('inf')
        
        # Stability score: lower is more stable
        self.stability_score = self.cv * math.sqrt(self.n_samples)

class EnhancedDiffTester(BaseTester):
    """Enhanced tester with variance-based relationship inference"""
    
    def __init__(self, config: DiffDecisionConfig):
        super().__init__(config)
        
        # Track additional statistics for variance analysis
        self.squared_diffs: List[float] = []
        self.running_variance = 0.0
        self.variance_m2 = 0.0  # For Welford's variance computation
        
        # Expected variance baselines (can be calibrated)
        self.expected_variance_same = 1e-6  # Expected for identical models
        self.expected_variance_scale = 0.01  # Expected for scale differences
        self.expected_variance_arch = 0.1  # Expected for architecture differences
    
    def update(self, x: float) -> None:
        """Enhanced update with variance tracking"""
        super().update(x)
        
        # Track squared differences for variance analysis
        if self.n > 1:
            self.squared_diffs.append((x - self.mean) ** 2)
            
            # Update running variance using Welford's method
            delta = x - self.mean
            self.variance_m2 += delta * delta * (self.n - 1) / self.n
    
    def compute_variance_signature(self) -> VarianceSignature:
        """Compute comprehensive variance signature"""
        
        if self.n < 2:
            return VarianceSignature(
                mean_effect=0, variance=0, cv=0, 
                variance_ratio=0, n_samples=self.n, ci_width=float('inf')
            )
        
        # Get CI for width calculation
        (ci_lo, ci_hi), ci_width = self.compute_ci()
        
        # Coefficient of variation
        cv = self.std_dev / abs(self.mean) if abs(self.mean) > 1e-10 else float('inf')
        
        # Variance ratio compared to expected for same model
        variance_ratio = self.variance / self.expected_variance_same if self.expected_variance_same > 0 else float('inf')
        
        return VarianceSignature(
            mean_effect=self.mean,
            variance=self.variance,
            cv=cv,
            variance_ratio=variance_ratio,
            n_samples=self.n,
            ci_width=ci_width
        )
    
    def infer_relationship(self, signature: VarianceSignature) -> ModelRelationship:
        """Infer model relationship from variance signature"""
        
        # Check for identical models first
        if (abs(signature.mean_effect) < 1e-6 and 
            signature.variance < self.expected_variance_same * 10):
            return ModelRelationship.IDENTICAL
        
        # Near-clone detection
        if (abs(signature.mean_effect) < 0.001 and
            signature.variance < self.expected_variance_same * 100):
            return ModelRelationship.NEAR_CLONE
        
        # Same architecture, different scale
        # Characterized by: moderate mean difference, moderate-high variance, stable CV
        if (0.001 < abs(signature.mean_effect) < 0.5 and
            self.expected_variance_same * 100 < signature.variance < self.expected_variance_arch and
            signature.cv < 2.0):  # Relatively stable coefficient of variation
            
            # Further distinguish based on variance patterns
            if signature.normalized_variance < 10:
                return ModelRelationship.SAME_ARCHITECTURE_DIFFERENT_SCALE
            elif signature.normalized_variance < 50:
                return ModelRelationship.SAME_ARCHITECTURE_FINE_TUNED
            else:
                return ModelRelationship.SAME_ARCHITECTURE_QUANTIZED
        
        # Distillation detection
        # Characterized by: large mean difference, moderate variance, specific patterns
        if (abs(signature.mean_effect) > 0.5 and
            signature.variance < self.expected_variance_arch * 0.5 and
            signature.cv < 1.0):  # Very stable despite large difference
            return ModelRelationship.DISTILLED
        
        # Different architecture
        # Characterized by: large mean difference, high variance, unstable CV
        if (abs(signature.mean_effect) > 0.1 and
            (signature.variance > self.expected_variance_arch or signature.cv > 2.0)):
            return ModelRelationship.DIFFERENT_ARCHITECTURE
        
        # Fine-tuning detection (fallback)
        # Characterized by: small-moderate mean difference, low-moderate variance
        if (0.01 < abs(signature.mean_effect) < 0.1 and
            signature.variance < self.expected_variance_scale):
            return ModelRelationship.SAME_ARCHITECTURE_FINE_TUNED
        
        # If we can't determine, it's truly inconclusive
        return ModelRelationship.INCONCLUSIVE
    
    def should_stop(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Enhanced stopping logic with relationship inference"""
        
        # First check traditional stopping criteria
        should_stop_base, base_info = super().should_stop()
        
        # If base says to stop with UNDECIDED, enhance the decision
        if should_stop_base and base_info and base_info.get("decision") == "UNDECIDED":
            
            # Compute variance signature
            signature = self.compute_variance_signature()
            
            # Infer relationship
            relationship = self.infer_relationship(signature)
            
            # Build enhanced decision info
            enhanced_info = {
                "decision": relationship.value,
                "relationship": relationship.name,
                "confidence": self.config.confidence,
                "mean_effect": signature.mean_effect,
                "variance": signature.variance,
                "cv": signature.cv,
                "variance_ratio": signature.variance_ratio,
                "n_samples": self.n,
                "ci": base_info.get("ci"),
                "ci_width": signature.ci_width,
                "normalized_variance": signature.normalized_variance,
                "stability_score": signature.stability_score,
                "inference_basis": self._explain_inference(relationship, signature),
                "original_diagnostics": base_info.get("diagnostics")
            }
            
            # Map to traditional decisions for compatibility
            if relationship in [ModelRelationship.IDENTICAL, ModelRelationship.NEAR_CLONE]:
                enhanced_info["traditional_decision"] = "SAME"
            elif relationship == ModelRelationship.INCONCLUSIVE:
                enhanced_info["traditional_decision"] = "UNDECIDED"
            else:
                enhanced_info["traditional_decision"] = "DIFFERENT"
            
            return True, enhanced_info
        
        # For non-UNDECIDED cases, enhance with relationship info
        if should_stop_base and base_info:
            signature = self.compute_variance_signature()
            
            # Quick relationship inference for SAME/DIFFERENT cases
            if base_info.get("decision") == "SAME":
                relationship = ModelRelationship.IDENTICAL
            elif base_info.get("decision") == "DIFFERENT":
                # Use variance to distinguish type of difference
                if signature.cv < 1.0 and abs(signature.mean_effect) > 0.5:
                    relationship = ModelRelationship.DISTILLED
                elif signature.variance > self.expected_variance_arch:
                    relationship = ModelRelationship.DIFFERENT_ARCHITECTURE
                else:
                    relationship = ModelRelationship.SAME_ARCHITECTURE_DIFFERENT_SCALE
            else:
                relationship = ModelRelationship.INCONCLUSIVE
            
            base_info["relationship"] = relationship.name
            base_info["variance_signature"] = {
                "mean_effect": signature.mean_effect,
                "variance": signature.variance,
                "cv": signature.cv,
                "variance_ratio": signature.variance_ratio
            }
        
        return should_stop_base, base_info
    
    def _explain_inference(self, relationship: ModelRelationship, signature: VarianceSignature) -> str:
        """Explain the basis for the relationship inference"""
        
        explanations = {
            ModelRelationship.IDENTICAL: 
                f"Near-zero mean effect ({signature.mean_effect:.6f}) with minimal variance ({signature.variance:.6f})",
            
            ModelRelationship.SAME_ARCHITECTURE_DIFFERENT_SCALE:
                f"Moderate mean effect ({signature.mean_effect:.3f}) with variance ratio {signature.variance_ratio:.1f}x "
                f"suggests same architecture at different scales (e.g., 125M vs 1.3B parameters)",
            
            ModelRelationship.SAME_ARCHITECTURE_FINE_TUNED:
                f"Small mean effect ({signature.mean_effect:.3f}) with low variance suggests fine-tuning "
                f"or domain adaptation of same base model",
            
            ModelRelationship.SAME_ARCHITECTURE_QUANTIZED:
                f"Variance pattern (CV={signature.cv:.2f}) consistent with quantization effects",
            
            ModelRelationship.DISTILLED:
                f"Large stable difference (mean={signature.mean_effect:.3f}, CV={signature.cv:.2f}) "
                f"characteristic of student-teacher distillation",
            
            ModelRelationship.DIFFERENT_ARCHITECTURE:
                f"High variance (ratio={signature.variance_ratio:.1f}x) and instability (CV={signature.cv:.2f}) "
                f"indicates fundamentally different architectures",
            
            ModelRelationship.NEAR_CLONE:
                f"Tiny differences (mean={signature.mean_effect:.6f}) suggest near-identical models "
                f"with possible version or seed differences",
            
            ModelRelationship.INCONCLUSIVE:
                f"Statistical patterns do not match known relationships "
                f"(mean={signature.mean_effect:.3f}, variance={signature.variance:.3f}, n={signature.n_samples})"
        }
        
        return explanations.get(relationship, "Unknown relationship pattern")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary with relationship inference"""
        
        base_summary = super().get_state()
        signature = self.compute_variance_signature()
        relationship = self.infer_relationship(signature)
        
        return {
            **base_summary,
            "relationship": relationship.name,
            "relationship_confidence": self._compute_relationship_confidence(signature),
            "variance_signature": {
                "mean_effect": signature.mean_effect,
                "variance": signature.variance,  
                "cv": signature.cv,
                "variance_ratio": signature.variance_ratio,
                "normalized_variance": signature.normalized_variance,
                "stability_score": signature.stability_score
            },
            "inference_explanation": self._explain_inference(relationship, signature)
        }
    
    def _compute_relationship_confidence(self, signature: VarianceSignature) -> float:
        """Compute confidence in the relationship inference"""
        
        # Base confidence on number of samples
        sample_confidence = min(signature.n_samples / 100, 1.0)
        
        # Adjust based on stability
        if signature.stability_score < 1.0:
            stability_factor = 1.0
        elif signature.stability_score < 10.0:
            stability_factor = 0.9
        else:
            stability_factor = 0.7
        
        # Adjust based on CI width relative to mean
        if abs(signature.mean_effect) > 1e-10:
            ci_factor = max(0.5, 1.0 - signature.ci_width / abs(signature.mean_effect))
        else:
            ci_factor = 0.5
        
        return sample_confidence * stability_factor * ci_factor


def create_enhanced_tester(mode: TestingMode = TestingMode.AUDIT_GRADE) -> EnhancedDiffTester:
    """Factory function to create enhanced tester with appropriate config"""
    
    config = DiffDecisionConfig(mode=mode)
    return EnhancedDiffTester(config)


# Example usage and testing
if __name__ == "__main__":
    import random
    
    # Simulate different model relationships
    def simulate_model_comparison(relationship_type: str, n_samples: int = 100):
        """Simulate score differences for different model relationships"""
        
        tester = create_enhanced_tester(TestingMode.AUDIT_GRADE)
        
        for i in range(n_samples):
            if relationship_type == "identical":
                # Identical models: near-zero differences
                score_diff = random.gauss(0, 0.0001)
            
            elif relationship_type == "same_arch_diff_scale":
                # Same architecture, different scale (e.g., GPT-Neo 125M vs 1.3B)
                # Moderate mean difference with moderate variance
                score_diff = random.gauss(0.1, 0.05) + random.uniform(-0.02, 0.02)
            
            elif relationship_type == "fine_tuned":
                # Fine-tuned variant
                score_diff = random.gauss(0.05, 0.01)
            
            elif relationship_type == "distilled":
                # Distilled model (student-teacher)
                score_diff = random.gauss(0.8, 0.1)
            
            elif relationship_type == "different_arch":
                # Completely different architectures
                score_diff = random.gauss(0.5, 0.3) + random.uniform(-0.5, 0.5)
            
            else:
                score_diff = random.gauss(0, 0.1)
            
            tester.update(score_diff)
            
            # Check if we should stop
            should_stop, info = tester.should_stop()
            if should_stop:
                print(f"\n{relationship_type.upper()} simulation:")
                print(f"  Decision: {info.get('decision')}")
                print(f"  Relationship: {info.get('relationship')}")
                print(f"  N samples: {tester.n}")
                print(f"  Mean effect: {tester.mean:.4f}")
                print(f"  Variance: {tester.variance:.6f}")
                if 'inference_basis' in info:
                    print(f"  Reasoning: {info['inference_basis']}")
                break
        
        return tester
    
    # Test different scenarios
    print("Testing Enhanced Diff Decision Framework")
    print("=" * 60)
    
    for relationship in ["identical", "same_arch_diff_scale", "fine_tuned", 
                         "distilled", "different_arch"]:
        simulate_model_comparison(relationship)
        print()