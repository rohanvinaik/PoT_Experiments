from dataclasses import dataclass
from typing import Optional
from enum import Enum

class ValidationMode(Enum):
    QUICK_GATE = "quick_gate"
    AUDIT_GRADE = "audit_grade"

@dataclass
class CalibratedConfig:
    """Configuration with properly calibrated thresholds based on actual measurements"""
    
    mode: ValidationMode
    
    # Calibrated thresholds from actual runs
    # Same-model p95 ≈ 3.39e-4, Near-clone p5 ≈ 0.0763
    
    # Core thresholds
    gamma: float = None  # Equivalence band for SAME
    delta_star: float = None  # Minimum effect for DIFFERENT
    
    # Precision targets
    epsilon_diff: float = None  # RME target for DIFFERENT
    eta: float = 0.5  # Precision factor for SAME (half_width ≤ η·γ)
    
    # Confidence
    confidence: float = None
    
    # Sampling parameters
    k_positions: int = None
    n_min: int = None
    n_max: int = None
    batch_size: int = 8
    
    # Score normalization
    clip_range: tuple = (0.0, 0.3)
    
    def __post_init__(self):
        """Set calibrated defaults based on mode"""
        
        # Calibrated values from experimental_results/calibration_test_results
        SAME_MODEL_P95 = 3.39e-4
        NEAR_CLONE_P5 = 0.0763
        
        if self.mode == ValidationMode.QUICK_GATE:
            # Quick gate settings
            self.gamma = self.gamma or 0.00102  # 3 * SAME_MODEL_P95
            self.delta_star = self.delta_star or 0.0383  # (NEAR_CLONE_P5 + SAME_MODEL_P95) / 2
            self.epsilon_diff = self.epsilon_diff or 0.20
            self.confidence = self.confidence or 0.975
            self.k_positions = self.k_positions or 64
            self.n_min = self.n_min or 12
            self.n_max = self.n_max or 120
            
        else:  # AUDIT_GRADE
            # Audit grade settings
            self.gamma = self.gamma or 0.00102
            self.delta_star = self.delta_star or 0.0383
            self.epsilon_diff = self.epsilon_diff or 0.10
            self.confidence = self.confidence or 0.99
            self.k_positions = self.k_positions or 128
            self.n_min = self.n_min or 30
            self.n_max = self.n_max or 400
    
    @property
    def alpha(self) -> float:
        """Convert confidence to alpha for CI computation"""
        return 1.0 - self.confidence
    
    @property
    def half_width_target_same(self) -> float:
        """Target half-width for SAME decision"""
        return self.eta * self.gamma  # 0.5 * 0.00102 = 0.00051
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary"""
        return {
            "mode": self.mode.value,
            "gamma": self.gamma,
            "delta_star": self.delta_star,
            "epsilon_diff": self.epsilon_diff,
            "eta": self.eta,
            "confidence": self.confidence,
            "alpha": self.alpha,
            "k_positions": self.k_positions,
            "n_min": self.n_min,
            "n_max": self.n_max,
            "half_width_target_same": self.half_width_target_same,
            "clip_range": self.clip_range
        }

def get_calibrated_config(mode: str = "audit") -> CalibratedConfig:
    """Factory function to get calibrated configuration"""
    if mode == "quick":
        return CalibratedConfig(mode=ValidationMode.QUICK_GATE)
    else:
        return CalibratedConfig(mode=ValidationMode.AUDIT_GRADE)