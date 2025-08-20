
# Updated DiffDecisionConfig defaults based on calibration

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class TestingMode(Enum):
    QUICK_GATE = "quick_gate"
    AUDIT_GRADE = "audit_grade"

@dataclass
class CalibratedDiffDecisionConfig:
    '''Calibrated configuration based on empirical testing'''
    
    mode: TestingMode = TestingMode.QUICK_GATE
    
    def __post_init__(self):
        if self.mode == TestingMode.QUICK_GATE:
            # Calibrated for GPT-2 family
            self.confidence = 0.975
            self.gamma = 0.35  # SAME threshold
            self.delta_star = 0.20  # DIFFERENT threshold
            self.eta = 0.5
            self.epsilon_diff = 0.15
            self.n_min = 15
            self.n_max = 100
            self.positions_per_prompt = 32
        else:  # AUDIT_GRADE
            self.confidence = 0.99
            self.gamma = 0.30
            self.delta_star = 0.25
            self.eta = 0.4
            self.epsilon_diff = 0.10
            self.n_min = 25
            self.n_max = 200
            self.positions_per_prompt = 64
    
    # Score clipping for stability
    score_clip_low: float = -100.0
    score_clip_high: float = 100.0
    
    # Legacy compatibility
    @property
    def alpha(self):
        return (1 - self.confidence) / 2
    
    @property
    def beta(self):
        return (1 - self.confidence) / 2
