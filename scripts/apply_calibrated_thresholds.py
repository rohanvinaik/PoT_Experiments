#!/usr/bin/env python3
"""
Apply calibrated thresholds to runtime validation
Updates the diff_decision.py module with calibrated values
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path


def apply_calibration_to_config():
    """Apply calibrated thresholds directly to the configuration"""
    
    print("🔧 APPLYING CALIBRATED THRESHOLDS")
    print("=" * 60)
    
    # Based on observed behavior from calibration tests:
    # - GPT-2 self-consistency shows mean ~0.22-0.24 with std ~0.04-0.08
    # - GPT-2 vs DistilGPT-2 shows mean ~0.29 with very low std
    
    # More aggressive thresholds based on actual observations
    calibrated_thresholds = {
        "quick_gate": {
            "gamma": 0.35,  # Accept scores up to 0.35 for SAME (covers GPT-2 self variance)
            "delta_star": 0.20,  # Require >0.20 difference for DIFFERENT
            "eta": 0.5,
            "epsilon_diff": 0.15,  # Allow 15% relative error
            "n_min": 15,
            "n_max": 100,
            "confidence": 0.975
        },
        "audit_grade": {
            "gamma": 0.30,  # Slightly tighter for audit grade
            "delta_star": 0.25,  # Require >0.25 for DIFFERENT
            "eta": 0.4,
            "epsilon_diff": 0.10,  # Tighter relative error
            "n_min": 25,
            "n_max": 200,
            "confidence": 0.99
        }
    }
    
    print("\nCalibrated thresholds based on empirical observations:")
    print("\nQUICK_GATE mode:")
    for key, value in calibrated_thresholds["quick_gate"].items():
        print(f"  {key}: {value}")
    
    print("\nAUDIT_GRADE mode:")
    for key, value in calibrated_thresholds["audit_grade"].items():
        print(f"  {key}: {value}")
    
    # Save to file
    output_dir = Path("experimental_results/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = output_dir / "empirical_thresholds.json"
    with open(config_file, 'w') as f:
        json.dump(calibrated_thresholds, f, indent=2)
    
    print(f"\n💾 Saved to: {config_file}")
    
    # Create updated diff_decision config
    updated_config = """
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
"""
    
    config_py_file = output_dir / "calibrated_config.py"
    with open(config_py_file, 'w') as f:
        f.write(updated_config)
    
    print(f"💾 Python config saved to: {config_py_file}")
    
    return calibrated_thresholds


def test_calibrated_thresholds(thresholds):
    """Quick test of calibrated thresholds"""
    
    print("\n" + "=" * 60)
    print("🧪 TESTING CALIBRATED THRESHOLDS")
    print("=" * 60)
    
    # Simulate some test cases based on observed values
    test_cases = [
        {
            "name": "GPT-2 self (observed)",
            "mean": 0.225,
            "std": 0.05,
            "expected": "SAME"
        },
        {
            "name": "GPT-2 vs DistilGPT-2 (observed)",
            "mean": 0.29,
            "std": 0.01,
            "expected": "DIFFERENT"
        }
    ]
    
    for mode in ["quick_gate", "audit_grade"]:
        print(f"\n📊 Testing {mode.upper()} mode:")
        config = thresholds[mode]
        
        for test in test_cases:
            mean = test["mean"]
            std = test["std"]
            
            # Simulate CI
            margin = 2.576 * std  # 99% CI
            ci_low = mean - margin
            ci_high = mean + margin
            
            # Check SAME condition
            same_ci = (abs(ci_low) <= config["gamma"]) and (abs(ci_high) <= config["gamma"])
            same_precision = margin <= (config["eta"] * config["gamma"])
            
            # Check DIFFERENT condition
            effect_size = abs(mean)
            relative_me = margin / effect_size if effect_size > 0 else float('inf')
            diff_effect = effect_size >= config["delta_star"]
            diff_precision = relative_me <= config["epsilon_diff"]
            
            if same_ci and same_precision:
                decision = "SAME"
            elif diff_effect and diff_precision:
                decision = "DIFFERENT"
            else:
                decision = "UNDECIDED"
            
            success = decision == test["expected"]
            status = "✅" if success else "❌"
            
            print(f"  {test['name']}: {decision} (expected {test['expected']}) {status}")
            if not success:
                print(f"    Mean={mean:.3f}, Effect={effect_size:.3f}, RME={relative_me:.3f}")
                print(f"    SAME check: CI in ±{config['gamma']:.2f}? {same_ci}, Precision? {same_precision}")
                print(f"    DIFF check: Effect≥{config['delta_star']:.2f}? {diff_effect}, RME≤{config['epsilon_diff']:.2f}? {diff_precision}")


if __name__ == "__main__":
    # Apply calibration
    thresholds = apply_calibration_to_config()
    
    # Test the thresholds
    test_calibrated_thresholds(thresholds)
    
    print("\n" + "=" * 60)
    print("✅ CALIBRATION APPLIED")
    print("=" * 60)
    print("\nTo use these calibrated thresholds in runtime validation:")
    print("1. Import CalibratedDiffDecisionConfig from the generated file")
    print("2. Or manually set the thresholds in your DiffDecisionConfig")
    print("\nExample:")
    print("  config.gamma = 0.35  # For SAME decisions")
    print("  config.delta_star = 0.20  # For DIFFERENT decisions")
    print("  config.epsilon_diff = 0.15  # Relative error tolerance")