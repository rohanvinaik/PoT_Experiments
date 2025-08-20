#!/usr/bin/env python3
"""
Calibrate decision thresholds based on actual model behavior
Fixes UNDECIDED outcomes by setting appropriate thresholds
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run threshold calibration"""
    
    print("🎯 THRESHOLD CALIBRATION FOR DECISION FRAMEWORK")
    print("=" * 60)
    
    from pot.core.threshold_calibration import (
        ThresholdCalibrator, 
        CalibrationConfig,
        AutoCalibrator
    )
    
    # Quick calibration for GPT-2
    print("\n📊 Quick Calibration for GPT-2 Self-Consistency")
    print("-" * 40)
    
    quick_calib = AutoCalibrator.quick_auto_calibrate("gpt2")
    
    print(f"\n📈 Same-Model Statistics:")
    print(f"   Mean: {quick_calib['same_model_mean']:.6f}")
    print(f"   Std: {quick_calib['same_model_std']:.6f}")
    print(f"   P95: {quick_calib['same_model_p95']:.6f}")
    print(f"   P99: {quick_calib['same_model_p99']:.6f}")
    
    print(f"\n🎯 Recommended Thresholds:")
    print(f"   γ (SAME threshold): {quick_calib['recommended_gamma']:.6f}")
    print(f"   δ* conservative: {quick_calib['recommended_delta_star_conservative']:.6f}")
    print(f"   δ* moderate: {quick_calib['recommended_delta_star_moderate']:.6f}")
    print(f"   δ* aggressive: {quick_calib['recommended_delta_star_aggressive']:.6f}")
    
    # Full calibration for GPT-2 family
    print("\n📊 Full Calibration for GPT-2 Family")
    print("-" * 40)
    
    calibrator = ThresholdCalibrator()
    
    config = CalibrationConfig(
        n_samples_same=30,
        n_samples_different=30,
        positions_per_prompt=32,
        save_scores=False  # Don't save raw scores to reduce file size
    )
    
    same_models = ["gpt2"]
    different_pairs = [("gpt2", "distilgpt2")]
    
    print(f"Calibrating with:")
    print(f"   Same-model: {same_models}")
    print(f"   Different pairs: {different_pairs}")
    
    full_calib = calibrator.full_calibration(
        same_models,
        different_pairs,
        config
    )
    
    # Display results
    print(f"\n📈 Aggregate Statistics:")
    print(f"   Same-model P99: {full_calib['same_model']['p99']:.6f}")
    if full_calib['different_model']['p10']:
        print(f"   Different-model P10: {full_calib['different_model']['p10']:.6f}")
        print(f"   Separation ratio: {full_calib['separation']['separation_ratio']:.2f}x")
    
    print(f"\n🎯 Optimal Thresholds:")
    opt = full_calib['optimal_thresholds']
    print(f"   γ (SAME): {opt['gamma']:.6f}")
    print(f"   δ* (DIFFERENT): {opt['delta_star']:.6f}")
    print(f"   η (precision): {opt['eta']:.3f}")
    print(f"   ε_diff (RME): {opt['epsilon_diff']:.3f}")
    
    print(f"\n📋 Recommendations:")
    rec = full_calib['recommendations']
    print(f"   Assessment: {rec['assessment']}")
    print(f"   Mode: {rec['mode']}")
    print(f"   n_min: {rec['n_min']}, n_max: {rec['n_max']}")
    print(f"   Expected samples (SAME): {rec['expected_samples_same']}")
    print(f"   Expected samples (DIFFERENT): {rec['expected_samples_different']}")
    if 'notes' in rec:
        print(f"   Notes: {rec['notes']}")
    
    # Save calibration
    output_dir = Path("experimental_results/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save quick calibration
    quick_file = output_dir / f"quick_calibration_{timestamp}.json"
    with open(quick_file, 'w') as f:
        json.dump(quick_calib, f, indent=2)
    print(f"\n💾 Quick calibration saved to: {quick_file}")
    
    # Save full calibration
    full_file = output_dir / f"full_calibration_{timestamp}.json"
    calibrator.save_calibration(full_calib, str(full_file))
    print(f"💾 Full calibration saved to: {full_file}")
    
    # Create recommended configuration
    recommended_config = {
        "timestamp": datetime.now().isoformat(),
        "description": "Calibrated thresholds for GPT-2 family models",
        "quick_gate": {
            "gamma": opt['gamma'] * 1.2,  # Slightly relaxed for quick mode
            "delta_star": opt['delta_star'] * 0.8,  # Slightly tighter for quick mode
            "eta": 0.5,
            "epsilon_diff": 0.25,
            "n_min": 20,
            "n_max": 150,
            "confidence": 0.975
        },
        "audit_grade": {
            "gamma": opt['gamma'],
            "delta_star": opt['delta_star'],
            "eta": opt['eta'],
            "epsilon_diff": opt['epsilon_diff'],
            "n_min": rec['n_min'],
            "n_max": rec['n_max'],
            "confidence": 0.99
        },
        "calibration_source": {
            "quick": str(quick_file),
            "full": str(full_file)
        }
    }
    
    config_file = output_dir / f"recommended_config_{timestamp}.json"
    with open(config_file, 'w') as f:
        json.dump(recommended_config, f, indent=2)
    print(f"💾 Recommended config saved to: {config_file}")
    
    # Display fix for UNDECIDED
    print("\n" + "=" * 60)
    print("🔧 FIX FOR UNDECIDED OUTCOMES")
    print("=" * 60)
    
    print("\nTo fix UNDECIDED outcomes in runtime validation, use these thresholds:")
    print(f"\n# For QUICK_GATE mode:")
    print(f"gamma = {recommended_config['quick_gate']['gamma']:.6f}")
    print(f"delta_star = {recommended_config['quick_gate']['delta_star']:.6f}")
    print(f"epsilon_diff = {recommended_config['quick_gate']['epsilon_diff']:.3f}")
    
    print(f"\n# For AUDIT_GRADE mode:")
    print(f"gamma = {recommended_config['audit_grade']['gamma']:.6f}")
    print(f"delta_star = {recommended_config['audit_grade']['delta_star']:.6f}")
    print(f"epsilon_diff = {recommended_config['audit_grade']['epsilon_diff']:.3f}")
    
    print("\n✅ Calibration complete! These thresholds should resolve UNDECIDED outcomes.")
    
    return recommended_config


if __name__ == "__main__":
    config = main()
    
    # Test the calibrated thresholds
    print("\n" + "=" * 60)
    print("🧪 TESTING CALIBRATED THRESHOLDS")
    print("=" * 60)
    
    # Import test function
    try:
        from scripts.test_calibrated_thresholds import test_with_calibration
        print("\nTesting calibrated thresholds with real models...")
        test_with_calibration(config)
    except ImportError:
        print("\nTo test calibrated thresholds, run:")
        print("python scripts/test_calibrated_thresholds.py")