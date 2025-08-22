#!/usr/bin/env python3
"""Test the enhanced diff decision framework with variance-based relationship inference"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pot.core.enhanced_diff_decision import (
    EnhancedDiffTester,
    ModelRelationship,
    TestingMode,
    DiffDecisionConfig,
    create_enhanced_tester
)
import random
import json

def test_relationship_detection():
    """Test detection of different model relationships"""
    
    test_cases = [
        {
            "name": "Identical Models (GPT-2 vs GPT-2)",
            "mean": 0.0,
            "std": 0.0001,
            "n_samples": 30,
            "expected": ModelRelationship.IDENTICAL
        },
        {
            "name": "Same Architecture Different Scale (GPT-Neo 125M vs 1.3B)",
            "mean": 0.15,
            "std": 0.08,
            "n_samples": 100,
            "expected": ModelRelationship.SAME_ARCHITECTURE_DIFFERENT_SCALE
        },
        {
            "name": "Fine-tuned Model (GPT-2 vs DialoGPT)",
            "mean": 0.05,
            "std": 0.02,
            "n_samples": 32,
            "expected": ModelRelationship.SAME_ARCHITECTURE_FINE_TUNED
        },
        {
            "name": "Distilled Model (GPT-2 vs DistilGPT-2)",
            "mean": 0.8,
            "std": 0.15,
            "n_samples": 32,
            "expected": ModelRelationship.DISTILLED
        },
        {
            "name": "Different Architecture (Pythia vs GPT-Neo)",
            "mean": 0.4,
            "std": 0.35,
            "n_samples": 32,
            "expected": ModelRelationship.DIFFERENT_ARCHITECTURE
        }
    ]
    
    print("Testing Enhanced Model Relationship Detection")
    print("=" * 70)
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print("-" * 50)
        
        # Create tester
        tester = create_enhanced_tester(TestingMode.AUDIT_GRADE)
        
        # Simulate observations
        for i in range(test_case['n_samples']):
            # Generate score difference with specified statistics
            score_diff = random.gauss(test_case['mean'], test_case['std'])
            
            # Add some realistic noise/variation
            if test_case['expected'] == ModelRelationship.SAME_ARCHITECTURE_DIFFERENT_SCALE:
                # Add characteristic variance pattern for scale differences
                score_diff += random.uniform(-0.05, 0.05) * (i % 10) / 10
            
            tester.update(score_diff)
            
            # Check stopping condition
            should_stop, info = tester.should_stop()
            
            if should_stop and info:
                print(f"  Samples used: {tester.n}")
                print(f"  Decision: {info.get('decision', 'UNKNOWN')}")
                print(f"  Relationship: {info.get('relationship', 'UNKNOWN')}")
                print(f"  Mean effect: {tester.mean:.4f}")
                print(f"  Variance: {tester.variance:.6f}")
                
                if 'variance_signature' in info:
                    sig = info['variance_signature']
                    print(f"  CV: {sig.get('cv', 0):.3f}")
                    print(f"  Variance ratio: {sig.get('variance_ratio', 0):.1f}x")
                
                if 'inference_basis' in info:
                    print(f"  Reasoning: {info['inference_basis']}")
                
                # Check if it matches expected
                if info.get('relationship') == test_case['expected'].name:
                    print(f"  ✅ CORRECT: Detected {test_case['expected'].name}")
                else:
                    print(f"  ❌ MISMATCH: Expected {test_case['expected'].name}, got {info.get('relationship')}")
                
                break
        else:
            print(f"  ⚠️ Did not reach decision after {test_case['n_samples']} samples")
            summary = tester.get_summary()
            print(f"  Final state: {summary}")

def analyze_real_run(evidence_file: str):
    """Analyze a real validation run with the enhanced framework"""
    
    print(f"\n\nAnalyzing Real Validation Run: {evidence_file}")
    print("=" * 70)
    
    try:
        with open(evidence_file, 'r') as f:
            evidence = json.load(f)
        
        # Extract transcript entries
        transcript = evidence.get('transcript_entries', [])
        
        if not transcript:
            print("No transcript entries found")
            return
        
        # Create enhanced tester
        tester = create_enhanced_tester(TestingMode.AUDIT_GRADE)
        
        # Process transcript entries
        for entry in transcript:
            if 'score_diff' in entry:
                tester.update(entry['score_diff'])
        
        # Get final analysis
        summary = tester.get_summary()
        
        print(f"Processed {tester.n} samples")
        print(f"Mean effect: {summary['mean']:.4f}")
        print(f"Variance: {summary['variance']:.6f}")
        print(f"Relationship: {summary.get('relationship', 'UNKNOWN')}")
        print(f"Relationship confidence: {summary.get('relationship_confidence', 0):.2%}")
        print(f"\nInference explanation:")
        print(f"  {summary.get('inference_explanation', 'No explanation available')}")
        
        if 'variance_signature' in summary:
            sig = summary['variance_signature']
            print(f"\nVariance Signature:")
            print(f"  CV: {sig['cv']:.3f}")
            print(f"  Variance ratio: {sig['variance_ratio']:.1f}x")
            print(f"  Normalized variance: {sig['normalized_variance']:.3f}")
            print(f"  Stability score: {sig['stability_score']:.3f}")
        
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    # Run synthetic tests
    test_relationship_detection()
    
    # Analyze the GPT-Neo 125M vs 1.3B run that returned UNDECIDED
    evidence_file = "outputs/validation_reports/evidence_bundle_validation_20250822_113616.json"
    if os.path.exists(evidence_file):
        analyze_real_run(evidence_file)
    
    # Also analyze a few other interesting runs
    other_runs = [
        "outputs/validation_reports/evidence_bundle_validation_20250822_111651.json",  # Pythia 70m vs 160m
        "outputs/validation_reports/evidence_bundle_validation_20250822_113017.json",  # DistilGPT-2 vs GPT-2
        "outputs/validation_reports/evidence_bundle_validation_20250822_114124.json",  # Pythia 70m vs 70m (SAME)
    ]
    
    for run_file in other_runs:
        if os.path.exists(run_file):
            analyze_real_run(run_file)