#!/usr/bin/env python3
"""
Test Pythia-70M vs Pythia-160M for size fraud detection using the working framework.
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pot.core.diff_decision import (
    DiffDecisionConfig, 
    TestingMode,
    EnhancedSequentialTester,
    create_enhanced_verifier
)

def test_pythia_size_fraud():
    """Test Pythia size fraud detection."""
    
    print("🎯 PYTHIA SIZE FRAUD DETECTION")
    print("Testing 70M vs 160M parameter models")
    print("Expected: DIFFERENT (size fraud detection)")
    print()
    
    try:
        # Use the working framework pattern
        verifier = create_enhanced_verifier(
            mode=TestingMode.AUDIT_GRADE,
            ref_model_name='/Users/rohanvinaik/LLM_Models/pythia-70m',
            cand_model_name='/Users/rohanvinaik/LLM_Models/pythia-160m',
            prf_key=b'deadbeefcafebabe1234567890abcdef',
            calibration_file=None
        )
        
        print("🔍 Running verification...")
        start_time = time.time()
        
        result = verifier.verify()
        
        verification_time = time.time() - start_time
        
        print(f"📊 RESULTS:")
        print(f"   Decision: {result.decision}")
        print(f"   Expected: DIFFERENT")
        print(f"   Test Passed: {'✅' if result.decision == 'DIFFERENT' else '❌'}")
        print(f"   Queries Used: {result.n_used}")
        print(f"   Effect Size: {result.mean:.6f}")
        print(f"   Confidence Interval: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
        print(f"   Verification Time: {verification_time:.3f}s")
        print(f"   Time per Query: {verification_time/result.n_used:.3f}s")
        
        # Save results
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "size_fraud_detection",
            "model_pair": "pythia_70m_vs_160m",
            "parameter_ratio": 2.3,
            "result": {
                "decision": result.decision,
                "expected": "DIFFERENT",
                "test_passed": result.decision == "DIFFERENT",
                "n_used": result.n_used,
                "effect_size": result.mean,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "verification_time": verification_time,
                "time_per_query": verification_time / result.n_used
            }
        }
        
        output_file = f"experimental_results/pythia_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Results saved to: {output_file}")
        
        return result.decision == "DIFFERENT"
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pythia_size_fraud()
    sys.exit(0 if success else 1)