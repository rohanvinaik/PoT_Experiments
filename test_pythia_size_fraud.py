#!/usr/bin/env python3
"""
Test Pythia-70M vs Pythia-160M for size fraud detection.
This validates the framework's ability to detect when a smaller model
is maliciously served instead of a larger claimed model.
"""

import sys
import os
import json
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pot.core.diff_verifier import EnhancedDifferenceVerifier
from pot.core.diff_decision import TestingMode
from pot.core.model_loader import UnifiedModelLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_pythia_size_fraud():
    """Test Pythia-70M vs Pythia-160M size fraud detection."""
    
    print("🎯 PYTHIA SIZE FRAUD DETECTION TEST")
    print("=" * 50)
    print("Testing framework's ability to detect smaller model substitution")
    print("70M parameters vs 160M parameters (2.3x size difference)")
    print()
    
    # Model paths
    pythia_70m_path = "/Users/rohanvinaik/LLM_Models/pythia-70m"
    pythia_160m_path = "/Users/rohanvinaik/LLM_Models/pythia-160m"
    
    # Test parameters
    test_config = {
        "model_a_path": pythia_70m_path,
        "model_b_path": pythia_160m_path, 
        "testing_mode": TestingMode.AUDIT_GRADE,
        "prf_key": "deadbeefcafebabe1234567890abcdef",
        "expected_result": "DIFFERENT",
        "test_purpose": "Size fraud detection"
    }
    
    print(f"📊 Test Configuration:")
    print(f"   • Model A: Pythia-70M ({pythia_70m_path})")
    print(f"   • Model B: Pythia-160M ({pythia_160m_path})")
    print(f"   • Testing Mode: {test_config['testing_mode'].name}")
    print(f"   • Expected: {test_config['expected_result']}")
    print()
    
    try:
        # Initialize verifier
        print("🔧 Initializing Enhanced Difference Verifier...")
        verifier = EnhancedDifferenceVerifier(
            mode=test_config["testing_mode"],
            prf_key=bytes.fromhex(test_config["prf_key"])
        )
        
        # Load models
        print("📥 Loading models...")
        
        start_time = time.time()
        model_a = AutoModelForCausalLM.from_pretrained(test_config["model_a_path"])
        tokenizer_a = AutoTokenizer.from_pretrained(test_config["model_a_path"])
        load_time_a = time.time() - start_time
        print(f"   ✅ Loaded Pythia-70M in {load_time_a:.3f}s")
        
        start_time = time.time()
        model_b = AutoModelForCausalLM.from_pretrained(test_config["model_b_path"])
        tokenizer_b = AutoTokenizer.from_pretrained(test_config["model_b_path"])
        load_time_b = time.time() - start_time
        print(f"   ✅ Loaded Pythia-160M in {load_time_b:.3f}s")
        
        # Get model info
        num_params_a = sum(p.numel() for p in model_a.parameters())
        num_params_b = sum(p.numel() for p in model_b.parameters())
        size_ratio = num_params_b / num_params_a
        
        print(f"📈 Model Statistics:")
        print(f"   • Pythia-70M: {num_params_a:,} parameters")
        print(f"   • Pythia-160M: {num_params_b:,} parameters")
        print(f"   • Size ratio: {size_ratio:.1f}x larger")
        print()
        
        # Run verification
        print("🔍 Running statistical difference verification...")
        verification_start = time.time()
        
        result = verifier.verify_different_models(
            model_a, model_b,
            model_a_name="pythia-70m",
            model_b_name="pythia-160m"
        )
        
        verification_time = time.time() - verification_start
        
        # Analyze results
        print("📊 VERIFICATION RESULTS")
        print("=" * 30)
        print(f"Decision: {result.decision}")
        print(f"Expected: {test_config['expected_result']}")
        print(f"Test Passed: {'✅ YES' if result.decision == test_config['expected_result'] else '❌ NO'}")
        print(f"Queries Used: {result.n_used}")
        print(f"Effect Size: {result.mean:.6f}")
        print(f"Confidence Interval: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
        print(f"Half Width: {result.half_width:.6f}")
        print(f"Verification Time: {verification_time:.3f}s")
        print(f"Time per Query: {verification_time/result.n_used:.3f}s")
        print()
        
        # Generate detailed report
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "size_fraud_detection",
            "test_case": "pythia_70m_vs_160m", 
            "models": {
                "model_a": {
                    "name": "pythia-70m",
                    "path": pythia_70m_path,
                    "parameters": num_params_a,
                    "load_time": load_time_a
                },
                "model_b": {
                    "name": "pythia-160m", 
                    "path": pythia_160m_path,
                    "parameters": num_params_b,
                    "load_time": load_time_b
                }
            },
            "size_analysis": {
                "parameter_ratio": size_ratio,
                "size_category": "2.3x fraud detection"
            },
            "verification_result": {
                "decision": result.decision,
                "expected": test_config["expected_result"],
                "test_passed": result.decision == test_config["expected_result"],
                "n_used": result.n_used,
                "effect_size": result.mean,
                "confidence_interval": [result.ci_lower, result.ci_upper],
                "half_width": result.half_width,
                "verification_time": verification_time,
                "time_per_query": verification_time / result.n_used
            },
            "testing_config": {
                "mode": test_config["testing_mode"].name,
                "confidence": verifier.config.confidence,
                "gamma": verifier.config.gamma,
                "delta_star": verifier.config.delta_star,
                "n_max": verifier.config.n_max
            }
        }
        
        # Save report
        output_file = f"experimental_results/pythia_size_fraud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Detailed report saved to: {output_file}")
        
        # Update rolling metrics
        try:
            rolling_file = "experimental_results/rolling_metrics.json"
            if os.path.exists(rolling_file):
                with open(rolling_file, 'r') as f:
                    rolling_data = json.load(f)
                
                # Add timing sample
                rolling_data["timing_samples"].append({
                    "t_per_query": verification_time / result.n_used,
                    "t_total": verification_time,
                    "hardware": "mps"
                })
                
                # Add statistical sample
                rolling_data["statistical_samples"].append({
                    "decision": result.decision,
                    "confidence": verifier.config.confidence,
                    "n_used": result.n_used,
                    "effect_size": abs(result.mean)
                })
                
                rolling_data["total_runs"] += 1
                rolling_data["successful_runs"] += 1
                rolling_data["last_updated"] = datetime.now().isoformat()
                
                with open(rolling_file, 'w') as f:
                    json.dump(rolling_data, f, indent=2)
                
                print(f"📊 Updated rolling metrics")
        except Exception as e:
            print(f"⚠️  Could not update rolling metrics: {e}")
        
        print()
        print("🎉 PYTHIA SIZE FRAUD DETECTION TEST COMPLETED!")
        
        if result.decision == test_config["expected_result"]:
            print("✅ SUCCESS: Framework correctly detected size difference")
            print(f"   • Detected {size_ratio:.1f}x parameter difference")
            print(f"   • Effect size: {result.mean:.3f}")
            print(f"   • High confidence with {result.n_used} queries")
        else:
            print("❌ FAILURE: Framework did not detect size difference")
            
        return report
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_pythia_size_fraud()