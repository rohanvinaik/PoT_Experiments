#!/usr/bin/env python3
"""
Google Colab Runner for PoT Experimental Suite
==============================================
This script runs the experimental PoT test suite in Google Colab.
Uses ONLY open models (GPT-2, DistilGPT-2) - NO AUTHENTICATION REQUIRED.

TO RUN IN GOOGLE COLAB:
1. Upload this file to Colab
2. Run these commands:
   !pip install transformers torch numpy scipy scikit-learn
   !python colab_experimental_runner.py
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def run_test_script(script_name, description):
    """Run a test script and capture results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ Test timed out after 5 minutes")
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: {script_name}")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def main():
    print("="*70)
    print("POT EXPERIMENTAL TEST SUITE - GOOGLE COLAB VERSION")
    print("="*70)
    print("This runner executes the experimental PoT tests")
    print("Using ONLY open models: GPT-2 and DistilGPT-2")
    print("NO AUTHENTICATION TOKENS REQUIRED")
    print("="*70)
    
    # Test scripts to run
    test_scripts = [
        ("scripts/run_statistical_verification.py", "Statistical Identity Verification"),
        ("scripts/test_llm_open_models_only.py", "LLM Verification (Open Models Only)"),
        ("scripts/run_fuzzy_verification.py", "Fuzzy Hash Verification"),
        ("scripts/run_provenance_verification.py", "Provenance Auditing"),
        ("scripts/experimental_report_clean.py", "Clean Reporting Format")
    ]
    
    results = {}
    
    # Run each test
    for script, description in test_scripts:
        success = run_test_script(script, description)
        results[description] = success
        
        if success:
            print(f"✅ {description} completed successfully")
        else:
            print(f"❌ {description} failed")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "environment": "Google Colab",
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "results": results
    }
    
    with open("colab_test_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: colab_test_results.json")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Review output above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())