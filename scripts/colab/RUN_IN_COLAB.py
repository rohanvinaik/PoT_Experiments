#!/usr/bin/env python3
"""
GOOGLE COLAB SETUP AND RUNNER FOR POT EXPERIMENTS
==================================================

INSTRUCTIONS TO RUN IN GOOGLE COLAB:

1. Open Google Colab (colab.research.google.com)
2. Create a new notebook
3. Run this single cell:

!git clone https://github.com/yourusername/PoT_Experiments.git
!cd PoT_Experiments && python RUN_IN_COLAB.py

That's it! This script will handle everything else.
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

def install_dependencies():
    """Install all required dependencies."""
    print("📦 Installing dependencies...")
    packages = [
        "transformers",
        "torch", 
        "numpy",
        "scipy",
        "scikit-learn"
    ]
    
    for pkg in packages:
        print(f"  Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
    
    print("✅ Dependencies installed\n")

def run_test_file(filepath, description):
    """Run a single test file."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"File: {filepath}")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, filepath],
            capture_output=True,
            text=True,
            timeout=120
        )
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("="*70)
    print("POT EXPERIMENTAL SUITE - GOOGLE COLAB")
    print("="*70)
    print("This suite uses ONLY open models:")
    print("  • GPT-2 (124M params)")
    print("  • DistilGPT-2 (82M params)")
    print("NO AUTHENTICATION TOKENS REQUIRED!")
    print("="*70)
    
    # Install dependencies
    install_dependencies()
    
    # Ensure we're in the right directory
    if os.path.exists("PoT_Experiments"):
        os.chdir("PoT_Experiments")
    
    print(f"📍 Working directory: {os.getcwd()}\n")
    
    # Create results directory
    Path("experimental_results").mkdir(exist_ok=True)
    
    # Tests to run
    tests = [
        ("scripts/run_statistical_verification.py", "Statistical Identity"),
        ("scripts/test_llm_open_models_only.py", "LLM Open Models"), 
        ("scripts/run_fuzzy_verification.py", "Fuzzy Hash"),
        ("scripts/run_provenance_verification.py", "Provenance"),
        ("scripts/experimental_report_clean.py", "Clean Reporting")
    ]
    
    results = {}
    
    # Run each test
    for filepath, name in tests:
        if os.path.exists(filepath):
            success = run_test_file(filepath, name)
            results[name] = success
        else:
            print(f"⚠️ Skipping {name} - file not found: {filepath}")
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.0f}%\n")
    
    for test, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {test}")
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "environment": "Google Colab",
        "models_used": ["gpt2", "distilgpt2"],
        "tests_run": total,
        "tests_passed": passed,
        "results": results
    }
    
    with open("colab_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Results saved to: colab_summary.json")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed")
    
    # Show key results files
    print("\n📊 KEY RESULTS:")
    results_dir = Path("experimental_results")
    if results_dir.exists():
        for file in results_dir.glob("*.json"):
            print(f"  • {file.name}")

if __name__ == "__main__":
    main()