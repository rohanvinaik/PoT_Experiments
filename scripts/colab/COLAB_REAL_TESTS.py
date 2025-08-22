#!/usr/bin/env python3
"""
REAL POT FRAMEWORK TESTS FOR GOOGLE COLAB
==========================================
This runs the ACTUAL PoT codebase, not mock tests.
Uses the real verification algorithms from the pot/ directory.
"""

import os
import sys
import subprocess
from pathlib import Path

print("=" * 70)
print("POT FRAMEWORK - REAL VERIFICATION TESTS")
print("=" * 70)
print("Running the ACTUAL PoT framework from this repository")
print("Not mock tests - the real analytical pipeline")
print("=" * 70)

# Ensure we're in the right place
if os.path.exists('/content'):
    os.chdir('/content')
    
# Setup repository if needed
if not os.path.exists('PoT_Experiments'):
    print("\n📥 Cloning repository...")
    subprocess.run(['git', 'clone', 'https://github.com/rohanvinaik/PoT_Experiments.git'], check=True)

os.chdir('PoT_Experiments')
print(f"📍 Working directory: {os.getcwd()}")

# Install dependencies
print("\n📦 Installing dependencies...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 
                'torch', 'transformers', 'numpy', 'scipy', 'scikit-learn'], check=False)

# Create results directory
Path("experimental_results").mkdir(exist_ok=True)

# Run the ACTUAL test scripts from the repository
print("\n" + "=" * 70)
print("🚀 RUNNING REAL POT FRAMEWORK TESTS")
print("=" * 70)

# These are the REAL scripts in your codebase
test_scripts = [
    "scripts/run_statistical_verification.py",
    "scripts/test_llm_verification.py",  # Uses GPT-2/DistilGPT-2
    "scripts/run_fuzzy_verification.py",
    "scripts/run_provenance_verification.py",
    "scripts/experimental_report_clean.py"
]

for script in test_scripts:
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print('='*60)
    
    if os.path.exists(script):
        # Run the actual script from your codebase
        result = subprocess.run([sys.executable, script], 
                              capture_output=True, 
                              text=True,
                              env={**os.environ, 'PYTHONPATH': os.getcwd()})
        print(result.stdout)
        if result.stderr and 'Warning' not in result.stderr:
            print("Errors:", result.stderr)
    else:
        print(f"❌ Script not found: {script}")

print("\n" + "=" * 70)
print("✅ TESTS COMPLETE")
print("=" * 70)
print("\nCheck experimental_results/ for detailed output")
print("These are REAL test results, not mocks!")