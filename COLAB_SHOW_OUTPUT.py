#!/usr/bin/env python3
"""
RUN THIS IN GOOGLE COLAB - SHOWS ALL OUTPUT
"""

import os
import subprocess
import sys

# Setup
os.chdir('/content')
subprocess.run(['rm', '-rf', 'PoT_Experiments'])
subprocess.run(['git', 'clone', 'https://github.com/rohanvinaik/PoT_Experiments.git'])
os.chdir('/content/PoT_Experiments')

# Install dependencies
print("Installing dependencies...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch', 'transformers', 'numpy', 'scipy', 'scikit-learn'])

print("=" * 70)
print("RUNNING POT FRAMEWORK TESTS")
print("=" * 70)

# Run tests and SHOW OUTPUT
test_scripts = [
    ('scripts/run_statistical_verification.py', 'STATISTICAL VERIFICATION'),
    ('scripts/test_llm_verification.py', 'LLM VERIFICATION'),
    ('scripts/run_fuzzy_verification.py', 'FUZZY HASH VERIFICATION'),
    ('scripts/run_provenance_verification.py', 'PROVENANCE AUDIT'),
    ('scripts/experimental_report_clean.py', 'COMPREHENSIVE REPORT')
]

for script, name in test_scripts:
    print(f"\n{'='*70}")
    print(f"{name}")
    print('='*70)
    
    if os.path.exists(script):
        # Run and capture output
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            env={**os.environ, 'PYTHONPATH': os.getcwd()}
        )
        
        # PRINT THE ACTUAL OUTPUT
        print(result.stdout)
        
        if result.stderr:
            # Only show real errors, not warnings
            errors = [l for l in result.stderr.split('\n') 
                     if l and 'Warning' not in l and 'FutureWarning' not in l]
            if errors:
                print("ERRORS:", '\n'.join(errors))
    else:
        print(f"Script not found: {script}")

# Show generated files
print("\n" + "="*70)
print("GENERATED RESULTS")
print("="*70)

import json
from pathlib import Path

exp_dir = Path("experimental_results")
if exp_dir.exists():
    for jf in sorted(exp_dir.glob("*.json"))[-5:]:
        print(f"\n📄 {jf.name}:")
        try:
            with open(jf) as f:
                data = json.load(f)
                print(json.dumps(data, indent=2)[:1000])  # Show first 1000 chars
        except:
            pass

print("\n✅ COMPLETE")