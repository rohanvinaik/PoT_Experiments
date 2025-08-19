#!/usr/bin/env python3
"""
COMPLETE GOOGLE COLAB NOTEBOOK FOR POT EXPERIMENTS
===================================================
Copy this entire script into a Google Colab cell and run it.
It will handle everything automatically.
"""

# ============================================================================
# CELL 1: Complete Setup and Execution
# ============================================================================

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

print("=" * 70)
print("POT EXPERIMENTS - COMPLETE TEST SUITE")
print("=" * 70)
print("This suite uses ONLY open models (GPT-2, DistilGPT-2)")
print("NO AUTHENTICATION TOKENS REQUIRED")
print("=" * 70)

# 1. Clone or update repository
print("\n📥 Setting up PoT repository...")
if not os.path.exists('/content/PoT_Experiments'):
    subprocess.run(['git', 'clone', 'https://github.com/rohanvinaik/PoT_Experiments.git'], check=True)
else:
    subprocess.run(['git', '-C', '/content/PoT_Experiments', 'pull'], check=True)

os.chdir('/content/PoT_Experiments')
print(f"📍 Working directory: {os.getcwd()}")

# 2. Install dependencies
print("\n📦 Installing dependencies...")
packages = ['torch', 'transformers', 'numpy', 'scipy', 'scikit-learn']
for pkg in packages:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)
print("✅ Dependencies installed")

# 3. Check environment
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️ Device: {device}")

# 4. Create results directory
Path("experimental_results").mkdir(exist_ok=True)

# 5. Run all tests
print("\n" + "=" * 70)
print("🚀 RUNNING TEST SUITE")
print("=" * 70)

test_scripts = [
    ("scripts/run_statistical_verification.py", "Statistical Identity"),
    ("scripts/test_llm_open_models_only.py", "LLM Verification (Open Models)"),
    ("scripts/run_fuzzy_verification.py", "Fuzzy Hash"),
    ("scripts/run_provenance_verification.py", "Provenance Audit"),
    ("scripts/experimental_report_clean.py", "Clean Reporting")
]

results = {}
for script_path, test_name in test_scripts:
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"Script: {script_path}")
    print('='*60)
    
    if os.path.exists(script_path):
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=120
            )
            print(result.stdout)
            if result.stderr and "Warning" not in result.stderr:
                print("Errors:", result.stderr)
            results[test_name] = result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"❌ Test timed out")
            results[test_name] = False
        except Exception as e:
            print(f"❌ Error: {e}")
            results[test_name] = False
    else:
        print(f"❌ Script not found: {script_path}")
        results[test_name] = False

# 6. Generate summary
print("\n" + "=" * 70)
print("📊 FINAL SUMMARY")
print("=" * 70)

passed = sum(1 for v in results.values() if v)
total = len(results)

print(f"\nTests Run: {total}")
print(f"Tests Passed: {passed}")
print(f"Success Rate: {(passed/total)*100:.0f}%")

print("\nDetailed Results:")
for test, success in results.items():
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status} - {test}")

# 7. Save summary
summary = {
    "timestamp": datetime.now().isoformat(),
    "environment": "Google Colab",
    "device": device,
    "models": ["gpt2", "distilgpt2"],
    "tests_total": total,
    "tests_passed": passed,
    "success_rate": passed/total,
    "results": results
}

with open("experimental_results/colab_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# 8. Display key results
print("\n" + "=" * 70)
print("📁 KEY RESULTS FILES")
print("=" * 70)

exp_dir = Path("experimental_results")
if exp_dir.exists():
    json_files = sorted(exp_dir.glob("*.json"), key=os.path.getmtime, reverse=True)[:10]
    for jf in json_files:
        size_kb = jf.stat().st_size / 1024
        print(f"  • {jf.name} ({size_kb:.1f} KB)")

print("\n" + "=" * 70)
print("✅ VALIDATION COMPLETE")
print("=" * 70)

if passed == total:
    print("\n🎉 ALL TESTS PASSED!")
    print("The PoT framework is working correctly with open models.")
else:
    print(f"\n⚠️ {total - passed} test(s) failed.")
    print("Review the output above for details.")

print("\n📝 Evidence for your paper:")
print("  • experimental_results/*.json - Test metrics and results")
print("  • Statistical identity verification results")
print("  • LLM verification using only open models")
print("  • Fuzzy hash algorithm comparisons")
print("  • Merkle tree provenance proofs")