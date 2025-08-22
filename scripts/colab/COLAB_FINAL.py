#!/usr/bin/env python3
"""
FINAL COLAB RUNNER - HANDLES ALL EDGE CASES
============================================
This is the definitive version for Google Colab execution.
"""

import os
import sys
import subprocess
import json
import shutil
from datetime import datetime
from pathlib import Path

def setup_environment():
    """Setup clean environment and clone repository."""
    print("=" * 70)
    print("POT EXPERIMENTS - COMPLETE TEST SUITE")
    print("=" * 70)
    print("Using ONLY open models (GPT-2, DistilGPT-2)")
    print("NO AUTHENTICATION TOKENS REQUIRED")
    print("=" * 70)
    
    # Ensure we're in a valid directory
    try:
        os.chdir('/content')
    except:
        os.chdir(os.path.expanduser('~'))
    
    print(f"\n📍 Base directory: {os.getcwd()}")
    
    # Clean up old directory if exists
    if os.path.exists('PoT_Experiments'):
        print("📧 Cleaning up old directory...")
        shutil.rmtree('PoT_Experiments')
    
    # Clone repository
    print("📥 Cloning repository...")
    result = subprocess.run(
        ['git', 'clone', 'https://github.com/rohanvinaik/PoT_Experiments.git'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Failed to clone repository: {result.stderr}")
        return False
    
    # Change to repository directory
    os.chdir('PoT_Experiments')
    print(f"✅ Repository cloned to: {os.getcwd()}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    packages = ['torch', 'transformers', 'numpy', 'scipy', 'scikit-learn']
    for pkg in packages:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)
    print("✅ Dependencies installed")
    
    return True

def run_test(script_path, test_name):
    """Run a single test script."""
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print('='*60)
    
    if not os.path.exists(script_path):
        print(f"⚠️ Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, 'PYTHONPATH': os.getcwd()}
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        # Print errors (excluding warnings)
        if result.stderr:
            errors = [l for l in result.stderr.split('\n') 
                     if l and 'Warning' not in l and 'FutureWarning' not in l]
            if errors:
                print("Errors:", '\n'.join(errors))
        
        success = result.returncode == 0
        print(f"\n{'✅' if success else '❌'} {test_name} {'passed' if success else 'failed'}")
        return success
        
    except subprocess.TimeoutExpired:
        print(f"❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main execution function."""
    
    # Setup environment
    if not setup_environment():
        print("❌ Failed to setup environment")
        return 1
    
    # Check device
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n🖥️ Device: {device}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name()}")
    except:
        device = "cpu"
        print(f"\n🖥️ Device: {device} (PyTorch check failed)")
    
    # Create results directory
    Path("experimental_results").mkdir(exist_ok=True)
    
    # Define tests
    tests = [
        ("scripts/run_statistical_verification.py", "Statistical Identity"),
        ("scripts/test_llm_open_models_only.py", "LLM Open Models"),
        ("scripts/run_fuzzy_verification.py", "Fuzzy Hash"),
        ("scripts/run_provenance_verification.py", "Provenance"),
        ("scripts/experimental_report_clean.py", "Clean Reporting")
    ]
    
    # Run tests
    print("\n" + "=" * 70)
    print("🚀 RUNNING TEST SUITE")
    print("=" * 70)
    
    results = {}
    for script, name in tests:
        results[name] = run_test(script, name)
    
    # Generate summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nTests: {passed}/{total} passed ({(passed/total)*100:.0f}%)")
    
    print("\nResults:")
    for test, success in results.items():
        print(f"  {'✅' if success else '❌'} {test}")
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "environment": "Google Colab",
        "device": device,
        "models": ["gpt2", "distilgpt2"],
        "tests_total": total,
        "tests_passed": passed,
        "success_rate": passed/total if total > 0 else 0,
        "results": results
    }
    
    with open("experimental_results/colab_final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # List result files
    print("\n📁 Generated files:")
    exp_dir = Path("experimental_results")
    if exp_dir.exists():
        for f in sorted(exp_dir.glob("*.json"))[-5:]:
            print(f"  • {f.name}")
    
    # Final status
    print("\n" + "=" * 70)
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️ {total - passed} test(s) failed")
    
    print("\n✅ Execution complete")
    print("All tests used open models - no authentication required!")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())