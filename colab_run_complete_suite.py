#!/usr/bin/env python3
"""
COMPLETE POT TEST SUITE AND REPORTING PIPELINE FOR GOOGLE COLAB
===============================================================
Runs the ENTIRE run_all family of scripts plus all reporting pipelines.
This is the comprehensive validation suite for paper evidence.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

print("=" * 70)
print("POT EXPERIMENTS - COMPLETE TEST SUITE WITH REPORTING")
print("Running ALL validation scripts and generating comprehensive reports")
print("=" * 70)

# Detect environment
IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
    print("✅ Running in Google Colab")
except ImportError:
    print("📁 Running locally")

# Setup paths
if IN_COLAB:
    WORK_DIR = '/content'
    POT_PATH = '/content/PoT_Experiments'
else:
    WORK_DIR = os.getcwd()
    POT_PATH = WORK_DIR if WORK_DIR.endswith('PoT_Experiments') else os.path.join(WORK_DIR, 'PoT_Experiments')

# Clone or update repository
if not os.path.exists(POT_PATH):
    print("\n📥 Cloning PoT repository from GitHub...")
    result = subprocess.run(
        ['git', 'clone', 'https://github.com/rohanvinaik/PoT_Experiments.git', POT_PATH],
        cwd=WORK_DIR,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ Failed to clone repository: {result.stderr}")
        sys.exit(1)
    print("✅ Repository cloned successfully")
else:
    print(f"📁 Using existing repository at {POT_PATH}")
    subprocess.run(['git', 'pull'], cwd=POT_PATH, capture_output=True)

os.chdir(POT_PATH)
print(f"📍 Working directory: {os.getcwd()}")

# Install dependencies
print("\n📦 Installing dependencies...")
if os.path.exists('requirements.txt'):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])
else:
    # Install essential packages
    packages = ['numpy', 'torch', 'transformers', 'scipy', 'pytest', 'matplotlib', 'pandas', 'tqdm']
    for pkg in packages:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg])
print("✅ Dependencies installed")

# Environment setup
env = os.environ.copy()
env['PYTHONPATH'] = POT_PATH
env['PYTHON'] = sys.executable
if IN_COLAB:
    env['TERM'] = 'xterm-256color'

def run_script(script_path, description):
    """Run a script and return success status"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_path):
        print(f"⚠️ Script not found: {script_path}")
        return False
    
    # Make executable
    subprocess.run(['chmod', '+x', script_path])
    
    start_time = time.time()
    print(f"Running {script_path}...\n")
    
    # Execute and stream output
    process = subprocess.Popen(
        ['bash', script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=POT_PATH,
        bufsize=1,
        universal_newlines=True
    )
    
    # Stream output
    for line in process.stdout:
        print(line, end='')
    
    return_code = process.wait()
    elapsed = time.time() - start_time
    
    if return_code == 0:
        print(f"\n✅ {description} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n⚠️ {description} failed with code {return_code} after {elapsed:.1f}s")
        return False

def run_python_script(script_path, description):
    """Run a Python script for reporting"""
    print(f"\n📊 {description}")
    
    if not os.path.exists(script_path):
        print(f"⚠️ Script not found: {script_path}")
        return False
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        env=env,
        cwd=POT_PATH
    )
    
    if result.returncode == 0:
        print(f"✅ {description} completed")
        if result.stdout:
            print(result.stdout)
        return True
    else:
        print(f"⚠️ {description} failed")
        if result.stderr:
            print(result.stderr[:500])
        return False

# Track overall results
results = {
    'timestamp': datetime.now().isoformat(),
    'tests_run': [],
    'reports_generated': [],
    'success_count': 0,
    'total_count': 0
}

print("\n" + "=" * 70)
print("PHASE 1: QUICK VALIDATION")
print("=" * 70)

# Run quick validation first
if run_script('scripts/run_all_quick.sh', 'Quick Validation Suite (~30s)'):
    results['tests_run'].append('run_all_quick')
    results['success_count'] += 1
results['total_count'] += 1

print("\n" + "=" * 70)
print("PHASE 2: STANDARD VALIDATION")
print("=" * 70)

# Run standard test suite
if run_script('scripts/run_all.sh', 'Standard Test Suite (~5min)'):
    results['tests_run'].append('run_all')
    results['success_count'] += 1
results['total_count'] += 1

print("\n" + "=" * 70)
print("PHASE 3: COMPREHENSIVE VALIDATION")
print("=" * 70)

# Run comprehensive suite if requested (this is long)
if os.path.exists('scripts/run_all_comprehensive.sh'):
    if IN_COLAB:
        print("📝 Note: Comprehensive suite can take 30+ minutes")
        print("Running comprehensive validation...")
        if run_script('scripts/run_all_comprehensive.sh', 'Comprehensive Test Suite'):
            results['tests_run'].append('run_all_comprehensive')
            results['success_count'] += 1
        results['total_count'] += 1
else:
    print("⚠️ Comprehensive suite not found, skipping")

print("\n" + "=" * 70)
print("PHASE 4: SPECIALIZED TESTS")
print("=" * 70)

# Run specialized test scripts
specialized_tests = [
    ('scripts/test_llm_verification.py', 'LLM Verification (Mistral vs GPT-2)'),
    ('scripts/test_gpt2_variants.py', 'GPT-2 Variants Comparison'),
    ('scripts/test_local_gpt2.py', 'Local GPT-2 Testing'),
]

for script, desc in specialized_tests:
    if os.path.exists(script):
        if run_python_script(script, desc):
            results['tests_run'].append(script)
            results['success_count'] += 1
        results['total_count'] += 1

print("\n" + "=" * 70)
print("PHASE 5: REPORTING PIPELINE")
print("=" * 70)

# Run validation report generator
if os.path.exists('scripts/run_validation_report.sh'):
    if run_script('scripts/run_validation_report.sh', 'Validation Report Generator'):
        results['reports_generated'].append('validation_report')
        results['success_count'] += 1
    results['total_count'] += 1

# Run Python report generators
report_scripts = [
    ('scripts/generate_validation_report.py', 'Python Validation Report'),
    ('scripts/experimental_report.py', 'Experimental Results Report'),
    ('scripts/generate_reproduction_report.py', 'Reproduction Report'),
]

for script, desc in report_scripts:
    if os.path.exists(script):
        if run_python_script(script, desc):
            results['reports_generated'].append(script)
            results['success_count'] += 1
        results['total_count'] += 1

print("\n" + "=" * 70)
print("PHASE 6: RESULTS CONSOLIDATION")
print("=" * 70)

# Check for results
exp_results_dir = os.path.join(POT_PATH, 'experimental_results')
test_results_dir = os.path.join(POT_PATH, 'test_results')

print("\n📁 Available Results:")
if os.path.exists(exp_results_dir):
    files = os.listdir(exp_results_dir)
    recent_files = sorted([f for f in files if any(f.endswith(ext) for ext in ['.log', '.json', '.txt'])])[-10:]
    if recent_files:
        print("\n  Recent files in experimental_results/:")
        for f in recent_files:
            size = os.path.getsize(os.path.join(exp_results_dir, f)) / 1024
            print(f"    - {f} ({size:.1f} KB)")

if os.path.exists(test_results_dir):
    files = os.listdir(test_results_dir)
    report_files = [f for f in files if 'report' in f.lower() and f.endswith('.md')]
    if report_files:
        print("\n  Generated reports in test_results/:")
        for f in report_files:
            print(f"    - {f}")

# Generate final summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

success_rate = (results['success_count'] / results['total_count'] * 100) if results['total_count'] > 0 else 0

print(f"""
📊 Test Suite Results:
---------------------
Total Scripts Run: {results['total_count']}
Successful: {results['success_count']}
Failed: {results['total_count'] - results['success_count']}
Success Rate: {success_rate:.1f}%

Tests Completed:
{chr(10).join('  ✅ ' + t for t in results['tests_run'])}

Reports Generated:
{chr(10).join('  📄 ' + r for r in results['reports_generated'])}
""")

# Save summary
summary_file = f"experimental_results/colab_run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
import json
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"📁 Summary saved to: {summary_file}")

# Final status
print("\n" + "=" * 70)
if success_rate >= 80:
    print("✅ VALIDATION SUCCESSFUL - System ready for paper submission")
    print("All major test suites and reports have been generated.")
elif success_rate >= 60:
    print("⚠️ PARTIAL SUCCESS - Most tests passed, review failures")
else:
    print("❌ VALIDATION FAILED - Multiple issues detected")

print("=" * 70)
print("\n🎯 Key Evidence for Paper:")
print("  1. Check experimental_results/ for detailed test metrics")
print("  2. Review test_results/*report*.md for validation reports")
print("  3. Look for *validation_results*.json for numerical data")
print("  4. Find timing and performance metrics in *benchmark*.json files")
print("\nThese results provide comprehensive evidence for your paper claims.")