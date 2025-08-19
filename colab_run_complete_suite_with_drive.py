#!/usr/bin/env python3
"""
COMPLETE POT TEST SUITE FOR GOOGLE COLAB WITH GOOGLE DRIVE SAVING
=================================================================
Runs all validation scripts and saves results to Google Drive for persistence.
"""

import os
import sys
import subprocess
import time
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

print("=" * 70)
print("POT EXPERIMENTS - COMPLETE TEST SUITE WITH DRIVE SAVING")
print("=" * 70)

# Mount Google Drive for persistent storage
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_MOUNTED = True
    print("✅ Google Drive mounted")
    
    # Create results directory in Drive
    DRIVE_RESULTS_DIR = f"/content/drive/MyDrive/PoT_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(DRIVE_RESULTS_DIR, exist_ok=True)
    print(f"📁 Results will be saved to: {DRIVE_RESULTS_DIR}")
except:
    DRIVE_MOUNTED = False
    print("⚠️ Google Drive not mounted - results will be temporary")
    DRIVE_RESULTS_DIR = None

# Setup paths
WORK_DIR = '/content'
POT_PATH = '/content/PoT_Experiments'

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
packages = ['numpy', 'torch', 'transformers', 'scipy', 'pytest', 'matplotlib', 'pandas', 'tqdm', 'accelerate']
for pkg in packages:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg])
print("✅ Dependencies installed")

# Check device
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️ Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Environment setup
env = os.environ.copy()
env['PYTHONPATH'] = POT_PATH
env['PYTHON'] = sys.executable
env['TERM'] = 'xterm-256color'

def save_to_drive(src_path, dest_name):
    """Copy file to Google Drive if mounted"""
    if DRIVE_MOUNTED and DRIVE_RESULTS_DIR and os.path.exists(src_path):
        dest_path = os.path.join(DRIVE_RESULTS_DIR, dest_name)
        try:
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dest_path)
            print(f"  💾 Saved to Drive: {dest_name}")
            return True
        except Exception as e:
            print(f"  ⚠️ Failed to save {dest_name}: {e}")
    return False

def run_script(script_path, description, save_output=True):
    """Run a script and save output to Drive"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_path):
        print(f"⚠️ Script not found: {script_path}")
        return False
    
    # Make executable
    subprocess.run(['chmod', '+x', script_path])
    
    start_time = time.time()
    script_name = os.path.basename(script_path).replace('.sh', '').replace('.py', '')
    output_file = f"{script_name}_{datetime.now().strftime('%H%M%S')}.log"
    
    print(f"Running {script_path}...\n")
    
    # Execute and capture output
    if save_output:
        output_path = os.path.join('/content', output_file)
        with open(output_path, 'w') as log_file:
            process = subprocess.Popen(
                ['bash', script_path] if script_path.endswith('.sh') else [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=POT_PATH,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream and save output
            for line in process.stdout:
                print(line, end='')
                log_file.write(line)
            
            return_code = process.wait()
            
        # Save log to Drive
        save_to_drive(output_path, output_file)
    else:
        # Just run without saving
        process = subprocess.Popen(
            ['bash', script_path] if script_path.endswith('.sh') else [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=POT_PATH,
            bufsize=1,
            universal_newlines=True
        )
        
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

# Track results
results = {
    'timestamp': datetime.now().isoformat(),
    'device': str(device),
    'drive_results_dir': DRIVE_RESULTS_DIR,
    'tests_run': [],
    'success_count': 0,
    'total_count': 0
}

print("\n" + "=" * 70)
print("PHASE 1: QUICK VALIDATION")
print("=" * 70)

if run_script('scripts/run_all_quick.sh', 'Quick Validation Suite'):
    results['tests_run'].append('run_all_quick')
    results['success_count'] += 1
results['total_count'] += 1

print("\n" + "=" * 70)
print("PHASE 2: STANDARD VALIDATION")
print("=" * 70)

if run_script('scripts/run_all.sh', 'Standard Test Suite'):
    results['tests_run'].append('run_all')
    results['success_count'] += 1
results['total_count'] += 1

# Copy experimental_results directory after each major test
if os.path.exists('experimental_results'):
    save_to_drive('experimental_results', 'experimental_results')

print("\n" + "=" * 70)
print("PHASE 3: SPECIALIZED TESTS")
print("=" * 70)

specialized_tests = [
    ('scripts/test_llm_verification.py', 'LLM Verification'),
    ('scripts/test_gpt2_variants.py', 'GPT-2 Variants'),
]

for script, desc in specialized_tests:
    if os.path.exists(script):
        if run_script(script, desc):
            results['tests_run'].append(script)
            results['success_count'] += 1
        results['total_count'] += 1

print("\n" + "=" * 70)
print("PHASE 4: REPORTING")
print("=" * 70)

if os.path.exists('scripts/run_validation_report.sh'):
    if run_script('scripts/run_validation_report.sh', 'Validation Report'):
        results['success_count'] += 1
    results['total_count'] += 1

# Save test_results directory
if os.path.exists('test_results'):
    save_to_drive('test_results', 'test_results')

print("\n" + "=" * 70)
print("PHASE 5: RESULTS PACKAGING")
print("=" * 70)

# Create a ZIP file with all results
zip_filename = f"PoT_Complete_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
zip_path = f"/content/{zip_filename}"

print(f"\n📦 Creating results archive: {zip_filename}")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add experimental_results
    if os.path.exists('experimental_results'):
        for root, dirs, files in os.walk('experimental_results'):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, POT_PATH)
                zipf.write(file_path, arcname)
                print(f"  Added: {arcname}")
    
    # Add test_results
    if os.path.exists('test_results'):
        for root, dirs, files in os.walk('test_results'):
            for file in files:
                if file.endswith(('.md', '.json', '.txt')):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, POT_PATH)
                    zipf.write(file_path, arcname)
                    print(f"  Added: {arcname}")

print(f"\n✅ Results archive created: {zip_path}")
print(f"   Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")

# Save ZIP to Drive
if DRIVE_MOUNTED:
    save_to_drive(zip_path, zip_filename)
    print(f"\n📁 Complete results saved to Google Drive:")
    print(f"   {DRIVE_RESULTS_DIR}")

# Generate download link for Colab
from google.colab import files
print("\n📥 Download results archive:")
files.download(zip_path)

# Final summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

success_rate = (results['success_count'] / results['total_count'] * 100) if results['total_count'] > 0 else 0

print(f"""
📊 Test Suite Results:
---------------------
Total Tests: {results['total_count']}
Successful: {results['success_count']}
Failed: {results['total_count'] - results['success_count']}
Success Rate: {success_rate:.1f}%

Tests Completed:
{chr(10).join('  ✅ ' + t for t in results['tests_run'])}
""")

if DRIVE_MOUNTED:
    print(f"""
📁 Results Location:
-------------------
Google Drive: {DRIVE_RESULTS_DIR}
  - experimental_results/ (test outputs)
  - test_results/ (validation reports)
  - *.log files (execution logs)
  - {zip_filename} (complete archive)

To access your results:
1. Go to Google Drive
2. Navigate to: {os.path.basename(DRIVE_RESULTS_DIR)}
3. All test results are permanently saved there
""")

# Save summary
import json
summary_path = '/content/test_summary.json'
with open(summary_path, 'w') as f:
    json.dump(results, f, indent=2)
save_to_drive(summary_path, 'test_summary.json')

print("\n" + "=" * 70)
if success_rate >= 80:
    print("✅ VALIDATION SUCCESSFUL - Results ready for paper")
else:
    print("⚠️ Some tests failed - review results in Drive")
print("=" * 70)