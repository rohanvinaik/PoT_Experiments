#!/usr/bin/env python3
"""
COMPLETE POT TEST SUITE FOR GOOGLE COLAB - UNICODE SAFE VERSION
===============================================================
Runs all validation scripts with proper encoding handling.
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
print("POT EXPERIMENTS - COMPLETE TEST SUITE (UNICODE SAFE)")
print("=" * 70)

# Mount Google Drive for persistent storage
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_MOUNTED = True
    print("✅ Google Drive mounted")
    
    # Create timestamped results directory in Drive
    DRIVE_RESULTS_DIR = f"/content/drive/MyDrive/PoT_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(DRIVE_RESULTS_DIR, exist_ok=True)
    print(f"📁 Results will be saved to: {DRIVE_RESULTS_DIR}")
except Exception as e:
    DRIVE_MOUNTED = False
    print(f"⚠️ Google Drive not mounted: {e}")
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

# Environment setup - disable color codes to prevent Unicode issues
env = os.environ.copy()
env['PYTHONPATH'] = POT_PATH
env['PYTHON'] = sys.executable
env['NO_COLOR'] = '1'  # Disable color codes
env['TERM'] = 'dumb'   # Use dumb terminal to avoid color codes
env['PYTHONIOENCODING'] = 'utf-8'

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

def run_script_safe(script_path, description):
    """Run a script with proper encoding handling"""
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
    output_path = os.path.join('/content', output_file)
    
    print(f"Running {script_path}...")
    
    try:
        # Use subprocess.run with proper encoding handling
        if script_path.endswith('.sh'):
            cmd = ['bash', script_path]
        else:
            cmd = [sys.executable, script_path]
            
        # Run with encoding handling and save output
        with open(output_path, 'w', encoding='utf-8', errors='replace') as log_file:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace invalid characters
                env=env,
                cwd=POT_PATH,
                timeout=1800  # 30 minute timeout
            )
            
            # Write output to both console and file
            output = process.stdout
            print(output)
            log_file.write(output)
            
        # Save log to Drive
        save_to_drive(output_path, output_file)
        
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print(f"\n✅ {description} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"\n⚠️ {description} failed with code {process.returncode} after {elapsed:.1f}s")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⚠️ {description} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"\n❌ {description} failed with error: {e}")
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

if run_script_safe('scripts/run_all_quick.sh', 'Quick Validation Suite'):
    results['tests_run'].append('run_all_quick')
    results['success_count'] += 1
results['total_count'] += 1

# Save results after each phase
if os.path.exists('experimental_results'):
    save_to_drive('experimental_results', 'experimental_results_phase1')

print("\n" + "=" * 70)
print("PHASE 2: STANDARD VALIDATION")
print("=" * 70)

if run_script_safe('scripts/run_all.sh', 'Standard Test Suite'):
    results['tests_run'].append('run_all')
    results['success_count'] += 1
results['total_count'] += 1

# Save results after standard tests
if os.path.exists('experimental_results'):
    save_to_drive('experimental_results', 'experimental_results_phase2')

print("\n" + "=" * 70)
print("PHASE 3: SPECIALIZED TESTS")
print("=" * 70)

# Test individual Python scripts
specialized_tests = [
    ('scripts/test_llm_verification.py', 'LLM Verification'),
    ('scripts/test_gpt2_variants.py', 'GPT-2 Variants'),
]

for script, desc in specialized_tests:
    if os.path.exists(script):
        if run_script_safe(script, desc):
            results['tests_run'].append(script)
            results['success_count'] += 1
        results['total_count'] += 1

print("\n" + "=" * 70)
print("PHASE 4: RESULTS PACKAGING")
print("=" * 70)

# Save test_results if it exists
if os.path.exists('test_results'):
    save_to_drive('test_results', 'test_results')

# Create comprehensive ZIP file
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
                if file.endswith(('.md', '.json', '.txt', '.log')):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, POT_PATH)
                    zipf.write(file_path, arcname)
                    print(f"  Added: {arcname}")
    
    # Add any log files from /content
    for file in os.listdir('/content'):
        if file.endswith('.log') and 'run_all' in file:
            file_path = f"/content/{file}"
            zipf.write(file_path, f"logs/{file}")
            print(f"  Added: logs/{file}")

print(f"\n✅ Results archive created: {zip_path}")
print(f"   Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")

# Save ZIP to Drive
if DRIVE_MOUNTED:
    save_to_drive(zip_path, zip_filename)

# Download the ZIP file
try:
    from google.colab import files
    print("\n📥 Downloading results archive...")
    files.download(zip_path)
    print("✅ Download started!")
except:
    print("⚠️ Could not initiate download, but file is saved to Drive")

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

# Show key results
print("\n📄 Key Results for Paper:")
if os.path.exists('experimental_results'):
    import json
    json_files = [f for f in os.listdir('experimental_results') if f.endswith('.json')]
    for jf in sorted(json_files)[-3:]:  # Last 3 JSON files
        try:
            with open(f'experimental_results/{jf}', 'r') as f:
                data = json.load(f)
                print(f"\n  📊 {jf}:")
                if 'summary' in data:
                    s = data['summary']
                    print(f"     Success rate: {s.get('success_rate', 'N/A')}")
                    print(f"     Tests passed: {s.get('passed_tests', 'N/A')}/{s.get('total_tests', 'N/A')}")
        except:
            print(f"  📊 {jf}: (could not parse)")

if DRIVE_MOUNTED:
    print(f"""
📁 Results Saved To:
-------------------
Google Drive: {DRIVE_RESULTS_DIR}
ZIP File: {zip_filename} (downloaded + in Drive)

Access anytime: Google Drive > {os.path.basename(DRIVE_RESULTS_DIR)}
""")

print("\n" + "=" * 70)
if success_rate >= 80:
    print("✅ VALIDATION SUCCESSFUL - Results ready for paper")
else:
    print("⚠️ Some tests failed - review results in archive")
print("=" * 70)

# Save final summary
import json
summary_path = '/content/final_test_summary.json'
with open(summary_path, 'w') as f:
    json.dump(results, f, indent=2)
save_to_drive(summary_path, 'final_test_summary.json')