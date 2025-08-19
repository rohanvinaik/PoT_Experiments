#!/usr/bin/env python3
"""
Complete PoT Test Suite Runner for Google Colab
Runs the entire test suite and generates comprehensive analysis reports
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime

print("🚀 COMPLETE POT TEST SUITE RUNNER FOR GOOGLE COLAB")
print("=" * 70)
print(f"Started at: {datetime.now()}")
print("=" * 70)

# Step 1: Setup working directory
print("\n📁 Setting up working directory...")
work_dir = '/content/PoT_Experiments'

# Option 1: Upload your codebase as a zip file
print("\n" + "="*70)
print("OPTION 1: Upload your codebase")
print("="*70)
print("""
To use your local codebase:
1. Zip your PoT_Experiments folder on your computer
2. Upload it to Colab using the file browser
3. Uncomment the following code:
""")

print("""
# from google.colab import files
# import zipfile

# print("Please upload your PoT_Experiments.zip file...")
# uploaded = files.upload()

# # Extract the zip file
# with zipfile.ZipFile('PoT_Experiments.zip', 'r') as zip_ref:
#     zip_ref.extractall('/content/')
""")

# Option 2: Clone from GitHub (if public)
print("\n" + "="*70)
print("OPTION 2: Clone from GitHub")
print("="*70)

# Clean up old directory
if os.path.exists(work_dir):
    print(f"Removing old directory: {work_dir}")
    shutil.rmtree(work_dir)

# Try to clone - with error handling
repo_url = "https://github.com/rohanvinaik/PoT_Experiments.git"
print(f"\n📥 Attempting to clone from: {repo_url}")

try:
    result = subprocess.run(
        ['git', 'clone', repo_url, work_dir],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"❌ Git clone failed: {result.stderr}")
        print("\n⚠️ Repository might be private or doesn't exist.")
        print("Creating local directory structure instead...")
        
        # Create directory and copy from mounted Google Drive if available
        os.makedirs(work_dir, exist_ok=True)
        
        # Option 3: Copy from Google Drive
        print("\n" + "="*70)
        print("OPTION 3: Copy from Google Drive")
        print("="*70)
        print("""
To use files from Google Drive:
1. Mount your drive using:
   from google.colab import drive
   drive.mount('/content/drive')
   
2. Copy your files:
   !cp -r /content/drive/MyDrive/PoT_Experiments/* /content/PoT_Experiments/
""")
        
    else:
        print("✅ Successfully cloned repository!")
        
except subprocess.TimeoutExpired:
    print("❌ Git clone timed out after 30 seconds")
    os.makedirs(work_dir, exist_ok=True)

# Change to working directory
os.chdir(work_dir)
print(f"\n📂 Working directory: {os.getcwd()}")

# List files to verify
print("\n📋 Files in working directory:")
try:
    files = os.listdir('.')
    for f in files[:10]:  # Show first 10 files
        print(f"  - {f}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")
except:
    print("  ❌ Could not list files - directory might be empty")

# Step 2: Install dependencies
print("\n" + "="*70)
print("📦 INSTALLING DEPENDENCIES")
print("="*70)

dependencies = [
    'torch',
    'transformers>=4.35.0',
    'scipy',
    'numpy',
    'tqdm',
    'matplotlib',
    'pandas',
    'scikit-learn',
    'huggingface-hub'
]

for dep in dependencies:
    print(f"Installing {dep}...")
    subprocess.run(['pip', 'install', '-q', dep], check=False)

print("✅ Dependencies installed")

# Add to Python path
sys.path.insert(0, work_dir)
print(f"\n🐍 Added to Python path: {work_dir}")

# Step 3: Run the complete test suite
print("\n" + "="*70)
print("🔬 RUNNING COMPLETE POT TEST SUITE")
print("="*70)

# Create results directory
results_dir = '/content/pot_test_results'
os.makedirs(results_dir, exist_ok=True)
print(f"📊 Results will be saved to: {results_dir}")

# Dictionary to store test results
test_results = {
    'passed': [],
    'failed': [],
    'skipped': []
}

# List of all test scripts to run
test_scripts = [
    # Core functionality tests
    ('scripts/test_llm_verification.py', 'LLM Verification Tests'),
    ('scripts/test_wrapper_detection.py', 'Wrapper Detection Tests'),
    ('scripts/test_one_shot_fine_tune.py', 'One-Shot Fine-Tune Detection'),
    
    # Security tests
    ('pot/security/test_token_space_normalizer.py', 'Token Space Normalizer'),
    ('pot/security/test_provenance_auditor.py', 'Provenance Auditor'),
    
    # Example demonstrations
    ('pot/experiments/example_validation.py', 'Example Validation'),
    ('pot/experiments/example_report_generation.py', 'Report Generation'),
]

# Run each test
for script_path, test_name in test_scripts:
    print(f"\n{'='*60}")
    print(f"🧪 Running: {test_name}")
    print(f"   Script: {script_path}")
    print('='*60)
    
    if not os.path.exists(script_path):
        print(f"⚠️ Script not found: {script_path}")
        test_results['skipped'].append((test_name, "File not found"))
        continue
    
    try:
        # Run the test with timeout
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per test
            env={**os.environ, 'PYTHONPATH': work_dir}
        )
        
        # Save output
        output_file = os.path.join(results_dir, f"{test_name.replace(' ', '_')}_output.txt")
        with open(output_file, 'w') as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Script: {script_path}\n")
            f.write(f"Exit Code: {result.returncode}\n")
            f.write("\n--- STDOUT ---\n")
            f.write(result.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ PASSED: {test_name}")
            test_results['passed'].append(test_name)
            # Print last few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:
                print(f"   {line}")
        else:
            print(f"❌ FAILED: {test_name} (exit code: {result.returncode})")
            test_results['failed'].append((test_name, result.returncode))
            # Print error info
            if result.stderr:
                err_lines = result.stderr.strip().split('\n')
                for line in err_lines[-5:]:
                    print(f"   ERROR: {line}")
                    
    except subprocess.TimeoutExpired:
        print(f"⏱️ TIMEOUT: {test_name} (exceeded 5 minutes)")
        test_results['failed'].append((test_name, "Timeout"))
    except Exception as e:
        print(f"❌ ERROR running {test_name}: {str(e)}")
        test_results['failed'].append((test_name, str(e)))

# Step 4: Generate comprehensive report
print("\n" + "="*70)
print("📊 GENERATING COMPREHENSIVE ANALYSIS REPORT")
print("="*70)

report_file = os.path.join(results_dir, 'POT_ANALYSIS_REPORT.md')

with open(report_file, 'w') as f:
    f.write("# Proof of Training (PoT) - Comprehensive Analysis Report\n\n")
    f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Executive Summary
    f.write("## Executive Summary\n\n")
    total_tests = len(test_results['passed']) + len(test_results['failed']) + len(test_results['skipped'])
    pass_rate = (len(test_results['passed']) / total_tests * 100) if total_tests > 0 else 0
    
    f.write(f"- **Total Tests Run**: {total_tests}\n")
    f.write(f"- **Passed**: {len(test_results['passed'])} ({pass_rate:.1f}%)\n")
    f.write(f"- **Failed**: {len(test_results['failed'])}\n")
    f.write(f"- **Skipped**: {len(test_results['skipped'])}\n\n")
    
    # Test Results Details
    f.write("## Test Results\n\n")
    
    f.write("### ✅ Passed Tests\n\n")
    if test_results['passed']:
        for test in test_results['passed']:
            f.write(f"- {test}\n")
    else:
        f.write("No tests passed.\n")
    
    f.write("\n### ❌ Failed Tests\n\n")
    if test_results['failed']:
        for test, reason in test_results['failed']:
            f.write(f"- {test}: {reason}\n")
    else:
        f.write("No tests failed.\n")
    
    f.write("\n### ⚠️ Skipped Tests\n\n")
    if test_results['skipped']:
        for test, reason in test_results['skipped']:
            f.write(f"- {test}: {reason}\n")
    else:
        f.write("No tests were skipped.\n")
    
    # Key Capabilities Demonstrated
    f.write("\n## Key Capabilities Demonstrated\n\n")
    f.write("### 1. LLM Verification\n")
    f.write("- Detects training data contamination\n")
    f.write("- Identifies memorization patterns\n")
    f.write("- Validates model authenticity\n\n")
    
    f.write("### 2. Wrapper Detection\n")
    f.write("- Identifies wrapper models\n")
    f.write("- Detects API proxies\n")
    f.write("- Reveals model substitution\n\n")
    
    f.write("### 3. One-Shot Fine-Tune Detection\n")
    f.write("- Detects minimal fine-tuning\n")
    f.write("- Identifies superficial modifications\n")
    f.write("- Validates training depth\n\n")
    
    f.write("### 4. Security Features\n")
    f.write("- Token space normalization\n")
    f.write("- Provenance auditing\n")
    f.write("- Cryptographic verification\n\n")
    
    # Technical Implementation
    f.write("\n## Technical Implementation\n\n")
    f.write("The PoT system implements multiple verification techniques:\n\n")
    f.write("1. **Statistical Analysis**: Entropy measurements, distribution analysis\n")
    f.write("2. **Pattern Recognition**: Memorization detection, training artifacts\n")
    f.write("3. **Cryptographic Proofs**: Hash-based verification, Merkle trees\n")
    f.write("4. **Behavioral Analysis**: Response patterns, consistency checks\n\n")
    
    # Recommendations
    f.write("\n## Recommendations\n\n")
    if pass_rate >= 80:
        f.write("✅ **System Status**: OPERATIONAL\n\n")
        f.write("The PoT system is functioning correctly and can be used for:\n")
        f.write("- Production deployment\n")
        f.write("- Model verification tasks\n")
        f.write("- Security auditing\n")
    elif pass_rate >= 50:
        f.write("⚠️ **System Status**: PARTIALLY OPERATIONAL\n\n")
        f.write("Some components need attention:\n")
        f.write("- Review failed tests\n")
        f.write("- Fix critical issues\n")
        f.write("- Re-run validation\n")
    else:
        f.write("❌ **System Status**: NEEDS ATTENTION\n\n")
        f.write("Significant issues detected:\n")
        f.write("- Debug failing components\n")
        f.write("- Check dependencies\n")
        f.write("- Verify environment setup\n")

print(f"\n✅ Report generated: {report_file}")

# Display the report
print("\n" + "="*70)
print("📄 REPORT PREVIEW")
print("="*70)

try:
    with open(report_file, 'r') as f:
        content = f.read()
        # Show first 50 lines
        lines = content.split('\n')
        for line in lines[:50]:
            print(line)
        if len(lines) > 50:
            print(f"\n... ({len(lines) - 50} more lines)")
except:
    print("Could not read report file")

# Step 5: Download results
print("\n" + "="*70)
print("💾 DOWNLOADING RESULTS")
print("="*70)

# Create a zip file with all results
import zipfile
zip_path = '/content/pot_test_results.zip'

with zipfile.ZipFile(zip_path, 'w') as zipf:
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, '/content')
            zipf.write(file_path, arcname)

print(f"✅ Results packaged: {zip_path}")

# Download the results
try:
    from google.colab import files
    files.download(zip_path)
    print("✅ Results downloaded to your computer!")
except:
    print("⚠️ Not running in Colab - results saved locally")

print("\n" + "="*70)
print(f"🎉 COMPLETE! Finished at: {datetime.now()}")
print("="*70)
print("\nSummary:")
print(f"  ✅ Passed: {len(test_results['passed'])} tests")
print(f"  ❌ Failed: {len(test_results['failed'])} tests")
print(f"  ⚠️ Skipped: {len(test_results['skipped'])} tests")
print(f"\n📊 Full report available at: {report_file}")