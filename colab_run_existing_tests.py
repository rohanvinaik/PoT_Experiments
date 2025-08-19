#!/usr/bin/env python3
"""
Run EXISTING PoT Test Suite in Google Colab
Simply executes your already-working test suite with better bandwidth for HF models
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime

print("🚀 RUNNING YOUR EXISTING POT TEST SUITE IN COLAB")
print("=" * 70)
print("Using Colab's bandwidth to download Hugging Face models")
print(f"Started at: {datetime.now()}")
print("=" * 70)

# Mount Google Drive
print("\n📁 Mounting Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted")
except:
    print("⚠️ Not in Colab")

# Setup working directory
work_dir = '/content/PoT_Experiments'
source_dir = '/content/drive/MyDrive/pot_to_upload'

# Copy your codebase from Drive
if os.path.exists(source_dir):
    print(f"\n📥 Copying your codebase from Google Drive...")
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    shutil.copytree(source_dir, work_dir)
    print("✅ Codebase copied")
elif not os.path.exists(work_dir):
    print("\n❌ Please upload your PoT codebase to:")
    print("   Google Drive: /My Drive/pot_to_upload/")
    print("   OR directly to: /content/PoT_Experiments/")
    sys.exit(1)

os.chdir(work_dir)
print(f"📂 Working directory: {work_dir}")

# Install dependencies
print("\n📦 Installing dependencies...")
subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=False)
print("✅ Dependencies installed")

# Set Python path
os.environ['PYTHONPATH'] = work_dir

print("\n" + "="*70)
print("🧪 RUNNING YOUR EXISTING TEST SUITE")
print("=" * 70)

# Run your existing test scripts
test_commands = [
    # Your actual test commands that work locally
    ('bash scripts/run_all_quick.sh', 'Quick validation tests'),
    ('bash scripts/run_all.sh', 'Full test suite'),
    ('python scripts/test_llm_verification.py', 'LLM verification (Mistral-7B vs GPT-2)'),
    ('python experimental_results/reliable_validation.py', 'Deterministic validation'),
    ('bash scripts/run_validation_report.sh', 'Generate validation report'),
]

results_dir = 'experimental_results'
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for command, description in test_commands:
    # Check if the script exists
    script_path = command.split()[1] if len(command.split()) > 1 else command
    
    if os.path.exists(script_path):
        print(f"\n### Running: {description} ###")
        print(f"Command: {command}")
        
        try:
            # Run the command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd=work_dir
            )
            
            # Save output
            output_file = f"{results_dir}/{script_path.replace('/', '_')}_{timestamp}.txt"
            with open(output_file, 'w') as f:
                f.write(f"Command: {command}\n")
                f.write(f"Description: {description}\n")
                f.write(f"Exit code: {result.returncode}\n")
                f.write("\n--- STDOUT ---\n")
                f.write(result.stdout)
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)
            
            if result.returncode == 0:
                print(f"✅ {description}: PASSED")
            else:
                print(f"⚠️ {description}: Exit code {result.returncode}")
                # Show last few lines of error if any
                if result.stderr:
                    err_lines = result.stderr.strip().split('\n')
                    for line in err_lines[-3:]:
                        print(f"  {line}")
            
        except subprocess.TimeoutExpired:
            print(f"⏱️ {description}: Timeout (>10 minutes)")
        except Exception as e:
            print(f"❌ {description}: Error - {e}")
    else:
        print(f"⚠️ Skipping {description} - script not found: {script_path}")

# Check for generated reports
print("\n" + "="*70)
print("📊 CHECKING FOR GENERATED REPORTS")
print("=" * 70)

report_locations = [
    'experimental_results/reports/',
    'test_results/',
    'outputs/',
    'reports/',
]

for location in report_locations:
    if os.path.exists(location):
        files = os.listdir(location)
        if files:
            print(f"\n📁 Found reports in {location}:")
            for f in files[:10]:  # Show first 10
                print(f"  - {f}")

# Look for specific result files
result_files = [
    'reliable_validation_results_*.json',
    'validation_results.json',
    'llm_result_*.json',
    'experimental_results/summary_*.txt',
    'test_results/validation_report_latest.md',
]

print("\n📄 Looking for result files...")
import glob
for pattern in result_files:
    matches = glob.glob(pattern, recursive=True)
    if matches:
        for match in matches:
            print(f"  ✅ Found: {match}")
            # Show file size to confirm it has content
            size = os.path.getsize(match)
            print(f"     Size: {size:,} bytes")

# Create a package of all results
print("\n📦 Creating results package...")
import zipfile

zip_path = f'/content/pot_results_{timestamp}.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add all results
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, work_dir)
            zipf.write(file_path, arcname)
    
    # Add any JSON result files
    for pattern in ['*.json', '**/*.json']:
        for file in glob.glob(pattern, recursive=True):
            if 'node_modules' not in file and '.git' not in file:
                zipf.write(file, os.path.relpath(file, work_dir))

print(f"✅ Results package created: {zip_path}")
print(f"   Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")

# Download the results
try:
    from google.colab import files
    files.download(zip_path)
    print("✅ Downloading results package...")
except:
    print(f"📥 Please download manually from: {zip_path}")

# Final summary
print("\n" + "="*70)
print("🎉 TEST SUITE EXECUTION COMPLETE!")
print("=" * 70)

print("\n✅ What happened:")
print("  1. Copied your working codebase to Colab")
print("  2. Installed all dependencies")  
print("  3. Ran your existing test suite")
print("  4. Collected all results and reports")
print("  5. Created downloadable package")

print("\n📊 Your tests should now have access to:")
print("  • Full Colab bandwidth for HF model downloads")
print("  • GPU acceleration (if using GPU runtime)")
print("  • No ISP throttling issues")

print("\n💡 Tips:")
print("  • Use GPU runtime in Colab for faster model inference")
print("  • Models are cached after first download")
print("  • Check /content/.cache/huggingface/ for cached models")

print("\n🔍 Check the downloaded results package for:")
print("  • Test outputs")
print("  • Validation reports")
print("  • JSON result files")
print("  • Any generated visualizations")