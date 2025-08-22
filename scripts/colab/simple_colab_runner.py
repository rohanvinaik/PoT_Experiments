#!/usr/bin/env python3
"""
SIMPLE RELIABLE COLAB RUNNER FOR POT TESTS
==========================================
Runs tests and ensures results are saved to accessible locations.
"""

import os
import sys
import subprocess
import shutil
import zipfile
from datetime import datetime

print("=" * 70)
print("POT EXPERIMENTS - SIMPLE RELIABLE RUNNER")
print("=" * 70)

# Step 1: Setup repository
POT_PATH = '/content/PoT_Experiments'
if not os.path.exists(POT_PATH):
    print("📥 Cloning repository...")
    subprocess.run(['git', 'clone', 'https://github.com/rohanvinaik/PoT_Experiments.git', POT_PATH])
    
os.chdir(POT_PATH)
print(f"📍 Working in: {os.getcwd()}")

# Step 2: Install basic dependencies
print("\n📦 Installing dependencies...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 
                'numpy', 'torch', 'transformers', 'scipy', 'pytest'])

# Step 3: Set up environment
env = os.environ.copy()
env['PYTHONPATH'] = POT_PATH
env['PYTHON'] = sys.executable
env['NO_COLOR'] = '1'

# Step 4: Run the main test script
print("\n🚀 Running main test suite...")
print("This may take several minutes...")

try:
    # Run run_all.sh with timeout
    result = subprocess.run(
        ['bash', 'scripts/run_all.sh'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env,
        cwd=POT_PATH,
        timeout=1800  # 30 minutes
    )
    
    print("✅ Main test suite completed")
    print(f"Return code: {result.returncode}")
    
    # Show last part of output
    if result.stdout:
        lines = result.stdout.split('\n')
        print("\nLast 20 lines of output:")
        for line in lines[-20:]:
            if line.strip():
                print(f"  {line}")
                
except subprocess.TimeoutExpired:
    print("⚠️ Test suite timed out after 30 minutes")
except Exception as e:
    print(f"⚠️ Error running tests: {e}")

# Step 5: Package results for download
print("\n📦 Packaging results...")

# Create a simple results directory in /content for easy access
RESULTS_DIR = '/content/PoT_Results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Copy experimental_results if it exists
if os.path.exists('experimental_results'):
    shutil.copytree('experimental_results', f'{RESULTS_DIR}/experimental_results', dirs_exist_ok=True)
    print(f"✅ Copied experimental_results/ to {RESULTS_DIR}")

# Copy test_results if it exists  
if os.path.exists('test_results'):
    shutil.copytree('test_results', f'{RESULTS_DIR}/test_results', dirs_exist_ok=True)
    print(f"✅ Copied test_results/ to {RESULTS_DIR}")

# Create ZIP file in /content for easy download
zip_filename = f'/content/PoT_Results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
print(f"\n📦 Creating downloadable ZIP: {zip_filename}")

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add all files from experimental_results
    if os.path.exists('experimental_results'):
        for root, dirs, files in os.walk('experimental_results'):
            for file in files:
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, POT_PATH)
                zipf.write(file_path, arc_path)
                print(f"  Added: {arc_path}")
    
    # Add all files from test_results
    if os.path.exists('test_results'):
        for root, dirs, files in os.walk('test_results'):
            for file in files:
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, POT_PATH)
                zipf.write(file_path, arc_path)
                print(f"  Added: {arc_path}")

zip_size = os.path.getsize(zip_filename) / (1024 * 1024)
print(f"✅ ZIP created: {zip_size:.2f} MB")

# Step 6: Try to mount Google Drive and save there
print("\n📁 Attempting to save to Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    
    # Create results folder in Drive
    drive_folder = f'/content/drive/MyDrive/PoT_Results_{datetime.now().strftime("%Y%m%d_%H%M")}'
    os.makedirs(drive_folder, exist_ok=True)
    
    # Copy ZIP to Drive
    drive_zip = f'{drive_folder}/PoT_Results.zip'
    shutil.copy2(zip_filename, drive_zip)
    print(f"✅ Results saved to Google Drive: {drive_folder}")
    
    # Copy results directory to Drive
    if os.path.exists(RESULTS_DIR):
        shutil.copytree(RESULTS_DIR, f'{drive_folder}/PoT_Results', dirs_exist_ok=True)
        print(f"✅ Raw results copied to Drive")
        
except Exception as e:
    print(f"⚠️ Could not save to Google Drive: {e}")
    print("Results are still available in /content/PoT_Results and ZIP file")

# Step 7: Make ZIP downloadable
print("\n📥 Initiating download...")
try:
    from google.colab import files
    files.download(zip_filename)
    print("✅ Download initiated!")
except Exception as e:
    print(f"⚠️ Could not initiate download: {e}")

# Step 8: Show what we have
print("\n" + "=" * 70)
print("📊 RESULTS SUMMARY")
print("=" * 70)

print(f"\n📁 Available in /content/PoT_Results:")
if os.path.exists(RESULTS_DIR):
    for item in os.listdir(RESULTS_DIR):
        item_path = os.path.join(RESULTS_DIR, item)
        if os.path.isdir(item_path):
            file_count = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
            print(f"  📁 {item}/ ({file_count} files)")
        else:
            size = os.path.getsize(item_path) / 1024
            print(f"  📄 {item} ({size:.1f} KB)")

print(f"\n📦 ZIP file: {zip_filename} ({zip_size:.2f} MB)")

# Show some key results if available
print(f"\n🎯 Key Results Preview:")
if os.path.exists('experimental_results'):
    json_files = [f for f in os.listdir('experimental_results') if f.endswith('.json')]
    for jf in sorted(json_files)[-3:]:
        print(f"  📊 {jf}")
        
if os.path.exists('test_results'):
    md_files = [f for f in os.listdir('test_results') if f.endswith('.md')]
    for mf in sorted(md_files):
        print(f"  📄 {mf}")

print(f"\n✅ All results packaged and ready!")
print(f"📁 Access files in Colab file browser: /content/PoT_Results/")
print(f"💾 Download ZIP file to your computer")
print(f"☁️ Check Google Drive for permanent backup")

print("\n" + "=" * 70)
print("DONE - Your test results are ready for paper evidence!")
print("=" * 70)