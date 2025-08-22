#!/usr/bin/env python3
# RUN YOUR POT CODEBASE IN GOOGLE COLAB - CORRECTED VERSION

import os
import sys
import subprocess
import tarfile
from google.colab import drive

print("Setting up YOUR PoT codebase in Colab")
print("=" * 60)

# Mount Google Drive
drive.mount('/content/drive')

# Extract your PoT codebase
print("\nExtracting PoT codebase...")
package_path = "/content/drive/MyDrive/pot_package.tar.gz"

if not os.path.exists(package_path):
    print("ERROR: pot_package.tar.gz not found!")
    print("\nTo create it on your Mac:")
    print("cd /Users/rohanvinaik/PoT_Experiments")
    print("tar -czf ~/Desktop/pot_package.tar.gz .")
    print("\nThen upload to Google Drive")
    sys.exit(1)

# Extract directly to /content
print("Extracting to /content...")
os.chdir('/content')
with tarfile.open(package_path, 'r:gz') as tar:
    tar.extractall()

# Now we should have everything in /content
print("\nChecking extracted contents:")
os.system('ls -la /content/ | grep -E "(pot|scripts|run_)"')

# Verify scripts directory
if os.path.exists('/content/scripts'):
    print("\n✅ Found scripts directory")
    os.chdir('/content')
else:
    print("\n❌ Scripts directory not found. Contents of /content:")
    os.system('ls -la /content/')
    sys.exit(1)

print(f"\nCurrent directory: {os.getcwd()}")

# Install dependencies
print("\nInstalling dependencies...")
os.system('pip install -q torch transformers scipy numpy')

# Add to Python path
sys.path.insert(0, '/content')

print("\n" + "="*60)
print("RUNNING YOUR POT TESTS")
print("=" * 60)

# Run your test script
print("\nRunning scripts/test_llm_verification.py...")
result = subprocess.run(
    ['python', 'scripts/test_llm_verification.py'],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print(f"Error code: {result.returncode}")
    print("\n--- STDOUT ---")
    print(result.stdout)
    print("\n--- STDERR ---")
    print(result.stderr)
    
    # Show the actual error
    print("\n--- Trying to diagnose the issue ---")
    print("First few lines of the script:")
    os.system('head -20 scripts/test_llm_verification.py')
else:
    print("✅ Success!")
    print(result.stdout)

print("\nDone!")
