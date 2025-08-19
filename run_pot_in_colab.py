# RUN YOUR ACTUAL POT CODEBASE IN GOOGLE COLAB
# This script runs YOUR existing PoT tests in Colab

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

with tarfile.open(package_path, 'r:gz') as tar:
    tar.extractall('/content/PoT_Experiments/')

os.chdir('/content/PoT_Experiments')

# Install dependencies
print("\nInstalling dependencies...")
subprocess.run(['pip', 'install', '-q', 'torch', 'transformers', 'scipy', 'numpy'], check=True)

# Add to path
sys.path.insert(0, '/content/PoT_Experiments')

print("\n" + "="*60)
print("RUNNING YOUR POT TESTS")
print("=" * 60)

# Run your existing test scripts
print("\nRunning scripts/test_llm_verification.py...")
subprocess.run(['python', 'scripts/test_llm_verification.py'], check=True)

print("\nDone! Check results in Google Drive.")