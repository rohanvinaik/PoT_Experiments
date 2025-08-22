# RUN YOUR POT CODEBASE IN GOOGLE COLAB
# Fixed version with better error handling

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

# Extract
with tarfile.open(package_path, 'r:gz') as tar:
    tar.extractall('/content/')

# Check what was extracted
print("\nExtracted files:")
os.system('ls -la /content/PoT_Experiments/ | head -10')

# Change to the directory
if os.path.exists('/content/PoT_Experiments'):
    os.chdir('/content/PoT_Experiments')
elif os.path.exists('/content/pot'):
    os.chdir('/content')
else:
    print("Checking what was extracted...")
    os.system('ls -la /content/')

print(f"\nCurrent directory: {os.getcwd()}")

# Install dependencies
print("\nInstalling dependencies...")
os.system('pip install -q torch transformers scipy numpy')

# Add to path
sys.path.insert(0, os.getcwd())

print("\n" + "="*60)
print("RUNNING YOUR POT TESTS")
print("=" * 60)

# Check if scripts directory exists
if os.path.exists('scripts'):
    print("\nFound scripts directory:")
    os.system('ls scripts/*.py | head -5')
    
    # Try to run the test
    print("\nRunning test_llm_verification.py...")
    result = subprocess.run(
        ['python', 'scripts/test_llm_verification.py'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error running script: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        # Try running it differently
        print("\nTrying direct import...")
        sys.path.insert(0, 'scripts')
        try:
            import test_llm_verification
        except Exception as e:
            print(f"Import error: {e}")
    else:
        print("Success!")
        print(result.stdout)
else:
    print("ERROR: scripts directory not found")
    print("Current directory contents:")
    os.system('ls -la')