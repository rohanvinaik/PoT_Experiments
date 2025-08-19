#!/usr/bin/env python3
"""
RUN THE COMPLETE POT TEST SUITE ON GOOGLE COLAB
===============================================
This simply clones the PoT repository and runs the full run_all.sh test suite.
Nothing more, nothing less.

Usage in Google Colab:
1. Upload this file to Colab
2. Run it
3. Get the same comprehensive test results as running locally
"""

import os
import sys
import subprocess

print("=" * 70)
print("POT EXPERIMENTS - FULL TEST SUITE")
print("Running complete run_all.sh pipeline on Google Colab")
print("=" * 70)

# Step 1: Clone the repository from GitHub
print("\n📥 Cloning PoT repository from GitHub...")
if os.path.exists('/content/PoT_Experiments'):
    print("Repository already exists, pulling latest...")
    os.chdir('/content/PoT_Experiments')
    subprocess.run(['git', 'pull'], check=True)
else:
    subprocess.run(['git', 'clone', 'https://github.com/rohanvinaik/PoT_Experiments.git'], 
                   cwd='/content', check=True)
    os.chdir('/content/PoT_Experiments')

print("✅ Repository ready")

# Step 2: Install dependencies
print("\n📦 Installing dependencies...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
print("✅ Dependencies installed")

# Step 3: Run the complete test suite
print("\n🚀 Running complete test suite (scripts/run_all.sh)...")
print("=" * 70)

# Convert bash script to Python execution for Colab compatibility
env = os.environ.copy()
env['PYTHONPATH'] = os.getcwd()
env['PYTHON'] = sys.executable

# Execute run_all.sh
result = subprocess.run(['bash', 'scripts/run_all.sh'], 
                       capture_output=False,  # Show output in real-time
                       text=True,
                       env=env)

if result.returncode == 0:
    print("\n✅ All tests completed successfully!")
else:
    print(f"\n⚠️ Tests completed with return code: {result.returncode}")

print("\n📁 Check experimental_results/ directory for detailed test results")
print("=" * 70)