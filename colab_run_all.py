#!/usr/bin/env python3
"""
RUN THE ACTUAL run_all.sh TEST SUITE ON GOOGLE COLAB
====================================================
This properly executes the existing run_all.sh script from the codebase.
No reinventing tests, no simplifications - just run the actual test pipeline.
"""

import os
import sys
import subprocess

print("=" * 70)
print("POT EXPERIMENTS - RUNNING ACTUAL run_all.sh")
print("=" * 70)

# Detect environment
IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
    print("✅ Running in Google Colab")
except ImportError:
    print("📁 Running locally")

# Step 1: Setup repository
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
    print("✅ Repository cloned")
else:
    print(f"📁 Using existing repository at {POT_PATH}")
    # Pull latest changes
    subprocess.run(['git', 'pull'], cwd=POT_PATH)

os.chdir(POT_PATH)
print(f"📍 Working directory: {os.getcwd()}")

# Step 2: Install dependencies
print("\n📦 Installing dependencies...")
if os.path.exists('requirements.txt'):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])
    print("✅ Dependencies installed")
else:
    print("⚠️ No requirements.txt found, installing basic packages...")
    packages = ['numpy', 'torch', 'transformers', 'scipy', 'pytest']
    for pkg in packages:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg])

# Step 3: Make sure the run_all.sh script exists and is executable
run_all_script = 'scripts/run_all.sh'
if not os.path.exists(run_all_script):
    print(f"❌ {run_all_script} not found!")
    print("Available scripts:")
    if os.path.exists('scripts'):
        for f in os.listdir('scripts'):
            if f.endswith('.sh'):
                print(f"  - scripts/{f}")
    sys.exit(1)

# Make it executable
subprocess.run(['chmod', '+x', run_all_script])

# Step 4: Run the actual run_all.sh script
print("\n" + "=" * 70)
print("🚀 RUNNING THE ACTUAL run_all.sh SCRIPT")
print("=" * 70 + "\n")

# Set up environment
env = os.environ.copy()
env['PYTHONPATH'] = POT_PATH
env['PYTHON'] = sys.executable

# For Colab, we might need to handle color codes differently
if IN_COLAB:
    env['TERM'] = 'xterm-256color'

# Execute run_all.sh and stream output
print("Executing scripts/run_all.sh...\n")
process = subprocess.Popen(
    ['bash', run_all_script],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env=env,
    cwd=POT_PATH,
    bufsize=1,
    universal_newlines=True
)

# Stream output in real-time
for line in process.stdout:
    print(line, end='')

# Wait for completion
return_code = process.wait()

print("\n" + "=" * 70)
if return_code == 0:
    print("✅ run_all.sh completed successfully!")
else:
    print(f"⚠️ run_all.sh exited with code {return_code}")

# Show where results are
print("\n📁 Test results are in:")
print(f"  - {POT_PATH}/experimental_results/")
if os.path.exists('experimental_results'):
    files = os.listdir('experimental_results')
    recent_files = sorted([f for f in files if f.endswith('.log') or f.endswith('.json')])[-5:]
    if recent_files:
        print("\n  Recent result files:")
        for f in recent_files:
            print(f"    - {f}")

print("\n" + "=" * 70)
print("DONE - Check experimental_results/ for detailed test outputs")
print("=" * 70)