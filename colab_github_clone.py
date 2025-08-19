# RUN YOUR POT CODEBASE FROM GITHUB IN COLAB

import os
import sys
import subprocess

print("🚀 Setting up PoT from GitHub")
print("=" * 60)

# Clone your repository
print("\n📥 Cloning your PoT repository...")

# Remove old directory if it exists
if os.path.exists('/content/PoT_Experiments'):
    print("Removing old directory...")
    subprocess.run(['rm', '-rf', '/content/PoT_Experiments'], check=True)

# Clone the repo (replace with your actual GitHub URL)
# Example: https://github.com/yourusername/PoT_Experiments.git
repo_url = "https://github.com/rohanvinaik/PoT_Experiments.git"  # REPLACE THIS

# For public repo:
subprocess.run(['git', 'clone', repo_url, '/content/PoT_Experiments'], check=True)

# For private repo (uncomment and use your token):
# token = "ghp_YOUR_GITHUB_TOKEN"  # REPLACE THIS
# repo_url_with_token = f"https://{token}@github.com/YOUR_USERNAME/PoT_Experiments.git"
# subprocess.run(['git', 'clone', repo_url_with_token, '/content/PoT_Experiments'], check=True)

os.chdir('/content/PoT_Experiments')
print(f"✅ Cloned to: {os.getcwd()}")

# Install dependencies
print("\n📦 Installing dependencies...")
subprocess.run(['pip', 'install', '-q', 'torch', 'transformers', 'scipy', 'numpy'], check=True)

# Add to Python path
sys.path.insert(0, '/content/PoT_Experiments')

print("\n" + "="*60)
print("RUNNING YOUR POT TESTS")
print("=" * 60)

# Run your test
print("\n🔬 Running test_llm_verification.py...")
subprocess.run(['python', 'scripts/test_llm_verification.py'], check=True)

print("\n✅ Done!")