# USE THE ACTUAL POT CODEBASE IN COLAB
# This script clones your repo and runs the real PoT framework tests

import os
import sys
import subprocess
from google.colab import drive

print("🚀 Setting up REAL PoT Framework from your codebase")
print("=" * 60)

# Mount Drive
drive.mount('/content/drive')

# Clone your PoT repository
print("\n📥 Cloning PoT repository...")
if os.path.exists('/content/PoT_Experiments'):
    print("Repository already exists, pulling latest...")
    os.chdir('/content/PoT_Experiments')
    subprocess.run(['git', 'pull'], check=True)
else:
    # Replace with your actual repo URL
    subprocess.run(['git', 'clone', 'https://github.com/YOUR_USERNAME/PoT_Experiments.git', '/content/PoT_Experiments'], check=True)

os.chdir('/content/PoT_Experiments')

# Install dependencies
print("\n📦 Installing PoT dependencies...")
subprocess.run(['pip', 'install', '-q', 'torch', 'transformers', 'numpy', 'scipy', 'hashlib', 'matplotlib'], check=True)

# Add PoT to Python path
sys.path.insert(0, '/content/PoT_Experiments')

print("\n" + "="*60)
print("IMPORTING REAL POT FRAMEWORK COMPONENTS")
print("="*60)

# Import actual PoT modules
from pot.lm.verifier import LMVerifier, LMVerifierConfig
from pot.lm.models import HFAdapterLM
from pot.security.token_space_normalizer import TokenSpaceNormalizer
from pot.security.provenance_auditor import ProvenanceAuditor
from pot.security.merkle_tree import MerkleTree
from pot.core.challenge import ChallengeGenerator
from pot.core.fuzzy_hash import FuzzyHasher
from pot.experiments.behavioral_fingerprint import BehavioralFingerprinter
from pot.experiments.jacobian_sketch import JacobianSketcher
from pot.reporting.comprehensive_report import ComprehensiveReporter

print("✅ Successfully imported PoT framework modules!")

print("\n" + "="*60)
print("RUNNING REAL POT FRAMEWORK TESTS")
print("="*60)

# Initialize models using actual PoT adapters
print("\n📊 Initializing models with PoT adapters...")
gpt2_model = HFAdapterLM("gpt2")
distil_model = HFAdapterLM("distilgpt2")

# Configure verifier with actual PoT config
config = LMVerifierConfig(
    model_name="hf",
    num_challenges=32,
    verification_method="sequential",
    sprt_alpha=0.001,
    sprt_beta=0.01,
    fuzzy_threshold=0.20,
    enable_jacobian=True,
    enable_merkle=True,
    enable_provenance=True
)

print("\n🔬 Running comprehensive PoT verification...")
verifier = LMVerifier(config)

# Run actual verification
result = verifier.verify(
    reference_model=gpt2_model,
    candidate_model=distil_model,
    verbose=True
)

print("\n" + "="*60)
print("GENERATING POT FRAMEWORK REPORT")
print("="*60)

# Use actual PoT reporter
reporter = ComprehensiveReporter(verifier)
report = reporter.generate_report(
    result,
    save_path="/content/drive/MyDrive/pot_real_framework_report.html"
)

print("\n📊 Test Results from REAL PoT Framework:")
print("-" * 40)
print(f"Overall verdict: {result.verdict}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Tests passed: {result.tests_passed}/{result.total_tests}")

# Display individual test results
for test_name, test_result in result.test_results.items():
    status = "✅ PASS" if test_result.passed else "❌ FAIL"
    print(f"  {test_name}: {status} ({test_result.metric:.2%})")

print("\n💾 Full report saved to Google Drive")
print("🎉 REAL PoT Framework testing complete!")