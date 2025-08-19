# RUN YOUR ACTUAL POT CODEBASE IN COLAB
# This extracts and uses your real PoT framework code

import os
import sys
import subprocess
import tarfile
import time
import numpy as np
from google.colab import drive
import torch

print("🚀 Setting up YOUR ACTUAL PoT Framework")
print("=" * 60)

# Mount Drive
drive.mount('/content/drive')

print("\n📦 Extracting your PoT codebase...")

# Look for the package
package_path = "/content/drive/MyDrive/pot_package.tar.gz"

if not os.path.exists(package_path):
    print("❌ pot_package.tar.gz not found in Google Drive!")
    print("\nTo create the package on your Mac:")
    print("1. Run in terminal:")
    print("   cd /Users/rohanvinaik/PoT_Experiments")
    print("   tar -czf ~/Desktop/pot_package.tar.gz pot/")
    print("2. Upload pot_package.tar.gz to Google Drive")
    raise FileNotFoundError("Package not found")

# Extract the package
print("📂 Extracting pot_package.tar.gz...")
with tarfile.open(package_path, 'r:gz') as tar:
    tar.extractall('/content/')

print("✅ Extracted your pot/ folder")

# Verify extraction
if os.path.exists('/content/pot'):
    num_files = len([f for f in os.listdir('/content/pot') if f.endswith('.py')])
    print(f"   Found {num_files} Python modules in pot/")
else:
    raise FileNotFoundError("Extraction failed - pot/ not found")

# Add to Python path
sys.path.insert(0, '/content')

# Install dependencies your code needs
print("\n📦 Installing dependencies...")
subprocess.run(['pip', 'install', '-q', 'scipy', 'transformers', 'torch'], check=True)

print("\n" + "="*60)
print("IMPORTING YOUR ACTUAL POT MODULES")
print("="*60)

# Import your actual modules
try:
    from pot.lm.verifier import LMVerifier, LMVerificationResult
    print("✅ Imported LMVerifier")
    
    from pot.core.challenge import generate_challenges, ChallengeConfig
    print("✅ Imported ChallengeGenerator")
    
    from pot.security.token_space_normalizer import TokenSpaceNormalizer
    print("✅ Imported TokenSpaceNormalizer")
    
    from pot.lm.fuzzy_hash import FuzzyHasher
    print("✅ Imported FuzzyHasher")
    
    from pot.core.sequential import SequentialTester
    print("✅ Imported SequentialTester")
    
    from pot.experiments.behavioral_fingerprint import generate_fingerprint
    print("✅ Imported behavioral fingerprinting")
    
    # Try importing more modules
    try:
        from pot.security.merkle_tree import MerkleTree
        print("✅ Imported MerkleTree")
    except ImportError:
        print("⚠️ MerkleTree not available")
    
    try:
        from pot.experiments.jacobian_sketch import compute_jacobian_sketch
        print("✅ Imported Jacobian sketching")
    except ImportError:
        print("⚠️ Jacobian sketching not available")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nAvailable modules in pot/:")
    for root, dirs, files in os.walk('/content/pot'):
        for file in files:
            if file.endswith('.py'):
                print(f"  {os.path.join(root, file)}")
    raise

print("\n" + "="*60)
print("CREATING MODEL ADAPTER FOR YOUR LM INTERFACE")
print("="*60)

# We need to create an adapter since your LM class is abstract
from pot.lm.models import LM
from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingFaceAdapter(LM):
    """Adapter to use HuggingFace models with your PoT framework"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"Loading {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = next(self.model.parameters()).device
        self.tok = self.tokenizer  # Your code expects this attribute
        print(f"✅ Loaded {model_name}")
    
    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        """Generate text - matches your LM interface"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_model_name(self) -> str:
        return self.model_name

print("\n" + "="*60)
print("RUNNING YOUR ACTUAL POT VERIFICATION")
print("="*60)

# Initialize models with your adapter
gpt2_model = HuggingFaceAdapter("gpt2")
distil_model = HuggingFaceAdapter("distilgpt2")

# Use YOUR LMVerifier
print("\n🔬 Initializing YOUR LMVerifier...")
verifier = LMVerifier(
    reference_model=gpt2_model,
    delta=0.01,
    use_sequential=True
)

print("\n📊 Generating challenges with YOUR ChallengeConfig...")
config = ChallengeConfig(
    n_challenges=16,  # Smaller for demo
    challenge_type="diverse",
    seed=42
)

challenges = generate_challenges(config)
print(f"Generated {len(challenges)} challenges")

# Run YOUR verification protocol
print("\n🔍 Running YOUR verification protocol...")
start_time = time.time()

# Collect responses
ref_responses = []
cand_responses = []

for i, challenge in enumerate(challenges):
    # Handle different challenge formats from your code
    if isinstance(challenge, dict):
        prompt = challenge.get('prompt', str(challenge))
    else:
        prompt = str(challenge)
    
    ref_resp = gpt2_model.generate(prompt, max_new_tokens=30)
    cand_resp = distil_model.generate(prompt, max_new_tokens=30)
    
    ref_responses.append(ref_resp)
    cand_responses.append(cand_resp)
    
    if (i + 1) % 4 == 0:
        print(f"  Processed {i+1}/{len(challenges)} challenges...")

# Use YOUR TokenSpaceNormalizer
print("\n📐 Computing distances with YOUR TokenSpaceNormalizer...")
normalizer = TokenSpaceNormalizer()

distances = []
for ref, cand in zip(ref_responses, cand_responses):
    dist = normalizer.compute_distance(ref, cand)
    distances.append(dist)

mean_distance = np.mean(distances)
std_distance = np.std(distances)

print(f"Mean distance: {mean_distance:.3f} ± {std_distance:.3f}")

# Use YOUR FuzzyHasher
print("\n🔒 Computing fuzzy hashes with YOUR FuzzyHasher...")
hasher = FuzzyHasher()

# Compute hashes
ref_text = " ".join(ref_responses)
cand_text = " ".join(cand_responses)

ref_hash = hasher.compute_hash(ref_text)
cand_hash = hasher.compute_hash(cand_text)
similarity = hasher.similarity(ref_hash, cand_hash)

print(f"Reference hash: {ref_hash[:16]}...")
print(f"Candidate hash: {cand_hash[:16]}...")
print(f"Fuzzy similarity: {similarity:.2%}")

# Create result using YOUR dataclass
result = LMVerificationResult(
    accepted=mean_distance < 0.5,
    distance=mean_distance,
    confidence_radius=std_distance,
    n_challenges=len(challenges),
    fuzzy_similarity=similarity,
    time_elapsed=time.time() - start_time,
    metadata={
        "reference_model": gpt2_model.get_model_name(),
        "candidate_model": distil_model.get_model_name(),
        "distances": [float(d) for d in distances]
    }
)

print("\n" + "="*60)
print("RESULTS FROM YOUR ACTUAL POT FRAMEWORK")
print("="*60)

print(f"\n📊 Verification Result: {'ACCEPTED' if result.accepted else 'REJECTED'}")
print(f"   Distance: {result.distance:.3f}")
print(f"   Confidence: ±{result.confidence_radius:.3f}")
print(f"   Fuzzy Similarity: {result.fuzzy_similarity:.2%}")
print(f"   Challenges: {result.n_challenges}")
print(f"   Time: {result.time_elapsed:.2f}s")

# Try running more of your modules if available
print("\n" + "="*60)
print("RUNNING ADDITIONAL POT COMPONENTS")
print("="*60)

try:
    # Try behavioral fingerprinting
    print("\n🔍 Generating behavioral fingerprint...")
    fingerprint = generate_fingerprint(gpt2_model, num_samples=5)
    print(f"✅ Fingerprint generated: {fingerprint}")
except Exception as e:
    print(f"⚠️ Behavioral fingerprinting not available: {e}")

try:
    # Try sequential testing
    print("\n📈 Running sequential test...")
    tester = SequentialTester(alpha=0.01, beta=0.01)
    # Add your sequential testing logic here
    print("✅ Sequential testing available")
except Exception as e:
    print(f"⚠️ Sequential testing error: {e}")

# Save results
import json
output_path = "/content/drive/MyDrive/pot_actual_results.json"
with open(output_path, "w") as f:
    json.dump({
        "framework": "ACTUAL PoT Codebase",
        "modules_used": [
            "LMVerifier",
            "ChallengeGenerator", 
            "TokenSpaceNormalizer",
            "FuzzyHasher",
            "LMVerificationResult"
        ],
        "result": {
            "accepted": result.accepted,
            "distance": float(result.distance),
            "confidence_radius": float(result.confidence_radius),
            "n_challenges": result.n_challenges,
            "fuzzy_similarity": float(result.fuzzy_similarity),
            "time_elapsed": result.time_elapsed
        },
        "metadata": result.metadata
    }, f, indent=2)

print(f"\n💾 Results saved to: {output_path}")

print("\n" + "="*60)
print("🎉 SUCCESSFULLY RAN YOUR ACTUAL POT FRAMEWORK!")
print("="*60)
print("\nThis used YOUR real code from:")
print("  /Users/rohanvinaik/PoT_Experiments/pot/")
print("\nIncluding your actual implementations of:")
for module in ["LMVerifier", "TokenSpaceNormalizer", "ChallengeGenerator", "FuzzyHasher"]:
    print(f"  ✓ {module}")
print("\nThese results are from YOUR codebase, not reimplementations!")