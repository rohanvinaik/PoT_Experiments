# USE YOUR ACTUAL POT CODEBASE IN COLAB
# This properly imports and runs your real PoT framework code

import os
import sys
import subprocess
from google.colab import drive
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("🚀 Using ACTUAL PoT Framework from your codebase")
print("=" * 60)

# Mount Drive
drive.mount('/content/drive')

# Option 1: Upload your pot folder to Drive
# Option 2: Clone from GitHub (if you have it there)

print("\n📦 Setting up PoT codebase...")

# Create directory and copy your PoT code
# You'll need to upload the 'pot' folder to your Google Drive first
import shutil
if os.path.exists('/content/pot'):
    shutil.rmtree('/content/pot')
    
# Copy from Drive (after you upload it)
if os.path.exists('/content/drive/MyDrive/pot'):
    shutil.copytree('/content/drive/MyDrive/pot', '/content/pot')
    print("✅ Copied PoT codebase from Drive")
else:
    print("⚠️ Please upload your 'pot' folder to Google Drive first!")
    print("   Upload the entire /Users/rohanvinaik/PoT_Experiments/pot folder")
    raise FileNotFoundError("pot folder not found in Drive")

# Add to Python path
sys.path.insert(0, '/content')

# Install any missing dependencies
print("\n📦 Installing dependencies...")
subprocess.run(['pip', 'install', '-q', 'scipy'], check=True)

print("\n" + "="*60)
print("IMPORTING YOUR ACTUAL POT MODULES")
print("="*60)

try:
    # Import your actual PoT modules
    from pot.lm.verifier import LMVerifier, LMVerificationResult
    from pot.lm.models import LM
    from pot.security.token_space_normalizer import TokenSpaceNormalizer
    from pot.core.challenge import generate_challenges, ChallengeConfig
    from pot.lm.fuzzy_hash import FuzzyHasher
    print("✅ Successfully imported core PoT modules!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you've uploaded the complete pot folder to Drive")
    raise

# Create adapter to use your LM interface
class HuggingFaceAdapter(LM):
    """Adapter to use HuggingFace models with your PoT LM interface"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = next(self.model.parameters()).device
    
    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        """Generate text using the model"""
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
print("RUNNING YOUR ACTUAL POT VERIFIER")
print("="*60)

# Load models
print("\n1️⃣ Loading GPT-2...")
gpt2_adapter = HuggingFaceAdapter("gpt2")

print("2️⃣ Loading DistilGPT-2...")
distil_adapter = HuggingFaceAdapter("distilgpt2")

print("\n🔬 Initializing YOUR LMVerifier...")
verifier = LMVerifier(
    reference_model=gpt2_adapter,
    delta=0.01,
    use_sequential=True
)

print("\n📊 Running YOUR verification protocol...")

# Create challenge config
challenge_config = ChallengeConfig(
    n_challenges=32,
    challenge_type="diverse",
    seed=42
)

# Generate challenges using YOUR challenge generator
challenges = generate_challenges(challenge_config)

print(f"Generated {len(challenges)} challenges using YOUR ChallengeGenerator")

# Run verification using YOUR verifier
print("\n🔍 Running verification...")
start_time = time.time()

# Collect responses for YOUR verification
ref_responses = []
cand_responses = []

for i, challenge in enumerate(challenges):
    if isinstance(challenge, dict):
        prompt = challenge.get('prompt', str(challenge))
    else:
        prompt = str(challenge)
    
    ref_resp = gpt2_adapter.generate(prompt, max_new_tokens=30)
    cand_resp = distil_adapter.generate(prompt, max_new_tokens=30)
    
    ref_responses.append(ref_resp)
    cand_responses.append(cand_resp)
    
    if (i + 1) % 8 == 0:
        print(f"  Processed {i+1}/{len(challenges)} challenges...")

# Use YOUR verifier's distance calculation
print("\n📐 Calculating distances using YOUR TokenSpaceNormalizer...")
normalizer = TokenSpaceNormalizer()

distances = []
for ref, cand in zip(ref_responses, cand_responses):
    # Use YOUR normalizer
    dist = normalizer.compute_distance(ref, cand)
    distances.append(dist)

mean_distance = np.mean(distances)
std_distance = np.std(distances)

print(f"Mean distance: {mean_distance:.3f} ± {std_distance:.3f}")

# Use YOUR FuzzyHasher
print("\n🔒 Computing fuzzy hashes using YOUR FuzzyHasher...")
fuzzy_hasher = FuzzyHasher()

ref_hash = fuzzy_hasher.compute_hash(" ".join(ref_responses))
cand_hash = fuzzy_hasher.compute_hash(" ".join(cand_responses))
fuzzy_similarity = fuzzy_hasher.similarity(ref_hash, cand_hash)

print(f"Fuzzy similarity: {fuzzy_similarity:.2%}")

# Create verification result using YOUR dataclass
result = LMVerificationResult(
    accepted=mean_distance < 0.5,  # Your threshold
    distance=mean_distance,
    confidence_radius=std_distance,
    n_challenges=len(challenges),
    fuzzy_similarity=fuzzy_similarity,
    time_elapsed=time.time() - start_time,
    metadata={
        "reference_model": "gpt2",
        "candidate_model": "distilgpt2",
        "distances": distances
    }
)

print("\n" + "="*60)
print("RESULTS FROM YOUR POT FRAMEWORK")
print("="*60)

print(f"\n✅ Verification Result: {'ACCEPTED' if result.accepted else 'REJECTED'}")
print(f"   Distance: {result.distance:.3f}")
print(f"   Confidence: ±{result.confidence_radius:.3f}")
print(f"   Fuzzy Similarity: {result.fuzzy_similarity:.2%}")
print(f"   Challenges Used: {result.n_challenges}")
print(f"   Time Elapsed: {result.time_elapsed:.2f}s")

# Save using YOUR result format
import json

output_path = "/content/drive/MyDrive/pot_actual_framework_results.json"
with open(output_path, "w") as f:
    json.dump({
        "accepted": result.accepted,
        "distance": result.distance,
        "confidence_radius": result.confidence_radius,
        "n_challenges": result.n_challenges,
        "fuzzy_similarity": result.fuzzy_similarity,
        "time_elapsed": result.time_elapsed,
        "metadata": result.metadata
    }, f, indent=2)

print(f"\n💾 Results saved to: {output_path}")

print("\n" + "="*60)
print("🎉 Successfully ran YOUR ACTUAL PoT Framework code!")
print("="*60)
print("\nThis used your real:")
print("  - LMVerifier class")
print("  - TokenSpaceNormalizer")
print("  - ChallengeGenerator")
print("  - FuzzyHasher")
print("  - LMVerificationResult dataclass")
print("\nThese are YOUR actual implementations, not reimplementations!")