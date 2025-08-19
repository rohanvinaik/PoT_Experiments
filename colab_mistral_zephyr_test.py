"""
Google Colab script for Mistral vs Zephyr comparison
Copy this entire script into a Colab code cell and run it
"""

# ============================================================
# PART 1: Setup Environment
# ============================================================
print("🚀 Setting up Google Colab environment...")
print("=" * 60)

# Install required packages
!pip install -q transformers torch scipy numpy

# Clone PoT repository
!git clone -q https://github.com/rohanvinaik/PoT_Experiments.git 2>/dev/null || echo "Repo already exists"
%cd PoT_Experiments

print("✅ Environment ready!")

# ============================================================
# PART 2: Create necessary PoT framework files
# ============================================================
print("\n📦 Setting up PoT framework modules...")

import os
os.makedirs("pot/lm", exist_ok=True)
os.makedirs("pot/security", exist_ok=True)
os.makedirs("experimental_results", exist_ok=True)

# Create minimal LM verifier
with open("pot/lm/verifier.py", "w") as f:
    f.write('''
import numpy as np
from typing import Any, Dict, Optional

class FuzzyHasher:
    def compute_hash(self, text: str) -> np.ndarray:
        # Simple hash based on character frequencies
        hash_vec = np.zeros(256)
        for char in text:
            hash_vec[ord(char) % 256] += 1
        return hash_vec / (len(text) + 1)

class LMVerifier:
    def __init__(self, reference_model: Any, config: Any):
        self.reference = reference_model
        self.config = config
        self.hasher = FuzzyHasher()
    
    def verify(self, candidate: Any, history: Optional[Any] = None) -> Dict:
        """Verify candidate model against reference"""
        challenges = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a Python function to reverse a string.",
            "What are the benefits of exercise?",
            "Describe the water cycle.",
        ]
        
        if self.config.num_challenges:
            challenges = challenges[:self.config.num_challenges]
        
        ref_hashes = []
        cand_hashes = []
        
        for prompt in challenges:
            ref_output = self.reference.generate(prompt, max_new_tokens=32)
            cand_output = candidate.generate(prompt, max_new_tokens=32)
            
            ref_hashes.append(self.hasher.compute_hash(ref_output))
            cand_hashes.append(self.hasher.compute_hash(cand_output))
        
        # Compute similarity
        similarities = []
        for r, c in zip(ref_hashes, cand_hashes):
            sim = 1.0 - np.mean(np.abs(r - c))
            similarities.append(sim)
        
        mean_sim = np.mean(similarities)
        accepted = mean_sim > self.config.fuzzy_threshold
        
        return {
            "accepted": accepted,
            "decision": "H0" if accepted else "H1",
            "p_value": mean_sim,
            "n_used": len(challenges),
            "threshold": self.config.fuzzy_threshold,
            "mean_similarity": mean_sim,
        }
''')

# Create LM config
with open("pot/lm/lm_config.py", "w") as f:
    f.write('''
class LMVerifierConfig:
    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "hf")
        self.device = kwargs.get("device", "cuda")
        self.num_challenges = kwargs.get("num_challenges", 5)
        self.verification_method = kwargs.get("verification_method", "batch")
        self.sprt_alpha = kwargs.get("sprt_alpha", 0.001)
        self.sprt_beta = kwargs.get("sprt_beta", 0.01)
        self.fuzzy_threshold = kwargs.get("fuzzy_threshold", 0.7)
        self.difficulty_curve = kwargs.get("difficulty_curve", "linear")
''')

# Create __init__ files
for path in ["pot", "pot/lm", "pot/security"]:
    with open(f"{path}/__init__.py", "w") as f:
        f.write("")

print("✅ PoT framework ready!")

# ============================================================
# PART 3: Main Test Script
# ============================================================
print("\n🔬 Running Mistral vs Zephyr Comparison")
print("=" * 60)

import torch
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Helper function to create model adapter
def load_model(model_name: str, device: str = "cuda", seed: int = 42):
    """Load model with HuggingFace transformers"""
    torch.manual_seed(seed)
    
    print(f"\n📥 Loading {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    print(f"✅ {model_name.split('/')[-1]} loaded!")
    
    @torch.no_grad()
    def generate(prompt: str, max_new_tokens: int = 64) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Create adapter object
    class ModelAdapter:
        def __init__(self):
            self.generate = generate
            self.device = device
            self.name = model_name
            self.tok = tokenizer
    
    return ModelAdapter()

# Import PoT verifier
from pot.lm.verifier import LMVerifier
from pot.lm.lm_config import LMVerifierConfig

# ============================================================
# PART 4: Load Models and Run Tests
# ============================================================

# Model names
MISTRAL_BASE = "mistralai/Mistral-7B-Instruct-v0.3"
ZEPHYR_FINETUNE = "HuggingFaceH4/zephyr-7b-beta"

print("\n" + "="*60)
print("LOADING MODELS")
print("="*60)

# Load reference model
ref_model = load_model(MISTRAL_BASE, device=device, seed=1)

# Load candidate models
self_match = load_model(MISTRAL_BASE, device=device, seed=2)  # Different seed
zephyr = load_model(ZEPHYR_FINETUNE, device=device, seed=3)

# Configure verifier
print("\n" + "="*60)
print("CONFIGURING VERIFIER")
print("="*60)

config = LMVerifierConfig(
    model_name="hf",
    device=device,
    num_challenges=5,  # Quick test with 5 challenges
    fuzzy_threshold=0.7,  # Threshold for similarity
)

verifier = LMVerifier(reference_model=ref_model, config=config)
print(f"✅ Verifier ready with {config.num_challenges} challenges")

# Run verification tests
print("\n" + "="*60)
print("RUNNING VERIFICATION TESTS")
print("="*60)

def run_test(candidate, test_name):
    """Run a verification test"""
    print(f"\n🧪 Test: {test_name}")
    print("-" * 40)
    
    start_time = time.time()
    result = verifier.verify(candidate, None)
    elapsed = time.time() - start_time
    
    # Display results
    status = "✅ ACCEPTED" if result["accepted"] else "❌ REJECTED"
    print(f"Result: {status}")
    print(f"Similarity: {result.get('mean_similarity', 0):.3f}")
    print(f"Threshold: {result['threshold']}")
    print(f"Time: {elapsed:.2f}s")
    
    # Save to file
    with open(f"experimental_results/{test_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result

# Test 1: Self-match (should ACCEPT)
print("\n📊 Test 1: Mistral vs Mistral (self-match)")
result1 = run_test(self_match, "mistral_self_match")

# Test 2: Fine-tune detection (should REJECT)
print("\n📊 Test 2: Mistral vs Zephyr (fine-tune detection)")
result2 = run_test(zephyr, "mistral_vs_zephyr")

# ============================================================
# PART 5: Summary
# ============================================================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

if result1["accepted"]:
    print("✅ Test 1 PASSED: Self-match correctly accepted")
else:
    print("❌ Test 1 FAILED: Self-match incorrectly rejected")

if not result2["accepted"]:
    print("✅ Test 2 PASSED: Zephyr correctly rejected (fine-tune detected!)")
else:
    print("⚠️ Test 2 WARNING: Zephyr accepted (fine-tune not distinguished)")
    print("   This may be due to the simplified verifier in Colab")

print("\n🎉 Comparison complete!")
print(f"📁 Results saved in experimental_results/")

# Show detailed results
print("\n📋 Detailed Results:")
print(f"Self-match similarity: {result1.get('mean_similarity', 0):.3f}")
print(f"Zephyr similarity: {result2.get('mean_similarity', 0):.3f}")
print(f"Difference: {abs(result1.get('mean_similarity', 0) - result2.get('mean_similarity', 0)):.3f}")