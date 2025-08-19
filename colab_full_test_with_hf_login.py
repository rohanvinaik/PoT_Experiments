# COMPLETE COLAB SCRIPT WITH HUGGINGFACE LOGIN AND FULL POT TEST
# Uses your uploaded Mistral model and downloads Zephyr

# ============================================================
# PART 1: Setup and Login
# ============================================================
import os
import torch
import numpy as np
import time
import json
from google.colab import drive, userdata
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

print("🚀 Complete PoT Framework Test with HuggingFace Login")
print("=" * 60)

# Login to HuggingFace using Colab secret
print("🔐 Logging into HuggingFace...")
from huggingface_hub import login

try:
    # Get token from Colab secrets
    hf_token = userdata.get('HF_TOKEN')  # Assumes you named it HF_TOKEN in secrets
    login(token=hf_token, add_to_git_credential=True)
    print("✅ Logged into HuggingFace successfully!")
except Exception as e:
    print(f"⚠️ HuggingFace login failed: {e}")
    print("Continuing without login (public models should still work)")

# Mount Google Drive
print("\n📁 Mounting Google Drive...")
drive.mount('/content/drive')

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️ Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Clear memory
torch.cuda.empty_cache()
gc.collect()

# ============================================================
# PART 2: Load Models
# ============================================================
print("\n" + "="*60)
print("LOADING MODELS")
print("="*60)

# Load Mistral - just download it since Drive only has model files
print("\n1️⃣ Downloading Mistral-7B...")
print("(Google Drive files incomplete - missing config/tokenizer)")
mistral = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
if mistral_tok.pad_token is None:
    mistral_tok.pad_token = mistral_tok.eos_token
print("✅ Mistral downloaded!")

# Load Zephyr (download)
print("\n2️⃣ Downloading Zephyr-7B...")
zephyr = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
zephyr_tok = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
if zephyr_tok.pad_token is None:
    zephyr_tok.pad_token = zephyr_tok.eos_token
print("✅ Zephyr downloaded!")

# Create model adapters for PoT framework
class ModelAdapter:
    def __init__(self, model, tokenizer, name):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.device = next(model.parameters()).device
        
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=64):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

mistral_adapter = ModelAdapter(mistral, mistral_tok, "Mistral-7B")
zephyr_adapter = ModelAdapter(zephyr, zephyr_tok, "Zephyr-7B")

# ============================================================
# PART 3: PoT Framework Implementation
# ============================================================
print("\n" + "="*60)
print("POT FRAMEWORK SETUP")
print("="*60)

import hashlib
import hmac

class CryptoChallenger:
    """Generate cryptographic challenges"""
    def __init__(self, seed="pot_test"):
        self.seed = seed.encode()
        
    def generate(self, index):
        """Generate deterministic challenge"""
        key = hmac.new(self.seed, f"challenge_{index}".encode(), hashlib.sha256).digest()
        
        prompts = [
            "Explain the concept of",
            "What are the benefits of",
            "Describe the process of",
            "Compare and contrast",
            "Analyze the importance of"
        ]
        
        topics = [
            "artificial intelligence",
            "quantum computing",
            "climate change",
            "democracy",
            "evolution",
            "consciousness",
            "economics",
            "space exploration"
        ]
        
        prompt_idx = int.from_bytes(key[:1], 'big') % len(prompts)
        topic_idx = int.from_bytes(key[1:2], 'big') % len(topics)
        
        return f"{prompts[prompt_idx]} {topics[topic_idx]}"

class BehavioralVerifier:
    """Verify model behavior"""
    def __init__(self, reference_model, threshold=0.7):
        self.reference = reference_model
        self.threshold = threshold
        self.challenger = CryptoChallenger()
        
    def verify(self, candidate_model, num_challenges=5):
        """Run verification test"""
        similarities = []
        results = []
        
        print(f"\nRunning {num_challenges} cryptographic challenges...")
        
        for i in range(num_challenges):
            # Generate challenge
            challenge = self.challenger.generate(i)
            print(f"\n🔐 Challenge {i+1}: {challenge[:50]}...")
            
            # Get responses
            ref_response = self.reference.generate(challenge, max_new_tokens=30)
            cand_response = candidate_model.generate(challenge, max_new_tokens=30)
            
            # Calculate similarity (word overlap)
            ref_words = set(ref_response.lower().split())
            cand_words = set(cand_response.lower().split())
            
            if len(ref_words | cand_words) > 0:
                similarity = len(ref_words & cand_words) / len(ref_words | cand_words)
            else:
                similarity = 0
                
            similarities.append(similarity)
            results.append({
                "challenge": challenge,
                "similarity": similarity,
                "ref_length": len(ref_response),
                "cand_length": len(cand_response)
            })
            
            print(f"  Similarity: {similarity:.2%}")
        
        # Calculate final verdict
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        
        return {
            "accepted": mean_similarity > self.threshold,
            "mean_similarity": mean_similarity,
            "std_similarity": std_similarity,
            "threshold": self.threshold,
            "num_challenges": num_challenges,
            "details": results
        }

# ============================================================
# PART 4: Run Tests
# ============================================================
print("\n" + "="*60)
print("RUNNING VERIFICATION TESTS")
print("="*60)

# Initialize verifier
verifier = BehavioralVerifier(mistral_adapter, threshold=0.7)

# Test 1: Self-verification (baseline)
print("\n📊 Test 1: Mistral vs Mistral (self-verification)")
print("-" * 40)
mistral_self = ModelAdapter(mistral, mistral_tok, "Mistral-Self")
result_self = verifier.verify(mistral_self, num_challenges=3)

# Test 2: Fine-tune detection
print("\n📊 Test 2: Mistral vs Zephyr (fine-tune detection)")
print("-" * 40)
result_zephyr = verifier.verify(zephyr_adapter, num_challenges=5)

# ============================================================
# PART 5: Results and Analysis
# ============================================================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

def print_result(name, result):
    status = "✅ ACCEPTED" if result["accepted"] else "❌ REJECTED"
    print(f"\n{name}:")
    print(f"  Status: {status}")
    print(f"  Mean Similarity: {result['mean_similarity']:.2%} ± {result['std_similarity']:.2%}")
    print(f"  Threshold: {result['threshold']:.2%}")
    print(f"  Challenges Used: {result['num_challenges']}")

print_result("Self-Verification", result_self)
print_result("Zephyr Verification", result_zephyr)

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if result_self["accepted"] and not result_zephyr["accepted"]:
    print("✅ SUCCESS: System correctly identified:")
    print("  - Same model (Mistral self) as authentic")
    print("  - Fine-tuned model (Zephyr) as modified")
    print("\n🎯 The PoT framework successfully detects fine-tuning!")
elif not result_self["accepted"]:
    print("⚠️ WARNING: Self-verification failed")
    print("  This suggests the threshold may be too strict")
elif result_zephyr["accepted"]:
    print("⚠️ WARNING: Failed to detect fine-tuning")
    print("  Zephyr was accepted as authentic")
    print("  Consider: More challenges, stricter threshold, or better similarity metric")

# Save results
results_data = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "device": str(device),
    "self_verification": result_self,
    "zephyr_verification": result_zephyr,
    "models": {
        "reference": "Mistral-7B-Instruct-v0.3",
        "candidate": "Zephyr-7B-beta"
    }
}

# Save to Drive
output_path = "/content/drive/MyDrive/pot_test_results.json"
with open(output_path, "w") as f:
    json.dump(results_data, f, indent=2)
print(f"\n💾 Results saved to: {output_path}")

print("\n" + "="*60)
print("🎉 TEST COMPLETE!")
print("="*60)
print("Results are ready for your paper.")
print(f"Similarity difference: {abs(result_self['mean_similarity'] - result_zephyr['mean_similarity']):.2%}")

# Memory cleanup
del mistral, zephyr
torch.cuda.empty_cache()
gc.collect()

print("\n✅ Memory cleaned. You can now run additional tests if needed.")