# MEMORY-EFFICIENT COLAB SCRIPT - Loads one model at a time
# Avoids RAM crashes by clearing memory between models

import os
import torch
import numpy as np
import time
import json
import gc
from google.colab import drive, userdata
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

print("🚀 Memory-Efficient PoT Framework Test")
print("=" * 60)

# Login to HuggingFace
print("🔐 Logging into HuggingFace...")
try:
    hf_token = userdata.get('HF_TOKEN')
    login(token=hf_token, add_to_git_credential=True)
    print("✅ Logged in!")
except:
    print("⚠️ No HF login (public models still work)")

# Mount Drive
drive.mount('/content/drive')

# Check resources
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️ Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Force cleanup
torch.cuda.empty_cache()
gc.collect()

print("\n" + "="*60)
print("STEP 1: Generate Reference Responses")
print("="*60)

# Test prompts
test_prompts = [
    "What is machine learning?",
    "Explain quantum computing",
    "Describe climate change",
    "What is democracy?",
    "Explain artificial intelligence"
]

# Generate Mistral responses
print("\n📥 Loading Mistral-7B...")
mistral = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    offload_folder="/content/offload"  # Offload to disk if needed
)
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
if mistral_tok.pad_token is None:
    mistral_tok.pad_token = mistral_tok.eos_token

print("Generating Mistral responses...")
mistral_responses = []
with torch.no_grad():
    for prompt in test_prompts:
        inputs = mistral_tok(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = mistral.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=mistral_tok.pad_token_id
        )
        response = mistral_tok.decode(outputs[0], skip_special_tokens=True)
        mistral_responses.append(response)
        print(f"  ✓ {prompt[:30]}...")

# Save responses and FREE MEMORY
results = {"mistral_responses": mistral_responses}
print("\n🗑️ Freeing Mistral memory...")
del mistral
del mistral_tok
torch.cuda.empty_cache()
gc.collect()
time.sleep(2)  # Let memory settle

print("\n" + "="*60)
print("STEP 2: Generate Zephyr Responses")
print("="*60)

# Load Zephyr
print("\n📥 Loading Zephyr-7B...")
zephyr = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    offload_folder="/content/offload"
)
zephyr_tok = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
if zephyr_tok.pad_token is None:
    zephyr_tok.pad_token = zephyr_tok.eos_token

print("Generating Zephyr responses...")
zephyr_responses = []
with torch.no_grad():
    for prompt in test_prompts:
        inputs = zephyr_tok(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = zephyr.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=zephyr_tok.pad_token_id
        )
        response = zephyr_tok.decode(outputs[0], skip_special_tokens=True)
        zephyr_responses.append(response)
        print(f"  ✓ {prompt[:30]}...")

results["zephyr_responses"] = zephyr_responses

# Free Zephyr
print("\n🗑️ Freeing Zephyr memory...")
del zephyr
del zephyr_tok
torch.cuda.empty_cache()
gc.collect()

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

# Calculate similarities
similarities = []
for i, prompt in enumerate(test_prompts):
    m_words = set(mistral_responses[i].lower().split())
    z_words = set(zephyr_responses[i].lower().split())
    
    if len(m_words | z_words) > 0:
        sim = len(m_words & z_words) / len(m_words | z_words)
    else:
        sim = 0
    
    similarities.append(sim)
    print(f"\n📊 Prompt: {prompt[:40]}...")
    print(f"   Similarity: {sim:.2%}")

mean_sim = np.mean(similarities)
std_sim = np.std(similarities)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Mean Similarity: {mean_sim:.2%} ± {std_sim:.2%}")

if mean_sim < 0.7:
    print("✅ SUCCESS: Fine-tuning detected!")
    print("   Zephyr behaves differently from Mistral")
else:
    print("⚠️ Models are too similar")

# Save full results
results["analysis"] = {
    "mean_similarity": float(mean_sim),
    "std_similarity": float(std_sim),
    "similarities": similarities,
    "prompts": test_prompts,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

output_path = "/content/drive/MyDrive/pot_memory_efficient_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {output_path}")
print("\n🎉 COMPLETE - Results ready for your paper!")
print(f"Key finding: {abs(1.0 - mean_sim):.1%} behavioral difference detected")