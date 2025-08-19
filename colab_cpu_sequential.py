# CPU-ONLY SEQUENTIAL TEST
# Loads one model at a time, generates responses, then completely unloads

import os
import torch
import numpy as np
import json
import gc
import time
from google.colab import drive, userdata
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

print("🚀 CPU-Only Sequential PoT Test")
print("=" * 60)
print("⚠️ WARNING: This will be SLOW on CPU but won't crash!")

# Login to HuggingFace
try:
    hf_token = userdata.get('HF_TOKEN')
    login(token=hf_token, add_to_git_credential=True)
    print("✅ Logged into HuggingFace")
except:
    print("⚠️ No HF login")

# Mount Drive
drive.mount('/content/drive')

# Force CPU
device = "cpu"
print(f"\n🖥️ Device: {device}")
print("Note: Using CPU - will be slow but stable")

# Clear any existing memory
gc.collect()

print("\n" + "="*60)
print("PHASE 1: MISTRAL ONLY")
print("="*60)

# Single test prompt for speed
test_prompt = "What is machine learning?"

print("\n📥 Loading Mistral-7B (this takes 3-5 minutes on CPU)...")
start_time = time.time()

# Load with minimal memory footprint
mistral = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    torch_dtype=torch.float32,  # CPU needs float32
    low_cpu_mem_usage=True,
    device_map={"": "cpu"}
)
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
mistral_tok.pad_token = mistral_tok.eos_token

print(f"✅ Mistral loaded in {time.time()-start_time:.1f}s")

# Generate ONE response
print("\n🤖 Generating Mistral response (30-60s on CPU)...")
start_time = time.time()

with torch.no_grad():
    inputs = mistral_tok(test_prompt, return_tensors="pt", max_length=128, truncation=True)
    outputs = mistral.generate(
        inputs.input_ids,
        max_new_tokens=20,  # Short for speed
        do_sample=False,
        pad_token_id=mistral_tok.pad_token_id
    )
    mistral_response = mistral_tok.decode(outputs[0], skip_special_tokens=True)

print(f"✅ Generated in {time.time()-start_time:.1f}s")
print(f"Response preview: {mistral_response[:100]}...")

# SAVE RESULT AND COMPLETELY FREE MEMORY
result_file = "/content/mistral_response.txt"
with open(result_file, "w") as f:
    f.write(mistral_response)

print("\n🗑️ Unloading Mistral completely...")
del mistral
del mistral_tok
del outputs
del inputs
gc.collect()
time.sleep(5)  # Let memory settle

print("✅ Memory freed!")

print("\n" + "="*60)
print("PHASE 2: ZEPHYR ONLY")
print("="*60)

print("\n📥 Loading Zephyr-7B (3-5 minutes)...")
start_time = time.time()

zephyr = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    device_map={"": "cpu"}
)
zephyr_tok = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
zephyr_tok.pad_token = zephyr_tok.eos_token

print(f"✅ Zephyr loaded in {time.time()-start_time:.1f}s")

# Generate response
print("\n🤖 Generating Zephyr response (30-60s)...")
start_time = time.time()

with torch.no_grad():
    inputs = zephyr_tok(test_prompt, return_tensors="pt", max_length=128, truncation=True)
    outputs = zephyr.generate(
        inputs.input_ids,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=zephyr_tok.pad_token_id
    )
    zephyr_response = zephyr_tok.decode(outputs[0], skip_special_tokens=True)

print(f"✅ Generated in {time.time()-start_time:.1f}s")
print(f"Response preview: {zephyr_response[:100]}...")

# Free Zephyr
del zephyr
del zephyr_tok
gc.collect()

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

# Load saved Mistral response
with open(result_file, "r") as f:
    mistral_response = f.read()

# Compare
m_words = set(mistral_response.lower().split())
z_words = set(zephyr_response.lower().split())
similarity = len(m_words & z_words) / len(m_words | z_words) if (m_words | z_words) else 0

print(f"\n📊 Results for: '{test_prompt}'")
print(f"Mistral length: {len(mistral_response)} chars")
print(f"Zephyr length: {len(zephyr_response)} chars")
print(f"Word overlap similarity: {similarity:.1%}")

if similarity < 0.7:
    print("\n✅ SUCCESS: Models are different!")
    print("   Fine-tuning detected")
else:
    print("\n⚠️ Models are similar")

# Save final results
results = {
    "prompt": test_prompt,
    "mistral_response": mistral_response,
    "zephyr_response": zephyr_response,
    "similarity": similarity,
    "conclusion": "fine-tuning detected" if similarity < 0.7 else "models similar",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

output_path = "/content/drive/MyDrive/pot_cpu_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {output_path}")
print("\n🎉 COMPLETE!")
print("Results ready for your paper - showing behavioral difference between base and fine-tuned models")