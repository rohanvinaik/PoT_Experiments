# ULTRA-LIGHTWEIGHT 8-BIT QUANTIZED TEST
# Uses 8-bit quantization to fit both models in Colab's limited RAM

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import json
import gc
from google.colab import drive, userdata
from huggingface_hub import login

print("🚀 8-bit Quantized PoT Test (Ultra Memory Efficient)")
print("=" * 60)

# Login to HuggingFace
try:
    hf_token = userdata.get('HF_TOKEN')
    login(token=hf_token, add_to_git_credential=True)
    print("✅ Logged into HuggingFace")
except:
    print("⚠️ No HF login")

# Mount Drive for results
drive.mount('/content/drive')

# Check GPU
if not torch.cuda.is_available():
    print("❌ ERROR: GPU required for quantization!")
    raise RuntimeError("Need GPU")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Configure 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

print("\n" + "="*60)
print("LOADING MODELS (8-bit quantized)")
print("="*60)

# Load Mistral in 8-bit
print("\n1️⃣ Loading Mistral (8-bit)...")
mistral = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
mistral_tok.pad_token = mistral_tok.eos_token
print("✅ Mistral loaded (using ~7GB)")

# Load Zephyr in 8-bit
print("\n2️⃣ Loading Zephyr (8-bit)...")
zephyr = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
zephyr_tok = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
zephyr_tok.pad_token = zephyr_tok.eos_token
print("✅ Zephyr loaded (using ~7GB)")

print("\n" + "="*60)
print("RUNNING COMPARISON")
print("="*60)

# Simple test prompts
prompts = [
    "What is artificial intelligence?",
    "Explain climate change",
    "What is democracy?"
]

results = []
for prompt in prompts:
    print(f"\n🔍 Testing: {prompt[:40]}...")
    
    # Mistral
    with torch.no_grad():
        inputs = mistral_tok(prompt, return_tensors="pt", max_length=512, truncation=True)
        m_out = mistral.generate(
            inputs.input_ids.cuda(),
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=mistral_tok.pad_token_id
        )
        m_text = mistral_tok.decode(m_out[0], skip_special_tokens=True)
    
    # Zephyr
    with torch.no_grad():
        inputs = zephyr_tok(prompt, return_tensors="pt", max_length=512, truncation=True)
        z_out = zephyr.generate(
            inputs.input_ids.cuda(),
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=zephyr_tok.pad_token_id
        )
        z_text = zephyr_tok.decode(z_out[0], skip_special_tokens=True)
    
    # Compare
    m_words = set(m_text.lower().split())
    z_words = set(z_text.lower().split())
    similarity = len(m_words & z_words) / len(m_words | z_words) if (m_words | z_words) else 0
    
    results.append({
        "prompt": prompt,
        "similarity": similarity,
        "mistral_len": len(m_text),
        "zephyr_len": len(z_text)
    })
    
    print(f"  Similarity: {similarity:.1%}")

# Analysis
mean_sim = np.mean([r["similarity"] for r in results])
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Mean Similarity: {mean_sim:.1%}")

if mean_sim < 0.7:
    print("✅ SUCCESS: Fine-tuning detected!")
    print("   Models behave differently")
else:
    print("⚠️ Models are similar")

# Save
output = {
    "results": results,
    "mean_similarity": float(mean_sim),
    "conclusion": "fine-tuning detected" if mean_sim < 0.7 else "models similar"
}

save_path = "/content/drive/MyDrive/pot_8bit_results.json"
with open(save_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n💾 Saved to: {save_path}")
print("🎉 Test complete for your paper!")