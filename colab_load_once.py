# LOAD MODELS ONLY ONCE - No duplicates!

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("📍 Smart Model Loading - No Duplicates")
print("=" * 60)

# Set cache directory explicitly
CACHE_DIR = "/content/model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR

print(f"Cache directory: {CACHE_DIR}")

# Check what's already downloaded
!echo "Current cache contents:"
!ls -la /content/model_cache/ 2>/dev/null || echo "Cache empty"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Function to safely load model ONCE
def load_model_once(model_name, cache_dir=CACHE_DIR):
    """Load model, checking cache first"""
    model_path = f"{cache_dir}/{model_name.replace('/', '_')}"
    
    # Check if already downloaded
    if os.path.exists(f"{model_path}/config.json"):
        print(f"✅ Found {model_name} in cache, loading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    else:
        print(f"📥 Downloading {model_name} (only once!)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Save to avoid re-download
        print(f"💾 Saving to {model_path}")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
    
    return model, tokenizer

# Load models (will only download if not in cache)
print("\n1️⃣ Loading Mistral...")
mistral, mistral_tok = load_model_once("mistralai/Mistral-7B-Instruct-v0.3")

print("\n2️⃣ Loading Zephyr...")
zephyr, zephyr_tok = load_model_once("HuggingFaceH4/zephyr-7b-beta")

print("\n✅ Both models loaded!")
print("\n📊 Final disk usage:")
!df -h /
!du -sh /content/model_cache/

# Quick test
print("\n🧪 Quick verification test...")
prompt = "Hello world"
with torch.no_grad():
    # Test Mistral
    inputs = mistral_tok(prompt, return_tensors="pt").to(device)
    m_out = mistral.generate(**inputs, max_new_tokens=10)
    print(f"Mistral: {mistral_tok.decode(m_out[0])[:50]}")
    
    # Test Zephyr  
    inputs = zephyr_tok(prompt, return_tensors="pt").to(device)
    z_out = zephyr.generate(**inputs, max_new_tokens=10)
    print(f"Zephyr: {zephyr_tok.decode(z_out[0])[:50]}")

print("\n🎉 Ready for comparison tests!")
print("Models are loaded in memory, no more downloads needed!")