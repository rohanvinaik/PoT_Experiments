# QUICK SETUP: Use Your Local Mistral Model in Colab

# ============================================
# STEP 1: On your Mac, create a zip file
# ============================================
# Run this in your Mac terminal:
'''
cd ~/.cache/huggingface/hub/
zip -r ~/Desktop/mistral_model.zip models--mistralai--Mistral-7B-Instruct-v0.3/
'''
# This creates mistral_model.zip on your Desktop (~13-14GB compressed)

# ============================================
# STEP 2: Upload to Google Drive
# ============================================
# 1. Go to drive.google.com
# 2. Upload mistral_model.zip from Desktop
# 3. Wait for upload (10-20 min depending on connection)

# ============================================
# STEP 3: In Colab, run this code
# ============================================

print("🚀 Setting up Mistral from your uploaded model")
print("=" * 60)

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Check if zip exists
import os
zip_path = "/content/drive/MyDrive/mistral_model.zip"
if os.path.exists(zip_path):
    print("✅ Found mistral_model.zip in Drive!")
    
    # Unzip (takes ~1 minute)
    print("📦 Unzipping model files...")
    !unzip -q /content/drive/MyDrive/mistral_model.zip -d /content/
    
    # Set HuggingFace to use these files
    os.environ['HF_HOME'] = '/content/'
    os.environ['TRANSFORMERS_CACHE'] = '/content/'
    
    print("✅ Model extracted and cache configured!")
else:
    print("❌ mistral_model.zip not found in Drive")
    print("Please upload it first")

# ============================================
# STEP 4: Load model from cache (no download!)
# ============================================

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🔧 Using device: {device}")

print("\n📥 Loading Mistral from cache (no download needed)...")
try:
    mistral = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        local_files_only=True,  # ONLY use cached files
        torch_dtype=torch.float16,
        device_map="auto"
    )
    mistral_tok = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        local_files_only=True
    )
    print("✅ Mistral loaded from your cache!")
    
    # Test it
    print("\n🧪 Testing model...")
    inputs = mistral_tok("Hello, world!", return_tensors="pt").to(device)
    with torch.no_grad():
        output = mistral.generate(**inputs, max_new_tokens=10)
    print(f"Output: {mistral_tok.decode(output[0])}")
    
except Exception as e:
    print(f"❌ Error loading from cache: {e}")
    print("\nTroubleshooting:")
    print("1. Check if unzip completed")
    print("2. Check cache structure:")
    !ls -la /content/models--mistralai--Mistral-7B-Instruct-v0.3/

print("\n" + "="*60)
print("Now only Zephyr needs to download!")
print("This saves 50% of download time!")

# ============================================
# STEP 5: Continue with Zephyr download
# ============================================

print("\n📥 Downloading only Zephyr (Mistral already loaded)...")
zephyr = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta",
    torch_dtype=torch.float16,
    device_map="auto"
)
zephyr_tok = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

print("✅ Both models ready!")
print("\nNow run your comparison tests...")