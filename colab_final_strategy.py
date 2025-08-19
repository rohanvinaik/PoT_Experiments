# FINAL OPTIMIZED COLAB SCRIPT
# Upload only Mistral, download only Zephyr

from google.colab import drive
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

print("🚀 Optimized Model Loading Strategy")
print("=" * 60)

# Mount Drive
drive.mount('/content/drive')

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
# Clear any previous runs
torch.cuda.empty_cache()
gc.collect()

print("\n1️⃣ Loading Mistral from your Drive (no download!)...")
mistral_path = "/content/drive/MyDrive/mistral_model/"

try:
    mistral = AutoModelForCausalLM.from_pretrained(
        mistral_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True  # Reduce RAM usage
    )
    mistral_tok = AutoTokenizer.from_pretrained(mistral_path, local_files_only=True)
    print("✅ Mistral loaded from Drive!")
except:
    print("❌ Mistral not found in Drive. Upload files first!")
    print("Need: model-00001/2/3-of-00003.safetensors + config files")

print("\n2️⃣ Downloading Zephyr (14GB)...")
zephyr = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
zephyr_tok = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
print("✅ Zephyr downloaded!")

print("\n" + "="*60)
print("🔬 Running Comparison Test")
print("="*60)

# Quick comparison
test_prompt = "What is machine learning?"

with torch.no_grad():
    # Mistral
    inputs = mistral_tok(test_prompt, return_tensors="pt").to(device)
    m_out = mistral.generate(**inputs, max_new_tokens=30)
    m_text = mistral_tok.decode(m_out[0], skip_special_tokens=True)
    
    # Zephyr
    inputs = zephyr_tok(test_prompt, return_tensors="pt").to(device)
    z_out = zephyr.generate(**inputs, max_new_tokens=30)
    z_text = zephyr_tok.decode(z_out[0], skip_special_tokens=True)

print(f"Mistral: {m_text[:100]}...")
print(f"Zephyr: {z_text[:100]}...")

# Calculate similarity
m_words = set(m_text.lower().split())
z_words = set(z_text.lower().split())
similarity = len(m_words & z_words) / len(m_words | z_words)

print(f"\nSimilarity: {similarity:.1%}")
if similarity < 0.7:
    print("✅ Fine-tuning detected - models are different!")
else:
    print("⚠️ Models are similar")

print("\n🎉 Test complete! Now run full PoT framework test for paper.")