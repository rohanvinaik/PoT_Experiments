# Colab Debug Commands - Run these in separate cells to diagnose issues

# Cell 1: Check environment
import torch
import sys
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Cell 2: Check disk space
!df -h

# Cell 3: Check what's downloaded
!ls -la ~/.cache/huggingface/hub/ 2>/dev/null || echo "No HF cache yet"

# Cell 4: Test simple download
from transformers import AutoTokenizer
print("Testing tokenizer download...")
tok = AutoTokenizer.from_pretrained("gpt2")
print(f"✅ Tokenizer works! Vocab size: {len(tok)}")

# Cell 5: Monitor download with progress
from transformers import AutoModelForCausalLM
import sys

print("Downloading small model with progress...")
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",  # Small 124M model
    cache_dir="/content/models",
)
print("✅ GPT-2 downloaded successfully!")

# Cell 6: Check network speed
!curl -s https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py | python - --simple

# Cell 7: Try downloading Mistral with error handling
try:
    print("Attempting Mistral download...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print("✅ Success!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTrying alternative:")
    # Try with different settings
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.float32,  # Full precision
        device_map="cpu",  # CPU only
        low_cpu_mem_usage=True,
    )