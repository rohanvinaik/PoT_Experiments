# Run this in a NEW CELL in Colab to check what's happening

# Cell 1: Check if models are loaded in memory
import torch
import gc

print("🔍 Checking what's in memory...")
print("=" * 60)

# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"GPU Memory Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")
else:
    print("No GPU detected")

# Check what objects are loaded
large_objects = []
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) and obj.numel() > 1000000:  # Tensors > 1M params
            large_objects.append(f"Tensor: shape={obj.shape}, device={obj.device}")
    except:
        pass

print(f"\nLarge tensors in memory: {len(large_objects)}")
if len(large_objects) > 0:
    print("Models appear to be loaded!")

# Cell 2: Check disk usage
!df -h /content

# Cell 3: Check what's been downloaded
!du -sh ~/.cache/huggingface/hub/* 2>/dev/null | sort -h | tail -5

# Cell 4: Check running processes
!ps aux | grep python | head -10

# Cell 5: Test if transformers is working
from transformers import AutoTokenizer
print("\n🧪 Testing basic functionality...")
try:
    tok = AutoTokenizer.from_pretrained("gpt2")
    print("✅ Transformers library works")
except Exception as e:
    print(f"❌ Error: {e}")

# Cell 6: Try a minimal test
print("\n🚀 Running minimal model test...")
from transformers import pipeline

try:
    # Use tiny model for quick test
    generator = pipeline('text-generation', model='gpt2')
    result = generator("Hello, I am", max_length=20, num_return_sequences=1)
    print("✅ Pipeline works!")
    print(f"Output: {result[0]['generated_text']}")
except Exception as e:
    print(f"❌ Pipeline failed: {e}")

# Cell 7: Check if models are actually downloading
!ls -la ~/.cache/huggingface/hub/models--*/blobs/*.incomplete 2>/dev/null | head -5

# Cell 8: Kill stuck processes and restart
import os
print("\n🔄 Restarting kernel might help...")
print("Click: Runtime -> Restart runtime")
print("Then run your script again")

# Cell 9: Alternative - try with much smaller model first
print("\n💡 Try this simpler test first:")
print("""
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use tiny GPT-2 (124M params) instead of 7B models
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2") 

inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model.generate(**inputs, max_length=20)
print(tokenizer.decode(outputs[0]))
""")

# Cell 10: Check Colab resources
!cat /proc/meminfo | grep MemTotal
!cat /proc/cpuinfo | grep "model name" | head -1
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader