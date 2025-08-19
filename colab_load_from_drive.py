# Load Mistral from Google Drive in Colab

from google.colab import drive
drive.mount('/content/drive')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("📥 Loading Mistral from your Google Drive...")

# Path to your uploaded files
model_path = "/content/drive/MyDrive/mistral_model/"

# Load model directly from Drive files
mistral = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

mistral_tok = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

print("✅ Mistral loaded from Drive!")

# Now only download Zephyr
print("\n📥 Downloading Zephyr (only 14GB instead of 28GB)...")
zephyr = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta",
    torch_dtype=torch.float16,
    device_map="auto"
)
zephyr_tok = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

print("✅ Both models ready!")