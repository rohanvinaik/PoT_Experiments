# Google Colab: Mistral vs Zephyr Fine-tune Detection
# Just copy this entire cell into Colab and run!

!pip install -q transformers torch accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

print("🚀 Mistral vs Zephyr Comparison on Colab")
print("=" * 60)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n📥 Loading models (this will download ~28GB)...")

# Load Mistral base model
print("\n1. Loading Mistral-7B base...")
mistral_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    torch_dtype=torch.float16,
    device_map="auto"
)
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
print("✅ Mistral loaded!")

# Load Zephyr (fine-tuned Mistral)
print("\n2. Loading Zephyr-7B (fine-tuned)...")
zephyr_model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta",
    torch_dtype=torch.float16,
    device_map="auto"
)
zephyr_tok = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
print("✅ Zephyr loaded!")

# Test prompts
test_prompts = [
    "What is machine learning?",
    "Write a Python hello world program.",
    "Explain gravity in simple terms.",
    "What are the benefits of reading?",
    "How do computers work?",
]

print("\n🔬 Running comparison tests...")
print("=" * 60)

@torch.no_grad()
def generate(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Compare outputs
similarities = []
for i, prompt in enumerate(test_prompts, 1):
    print(f"\n📝 Test {i}: {prompt}")
    
    # Generate responses
    mistral_out = generate(mistral_model, mistral_tok, prompt, 50)
    zephyr_out = generate(zephyr_model, zephyr_tok, prompt, 50)
    
    # Simple similarity check (character overlap)
    mistral_words = set(mistral_out.lower().split())
    zephyr_words = set(zephyr_out.lower().split())
    
    if len(mistral_words | zephyr_words) > 0:
        similarity = len(mistral_words & zephyr_words) / len(mistral_words | zephyr_words)
    else:
        similarity = 0
    
    similarities.append(similarity)
    print(f"   Similarity: {similarity:.2%}")

# Results
avg_similarity = sum(similarities) / len(similarities)
print("\n" + "=" * 60)
print("📊 RESULTS")
print("=" * 60)
print(f"Average similarity: {avg_similarity:.2%}")

if avg_similarity < 0.7:
    print("✅ FINE-TUNE DETECTED: Models are significantly different")
    print("   Zephyr shows distinct behavior from base Mistral")
else:
    print("⚠️ Models are very similar")
    print("   This could indicate same base model or minimal fine-tuning")

print("\n🎉 Test complete!")
print("For paper results, use the full PoT framework with more challenges")