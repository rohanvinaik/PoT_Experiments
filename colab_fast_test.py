# FAST VERSION - Run this instead!
# Should complete in 2-3 minutes on GPU after models load

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

print("⚡ FAST Mistral vs Zephyr Test")
print("=" * 60)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ WARNING: No GPU! This will be slow!")
    print("Go to Runtime -> Change runtime type -> GPU")

# CRITICAL: Load models properly for GPU
print("\n📥 Loading models WITH GPU optimization...")

# Model 1: Mistral
print("Loading Mistral...")
start = time.time()
mistral = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    torch_dtype=torch.float16,  # MUST use float16 for speed
    device_map="auto",  # MUST auto-map to GPU
    use_cache=True,  # Enable KV cache
)
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
mistral_tok.pad_token = mistral_tok.eos_token
print(f"✅ Mistral loaded in {time.time()-start:.1f}s")

# Model 2: Zephyr  
print("Loading Zephyr...")
start = time.time()
zephyr = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta",
    torch_dtype=torch.float16,
    device_map="auto",
    use_cache=True,
)
zephyr_tok = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
zephyr_tok.pad_token = zephyr_tok.eos_token
print(f"✅ Zephyr loaded in {time.time()-start:.1f}s")

# FAST generation function
@torch.no_grad()
def fast_generate(model, tokenizer, prompt):
    """Optimized generation - should take <1 second on GPU"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with speed optimizations
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,  # SHORT outputs for speed
        min_new_tokens=10,  # Don't stop too early
        do_sample=False,  # Deterministic
        use_cache=True,  # Use KV cache
        pad_token_id=tokenizer.eos_token_id,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Quick test with just 3 prompts
test_prompts = [
    "What is 2+2?",  # Very short
    "Name a color.",  # Very short
    "Define AI.",  # Short
]

print("\n⚡ Running FAST comparison (3 quick tests)...")
print("=" * 60)

similarities = []
total_time = time.time()

for i, prompt in enumerate(test_prompts, 1):
    test_start = time.time()
    print(f"\nTest {i}: {prompt}")
    
    # Generate
    m_out = fast_generate(mistral, mistral_tok, prompt)
    z_out = fast_generate(zephyr, zephyr_tok, prompt)
    
    # Quick similarity
    m_words = set(m_out.lower().split())
    z_words = set(z_out.lower().split())
    sim = len(m_words & z_words) / max(len(m_words | z_words), 1)
    similarities.append(sim)
    
    print(f"  Mistral: {m_out[:50]}...")
    print(f"  Zephyr:  {z_out[:50]}...")
    print(f"  Similarity: {sim:.1%}")
    print(f"  Time: {time.time()-test_start:.1f}s")

avg_sim = sum(similarities) / len(similarities)

print("\n" + "=" * 60)
print("📊 RESULTS")
print("=" * 60)
print(f"Average similarity: {avg_sim:.1%}")
print(f"Total time: {time.time()-total_time:.1f}s")

if avg_sim < 0.8:
    print("✅ FINE-TUNE DETECTED - Models are different!")
else:
    print("⚠️ Models are similar")

# Debug info
print("\n🔍 Debug Info:")
print(f"Mistral on: {next(mistral.parameters()).device}")
print(f"Zephyr on: {next(zephyr.parameters()).device}")
if device == "cuda":
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB used")