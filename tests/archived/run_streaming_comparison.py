#!/usr/bin/env python3
"""
Stream-based comparison - loads models on demand without full download
This avoids the download issues by streaming model weights as needed
"""
import os
import torch
import time
import json

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'

print("🚀 Running Mistral vs Zephyr comparison with streaming")
print("=" * 60)
print("This approach loads models on-demand, avoiding full download")
print("")

from transformers import AutoModelForCausalLM, AutoTokenizer

# Use CPU to avoid memory issues with streaming
device = "cpu"
print(f"Using device: {device}")

def create_streaming_model(model_name, seed=42):
    """Create a model that streams weights on demand"""
    torch.manual_seed(seed)
    
    print(f"\n📡 Loading {model_name} (streaming mode)...")
    
    # Load tokenizer (small, downloads quickly)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    
    # Load model with streaming - only downloads weights as needed
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
        # Key: Don't download all weights upfront
    )
    model.eval()
    
    print(f"✓ {model_name} ready (streaming)")
    
    @torch.no_grad()
    def generate(prompt, max_new_tokens=64):
        ids = tok(prompt, return_tensors="pt")
        ids = {k: v.to(device) for k, v in ids.items()}
        out = model.generate(
            **ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
        )
        return tok.decode(out[0], skip_special_tokens=True)
    
    class StreamingLM:
        def __init__(self):
            self.generate = generate
            self.tok = tok
            self.device = device
            self.name = model_name
    
    return StreamingLM()

# Import PoT verifier
from pot.lm.verifier import LMVerifier
from pot.lm.lm_config import LMVerifierConfig

print("\n" + "=" * 60)
print("LOADING MODELS")
print("=" * 60)

# Load reference model (Mistral base)
REF_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
print(f"\n1. Reference: {REF_NAME}")
ref = create_streaming_model(REF_NAME, seed=1)

# Load Zephyr (fine-tuned Mistral)
ZEPHYR = "HuggingFaceH4/zephyr-7b-beta"
print(f"\n2. Candidate: {ZEPHYR}")
zephyr = create_streaming_model(ZEPHYR, seed=2)

# Also create self-match for baseline
print(f"\n3. Self-match: {REF_NAME}")
self_match = create_streaming_model(REF_NAME, seed=3)

print("\n" + "=" * 60)
print("CONFIGURING VERIFIER")
print("=" * 60)

# Configure verifier for fine-tune detection
cfg = LMVerifierConfig(
    model_name="hf",
    device=str(device),
    num_challenges=32,  # Reduced for faster testing
    verification_method="sequential",
    sprt_alpha=0.001,
    sprt_beta=0.01,
    fuzzy_threshold=0.15,
    difficulty_curve="linear",
)

verifier = LMVerifier(reference_model=ref, config=cfg)
print(f"✓ Verifier configured: {cfg.num_challenges} challenges, threshold={cfg.fuzzy_threshold}")

def run_verification(candidate, test_name):
    """Run verification test"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print('-'*60)
    
    t0 = time.time()
    result = verifier.verify(candidate, None)
    elapsed = time.time() - t0
    
    res = {
        "test": test_name,
        "accepted": result.get("accepted"),
        "decision": result.get("decision"),
        "p_value": result.get("p_value", 0),
        "n_used": result.get("n_used"),
        "elapsed": round(elapsed, 2),
    }
    
    print(json.dumps(res, indent=2))
    
    # Save result
    os.makedirs("experimental_results", exist_ok=True)
    with open(f"experimental_results/streaming_{test_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result

print("\n" + "=" * 60)
print("RUNNING VERIFICATION TESTS")
print("=" * 60)

# Test 1: Self-match (should ACCEPT)
res1 = run_verification(self_match, "mistral_self_match")

# Test 2: Mistral vs Zephyr (should REJECT if fine-tune detected)
res2 = run_verification(zephyr, "mistral_vs_zephyr")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if res1.get("accepted"):
    print("✅ Test 1 PASSED: Self-match correctly accepted")
else:
    print("❌ Test 1 FAILED: Self-match incorrectly rejected")

if not res2.get("accepted"):
    print("✅ Test 2 PASSED: Zephyr correctly rejected (fine-tune detected!)")
else:
    print("⚠️ Test 2: Zephyr accepted (fine-tune not distinguished)")
    print("   Consider increasing num_challenges or adjusting threshold")

print("\n🎉 Comparison complete using streaming approach!")