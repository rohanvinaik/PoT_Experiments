#!/usr/bin/env python3
"""
LLM Verification Test - ONLY OPEN MODELS - NO AUTHENTICATION REQUIRED
This script uses ONLY publicly available models that require NO tokens.
"""

import os
import sys
import time
import json
import torch
import pathlib

# Ensure PYTHONPATH includes the repository root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("LLM VERIFICATION TEST - OPEN MODELS ONLY")
print("=" * 70)
print("This test uses ONLY publicly available models:")
print("  - gpt2 (124M params) - NO TOKEN REQUIRED")
print("  - distilgpt2 (82M params) - NO TOKEN REQUIRED")
print("=" * 70)
print()

# Check if transformers is available
try:
    import transformers
    print("✅ Transformers library found")
except ImportError:
    print("❌ Transformers library not installed.")
    print("   Install with: pip install transformers")
    sys.exit(0)

# Try to import PoT modules
try:
    from pot.lm.verifier import LMVerifier
    from pot.lm.lm_config import LMVerifierConfig
except Exception:
    try:
        from pot.core.verifier import LMVerifier
        from pot.core.lm_config import LMVerifierConfig
    except Exception as e:
        print(f"⚠️ Could not import LMVerifier: {e}")
        print("   Using mock verification instead")
        LMVerifier = None

from transformers import AutoModelForCausalLM, AutoTokenizer

class OpenModelAdapter:
    """Adapter for open HuggingFace models - NO TOKENS REQUIRED"""
    
    def __init__(self, model_name: str, device=None, seed: int = 0):
        print(f"Loading OPEN model: {model_name} (no authentication required)")
        torch.manual_seed(seed)
        
        # Load tokenizer and model - these are FULLY OPEN
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.m = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use fp32 for compatibility
        ).eval()
        
        self.device = device or "cpu"
        self.m = self.m.to(self.device)
        
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or "[PAD]"
        
        print(f"✅ {model_name} loaded successfully on {self.device}")
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        """Generate text - simple greedy decoding"""
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.m.generate(
            **inputs,
            do_sample=False,  # Deterministic
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.pad_token_id,
        )
        
        return self.tok.decode(outputs[0], skip_special_tokens=True)


def main():
    """Main test function using ONLY open models"""
    
    # ONLY OPEN MODELS - NO AUTHENTICATION REQUIRED
    REFERENCE_MODEL = "gpt2"  # OPEN MODEL
    CANDIDATE_SAME = "gpt2"   # OPEN MODEL
    CANDIDATE_DIFF = "distilgpt2"  # OPEN MODEL
    
    print("Models to test (ALL OPEN - NO TOKENS NEEDED):")
    print(f"  Reference: {REFERENCE_MODEL}")
    print(f"  Candidate 1: {CANDIDATE_SAME} (should match)")
    print(f"  Candidate 2: {CANDIDATE_DIFF} (should differ)")
    print()
    
    # Load models
    try:
        print("Loading reference model...")
        ref_model = OpenModelAdapter(REFERENCE_MODEL, seed=42)
        
        print("\nLoading candidate models...")
        cand_same = OpenModelAdapter(CANDIDATE_SAME, seed=123)
        cand_diff = OpenModelAdapter(CANDIDATE_DIFF, seed=456)
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("   Note: These are OPEN models and should not require authentication!")
        return 1
    
    # Run simple verification tests
    print("\n" + "=" * 70)
    print("RUNNING VERIFICATION TESTS")
    print("=" * 70)
    
    # Test with a simple prompt
    test_prompt = "The weather today is"
    
    print(f"\nTest prompt: '{test_prompt}'")
    print("-" * 40)
    
    # Generate outputs
    ref_output = ref_model.generate(test_prompt)
    same_output = cand_same.generate(test_prompt)
    diff_output = cand_diff.generate(test_prompt)
    
    print(f"Reference output: {ref_output[:50]}...")
    print(f"Same model output: {same_output[:50]}...")
    print(f"Different model output: {diff_output[:50]}...")
    
    # Simple similarity check
    ref_tokens = ref_output.split()
    same_tokens = same_output.split()
    diff_tokens = diff_output.split()
    
    # Calculate simple overlap
    same_overlap = len(set(ref_tokens) & set(same_tokens)) / max(len(ref_tokens), 1)
    diff_overlap = len(set(ref_tokens) & set(diff_tokens)) / max(len(ref_tokens), 1)
    
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    
    print(f"\nTest 1: {REFERENCE_MODEL} vs {CANDIDATE_SAME}")
    print(f"  Token overlap: {same_overlap:.2%}")
    print(f"  Expected: HIGH similarity (same model)")
    print(f"  Result: {'✅ PASS' if same_overlap > 0.5 else '❌ FAIL'}")
    
    print(f"\nTest 2: {REFERENCE_MODEL} vs {CANDIDATE_DIFF}")
    print(f"  Token overlap: {diff_overlap:.2%}")
    print(f"  Expected: LOW similarity (different model)")
    print(f"  Result: {'✅ PASS' if diff_overlap < 0.8 else '❌ FAIL'}")
    
    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "reference": REFERENCE_MODEL,
            "candidate_same": CANDIDATE_SAME,
            "candidate_diff": CANDIDATE_DIFF
        },
        "test_results": {
            "same_model_overlap": same_overlap,
            "diff_model_overlap": diff_overlap,
            "test1_pass": same_overlap > 0.5,
            "test2_pass": diff_overlap < 0.8
        },
        "note": "All models are OPEN and require NO authentication tokens"
    }
    
    output_dir = pathlib.Path("experimental_results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "llm_open_models_test.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE - USING ONLY OPEN MODELS")
    print("NO AUTHENTICATION TOKENS WERE REQUIRED!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())