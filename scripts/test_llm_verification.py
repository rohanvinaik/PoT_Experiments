#!/usr/bin/env python3
"""
LLM Verification Test - Comparing GPT-2 and DistilGPT-2
Tests the LMVerifier module with real language models (no tokens required)
"""

import os
import time
import json
import torch
import pathlib
import sys

# Ensure PYTHONPATH includes the repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- tiny HF adapter (MPS-safe: fp16 + eager attention) ---
from transformers import AutoModelForCausalLM, AutoTokenizer

class HFAdapterLM:
    def __init__(self, model_name: str, device=None, seed: int = 0):
        torch.manual_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.m = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=(torch.float16 if torch.backends.mps.is_available() else None),
            attn_implementation="eager",
        ).eval()
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.m = self.m.to(self.device)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or self.tok.unk_token

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        dev = next(self.m.parameters()).device
        enc = self.tok(prompt, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        out = self.m.generate(
            **enc,
            do_sample=False,            # greedy; no temperature/top_k
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.pad_token_id,
            return_dict_in_generate=True,
        )
        return self.tok.decode(out.sequences[0], skip_special_tokens=True)

def main():
    print("=== LLM VERIFICATION TEST ===")
    print("Testing LMVerifier with GPT-2 vs DistilGPT-2")
    print()
    
    # Check if transformers is available
    try:
        import transformers
    except ImportError:
        print("⚠️  Transformers library not installed. Skipping LLM verification test.")
        print("   Install with: pip install transformers")
        return 0
    
    # --- import PoT verifier (two possible module paths) ---
    try:
        from pot.lm.verifier import LMVerifier
    except Exception:
        try:
            from pot.core.verifier import LMVerifier
        except Exception as e:
            print(f"❌ Could not import LMVerifier: {e}")
            return 1
    
    # --- models (open, no gating) ---
    REF = "gpt2"  # Reference: GPT-2 (124M params)
    NEG = "distilgpt2"  # Candidate: DistilGPT-2 (82M params, distilled version)
    
    print(f"Loading reference model: {REF}")
    try:
        ref = HFAdapterLM(REF, seed=1)
        print(f"✓ Reference model loaded on device: {ref.device}")
    except Exception as e:
        print(f"❌ Could not load reference model {REF}: {e}")
        return 1
    
    print(f"Loading candidate models...")
    cand_pos = HFAdapterLM(REF, device=ref.device, seed=2)
    print(f"✓ Positive candidate (same model, different seed) loaded")
    
    cand_neg = HFAdapterLM(NEG, device=ref.device, seed=3)
    print(f"✓ Negative candidate (DistilGPT-2) loaded")
    print()
    
    # --- verifier setup ---
    print("Initializing LMVerifier...")
    verifier = LMVerifier(
        reference_model=ref, 
        delta=0.01,  # 99% confidence
        use_sequential=True  # Enable sequential testing
    )
    print(f"✓ Verifier initialized")
    print()
    
    def run_case(cand, tag):
        print(f"Running verification: {tag}")
        t0 = time.time()
        
        # Generate challenges
        from pot.core.challenge import generate_challenges
        challenges = generate_challenges(n_challenges=32)
        
        # Run verification
        res = verifier.verify(
            model=cand,
            challenges=challenges,
            tolerance=0.5,  # Distance threshold
            method='fuzzy'  # Use fuzzy hashing
        )
        
        elapsed = time.time() - t0
        
        # Extract results from LMVerificationResult object
        result_data = {
            "case": tag,
            "accepted": res.accepted,
            "distance": res.distance,
            "confidence_radius": res.confidence_radius,
            "n_challenges": res.n_challenges,
            "fuzzy_similarity": res.fuzzy_similarity,
            "elapsed_sec": round(elapsed, 2),
        }
        
        print(json.dumps(result_data, indent=2))
        
        # Save detailed results
        pathlib.Path("experimental_results").mkdir(exist_ok=True)
        with open(f"experimental_results/llm_result_{tag}.json", "w") as f:
            json.dump(result_data, f, indent=2)
        
        return result_data
    
    print("=" * 60)
    print("TEST 1: GPT-2 vs GPT-2 (same model, different seed)")
    print("Reference: GPT-2 (124M params, seed=1)")
    print("Candidate: GPT-2 (124M params, seed=2)")
    print("Expected: ACCEPT / H0 (should identify as SAME model)")
    print("-" * 60)
    res1 = run_case(cand_pos, "gpt2_vs_gpt2_same")
    
    print()
    print("=" * 60)
    print("TEST 2: GPT-2 vs DistilGPT-2 (different models)")
    print("Reference: GPT-2 (124M params)")
    print("Candidate: DistilGPT-2 (82M params, distilled)")
    print("Expected: REJECT / H1 (should identify as DIFFERENT models)")
    print("-" * 60)
    res2 = run_case(cand_neg, "gpt2_vs_distilgpt2_different")
    
    print()
    print("=" * 50)
    print("SUMMARY")
    print("-" * 50)
    
    # Check results
    test1_passed = res1.get("accepted", False) == True
    test2_passed = res2.get("accepted", False) == False
    
    if test1_passed:
        print("✅ Test 1 PASSED: GPT-2 vs Self correctly accepted")
    else:
        print("❌ Test 1 FAILED: GPT-2 vs Self should have been accepted")
    
    if test2_passed:
        print("✅ Test 2 PASSED: GPT-2 vs DistilGPT-2 correctly rejected")
    else:
        print("❌ Test 2 FAILED: GPT-2 vs DistilGPT-2 should have been rejected")
    
    print()
    if test1_passed and test2_passed:
        print("🎉 LLM VERIFICATION TEST: ALL TESTS PASSED")
        return 0
    else:
        print("❌ LLM VERIFICATION TEST: SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())