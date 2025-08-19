#!/usr/bin/env python3
"""
Test Mistral-7B vs Zephyr-7B (fine-tuned version)
This script compares the base Mistral model against Zephyr, which is a fine-tuned version.
"""

import os, json, time, pathlib, torch
import sys

# Ensure PYTHONPATH includes the repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- MPS-safe HF adapter (fp16 + eager attention on MPS) ---
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_lm(model_name: str, device: str | None = None, seed: int = 0):
    """Load a language model with MPS-safe configuration"""
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    kwargs = {}
    if torch.backends.mps.is_available():
        kwargs["torch_dtype"] = torch.float16
        kwargs["attn_implementation"] = "eager"
    try:
        m = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
    except Exception:
        # very rare: try trust_remote_code if model requires it
        m = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kwargs).eval()
    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    m = m.to(dev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token

    @torch.no_grad()
    def generate(prompt: str, max_new_tokens: int = 64) -> str:
        ids = tok(prompt, return_tensors="pt")
        ids = {k: v.to(next(m.parameters()).device) for k, v in ids.items()}
        out = m.generate(
            **ids,
            do_sample=False,             # deterministic
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            return_dict_in_generate=True,
        )
        return tok.decode(out.sequences[0], skip_special_tokens=True)

    # simple adapter object with .generate()
    class _LM: pass
    lm = _LM()
    lm.generate = generate
    lm.device = dev
    lm.name = model_name
    return lm

def main():
    print("=== MISTRAL VS ZEPHYR COMPARISON TEST ===")
    print("Testing base Mistral-7B against Zephyr-7B (fine-tuned)")
    print()
    
    # Check if transformers is available
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers library not installed.")
        print("   Install with: pip install transformers")
        return 1
    
    # Check device availability
    if torch.backends.mps.is_available():
        print(f"✓ Using MPS (Apple Metal) acceleration")
    elif torch.cuda.is_available():
        print(f"✓ Using CUDA GPU acceleration")
    else:
        print(f"ℹ Using CPU (slower performance expected)")
    print()

    # --- import PoT verifier (+config), handle either package layout ---
    try:
        from pot.lm.verifier import LMVerifier
        from pot.lm.lm_config import LMVerifierConfig
        print("✓ Loaded LMVerifier from pot.lm")
    except Exception:
        try:
            from pot.core.verifier import LMVerifier
            from pot.core.lm_config import LMVerifierConfig
            print("✓ Loaded LMVerifier from pot.core")
        except Exception as e:
            print(f"❌ Could not import LMVerifier: {e}")
            return 1

    REF_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    ZEPHYR   = "HuggingFaceH4/zephyr-7b-beta"

    print(f"Loading reference model: {REF_NAME}")
    try:
        # Load reference on MPS if available
        ref = load_lm(REF_NAME, device=None, seed=1)
        print(f"✓ Reference model loaded on {ref.device}")
    except Exception as e:
        print(f"❌ Failed to load reference model: {e}")
        print("   This test requires downloading large models (~14GB each)")
        return 1

    print(f"\nLoading candidate models...")
    
    # Load self-match candidate for baseline
    try:
        cand_self = load_lm(REF_NAME, device=ref.device, seed=3)
        print(f"✓ Self-match candidate loaded (same model, different seed)")
    except Exception as e:
        print(f"⚠ Failed to load self-match: {e}")
        cand_self = None

    # Try loading Zephyr on MPS; if OOM, fall back to CPU
    print(f"Loading Zephyr model: {ZEPHYR}")
    try:
        cand_zephyr = load_lm(ZEPHYR, device=ref.device, seed=2)
        print(f"✓ Zephyr model loaded on {cand_zephyr.device}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "mps" in str(e).lower():
            print(f"⚠ Memory issue on {ref.device}, falling back to CPU for Zephyr")
            try:
                cand_zephyr = load_lm(ZEPHYR, device="cpu", seed=2)
                print(f"✓ Zephyr model loaded on CPU")
            except Exception as e2:
                print(f"❌ Failed to load Zephyr: {e2}")
                return 1
        else:
            print(f"❌ Failed to load Zephyr: {e}")
            return 1
    print()

    # For near-siblings like Zephyr (a Mistral fine-tune), use more challenges + stricter threshold
    print("Configuring verifier for fine-tune detection...")
    cfg = LMVerifierConfig(
        model_name="hf",
        device=str(ref.device),
        num_challenges=96,          # More challenges for fine-tune detection
        verification_method="sequential",
        sprt_alpha=0.001,           # FAR target
        sprt_beta=0.01,             # FRR target
        fuzzy_threshold=0.15,       # Stricter than default 0.20
        difficulty_curve="linear",
    )

    print(f"✓ Verifier configured:")
    print(f"  - Challenges: {cfg.num_challenges}")
    print(f"  - Fuzzy threshold: {cfg.fuzzy_threshold} (stricter for fine-tune detection)")
    print(f"  - Method: {cfg.verification_method}")
    print()

    verifier = LMVerifier(reference_model=ref, config=cfg)
    
    # Create output directory
    pathlib.Path("experimental_results").mkdir(exist_ok=True)

    def run_case(cand, tag, expected):
        """Run verification case and display results"""
        print("=" * 60)
        print(f"TEST: {tag}")
        print(f"Expected: {expected}")
        print("-" * 60)
        
        t0 = time.time()
        if hasattr(verifier, "verify_enhanced"):
            res = verifier.verify_enhanced(cand, None)
        else:
            res = verifier.verify(cand, None)
        res["elapsed_sec"] = time.time() - t0
        
        # Display results
        accepted = res.get("accepted", False)
        decision = res.get("decision", "N/A")
        
        result_summary = {
            "case": tag,
            "accepted": accepted,
            "decision": decision,
            "p_value": res.get("p_value"),
            "threshold": res.get("threshold"),
            "n_used": res.get("n_used"),
            "elapsed_sec": round(res["elapsed_sec"], 2),
            "ref_device": str(ref.device),
            "cand_device": str(cand.device) if hasattr(cand, 'device') else "?",
        }
        
        print(json.dumps(result_summary, indent=2))
        
        # Save detailed results
        output_file = f"experimental_results/mistral_zephyr_{tag}.json"
        with open(output_file, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
        
        # Check if result matches expectation
        if "ACCEPT" in expected and accepted:
            print("✅ PASSED: Result matches expectation (ACCEPT)")
            return True
        elif "REJECT" in expected and not accepted:
            print("✅ PASSED: Result matches expectation (REJECT)")
            return True
        else:
            print(f"❌ FAILED: Expected {expected}, got {'ACCEPT' if accepted else 'REJECT'}")
            return False

    print("\n" + "=" * 60)
    print("RUNNING VERIFICATION TESTS")
    print("=" * 60)
    
    results = []
    
    # 1) Baseline: Mistral vs Mistral (EXPECT: ACCEPT / H0)
    if cand_self:
        passed = run_case(cand_self, "mistral_vs_mistral", "ACCEPT (same model)")
        results.append(("Mistral vs Mistral", passed))
        print()

    # 2) Main test: Mistral vs Zephyr (EXPECT: REJECT / H1 if verifier can distinguish)
    passed = run_case(cand_zephyr, "mistral_vs_zephyr", "REJECT (fine-tuned model)")
    results.append(("Mistral vs Zephyr", passed))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 SUCCESS: The verifier correctly distinguishes between base and fine-tuned models!")
    elif results[-1][1]:  # If Zephyr test passed
        print("\n✓ The verifier successfully detected the fine-tuned model (Zephyr)")
    else:
        print("\n⚠ The verifier could not distinguish the fine-tuned model from the base model")
        print("  Consider increasing num_challenges or adjusting fuzzy_threshold")
    
    return 0 if total_passed == total_tests else 1

if __name__ == "__main__":
    sys.exit(main())