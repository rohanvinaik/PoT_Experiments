#!/usr/bin/env python3
import os, json, time, pathlib, torch
# --- MPS-safe HF adapter (fp16 + eager attention on MPS) ---
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_lm(model_name: str, device: str | None = None, seed: int = 0):
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

    # simple adapter object with .generate() and .tok for verifier
    class _LM: pass
    lm = _LM()
    lm.generate = generate
    lm.device = dev
    lm.name = model_name
    lm.tok = tok  # Add tokenizer for verifier
    return lm

# --- import PoT verifier (+config) ---
from pot.lm.verifier import LMVerifier
from pot.lm.lm_config import LMVerifierConfig

REF_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
ZEPHYR   = "HuggingFaceH4/zephyr-7b-beta"

print(f"Loading reference model: {REF_NAME}")
# Load reference on MPS if available
ref = load_lm(REF_NAME, device=None, seed=1)
print(f"✓ Reference loaded on {ref.device}")

print(f"\nLoading Zephyr model: {ZEPHYR}")
# Try loading Zephyr on MPS; if OOM, fall back to CPU
try:
    cand_zephyr = load_lm(ZEPHYR, device=ref.device, seed=2)
    print(f"✓ Zephyr loaded on {cand_zephyr.device}")
except RuntimeError as e:
    if "out of memory" in str(e).lower() or "mps" in str(e).lower():
        print(f"⚠ Memory issue, loading Zephyr on CPU")
        cand_zephyr = load_lm(ZEPHYR, device="cpu", seed=2)
        print(f"✓ Zephyr loaded on CPU")
    else:
        raise

# Also build a self-match candidate for baseline ACCEPT
print(f"\nLoading self-match candidate...")
cand_self = load_lm(REF_NAME, device=ref.device, seed=3)
print(f"✓ Self-match loaded on {cand_self.device}")

# For near-siblings like Zephyr (a Mistral fine-tune), use more challenges + slightly stricter fuzzy threshold
print("\nConfiguring verifier for fine-tune detection...")
cfg = LMVerifierConfig(
    model_name="hf",
    device=str(ref.device),          # "mps"
    num_challenges=96,          # bump if you want even more bite (e.g., 128–192)
    verification_method="sequential",
    sprt_alpha=0.001,           # FAR
    sprt_beta=0.01,             # FRR
    fuzzy_threshold=0.15,       # stricter than 0.20
    difficulty_curve="linear",
)

verifier = LMVerifier(reference_model=ref, config=cfg)
print(f"✓ Verifier configured with {cfg.num_challenges} challenges, threshold={cfg.fuzzy_threshold}")

pathlib.Path("experimental_results").mkdir(exist_ok=True)

def run_case(cand, tag):
    print(f"\n{'='*60}")
    print(f"TEST: {tag}")
    print('-'*60)
    t0 = time.time()
    res = verifier.verify_enhanced(cand, None) if hasattr(verifier, "verify_enhanced") else verifier.verify(cand, None)
    res["elapsed_sec"] = time.time() - t0
    result = {
        "case": tag,
        "accepted": res.get("accepted"),
        "decision": res.get("decision"),
        "p_value": res.get("p_value"),
        "threshold": res.get("threshold"),
        "n_used": res.get("n_used"),
        "elapsed_sec": round(res["elapsed_sec"], 2),
        "ref_device": str(getattr(ref, "device", "?")),
        "cand_device": str(getattr(cand, "device", "?")),
    }
    print(json.dumps(result, indent=2))
    with open(f"experimental_results/result_{tag}.json", "w") as f:
        json.dump(res, f, indent=2)
    return res

print("\n" + "="*60)
print("RUNNING VERIFICATION TESTS")
print("="*60)

# 1) Baseline: Mistral vs Mistral (EXPECT: ACCEPT / H0)
res1 = run_case(cand_self, "mistral_vs_mistral")

# 2) Probe: Mistral vs Zephyr (EXPECT: REJECT / H1 if verifier separates fine-tune from base)
res2 = run_case(cand_zephyr, "mistral_vs_zephyr")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
if res1.get("accepted"):
    print("✅ Test 1: Mistral vs Mistral - ACCEPTED (as expected)")
else:
    print("❌ Test 1: Mistral vs Mistral - REJECTED (unexpected)")

if not res2.get("accepted"):
    print("✅ Test 2: Mistral vs Zephyr - REJECTED (fine-tune detected!)")
else:
    print("⚠️  Test 2: Mistral vs Zephyr - ACCEPTED (fine-tune not distinguished)")
    print("   Consider increasing num_challenges or adjusting threshold")