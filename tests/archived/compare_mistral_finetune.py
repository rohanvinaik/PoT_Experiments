#!/usr/bin/env python3
"""
One-shot script to compare Mistral-7B base model against fine-tuned versions.
This demonstrates the PoT framework's ability to detect fine-tuning modifications.

Usage:
    # Default: Compare Mistral vs Zephyr (a popular fine-tune)
    python scripts/compare_mistral_finetune.py
    
    # Custom fine-tuned model
    python scripts/compare_mistral_finetune.py --finetuned "your-org/your-fine-tuned-model"
    
    # Adjust verification strictness
    python scripts/compare_mistral_finetune.py --num-challenges 128 --threshold 0.10
"""

import os
import sys
import json
import time
import torch
import pathlib
import argparse
from typing import Optional

# Ensure repository root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set MPS fallback for Apple Silicon
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


class HFModelAdapter:
    """Lightweight adapter for HuggingFace language models with PoT compatibility"""
    
    def __init__(self, model_name: str, device: Optional[str] = None, seed: int = 0):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        torch.manual_seed(seed)
        print(f"Loading {model_name}...")
        
        # Load tokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        # Configure for device (MPS-safe settings)
        kwargs = {}
        if torch.backends.mps.is_available():
            kwargs["torch_dtype"] = torch.float16
            kwargs["attn_implementation"] = "eager"
        
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **kwargs
            ).eval()
        except Exception:
            # Some models require trust_remote_code
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, **kwargs
            ).eval()
        
        # Set device
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Ensure pad token is set
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or self.tok.unk_token
        
        self.name = model_name
        print(f"✓ Loaded on {self.device}")
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Generate text from prompt (deterministic/greedy decoding)"""
        ids = self.tok(prompt, return_tensors="pt")
        ids = {k: v.to(self.device) for k, v in ids.items()}
        
        out = self.model.generate(
            **ids,
            do_sample=False,  # Deterministic
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.pad_token_id,
            return_dict_in_generate=True,
        )
        
        return self.tok.decode(out.sequences[0], skip_special_tokens=True)


def run_verification(
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    finetuned_model: str = "HuggingFaceH4/zephyr-7b-beta",
    num_challenges: int = 96,
    fuzzy_threshold: float = 0.15,
    output_dir: str = "experimental_results",
):
    """
    Run verification comparing base model against fine-tuned version.
    
    Args:
        base_model: HuggingFace model ID for base model
        finetuned_model: HuggingFace model ID for fine-tuned model
        num_challenges: Number of challenges for verification (more = stricter)
        fuzzy_threshold: Similarity threshold (lower = stricter)
        output_dir: Directory to save results
    """
    
    print("=" * 70)
    print("PROOF-OF-TRAINING: FINE-TUNE DETECTION TEST")
    print("=" * 70)
    print(f"Base Model:      {base_model}")
    print(f"Fine-tuned Model: {finetuned_model}")
    print(f"Challenges:      {num_challenges}")
    print(f"Threshold:       {fuzzy_threshold}")
    print("=" * 70)
    print()
    
    # Check dependencies
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Error: transformers not installed")
        print("   Install with: pip install transformers torch")
        return 1
    
    # Check device
    if torch.backends.mps.is_available():
        print("✓ Using MPS (Apple Metal) acceleration")
    elif torch.cuda.is_available():
        print("✓ Using CUDA GPU acceleration")
    else:
        print("ℹ Using CPU (slower performance)")
    print()
    
    # Import PoT verifier
    try:
        from pot.lm.verifier import LMVerifier
        from pot.lm.lm_config import LMVerifierConfig
        print("✓ PoT LMVerifier loaded")
    except ImportError as e:
        print(f"❌ Error loading LMVerifier: {e}")
        return 1
    print()
    
    # Load models
    print("LOADING MODELS")
    print("-" * 70)
    
    # Load base model (reference)
    try:
        ref_model = HFModelAdapter(base_model, seed=1)
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        print("   Note: Model will be downloaded on first run (~14GB)")
        return 1
    
    # Load fine-tuned model (candidate)
    try:
        cand_finetuned = HFModelAdapter(finetuned_model, device=ref_model.device, seed=2)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠ Memory issue on GPU, falling back to CPU for fine-tuned model")
            cand_finetuned = HFModelAdapter(finetuned_model, device="cpu", seed=2)
        else:
            raise
    except Exception as e:
        print(f"❌ Failed to load fine-tuned model: {e}")
        print("   Note: Model will be downloaded on first run (~14GB)")
        return 1
    
    # Also load self-match for baseline
    print("\nLoading self-match baseline...")
    cand_self = HFModelAdapter(base_model, device=ref_model.device, seed=3)
    print()
    
    # Configure verifier
    print("CONFIGURING VERIFIER")
    print("-" * 70)
    config = LMVerifierConfig(
        model_name="hf",
        device=str(ref_model.device),
        num_challenges=num_challenges,
        verification_method="sequential",
        sprt_alpha=0.001,  # False Accept Rate target
        sprt_beta=0.01,    # False Reject Rate target
        fuzzy_threshold=fuzzy_threshold,
        difficulty_curve="linear",
    )
    
    verifier = LMVerifier(reference_model=ref_model, config=config)
    print(f"✓ Verifier configured")
    print(f"  - Method: Sequential testing with early stopping")
    print(f"  - FAR target: 0.1%")
    print(f"  - FRR target: 1%")
    print()
    
    # Create output directory
    pathlib.Path(output_dir).mkdir(exist_ok=True)
    
    def run_test(candidate, test_name, expected_result):
        """Run a single verification test"""
        print("=" * 70)
        print(f"TEST: {test_name}")
        print(f"Expected: {expected_result}")
        print("-" * 70)
        
        t0 = time.time()
        
        # Run verification
        if hasattr(verifier, "verify_enhanced"):
            result = verifier.verify_enhanced(candidate, None)
        else:
            result = verifier.verify(candidate, None)
        
        elapsed = time.time() - t0
        
        # Prepare summary
        accepted = result.get("accepted", False)
        summary = {
            "test": test_name,
            "accepted": accepted,
            "decision": result.get("decision", "N/A"),
            "p_value": result.get("p_value"),
            "threshold": result.get("threshold"),
            "challenges_used": result.get("n_used"),
            "time_seconds": round(elapsed, 2),
            "ref_device": str(ref_model.device),
            "cand_device": str(candidate.device),
        }
        
        # Display result
        print(json.dumps(summary, indent=2))
        
        # Save detailed results
        output_file = f"{output_dir}/finetune_test_{test_name.replace(' ', '_').lower()}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
        
        # Check if result matches expectation
        if expected_result == "ACCEPT" and accepted:
            print("✅ PASSED: Model correctly accepted")
            return True
        elif expected_result == "REJECT" and not accepted:
            print("✅ PASSED: Model correctly rejected")
            return True
        else:
            print(f"❌ FAILED: Expected {expected_result}, got {'ACCEPT' if accepted else 'REJECT'}")
            return False
    
    # Run tests
    print("RUNNING VERIFICATION TESTS")
    print("=" * 70)
    
    results = []
    
    # Test 1: Baseline (same model, different seed)
    passed = run_test(cand_self, "Self Match", "ACCEPT")
    results.append(("Self Match (baseline)", passed))
    print()
    
    # Test 2: Fine-tuned model
    model_name = finetuned_model.split("/")[-1]
    passed = run_test(cand_finetuned, f"Fine-tune ({model_name})", "REJECT")
    results.append((f"Fine-tune Detection ({model_name})", passed))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 SUCCESS: The PoT verifier correctly distinguishes fine-tuned models!")
        print("This demonstrates the framework's ability to detect model modifications.")
    elif results[1][1]:  # If fine-tune test passed
        print("\n✓ The verifier successfully detected the fine-tuned model")
        print("This validates the PoT framework's fine-tune detection capability.")
    else:
        print("\n⚠ The verifier could not distinguish the fine-tuned model")
        print("Consider adjusting parameters:")
        print("  - Increase --num-challenges (e.g., 128 or 192)")
        print("  - Decrease --threshold (e.g., 0.10 or 0.05)")
    
    return 0 if total_passed == total_tests else 1


def main():
    parser = argparse.ArgumentParser(
        description="Compare Mistral base model against fine-tuned versions"
    )
    parser.add_argument(
        "--base",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Base model ID (default: Mistral-7B-Instruct-v0.3)"
    )
    parser.add_argument(
        "--finetuned",
        default="HuggingFaceH4/zephyr-7b-beta",
        help="Fine-tuned model ID (default: Zephyr-7B-beta)"
    )
    parser.add_argument(
        "--num-challenges",
        type=int,
        default=96,
        help="Number of challenges (default: 96, more = stricter)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Fuzzy similarity threshold (default: 0.15, lower = stricter)"
    )
    parser.add_argument(
        "--output-dir",
        default="experimental_results",
        help="Output directory for results (default: experimental_results)"
    )
    
    args = parser.parse_args()
    
    return run_verification(
        base_model=args.base,
        finetuned_model=args.finetuned,
        num_challenges=args.num_challenges,
        fuzzy_threshold=args.threshold,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())