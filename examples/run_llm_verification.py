#!/usr/bin/env python3
"""
Example script for Language Model Verification using the PoT system.

This demonstrates how to:
1. Load reference and candidate models
2. Generate verification challenges
3. Run the verification protocol
4. Interpret the results

Requirements:
    pip install torch transformers sentencepiece
"""

import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import PoT verification components
from pot.lm.verifier import LMVerifier
from pot.lm.models import LM


class HuggingFaceLM:
    """Adapter for HuggingFace models to work with PoT verifier."""
    
    def __init__(self, model_name: str, device=None, seed: int = 0):
        """
        Initialize HuggingFace language model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on (auto-detect if None)
            seed: Random seed for reproducibility
        """
        torch.manual_seed(seed)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        # Determine device and dtype
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                dtype = torch.float16
            elif torch.backends.mps.is_available():
                self.device = "mps"
                dtype = torch.float16
            else:
                self.device = "cpu"
                dtype = torch.float32
        else:
            self.device = device
            dtype = torch.float16 if device != "cpu" else torch.float32
        
        # Load model with appropriate settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if self.device == "cuda" else None
        ).eval()
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        # Set pad token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text including prompt
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            do_sample=False,  # Deterministic generation
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Decode and return
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def create_test_challenges():
    """Create a set of test challenges for verification."""
    
    # Mix of different challenge types
    challenges = [
        # Simple string prompts
        "Complete this sentence: The sky is",
        "What is 2 + 2?",
        "Continue the story: Once upon a time",
        
        # Dict-based challenges with prompts
        {"prompt": "Translate to French: Hello", "difficulty": 0.3},
        {"prompt": "What color is grass?", "difficulty": 0.2},
        {"prompt": "Complete: Roses are red, violets are", "difficulty": 0.4},
        
        # More complex prompts
        {"prompt": "Explain quantum computing in one sentence:", "difficulty": 0.8},
        {"prompt": "Write a haiku about artificial intelligence:", "difficulty": 0.9},
    ]
    
    return challenges


def run_verification(reference_model, candidate_model, model_name="candidate"):
    """
    Run verification protocol on a candidate model.
    
    Args:
        reference_model: Reference model to verify against
        candidate_model: Model to verify
        model_name: Name for logging
        
    Returns:
        Verification result object
    """
    print(f"\n{'='*60}")
    print(f"Verifying: {model_name}")
    print('='*60)
    
    # Create verifier
    verifier = LMVerifier(
        reference_model=reference_model,
        delta=0.01,  # 99% confidence
        use_sequential=True  # Enable early stopping
    )
    
    # Generate challenges
    challenges = create_test_challenges()
    print(f"Using {len(challenges)} challenges")
    
    # Run verification
    start_time = time.time()
    result = verifier.verify(
        model=candidate_model,
        challenges=challenges,
        tolerance=0.5,  # Distance threshold
        method='fuzzy'  # Use fuzzy hashing
    )
    elapsed = time.time() - start_time
    
    # Display results
    print(f"\nResult: {'✅ ACCEPTED' if result.accepted else '❌ REJECTED'}")
    print(f"Distance: {result.distance:.4f}")
    print(f"Confidence radius: {result.confidence_radius:.4f}")
    print(f"Fuzzy similarity: {result.fuzzy_similarity:.4f}")
    print(f"Challenges evaluated: {result.n_challenges}")
    print(f"Time taken: {elapsed:.2f} seconds")
    
    return result


def main():
    """Main execution function."""
    
    print("🚀 PoT Language Model Verification Example")
    print("="*60)
    
    # Configuration
    # For testing, you can use smaller models like:
    # - "gpt2" (124M params)
    # - "distilgpt2" (82M params)
    # - "microsoft/DialoGPT-small" (117M params)
    
    # For production, use larger models like:
    # - "mistralai/Mistral-7B-Instruct-v0.3"
    # - "meta-llama/Llama-2-7b-chat-hf"
    
    REFERENCE_MODEL = "gpt2"  # Change to your preferred model
    CANDIDATE_SAME = "gpt2"   # Same model (should pass)
    CANDIDATE_DIFF = "distilgpt2"  # Different model (should fail)
    
    # Load models
    print(f"\n📦 Loading models...")
    print(f"Reference: {REFERENCE_MODEL}")
    reference = HuggingFaceLM(REFERENCE_MODEL, seed=42)
    
    print(f"Candidate 1 (same): {CANDIDATE_SAME}")
    candidate_same = HuggingFaceLM(CANDIDATE_SAME, device=reference.device, seed=123)
    
    print(f"Candidate 2 (different): {CANDIDATE_DIFF}")
    candidate_diff = HuggingFaceLM(CANDIDATE_DIFF, device=reference.device, seed=456)
    
    # Create output directory
    output_dir = Path("verification_results")
    output_dir.mkdir(exist_ok=True)
    
    # Test 1: Same model (should ACCEPT)
    print("\n" + "="*60)
    print("TEST 1: Verifying same model with different seed")
    result1 = run_verification(reference, candidate_same, f"{CANDIDATE_SAME} (same)")
    
    # Save results
    with open(output_dir / "same_model_result.json", "w") as f:
        json.dump({
            "accepted": result1.accepted,
            "distance": result1.distance,
            "confidence_radius": result1.confidence_radius,
            "fuzzy_similarity": result1.fuzzy_similarity,
            "n_challenges": result1.n_challenges,
            "metadata": result1.metadata
        }, f, indent=2)
    
    # Test 2: Different model (should REJECT)
    print("\n" + "="*60)
    print("TEST 2: Verifying different model")
    result2 = run_verification(reference, candidate_diff, f"{CANDIDATE_DIFF} (different)")
    
    # Save results
    with open(output_dir / "different_model_result.json", "w") as f:
        json.dump({
            "accepted": result2.accepted,
            "distance": result2.distance,
            "confidence_radius": result2.confidence_radius,
            "fuzzy_similarity": result2.fuzzy_similarity,
            "n_challenges": result2.n_challenges,
            "metadata": result2.metadata
        }, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    test1_correct = result1.accepted == True
    test2_correct = result2.accepted == False
    
    print(f"Test 1 (same model): {'✅ PASS' if test1_correct else '❌ FAIL'}")
    print(f"  Expected: ACCEPT, Got: {'ACCEPT' if result1.accepted else 'REJECT'}")
    
    print(f"Test 2 (different model): {'✅ PASS' if test2_correct else '❌ FAIL'}")
    print(f"  Expected: REJECT, Got: {'ACCEPT' if result2.accepted else 'REJECT'}")
    
    print(f"\nOverall: {test1_correct + test2_correct}/2 tests passed")
    
    if test1_correct and test2_correct:
        print("\n✅ SUCCESS: The PoT system correctly verified both cases!")
    else:
        print("\n⚠️ WARNING: Some tests did not produce expected results.")
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()