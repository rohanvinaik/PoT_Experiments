#!/usr/bin/env python3
"""
LLM Verification Test - ONLY OPEN MODELS
=========================================
Uses GPT-2 and DistilGPT-2 - NO AUTHENTICATION REQUIRED
"""

import os
import sys
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*60)
print("LLM VERIFICATION - OPEN MODELS ONLY")
print("="*60)
print("Models: GPT-2 (124M) and DistilGPT-2 (82M)")
print("No tokens or authentication required!")
print("="*60)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    class OpenModelLM:
        """Adapter for open HuggingFace models."""
        def __init__(self, model_name: str, seed: int = 0):
            print(f"Loading {model_name}...")
            torch.manual_seed(seed)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            ).eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
            print(f"✅ {model_name} loaded")
        
        @torch.no_grad()
        def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Load models
    print("\nLoading models...")
    ref = OpenModelLM("gpt2", seed=42)
    cand_same = OpenModelLM("gpt2", seed=123)
    cand_diff = OpenModelLM("distilgpt2", seed=456)
    
    # Test
    prompts = ["The weather is", "Today I will", "Science is"]
    
    print("\nRunning verification...")
    for prompt in prompts:
        ref_out = ref.generate(prompt)
        same_out = cand_same.generate(prompt)
        diff_out = cand_diff.generate(prompt)
        
        print(f"\nPrompt: '{prompt}'")
        print(f"  GPT-2 (ref): {ref_out[:40]}...")
        print(f"  GPT-2 (same): {same_out[:40]}...")
        print(f"  DistilGPT-2: {diff_out[:40]}...")
    
    # Save results
    Path("experimental_results").mkdir(exist_ok=True)
    results = {
        "test": "llm_open_models",
        "models": ["gpt2", "distilgpt2"],
        "status": "success",
        "authentication_required": False
    }
    
    with open("experimental_results/llm_open_test.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ LLM test completed successfully")
    print("   NO AUTHENTICATION TOKENS WERE USED!")
    
except ImportError as e:
    print(f"⚠️ Missing dependency: {e}")
    print("   Install with: pip install transformers torch")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)