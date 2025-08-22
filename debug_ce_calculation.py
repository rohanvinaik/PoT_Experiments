#!/usr/bin/env python3
"""Debug CE calculation differences between implementations"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_ce_calculation():
    """Test CE calculation to understand magnitude differences"""
    
    # Load models
    print("Loading models...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    
    distil_tok = AutoTokenizer.from_pretrained("distilgpt2")
    distil_tok.pad_token = distil_tok.eos_token
    distil_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    
    # Test prompt
    prompt = "The capital of France is"
    K = 32  # Generate 32 tokens
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Generating {K} tokens...")
    
    # Generate target using GPT-2
    inputs = gpt2_tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gpt2_model.generate(
            **inputs,
            max_new_tokens=K,
            do_sample=False,
            pad_token_id=gpt2_tok.pad_token_id
        )
    
    prompt_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][prompt_length:]
    target_text = gpt2_tok.decode(generated_tokens, skip_special_tokens=True)
    
    print(f"Generated target: '{target_text[:50]}...'")
    print(f"Target length: {len(generated_tokens)} tokens")
    
    # Method 1: Full text CE (like our current implementation)
    full_text = prompt + target_text
    
    # GPT-2 CE
    gpt2_full_inputs = gpt2_tok(full_text, return_tensors="pt").to(device)
    with torch.no_grad():
        gpt2_outputs = gpt2_model(**gpt2_full_inputs, labels=gpt2_full_inputs.input_ids)
        gpt2_loss = gpt2_outputs.loss.item()
    
    # DistilGPT-2 CE
    distil_full_inputs = distil_tok(full_text, return_tensors="pt").to(device)
    with torch.no_grad():
        distil_outputs = distil_model(**distil_full_inputs, labels=distil_full_inputs.input_ids)
        distil_loss = distil_outputs.loss.item()
    
    # Scale by target length (like runtime_blackbox_validation.py)
    target_length = len(generated_tokens)
    gpt2_ce_scaled = gpt2_loss * target_length
    distil_ce_scaled = distil_loss * target_length
    
    print(f"\n=== Cross-Entropy Results ===")
    print(f"GPT-2 loss (avg): {gpt2_loss:.6f}")
    print(f"DistilGPT-2 loss (avg): {distil_loss:.6f}")
    print(f"Difference (avg): {gpt2_loss - distil_loss:.6f}")
    print()
    print(f"GPT-2 CE (scaled): {gpt2_ce_scaled:.6f}")
    print(f"DistilGPT-2 CE (scaled): {distil_ce_scaled:.6f}")
    print(f"Difference (scaled): {gpt2_ce_scaled - distil_ce_scaled:.6f}")
    
    # Method 2: Target-only CE (what runtime_blackbox might be doing differently)
    print("\n=== Alternative: Compute CE only on target tokens ===")
    
    # Tokenize prompt and full text separately
    prompt_ids = gpt2_tok(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = gpt2_tok(full_text, return_tensors="pt").input_ids.to(device)
    
    # Create attention mask that focuses only on target tokens
    # This is more complex and may not be what's happening
    
    print("\nNote: The scaled difference should be around -11 to -15 for GPT-2 vs DistilGPT-2")
    print("based on the runtime_blackbox_validation.py results.")

if __name__ == "__main__":
    test_ce_calculation()