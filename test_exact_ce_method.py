#!/usr/bin/env python3
"""Test exact CE calculation method from runtime_blackbox_validation.py"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_cross_entropy_exact(model, tokenizer, prompt, target, device):
    """Exact method from runtime_blackbox_validation.py"""
    full_text = prompt + target
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        
    # Get loss only for the target tokens
    loss = outputs.loss.item()
    target_length = inputs.input_ids.shape[1] - prompt_inputs.input_ids.shape[1]
    
    # Normalize by target length for fair comparison
    return loss * target_length if target_length > 0 else loss

def test_exact_method():
    """Test exact CE method from runtime_blackbox"""
    
    # Load models
    print("Loading models...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    
    distil_tok = AutoTokenizer.from_pretrained("distilgpt2")
    distil_tok.pad_token = distil_tok.eos_token
    distil_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    
    # Test multiple prompts
    prompts = [
        "The capital of France is",
        "A unusual dog will transforms the equation.",
        "The sky is blue because",
        "In programming, a function is"
    ]
    
    total_diff = 0.0
    diffs = []
    
    for prompt in prompts:
        # Generate target using GPT-2
        inputs = gpt2_tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = gpt2_model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=gpt2_tok.pad_token_id
            )
        
        prompt_length = inputs.input_ids.shape[1]
        target_text = gpt2_tok.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        
        # Compute CE using exact method
        gpt2_ce = get_cross_entropy_exact(gpt2_model, gpt2_tok, prompt, target_text, device)
        distil_ce = get_cross_entropy_exact(distil_model, distil_tok, prompt, target_text, device)
        
        diff = gpt2_ce - distil_ce
        diffs.append(diff)
        total_diff += diff
        
        print(f"\nPrompt: '{prompt[:30]}...'")
        print(f"Target: '{target_text[:30]}...'")
        print(f"GPT-2 CE: {gpt2_ce:.6f}")
        print(f"DistilGPT-2 CE: {distil_ce:.6f}")
        print(f"Difference: {diff:.6f}")
    
    print(f"\n=== Summary ===")
    print(f"Mean difference: {total_diff/len(prompts):.6f}")
    print(f"All differences: {diffs}")
    print(f"\nExpected: Mean around -11 for distillation detection")

if __name__ == "__main__":
    test_exact_method()