#!/usr/bin/env python3
"""Debug CE calculation - target tokens only"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def compute_target_only_ce(model, tokenizer, prompt, target, device):
    """Compute CE only on target tokens (teacher-forced)"""
    
    # Tokenize separately
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_text = prompt + target
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    
    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]
    target_len = full_len - prompt_len
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
    
    # Compute CE only on target positions
    total_ce = 0.0
    for i in range(prompt_len, full_len):
        # Logits at position i-1 predict token at position i
        pred_logits = logits[i-1]
        target_token = full_ids[0, i]
        
        # Cross-entropy for this position
        ce = F.cross_entropy(pred_logits.unsqueeze(0), target_token.unsqueeze(0))
        total_ce += ce.item()
    
    return total_ce  # Total CE for target tokens

def test_target_only_ce():
    """Test target-only CE calculation"""
    
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
    print(f"Target tokens: {len(generated_tokens)}")
    
    # Compute target-only CE
    gpt2_ce = compute_target_only_ce(gpt2_model, gpt2_tok, prompt, target_text, device)
    distil_ce = compute_target_only_ce(distil_model, distil_tok, prompt, target_text, device)
    
    print(f"\n=== Target-Only Cross-Entropy ===")
    print(f"GPT-2 CE (total): {gpt2_ce:.6f}")
    print(f"DistilGPT-2 CE (total): {distil_ce:.6f}")
    print(f"Difference: {gpt2_ce - distil_ce:.6f}")
    
    print(f"\nGPT-2 CE (per token): {gpt2_ce/len(generated_tokens):.6f}")
    print(f"DistilGPT-2 CE (per token): {distil_ce/len(generated_tokens):.6f}")

if __name__ == "__main__":
    test_target_only_ce()