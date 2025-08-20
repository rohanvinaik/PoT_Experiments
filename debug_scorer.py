#!/usr/bin/env python3
"""Debug the scoring issue with identical models"""

from pot.scoring.optimized_scorer import FastScorer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

device = torch.device('cpu')
model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)

# Test prompt
prompt = "The capital of France is"
full_text = prompt + " The answer is"

# Tokenize
inputs = tokenizer(
    full_text,
    return_tensors="pt",
    truncation=True,
    max_length=128
)

prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
print(f"Prompt: '{prompt}'")
print(f"Full text: '{full_text}'")
print(f"Prompt length: {prompt_len} tokens")
print(f"Total length: {inputs['input_ids'].shape[1]} tokens")

# Get logits
with torch.no_grad():
    out = model(**inputs)

# Check scoring
k = 32
top_k = 100
scores = []
penalties = 0
seq_len = inputs["input_ids"].shape[1]

print(f"\nScoring positions {prompt_len} to {min(prompt_len + k, seq_len - 1)}:")

for i in range(prompt_len, min(prompt_len + k, seq_len - 1)):
    target = inputs["input_ids"][0, i + 1]
    
    # Top-k tokens
    topk_vals, topk_idx = torch.topk(out.logits[0, i], top_k)
    
    if target in topk_idx:
        idx = (topk_idx == target).nonzero(as_tuple=True)[0][0]
        
        # Log probabilities (identical for same model)
        lp = F.log_softmax(topk_vals, dim=-1)[idx]
        
        # Difference should be 0
        diff = abs((lp - lp).item())
        scores.append(diff)
        
        token_str = tokenizer.decode([target.item()])
        print(f"  Position {i}: token='{token_str}', diff={diff:.6f} (in top-{top_k})")
    else:
        scores.append(0.5)
        penalties += 1
        token_str = tokenizer.decode([target.item()])
        print(f"  Position {i}: token='{token_str}', PENALTY=0.5 (NOT in top-{top_k})")

mean_score = np.mean(scores) if scores else 0.0
print(f"\nScores: {scores[:10]}...")
print(f"Mean score: {mean_score:.6f}")
print(f"Penalties applied: {penalties}/{len(scores)}")

if mean_score == 0.25:
    print("\n⚠️ Score is exactly 0.25 - suggests 50% penalties")
    print("This happens when half the tokens are not in top-k")