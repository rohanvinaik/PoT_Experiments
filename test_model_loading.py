#!/usr/bin/env python3
"""Test script to debug model loading issues on macOS"""

import sys
import os

# Try different approaches
print("Testing model loading approaches...")

# Test 1: Basic transformers import
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✓ Transformers imported successfully")
except Exception as e:
    print(f"✗ Failed to import transformers: {e}")
    sys.exit(1)

# Test 2: Load GPT2
try:
    print("\nAttempting to load GPT2...")
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ GPT2 loaded from HuggingFace hub")
    
    # Test generation
    inputs = tokenizer("Hello world", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✓ Generation successful: {result[:50]}...")
    
except Exception as e:
    print(f"✗ Failed to load/generate with GPT2: {e}")
    
# Test 3: Load from local path
try:
    local_path = "/Users/rohanvinaik/LLM_Models/gpt2"
    if os.path.exists(local_path):
        print(f"\nAttempting to load from local path: {local_path}")
        model = AutoModelForCausalLM.from_pretrained(local_path)
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Model loaded from local path")
        
        # Test generation
        inputs = tokenizer("Hello world", return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Generation successful: {result[:50]}...")
    else:
        print(f"✗ Local path does not exist: {local_path}")
        
except Exception as e:
    print(f"✗ Failed with local model: {e}")
    import traceback
    traceback.print_exc()

print("\nDiagnostic complete.")