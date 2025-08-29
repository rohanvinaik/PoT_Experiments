#!/usr/bin/env python3
"""
Create a quantized version of GPT-2 using PyTorch dynamic quantization.
This works on CPU/MPS without requiring CUDA.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("="*60)
    print("Creating Quantized GPT-2 Model (PyTorch Dynamic Quantization)")
    print("="*60)
    
    model_path = os.path.expanduser("~/LLM_Models/gpt2")
    output_path = os.path.expanduser("~/LLM_Models/gpt2-pytorch-quantized")
    
    print(f"\nSource model: {model_path}")
    print(f"Output path: {output_path}")
    
    print("\nLoading original model...")
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\nApplying dynamic quantization...")
    # Apply dynamic quantization to the model
    # This quantizes Linear layers to int8
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8  # Use int8 quantization
    )
    
    print(f"\nOriginal model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\nSaving quantized model to {output_path}...")
    
    # Save the quantized model state dict
    torch.save(quantized_model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
    
    # Save tokenizer and config
    tokenizer.save_pretrained(output_path)
    model.config.save_pretrained(output_path)
    
    # Add quantization info to config
    config_path = os.path.join(output_path, "config.json")
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    config["quantization_method"] = "pytorch_dynamic_int8"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Quantized model saved successfully!")
    print(f"   Location: {output_path}")
    
    # Test the model to ensure it works
    print("\nTesting quantized model...")
    test_input = "The quick brown fox"
    inputs = tokenizer(test_input, return_tensors="pt")
    
    with torch.no_grad():
        # For testing, we need to generate with the quantized model
        input_ids = inputs["input_ids"]
        for _ in range(10):  # Generate 10 tokens
            outputs = quantized_model(input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"Test generation: {generated_text}")
    
    print("\n✅ Quantized model is working correctly!")
    
    # Check file size difference
    original_size = os.path.getsize(os.path.join(model_path, "pytorch_model.bin")) / (1024**2)
    quantized_size = os.path.getsize(os.path.join(output_path, "pytorch_model.bin")) / (1024**2)
    print(f"\nModel size comparison:")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  Quantized: {quantized_size:.2f} MB")
    print(f"  Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    
    return output_path

if __name__ == "__main__":
    output_path = main()
    print(f"\n📝 Ready to test with PoT framework:")
    print(f"   python scripts/run_e2e_validation.py \\")
    print(f"       --ref-model ~/LLM_Models/gpt2 \\")
    print(f"       --cand-model {output_path} \\")
    print(f"       --mode audit --enable-zk")