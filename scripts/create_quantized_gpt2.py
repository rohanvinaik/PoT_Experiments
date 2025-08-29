#!/usr/bin/env python3
"""
Create a quantized version of GPT-2 using bitsandbytes int8 quantization.
This creates a real quantized model for testing PoT framework.
"""

import os
import sys
import torch
from pathlib import Path
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    print("="*60)
    print("Creating Quantized GPT-2 Model")
    print("="*60)
    
    # First, install bitsandbytes if not available
    try:
        import bitsandbytes as bnb
        print("✅ bitsandbytes is installed")
    except ImportError:
        print("Installing bitsandbytes...")
        subprocess.run([sys.executable, "-m", "pip", "install", "bitsandbytes"], check=True)
        import bitsandbytes as bnb
    
    model_path = os.path.expanduser("~/LLM_Models/gpt2")
    output_path = os.path.expanduser("~/LLM_Models/gpt2-int8-quantized")
    
    print(f"\nSource model: {model_path}")
    print(f"Output path: {output_path}")
    
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16
    )
    
    print("\nLoading and quantizing model...")
    # Load model with int8 quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"\nSaving quantized model to {output_path}...")
    # Save the quantized model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Also save the config to indicate it's quantized
    config = model.config
    config.quantization_config = {
        "load_in_8bit": True,
        "bnb_8bit_compute_dtype": "float16"
    }
    config.save_pretrained(output_path)
    
    print("\n✅ Quantized model saved successfully!")
    print(f"   Location: {output_path}")
    
    # Test the model to ensure it works
    print("\nTesting quantized model...")
    test_input = "The quick brown fox"
    inputs = tokenizer(test_input, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=20)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test generation: {generated_text}")
    
    print("\n✅ Quantized model is working correctly!")
    return output_path

if __name__ == "__main__":
    output_path = main()
    print(f"\n📝 Ready to test with PoT framework:")
    print(f"   python scripts/run_e2e_validation.py \\")
    print(f"       --ref-model ~/LLM_Models/gpt2 \\")
    print(f"       --cand-model {output_path} \\")
    print(f"       --mode audit --enable-zk")