#!/usr/bin/env python3
"""
Simple Yi-34B test - just verify we can load and use the models.
NO TIMEOUTS - these are 34B parameter models!
"""

import os
import sys
import json
import time
from datetime import datetime

print(f"""
========================================
Yi-34B Simple Model Test
Time: {datetime.now()}
========================================
""")

# Model paths
base_model = "/Users/rohanvinaik/LLM_Models/yi-34b"
chat_model = "/Users/rohanvinaik/LLM_Models/yi-34b-chat"

# First, just check the models exist
print("\n1. Checking model directories...")
for model_name, model_path in [("Base", base_model), ("Chat", chat_model)]:
    if os.path.exists(model_path):
        print(f"✅ {model_name} model found: {model_path}")
        
        # Check config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                model_type = config.get("model_type", "unknown")
                hidden_size = config.get("hidden_size", 0)
                num_layers = config.get("num_hidden_layers", 0)
                vocab_size = config.get("vocab_size", 0)
                
                # Estimate parameters
                # Rough estimate: hidden_size * num_layers * 4 (for typical transformer)
                est_params = hidden_size * num_layers * 12  # More accurate for Yi
                
                print(f"   Model type: {model_type}")
                print(f"   Hidden size: {hidden_size}")
                print(f"   Layers: {num_layers}")
                print(f"   Vocab size: {vocab_size}")
                print(f"   Estimated params: {est_params/1e9:.1f}B")
        
        # Check model files
        model_files = os.listdir(model_path)
        safetensor_files = [f for f in model_files if f.endswith('.safetensors')]
        print(f"   Model files: {len(safetensor_files)} safetensor files")
        
        # Check total size
        total_size = 0
        for f in safetensor_files:
            file_path = os.path.join(model_path, f)
            total_size += os.path.getsize(file_path)
        print(f"   Total model size: {total_size/1e9:.1f}GB")
    else:
        print(f"❌ {model_name} model NOT found: {model_path}")
        sys.exit(1)

print("\n2. Testing model loading with transformers...")
print("⚠️ This will take several minutes for 34B models!")
print("⚠️ NO TIMEOUT - letting it run to completion")

try:
    print("\nImporting transformers...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    print("Setting up memory-efficient loading...")
    
    # Configure for CPU with memory efficiency
    print("\n3. Loading tokenizer...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    print(f"✅ Tokenizer loaded in {time.time()-start:.2f}s")
    
    print("\n4. Loading Yi-34B base model (this will take 2-5 minutes)...")
    print("   Using low_cpu_mem_usage=True")
    print("   Using torch.float16 for efficiency")
    
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=True
    )
    load_time = time.time() - start
    
    print(f"✅ Model loaded in {load_time:.2f} seconds ({load_time/60:.1f} minutes)")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Parameters: {model.num_parameters()/1e9:.2f}B")
    
    print("\n5. Testing inference...")
    test_prompt = "The future of artificial intelligence is"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    print(f"   Input: '{test_prompt}'")
    print("   Generating (deterministic, max_length=20)...")
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_time = time.time() - start
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Output: '{generated_text}'")
    print(f"   Generation time: {gen_time:.2f}s")
    
    print("\n✅ SUCCESS: Yi-34B model loads and runs!")
    print(f"   Total test time: {(time.time()-start):.1f}s")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": base_model,
        "success": True,
        "load_time": load_time,
        "generation_time": gen_time,
        "model_params": f"{model.num_parameters()/1e9:.2f}B",
        "generated_text": generated_text
    }
    
    result_file = f"experimental_results/yi34b_simple_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to: {result_file}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Save error results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": base_model,
        "success": False,
        "error": str(e)
    }
    
    result_file = f"experimental_results/yi34b_simple_test_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    sys.exit(1)

print("\n" + "="*60)
print("Test complete!")
print("="*60)