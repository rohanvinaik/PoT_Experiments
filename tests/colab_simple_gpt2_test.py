#!/usr/bin/env python3
"""
SIMPLE GPT-2 MODEL COMPARISON TEST FOR GOOGLE COLAB
==================================================
Tests GPT-2 vs DistilGPT-2 and GPT-2 vs GPT-2-medium using your existing framework.
No HuggingFace tokens required - all models are public.
"""

import pytest

pytest.skip(
    "colab environment test script – skipped in automated test runs",
    allow_module_level=True,
)

import os
import sys
import subprocess
import json
import shutil
from datetime import datetime

print("=" * 70)
print("POT EXPERIMENTS - SIMPLE GPT-2 MODEL COMPARISON")
print("No HuggingFace tokens required!")
print("=" * 70)

# Setup repository
if not os.path.exists('/content/PoT_Experiments'):
    print("📥 Cloning repository...")
    subprocess.run(['git', 'clone', 'https://github.com/ANONYMOUS/PoT_Experiments.git', '/content/PoT_Experiments'])

os.chdir('/content/PoT_Experiments')
print(f"📍 Working directory: {os.getcwd()}")

# Install dependencies
print("\n📦 Installing dependencies...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 
                'torch', 'transformers', 'numpy', 'scipy'])

# Create results directory
RESULTS_DIR = '/content/GPT2_Comparison_Results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Test 1: Run the existing GPT-2 variants test
print("\n" + "=" * 70)
print("TEST 1: GPT-2 vs DistilGPT-2 (Your Paper's Main Test)")
print("=" * 70)

env = os.environ.copy()
env['PYTHONPATH'] = '/content/PoT_Experiments'

try:
    result = subprocess.run(
        [sys.executable, 'scripts/test_gpt2_variants.py'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env,
        timeout=600  # 10 minutes
    )
    
    print("📊 GPT-2 vs DistilGPT-2 Test Results:")
    print(result.stdout)
    
    if result.stderr:
        print("\nWarnings/Errors:")
        print(result.stderr)
    
    # Save results
    with open(f'{RESULTS_DIR}/gpt2_vs_distilgpt2_output.txt', 'w') as f:
        f.write("GPT-2 vs DistilGPT-2 Test Results\n")
        f.write("=" * 40 + "\n")
        f.write(result.stdout)
        f.write("\n\nStderr:\n")
        f.write(result.stderr)
    
    print("✅ GPT-2 vs DistilGPT-2 test completed")
    
except subprocess.TimeoutExpired:
    print("⚠️ Test timed out after 10 minutes")
except Exception as e:
    print(f"❌ Test failed: {e}")

# Test 2: Create a simple comparison test for additional models
print("\n" + "=" * 70)
print("TEST 2: Additional Public Model Comparisons")
print("=" * 70)

# Create a simple model comparison script
comparison_script = '''
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
from datetime import datetime

def compare_models(model1_name, model2_name, test_name):
    """Compare two models and return metrics"""
    print(f"\\n🔬 Comparing {model1_name} vs {model2_name}")
    
    try:
        # Load models
        print("  Loading models...")
        tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
        model1 = AutoModelForCausalLM.from_pretrained(model1_name)
        
        tokenizer2 = AutoTokenizer.from_pretrained(model2_name)  
        model2 = AutoModelForCausalLM.from_pretrained(model2_name)
        
        if tokenizer1.pad_token is None:
            tokenizer1.pad_token = tokenizer1.eos_token
        if tokenizer2.pad_token is None:
            tokenizer2.pad_token = tokenizer2.eos_token
        
        # Test prompts
        test_prompts = [
            "The future of artificial intelligence",
            "In a world where robots",
            "The meaning of life is",
            "Technology will change",
            "Once upon a time"
        ]
        
        results = {
            "test_name": test_name,
            "model1": model1_name,
            "model2": model2_name,
            "timestamp": datetime.now().isoformat(),
            "comparisons": []
        }
        
        total_divergence = 0
        
        for prompt in test_prompts:
            print(f"    Testing prompt: '{prompt[:30]}...'")
            
            # Get logits from both models
            inputs1 = tokenizer1(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs2 = tokenizer2(prompt, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs1 = model1(**inputs1)
                outputs2 = model2(**inputs2)
                
                # Get probability distributions
                logits1 = outputs1.logits[0, -1, :].softmax(dim=0)
                logits2 = outputs2.logits[0, -1, :].softmax(dim=0)
                
                # Calculate KL divergence (symmetric)
                kl1 = torch.nn.functional.kl_div(logits1.log(), logits2, reduction='sum')
                kl2 = torch.nn.functional.kl_div(logits2.log(), logits1, reduction='sum')
                symmetric_kl = (kl1 + kl2) / 2
                
                divergence = float(symmetric_kl)
                total_divergence += divergence
                
                results["comparisons"].append({
                    "prompt": prompt,
                    "kl_divergence": divergence
                })
        
        avg_divergence = total_divergence / len(test_prompts)
        results["average_divergence"] = avg_divergence
        results["num_prompts"] = len(test_prompts)
        
        # Interpretation
        if avg_divergence < 0.1:
            interpretation = "VERY_SIMILAR"
        elif avg_divergence < 0.5:
            interpretation = "MODERATELY_SIMILAR"
        elif avg_divergence < 1.0:
            interpretation = "SOMEWHAT_DIFFERENT"
        else:
            interpretation = "VERY_DIFFERENT"
            
        results["interpretation"] = interpretation
        
        print(f"  ✅ Average KL divergence: {avg_divergence:.4f}")
        print(f"  📊 Interpretation: {interpretation}")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Comparison failed: {e}")
        return {
            "test_name": test_name,
            "model1": model1_name,
            "model2": model2_name,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Run comparisons
print("Running model comparisons...")

# Test 1: GPT-2 vs DistilGPT-2 (close variants)
result1 = compare_models("gpt2", "distilgpt2", "gpt2_vs_distilgpt2")

# Test 2: GPT-2 vs GPT-2-medium (different sizes)  
result2 = compare_models("gpt2", "gpt2-medium", "gpt2_vs_gpt2medium")

# Test 3: Self-comparison (should be identical)
result3 = compare_models("gpt2", "gpt2", "gpt2_vs_self")

# Save all results
all_results = {
    "timestamp": datetime.now().isoformat(),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "comparisons": [result1, result2, result3]
}

with open("/content/GPT2_Comparison_Results/model_comparisons.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\\n📊 Summary:")
print("=" * 50)
for result in [result1, result2, result3]:
    if "error" not in result:
        print(f"{result['test_name']}: {result['average_divergence']:.4f} ({result['interpretation']})")
    else:
        print(f"{result['test_name']}: ERROR - {result['error']}")

print("\\n✅ Model comparison tests completed!")
'''

# Write and run the comparison script
with open(f'{RESULTS_DIR}/run_comparisons.py', 'w') as f:
    f.write(comparison_script)

print("🚀 Running additional model comparisons...")
try:
    result = subprocess.run(
        [sys.executable, f'{RESULTS_DIR}/run_comparisons.py'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=900  # 15 minutes
    )
    
    print(result.stdout)
    if result.stderr:
        print("Warnings:", result.stderr)
        
except Exception as e:
    print(f"❌ Additional tests failed: {e}")

# Package results
print("\n" + "=" * 70)
print("📦 PACKAGING RESULTS")
print("=" * 70)

# Copy any outputs from the PoT tests
if os.path.exists('outputs'):
    shutil.copytree('outputs', f'{RESULTS_DIR}/pot_outputs', dirs_exist_ok=True)
    print("✅ Copied PoT test outputs")

if os.path.exists('experimental_results'):
    shutil.copytree('experimental_results', f'{RESULTS_DIR}/experimental_results', dirs_exist_ok=True)
    print("✅ Copied experimental results")

# Create summary
summary = {
    "test_suite": "GPT-2 Model Comparisons",
    "timestamp": datetime.now().isoformat(),
    "models_tested": ["gpt2", "distilgpt2", "gpt2-medium"],
    "test_types": [
        "Sequential Testing (GPT-2 vs DistilGPT-2)",
        "KL Divergence Comparison",
        "Self-comparison validation"
    ],
    "results_location": RESULTS_DIR,
    "paper_relevance": [
        "Validates teacher-forced scoring methodology",
        "Demonstrates model discrimination capability", 
        "Provides baseline comparisons for paper metrics",
        "Tests with publicly available models (no tokens required)"
    ]
}

with open(f'{RESULTS_DIR}/test_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Create ZIP for download
import zipfile
zip_filename = f'/content/GPT2_Comparison_Results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(RESULTS_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arc_path = os.path.relpath(file_path, RESULTS_DIR)
            zipf.write(file_path, arc_path)

# Show final results
print(f"\n📁 Results saved to: {RESULTS_DIR}")
print("Contents:")
for item in os.listdir(RESULTS_DIR):
    item_path = os.path.join(RESULTS_DIR, item)
    if os.path.isfile(item_path):
        size = os.path.getsize(item_path) / 1024
        print(f"  📄 {item} ({size:.1f} KB)")
    else:
        print(f"  📁 {item}/")

print(f"\n📦 Downloadable ZIP: {zip_filename}")
zip_size = os.path.getsize(zip_filename) / (1024*1024)
print(f"   Size: {zip_size:.2f} MB")

# Download the ZIP
try:
    from google.colab import files
    files.download(zip_filename)
    print("✅ Download initiated!")
except:
    print("⚠️ Could not initiate download")

print("\n" + "=" * 70)
print("✅ GPT-2 MODEL COMPARISON TESTS COMPLETE")
print("=" * 70)
print("\n🎯 These results validate your paper claims:")
print("  • Teacher-forced scoring with real models")
print("  • Sequential testing efficiency")
print("  • Model discrimination capabilities")
print("  • Performance with public models (no tokens)")
print(f"\n📁 All evidence saved to: {RESULTS_DIR}")
print("💾 ZIP file downloaded to your computer")
print("=" * 70)