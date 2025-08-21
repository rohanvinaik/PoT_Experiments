#!/usr/bin/env python3
"""
Phi-2 vs GPT-2 Architecture & Training Quality Detection

This test validates detection of:
- Modern architecture improvements (Phi-2's optimizations)
- Training data quality differences (Phi-2's curated dataset)
- Size vs quality trade-offs (2.7B vs 124M parameters)

Real-world scenarios:
- API providers claiming GPT-2 performance but using different architectures
- Detecting when models use proprietary training improvements
- Verifying claimed model architectures in production systems
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from pot.core.diff_decision import EnhancedSequentialTester, TestingMode, DiffDecisionConfig

# Test scenario
TEST_SCENARIO = {
    'name': 'Architecture & Training Quality Detection',
    'description': 'Can we detect Phi-2 vs GPT-2 differences?',
    'model_a': 'gpt2',
    'model_b': 'microsoft/phi-2',
    'expected': 'DIFFERENT',
    'size_ratio': '22x larger (2.7B vs 124M)',
    'key_differences': [
        'Modern architecture optimizations',
        'High-quality curated training data',
        'Advanced positional encoding',
        'Improved attention mechanisms'
    ],
    'real_scenario': 'Detecting undisclosed architecture improvements in APIs'
}

# Prompts that highlight training quality and architecture differences
QUALITY_PROMPTS = [
    "Explain quantum mechanics:",
    "Write a Python function to sort a list:",
    "What is the capital of France?",
    "Describe the process of photosynthesis:",
    "Solve: 2x + 5 = 13",
    "Write a short story about a robot:",
    "Explain machine learning in simple terms:",
    "What are the benefits of renewable energy?",
    "Describe the water cycle:",
    "How does the internet work?",
    "What is artificial intelligence?",
    "Explain the theory of evolution:",
    "Describe how computers process information:",
    "What is climate change?",
    "Explain the concept of gravity:",
]

def test_phi2_vs_gpt2() -> Dict:
    """Test if we can detect Phi-2 vs GPT-2 differences."""
    
    print(f"{'='*70}")
    print(f"🧠 {TEST_SCENARIO['name']}")
    print(f"{'='*70}")
    print(f"Description: {TEST_SCENARIO['description']}")
    print(f"Size difference: {TEST_SCENARIO['size_ratio']}")
    print(f"Expected result: {TEST_SCENARIO['expected']}")
    print(f"Real-world scenario: {TEST_SCENARIO['real_scenario']}")
    
    print(f"\n🔍 Key Differences Being Tested:")
    for diff in TEST_SCENARIO['key_differences']:
        print(f"  • {diff}")
    
    results = {
        'scenario': TEST_SCENARIO,
        'success': False
    }
    
    try:
        # Load models
        print(f"\n📥 Loading models...")
        print(f"  Model A: {TEST_SCENARIO['model_a']}")
        print(f"  Model B: {TEST_SCENARIO['model_b']}")
        
        # Check local models first
        model_b_local_path = str(Path.home() / "LLM_Models" / "phi-2")
        
        # Use appropriate dtypes
        model_a = AutoModelForCausalLM.from_pretrained(
            TEST_SCENARIO['model_a'],
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Try local path first, fallback to HuggingFace
        try:
            model_b = AutoModelForCausalLM.from_pretrained(
                model_b_local_path,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for larger Phi-2
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            phi2_path = model_b_local_path
        except:
            model_b = AutoModelForCausalLM.from_pretrained(
                TEST_SCENARIO['model_b'],
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            phi2_path = TEST_SCENARIO['model_b']
        
        tokenizer_a = AutoTokenizer.from_pretrained(TEST_SCENARIO['model_a'])
        tokenizer_b = AutoTokenizer.from_pretrained(
            phi2_path,
            trust_remote_code=True
        )
        
        # Set padding tokens
        tokenizer_a.pad_token = tokenizer_a.eos_token if tokenizer_a.eos_token else tokenizer_a.unk_token
        tokenizer_b.pad_token = tokenizer_b.eos_token if tokenizer_b.eos_token else tokenizer_b.unk_token
        
        # Get model sizes
        params_a = sum(p.numel() for p in model_a.parameters())
        params_b = sum(p.numel() for p in model_b.parameters())
        
        print(f"✓ Models loaded successfully")
        print(f"  {TEST_SCENARIO['model_a']}: {params_a/1e6:.1f}M parameters")
        print(f"  Phi-2: {params_b/1e6:.1f}M parameters")
        print(f"  Actual size ratio: {params_b/params_a:.1f}x")
        
        # Check vocabulary compatibility
        vocab_a = len(tokenizer_a.get_vocab())
        vocab_b = len(tokenizer_b.get_vocab())
        print(f"  Vocabulary sizes: {vocab_a} vs {vocab_b}")
        
        # Create enhanced tester for architecture detection
        config = DiffDecisionConfig(
            mode=TestingMode.QUICK_GATE,
            n_min=10,
            n_max=50,
            gamma=0.02,  # Tight threshold for architecture differences
            delta_star=0.1
        )
        tester = EnhancedSequentialTester(config)
        
        print(f"\n🧪 Testing architecture and training differences...")
        
        differences = []
        
        for i, prompt in enumerate(QUALITY_PROMPTS * 2):  # Double for more samples
            # Get inputs
            inputs_a = tokenizer_a(prompt, return_tensors='pt', padding=True, truncation=True, max_length=64)
            inputs_b = tokenizer_b(prompt, return_tensors='pt', padding=True, truncation=True, max_length=64)
            
            with torch.no_grad():
                # Get logits
                outputs_a = model_a(**inputs_a)
                outputs_b = model_b(**inputs_b)
                
                logits_a = outputs_a.logits[0, -1, :].float()
                logits_b = outputs_b.logits[0, -1, :].float()
                
                # Align vocabulary sizes
                min_vocab = min(logits_a.shape[0], logits_b.shape[0])
                logits_a = logits_a[:min_vocab]
                logits_b = logits_b[:min_vocab]
                
                # Convert to probabilities
                probs_a = torch.softmax(logits_a, dim=-1)
                probs_b = torch.softmax(logits_b, dim=-1)
                
                # Calculate Total Variation Distance
                tv_distance = 0.5 * torch.abs(probs_a - probs_b).sum().item()
                
                differences.append(tv_distance)
                tester.update(tv_distance)
                
                # Check stopping condition
                should_stop, decision_info = tester.should_stop()
                
                if (i + 1) % 5 == 0:
                    current = decision_info.get('decision', 'UNDECIDED') if decision_info else 'UNDECIDED'
                    print(f"  After {i+1} tests: {current} (avg TV distance: {np.mean(differences):.4f})")
                
                if should_stop and len(differences) >= config.n_min:
                    break
        
        # Final decision
        final_decision = decision_info.get('decision', 'UNDECIDED') if decision_info else 'UNDECIDED'
        
        print(f"\n📊 RESULTS:")
        print(f"  Decision: {final_decision}")
        print(f"  Expected: {TEST_SCENARIO['expected']}")
        print(f"  Samples used: {len(differences)}")
        print(f"  Avg TV distance: {np.mean(differences):.4f}")
        print(f"  Std deviation: {np.std(differences):.4f}")
        
        # Check vocabulary overlap
        common_tokens = set(tokenizer_a.get_vocab().keys()) & set(tokenizer_b.get_vocab().keys())
        overlap_ratio = len(common_tokens) / max(vocab_a, vocab_b)
        print(f"  Vocabulary overlap: {overlap_ratio:.1%}")
        
        # Interpret results
        success = (final_decision == TEST_SCENARIO['expected'])
        results['success'] = success
        results['decision'] = final_decision
        results['samples'] = len(differences)
        results['avg_distance'] = float(np.mean(differences))
        results['std_distance'] = float(np.std(differences))
        results['params_a'] = params_a
        results['params_b'] = params_b
        results['vocab_overlap'] = overlap_ratio
        
        print(f"\n🎯 DETECTION RESULT: {'✅ SUCCESS' if success else '❌ FAILURE'}")
        
        if success:
            print(f"  Successfully detected architecture/training differences!")
            print(f"  The framework can distinguish between:")
            print(f"    • Classical GPT-2 architecture")
            print(f"    • Modern Phi-2 optimizations")
            print(f"    • Training data quality improvements")
            print(f"  This prevents substitution fraud in production APIs")
        
        # Clean up
        del model_a, model_b
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        results['error'] = str(e)
    
    return results

def main():
    """Run Phi-2 vs GPT-2 architecture detection test."""
    
    print("="*70)
    print("🏗️ ARCHITECTURE & TRAINING QUALITY DETECTION")
    print("="*70)
    print("\nTesting the framework's ability to detect:")
    print("• Modern architecture improvements (Phi-2 vs GPT-2)")
    print("• Training data quality differences")
    print("• Advanced optimization techniques")
    print("\nWhy this matters:")
    print("• API transparency (detecting architecture substitution)")
    print("• Intellectual property verification")
    print("• Performance guarantee validation")
    print("• Model authenticity in production systems")
    
    result = test_phi2_vs_gpt2()
    
    # Save results
    output_dir = Path('experimental_results')
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f'phi2_vs_gpt2_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    # Final summary
    print("\n" + "="*70)
    print("📊 ARCHITECTURE DETECTION SUMMARY")
    print("="*70)
    
    if 'decision' in result:
        scenario = result['scenario']
        print(f"\n**{scenario['name']}**")
        print(f"  Models: {scenario['model_a']} vs Phi-2")
        print(f"  Size difference: {scenario['size_ratio']}")
        print(f"  Detection: {'✅' if result['success'] else '❌'} "
              f"{result['decision']} (expected: {scenario['expected']})")
        print(f"  TV distance: {result.get('avg_distance', 0):.4f}")
        print(f"  Samples needed: {result.get('samples', 0)}")
        print(f"  Vocabulary overlap: {result.get('vocab_overlap', 0):.1%}")
    
    print("\n🎯 KEY INSIGHT:")
    print("Successfully detecting Phi-2 vs GPT-2 proves the framework can")
    print("identify modern architecture improvements and training quality")
    print("differences, ensuring API authenticity and preventing model")
    print("substitution fraud in production systems.")

if __name__ == "__main__":
    main()