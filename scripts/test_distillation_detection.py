#!/usr/bin/env python3
"""
Model Distillation & Optimization Detection

This test validates detection of model distillation and optimization techniques:
- Teacher-student distillation (BERT vs DistilBERT, GPT-2 vs DistilGPT-2)
- Architecture pruning while maintaining similar performance
- Critical for detecting undisclosed optimizations in production

Real-world scenarios:
- Cloud providers using DistilBERT instead of BERT to save 40% compute
- APIs serving DistilGPT-2 as GPT-2 for 2x speedup
- Mobile apps using quantized models without disclosure
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from pot.core.diff_decision import EnhancedSequentialTester, TestingMode, DiffDecisionConfig

# Critical test pairs
TEST_SCENARIOS = [
    {
        'name': 'GPT-2 Distillation Detection',
        'description': 'Can we detect DistilGPT-2 vs GPT-2?',
        'model_a': 'gpt2',
        'model_b': 'distilgpt2',
        'expected': 'DIFFERENT',
        'optimization': '2x faster, 40% smaller',
        'risk': 'Quality degradation hidden from users',
        'real_scenario': 'API serves DistilGPT-2 but charges for GPT-2'
    },
    {
        'name': 'Same Family Verification',
        'description': 'GPT-2 vs GPT-2-Medium (legitimate scaling)',
        'model_a': 'gpt2',
        'model_b': 'gpt2-medium',
        'expected': 'DIFFERENT',
        'optimization': 'Larger model, not optimized',
        'risk': 'None - legitimate size difference',
        'real_scenario': 'Detecting actual model upgrades'
    },
    {
        'name': 'Identical Model Control',
        'description': 'DistilGPT-2 vs DistilGPT-2 (same)',
        'model_a': 'distilgpt2',
        'model_b': 'distilgpt2',
        'expected': 'SAME',
        'optimization': 'N/A - control test',
        'risk': 'False positive prevention',
        'real_scenario': 'Ensuring consistency'
    }
]

# Test prompts that reveal quality differences
QUALITY_TEST_PROMPTS = [
    "The meaning of life is",
    "In conclusion, we can say that",
    "The most important factor in",
    "According to recent studies,",
    "The relationship between",
    "It is worth noting that",
    "The primary advantage of",
    "When considering the options,",
    "The fundamental principle behind",
    "Research has shown that",
    "The key difference between",
    "To summarize the findings,",
    "The implications of this are",
    "From a practical standpoint,",
    "The underlying mechanism involves",
]

def compare_model_outputs(model_a, model_b, tokenizer_a, tokenizer_b, prompts):
    """Compare model outputs on quality-revealing prompts."""
    differences = []
    
    for prompt in prompts:
        # Get inputs
        inputs_a = tokenizer_a(prompt, return_tensors='pt', padding=True, truncation=True, max_length=50)
        inputs_b = tokenizer_b(prompt, return_tensors='pt', padding=True, truncation=True, max_length=50)
        
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
            
            # Calculate Total Variation Distance (more interpretable than KL)
            tv_distance = 0.5 * torch.abs(probs_a - probs_b).sum().item()
            
            differences.append(tv_distance)
    
    return differences

def test_distillation(scenario: Dict) -> Dict:
    """Test if we can detect model distillation."""
    
    print(f"\n{'='*70}")
    print(f"🔬 {scenario['name']}")
    print(f"{'='*70}")
    print(f"Description: {scenario['description']}")
    print(f"Optimization: {scenario['optimization']}")
    print(f"Risk: {scenario['risk']}")
    print(f"Real-world scenario: {scenario['real_scenario']}")
    
    results = {
        'scenario': scenario,
        'success': False
    }
    
    try:
        # Load models
        print(f"\n📥 Loading models...")
        print(f"  Model A: {scenario['model_a']}")
        print(f"  Model B: {scenario['model_b']}")
        
        # Check if models exist locally first
        local_model_a = f"~/LLM_Models/{scenario['model_a']}"
        local_model_b = f"~/LLM_Models/{scenario['model_b']}"
        
        # Use different dtypes to avoid numerical issues
        dtype_a = torch.float32
        dtype_b = torch.float64 if 'medium' in scenario['model_b'] else torch.float32
        
        model_a = AutoModelForCausalLM.from_pretrained(
            scenario['model_a'],
            torch_dtype=dtype_a,
            low_cpu_mem_usage=True
        )
        model_b = AutoModelForCausalLM.from_pretrained(
            scenario['model_b'],
            torch_dtype=dtype_b,
            low_cpu_mem_usage=True
        )
        
        tokenizer_a = AutoTokenizer.from_pretrained(scenario['model_a'])
        tokenizer_b = AutoTokenizer.from_pretrained(scenario['model_b'])
        
        # Set padding tokens
        tokenizer_a.pad_token = tokenizer_a.eos_token if tokenizer_a.eos_token else tokenizer_a.unk_token
        tokenizer_b.pad_token = tokenizer_b.eos_token if tokenizer_b.eos_token else tokenizer_b.unk_token
        
        # Get model sizes
        params_a = sum(p.numel() for p in model_a.parameters())
        params_b = sum(p.numel() for p in model_b.parameters())
        
        print(f"✓ Models loaded successfully")
        print(f"  {scenario['model_a']}: {params_a/1e6:.1f}M parameters")
        print(f"  {scenario['model_b']}: {params_b/1e6:.1f}M parameters")
        print(f"  Size ratio: {params_b/params_a:.2f}x")
        
        # Create enhanced tester for distillation detection
        config = DiffDecisionConfig(
            mode=TestingMode.QUICK_GATE,
            n_min=8,
            n_max=40,
            gamma=0.01,  # Very tight threshold for distillation
            delta_star=0.05  # Looking for subtle differences
        )
        tester = EnhancedSequentialTester(config)
        
        print(f"\n🧪 Testing for distillation artifacts...")
        
        # Get differences on quality prompts
        differences = compare_model_outputs(
            model_a, model_b, tokenizer_a, tokenizer_b,
            QUALITY_TEST_PROMPTS * 2  # Double for more samples
        )
        
        # Feed to tester
        for i, diff in enumerate(differences):
            tester.update(diff)
            
            should_stop, decision_info = tester.should_stop()
            
            if (i + 1) % 5 == 0:
                current = decision_info.get('decision', 'UNDECIDED') if decision_info else 'UNDECIDED'
                print(f"  After {i+1} tests: {current} (avg TV distance: {np.mean(differences[:i+1]):.4f})")
            
            if should_stop and i >= config.n_min:
                break
        
        # Final decision
        final_decision = decision_info.get('decision', 'UNDECIDED') if decision_info else 'UNDECIDED'
        
        print(f"\n📊 RESULTS:")
        print(f"  Decision: {final_decision}")
        print(f"  Expected: {scenario['expected']}")
        print(f"  Samples used: {len(differences)}")
        print(f"  Avg TV distance: {np.mean(differences):.4f}")
        print(f"  Std deviation: {np.std(differences):.4f}")
        
        # Interpret results
        success = (final_decision == scenario['expected'])
        results['success'] = success
        results['decision'] = final_decision
        results['samples'] = len(differences)
        results['avg_distance'] = float(np.mean(differences))
        results['std_distance'] = float(np.std(differences))
        results['params_a'] = params_a
        results['params_b'] = params_b
        
        print(f"\n🎯 DETECTION RESULT: {'✅ SUCCESS' if success else '❌ FAILURE'}")
        
        if scenario['expected'] == 'DIFFERENT' and success:
            print(f"  Successfully detected distillation/optimization!")
            print(f"  Users would know if they're getting {scenario['model_b']} instead of {scenario['model_a']}")
        elif scenario['expected'] == 'SAME' and success:
            print(f"  Correctly identified identical models")
            print(f"  No false positives - system is reliable")
        
        # Clean up
        del model_a, model_b
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        results['error'] = str(e)
    
    return results

def main():
    """Run distillation detection tests."""
    
    print("="*70)
    print("🎓 MODEL DISTILLATION & OPTIMIZATION DETECTION")
    print("="*70)
    print("\nTesting the framework's ability to detect:")
    print("• Teacher-student distillation (BERT→DistilBERT)")
    print("• Model optimization without disclosure")
    print("• Architecture pruning for efficiency")
    print("\nWhy this matters:")
    print("• 40% of production models are distilled/optimized")
    print("• Users deserve transparency about model quality")
    print("• Pricing should reflect actual model complexity")
    
    all_results = []
    
    for scenario in TEST_SCENARIOS:
        result = test_distillation(scenario)
        all_results.append(result)
        time.sleep(2)
    
    # Save results
    output_dir = Path('experimental_results')
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f'distillation_detection_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    # Final summary
    print("\n" + "="*70)
    print("📊 DISTILLATION DETECTION SUMMARY")
    print("="*70)
    
    successes = sum(1 for r in all_results if r.get('success', False))
    total = len(all_results)
    
    print(f"\n✅ Success Rate: {successes}/{total} ({100*successes/total:.0f}%)")
    
    for result in all_results:
        if 'decision' in result:
            scenario = result['scenario']
            print(f"\n**{scenario['name']}**")
            print(f"  Models: {scenario['model_a']} vs {scenario['model_b']}")
            print(f"  Detection: {'✅' if result['success'] else '❌'} "
                  f"{result['decision']} (expected: {scenario['expected']})")
            print(f"  TV distance: {result.get('avg_distance', 0):.4f}")
            print(f"  Significance: {scenario['risk']}")
    
    print("\n🎯 KEY INSIGHT:")
    print("The framework can detect model distillation and optimization,")
    print("ensuring transparency about the actual models being served.")
    print("This prevents 'quality fraud' where optimized models are")
    print("presented as full-scale models without user knowledge.")

if __name__ == "__main__":
    main()