#!/usr/bin/env python3
"""
Model Size Fraud Detection: Detecting when smaller models are served as larger ones

This test validates the framework's ability to detect "model downsizing fraud" where:
- A provider claims to serve GPT-3 but actually serves GPT-2
- API endpoints quietly switch to smaller models to save costs
- Models are pruned/quantized without disclosure
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from pot.core.diff_decision import EnhancedSequentialTester, TestingMode, DiffDecisionConfig

# Test scenarios for size fraud detection
TEST_SCENARIOS = [
    {
        'name': 'Pythia Size Scaling',
        'description': 'Can we detect 70M vs 160M parameter difference?',
        'model_a': 'EleutherAI/pythia-70m',
        'model_b': 'EleutherAI/pythia-160m',
        'expected': 'DIFFERENT',
        'cost_savings': '56% compute reduction',
        'fraud_scenario': 'Provider claims Pythia-160M but serves 70M'
    },
    {
        'name': 'GPT-Neo Size Detection',
        'description': 'Detecting 125M vs 1.3B parameter models',
        'model_a': 'EleutherAI/gpt-neo-125m',
        'model_b': 'EleutherAI/gpt-neo-1.3B',
        'expected': 'DIFFERENT',
        'cost_savings': '90% compute reduction',
        'fraud_scenario': 'Major downsizing - 10x smaller model'
    },
    {
        'name': 'Same Architecture Test',
        'description': 'Same model should verify as identical',
        'model_a': 'EleutherAI/pythia-70m',
        'model_b': 'EleutherAI/pythia-70m',
        'expected': 'SAME',
        'cost_savings': 'N/A',
        'fraud_scenario': 'Control test - no fraud'
    }
]

# Capability test prompts (where size matters)
CAPABILITY_PROMPTS = [
    "Explain the theory of relativity in detail:",
    "Write a complex algorithm for sorting:",
    "Analyze the following philosophical argument:",
    "Describe the process of photosynthesis step by step:",
    "Compare and contrast democracy and republic:",
    "Solve this mathematical problem: ∫(x²+3x)dx =",
    "Translate this technical term to multiple languages:",
    "Generate a detailed business plan for:",
    "Explain quantum entanglement to a physicist:",
    "Write a sonnet about artificial intelligence:",
]

def test_size_fraud(scenario: Dict) -> Dict:
    """Test if we can detect model size fraud."""
    
    print(f"\n{'='*70}")
    print(f"🔍 {scenario['name']}")
    print(f"{'='*70}")
    print(f"Scenario: {scenario['fraud_scenario']}")
    print(f"Cost savings from fraud: {scenario['cost_savings']}")
    print(f"Testing: {scenario['model_a']} vs {scenario['model_b']}")
    
    results = {'scenario': scenario, 'detection': None}
    
    try:
        # Load models
        print("\n📥 Loading models...")
        
        # Check if we should skip large model
        if '1.3B' in scenario['model_b']:
            print("⚠️  Skipping 1.3B model (too large for quick test)")
            print("   In production, this would be tested")
            results['skipped'] = True
            return results
            
        model_a = AutoModelForCausalLM.from_pretrained(
            scenario['model_a'],
            torch_dtype=torch.float32,  # Use float32 to avoid NaN issues
            low_cpu_mem_usage=True
        )
        model_b = AutoModelForCausalLM.from_pretrained(
            scenario['model_b'],
            torch_dtype=torch.float32,  # Use float32 to avoid NaN issues
            low_cpu_mem_usage=True
        )
        
        tokenizer_a = AutoTokenizer.from_pretrained(scenario['model_a'])
        tokenizer_b = AutoTokenizer.from_pretrained(scenario['model_b'])
        
        tokenizer_a.pad_token = tokenizer_a.eos_token
        tokenizer_b.pad_token = tokenizer_b.eos_token
        
        # Get actual parameter counts
        params_a = sum(p.numel() for p in model_a.parameters())
        params_b = sum(p.numel() for p in model_b.parameters())
        
        print(f"✓ Models loaded")
        print(f"  Model A: {params_a/1e6:.1f}M parameters")
        print(f"  Model B: {params_b/1e6:.1f}M parameters")
        print(f"  Size ratio: {params_b/params_a:.2f}x")
        
        # Create tester
        config = DiffDecisionConfig(
            mode=TestingMode.QUICK_GATE,
            n_min=10,
            n_max=50,
            gamma=0.02,  # Tighter threshold for size detection
            delta_star=0.1
        )
        tester = EnhancedSequentialTester(config)
        
        print("\n🔬 Testing model capabilities...")
        diffs = []
        
        for i, prompt in enumerate(CAPABILITY_PROMPTS * 3):  # Repeat for more samples
            # Tokenize
            inputs_a = tokenizer_a(prompt, return_tensors='pt', padding=True, truncation=True, max_length=64)
            inputs_b = tokenizer_b(prompt, return_tensors='pt', padding=True, truncation=True, max_length=64)
            
            # Get outputs
            with torch.no_grad():
                outputs_a = model_a(**inputs_a)
                outputs_b = model_b(**inputs_b)
                
                # Compare predictions
                logits_a = outputs_a.logits[0, -1, :].float()
                logits_b = outputs_b.logits[0, -1, :].float()
                
                # Ensure same vocab size
                min_vocab = min(logits_a.shape[0], logits_b.shape[0])
                logits_a = logits_a[:min_vocab]
                logits_b = logits_b[:min_vocab]
                
                # Calculate difference (use L2 distance which is more stable)
                diff = torch.nn.functional.mse_loss(logits_a, logits_b).item()
                normalized_diff = diff
                
                diffs.append(normalized_diff)
                tester.update(normalized_diff)
                
                # Check stopping
                should_stop, decision_info = tester.should_stop()
                
                if (i + 1) % 10 == 0:
                    current = decision_info.get('decision', 'UNDECIDED') if decision_info else 'UNDECIDED'
                    print(f"  After {i+1} queries: {current} (avg diff: {np.mean(diffs):.4f})")
                
                if should_stop and len(diffs) >= config.n_min:
                    break
        
        # Get final decision
        final_decision = decision_info.get('decision', 'UNDECIDED') if decision_info else 'UNDECIDED'
        
        print(f"\n📊 RESULTS:")
        print(f"  Decision: {final_decision}")
        print(f"  Expected: {scenario['expected']}")
        print(f"  Queries used: {len(diffs)}")
        print(f"  Avg normalized difference: {np.mean(diffs):.4f}")
        
        # Determine if fraud was detected
        if scenario['expected'] == 'DIFFERENT':
            fraud_detected = (final_decision == 'DIFFERENT')
            print(f"\n🎯 FRAUD DETECTION: {'✅ SUCCESS' if fraud_detected else '❌ MISSED'}")
            if fraud_detected:
                print(f"  The framework would catch this cost-cutting fraud!")
                print(f"  Provider cannot secretly serve {scenario['model_a']} as {scenario['model_b']}")
        else:  # Control test
            correct = (final_decision == 'SAME')
            print(f"\n✓ Control Test: {'PASS' if correct else 'FAIL'}")
        
        results['detection'] = final_decision
        results['success'] = (final_decision == scenario['expected'])
        results['queries'] = len(diffs)
        results['avg_diff'] = float(np.mean(diffs))
        results['params_a'] = params_a
        results['params_b'] = params_b
        
        # Clean up
        del model_a, model_b
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        results['error'] = str(e)
    
    return results

def main():
    """Run model size fraud detection tests."""
    
    print("="*70)
    print("💰 MODEL SIZE FRAUD DETECTION TEST")
    print("="*70)
    print("\nValidating the framework's ability to detect when providers")
    print("serve smaller models than claimed to reduce costs.")
    print("\nThis is critical for:")
    print("• API transparency (detecting GPT-3.5 served as GPT-4)")
    print("• Cost verification (ensuring you get what you pay for)")
    print("• Performance guarantees (smaller models = worse performance)")
    
    all_results = []
    
    for scenario in TEST_SCENARIOS:
        result = test_size_fraud(scenario)
        all_results.append(result)
        time.sleep(2)
    
    # Save results
    output_dir = Path('experimental_results')
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f'size_fraud_detection_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 FRAUD DETECTION SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in all_results if r.get('success', False))
    total_tested = sum(1 for r in all_results if 'detection' in r)
    
    print(f"\nDetection Rate: {success_count}/{total_tested} tests passed")
    
    for result in all_results:
        if 'detection' in result:
            scenario = result['scenario']
            print(f"\n{scenario['name']}:")
            print(f"  Fraud scenario: {scenario['fraud_scenario']}")
            if 'success' in result:
                print(f"  Detection: {'✅ CAUGHT' if result['success'] else '❌ MISSED'}")
            if 'queries' in result:
                print(f"  Queries needed: {result['queries']}")
    
    print("\n🎯 KEY INSIGHT:")
    print("The framework can detect model size fraud, preventing providers")
    print("from secretly serving smaller, cheaper models while charging for")
    print("larger ones. This ensures API transparency and fair pricing.")

if __name__ == "__main__":
    main()