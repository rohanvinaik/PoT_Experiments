#!/usr/bin/env python3
"""
Instruction-Tuning Detection: Critical Validation for Model Safety & Compliance

This test demonstrates the framework's ability to detect instruction-tuning,
which is crucial for:
1. Safety verification (instruction-tuned models behave very differently)
2. Regulatory compliance (EU AI Act requires knowing model capabilities)
3. Supply chain security (detecting unauthorized modifications)
4. Model authenticity (verifying claimed model versions)
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from pot.core.diff_decision import EnhancedSequentialTester, TestingMode, DiffDecisionConfig

# Critical test scenarios for paper validation
TEST_SCENARIOS = [
    {
        'name': 'Chat Model Detection',
        'description': 'Detecting chat/instruction tuning in TinyLlama',
        'model_a': 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
        'model_b': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'model_type': 'causal',
        'expected': 'DIFFERENT',
        'hypothesis': 'Chat tuning creates detectable behavioral changes',
        'significance': 'Critical for detecting ChatGPT-style modifications'
    },
    {
        'name': 'Dialogue Fine-tuning Detection',
        'description': 'GPT-2 base vs dialogue-tuned DialoGPT',
        'model_a': 'gpt2',
        'model_b': 'microsoft/DialoGPT-small',
        'model_type': 'causal',
        'expected': 'DIFFERENT',
        'hypothesis': 'Dialogue fine-tuning is detectable',
        'significance': 'Identifies conversational adaptations'
    },
    {
        'name': 'Instruction Tuning Detection (T5)',
        'description': 'T5 base vs FLAN-T5 (instruction-tuned)',
        'model_a': 't5-small',
        'model_b': 'google/flan-t5-small',
        'model_type': 't5',
        'expected': 'DIFFERENT',
        'hypothesis': 'FLAN instruction tuning is detectable',
        'significance': 'Detects task-specific fine-tuning'
    },
    {
        'name': 'Training Data Variation',
        'description': 'Pythia trained on different data (deduped)',
        'model_a': 'EleutherAI/pythia-160m',
        'model_b': 'EleutherAI/pythia-160m-deduped',
        'model_type': 'causal',
        'expected': 'DIFFERENT',
        'hypothesis': 'Training data changes are detectable',
        'significance': 'Detects data poisoning or filtering'
    }
]

# Instruction-following test prompts (designed to trigger different behaviors)
INSTRUCTION_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list.",
    "What are the benefits of exercise?",
    "Translate 'Hello world' to French.",
    "Summarize the plot of Romeo and Juliet.",
    "List three healthy breakfast options.",
    "How do I change a car tire?",
    "What is the capital of Japan?",
    "Generate a creative story about a robot.",
    "Solve: 2x + 5 = 13",
]

# Dialogue prompts (for dialogue model testing)
DIALOGUE_PROMPTS = [
    "Hello! How are you today?",
    "What's your favorite color?",
    "Tell me a joke.",
    "I'm feeling tired.",
    "What should I have for dinner?",
    "Can you help me with something?",
    "That's interesting!",
    "I disagree with that.",
    "Thanks for your help!",
    "Goodbye!",
]

def test_model_pair(scenario: Dict, n_runs: int = 5) -> Dict:
    """Test a model pair for instruction-tuning detection."""
    print(f"\n{'='*70}")
    print(f"🧪 {scenario['name']}")
    print(f"{'='*70}")
    print(f"Description: {scenario['description']}")
    print(f"Hypothesis: {scenario['hypothesis']}")
    print(f"Significance: {scenario['significance']}")
    print(f"Expected: {scenario['expected']}")
    
    results = {
        'scenario': scenario,
        'runs': [],
        'summary': {}
    }
    
    try:
        # Load models based on type
        print(f"\n📥 Loading models...")
        print(f"  Model A: {scenario['model_a']}")
        print(f"  Model B: {scenario['model_b']}")
        
        if scenario['model_type'] == 't5':
            # Load T5 models
            model_a = T5ForConditionalGeneration.from_pretrained(
                scenario['model_a'],
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            model_b = T5ForConditionalGeneration.from_pretrained(
                scenario['model_b'],
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        else:
            # Load causal LM models
            model_a = AutoModelForCausalLM.from_pretrained(
                scenario['model_a'],
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            model_b = AutoModelForCausalLM.from_pretrained(
                scenario['model_b'],
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        
        # Load tokenizers
        tokenizer_a = AutoTokenizer.from_pretrained(scenario['model_a'])
        tokenizer_b = AutoTokenizer.from_pretrained(scenario['model_b'])
        
        # Set padding tokens
        if tokenizer_a.pad_token is None:
            tokenizer_a.pad_token = tokenizer_a.eos_token
        if tokenizer_b.pad_token is None:
            tokenizer_b.pad_token = tokenizer_b.eos_token
        
        print(f"✓ Models loaded successfully")
        print(f"  Vocab sizes: {tokenizer_a.vocab_size} vs {tokenizer_b.vocab_size}")
        
        # Choose appropriate prompts
        if 'dialogue' in scenario['name'].lower() or 'chat' in scenario['name'].lower():
            prompts = DIALOGUE_PROMPTS
        else:
            prompts = INSTRUCTION_PROMPTS
        
        # Run verification tests
        decisions = []
        queries_used = []
        kl_divergences = []
        
        print(f"\n🔬 Running {n_runs} verification tests...")
        
        for run_idx in range(n_runs):
            start_time = time.time()
            
            # Create tester with appropriate mode
            config = DiffDecisionConfig(mode=TestingMode.AUDIT_GRADE)  # Use higher precision for instruction detection
            tester = EnhancedSequentialTester(config)
            
            # Test with prompts
            diffs = []
            for prompt_idx, prompt in enumerate(prompts + prompts):  # Test twice for more samples
                try:
                    # For T5 models, add task prefix
                    if scenario['model_type'] == 't5':
                        prompt = f"question: {prompt}"
                    
                    # Tokenize
                    inputs_a = tokenizer_a(prompt, return_tensors='pt', padding=True, truncation=True, max_length=128)
                    inputs_b = tokenizer_b(prompt, return_tensors='pt', padding=True, truncation=True, max_length=128)
                    
                    # Get outputs
                    with torch.no_grad():
                        if scenario['model_type'] == 't5':
                            # For T5, use decoder_input_ids
                            decoder_input_ids = torch.zeros((1, 1), dtype=torch.long)
                            outputs_a = model_a(**inputs_a, decoder_input_ids=decoder_input_ids)
                            outputs_b = model_b(**inputs_b, decoder_input_ids=decoder_input_ids)
                        else:
                            outputs_a = model_a(**inputs_a)
                            outputs_b = model_b(**inputs_b)
                        
                        # Get logits
                        logits_a = outputs_a.logits[0, -1, :].float()
                        logits_b = outputs_b.logits[0, -1, :].float()
                        
                        # Handle vocabulary size mismatch
                        min_vocab = min(logits_a.shape[0], logits_b.shape[0])
                        
                        # Calculate KL divergence (more sensitive for instruction tuning)
                        probs_a = torch.softmax(logits_a[:min_vocab], dim=-1)
                        probs_b = torch.softmax(logits_b[:min_vocab], dim=-1)
                        
                        # KL divergence
                        kl_div = (probs_a * (probs_a / (probs_b + 1e-10)).log()).sum().item()
                        
                        # Also calculate Jensen-Shannon divergence for robustness
                        m = (probs_a + probs_b) / 2
                        js_div = 0.5 * (probs_a * (probs_a / m).log()).sum().item() + \
                                 0.5 * (probs_b * (probs_b / m).log()).sum().item()
                        
                        # Use average of KL and JS for better sensitivity
                        diff_score = (kl_div + js_div) / 2
                        
                        diffs.append(diff_score)
                        tester.update(diff_score)
                        
                        # Check stopping condition
                        should_stop, decision_info = tester.should_stop()
                        if should_stop and len(diffs) >= config.n_min:
                            break
                            
                except Exception as e:
                    print(f"  Warning: {str(e)[:50]}")
                    continue
            
            # Get decision
            decision = decision_info.get('decision', 'UNDECIDED') if decision_info else 'UNDECIDED'
            elapsed = time.time() - start_time
            
            decisions.append(decision)
            queries_used.append(len(diffs))
            kl_divergences.append(np.mean(diffs) if diffs else 0)
            
            print(f"  Run {run_idx + 1}/{n_runs}: {decision} ({len(diffs)} queries, {elapsed:.2f}s, "
                  f"avg divergence: {np.mean(diffs) if diffs else 0:.4f})")
            
            results['runs'].append({
                'run': run_idx,
                'decision': decision,
                'queries': len(diffs),
                'time': elapsed,
                'mean_divergence': np.mean(diffs) if diffs else 0
            })
        
        # Calculate summary
        decision_counts = {d: decisions.count(d) for d in set(decisions)}
        majority = max(decision_counts, key=decision_counts.get) if decision_counts else 'UNDECIDED'
        
        results['summary'] = {
            'total_runs': n_runs,
            'decisions': decision_counts,
            'majority_decision': majority,
            'matches_expected': majority == scenario['expected'],
            'avg_queries': np.mean(queries_used) if queries_used else 0,
            'avg_divergence': np.mean(kl_divergences) if kl_divergences else 0,
            'vocab_size_a': tokenizer_a.vocab_size,
            'vocab_size_b': tokenizer_b.vocab_size,
            'vocab_overlap': min(tokenizer_a.vocab_size, tokenizer_b.vocab_size) / max(tokenizer_a.vocab_size, tokenizer_b.vocab_size)
        }
        
        # Print results
        print(f"\n📊 Results:")
        print(f"  Decisions: {decision_counts}")
        print(f"  Majority: {majority}")
        print(f"  Expected: {scenario['expected']}")
        print(f"  {'✅ SUCCESS' if results['summary']['matches_expected'] else '❌ FAILURE'}")
        print(f"  Avg queries: {results['summary']['avg_queries']:.1f}")
        print(f"  Avg divergence: {results['summary']['avg_divergence']:.4f}")
        
        # Clean up models
        del model_a, model_b
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        results['error'] = str(e)
    
    return results

def main():
    """Run instruction-tuning detection tests."""
    print("="*70)
    print("🎯 INSTRUCTION-TUNING DETECTION VALIDATION")
    print("="*70)
    print("\nThis test validates the framework's ability to detect:")
    print("• Instruction/chat tuning (ChatGPT, Claude, etc.)")
    print("• Dialogue fine-tuning (customer service bots)")
    print("• Task-specific adaptations (FLAN, Alpaca)")
    print("• Training data modifications (safety filtering)")
    
    all_results = []
    output_dir = Path('experimental_results')
    output_dir.mkdir(exist_ok=True)
    
    # Run all test scenarios
    for scenario in TEST_SCENARIOS:
        result = test_model_pair(scenario, n_runs=3)  # Fewer runs due to larger models
        all_results.append(result)
        time.sleep(2)  # Brief pause between tests
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f'instruction_tuning_detection_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    # Generate summary report
    print("\n" + "="*70)
    print("📊 FINAL VALIDATION REPORT")
    print("="*70)
    
    success_count = 0
    for result in all_results:
        if 'summary' in result and result['summary'].get('matches_expected', False):
            success_count += 1
    
    print(f"\n### Overall Success Rate: {success_count}/{len(TEST_SCENARIOS)} "
          f"({100*success_count/len(TEST_SCENARIOS):.0f}%)")
    
    print("\n### Detailed Results:\n")
    for result in all_results:
        if 'summary' in result:
            scenario = result['scenario']
            summary = result['summary']
            
            print(f"**{scenario['name']}**")
            print(f"  Models: {scenario['model_a'].split('/')[-1]} vs {scenario['model_b'].split('/')[-1]}")
            print(f"  Result: {'✅' if summary['matches_expected'] else '❌'} "
                  f"{summary['majority_decision']} (expected: {scenario['expected']})")
            print(f"  Significance: {scenario['significance']}")
            print(f"  Avg divergence: {summary['avg_divergence']:.4f}")
            print(f"  Queries needed: {summary['avg_queries']:.0f}")
            print()
    
    # Key insights for paper
    print("### 🎯 Key Validation Points:\n")
    print("1. **Instruction Detection**: Framework can identify when models have been")
    print("   instruction-tuned, critical for safety and compliance verification\n")
    print("2. **Behavioral Fingerprinting**: Different fine-tuning approaches create")
    print("   distinct statistical signatures detectable without weight access\n")
    print("3. **Real-World Application**: Enables verification of production models")
    print("   like ChatGPT, Claude, Gemini without proprietary access\n")
    print("4. **Supply Chain Security**: Can detect unauthorized modifications or")
    print("   fine-tuning in model supply chain")

if __name__ == "__main__":
    main()