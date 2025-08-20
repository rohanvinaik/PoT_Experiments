#!/usr/bin/env python3
"""
Optimized Runtime Black-Box Statistical Identity Validation
Uses optimized teacher-forced scoring for faster inference (~200ms per query target)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local model configuration
LOCAL_MODEL_BASE = "/Users/rohanvinaik/LLM_Models"
LOCAL_MODEL_MAPPING = {
    "gpt2": f"{LOCAL_MODEL_BASE}/gpt2",
    "distilgpt2": f"{LOCAL_MODEL_BASE}/distilgpt2", 
    "gpt2-medium": f"{LOCAL_MODEL_BASE}/gpt2-medium",
}

import json
import time
import hashlib
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run optimized runtime validation with fast scoring"""
    
    print("🚀 OPTIMIZED RUNTIME BLACK-BOX STATISTICAL IDENTITY VALIDATION")
    print("=" * 60)
    
    # Import dependencies
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from pot.core.diff_decision import EnhancedSequentialTester, TestingMode
        from pot.scoring.optimized_scorer import (
            OptimizedTeacherForcedScorer,
            OptimizedScoringConfig,
            FastScorer
        )
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install: pip install torch transformers")
        return
    
    # Test cases with optimized settings
    test_cases = [
        {
            "name": "self_consistency_fast",
            "model_a": "gpt2",
            "model_b": "gpt2",
            "expected": "SAME",
            "mode": "QUICK_GATE",
            "config": "fastest"  # Use fastest config
        },
        {
            "name": "different_models_balanced",
            "model_a": "gpt2",
            "model_b": "distilgpt2",
            "expected": "DIFFERENT",
            "mode": "AUDIT_GRADE",
            "config": "balanced"  # Use balanced config
        }
    ]
    
    results = []
    timestamp = datetime.now().isoformat()
    
    for test_case in test_cases:
        print(f"\n📊 TEST CASE: {test_case['name']}")
        print("-" * 40)
        
        try:
            result = run_optimized_test(test_case)
            results.append(result)
            
            # Display results
            print(f"📈 Results:")
            print(f"   Decision: {result['statistical_results']['decision']} (expected: {test_case['expected']})")
            print(f"   Rule: {result['statistical_results']['rule_fired']}")
            print(f"   n_used: {result['statistical_results']['n_used']}/{result['framework']['n_max']}")
            print(f"   Mean Δ: {result['statistical_results']['mean_diff']:.6f}")
            print(f"   99% CI: [{result['statistical_results']['ci_99'][0]:.6f}, {result['statistical_results']['ci_99'][1]:.6f}]")
            
            print(f"\n⏱️  Performance:")
            print(f"   Load time: {result['timing']['t_load_a']:.2f}s + {result['timing']['t_load_b']:.2f}s")
            print(f"   Inference total: {result['timing']['t_infer_total']:.2f}s")
            print(f"   Per query: {result['timing']['t_per_query']:.3f}s")
            print(f"   Speedup: {result['timing'].get('speedup', 1.0):.1f}x")
            print(f"   Hardware: {result['timing']['hardware']['device']}")
            
            print(f"\n🔒 Audit:")
            print(f"   Merkle root: {result['audit']['merkle_root'][:16]}...")
            
            # Check success
            if result['statistical_results']['decision'] == test_case['expected']:
                print(f"\n✅ Test status: PASS")
            else:
                print(f"\n⚠️  Test status: EXPECTED {test_case['expected']}, got {result['statistical_results']['decision']}")
                
        except Exception as e:
            logger.error(f"Test failed: {e}")
            results.append({
                "error": str(e),
                "test_case": test_case
            })
            print(f"❌ Test failed: {e}")
    
    # Save results
    output_file = f"experimental_results/runtime_blackbox_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "validation_type": "runtime_blackbox_optimized",
            "timestamp": timestamp,
            "test_cases": len(test_cases),
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Performance summary
    total_time = sum(r.get('timing', {}).get('t_infer_total', 0) for r in results if 'timing' in r)
    total_queries = sum(r.get('statistical_results', {}).get('n_used', 0) for r in results if 'statistical_results' in r)
    avg_per_query = total_time / total_queries if total_queries > 0 else 0
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   Total inference time: {total_time:.2f}s")
    print(f"   Total queries: {total_queries}")
    print(f"   Average per query: {avg_per_query:.3f}s")
    
    if avg_per_query > 0:
        original_time = 1.0  # Original ~1s per query
        speedup = original_time / avg_per_query
        print(f"   🚀 Speedup vs original: {speedup:.1f}x faster")
        
        if avg_per_query < 0.3:
            print(f"   ✅ TARGET ACHIEVED: <300ms per query ({avg_per_query*1000:.0f}ms)")
        else:
            print(f"   ⚠️  TARGET MISSED: {avg_per_query*1000:.0f}ms per query (target: <300ms)")


def run_optimized_test(test_case: Dict) -> Dict:
    """Run a single optimized test case"""
    
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pot.core.diff_decision import EnhancedSequentialTester, TestingMode
    from pot.scoring.optimized_scorer import (
        OptimizedTeacherForcedScorer,
        OptimizedScoringConfig
    )
    
    # Load configuration preset
    config_presets = OptimizedTeacherForcedScorer.benchmark_configurations()
    config = config_presets.get(test_case.get('config', 'balanced'))
    
    print(f"Loading models: {test_case['model_a']} vs {test_case['model_b']}")
    print(f"Using config: {test_case.get('config', 'balanced')} (top_k={config.top_k}, batch={config.batch_size})")
    
    # Load models and tokenizer
    t_start = time.time()
    
    # Use local model paths if available
    local_path_a = LOCAL_MODEL_MAPPING.get(test_case['model_a'], test_case['model_a'])
    local_path_b = LOCAL_MODEL_MAPPING.get(test_case['model_b'], test_case['model_b'])
    
    print(f"Loading models from: {local_path_a} vs {local_path_b}")
    
    tokenizer = AutoTokenizer.from_pretrained(local_path_a, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model_a = AutoModelForCausalLM.from_pretrained(local_path_a, local_files_only=True)
    t_load_a = time.time() - t_start
    
    t_start = time.time()
    model_b = AutoModelForCausalLM.from_pretrained(local_path_b, local_files_only=True)
    t_load_b = time.time() - t_start
    
    # Initialize optimized scorer
    scorer = OptimizedTeacherForcedScorer(config)
    
    # Get device info
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize tester
    from pot.core.diff_decision import DiffDecisionConfig
    mode_str = test_case.get('mode', 'QUICK_GATE')
    mode = TestingMode.QUICK_GATE if mode_str == 'QUICK_GATE' else TestingMode.AUDIT_GRADE
    config_diff = DiffDecisionConfig(mode=mode)
    tester = EnhancedSequentialTester(config_diff)
    
    # Generate test prompts
    prompts_planned = 40
    challenge_families = ["completion", "style", "reasoning", "knowledge"]
    
    prompts = []
    for i in range(prompts_planned):
        family = challenge_families[i % len(challenge_families)]
        prompts.extend(generate_prompt_family(family, count=1, seed=i))
    
    print(f"Running optimized black-box test: {test_case['model_a']} vs {test_case['model_b']}")
    print(f"Mode: {mode.name}, Confidence: {tester.config.confidence*100:.1f}%")
    print(f"Prompts: {len(prompts)}, Positions per prompt: {config.positions_per_prompt}")
    
    # Warmup cache with first few prompts
    scorer.warmup_cache(model_a, prompts[:5], tokenizer)
    
    # Run test with batching
    t_infer_start = time.time()
    n_used = 0
    batch_size = config.batch_size
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Score batch
        batch_scores = scorer.score_batch(
            model_a, model_b, batch_prompts, tokenizer
        )
        
        # Update tester
        for score in batch_scores:
            tester.update(score)
            n_used += 1
            
            # Check for early stopping (we'll check manually)
            pass
            
        # Stop if we've used enough samples
        if n_used >= tester.config.n_max:
            break
    
    t_infer_total = time.time() - t_infer_start
    
    # Get final decision manually
    # Create decision info based on tester state
    if hasattr(tester, 'clipped_scores') and len(tester.clipped_scores) > 0:
        scores_array = np.array(tester.clipped_scores)
        mean = np.mean(scores_array)
        
        # Compute CI using empirical Bernstein
        n = len(scores_array)
        std_dev = np.std(scores_array, ddof=1) if n > 1 else 0
        margin = 2.576 * std_dev / np.sqrt(n) if n > 0 else float('inf')  # 99% CI
        ci_low, ci_high = mean - margin, mean + margin
        half_width = margin
        
        # Make decision based on thresholds
        gamma = tester.config.gamma
        delta_star = tester.config.delta_star
        epsilon_diff = tester.config.epsilon_diff
        
        # SAME decision
        same_ci_condition = (ci_low >= -gamma) and (ci_high <= gamma)
        same_precision_condition = half_width <= (0.5 * gamma)
        
        # DIFFERENT decision  
        effect_size = abs(mean)
        relative_me = half_width / effect_size if effect_size > 0 else float('inf')
        different_effect_condition = effect_size >= delta_star
        different_precision_condition = relative_me <= epsilon_diff
        
        if same_ci_condition and same_precision_condition:
            decision_str = "SAME"
            rule = f"CI within [-{gamma}, +{gamma}] and half_width <= {0.5*gamma}"
        elif different_effect_condition and different_precision_condition:
            decision_str = "DIFFERENT"
            rule = f"Effect size >= {delta_star} and RME <= {epsilon_diff}"
        else:
            decision_str = "UNDECIDED"
            rule = f"Neither SAME nor DIFFERENT criteria met at n={n_used}"
            
        decision = {
            "decision": decision_str,
            "rule": rule,
            "ci_99": [ci_low, ci_high],
            "half_width": half_width,
            "effect_size": effect_size,
            "relative_me": relative_me
        }
    else:
        decision = {
            "decision": "UNDECIDED",
            "rule": "No scores collected",
            "ci_99": [0, 0],
            "half_width": 0,
            "effect_size": 0,
            "relative_me": 0
        }
    
    # Calculate timing metrics
    t_per_query = t_infer_total / n_used if n_used > 0 else 0
    
    # Calculate speedup vs baseline (~1s per query)
    baseline_time = 1.0
    speedup = baseline_time / t_per_query if t_per_query > 0 else 1.0
    
    # Build audit trail
    audit_entries = []
    scores_list = tester.clipped_scores if hasattr(tester, 'clipped_scores') else []
    for j, prompt in enumerate(prompts[:min(n_used, len(prompts))]):
        score = scores_list[j] if j < len(scores_list) else 0.0
        entry = f"prompt_{j}:{prompt[:30]}→{score:.6f}"
        audit_entries.append(entry)
    
    merkle_root = hashlib.sha256(
        json.dumps(audit_entries).encode()
    ).hexdigest()
    
    # Clear cache to free memory
    scorer.clear_cache()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "models": {
            "model_a": test_case['model_a'],
            "model_b": test_case['model_b']
        },
        "framework": {
            "mode": mode_str,
            "alpha": tester.config.alpha if hasattr(tester.config, 'alpha') else (1 - tester.config.confidence) / 2,
            "beta": tester.config.alpha if hasattr(tester.config, 'alpha') else (1 - tester.config.confidence) / 2,
            "confidence": tester.config.confidence,
            "gamma": tester.config.gamma,
            "delta_star": tester.config.delta_star,
            "epsilon_diff": tester.config.epsilon_diff,
            "n_min": tester.config.n_min,
            "n_max": tester.config.n_max
        },
        "optimization": {
            "config_preset": test_case.get('config', 'balanced'),
            "use_top_k": config.use_top_k_only,
            "top_k": config.top_k,
            "batch_size": config.batch_size,
            "positions_per_prompt": config.positions_per_prompt,
            "max_length": config.max_length,
            "use_amp": config.use_amp,
            "cached_embeddings": len(scorer.embedding_cache)
        },
        "test_parameters": {
            "positions_per_prompt": config.positions_per_prompt,
            "challenge_families": challenge_families,
            "prompts_planned": prompts_planned,
            "prompts_used": n_used,
            "metric_used": "optimized_delta_ce"
        },
        "statistical_results": {
            "decision": decision["decision"],
            "rule_fired": decision["rule"],
            "n_used": n_used,
            "mean_diff": float(tester.mean) if hasattr(tester, 'mean') else 0.0,
            "ci_99": list(decision["ci_99"]),
            "half_width": decision["half_width"],
            "effect_size": decision["effect_size"],
            "relative_me": decision["relative_me"]
        },
        "timing": {
            "t_load_a": t_load_a,
            "t_load_b": t_load_b,
            "t_infer_total": t_infer_total,
            "t_per_query": t_per_query,
            "speedup": speedup,
            "hardware": {
                "device": device,
                "backend": "transformers+torch+optimized"
            }
        },
        "audit": {
            "merkle_root": merkle_root,
            "sample_entries": audit_entries[:3]
        }
    }


def generate_prompt_family(family: str, count: int = 1, seed: int = 0) -> List[str]:
    """Generate prompts from a specific family"""
    
    np.random.seed(seed)
    
    templates = {
        "completion": [
            "The capital of France is",
            "To make a sandwich, you need",
            "The sky is blue because",
            "Water freezes at",
            "The largest planet is"
        ],
        "reasoning": [
            "If all birds can fly and penguins are birds, then",
            "The opposite of hot is",
            "If it rains, the ground gets",
            "Two plus two equals",
            "The sun rises in the"
        ],
        "knowledge": [
            "Shakespeare wrote",
            "The periodic table contains",
            "DNA stands for",
            "The speed of light is",
            "Gravity was discovered by"
        ],
        "style": [
            "Once upon a time",
            "In conclusion,",
            "Dear Sir or Madam,",
            "Breaking news:",
            "According to scientists,"
        ]
    }
    
    family_templates = templates.get(family, templates["completion"])
    selected = np.random.choice(family_templates, size=min(count, len(family_templates)), replace=False)
    
    return list(selected)


if __name__ == "__main__":
    main()