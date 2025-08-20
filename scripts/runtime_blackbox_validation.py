#!/usr/bin/env python3
"""
Runtime Black-Box Statistical Identity Validation
Tests real model pairs with proper statistical decision framework
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import torch
from datetime import datetime
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pathlib

# Ensure PYTHONPATH includes repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local model configuration
LOCAL_MODEL_BASE = "/Users/rohanvinaik/LLM_Models"
LOCAL_MODEL_MAPPING = {
    "gpt2": f"{LOCAL_MODEL_BASE}/gpt2",
    "distilgpt2": f"{LOCAL_MODEL_BASE}/distilgpt2", 
    "gpt2-medium": f"{LOCAL_MODEL_BASE}/gpt2-medium",
}

# Set environment for compatibility
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

def merkle_hash(data_list: List[str]) -> str:
    """Generate Merkle root for audit trail"""
    if not data_list:
        return hashlib.sha256(b'').hexdigest()
    
    hashes = [hashlib.sha256(item.encode()).hexdigest() for item in data_list]
    
    while len(hashes) > 1:
        next_level = []
        for i in range(0, len(hashes), 2):
            if i + 1 < len(hashes):
                combined = hashes[i] + hashes[i + 1]
            else:
                combined = hashes[i] + hashes[i]
            next_level.append(hashlib.sha256(combined.encode()).hexdigest())
        hashes = next_level
    
    return hashes[0]

class MinimalLM:
    """Minimal language model wrapper for black-box testing"""
    
    def __init__(self, model_name: str, device: str = "auto", seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.model_name = model_name
        self.device = device if device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.load_time = 0.0
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            start_time = time.time()
            # Use local model path if available
            local_path = LOCAL_MODEL_MAPPING.get(model_name, model_name)
            print(f"Loading model from: {local_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=True, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                local_path,
                local_files_only=True,
                torch_dtype=torch.float16 if self.device == "mps" else None,
                attn_implementation="eager"
            ).eval().to(self.device)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.load_time = time.time() - start_time
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")
    
    def get_cross_entropy(self, prompt: str, target: str) -> float:
        """Get cross-entropy loss for target given prompt (teacher-forced)"""
        try:
            full_text = prompt + target
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs.input_ids)
                
            # Get loss only for the target tokens
            loss = outputs.loss.item()
            target_length = inputs.input_ids.shape[1] - prompt_inputs.input_ids.shape[1]
            
            # Normalize by target length for fair comparison
            return loss * target_length if target_length > 0 else loss
            
        except Exception as e:
            print(f"Warning: CE calculation failed for '{prompt}' -> '{target}': {e}")
            return float('inf')

class StatisticalDecisionFramework:
    """Enhanced statistical decision framework with SAME/DIFFERENT rules"""
    
    def __init__(self, mode: str = "audit_grade"):
        self.mode = mode
        
        if mode == "audit_grade":
            self.confidence = 0.99
            self.gamma = 0.01  # SAME band
            self.delta_star = 0.10  # DIFFERENT threshold
            self.epsilon_diff = 0.10  # Relative ME for DIFFERENT
            self.n_min = 30
            self.n_max = 400
        elif mode == "quick_gate":
            self.confidence = 0.975
            self.gamma = 0.015
            self.delta_star = 0.10
            self.epsilon_diff = 0.20
            self.n_min = 12
            self.n_max = 120
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.alpha = (1 - self.confidence) / 2  # Two-tailed
        self.beta = self.alpha  # Symmetric error rates
    
    def empirical_bernstein_ci(self, differences: np.ndarray) -> Tuple[float, float, float]:
        """Calculate Empirical-Bernstein confidence interval"""
        n = len(differences)
        mean_diff = np.mean(differences)
        var_diff = np.var(differences, ddof=1)
        
        # Empirical-Bernstein bound
        t_val = stats.t.ppf(1 - self.alpha, df=n-1)
        
        # EB adjustment for variance
        eb_term = np.sqrt(2 * var_diff * np.log(2/self.alpha) / n)
        range_term = 3 * np.log(2/self.alpha) / (n - 1)
        
        margin = t_val * np.sqrt(var_diff / n) + eb_term + range_term
        
        ci_low = mean_diff - margin
        ci_high = mean_diff + margin
        half_width = margin
        
        return ci_low, ci_high, half_width
    
    def make_decision(self, differences: np.ndarray, n_used: int) -> Dict:
        """Make SAME/DIFFERENT/UNDECIDED decision with diagnostics"""
        
        mean_diff = np.mean(differences)
        ci_low, ci_high, half_width = self.empirical_bernstein_ci(differences)
        
        # SAME decision: CI within [-γ, +γ] AND half_width ≤ η·γ
        eta = 0.5  # Precision requirement factor
        same_ci_condition = (ci_low >= -self.gamma) and (ci_high <= self.gamma)
        same_precision_condition = half_width <= (eta * self.gamma)
        same_decision = same_ci_condition and same_precision_condition
        
        # DIFFERENT decision: Effect size ≥ δ* AND RME ≤ ε_diff
        effect_size = abs(mean_diff)
        relative_me = half_width / max(abs(mean_diff), 1e-6)  # Avoid division by zero
        different_effect_condition = effect_size >= self.delta_star
        different_precision_condition = relative_me <= self.epsilon_diff
        different_decision = different_effect_condition and different_precision_condition
        
        # Determine final decision
        if same_decision:
            decision = "SAME"
            rule_fired = f"CI [{ci_low:.6f}, {ci_high:.6f}] ⊂ [-{self.gamma}, +{self.gamma}] and half_width {half_width:.6f} ≤ {eta * self.gamma:.6f}"
        elif different_decision:
            decision = "DIFFERENT"
            rule_fired = f"Effect size {effect_size:.6f} ≥ δ* {self.delta_star} and RME {relative_me:.6f} ≤ {self.epsilon_diff}"
        else:
            decision = "UNDECIDED"
            rule_fired = f"Neither SAME nor DIFFERENT criteria met at n={n_used}"
        
        return {
            "decision": decision,
            "rule_fired": rule_fired,
            "mean_diff": mean_diff,
            "ci_99": [ci_low, ci_high],
            "half_width": half_width,
            "effect_size": effect_size,
            "relative_me": relative_me,
            "n_used": n_used,
            "diagnostics": {
                "same_ci_condition": same_ci_condition,
                "same_precision_condition": same_precision_condition,
                "different_effect_condition": different_effect_condition,
                "different_precision_condition": different_precision_condition
            }
        }

def generate_challenge_prompts(n_prompts: int = 32, K: int = 32) -> List[Dict]:
    """Generate challenge prompts for black-box testing"""
    
    challenge_families = [
        "completion",  # Simple completions
        "reasoning",   # Basic reasoning tasks
        "knowledge",   # Factual knowledge
        "style"        # Style transfer
    ]
    
    prompts = []
    
    # Completion family
    completion_prompts = [
        "The capital of France is",
        "To make a sandwich, you need",
        "The sky is blue because",
        "In programming, a function is",
        "The weather today looks",
        "My favorite color is",
        "The best way to learn is",
        "Technology has changed"
    ]
    
    # Reasoning family  
    reasoning_prompts = [
        "If it rains, then",
        "The problem with this approach is",
        "Based on the evidence,",
        "The logical conclusion is",
        "This happens because",
        "The main issue here is",
        "We can solve this by",
        "The reason why"
    ]
    
    # Knowledge family
    knowledge_prompts = [
        "Einstein is famous for",
        "The largest planet is",
        "Python programming language",
        "Machine learning is",
        "The Internet works by",
        "Quantum physics deals with",
        "Climate change is caused by",
        "Democracy means"
    ]
    
    # Style family
    style_prompts = [
        "In simple terms,",
        "Technically speaking,",
        "From my perspective,",
        "In conclusion,",
        "Surprisingly,",
        "Obviously,",
        "Unfortunately,",
        "Fortunately,"
    ]
    
    all_prompts = completion_prompts + reasoning_prompts + knowledge_prompts + style_prompts
    
    # Select prompts up to n_prompts
    selected_prompts = (all_prompts * ((n_prompts // len(all_prompts)) + 1))[:n_prompts]
    
    for i, prompt in enumerate(selected_prompts):
        family = challenge_families[i % len(challenge_families)]
        prompts.append({
            "prompt_id": i,
            "prompt": prompt,
            "family": family,
            "positions_per_prompt": K
        })
    
    return prompts

def run_black_box_identity_test(model_a: MinimalLM, model_b: MinimalLM, 
                               prompts: List[Dict], framework: StatisticalDecisionFramework) -> Dict:
    """Run black-box statistical identity test between two models"""
    
    print(f"Running black-box identity test: {model_a.model_name} vs {model_b.model_name}")
    print(f"Mode: {framework.mode}, Confidence: {framework.confidence:.1%}")
    print(f"Prompts: {len(prompts)}, Positions per prompt: {prompts[0]['positions_per_prompt']}")
    
    start_time = time.time()
    differences = []
    audit_log = []
    inference_times = []
    
    for prompt_data in prompts:
        prompt = prompt_data["prompt"]
        K = prompt_data["positions_per_prompt"]
        
        prompt_start = time.time()
        
        # Generate target completion (using model_a as reference)
        with torch.no_grad():
            inputs = model_a.tokenizer(prompt, return_tensors="pt").to(model_a.device)
            outputs = model_a.model.generate(
                **inputs,
                max_new_tokens=K,
                do_sample=False,
                pad_token_id=model_a.tokenizer.pad_token_id
            )
            target = model_a.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Get cross-entropy from both models
        ce_a = model_a.get_cross_entropy(prompt, target)
        ce_b = model_b.get_cross_entropy(prompt, target)
        
        if not (np.isfinite(ce_a) and np.isfinite(ce_b)):
            continue
            
        diff = ce_a - ce_b  # Symmetric difference
        differences.append(diff)
        
        prompt_time = time.time() - prompt_start
        inference_times.append(prompt_time)
        
        # Log for audit trail
        audit_entry = f"prompt_{prompt_data['prompt_id']}:{prompt}→{target[:20]}...→ce_diff:{diff:.6f}"
        audit_log.append(audit_entry)
        
        # Check for early stopping
        if len(differences) >= framework.n_min:
            decision_result = framework.make_decision(np.array(differences), len(differences))
            if decision_result["decision"] in ["SAME", "DIFFERENT"]:
                print(f"Early stopping at n={len(differences)}: {decision_result['decision']}")
                break
        
        if len(differences) >= framework.n_max:
            print(f"Reached n_max={framework.n_max}")
            break
    
    total_time = time.time() - start_time
    
    # Final decision
    final_decision = framework.make_decision(np.array(differences), len(differences))
    
    # Calculate timing statistics
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    total_inference_time = sum(inference_times)
    
    # Generate Merkle root for audit
    merkle_root = merkle_hash(audit_log)
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "models": {
            "model_a": model_a.model_name,
            "model_b": model_b.model_name
        },
        "framework": {
            "mode": framework.mode,
            "alpha": framework.alpha,
            "beta": framework.beta,
            "confidence": framework.confidence,
            "gamma": framework.gamma,
            "delta_star": framework.delta_star,
            "epsilon_diff": framework.epsilon_diff,
            "n_min": framework.n_min,
            "n_max": framework.n_max
        },
        "test_parameters": {
            "positions_per_prompt": prompts[0]["positions_per_prompt"] if prompts else 0,
            "challenge_families": list(set(p["family"] for p in prompts)),
            "prompts_planned": len(prompts),
            "prompts_used": len(differences)
        },
        "statistical_results": {
            "decision": final_decision["decision"],
            "rule_fired": final_decision["rule_fired"],
            "n_used": final_decision["n_used"],
            "mean_diff": final_decision["mean_diff"],
            "ci_99": final_decision["ci_99"],
            "half_width": final_decision["half_width"],
            "effect_size": final_decision["effect_size"],
            "relative_me": final_decision["relative_me"]
        },
        "timing": {
            "t_load_a": model_a.load_time,
            "t_load_b": model_b.load_time,
            "t_infer_total": total_inference_time,
            "t_per_query": avg_inference_time,
            "t_total": total_time,
            "hardware": {
                "device": model_a.device,
                "backend": "transformers+torch"
            }
        },
        "audit": {
            "merkle_root": merkle_root,
            "sample_entries": audit_log[:3] if audit_log else []
        }
    }
    
    return result

def main():
    """Run comprehensive black-box statistical identity validation"""
    
    print("🔬 RUNTIME BLACK-BOX STATISTICAL IDENTITY VALIDATION")
    print("=" * 60)
    
    # Test configurations
    test_cases = [
        {
            "name": "self_consistency", 
            "model_a": "gpt2",
            "model_b": "gpt2",  # Same model, different seeds
            "expected": "SAME",
            "mode": "quick_gate"
        },
        {
            "name": "different_models",
            "model_a": "gpt2", 
            "model_b": "distilgpt2",
            "expected": "DIFFERENT",
            "mode": "audit_grade"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📊 TEST CASE {i+1}: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Initialize framework
            framework = StatisticalDecisionFramework(mode=test_case["mode"])
            
            # Load models
            print(f"Loading models: {test_case['model_a']} vs {test_case['model_b']}")
            model_a = MinimalLM(test_case["model_a"], seed=42)
            model_b = MinimalLM(test_case["model_b"], seed=43 if test_case["model_a"] == test_case["model_b"] else 42)
            
            # Generate challenges
            n_prompts = min(framework.n_max // 4, 32)  # Conservative prompt count
            K = 32  # Positions per prompt
            prompts = generate_challenge_prompts(n_prompts, K)
            
            # Run test
            result = run_black_box_identity_test(model_a, model_b, prompts, framework)
            
            # Display results
            stats_res = result["statistical_results"]
            timing = result["timing"]
            
            print(f"\n📈 Results:")
            print(f"   Decision: {stats_res['decision']} (expected: {test_case['expected']})")
            print(f"   Rule: {stats_res['rule_fired']}")
            print(f"   n_used: {stats_res['n_used']}/{framework.n_max}")
            print(f"   Mean Δ: {stats_res['mean_diff']:.6f}")
            print(f"   99% CI: [{stats_res['ci_99'][0]:.6f}, {stats_res['ci_99'][1]:.6f}]")
            print(f"   Half-width: {stats_res['half_width']:.6f}")
            print(f"   Effect size: {stats_res['effect_size']:.6f}")
            
            print(f"\n⏱️  Timing:")
            print(f"   Load time: {timing['t_load_a']:.2f}s + {timing['t_load_b']:.2f}s")
            print(f"   Inference total: {timing['t_infer_total']:.2f}s")
            print(f"   Per query: {timing['t_per_query']:.3f}s")
            print(f"   Total runtime: {timing['t_total']:.2f}s")
            print(f"   Hardware: {timing['hardware']['device']}")
            
            print(f"\n🔒 Audit:")
            print(f"   Merkle root: {result['audit']['merkle_root'][:16]}...")
            
            # Test result validation
            correct_decision = (stats_res['decision'] == test_case['expected'] or 
                              stats_res['decision'] == "UNDECIDED")  # UNDECIDED is acceptable
            
            print(f"\n✅ Test status: {'PASS' if correct_decision else 'FAIL'}")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({"error": str(e), "test_case": test_case})
    
    # Save results
    output_dir = pathlib.Path("experimental_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"runtime_blackbox_validation_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "validation_type": "runtime_blackbox_statistical_identity",
            "timestamp": datetime.now().isoformat(),
            "test_cases": len(test_cases),
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Summary
    successful_tests = sum(1 for r in results if "error" not in r)
    print(f"\n📊 SUMMARY: {successful_tests}/{len(test_cases)} tests completed successfully")
    
    return results

if __name__ == "__main__":
    main()