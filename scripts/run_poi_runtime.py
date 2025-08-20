#!/usr/bin/env python3
"""
Runtime PoI verification with open model fallback and timing
"""

import time
import os
import sys
import logging
import json
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.scoring.teacher_forced import TeacherForcedScorer, ScoringConfig
from pot.core.statistical_policy import DiffDecisionConfig, SequentialDiffTester

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("poi_runtime")

def load_model(name, device="cpu", dtype=None):
    """Load model with proper configuration"""
    log.info(f"Loading model: {name}")
    
    try:
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            name, 
            torch_dtype=dtype or (torch.float16 if device == "cuda" else torch.float32)
        )
        mdl = mdl.eval().to(device)
        
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token
            
        return tok, mdl
    except Exception as e:
        log.error(f"Failed to load {name}: {e}")
        raise

def run_pair(ref_name, cand_name, mode="AUDIT_GRADE", device="cpu"):
    """Run verification for a model pair"""
    log.info(f"Testing {ref_name} vs {cand_name} in {mode} mode")
    
    # Load models
    t0 = time.time()
    tok_r, m_r = load_model(ref_name, device)
    tok_c, m_c = load_model(cand_name, device)
    t_load = time.time() - t0
    
    # Setup scorer with proper K for mode
    num_positions = 64 if mode == "QUICK_GATE" else 128
    scorer = TeacherForcedScorer(
        ScoringConfig(
            method="delta_ce",
            num_positions=num_positions,
            score_clip=(0.0, 1.0)  # Allow full range for teacher-forced scoring
        )
    )
    
    # Setup tester with calibrated values and practical settings for runtime
    epsilon_diff = 0.50 if mode == "QUICK_GATE" else 0.40  # Relaxed for teacher-forced variance
    cfg = DiffDecisionConfig(
        mode=mode,
        same_model_p95=0.00034,  # From calibration
        near_clone_p5=0.0763,     # From calibration
        use_calibration=True,
        positions_per_prompt=num_positions,  # Match scorer K
        score_clip_low=0.0,
        score_clip_high=1.0,  # Match scorer range
        epsilon_diff=epsilon_diff,  # Relaxed RME target
        force_decision_at_max=True,  # Force decision for runtime use
        n_max=200 if mode == "QUICK_GATE" else 300  # Reduce max samples for runtime
    )
    cfg.finalize()
    
    log.info(f"Config: γ={cfg.gamma:.6f}, δ*={cfg.delta_star:.4f}, K={cfg.positions_per_prompt}")
    
    tester = SequentialDiffTester(cfg)
    
    # Test prompts
    prompts = [
        "Explain evolution in simple terms.",
        "Translate to French: good morning.",
        "What is the capital of Japan?",
        "Summarize the purpose of a hash function.",
        "Write a short definition of entropy.",
        "What are the primary colors?",
        "Describe photosynthesis briefly.",
        "Define machine learning.",
        "What is quantum computing?",
        "Explain gravity to a child."
    ]
    
    # Run sampling
    t_inf = 0.0
    n_prompts_used = 0
    
    for round_idx in range(cfg.n_max // len(prompts) + 1):
        for prompt in prompts:
            if tester.n >= cfg.n_max:
                break
                
            # Tokenize
            full_text = prompt + " The answer is"
            inputs = tok_r(full_text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            prompt_tokens = tok_r(prompt, return_tensors="pt")
            prompt_len = prompt_tokens["input_ids"].shape[1]
            
            # Score
            q0 = time.time()
            score = scorer.score(m_r, m_c, inputs, prompt_len)
            t_inf += time.time() - q0
            
            tester.add_sample(score)
            n_prompts_used += 1
            
            # Progress logging (every 50 samples)
            if tester.n % 50 == 0 or tester.n <= 5:
                (lo, hi), h = tester.compute_ci()
                log.info(f"n={tester.n}: mean={tester.mean:.4f}, CI=[{lo:.4f}, {hi:.4f}]")
            
            # Check stopping
            if tester.n >= cfg.n_min:
                stop, info = tester.should_stop()
                if stop:
                    log.info(f"Decision: {info['decision']} | n={tester.n} | "
                            f"mean={tester.mean:.4f} | CI99={info['ci']}")
                    
                    return {
                        "ref": ref_name,
                        "cand": cand_name,
                        "mode": mode,
                        "alpha": cfg.alpha,
                        "beta": cfg.beta,
                        "gamma": cfg.gamma,
                        "delta_star": cfg.delta_star,
                        "n_used": tester.n,
                        "decision": info["decision"],
                        "rule": info.get("rule"),
                        "ci_99": list(info["ci"]),
                        "half_width": info["half_width"],
                        "mean": tester.mean,
                        "rme": info.get("rme", info["half_width"] / max(abs(tester.mean), cfg.min_effect_floor)),
                        "K": cfg.positions_per_prompt,
                        "t_load": t_load,
                        "t_infer_total": t_inf,
                        "t_per_query": t_inf / tester.n if tester.n else None,
                        "hardware": str(device)
                    }
    
    # Shouldn't reach here, but handle gracefully
    _, final_info = tester.should_stop()
    return {
        "ref": ref_name,
        "cand": cand_name,
        "mode": mode,
        "decision": "UNDECIDED",
        "n_used": tester.n,
        "mean": tester.mean,
        "t_load": t_load,
        "t_infer_total": t_inf,
        "t_per_query": t_inf / tester.n if tester.n else None
    }

def main():
    parser = argparse.ArgumentParser(description="Runtime PoI verification")
    parser.add_argument("--mode", choices=["quick", "audit"], default="audit")
    parser.add_argument("--pair", help="Model pair as ref:cand")
    parser.add_argument("--fallback", action="store_true", help="Use open models if gated fail")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    log.info(f"Using device: {device}")
    
    # Parse model pair
    if args.pair:
        ref, cand = args.pair.split(":")
        pairs = [(ref, cand)]
    else:
        # Default test pairs
        pairs = [
            ("gpt2", "gpt2"),
            ("gpt2", "distilgpt2")
        ]
    
    # Run tests
    results = []
    mode = "QUICK_GATE" if args.mode == "quick" else "AUDIT_GRADE"
    
    for ref, cand in pairs:
        try:
            result = run_pair(ref, cand, mode, device)
            results.append(result)
        except Exception as e:
            log.warning(f"Failed on {ref}/{cand}: {e}")
            
            if args.fallback and (ref != "gpt2" or cand != "distilgpt2"):
                log.info("Falling back to gpt2/distilgpt2")
                try:
                    result = run_pair("gpt2", "distilgpt2", mode, device)
                    result["note"] = f"Fallback from {ref}/{cand}"
                    results.append(result)
                except Exception as e2:
                    log.error(f"Fallback also failed: {e2}")
    
    # Output results
    output = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "mode": mode,
        "device": str(device),
        "results": results
    }
    
    print(json.dumps(output, indent=2))
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        log.info(f"Results saved to {args.output}")
    
    # Exit code based on results
    if all(r["decision"] != "UNDECIDED" for r in results):
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())