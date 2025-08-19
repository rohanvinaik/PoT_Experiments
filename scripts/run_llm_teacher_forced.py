#!/usr/bin/env python3
"""
LLM verification with teacher-forced scoring using open models.
Tests GPT-2 vs DistilGPT-2 with cross-entropy or symmetric KL divergence.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def compute_teacher_forced_scores(
    model1, 
    model2, 
    tokenizer,
    prompts: List[str],
    k_positions: int = 32,
    metric: str = "symmetric_kl"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute teacher-forced scoring between two models.
    
    Args:
        model1: Reference model
        model2: Candidate model
        tokenizer: Shared tokenizer
        prompts: List of prompts
        k_positions: Number of positions to evaluate per prompt
        metric: "ce" for cross-entropy, "symmetric_kl" for symmetric KL
    
    Returns:
        scores: Array of divergence scores
        stats: Statistics including CI and relative precision
    """
    scores = []
    
    for prompt in prompts:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        
        # Ensure we have at least k_positions
        if input_ids.shape[1] < k_positions + 1:
            # Pad or generate more tokens
            continue
        
        with torch.no_grad():
            # Get logits from both models
            logits1 = model1(**inputs).logits
            logits2 = model2(**inputs).logits
            
            # Sample K positions (excluding first token)
            positions = np.random.choice(
                range(1, min(logits1.shape[1], k_positions + 1)), 
                size=min(k_positions, logits1.shape[1] - 1),
                replace=False
            )
            
            for pos in positions:
                # Get distributions at position
                p1 = torch.softmax(logits1[0, pos], dim=-1)
                p2 = torch.softmax(logits2[0, pos], dim=-1)
                
                # Compute divergence
                if metric == "symmetric_kl":
                    # D_KL(p1||p2) + D_KL(p2||p1)
                    kl_12 = torch.sum(p1 * (torch.log(p1 + 1e-10) - torch.log(p2 + 1e-10)))
                    kl_21 = torch.sum(p2 * (torch.log(p2 + 1e-10) - torch.log(p1 + 1e-10)))
                    score = (kl_12 + kl_21).item() / 2
                else:  # cross-entropy
                    # -sum(p1 * log(p2))
                    score = -torch.sum(p1 * torch.log(p2 + 1e-10)).item()
                
                scores.append(score)
    
    scores = np.array(scores)
    
    # Calculate statistics
    mean = np.mean(scores)
    std = np.std(scores)
    n = len(scores)
    
    # 99% CI
    z_99 = 2.576
    std_err = std / np.sqrt(n)
    ci_lower = mean - z_99 * std_err
    ci_upper = mean + z_99 * std_err
    half_width = z_99 * std_err
    rel_precision = (half_width / mean * 100) if mean > 0 else 0
    
    # Check if CI excludes 0
    excludes_zero = ci_lower > 0 or ci_upper < 0
    
    stats = {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "n": n,
        "ci_99": [round(ci_lower, 6), round(ci_upper, 6)],
        "half_width": round(half_width, 6),
        "rel_precision": round(rel_precision, 2),
        "excludes_zero": excludes_zero,
        "decision": "DIFFERENT" if excludes_zero else "UNDECIDED"
    }
    
    return scores, stats


def main():
    """Run LLM verification with teacher-forced scoring."""
    logger.info("=" * 70)
    logger.info("LLM TEACHER-FORCED VERIFICATION")
    logger.info("=" * 70)
    
    # Check for transformers
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("❌ transformers not installed. Run: pip install transformers")
        return 1
    
    # Models to test (open, no auth required)
    model_pairs = [
        {
            "name": "GPT-2 vs Self",
            "model1": "gpt2",
            "model2": "gpt2",
            "expected": "SAME"
        },
        {
            "name": "GPT-2 vs DistilGPT-2",
            "model1": "gpt2",
            "model2": "distilgpt2",
            "expected": "DIFFERENT"
        }
    ]
    
    # Test prompts
    test_prompts = [
        "The weather today is",
        "In the future, artificial intelligence will",
        "The most important thing about",
        "Scientists have discovered that",
        "The economic impact of",
        "Recent studies show that",
        "Technology companies are developing",
        "The environmental effects of",
        "Healthcare systems around the world",
        "Educational institutions must adapt to"
    ]
    
    results = {}
    
    for pair in model_pairs:
        logger.info(f"\nTesting: {pair['name']}")
        logger.info("-" * 40)
        
        start_time = time.time()
        
        try:
            # Load models
            logger.info(f"Loading {pair['model1']}...")
            model1 = AutoModelForCausalLM.from_pretrained(pair['model1'])
            tokenizer = AutoTokenizer.from_pretrained(pair['model1'])
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"Loading {pair['model2']}...")
            model2 = AutoModelForCausalLM.from_pretrained(pair['model2'])
            
            load_time = time.time() - start_time
            
            # Compute scores
            inference_start = time.time()
            scores, stats = compute_teacher_forced_scores(
                model1, model2, tokenizer,
                prompts=test_prompts,
                k_positions=32,
                metric="symmetric_kl"
            )
            inference_time = time.time() - inference_start
            
            # Add timing info
            stats["time"] = {
                "load": round(load_time, 3),
                "infer_total": round(inference_time, 3),
                "per_query": round(inference_time / stats["n"] if stats["n"] > 0 else 0, 6)
            }
            stats["positions_per_prompt"] = 32
            stats["alpha"] = 0.01
            stats["beta"] = 0.01
            
            # Log results
            logger.info(f"Mean divergence: {stats['mean']:.6f}")
            logger.info(f"99% CI: [{stats['ci_99'][0]:.6f}, {stats['ci_99'][1]:.6f}]")
            logger.info(f"CI excludes 0: {stats['excludes_zero']}")
            logger.info(f"Relative precision: {stats['rel_precision']:.2f}%")
            logger.info(f"Decision: {stats['decision']}")
            logger.info(f"Expected: {pair['expected']}")
            
            # Check if result matches expectation
            if stats['decision'] == pair['expected'] or \
               (pair['expected'] == 'SAME' and not stats['excludes_zero']):
                logger.info("✅ Result matches expectation")
                stats['test_passed'] = True
            else:
                logger.info("❌ Result does not match expectation")
                stats['test_passed'] = False
            
            results[pair['name']] = stats
            
        except Exception as e:
            logger.error(f"Error testing {pair['name']}: {e}")
            results[pair['name']] = {"error": str(e), "test_passed": False}
    
    # Save results
    output_file = "experimental_results/llm_teacher_forced_results.json"
    os.makedirs("experimental_results", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    # Check overall success
    all_passed = all(r.get("test_passed", False) for r in results.values())
    
    if all_passed:
        logger.info("\n✅ LLM teacher-forced verification PASSED")
        return 0
    else:
        logger.info("\n❌ LLM teacher-forced verification FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())