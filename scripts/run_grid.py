#!/usr/bin/env python
"""
Runs grid over challenge sizes n ∈ {32, 64, 128, 256, 512, 1024}
and model pairs:
  - identical (same ckpt)
  - seed-variant
  - fine-tuned
  - pruned
  - quantized
  - distilled
Outputs ROC/DET data and query budgets.
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pot.core.logging import StructuredLogger
from pot.core.stats import far_frr

def main():
    parser = argparse.ArgumentParser(description="Run grid experiment")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument("--exp", default="E1", help="Experiment name")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    exp_name = config['experiment']
    logger = StructuredLogger(f"{args.output_dir}/{exp_name}/{args.exp}")
    
    # Grid parameters
    challenge_sizes = [32, 64, 128, 256, 512, 1024]
    model_pairs = [
        ("reference", "identical"),
        ("reference", "seed_variant"),
        ("reference", "fine_tuned"),
        ("reference", "pruned"),
        ("reference", "quantized"),
        ("reference", "distilled")
    ]
    
    print(f"Running grid experiment {args.exp}")
    print(f"Challenge sizes: {challenge_sizes}")
    print(f"Model pairs: {len(model_pairs)}")
    
    # Run grid
    for n in challenge_sizes:
        for ref_model, test_model in model_pairs:
            # Placeholder: simulate distances
            np.random.seed(n + hash(test_model) % 1000)
            
            if test_model == "identical":
                distances = np.random.normal(0.001, 0.0005, n)
            elif test_model == "seed_variant":
                distances = np.random.normal(0.01, 0.002, n)
            elif test_model == "fine_tuned":
                distances = np.random.normal(0.05, 0.01, n)
            else:
                distances = np.random.normal(0.1, 0.02, n)
            
            # Compute metrics
            distances_h1 = np.random.normal(0.2, 0.05, n)  # Different model baseline
            
            for tau in config['verification']['tau_grid']:
                far, frr = far_frr(distances, distances_h1, tau)
                
                # Log result
                entry = {
                    "exp": args.exp,
                    "n": n,
                    "ref_model": ref_model,
                    "test_model": test_model,
                    "tau": tau,
                    "far": far,
                    "frr": frr,
                    "auroc": 0.5 + 0.5 * (1 - far - frr),  # Placeholder AUROC
                    "mean_distance": float(np.mean(distances)),
                    "std_distance": float(np.std(distances))
                }
                
                logger.log_jsonl("grid_results.jsonl", entry)
    
    print(f"Grid complete. Results saved to {args.output_dir}/{exp_name}/{args.exp}/")

if __name__ == "__main__":
    main()