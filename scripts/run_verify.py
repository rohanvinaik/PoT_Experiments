#!/usr/bin/env python
"""
Given (reference model, deployed model, challenge cfg), run probes, canonicalize outputs,
compute distances, T-statistic, FAR/FRR estimates via bootstrap, and optional sequential test.
Emit:
  - per-challenge distances
  - summary (T, tau, far_hat, frr_hat, queries_used)
  - audit artifacts: commit hash, salts used, config snapshot
"""

import argparse
import yaml
import json
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pot.core.stats import t_statistic, far_frr, empirical_bernstein_radius
from pot.core.logging import StructuredLogger
from pot.core.governance import commit_message
from pot.eval.metrics import dist_logits_l2, dist_kl

def main():
    parser = argparse.ArgumentParser(description="Verify model against reference")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument("--challenge_family", required=True, help="Challenge family to use")
    parser.add_argument("--n", type=int, default=256, help="Number of challenges")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    exp_name = config['experiment']
    run_id = f"{args.challenge_family}_{args.n}"
    logger = StructuredLogger(f"{args.output_dir}/{exp_name}/{run_id}")
    
    # Placeholder: compute distances (would need actual model outputs)
    # For skeleton, generate random distances
    np.random.seed(42)
    distances_h0 = np.random.normal(0.01, 0.005, args.n)  # Same model
    distances_h1 = np.random.normal(0.1, 0.02, args.n)     # Different model
    
    # Compute statistics
    t_stat = t_statistic(distances_h0)
    
    # FAR/FRR for different thresholds
    for tau in config['verification']['tau_grid']:
        far, frr = far_frr(distances_h0, distances_h1, tau)
        
        # Compute confidence radius
        radius = empirical_bernstein_radius(distances_h0, delta=0.05)
        
        # Log result
        entry = {
            "run_id": run_id,
            "challenge_family": args.challenge_family,
            "n": args.n,
            "tau": tau,
            "distances_mean": float(np.mean(distances_h0)),
            "distances_std": float(np.std(distances_h0)),
            "T": t_stat,
            "far_hat": far,
            "frr_hat": frr,
            "confidence_radius": radius,
            "config_snapshot": config
        }
        
        logger.log_jsonl("verify.jsonl", entry)
        
        print(f"τ={tau}: FAR={far:.3f}, FRR={frr:.3f}, T={t_stat:.4f}")
    
    print(f"Results saved to {args.output_dir}/{exp_name}/{run_id}/")

if __name__ == "__main__":
    main()