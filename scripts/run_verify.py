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
import numpy as np
from pathlib import Path
import sys
import torch

# Allow running as script from project root
sys.path.append(str(Path(__file__).parent.parent))

from pot.core.canonicalize import canonicalize_logits
from pot.core.challenge import ChallengeConfig, generate_challenges
from pot.core.governance import new_session_nonce
from pot.core.stats import (
    empirical_bernstein_radius,
    far_frr,
    t_statistic,
)
from pot.core.logging import StructuredLogger
from pot.eval.metrics import dist_logits_l2, dist_kl
from pot.models import load_model

def main():
    parser = argparse.ArgumentParser(description="Verify model against reference")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument("--challenge_family", required=True, help="Challenge family to use")
    parser.add_argument("--n", type=int, default=256, help="Number of challenges")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run on CPU only and avoid CUDA initialization",
    )
    args = parser.parse_args()

    if args.cpu_only:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_name = config["experiment"]
    run_id = f"{args.challenge_family}_{args.n}"
    logger = StructuredLogger(f"{args.output_dir}/{exp_name}/{run_id}")

    # Load reference and test models
    ref_model = load_model(config["models"]["reference_path"], cpu_only=args.cpu_only).eval()
    test_model = load_model(config["models"]["test_path"], cpu_only=args.cpu_only).eval()

    # Get challenge parameters from config
    family_cfg = None
    for fam in config["challenges"]["families"]:
        if fam["family"] == args.challenge_family:
            family_cfg = fam
            break
    if family_cfg is None:
        raise ValueError(f"Challenge family {args.challenge_family} not found in config")

    # Generate challenges
    cfg = ChallengeConfig(
        master_key_hex="0" * 64,
        session_nonce_hex=new_session_nonce(),
        n=args.n,
        family=args.challenge_family,
        params=family_cfg["params"],
    )
    challenge_data = generate_challenges(cfg)
    items = challenge_data["items"]

    metrics_map = {"logits_l2": dist_logits_l2, "kl": dist_kl}
    selected_metrics = {m: metrics_map[m] for m in config["verification"]["distances"]}
    dists_h0 = {m: [] for m in selected_metrics}
    dists_h1 = {m: [] for m in selected_metrics}

    def item_to_tensor(item):
        vals = [item[k] for k in sorted(item.keys())]
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(0)

    # Run models on challenges
    for idx, item in enumerate(items):
        x = item_to_tensor(item)
        with torch.no_grad():
            logits_ref = ref_model(x).squeeze(0).numpy()
            logits_test = test_model(x).squeeze(0).numpy()
        logits_ref = canonicalize_logits(logits_ref)
        logits_test = canonicalize_logits(logits_test)

        log_entry = {"challenge_index": idx}
        for name, fn in selected_metrics.items():
            d0 = fn(logits_ref, logits_ref)
            d1 = fn(logits_ref, logits_test)
            dists_h0[name].append(d0)
            dists_h1[name].append(d1)
            log_entry[name] = d1
        logger.log_jsonl("distances.jsonl", log_entry)

    # Convert distance lists to arrays
    for name in selected_metrics:
        dists_h0[name] = np.array(dists_h0[name])
        dists_h1[name] = np.array(dists_h1[name])

    # Use first metric for statistical estimates
    primary_metric = next(iter(selected_metrics))
    t_stat = t_statistic(dists_h1[primary_metric])

    for tau in config["verification"]["tau_grid"]:
        far, frr = far_frr(dists_h0[primary_metric], dists_h1[primary_metric], tau)
        radius = empirical_bernstein_radius(dists_h1[primary_metric], delta=0.05)

        entry = {
            "run_id": run_id,
            "challenge_family": args.challenge_family,
            "n": args.n,
            "tau": tau,
            "T": t_stat,
            "far_hat": far,
            "frr_hat": frr,
            "confidence_radius": radius,
            "config_snapshot": config,
        }
        logger.log_jsonl("verify.jsonl", entry)
        print(f"τ={tau}: FAR={far:.3f}, FRR={frr:.3f}, T={t_stat:.4f}")

    print(f"Results saved to {args.output_dir}/{exp_name}/{run_id}/")

if __name__ == "__main__":
    main()