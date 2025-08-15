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
import importlib
import yaml
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pot.core.logging import StructuredLogger
from pot.core.stats import far_frr
from pot.core.challenge import ChallengeConfig, generate_challenges

def main():
    parser = argparse.ArgumentParser(description="Run grid experiment")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument("--exp", default="E1", help="Experiment name")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument(
        "--n_values",
        default="32,64,128,256,512,1024",
        help="Comma separated list of challenge sizes to evaluate",
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    exp_name = config['experiment']
    logger = StructuredLogger(f"{args.output_dir}/{exp_name}/{args.exp}")
    
    # Helper to load model definitions of form "module:Class"
    def load_model(path: str):
        module_name, attr = path.split(":")
        module = importlib.import_module(module_name)
        return getattr(module, attr)()

    # Instantiate models
    models_cfg = config.get("models", {})
    reference_model = load_model(models_cfg["reference"])
    identical_model = load_model(models_cfg.get("identical", models_cfg["reference"]))

    test_models = []
    for name, path in models_cfg.items():
        if name in {"reference", "identical"}:
            continue
        test_models.append((name, load_model(path)))

    # Challenge configuration
    chal_cfg = config.get("challenges", {})
    if isinstance(chal_cfg.get("families"), list):
        family_cfg = chal_cfg["families"][0]
        family = family_cfg["family"]
        params = family_cfg.get("params", {})
    else:
        family = chal_cfg.get("family", "vision:freq")
        params = chal_cfg.get("params", {})
    master_key = chal_cfg.get("master_key", "0" * 64)
    session_nonce = chal_cfg.get("session_nonce", "0" * 32)

    # Grid parameters
    challenge_sizes = [int(x) for x in args.n_values.split(",") if x]

    print(f"Running grid experiment {args.exp}")
    print(f"Challenge sizes: {challenge_sizes}")
    print(f"Model pairs: {len(test_models)}")

    # Run grid
    for n in challenge_sizes:
        # Generate challenges
        cfg = ChallengeConfig(
            master_key_hex=master_key,
            session_nonce_hex=session_nonce,
            n=n,
            family=family,
            params=params,
        )
        challenges = generate_challenges(cfg)["items"]

        # Compute baseline distances for identical model (H0)
        ref_outputs = [reference_model(ch) for ch in challenges]
        ident_outputs = [identical_model(ch) for ch in challenges]
        distances_h0 = np.array([
            np.linalg.norm(np.array(r) - np.array(i))
            for r, i in zip(ref_outputs, ident_outputs)
        ])

        for name, test_model in test_models:
            test_outputs = [test_model(ch) for ch in challenges]
            distances = np.array([
                np.linalg.norm(np.array(r) - np.array(t))
                for r, t in zip(ref_outputs, test_outputs)
            ])

            far_values = []
            frr_values = []
            for tau in config["verification"]["tau_grid"]:
                far, frr = far_frr(distances_h0, distances, tau)
                far_values.append(far)
                frr_values.append(frr)

            # Compute AUROC from FAR/FRR curves
            far_arr = np.array(far_values)
            tpr_arr = 1 - np.array(frr_values)
            order = np.argsort(far_arr)
            auroc = float(np.trapz(tpr_arr[order], far_arr[order]))

            for tau, far, frr in zip(config["verification"]["tau_grid"], far_values, frr_values):
                entry = {
                    "exp": args.exp,
                    "n": n,
                    "ref_model": "reference",
                    "test_model": name,
                    "tau": tau,
                    "far": far,
                    "frr": frr,
                    "auroc": auroc,
                    "mean_distance": float(np.mean(distances)),
                    "std_distance": float(np.std(distances)),
                }

                logger.log_jsonl("grid_results.jsonl", entry)
    
    print(f"Grid complete. Results saved to {args.output_dir}/{exp_name}/{args.exp}/")

if __name__ == "__main__":
    main()
