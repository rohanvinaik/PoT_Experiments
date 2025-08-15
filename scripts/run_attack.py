#!/usr/bin/env python
"""
Implements attacks:
  - wrapper mapping
  - targeted fine-tune (given leaked fraction rho)
  - limited distillation with query budget
Logs attack cost, queries used, and resulting PoT distance distributions.
"""

import argparse
import time
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from pot.core.logging import StructuredLogger

# Cost model constants
STEP_TIME = 0.05   # seconds per training step (simulated)
STEP_COST = 0.1    # cost units per training step
QUERY_COST = 0.01  # cost units per query
TIME_COST = 0.001  # cost units per second


def run_attack(config_path: str, attack: str, rho: float = 0.1,
               budget: int = 10000, output_dir: str = "outputs"):
    """Execute an attack simulation and log resource usage."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_name = config["experiment"]
    attack_id = f"{attack}_rho{rho}"
    if attack == "distillation":
        attack_id += f"_budget{budget}"
    logger = StructuredLogger(f"{output_dir}/{exp_name}/attacks/{attack_id}")

    print(f"Running {attack} attack with ρ={rho}")

    total_n = sum(fam.get("n", 0) for fam in config.get("challenges", {}).get("families", []))

    start = time.time()
    if attack == "wrapper":
        print("Applying wrapper mapping...")
        training_steps = 0
        queries_used = 0
    elif attack == "targeted_finetune":
        print(f"Fine-tuning on {rho*100:.0f}% leaked challenges...")
        queries_used = int(rho * total_n)
        training_steps = queries_used * 2  # assume 2 steps per leaked challenge
    elif attack == "distillation":
        print(f"Distilling with budget={budget} queries...")
        queries_used = budget
        training_steps = budget // 10  # assume 10 queries per step
    else:
        raise ValueError(f"Unknown attack type: {attack}")

    # Simulated time spent on attack
    time_sec = training_steps * STEP_TIME + (time.time() - start)
    attack_cost = (
        training_steps * STEP_COST
        + time_sec * TIME_COST
        + queries_used * QUERY_COST
    )

    entry = {
        "attack_type": attack,
        "rho": rho,
        "budget": budget if attack == "distillation" else None,
        "metrics": {
            "training_steps": training_steps,
            "time_sec": time_sec,
            "queries_used": queries_used,
            "attack_cost": attack_cost,
        },
        "config": config,
    }

    logger.log_jsonl("attack_log.jsonl", entry)

    print(
        f"Attack complete. Cost={attack_cost:.2f}, "
        f"Steps={training_steps}, Time={time_sec:.2f}s, Queries={queries_used}"
    )
    print(f"Results saved to {output_dir}/{exp_name}/attacks/{attack_id}/")
    return entry


def main():
    parser = argparse.ArgumentParser(description="Run attack experiments")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument(
        "--attack", choices=["wrapper", "targeted_finetune", "distillation"],
        required=True, help="Attack type",
    )
    parser.add_argument("--rho", type=float, default=0.1, help="Leakage fraction")
    parser.add_argument(
        "--budget", type=int, default=10000, help="Query budget for distillation"
    )
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    run_attack(
        args.config,
        args.attack,
        rho=args.rho,
        budget=args.budget,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()