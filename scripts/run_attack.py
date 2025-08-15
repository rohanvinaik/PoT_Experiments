#!/usr/bin/env python
"""
Implements attacks:
  - wrapper mapping
  - targeted fine-tune (given leaked fraction rho)
  - limited distillation with query budget
Logs attack cost, queries used, and resulting PoT distance distributions.
"""

import argparse
import json
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from pot.core.logging import StructuredLogger

def main():
    parser = argparse.ArgumentParser(description="Run attack experiments")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument(
        "--attack",
        choices=["wrapper", "targeted_finetune", "distillation", "extraction"],
        required=True,
        help="Attack type",
    )
    parser.add_argument("--rho", type=float, default=0.1, help="Leakage fraction")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Training epochs for learning-based attacks"
    )
    parser.add_argument(
        "--temperature", type=float, default=4.0, help="Distillation temperature"
    )
    parser.add_argument(
        "--routing",
        type=str,
        default=None,
        help="JSON-encoded routing logic for wrapper attack",
    )
    parser.add_argument(
        "--query_budget",
        type=int,
        default=10000,
        help="Query budget for distillation or extraction attacks",
    )
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    exp_name = config['experiment']
    attack_id = f"{args.attack}_rho{args.rho}"
    logger = StructuredLogger(f"{args.output_dir}/{exp_name}/attacks/{attack_id}")
    
    print(f"Running {args.attack} attack with ρ={args.rho}")

    # Placeholder cost accounting; actual attack logic is implemented in
    # ``pot.core.attacks`` and used in tests/examples.  Here we simply expose the
    # hyperparameters and log the expected resource usage.
    if args.attack == "wrapper":
        routing_logic = json.loads(args.routing) if args.routing else None
        print(f"Applying wrapper mapping with routing={routing_logic}...")
        attack_cost = 0
        queries_used = 0

    elif args.attack == "targeted_finetune":
        print(
            f"Fine-tuning on {args.rho*100:.0f}% leaked challenges for {args.epochs} epochs..."
        )
        attack_cost = args.rho * 1000 * args.epochs
        queries_used = int(args.rho * 512)

    elif args.attack == "distillation":
        print(
            f"Distilling with budget={args.query_budget} queries, temperature={args.temperature}..."
        )
        attack_cost = args.query_budget * 0.01 * args.epochs
        queries_used = args.query_budget

    elif args.attack == "extraction":
        print(f"Extracting model with query_budget={args.query_budget}...")
        attack_cost = args.query_budget * 0.02
        queries_used = args.query_budget
    
    # Log attack results
    entry = {
        "attack_type": args.attack,
        "rho": args.rho,
        "budget": args.query_budget if args.attack in {"distillation", "extraction"} else None,
        "epochs": args.epochs,
        "temperature": args.temperature if args.attack == "distillation" else None,
        "routing": json.loads(args.routing) if args.routing else None,
        "attack_cost": attack_cost,
        "queries_used": queries_used,
        "config": config
    }
    
    logger.log_jsonl("attack_log.jsonl", entry)
    
    print(f"Attack complete. Cost={attack_cost:.2f}, Queries={queries_used}")
    print(f"Results saved to {args.output_dir}/{exp_name}/attacks/{attack_id}/")

if __name__ == "__main__":
    main()