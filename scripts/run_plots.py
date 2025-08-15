#!/usr/bin/env python
"""
Creates:
  - ROC/DET curves (FAR vs FRR) as n varies
  - AUROC vs query budget
  - Leakage ρ vs detection rate
  - Non-IID drift sensitivity plots
  - Early stopping: average queries to decision under SPRT/e-values
"""

import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

def load_jsonl(filepath):
    """Load JSONL file"""
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate plots from experiment results")
    parser.add_argument("--exp_dir", required=True, help="Experiment directory")
    parser.add_argument(
        "--plot_type",
        choices=["roc", "det", "auroc", "leakage", "drift", "sequential"],
        default="roc",
        help="Plot type to generate",
    )
    parser.add_argument(
        "--input_files",
        nargs="*",
        help="Optional JSONL files to load (defaults depend on plot type)",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)

    print(f"Generating {args.plot_type} plots from {exp_dir}")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    if args.plot_type == "roc":
        # Load grid results
        grid_file = exp_dir / "grid_results.jsonl"
        if grid_file.exists():
            data = load_jsonl(grid_file)
            
            # Group by n
            n_values = sorted(set(d['n'] for d in data))
            
            plt.figure()
            for n in n_values:
                n_data = [d for d in data if d['n'] == n]
                far_values = [d['far'] for d in n_data]
                frr_values = [d['frr'] for d in n_data]
                
                # Sort by FAR for proper curve
                sorted_pairs = sorted(zip(far_values, frr_values))
                far_sorted = [p[0] for p in sorted_pairs]
                frr_sorted = [p[1] for p in sorted_pairs]
                
                plt.plot(far_sorted, 1 - np.array(frr_sorted), label=f'n={n}', linewidth=2)
            
            plt.xlabel('False Accept Rate (FAR)')
            plt.ylabel('True Accept Rate (1-FRR)')
            plt.title('ROC Curves for Different Challenge Sizes')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            
            output_file = exp_dir / "roc_curve.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"ROC curve saved to {output_file}")
            
    elif args.plot_type == "auroc":
        # AUROC vs query budget
        grid_file = exp_dir / "grid_results.jsonl"
        if grid_file.exists():
            data = load_jsonl(grid_file)
            
            n_values = sorted(set(d['n'] for d in data))
            auroc_values = []
            
            for n in n_values:
                n_data = [d for d in data if d['n'] == n]
                auroc = np.mean([d.get('auroc', 0.5) for d in n_data])
                auroc_values.append(auroc)
            
            plt.figure()
            plt.plot(n_values, auroc_values, 'o-', linewidth=2, markersize=8)
            plt.xlabel('Query Budget (n)')
            plt.ylabel('AUROC')
            plt.title('AUROC vs Query Budget')
            plt.grid(True, alpha=0.3)
            plt.xscale('log')
            
            output_file = exp_dir / "auroc_vs_queries.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"AUROC plot saved to {output_file}")
            
    elif args.plot_type == "leakage":
        # Load leakage experiment results
        files = [Path(f) for f in args.input_files] if args.input_files else [exp_dir / "leakage_log.jsonl"]

        rho_to_rates = {}
        for file in files:
            if file.exists():
                data = load_jsonl(file)
                for entry in data:
                    rho = entry.get("rho")
                    rate = entry.get("detection_rate")
                    if rho is None or rate is None:
                        continue
                    rho_to_rates.setdefault(rho, []).append(rate)
            else:
                print(f"Warning: {file} not found")

        if rho_to_rates:
            rho_values = sorted(rho_to_rates.keys())
            detection_rates = [np.mean(rates) for rates in (rho_to_rates[r] for r in rho_values)]

            plt.figure()
            plt.plot(rho_values, detection_rates, 'r^-', linewidth=2, markersize=10)
            plt.xlabel('Leakage Fraction (ρ)')
            plt.ylabel('Detection Rate')
            plt.title('Detection Rate vs Challenge Leakage')
            plt.grid(True, alpha=0.3)
            plt.xlim([0, max(rho_values)])
            plt.ylim([0, 1])

            output_file = exp_dir / "leakage_curve.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Leakage curve saved to {output_file}")
        else:
            print("No leakage data found; skipping plot")
    
    plt.show()

if __name__ == "__main__":
    main()