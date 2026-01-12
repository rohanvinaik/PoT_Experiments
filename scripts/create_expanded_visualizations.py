#!/usr/bin/env python3
"""
Create comprehensive visualizations for expanded behavioral fingerprinting results.

Generates:
1. Classification accuracy by category (bar chart)
2. Mean effect vs CV scatter plot with decision regions
3. Variance signature heatmap
4. Confusion matrix
5. Summary dashboard
6. Decision boundary visualization
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


def load_results(results_dir: Path) -> dict:
    """Load results from JSON file."""
    results_file = results_dir / "final_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file) as f:
        return json.load(f)


def create_accuracy_by_category(results: dict, output_dir: Path):
    """Create bar chart of accuracy by category."""
    categories = results.get("categories", {})
    if not categories:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    cat_names = list(categories.keys())
    accuracies = [categories[c]["correct"] / categories[c]["total"] * 100 for c in cat_names]
    correct = [categories[c]["correct"] for c in cat_names]
    total = [categories[c]["total"] for c in cat_names]

    # Color based on accuracy
    colors = []
    for acc in accuracies:
        if acc >= 90:
            colors.append('#2ecc71')  # Green
        elif acc >= 70:
            colors.append('#f39c12')  # Orange
        elif acc >= 50:
            colors.append('#e74c3c')  # Red
        else:
            colors.append('#95a5a6')  # Gray

    bars = ax.bar(range(len(cat_names)), accuracies, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, acc, c, t) in enumerate(zip(bars, accuracies, correct, total)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc:.0f}%\n({c}/{t})', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(cat_names)))
    ax.set_xticklabels([c.replace('_', '\n').title() for c in cat_names], fontsize=10)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax.set_title('Behavioral Fingerprinting Accuracy by Category', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 120)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='Excellent (>90%)'),
        mpatches.Patch(facecolor='#f39c12', edgecolor='black', label='Good (70-90%)'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Fair (50-70%)'),
        mpatches.Patch(facecolor='#95a5a6', edgecolor='black', label='Poor (<50%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Add overall accuracy
    overall = results.get("accuracy", 0)
    ax.axhline(y=overall, color='#3498db', linestyle='--', linewidth=2, label=f'Overall: {overall:.1f}%')
    ax.text(len(cat_names) - 0.5, overall + 2, f'Overall: {overall:.1f}%',
            color='#3498db', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_category.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_mean_cv_scatter(results: dict, output_dir: Path):
    """Create scatter plot of mean effect vs coefficient of variation with decision regions."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw decision regions first (as background)
    ax.axhspan(0, 0.25, xmin=0, xmax=0.5/10, alpha=0.2, color='#9b59b6', label='_nolegend_')
    ax.axhspan(0.35, 2.0, xmin=0.3/10, xmax=1.5/10, alpha=0.15, color='#3498db', label='_nolegend_')
    ax.axhspan(0.25, 0.6, xmin=0.1/10, xmax=2.0/10, alpha=0.15, color='#f39c12', label='_nolegend_')
    ax.axvspan(5.0, 10.0, alpha=0.15, color='#e74c3c', label='_nolegend_')
    ax.axvspan(2.0, 5.0, alpha=0.15, color='#e67e22', label='_nolegend_')

    # Add region labels
    ax.text(0.4, 0.12, 'DISTILLED', fontsize=9, color='#9b59b6', fontweight='bold', alpha=0.8)
    ax.text(0.9, 1.0, 'SCALE', fontsize=9, color='#3498db', fontweight='bold', alpha=0.8)
    ax.text(0.5, 0.4, 'FINE-TUNED', fontsize=9, color='#f39c12', fontweight='bold', alpha=0.8)
    ax.text(7.0, 0.3, 'DIFFERENT\nARCH', fontsize=9, color='#e74c3c', fontweight='bold', alpha=0.8, ha='center')
    ax.text(3.5, 0.3, 'EXTENSIVE\nFINE-TUNE', fontsize=9, color='#e67e22', fontweight='bold', alpha=0.8, ha='center')

    # Category colors and markers
    category_styles = {
        'self_consistency': {'color': '#27ae60', 'marker': 'o', 'size': 200},
        'distillation': {'color': '#9b59b6', 'marker': 's', 'size': 200},
        'scale': {'color': '#3498db', 'marker': '^', 'size': 200},
        'finetune_dialog': {'color': '#f39c12', 'marker': 'D', 'size': 200},
        'finetune_code': {'color': '#e67e22', 'marker': 'p', 'size': 200},
        'deduplication': {'color': '#1abc9c', 'marker': 'h', 'size': 200},
        'architecture': {'color': '#e74c3c', 'marker': 'v', 'size': 200},
        'cross_scale': {'color': '#c0392b', 'marker': '<', 'size': 200},
    }

    # Plot each result
    for r in results.get("results", []):
        cat = r.get("category", "unknown")
        style = category_styles.get(cat, {'color': 'gray', 'marker': 'x', 'size': 150})

        mean = abs(r.get("mean_effect", 0))
        cv = r.get("cv", 0)
        correct = r.get("correct", False)

        # Handle infinite CV
        if not np.isfinite(cv):
            cv = 0.01  # Place at bottom for identical models

        # Edge color indicates correct/incorrect
        edge_color = '#27ae60' if correct else '#e74c3c'
        edge_width = 3 if not correct else 2

        ax.scatter(mean, cv, c=style['color'], s=style['size'], marker=style['marker'],
                  edgecolors=edge_color, linewidths=edge_width, alpha=0.9, zorder=5)

        # Add label for each point
        ax.annotate(r.get("name", "")[:15], (mean, cv),
                   textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8)

    # Create legend for categories
    legend_elements = []
    for cat, style in category_styles.items():
        if any(r.get("category") == cat for r in results.get("results", [])):
            legend_elements.append(plt.scatter([], [], c=style['color'], s=100,
                                              marker=style['marker'], label=cat.replace('_', ' ').title()))

    # Add legend for correct/incorrect
    legend_elements.append(plt.scatter([], [], c='white', s=100, edgecolors='#27ae60',
                                       linewidths=3, label='Correct'))
    legend_elements.append(plt.scatter([], [], c='white', s=100, edgecolors='#e74c3c',
                                       linewidths=3, label='Incorrect'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.set_xlabel('Mean Effect (|Δ Cross-Entropy|)', fontsize=12)
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
    ax.set_title('Behavioral Fingerprinting: Mean Effect vs CV\nwith Decision Regions', fontsize=14, fontweight='bold')

    ax.set_xlim(-0.1, 10.5)
    ax.set_ylim(-0.05, 1.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'mean_vs_cv_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_variance_profiles(results: dict, output_dir: Path):
    """Create bar chart of variance signatures."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    experiment_results = results.get("results", [])
    names = [r.get("name", "")[:15] for r in experiment_results]
    means = [abs(r.get("mean_effect", 0)) for r in experiment_results]
    cvs = [r.get("cv", 0) if np.isfinite(r.get("cv", 0)) else 0 for r in experiment_results]
    correct = [r.get("correct", False) for r in experiment_results]

    colors = ['#27ae60' if c else '#e74c3c' for c in correct]

    # Mean effect chart
    bars1 = ax1.barh(range(len(names)), means, color=colors, edgecolor='black')
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('Mean Effect', fontsize=11)
    ax1.set_title('Mean Effect by Experiment', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()

    # Add threshold lines
    ax1.axvline(x=0.3, color='#9b59b6', linestyle='--', alpha=0.7, label='Distillation threshold')
    ax1.axvline(x=2.0, color='#e67e22', linestyle='--', alpha=0.7, label='Extensive fine-tune')
    ax1.axvline(x=5.0, color='#e74c3c', linestyle='--', alpha=0.7, label='Different arch')
    ax1.legend(fontsize=8, loc='lower right')

    # CV chart
    bars2 = ax2.barh(range(len(names)), cvs, color=colors, edgecolor='black')
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('Coefficient of Variation', fontsize=11)
    ax2.set_title('CV by Experiment', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()

    # Add threshold lines
    ax2.axvline(x=0.25, color='#9b59b6', linestyle='--', alpha=0.7, label='Low CV (distillation)')
    ax2.axvline(x=0.35, color='#3498db', linestyle='--', alpha=0.7, label='Scale threshold')
    ax2.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'variance_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_confusion_matrix(results: dict, output_dir: Path):
    """Create confusion matrix of expected vs actual classifications."""
    experiment_results = results.get("results", [])

    # Normalize relationship names
    def normalize(name):
        return name.upper().replace("_", " ").replace("-", " ")

    expected_labels = list(set(normalize(r.get("expected", "UNKNOWN")) for r in experiment_results))
    actual_labels = list(set(normalize(r.get("actual", "UNKNOWN")) for r in experiment_results))
    all_labels = sorted(list(set(expected_labels + actual_labels)))

    # Build confusion matrix
    n = len(all_labels)
    matrix = np.zeros((n, n))

    for r in experiment_results:
        exp = normalize(r.get("expected", "UNKNOWN"))
        act = normalize(r.get("actual", "UNKNOWN"))
        i = all_labels.index(exp)
        j = all_labels.index(act)
        matrix[i, j] += 1

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=11)

    # Set labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([l[:20] for l in all_labels], rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([l[:20] for l in all_labels], fontsize=9)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            value = int(matrix[i, j])
            if value > 0:
                color = 'white' if matrix[i, j] > matrix.max() / 2 else 'black'
                ax.text(j, i, value, ha='center', va='center', color=color, fontsize=12, fontweight='bold')

    ax.set_xlabel('Actual Classification', fontsize=12)
    ax.set_ylabel('Expected Classification', fontsize=12)
    ax.set_title('Confusion Matrix: Expected vs Actual', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_dashboard(results: dict, output_dir: Path):
    """Create summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Overall accuracy (top center)
    ax1 = fig.add_subplot(gs[0, 1])
    accuracy = results.get("accuracy", 0)
    colors_pie = ['#27ae60', '#e74c3c']
    ax1.pie([accuracy, 100-accuracy], colors=colors_pie, startangle=90,
            autopct='%1.1f%%', textprops={'fontsize': 14, 'fontweight': 'bold'})
    ax1.set_title(f'Overall Accuracy\n{results.get("correct", 0)}/{results.get("total_experiments", 0)} Correct',
                  fontsize=14, fontweight='bold')

    # Category breakdown (top left)
    ax2 = fig.add_subplot(gs[0, 0])
    categories = results.get("categories", {})
    cat_names = list(categories.keys())[:6]  # Limit to 6
    cat_accs = [categories[c]["correct"] / categories[c]["total"] * 100 for c in cat_names]
    colors_bar = ['#27ae60' if a >= 70 else '#f39c12' if a >= 50 else '#e74c3c' for a in cat_accs]
    ax2.bar(range(len(cat_names)), cat_accs, color=colors_bar, edgecolor='black')
    ax2.set_xticks(range(len(cat_names)))
    ax2.set_xticklabels([c.replace('_', '\n')[:10] for c in cat_names], fontsize=8)
    ax2.set_ylabel('Accuracy %')
    ax2.set_title('By Category', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 110)

    # Key metrics (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    metrics_text = f"""
    KEY METRICS

    Total Experiments: {results.get('total_experiments', 0)}
    Correct: {results.get('correct', 0)}
    Accuracy: {accuracy:.1f}%

    Categories Tested: {len(categories)}

    Perfect Categories:
    {', '.join([c for c, v in categories.items() if v['correct'] == v['total']][:3])}
    """
    ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Mean effect distribution (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    means = [abs(r.get("mean_effect", 0)) for r in results.get("results", [])]
    ax4.hist(means, bins=10, color='#3498db', edgecolor='black', alpha=0.7)
    ax4.axvline(x=np.mean(means), color='red', linestyle='--', label=f'Mean: {np.mean(means):.2f}')
    ax4.set_xlabel('Mean Effect')
    ax4.set_ylabel('Count')
    ax4.set_title('Mean Effect Distribution', fontsize=12, fontweight='bold')
    ax4.legend()

    # CV distribution (middle center)
    ax5 = fig.add_subplot(gs[1, 1])
    cvs = [r.get("cv", 0) for r in results.get("results", []) if np.isfinite(r.get("cv", 0))]
    if cvs:
        ax5.hist(cvs, bins=10, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax5.axvline(x=np.mean(cvs), color='red', linestyle='--', label=f'Mean: {np.mean(cvs):.2f}')
    ax5.set_xlabel('Coefficient of Variation')
    ax5.set_ylabel('Count')
    ax5.set_title('CV Distribution', fontsize=12, fontweight='bold')
    ax5.legend()

    # Query count (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    queries = [r.get("n_queries", 0) for r in results.get("results", [])]
    ax6.hist(queries, bins=10, color='#e67e22', edgecolor='black', alpha=0.7)
    ax6.axvline(x=np.mean(queries), color='red', linestyle='--', label=f'Mean: {np.mean(queries):.0f}')
    ax6.set_xlabel('Queries Used')
    ax6.set_ylabel('Count')
    ax6.set_title('Query Distribution', fontsize=12, fontweight='bold')
    ax6.legend()

    # Results table (bottom spanning all columns)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')

    # Create table data
    table_data = []
    for r in results.get("results", [])[:10]:  # Limit to 10 rows
        status = "✓" if r.get("correct") else "✗"
        cv_str = f"{r.get('cv', 0):.2f}" if np.isfinite(r.get('cv', 0)) else "∞"
        table_data.append([
            r.get("name", "")[:18],
            r.get("category", "")[:12],
            r.get("expected", "")[:15],
            r.get("actual", "")[:15],
            status,
            f"{abs(r.get('mean_effect', 0)):.3f}",
            cv_str,
            f"{r.get('confidence', 0):.0f}%"
        ])

    if table_data:
        table = ax7.table(cellText=table_data,
                         colLabels=['Experiment', 'Category', 'Expected', 'Actual', 'Match', 'Mean', 'CV', 'Conf'],
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Color cells based on correctness
        for i, row in enumerate(table_data):
            color = '#d5f4e6' if row[4] == "✓" else '#f4d5d5'
            for j in range(8):
                table[(i+1, j)].set_facecolor(color)

    plt.suptitle('Expanded Behavioral Fingerprinting Summary Dashboard',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_dir / 'summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create visualizations for expanded fingerprinting results')
    parser.add_argument('--results-dir', type=str, help='Path to results directory')
    args = parser.parse_args()

    # Find latest results directory if not specified
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        base_dir = Path("experimental_results/expanded_fingerprinting")
        if not base_dir.exists():
            print("No expanded fingerprinting results found")
            return

        # Get latest run
        runs = sorted(base_dir.glob("run_*"))
        if not runs:
            print("No runs found")
            return
        results_dir = runs[-1]

    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)

    # Create visualizations directory
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    print("Creating visualizations...")

    create_accuracy_by_category(results, viz_dir)
    print("  - accuracy_by_category.png")

    create_mean_cv_scatter(results, viz_dir)
    print("  - mean_vs_cv_scatter.png")

    create_variance_profiles(results, viz_dir)
    print("  - variance_profiles.png")

    create_confusion_matrix(results, viz_dir)
    print("  - confusion_matrix.png")

    create_summary_dashboard(results, viz_dir)
    print("  - summary_dashboard.png")

    print(f"\nVisualizations saved to: {viz_dir}")
    return viz_dir


if __name__ == "__main__":
    main()
