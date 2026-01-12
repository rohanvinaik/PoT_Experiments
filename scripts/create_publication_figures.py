#!/usr/bin/env python3
"""
Create publication-quality figures for behavioral fingerprinting results.

Matches the academic style of the original PoT paper figures:
- Clean, professional appearance
- Consistent color scheme
- Clear annotations and legends
- PDF output for publication
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
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Publication style settings
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_results(results_dir: Path) -> dict:
    """Load results from JSON file."""
    results_file = results_dir / "final_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file) as f:
        return json.load(f)


def create_fig3_variance_signature_space(results: dict, output_dir: Path):
    """
    Figure 3: Variance Signature Space

    Scatter plot showing how different model relationships cluster
    in the mean effect vs coefficient of variation space.
    Similar style to Fig 1 (time-to-decision) from the paper.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Define decision regions with soft colors (matching paper style)
    # IDENTICAL region (near origin)
    rect1 = FancyBboxPatch((0, 0), 0.01, 0.3, boxstyle="round,pad=0.01",
                           facecolor='#d5f4e6', alpha=0.5, edgecolor='none')
    ax.add_patch(rect1)

    # DISTILLED region (moderate mean, low CV)
    rect2 = FancyBboxPatch((0.3, 0), 0.5, 0.25, boxstyle="round,pad=0.01",
                           facecolor='#e8d5f4', alpha=0.5, edgecolor='none')
    ax.add_patch(rect2)

    # SCALE region (moderate mean, higher CV)
    rect3 = FancyBboxPatch((0.3, 0.35), 1.2, 1.0, boxstyle="round,pad=0.01",
                           facecolor='#d5e8f4', alpha=0.5, edgecolor='none')
    ax.add_patch(rect3)

    # DIFFERENT_ARCH region (high mean)
    rect4 = FancyBboxPatch((5.0, 0), 5.0, 1.5, boxstyle="round,pad=0.01",
                           facecolor='#f4d5d5', alpha=0.5, edgecolor='none')
    ax.add_patch(rect4)

    # Category markers matching paper style
    category_styles = {
        'self_consistency': {'color': '#2ca02c', 'marker': 'o', 'label': 'Self-consistency'},
        'distillation': {'color': '#9467bd', 'marker': 's', 'label': 'Distillation'},
        'scale': {'color': '#1f77b4', 'marker': '^', 'label': 'Scale variation'},
        'finetune_dialog': {'color': '#ff7f0e', 'marker': 'D', 'label': 'Fine-tuning (dialog)'},
        'finetune_code': {'color': '#d62728', 'marker': 'p', 'label': 'Fine-tuning (code)'},
        'deduplication': {'color': '#17becf', 'marker': 'h', 'label': 'Deduplication'},
        'architecture': {'color': '#e377c2', 'marker': 'v', 'label': 'Different architecture'},
        'cross_scale': {'color': '#bcbd22', 'marker': '<', 'label': 'Cross-architecture'},
    }

    # Plot data points
    plotted_categories = set()
    for r in results.get("results", []):
        cat = r.get("category", "unknown")
        style = category_styles.get(cat, {'color': 'gray', 'marker': 'x', 'label': 'Other'})

        mean = abs(r.get("mean_effect", 0))
        cv = r.get("cv", 0)
        correct = r.get("correct", False)

        # Handle infinite CV (place at 0.02 for visibility)
        if not np.isfinite(cv):
            cv = 0.02

        # Edge indicates correct/incorrect
        edge = '#2ca02c' if correct else '#d62728'
        edge_width = 2 if correct else 2.5

        label = style['label'] if cat not in plotted_categories else None
        ax.scatter(mean, cv, c=style['color'], s=120, marker=style['marker'],
                  edgecolors=edge, linewidths=edge_width, alpha=0.9, zorder=5, label=label)
        plotted_categories.add(cat)

    # Add threshold lines (matching paper style - dashed)
    ax.axvline(x=0.3, color='#9467bd', linestyle='--', linewidth=1.2, alpha=0.7, label='Distillation threshold')
    ax.axvline(x=5.0, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.7, label='Architecture threshold')
    ax.axhline(y=0.35, color='#1f77b4', linestyle=':', linewidth=1.2, alpha=0.7, label='CV threshold (scale)')

    # Region labels
    ax.text(0.005, 0.15, 'IDENTICAL', fontsize=8, color='#27ae60', fontweight='bold', ha='center')
    ax.text(0.55, 0.12, 'DISTILLED', fontsize=8, color='#9467bd', fontweight='bold', ha='center')
    ax.text(0.9, 0.8, 'SCALE\nVARIATION', fontsize=8, color='#1f77b4', fontweight='bold', ha='center')
    ax.text(7.5, 0.75, 'DIFFERENT\nARCHITECTURE', fontsize=8, color='#d62728', fontweight='bold', ha='center')

    ax.set_xlabel('Mean Effect (|Δ Cross-Entropy|)')
    ax.set_ylabel('Coefficient of Variation (CV)')
    ax.set_title('(a) Variance Signature Space for Model Relationship Classification')

    ax.set_xlim(-0.2, 10.5)
    ax.set_ylim(-0.05, 1.5)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Add correct/incorrect markers to legend
    handles.append(plt.scatter([], [], c='white', s=80, edgecolors='#2ca02c', linewidths=2, marker='o'))
    labels.append('Correct')
    handles.append(plt.scatter([], [], c='white', s=80, edgecolors='#d62728', linewidths=2, marker='o'))
    labels.append('Incorrect')

    ax.legend(handles, labels, loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_variance_signature_space.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_variance_signature_space.png', bbox_inches='tight')
    plt.close()


def create_fig4_accuracy_by_category(results: dict, output_dir: Path):
    """
    Figure 4: Classification Accuracy by Relationship Type

    Bar chart showing accuracy for each category, matching paper style.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    categories = results.get("categories", {})
    if not categories:
        return

    # Sort by accuracy descending
    sorted_cats = sorted(categories.items(), key=lambda x: x[1]["correct"]/x[1]["total"], reverse=True)
    cat_names = [c[0] for c in sorted_cats]
    accuracies = [c[1]["correct"] / c[1]["total"] * 100 for c in sorted_cats]
    correct = [c[1]["correct"] for c in sorted_cats]
    total = [c[1]["total"] for c in sorted_cats]

    # Color gradient based on accuracy (matching paper blue scheme)
    colors = []
    for acc in accuracies:
        if acc >= 90:
            colors.append('#1f77b4')  # Dark blue
        elif acc >= 70:
            colors.append('#7fcdbb')  # Medium
        elif acc >= 50:
            colors.append('#c7e9b4')  # Light
        else:
            colors.append('#ffffcc')  # Very light

    x = np.arange(len(cat_names))
    bars = ax.bar(x, accuracies, color=colors, edgecolor='black', linewidth=0.8, width=0.7)

    # Add value labels
    for bar, acc, c, t in zip(bars, accuracies, correct, total):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{c}/{t}', ha='center', va='bottom', fontsize=9)

    # Add overall accuracy line
    overall = results.get("accuracy", 0)
    ax.axhline(y=overall, color='#d62728', linestyle='--', linewidth=1.5, label=f'Overall: {overall:.1f}%')

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n').title() for c in cat_names], fontsize=9)
    ax.set_ylabel('Classification Accuracy (%)')
    ax.set_title('(b) Behavioral Fingerprinting Accuracy by Relationship Type')
    ax.set_ylim(0, 115)

    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_accuracy_by_category.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_accuracy_by_category.png', bbox_inches='tight')
    plt.close()


def create_fig5_confusion_matrix(results: dict, output_dir: Path):
    """
    Figure 5: Multi-class Confusion Matrix

    Extended confusion matrix matching paper style.
    """
    experiment_results = results.get("results", [])

    # Normalize names
    def normalize(name):
        name = name.upper().replace("_", " ").replace("-", " ")
        # Shorten long names
        name = name.replace("SAME ARCHITECTURE DIFFERENT SCALE", "SCALE")
        name = name.replace("DIFFERENT ARCHITECTURE", "DIFF ARCH")
        name = name.replace("EXTENSIVE FINE TUNING", "EXT FT")
        name = name.replace("SAME ARCH DIFF SCALE", "SCALE")
        name = name.replace("DIFFERENT ARCH", "DIFF ARCH")
        name = name.replace("FINE TUNED", "FINE-TUNED")
        return name

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

    fig, ax = plt.subplots(figsize=(6, 5))

    # Use paper's blue colormap
    cmap = LinearSegmentedColormap.from_list('paper_blue', ['#ffffff', '#1f77b4'])
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")

    # Set labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(all_labels, fontsize=9)

    # Add text annotations
    thresh = matrix.max() / 2
    for i in range(n):
        for j in range(n):
            value = int(matrix[i, j])
            if value > 0:
                color = 'white' if matrix[i, j] > thresh else 'black'
                ax.text(j, i, value, ha='center', va='center', color=color, fontsize=11, fontweight='bold')

    ax.set_xlabel('Predicted Classification')
    ax.set_ylabel('True Classification')
    ax.set_title(f'Confusion Matrix (n={len(experiment_results)} pairs)')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_confusion_matrix.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_confusion_matrix.png', bbox_inches='tight')
    plt.close()


def create_fig6_decision_boundaries(results: dict, output_dir: Path):
    """
    Figure 6: Decision Boundary Visualization

    Shows how thresholds separate different relationship classes.
    Similar to the error rates figure from the paper.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    experiment_results = results.get("results", [])

    # Extract data
    means = [abs(r.get("mean_effect", 0)) for r in experiment_results]
    cvs = [r.get("cv", 0) for r in experiment_results]
    cvs = [cv if np.isfinite(cv) else 0.01 for cv in cvs]
    categories = [r.get("category", "unknown") for r in experiment_results]
    correct = [r.get("correct", False) for r in experiment_results]

    # Left panel: Mean effect distribution by category
    cat_means = {}
    for m, cat in zip(means, categories):
        if cat not in cat_means:
            cat_means[cat] = []
        cat_means[cat].append(m)

    cat_order = ['self_consistency', 'deduplication', 'distillation', 'scale', 'finetune_dialog', 'architecture', 'cross_scale']
    cat_order = [c for c in cat_order if c in cat_means]

    positions = range(len(cat_order))
    bp = ax1.boxplot([cat_means[c] for c in cat_order], positions=positions, patch_artist=True)

    colors = ['#2ca02c', '#17becf', '#9467bd', '#1f77b4', '#ff7f0e', '#e377c2', '#bcbd22']
    for patch, color in zip(bp['boxes'], colors[:len(cat_order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_xticks(positions)
    ax1.set_xticklabels([c.replace('_', '\n')[:12] for c in cat_order], fontsize=8)
    ax1.set_ylabel('Mean Effect')
    ax1.set_title('(a) Mean Effect Distribution by Category')

    # Add threshold lines
    ax1.axhline(y=0.3, color='#9467bd', linestyle='--', alpha=0.7, label='Distillation threshold')
    ax1.axhline(y=2.0, color='#ff7f0e', linestyle='--', alpha=0.7, label='Extensive FT threshold')
    ax1.axhline(y=5.0, color='#d62728', linestyle='--', alpha=0.7, label='Diff Arch threshold')
    ax1.legend(fontsize=7, loc='upper right')

    # Right panel: CV distribution by category
    cat_cvs = {}
    for cv, cat in zip(cvs, categories):
        if cat not in cat_cvs:
            cat_cvs[cat] = []
        cat_cvs[cat].append(cv)

    cat_order2 = [c for c in cat_order if c in cat_cvs]
    bp2 = ax2.boxplot([cat_cvs[c] for c in cat_order2], positions=range(len(cat_order2)), patch_artist=True)

    for patch, color in zip(bp2['boxes'], colors[:len(cat_order2)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xticks(range(len(cat_order2)))
    ax2.set_xticklabels([c.replace('_', '\n')[:12] for c in cat_order2], fontsize=8)
    ax2.set_ylabel('Coefficient of Variation')
    ax2.set_title('(b) CV Distribution by Category')

    # Add threshold lines
    ax2.axhline(y=0.25, color='#9467bd', linestyle='--', alpha=0.7, label='Low CV (distillation)')
    ax2.axhline(y=0.35, color='#1f77b4', linestyle='--', alpha=0.7, label='Scale threshold')
    ax2.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_decision_boundaries.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig6_decision_boundaries.png', bbox_inches='tight')
    plt.close()


def create_summary_table(results: dict, output_dir: Path):
    """Create LaTeX-ready summary table."""
    table = r"""
\begin{table}[t]
\centering
\caption{Behavioral Fingerprinting Results Summary}
\label{tab:fingerprinting}
\begin{tabular}{lcccc}
\toprule
\textbf{Category} & \textbf{Experiments} & \textbf{Correct} & \textbf{Accuracy} \\
\midrule
"""
    categories = results.get("categories", {})
    for cat, stats in sorted(categories.items()):
        acc = stats["correct"] / stats["total"] * 100
        cat_name = cat.replace("_", " ").title()
        table += f"{cat_name} & {stats['total']} & {stats['correct']} & {acc:.1f}\\% \\\\\n"

    table += r"""\midrule
\textbf{Overall} & """ + f"{results.get('total_experiments', 0)} & {results.get('correct', 0)} & {results.get('accuracy', 0):.1f}\\% \\\\"
    table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    with open(output_dir / 'table_fingerprinting.tex', 'w') as f:
        f.write(table)


def main():
    parser = argparse.ArgumentParser(description='Create publication-quality figures')
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

        runs = sorted(base_dir.glob("run_*"))
        if not runs:
            print("No runs found")
            return
        results_dir = runs[-1]

    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)

    # Create figures directory
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("Creating publication-quality figures...")

    create_fig3_variance_signature_space(results, fig_dir)
    print("  - fig3_variance_signature_space.pdf/png")

    create_fig4_accuracy_by_category(results, fig_dir)
    print("  - fig4_accuracy_by_category.pdf/png")

    create_fig5_confusion_matrix(results, fig_dir)
    print("  - fig5_confusion_matrix.pdf/png")

    create_fig6_decision_boundaries(results, fig_dir)
    print("  - fig6_decision_boundaries.pdf/png")

    create_summary_table(results, fig_dir)
    print("  - table_fingerprinting.tex")

    print(f"\nFigures saved to: {fig_dir}")
    return fig_dir


if __name__ == "__main__":
    main()
