#!/usr/bin/env python3
"""
Create visualizations for behavioral fingerprinting results.

Generates:
1. Classification accuracy by category (bar chart)
2. Mean effect vs CV scatter plot (shows relationship clustering)
3. Decision boundary visualization
4. Variance signature comparison
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Find the most recent results
results_dir = Path("experimental_results/behavioral_fingerprinting")
run_dirs = sorted(results_dir.glob("run_*"), reverse=True)

if not run_dirs:
    print("No fingerprinting results found!")
    sys.exit(1)

latest_run = run_dirs[0]
print(f"Using results from: {latest_run}")

# Load results
with open(latest_run / "intermediate_results.json") as f:
    results = json.load(f)

# Output directory for visualizations
viz_dir = latest_run / "visualizations"
viz_dir.mkdir(exist_ok=True)

# Color scheme for relationships
RELATIONSHIP_COLORS = {
    "IDENTICAL": "#2ecc71",       # Green
    "NEAR_CLONE": "#27ae60",       # Darker green
    "DISTILLED": "#3498db",        # Blue
    "SAME_ARCHITECTURE_DIFFERENT_SCALE": "#9b59b6",  # Purple
    "SAME_ARCH_DIFF_SCALE": "#9b59b6",               # Purple (alias)
    "DIFFERENT_ARCHITECTURE": "#e74c3c",              # Red
    "DIFFERENT_ARCH": "#e74c3c",                      # Red (alias)
    "SAME_ARCHITECTURE_FINE_TUNED": "#f39c12",       # Orange
    "INCONCLUSIVE": "#95a5a6",     # Gray
}

CATEGORY_COLORS = {
    "self_consistency": "#2ecc71",
    "distillation": "#3498db",
    "scale": "#9b59b6",
    "architecture": "#e74c3c",
}

# ============================================================================
# 1. Accuracy by Category (Bar Chart)
# ============================================================================

def plot_accuracy_by_category():
    """Create bar chart of accuracy by category"""

    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r["relationship_match"]:
            categories[cat]["correct"] += 1

    cat_names = list(categories.keys())
    accuracies = [categories[c]["correct"] / categories[c]["total"] * 100 for c in cat_names]
    totals = [categories[c]["total"] for c in cat_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        range(len(cat_names)),
        accuracies,
        color=[CATEGORY_COLORS.get(c, "#95a5a6") for c in cat_names],
        edgecolor="white",
        linewidth=2
    )

    # Add value labels on bars
    for bar, acc, total in zip(bars, accuracies, totals):
        height = bar.get_height()
        ax.annotate(
            f'{acc:.0f}%\n({int(acc/100*total)}/{total})',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )

    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
    ax.set_title("Behavioral Fingerprinting: Accuracy by Category", fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(cat_names)))
    ax.set_xticklabels([c.replace("_", " ").title() for c in cat_names], fontsize=11)
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)

    # Overall accuracy annotation
    overall_correct = sum(categories[c]["correct"] for c in cat_names)
    overall_total = sum(categories[c]["total"] for c in cat_names)
    overall_acc = overall_correct / overall_total * 100

    ax.text(
        0.95, 0.95,
        f"Overall: {overall_acc:.1f}%\n({overall_correct}/{overall_total})",
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(viz_dir / "accuracy_by_category.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: accuracy_by_category.png")

# ============================================================================
# 2. Mean Effect vs CV Scatter Plot
# ============================================================================

def plot_mean_vs_cv():
    """Create scatter plot showing mean effect vs CV with relationship clustering"""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each result
    for r in results:
        mean = r["mean_effect"]
        cv = r["cv"] if r["cv"] != float('inf') else 0

        actual = r["actual_relationship"]
        expected = r["expected_relationship"]
        correct = r["relationship_match"]

        color = RELATIONSHIP_COLORS.get(actual, "#95a5a6")
        marker = 'o' if correct else 'x'
        size = 200 if correct else 150

        ax.scatter(
            mean, cv,
            c=color,
            marker=marker,
            s=size,
            edgecolors='black' if correct else 'red',
            linewidths=2,
            alpha=0.8,
            label=actual if actual not in [r["actual_relationship"] for r in results[:results.index(r)]] else ""
        )

        # Add experiment label
        ax.annotate(
            r["experiment_id"].split("_")[-1][:6],
            (mean, cv),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )

    # Draw decision boundaries (approximate)
    ax.axvline(x=3.0, color='red', linestyle='--', alpha=0.5, label='Arch threshold (mean=3.0)')
    ax.axhline(y=0.3, color='blue', linestyle='--', alpha=0.5, label='Distilled threshold (CV=0.3)')
    ax.axhline(y=1.5, color='purple', linestyle='--', alpha=0.5, label='Scale threshold (CV=1.5)')

    # Shade regions
    ax.fill_between([0, 0.001], 0, 3, alpha=0.1, color='green', label='IDENTICAL region')
    ax.fill_between([0.3, 3], 0, 0.3, alpha=0.1, color='blue', label='DISTILLED region')
    ax.fill_between([0.1, 3], 0.3, 1.5, alpha=0.1, color='purple', label='SCALE region')

    ax.set_xlabel("Mean Effect (|X̄n|)", fontsize=12)
    ax.set_ylabel("Coefficient of Variation (CV)", fontsize=12)
    ax.set_title("Behavioral Fingerprinting: Mean Effect vs CV\n(Decision Space)", fontsize=14, fontweight='bold')

    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.1, 2)

    # Create legend
    handles = [
        mpatches.Patch(color=RELATIONSHIP_COLORS["IDENTICAL"], label='IDENTICAL'),
        mpatches.Patch(color=RELATIONSHIP_COLORS["DISTILLED"], label='DISTILLED'),
        mpatches.Patch(color=RELATIONSHIP_COLORS["SAME_ARCHITECTURE_DIFFERENT_SCALE"], label='SCALE'),
        mpatches.Patch(color=RELATIONSHIP_COLORS["DIFFERENT_ARCHITECTURE"], label='DIFFERENT_ARCH'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Correct'),
        plt.Line2D([0], [0], marker='x', color='red', markersize=10, label='Incorrect'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(viz_dir / "mean_vs_cv_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: mean_vs_cv_scatter.png")

# ============================================================================
# 3. Variance Signature Profiles
# ============================================================================

def plot_variance_profiles():
    """Create bar chart comparing variance signatures across categories"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Organize by expected relationship
    relationship_data = {}
    for r in results:
        expected = r["expected_relationship"]
        if expected not in relationship_data:
            relationship_data[expected] = {"mean": [], "cv": [], "var": []}
        relationship_data[expected]["mean"].append(r["mean_effect"])
        relationship_data[expected]["cv"].append(r["cv"] if r["cv"] != float('inf') else 0)
        relationship_data[expected]["var"].append(r["variance"])

    # Mean Effect comparison
    relationships = list(relationship_data.keys())
    mean_values = [np.mean(relationship_data[r]["mean"]) for r in relationships]

    ax = axes[0]
    bars = ax.bar(
        range(len(relationships)),
        mean_values,
        color=[RELATIONSHIP_COLORS.get(r, "#95a5a6") for r in relationships],
        edgecolor="white",
        linewidth=2
    )
    ax.set_title("Mean Effect by Relationship", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Effect (|X̄n|)")
    ax.set_xticks(range(len(relationships)))
    ax.set_xticklabels([r.split("_")[0] for r in relationships], rotation=45, ha='right')

    # CV comparison
    cv_values = [np.mean([c for c in relationship_data[r]["cv"] if np.isfinite(c)]) if any(np.isfinite(c) for c in relationship_data[r]["cv"]) else 0 for r in relationships]

    ax = axes[1]
    bars = ax.bar(
        range(len(relationships)),
        cv_values,
        color=[RELATIONSHIP_COLORS.get(r, "#95a5a6") for r in relationships],
        edgecolor="white",
        linewidth=2
    )
    ax.set_title("CV by Relationship", fontsize=12, fontweight='bold')
    ax.set_ylabel("Coefficient of Variation")
    ax.set_xticks(range(len(relationships)))
    ax.set_xticklabels([r.split("_")[0] for r in relationships], rotation=45, ha='right')

    # Variance comparison
    var_values = [np.mean(relationship_data[r]["var"]) for r in relationships]

    ax = axes[2]
    bars = ax.bar(
        range(len(relationships)),
        var_values,
        color=[RELATIONSHIP_COLORS.get(r, "#95a5a6") for r in relationships],
        edgecolor="white",
        linewidth=2
    )
    ax.set_title("Variance by Relationship", fontsize=12, fontweight='bold')
    ax.set_ylabel("Variance")
    ax.set_xticks(range(len(relationships)))
    ax.set_xticklabels([r.split("_")[0] for r in relationships], rotation=45, ha='right')
    ax.set_yscale('log')

    plt.suptitle("Variance Signature Profiles by Relationship Type", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(viz_dir / "variance_profiles.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: variance_profiles.png")

# ============================================================================
# 4. Confusion Matrix Style Heatmap
# ============================================================================

def plot_classification_matrix():
    """Create heatmap showing expected vs actual classifications"""

    # Get unique relationships
    expected_set = set(r["expected_relationship"] for r in results)
    actual_set = set(r["actual_relationship"] for r in results)
    all_relationships = sorted(expected_set | actual_set)

    # Build confusion matrix
    n = len(all_relationships)
    matrix = np.zeros((n, n))

    for r in results:
        exp_idx = all_relationships.index(r["expected_relationship"])
        act_idx = all_relationships.index(r["actual_relationship"])
        matrix[exp_idx, act_idx] += 1

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap='Blues')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")

    # Add ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    short_labels = [r.replace("SAME_ARCH_DIFF_SCALE", "SCALE").replace("DIFFERENT_ARCH", "DIFF_ARCH").replace("SAME_ARCHITECTURE_DIFFERENT_SCALE", "SCALE").replace("DIFFERENT_ARCHITECTURE", "DIFF_ARCH") for r in all_relationships]
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.set_yticklabels(short_labels)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = int(matrix[i, j])
            if val > 0:
                color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                ax.text(j, i, val, ha="center", va="center", color=color, fontsize=14, fontweight='bold')

    ax.set_xlabel("Actual Classification", fontsize=12)
    ax.set_ylabel("Expected Classification", fontsize=12)
    ax.set_title("Classification Matrix: Expected vs Actual", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(viz_dir / "classification_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: classification_matrix.png")

# ============================================================================
# 5. Summary Dashboard
# ============================================================================

def plot_summary_dashboard():
    """Create a summary dashboard with key metrics"""

    fig = plt.figure(figsize=(16, 10))

    # Overall accuracy
    correct = sum(1 for r in results if r["relationship_match"])
    total = len(results)
    accuracy = correct / total * 100

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Accuracy pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pie(
        [correct, total - correct],
        labels=['Correct', 'Incorrect'],
        colors=['#2ecc71', '#e74c3c'],
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0)
    )
    ax1.set_title(f"Overall Accuracy\n{correct}/{total} = {accuracy:.1f}%", fontsize=12, fontweight='bold')

    # 2. Category breakdown
    ax2 = fig.add_subplot(gs[0, 1])
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r["relationship_match"]:
            categories[cat]["correct"] += 1

    cat_names = list(categories.keys())
    cat_correct = [categories[c]["correct"] for c in cat_names]
    cat_total = [categories[c]["total"] for c in cat_names]

    x = np.arange(len(cat_names))
    width = 0.35
    ax2.bar(x - width/2, cat_correct, width, label='Correct', color='#2ecc71')
    ax2.bar(x + width/2, cat_total, width, label='Total', color='#3498db', alpha=0.5)
    ax2.set_ylabel('Count')
    ax2.set_title('Results by Category', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("_", "\n") for c in cat_names], fontsize=8)
    ax2.legend()

    # 3. Mean effect distribution
    ax3 = fig.add_subplot(gs[0, 2])
    mean_values = [r["mean_effect"] for r in results]
    colors = ['#2ecc71' if r["relationship_match"] else '#e74c3c' for r in results]
    ax3.bar(range(len(results)), mean_values, color=colors)
    ax3.set_xlabel("Experiment")
    ax3.set_ylabel("Mean Effect")
    ax3.set_title("Mean Effect by Experiment", fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(results)))
    ax3.set_xticklabels([r["experiment_id"].split("_")[-1][:6] for r in results], rotation=45, ha='right', fontsize=8)

    # 4. Scatter plot (mean vs CV)
    ax4 = fig.add_subplot(gs[1, :2])
    for r in results:
        mean = r["mean_effect"]
        cv = r["cv"] if np.isfinite(r["cv"]) else 0
        color = '#2ecc71' if r["relationship_match"] else '#e74c3c'
        marker = 'o' if r["relationship_match"] else 'x'
        ax4.scatter(mean, cv, c=color, marker=marker, s=150, edgecolors='black', linewidths=1.5)
        ax4.annotate(r["experiment_id"].split("_")[-1][:6], (mean, cv), fontsize=8, alpha=0.7)

    ax4.axvline(x=3.0, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel("Mean Effect")
    ax4.set_ylabel("Coefficient of Variation")
    ax4.set_title("Decision Space: Mean Effect vs CV", fontsize=12, fontweight='bold')

    # 5. Summary table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    summary_text = f"""
    BEHAVIORAL FINGERPRINTING SUMMARY
    ================================

    Overall Accuracy: {accuracy:.1f}%
    Total Experiments: {total}
    Correct: {correct}

    Category Breakdown:
    """

    for cat in cat_names:
        cat_acc = categories[cat]["correct"] / categories[cat]["total"] * 100
        summary_text += f"\n    {cat}: {cat_acc:.0f}%"

    summary_text += f"""

    Key Findings:
    - IDENTICAL: Perfect detection (0 mean)
    - DISTILLED: Low CV (<0.3)
    - SCALE: Moderate CV (0.3-1.5)
    - DIFFERENT_ARCH: High mean (>3.0)

    Edge Case:
    GPT-2 vs GPT-Neo classified as
    DISTILLED due to similar architectures
    """

    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("Behavioral Fingerprinting: Experimental Results Dashboard",
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(viz_dir / "summary_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_dashboard.png")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\nGenerating behavioral fingerprinting visualizations...")
    print(f"Output directory: {viz_dir}\n")

    plot_accuracy_by_category()
    plot_mean_vs_cv()
    plot_variance_profiles()
    plot_classification_matrix()
    plot_summary_dashboard()

    print(f"\nAll visualizations saved to: {viz_dir}")
    print("\nVisualization files:")
    for f in viz_dir.glob("*.png"):
        print(f"  - {f.name}")
