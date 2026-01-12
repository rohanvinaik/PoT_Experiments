#!/usr/bin/env python3
"""
Publication-Quality Results Visualization and Report Generator

Generates comprehensive visualizations and reports from PoT experiment results:
- Summary statistics tables (LaTeX and Markdown)
- Accuracy heatmaps by experiment category
- Confidence interval plots
- Query efficiency analysis
- HTML report with embedded figures
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings

# Visualization dependencies
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
except ImportError as e:
    print(f"Missing visualization dependency: {e}")
    print("Install with: pip install numpy pandas matplotlib seaborn")
    sys.exit(1)

# Optional: plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


@dataclass
class ExperimentResult:
    """Single experiment result"""
    experiment_id: str
    category: str
    ref_model: str
    cand_model: str
    expected_decision: str
    actual_decision: str
    confidence: float
    queries_used: int
    duration_seconds: float
    correct: bool
    error: Optional[str] = None


def load_publication_results(results_dir: Path) -> Tuple[List[ExperimentResult], Dict]:
    """Load results from publication experiment run"""
    results = []
    metadata = {}

    # Find the publication_results.json file
    json_path = results_dir / "publication_results.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
            metadata = data.get("metadata", {})
            for r in data.get("results", []):
                results.append(ExperimentResult(**r))
    else:
        # Try to reconstruct from individual experiment directories
        for exp_dir in sorted(results_dir.iterdir()):
            if exp_dir.is_dir():
                pipeline_files = list(exp_dir.glob("pipeline_results_*.json"))
                if pipeline_files:
                    with open(pipeline_files[0]) as f:
                        data = json.load(f)
                        # Reconstruct result
                        results.append(ExperimentResult(
                            experiment_id=exp_dir.name,
                            category=_infer_category(exp_dir.name),
                            ref_model=data.get("ref_model", "unknown"),
                            cand_model=data.get("cand_model", "unknown"),
                            expected_decision=_infer_expected(exp_dir.name),
                            actual_decision=data.get("decision", "ERROR"),
                            confidence=data.get("confidence", 0.0),
                            queries_used=data.get("n_queries", 0),
                            duration_seconds=data.get("total_duration", 0.0),
                            correct=data.get("decision") == _infer_expected(exp_dir.name),
                            error=data.get("error")
                        ))

    return results, metadata


def _infer_category(exp_id: str) -> str:
    """Infer category from experiment ID"""
    if exp_id.startswith("self_"):
        return "self_consistency"
    elif exp_id.startswith("distill_"):
        return "distillation"
    elif exp_id.startswith("finetune_"):
        return "finetuning"
    elif exp_id.startswith("scale_"):
        return "scale"
    elif exp_id.startswith("arch_"):
        return "architecture"
    elif exp_id.startswith("large_"):
        return "large_models"
    elif exp_id.startswith("quant_"):
        return "quantization"
    else:
        return "unknown"


def _infer_expected(exp_id: str) -> str:
    """Infer expected decision from experiment ID"""
    if exp_id.startswith("self_"):
        return "SAME"
    else:
        return "DIFFERENT"


def create_summary_table(results: List[ExperimentResult]) -> pd.DataFrame:
    """Create summary statistics table"""
    data = []
    for r in results:
        data.append({
            "Experiment": r.experiment_id,
            "Category": r.category.replace("_", " ").title(),
            "Reference": r.ref_model.split("/")[-1][:20],
            "Candidate": r.cand_model.split("/")[-1][:20],
            "Expected": r.expected_decision,
            "Actual": r.actual_decision,
            "Correct": "✓" if r.correct else "✗",
            "Confidence": f"{r.confidence:.1%}" if r.confidence else "-",
            "Queries": r.queries_used,
            "Time (s)": f"{r.duration_seconds:.1f}"
        })

    df = pd.DataFrame(data)
    return df


def create_category_summary(results: List[ExperimentResult]) -> pd.DataFrame:
    """Create per-category accuracy summary"""
    categories = {}
    for r in results:
        cat = r.category.replace("_", " ").title()
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0, "queries": [], "times": []}
        categories[cat]["total"] += 1
        if r.correct:
            categories[cat]["correct"] += 1
        if r.queries_used > 0:
            categories[cat]["queries"].append(r.queries_used)
        if r.duration_seconds > 0:
            categories[cat]["times"].append(r.duration_seconds)

    data = []
    for cat, stats in categories.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        avg_queries = np.mean(stats["queries"]) if stats["queries"] else 0
        avg_time = np.mean(stats["times"]) if stats["times"] else 0
        data.append({
            "Category": cat,
            "Experiments": stats["total"],
            "Correct": stats["correct"],
            "Accuracy": f"{accuracy:.1%}",
            "Avg Queries": f"{avg_queries:.1f}",
            "Avg Time (s)": f"{avg_time:.1f}"
        })

    # Add overall row
    total = sum(c["total"] for c in categories.values())
    correct = sum(c["correct"] for c in categories.values())
    all_queries = [q for c in categories.values() for q in c["queries"]]
    all_times = [t for c in categories.values() for t in c["times"]]
    data.append({
        "Category": "**Overall**",
        "Experiments": total,
        "Correct": correct,
        "Accuracy": f"{correct/total:.1%}" if total > 0 else "-",
        "Avg Queries": f"{np.mean(all_queries):.1f}" if all_queries else "-",
        "Avg Time (s)": f"{np.mean(all_times):.1f}" if all_times else "-"
    })

    return pd.DataFrame(data)


def plot_accuracy_by_category(results: List[ExperimentResult], output_path: Path):
    """Create accuracy bar chart by category"""
    categories = {}
    for r in results:
        cat = r.category.replace("_", " ").title()
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r.correct:
            categories[cat]["correct"] += 1

    # Sort by accuracy
    cats = list(categories.keys())
    accuracies = [categories[c]["correct"] / categories[c]["total"] for c in cats]
    totals = [categories[c]["total"] for c in cats]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Color based on accuracy
    colors = ['#2ecc71' if acc >= 0.9 else '#f1c40f' if acc >= 0.7 else '#e74c3c'
              for acc in accuracies]

    bars = ax.barh(cats, accuracies, color=colors, edgecolor='black', linewidth=0.5)

    # Add labels
    for bar, total, acc in zip(bars, totals, accuracies):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{acc:.0%} ({total})', va='center', fontsize=10)

    ax.set_xlim(0, 1.2)
    ax.set_xlabel('Accuracy')
    ax.set_title('PoT Verification Accuracy by Experiment Category')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='≥90%'),
        mpatches.Patch(facecolor='#f1c40f', edgecolor='black', label='70-90%'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='<70%')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_by_category.png', dpi=300)
    plt.savefig(output_path / 'accuracy_by_category.pdf')
    plt.close()

    print(f"  Saved accuracy plot: {output_path / 'accuracy_by_category.png'}")


def plot_queries_distribution(results: List[ExperimentResult], output_path: Path):
    """Plot query count distribution by decision type"""
    same_queries = [r.queries_used for r in results
                    if r.actual_decision == "SAME" and r.queries_used > 0]
    diff_queries = [r.queries_used for r in results
                    if r.actual_decision == "DIFFERENT" and r.queries_used > 0]

    fig, ax = plt.subplots(figsize=(10, 5))

    if same_queries:
        ax.hist(same_queries, bins=15, alpha=0.7, label=f'SAME (n={len(same_queries)})',
                color='#3498db', edgecolor='black')
    if diff_queries:
        ax.hist(diff_queries, bins=15, alpha=0.7, label=f'DIFFERENT (n={len(diff_queries)})',
                color='#e74c3c', edgecolor='black')

    ax.axvline(x=30, color='gray', linestyle='--', alpha=0.7, label='n_min (AUDIT)')

    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Count')
    ax.set_title('Query Efficiency: Queries to Reach Decision')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'query_distribution.png', dpi=300)
    plt.savefig(output_path / 'query_distribution.pdf')
    plt.close()

    print(f"  Saved query distribution plot: {output_path / 'query_distribution.png'}")


def plot_confidence_by_experiment(results: List[ExperimentResult], output_path: Path):
    """Plot confidence levels for each experiment"""
    # Filter out errored results
    valid_results = [r for r in results if r.confidence > 0]

    if not valid_results:
        print("  No valid results for confidence plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    exp_ids = [r.experiment_id[:15] for r in valid_results]
    confidences = [r.confidence for r in valid_results]
    colors = ['#2ecc71' if r.correct else '#e74c3c' for r in valid_results]

    bars = ax.bar(range(len(exp_ids)), confidences, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(len(exp_ids)))
    ax.set_xticklabels(exp_ids, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Confidence')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.7, label='99% threshold')
    ax.axhline(y=0.975, color='gray', linestyle=':', alpha=0.5, label='97.5% threshold')
    ax.set_title('Verification Confidence by Experiment')
    ax.legend()

    # Add correct/incorrect legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='Correct'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Incorrect')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path / 'confidence_by_experiment.png', dpi=300)
    plt.savefig(output_path / 'confidence_by_experiment.pdf')
    plt.close()

    print(f"  Saved confidence plot: {output_path / 'confidence_by_experiment.png'}")


def plot_decision_heatmap(results: List[ExperimentResult], output_path: Path):
    """Create decision matrix heatmap"""
    # Build confusion-style matrix
    expected = ['SAME', 'DIFFERENT']
    actual = ['SAME', 'DIFFERENT', 'UNDECIDED', 'ERROR']

    matrix = np.zeros((len(expected), len(actual)))

    for r in results:
        if r.expected_decision in expected:
            i = expected.index(r.expected_decision)
            if r.actual_decision in actual:
                j = actual.index(r.actual_decision)
                matrix[i, j] += 1

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=actual, yticklabels=expected,
                ax=ax, cbar_kws={'label': 'Count'})

    ax.set_xlabel('Actual Decision')
    ax.set_ylabel('Expected Decision')
    ax.set_title('Decision Matrix: Expected vs Actual')

    plt.tight_layout()
    plt.savefig(output_path / 'decision_heatmap.png', dpi=300)
    plt.savefig(output_path / 'decision_heatmap.pdf')
    plt.close()

    print(f"  Saved decision heatmap: {output_path / 'decision_heatmap.png'}")


def plot_time_vs_model_size(results: List[ExperimentResult], output_path: Path):
    """Plot relationship between model size and verification time"""
    # Estimate model sizes (approximate)
    size_estimates = {
        "gpt2": 0.124, "distilgpt2": 0.082,
        "pythia-70m": 0.07, "pythia-160m": 0.16, "pythia-410m": 0.41,
        "phi-2": 2.7, "TinyLlama": 1.1, "gpt-neo-125m": 0.125, "gpt-neo-1.3B": 1.3,
        "Llama-2-7b": 7.0, "Mistral-7B": 7.0, "zephyr-7b": 7.0, "falcon-7b": 7.0
    }

    sizes = []
    times = []
    labels = []

    for r in results:
        if r.duration_seconds > 0:
            # Find size estimate
            size = 0.5  # default
            for key, val in size_estimates.items():
                if key.lower() in r.ref_model.lower() or key.lower() in r.cand_model.lower():
                    size = max(size, val)
            sizes.append(size)
            times.append(r.duration_seconds)
            labels.append(r.experiment_id[:10])

    if not sizes:
        print("  No data for time vs model size plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(sizes, times, c='#3498db', s=100, alpha=0.7, edgecolors='black')

    # Fit trend line
    if len(sizes) > 2:
        z = np.polyfit(sizes, times, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(sizes), max(sizes), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.5, label=f'Trend: {z[0]:.1f}x + {z[1]:.1f}')

    ax.set_xlabel('Larger Model Size (B params)')
    ax.set_ylabel('Verification Time (seconds)')
    ax.set_title('Verification Time vs Model Size')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'time_vs_size.png', dpi=300)
    plt.savefig(output_path / 'time_vs_size.pdf')
    plt.close()

    print(f"  Saved time vs size plot: {output_path / 'time_vs_size.png'}")


def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Generate LaTeX table from DataFrame"""
    latex = df.to_latex(index=False, escape=False)

    # Wrap in table environment
    full_latex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{latex}
\\end{{table}}
"""
    return full_latex


def generate_markdown_report(results: List[ExperimentResult], output_path: Path) -> str:
    """Generate comprehensive Markdown report"""

    # Calculate stats
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    errors = sum(1 for r in results if r.error)

    valid_results = [r for r in results if r.confidence > 0]
    avg_confidence = np.mean([r.confidence for r in valid_results]) if valid_results else 0
    avg_queries = np.mean([r.queries_used for r in valid_results]) if valid_results else 0
    avg_time = np.mean([r.duration_seconds for r in valid_results]) if valid_results else 0

    # Build report
    report = f"""# Proof-of-Training (PoT) Experimental Results

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Experiments | {total} |
| Correct Decisions | {correct} |
| **Overall Accuracy** | **{correct/total:.1%}** |
| Average Confidence | {avg_confidence:.1%} |
| Average Queries | {avg_queries:.1f} |
| Average Time | {avg_time:.1f}s |
| Errors | {errors} |

## Results by Category

"""

    # Category summary
    category_df = create_category_summary(results)
    report += category_df.to_markdown(index=False) + "\n\n"

    # Detailed results
    report += "## Detailed Results\n\n"
    summary_df = create_summary_table(results)
    report += summary_df.to_markdown(index=False) + "\n\n"

    # Figures
    report += """## Visualizations

### Accuracy by Category
![Accuracy by Category](accuracy_by_category.png)

### Query Efficiency
![Query Distribution](query_distribution.png)

### Confidence Levels
![Confidence by Experiment](confidence_by_experiment.png)

### Decision Matrix
![Decision Heatmap](decision_heatmap.png)

### Time vs Model Size
![Time vs Size](time_vs_size.png)

## Methodology

The Proof-of-Training framework uses **sequential statistical testing** with:
- **Empirical-Bernstein confidence intervals** for anytime-valid inference
- **HMAC-SHA256 pre-committed challenges** for reproducibility
- **Early stopping** based on explicit SAME/DIFFERENT decision rules

### Decision Rules

- **SAME**: Confidence interval within ±γ tolerance with sufficient precision
- **DIFFERENT**: Effect size exceeds δ* threshold with low relative margin of error
- **UNDECIDED**: Neither criterion met (continues sampling or returns at budget)

### Testing Mode: AUDIT

| Parameter | Value |
|-----------|-------|
| Confidence Level (1-α) | 99% |
| Tolerance (γ) | 0.10 |
| Effect Size Threshold (δ*) | 1.0 |
| Query Range | [30, 400] |

## Conclusions

"""

    if correct / total >= 0.95:
        report += f"""The PoT framework achieved **{correct/total:.1%} accuracy** across {total} experiments,
demonstrating robust behavioral verification for:
- Self-consistency testing (identical models)
- Distillation detection
- Fine-tuning detection
- Architecture differentiation
- Scale variation detection
"""
    elif correct / total >= 0.80:
        report += f"""The PoT framework achieved **{correct/total:.1%} accuracy** across {total} experiments.
Performance was strong for most categories, with some challenging edge cases identified.
"""
    else:
        report += f"""The PoT framework achieved **{correct/total:.1%} accuracy** across {total} experiments.
Further investigation is needed for categories with lower accuracy.
"""

    return report


def generate_html_report(results: List[ExperimentResult], output_path: Path):
    """Generate interactive HTML report"""
    if not PLOTLY_AVAILABLE:
        print("  Plotly not available - skipping interactive HTML report")
        return

    # Create interactive visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy by Category', 'Query Distribution',
                        'Confidence Levels', 'Time vs Queries'),
        specs=[[{'type': 'bar'}, {'type': 'histogram'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )

    # Accuracy by category
    categories = {}
    for r in results:
        cat = r.category.replace("_", " ").title()
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r.correct:
            categories[cat]["correct"] += 1

    cats = list(categories.keys())
    accuracies = [categories[c]["correct"] / categories[c]["total"] for c in cats]

    fig.add_trace(
        go.Bar(x=cats, y=accuracies, name='Accuracy',
               marker_color=['green' if a >= 0.9 else 'yellow' if a >= 0.7 else 'red' for a in accuracies]),
        row=1, col=1
    )

    # Query distribution
    queries = [r.queries_used for r in results if r.queries_used > 0]
    fig.add_trace(
        go.Histogram(x=queries, name='Queries', nbinsx=20),
        row=1, col=2
    )

    # Confidence levels
    valid_results = [r for r in results if r.confidence > 0]
    fig.add_trace(
        go.Bar(x=[r.experiment_id[:12] for r in valid_results],
               y=[r.confidence for r in valid_results],
               name='Confidence',
               marker_color=['green' if r.correct else 'red' for r in valid_results]),
        row=2, col=1
    )

    # Time vs Queries
    fig.add_trace(
        go.Scatter(x=[r.queries_used for r in valid_results],
                   y=[r.duration_seconds for r in valid_results],
                   mode='markers',
                   name='Time vs Queries',
                   text=[r.experiment_id for r in valid_results]),
        row=2, col=2
    )

    fig.update_layout(height=800, showlegend=False, title_text="PoT Experiment Results Dashboard")

    fig.write_html(output_path / 'interactive_report.html')
    print(f"  Saved interactive HTML report: {output_path / 'interactive_report.html'}")


def main():
    parser = argparse.ArgumentParser(description='Generate publication-quality PoT results report')
    parser.add_argument('--results-dir', type=str,
                        default='experimental_results/publication',
                        help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str,
                        default='experimental_results/publication_report',
                        help='Output directory for report')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Specific run ID to analyze (default: latest)')

    args = parser.parse_args()

    # Find results directory
    results_base = Path(args.results_dir)
    if args.run_id:
        results_dir = results_base / args.run_id
    else:
        # Find latest run
        runs = sorted([d for d in results_base.iterdir() if d.is_dir() and d.name.startswith('run_')])
        if not runs:
            print(f"No runs found in {results_base}")
            sys.exit(1)
        results_dir = runs[-1]

    print(f"Analyzing results from: {results_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results, metadata = load_publication_results(results_dir)

    if not results:
        print("No results found!")
        sys.exit(1)

    print(f"Loaded {len(results)} experiment results")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_accuracy_by_category(results, output_dir)
    plot_queries_distribution(results, output_dir)
    plot_confidence_by_experiment(results, output_dir)
    plot_decision_heatmap(results, output_dir)
    plot_time_vs_model_size(results, output_dir)

    # Generate reports
    print("\nGenerating reports...")

    # Markdown report
    md_report = generate_markdown_report(results, output_dir)
    md_path = output_dir / 'PUBLICATION_REPORT.md'
    with open(md_path, 'w') as f:
        f.write(md_report)
    print(f"  Saved Markdown report: {md_path}")

    # LaTeX tables
    summary_df = create_summary_table(results)
    latex_path = output_dir / 'results_table.tex'
    with open(latex_path, 'w') as f:
        f.write(generate_latex_table(summary_df,
                "PoT Verification Results", "tab:pot_results"))
    print(f"  Saved LaTeX tables: {latex_path}")

    # CSV for further analysis
    summary_df.to_csv(output_dir / 'results.csv', index=False)
    print(f"  Saved CSV: {output_dir / 'results.csv'}")

    # Interactive HTML
    generate_html_report(results, output_dir)

    # Summary
    print(f"\n{'='*60}")
    print("REPORT GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Total experiments: {len(results)}")
    correct = sum(1 for r in results if r.correct)
    print(f"Accuracy: {correct}/{len(results)} ({correct/len(results):.1%})")
    print(f"\nFiles generated:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
