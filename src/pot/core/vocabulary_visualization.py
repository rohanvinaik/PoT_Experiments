"""
Vocabulary Visualization Module

Provides visualization capabilities for vocabulary analysis including
overlap diagrams, token distribution histograms, and diff reports.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle, Circle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Visualization features will be limited.")


class VocabularyVisualizer:
    """
    Visualizer for vocabulary analysis results.
    
    Provides various visualization methods including Venn diagrams,
    histograms, and comparison charts.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir) if output_dir else Path("vocabulary_analysis")
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def create_overlap_diagram(
        self,
        ref_size: int,
        cand_size: int,
        shared_tokens: int,
        ref_name: str = "Reference",
        cand_name: str = "Candidate",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a Venn diagram showing vocabulary overlap.
        
        Args:
            ref_size: Reference vocabulary size
            cand_size: Candidate vocabulary size
            shared_tokens: Number of shared tokens
            ref_name: Name of reference model
            cand_name: Name of candidate model
            save_path: Path to save the diagram
        
        Returns:
            Path to saved diagram or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._create_text_venn_diagram(
                ref_size, cand_size, shared_tokens, ref_name, cand_name
            )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate unique tokens
        ref_unique = ref_size - shared_tokens
        cand_unique = cand_size - shared_tokens
        
        # Create circles for Venn diagram
        ref_circle = Circle((0.35, 0.5), 0.25, color='blue', alpha=0.3, label=ref_name)
        cand_circle = Circle((0.65, 0.5), 0.25, color='red', alpha=0.3, label=cand_name)
        
        ax.add_patch(ref_circle)
        ax.add_patch(cand_circle)
        
        # Add text labels
        ax.text(0.25, 0.5, f'{ref_unique:,}', fontsize=14, ha='center', va='center')
        ax.text(0.5, 0.5, f'{shared_tokens:,}\n(shared)', fontsize=14, ha='center', va='center')
        ax.text(0.75, 0.5, f'{cand_unique:,}', fontsize=14, ha='center', va='center')
        
        # Add model names
        ax.text(0.35, 0.8, f'{ref_name}\n{ref_size:,} tokens', fontsize=12, ha='center')
        ax.text(0.65, 0.8, f'{cand_name}\n{cand_size:,} tokens', fontsize=12, ha='center')
        
        # Calculate overlap percentage
        overlap_pct = shared_tokens / max(ref_size, cand_size) * 100
        ax.text(0.5, 0.15, f'Overlap: {overlap_pct:.1f}%', fontsize=14, ha='center', weight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title('Vocabulary Overlap Analysis', fontsize=16, weight='bold')
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "vocabulary_overlap.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _create_text_venn_diagram(
        self,
        ref_size: int,
        cand_size: int,
        shared_tokens: int,
        ref_name: str,
        cand_name: str
    ) -> str:
        """Create ASCII art Venn diagram"""
        ref_unique = ref_size - shared_tokens
        cand_unique = cand_size - shared_tokens
        overlap_pct = shared_tokens / max(ref_size, cand_size) * 100
        
        diagram = f"""
        Vocabulary Overlap Diagram
        ===========================
        
             {ref_name}                {cand_name}
           ({ref_size:,})              ({cand_size:,})
        ┌─────────────┐      ┌─────────────┐
        │             │      │             │
        │   {ref_unique:>7,}  ├──────┤  {cand_unique:>7,}   │
        │             │{shared_tokens:^6,}│             │
        └─────────────┘      └─────────────┘
                      shared
        
        Overlap: {overlap_pct:.1f}%
        """
        
        # Save to file
        output_path = self.output_dir / "vocabulary_overlap.txt"
        with open(output_path, 'w') as f:
            f.write(diagram)
        
        return diagram
    
    def create_token_distribution_histogram(
        self,
        token_categories: Dict[str, int],
        title: str = "Token Category Distribution",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create histogram showing token distribution by category.
        
        Args:
            token_categories: Dictionary of category -> count
            title: Chart title
            save_path: Path to save the histogram
        
        Returns:
            Path to saved histogram or text representation
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._create_text_histogram(token_categories, title)
        
        if not token_categories:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = list(token_categories.keys())
        counts = list(token_categories.values())
        
        # Create bar chart
        bars = ax.bar(categories, counts, color='steelblue', edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}', ha='center', va='bottom')
        
        ax.set_xlabel('Token Category', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        
        # Rotate x labels if needed
        if len(categories) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "token_distribution.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _create_text_histogram(
        self,
        token_categories: Dict[str, int],
        title: str
    ) -> str:
        """Create ASCII histogram"""
        if not token_categories:
            return "No data to display"
        
        max_count = max(token_categories.values())
        max_label_len = max(len(cat) for cat in token_categories.keys())
        
        lines = [title, "=" * (max_label_len + 40)]
        
        for category, count in sorted(token_categories.items(), key=lambda x: x[1], reverse=True):
            bar_len = int((count / max_count) * 30) if max_count > 0 else 0
            bar = "█" * bar_len
            lines.append(f"{category:<{max_label_len}} │ {bar} {count:,}")
        
        histogram = "\n".join(lines)
        
        # Save to file
        output_path = self.output_dir / "token_distribution.txt"
        with open(output_path, 'w') as f:
            f.write(histogram)
        
        return histogram
    
    def create_comparison_chart(
        self,
        metrics: Dict[str, Tuple[float, float]],
        title: str = "Model Vocabulary Comparison",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create comparison chart for multiple metrics.
        
        Args:
            metrics: Dictionary of metric_name -> (reference_value, candidate_value)
            title: Chart title
            save_path: Path to save the chart
        
        Returns:
            Path to saved chart or text representation
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._create_text_comparison(metrics, title)
        
        if not metrics:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = list(metrics.keys())
        ref_values = [v[0] for v in metrics.values()]
        cand_values = [v[1] for v in metrics.values()]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ref_values, width, label='Reference', color='skyblue')
        bars2 = ax.bar(x + width/2, cand_values, width, label='Candidate', color='lightcoral')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.legend()
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "vocabulary_comparison.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _create_text_comparison(
        self,
        metrics: Dict[str, Tuple[float, float]],
        title: str
    ) -> str:
        """Create text-based comparison"""
        lines = [title, "=" * 60]
        lines.append(f"{'Metric':<30} {'Reference':>12} {'Candidate':>12}")
        lines.append("-" * 60)
        
        for metric, (ref_val, cand_val) in metrics.items():
            lines.append(f"{metric:<30} {ref_val:>12.2f} {cand_val:>12.2f}")
        
        comparison = "\n".join(lines)
        
        # Save to file
        output_path = self.output_dir / "vocabulary_comparison.txt"
        with open(output_path, 'w') as f:
            f.write(comparison)
        
        return comparison
    
    def create_diff_visualization(
        self,
        added_tokens: List[str],
        removed_tokens: List[str],
        max_display: int = 50,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create a diff visualization showing added and removed tokens.
        
        Args:
            added_tokens: List of added token strings
            removed_tokens: List of removed token strings
            max_display: Maximum number of tokens to display
            save_path: Path to save the diff
        
        Returns:
            Path to saved diff or text representation
        """
        lines = [
            "Vocabulary Diff Report",
            "=" * 70,
            "",
            f"Added Tokens ({len(added_tokens)} total):",
            "-" * 40
        ]
        
        # Show added tokens
        for i, token in enumerate(added_tokens[:max_display]):
            lines.append(f"+ {token}")
        if len(added_tokens) > max_display:
            lines.append(f"... and {len(added_tokens) - max_display} more")
        
        lines.extend([
            "",
            f"Removed Tokens ({len(removed_tokens)} total):",
            "-" * 40
        ])
        
        # Show removed tokens
        for i, token in enumerate(removed_tokens[:max_display]):
            lines.append(f"- {token}")
        if len(removed_tokens) > max_display:
            lines.append(f"... and {len(removed_tokens) - max_display} more")
        
        diff_text = "\n".join(lines)
        
        # Save to file
        if save_path is None:
            save_path = self.output_dir / "vocabulary_diff.txt"
        
        with open(save_path, 'w') as f:
            f.write(diff_text)
        
        return str(save_path)
    
    def create_summary_dashboard(
        self,
        analysis_report: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create a comprehensive summary dashboard.
        
        Args:
            analysis_report: Complete vocabulary analysis report
            save_path: Path to save the dashboard
        
        Returns:
            Path to saved dashboard
        """
        lines = [
            "=" * 80,
            "VOCABULARY ANALYSIS DASHBOARD",
            "=" * 80,
            ""
        ]
        
        # Overview section
        lines.extend([
            "📊 OVERVIEW",
            "-" * 40,
            f"Reference Size: {analysis_report.get('reference_size', 0):,} tokens",
            f"Candidate Size: {analysis_report.get('candidate_size', 0):,} tokens",
            ""
        ])
        
        # Overlap analysis
        if 'overlap_analysis' in analysis_report:
            overlap = analysis_report['overlap_analysis']
            lines.extend([
                "🔄 OVERLAP ANALYSIS",
                "-" * 40,
                f"Shared Tokens: {overlap.get('shared_tokens', 0):,}",
                f"Overlap Ratio: {overlap.get('overlap_ratio', 0):.1%}",
                f"Core Vocabulary Overlap: {overlap.get('core_vocabulary_overlap', 0):.1%}",
                ""
            ])
        
        # Token categories
        if 'token_categories' in analysis_report:
            categories = analysis_report['token_categories']
            lines.extend([
                "📝 TOKEN CATEGORIES",
                "-" * 40
            ])
            
            if 'category_distribution' in categories:
                for cat, count in categories['category_distribution'].items():
                    lines.append(f"  {cat}: {count}")
            lines.append("")
        
        # Architectural impact
        if 'architectural_impact' in analysis_report:
            impact = analysis_report['architectural_impact']
            lines.extend([
                "🏗️ ARCHITECTURAL IMPACT",
                "-" * 40,
                f"Parameter Change: {impact.get('parameter_difference_ratio', 0):.2%}",
                f"Functional Impact: {impact.get('functional_impact', 'unknown')}",
                f"Backward Compatible: {impact.get('backward_compatible', False)}",
                ""
            ])
        
        # Recommendations
        if 'recommendations' in analysis_report:
            lines.extend([
                "💡 RECOMMENDATIONS",
                "-" * 40
            ])
            for i, rec in enumerate(analysis_report['recommendations'], 1):
                # Wrap long recommendations
                if len(rec) > 70:
                    words = rec.split()
                    current_line = f"{i}. "
                    for word in words:
                        if len(current_line) + len(word) + 1 > 70:
                            lines.append(current_line)
                            current_line = "   " + word
                        else:
                            current_line += " " + word if current_line != f"{i}. " else word
                    lines.append(current_line)
                else:
                    lines.append(f"{i}. {rec}")
            lines.append("")
        
        # Verification compatibility
        lines.extend([
            "✅ VERIFICATION COMPATIBILITY",
            "-" * 40,
            f"Can Verify: {analysis_report.get('can_verify', False)}",
            f"Strategy: {analysis_report.get('verification_strategy', 'unknown')}",
            f"Confidence Adjustment: {analysis_report.get('confidence_adjustment', 1.0):.2f}x",
            "",
            "=" * 80
        ])
        
        dashboard = "\n".join(lines)
        
        # Save to file
        if save_path is None:
            save_path = self.output_dir / "vocabulary_dashboard.txt"
        
        with open(save_path, 'w') as f:
            f.write(dashboard)
        
        # Also save as JSON for programmatic access
        json_path = Path(str(save_path).replace('.txt', '.json'))
        with open(json_path, 'w') as f:
            json.dump(analysis_report, f, indent=2)
        
        return str(save_path)


def visualize_vocabulary_analysis(
    analysis_report: Dict[str, Any],
    output_dir: Optional[Path] = None,
    create_plots: bool = True
) -> Dict[str, str]:
    """
    Convenience function to create all visualizations for a vocabulary analysis.
    
    Args:
        analysis_report: Complete vocabulary analysis report
        output_dir: Directory to save visualizations
        create_plots: Whether to create matplotlib plots (if available)
    
    Returns:
        Dictionary mapping visualization type to file path
    """
    visualizer = VocabularyVisualizer(output_dir)
    results = {}
    
    # Create summary dashboard
    results['dashboard'] = visualizer.create_summary_dashboard(analysis_report)
    
    # Create overlap diagram if we have the data
    if 'overlap_analysis' in analysis_report:
        overlap = analysis_report['overlap_analysis']
        results['overlap_diagram'] = visualizer.create_overlap_diagram(
            analysis_report.get('reference_size', 0),
            analysis_report.get('candidate_size', 0),
            overlap.get('shared_tokens', 0)
        )
    
    # Create token distribution histogram
    if 'token_categories' in analysis_report:
        categories = analysis_report['token_categories']
        if 'category_distribution' in categories:
            results['token_distribution'] = visualizer.create_token_distribution_histogram(
                categories['category_distribution']
            )
    
    # Create comparison chart
    if 'overlap_analysis' in analysis_report:
        overlap = analysis_report['overlap_analysis']
        metrics = {
            'Vocabulary Size': (
                analysis_report.get('reference_size', 0) / 1000,
                analysis_report.get('candidate_size', 0) / 1000
            ),
            'Overlap Ratio': (
                overlap.get('overlap_ratio', 0),
                overlap.get('overlap_ratio', 0)
            ),
            'Core Overlap': (
                overlap.get('core_vocabulary_overlap', 0),
                overlap.get('core_vocabulary_overlap', 0)
            )
        }
        results['comparison_chart'] = visualizer.create_comparison_chart(metrics)
    
    logger.info(f"Created {len(results)} visualizations in {visualizer.output_dir}")
    
    return results