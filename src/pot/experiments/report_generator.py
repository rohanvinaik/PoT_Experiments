#!/usr/bin/env python3
"""
Automated Report Generator for PoT Framework

Comprehensive report generation from experimental results including:
- Executive summaries with key metrics
- Detailed per-challenge analysis
- Statistical comparisons with paper claims
- Automated discrepancy detection
- Multiple output formats (Markdown, HTML, LaTeX, JSON)
- Interactive visualizations (ROC curves, histograms, distributions)

Usage:
    from pot.experiments.report_generator import ReportGenerator
    
    generator = ReportGenerator("path/to/results")
    markdown_report = generator.generate_markdown_report()
    plots = generator.generate_plots()
    latex_tables = generator.generate_latex_tables()
"""

import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings
import logging
import base64
from io import BytesIO

# Optional imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class ResultMetrics:
    """Container for experimental results and metrics."""
    far: float = 0.0
    frr: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc: float = 0.0
    total_queries: int = 0
    avg_queries: float = 0.0
    processing_time: float = 0.0
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    raw_data: List[Dict[str, Any]] = field(default_factory=list)

@dataclass 
class PaperClaims:
    """Container for paper claims to compare against."""
    far: float = 0.01
    frr: float = 0.01
    accuracy: float = 0.99
    efficiency_gain: float = 0.90
    average_queries: float = 10.0
    confidence_level: float = 0.95
    auc: float = 0.99
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperClaims':
        """Create PaperClaims from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

@dataclass
class Discrepancy:
    """Container for identified discrepancies."""
    metric: str
    claimed: float
    actual: float
    difference: float
    relative_difference: float
    severity: str  # 'minor', 'moderate', 'major'
    suggestion: str

class ReportGenerator:
    """Automated report generator for PoT experimental results."""
    
    def __init__(self, results_path: str, paper_claims_path: Optional[str] = None):
        """
        Initialize the report generator.
        
        Args:
            results_path: Path to results directory or file
            paper_claims_path: Optional path to paper claims file
        """
        self.results_path = Path(results_path)
        self.paper_claims = self._load_paper_claims(paper_claims_path)
        self.data = self.load_results(results_path)
        self.metrics = self._calculate_metrics()
        self.discrepancies = self._detect_discrepancies()
        
        # Create output directory
        self.output_dir = self.results_path.parent / "reports" / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ReportGenerator initialized with {len(self.data)} result files")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_results(self, results_path: str) -> List[Dict[str, Any]]:
        """
        Load experimental results from CSV/JSON files.
        
        Args:
            results_path: Path to results directory or file
            
        Returns:
            List of result dictionaries
        """
        results = []
        path = Path(results_path)
        
        if path.is_file():
            results.extend(self._load_single_file(path))
        elif path.is_dir():
            # Load all JSON and CSV files in directory
            for file_path in path.rglob("*.json"):
                if "config" not in file_path.name.lower():  # Skip config files
                    results.extend(self._load_single_file(file_path))
            
            for file_path in path.rglob("*.csv"):
                results.extend(self._load_single_file(file_path))
        
        logger.info(f"Loaded {len(results)} result records")
        return results
    
    def _load_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load results from a single file."""
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        return [data]
                    else:
                        logger.warning(f"Unexpected JSON structure in {file_path}")
                        return []
            
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                return df.to_dict('records')
                
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            
        return []
    
    def _load_paper_claims(self, paper_claims_path: Optional[str]) -> PaperClaims:
        """Load paper claims from file or use defaults."""
        if paper_claims_path and Path(paper_claims_path).exists():
            try:
                with open(paper_claims_path, 'r') as f:
                    claims_data = json.load(f)
                return PaperClaims.from_dict(claims_data)
            except Exception as e:
                logger.warning(f"Failed to load paper claims: {e}")
        
        return PaperClaims()  # Use defaults
    
    def _calculate_metrics(self) -> ResultMetrics:
        """Calculate comprehensive metrics from loaded data."""
        if not self.data:
            return ResultMetrics()
        
        # Extract key metrics from data
        fars = []
        frrs = []
        accuracies = []
        query_counts = []
        processing_times = []
        
        for result in self.data:
            # Extract FAR/FRR if available
            if 'far' in result:
                fars.append(float(result['far']))
            if 'frr' in result:
                frrs.append(float(result['frr']))
            if 'accuracy' in result:
                accuracies.append(float(result['accuracy']))
            
            # Extract query counts
            for key in ['queries', 'query_count', 'total_queries', 'n_queries']:
                if key in result:
                    query_counts.append(int(result[key]))
                    break
            
            # Extract timing
            for key in ['time', 'processing_time', 'duration']:
                if key in result:
                    processing_times.append(float(result[key]))
                    break
        
        # Calculate aggregate metrics
        metrics = ResultMetrics()
        
        if fars:
            metrics.far = np.mean(fars)
        if frrs:
            metrics.frr = np.mean(frrs)
        if accuracies:
            metrics.accuracy = np.mean(accuracies)
        if query_counts:
            metrics.total_queries = sum(query_counts)
            metrics.avg_queries = np.mean(query_counts)
        if processing_times:
            metrics.processing_time = np.mean(processing_times)
        
        # Calculate derived metrics
        if metrics.far != 0 or metrics.frr != 0:
            metrics.precision = 1 - metrics.far if metrics.far > 0 else 1.0
            metrics.recall = 1 - metrics.frr if metrics.frr > 0 else 1.0
            
            if metrics.precision + metrics.recall > 0:
                metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
        
        # Store raw data for detailed analysis
        metrics.raw_data = self.data
        
        # Calculate confidence intervals
        metrics.confidence_intervals = self._calculate_confidence_intervals(fars, frrs, accuracies)
        
        return metrics
    
    def _calculate_confidence_intervals(self, fars: List[float], frrs: List[float], 
                                     accuracies: List[float]) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals."""
        intervals = {}
        
        def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
            if len(data) < 2:
                return (0.0, 0.0)
            
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_means.append(np.mean(sample))
            
            alpha = 1 - confidence
            lower = np.percentile(bootstrap_means, 100 * alpha / 2)
            upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
            return (lower, upper)
        
        if fars:
            intervals['far'] = bootstrap_ci(fars)
        if frrs:
            intervals['frr'] = bootstrap_ci(frrs)
        if accuracies:
            intervals['accuracy'] = bootstrap_ci(accuracies)
        
        return intervals
    
    def _detect_discrepancies(self) -> List[Discrepancy]:
        """Detect discrepancies between actual results and paper claims."""
        discrepancies = []
        
        def classify_severity(relative_diff: float) -> str:
            if relative_diff < 0.1:  # < 10%
                return 'minor'
            elif relative_diff < 0.25:  # < 25%
                return 'moderate'
            else:
                return 'major'
        
        def suggest_reconciliation(metric: str, claimed: float, actual: float) -> str:
            suggestions = {
                'far': "Check threshold settings, challenge difficulty, or model calibration",
                'frr': "Verify challenge generation parameters and model evaluation conditions",
                'accuracy': "Review experimental setup, data preprocessing, and evaluation metrics",
                'efficiency_gain': "Validate query optimization and sequential testing implementation",
                'average_queries': "Check early stopping criteria and challenge generation strategy"
            }
            return suggestions.get(metric, "Review experimental conditions and implementation details")
        
        # Compare key metrics
        comparisons = [
            ('far', self.paper_claims.far, self.metrics.far),
            ('frr', self.paper_claims.frr, self.metrics.frr),
            ('accuracy', self.paper_claims.accuracy, self.metrics.accuracy),
            ('average_queries', self.paper_claims.average_queries, self.metrics.avg_queries)
        ]
        
        for metric, claimed, actual in comparisons:
            if claimed > 0 and actual > 0:  # Only compare if both values are meaningful
                difference = actual - claimed
                relative_difference = abs(difference) / claimed
                
                if relative_difference > 0.05:  # More than 5% difference
                    severity = classify_severity(relative_difference)
                    suggestion = suggest_reconciliation(metric, claimed, actual)
                    
                    discrepancies.append(Discrepancy(
                        metric=metric,
                        claimed=claimed,
                        actual=actual,
                        difference=difference,
                        relative_difference=relative_difference,
                        severity=severity,
                        suggestion=suggestion
                    ))
        
        return discrepancies
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        summary = []
        summary.append("# Executive Summary")
        summary.append("")
        summary.append(f"**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"**Results Path**: {self.results_path}")
        summary.append(f"**Total Experiments**: {len(self.data)}")
        summary.append("")
        
        # Key metrics overview
        summary.append("## Key Performance Metrics")
        summary.append("")
        summary.append(f"- **False Accept Rate (FAR)**: {self.metrics.far:.4f}")
        summary.append(f"- **False Reject Rate (FRR)**: {self.metrics.frr:.4f}")
        summary.append(f"- **Overall Accuracy**: {self.metrics.accuracy:.4f}")
        summary.append(f"- **Average Queries**: {self.metrics.avg_queries:.1f}")
        summary.append(f"- **Total Queries**: {self.metrics.total_queries}")
        summary.append("")
        
        # Performance assessment
        overall_accuracy = 1 - (self.metrics.far + self.metrics.frr) / 2
        if overall_accuracy >= 0.95:
            assessment = "✅ **EXCELLENT** - Meets production standards"
        elif overall_accuracy >= 0.90:
            assessment = "🟡 **GOOD** - Suitable for most applications"
        elif overall_accuracy >= 0.80:
            assessment = "⚠️ **ACCEPTABLE** - May need optimization"
        else:
            assessment = "❌ **POOR** - Requires significant improvement"
        
        summary.append(f"**Overall Assessment**: {assessment}")
        summary.append("")
        
        # Discrepancy summary
        if self.discrepancies:
            major_count = sum(1 for d in self.discrepancies if d.severity == 'major')
            moderate_count = sum(1 for d in self.discrepancies if d.severity == 'moderate')
            minor_count = sum(1 for d in self.discrepancies if d.severity == 'minor')
            
            summary.append("## Discrepancy Analysis")
            summary.append("")
            summary.append(f"- **Major Discrepancies**: {major_count}")
            summary.append(f"- **Moderate Discrepancies**: {moderate_count}")
            summary.append(f"- **Minor Discrepancies**: {minor_count}")
            summary.append("")
            
            if major_count > 0:
                summary.append("⚠️ **Action Required**: Major discrepancies detected that require investigation.")
            elif moderate_count > 0:
                summary.append("📋 **Review Recommended**: Moderate discrepancies should be reviewed.")
            else:
                summary.append("✅ **Results Aligned**: Results generally align with expectations.")
        else:
            summary.append("✅ **No Significant Discrepancies**: Results align well with paper claims.")
        
        summary.append("")
        return "\n".join(summary)
    
    def generate_detailed_results(self) -> str:
        """Generate detailed per-challenge results section."""
        details = []
        details.append("# Detailed Results Analysis")
        details.append("")
        
        # Statistical overview
        details.append("## Statistical Summary")
        details.append("")
        
        if TABULATE_AVAILABLE:
            stats_data = [
                ["Metric", "Value", "Confidence Interval (95%)", "Paper Claim", "Status"],
                ["FAR", f"{self.metrics.far:.4f}", 
                 f"({self.metrics.confidence_intervals.get('far', (0, 0))[0]:.4f}, {self.metrics.confidence_intervals.get('far', (0, 0))[1]:.4f})",
                 f"{self.paper_claims.far:.4f}", 
                 "✅" if abs(self.metrics.far - self.paper_claims.far) < 0.05 else "⚠️"],
                ["FRR", f"{self.metrics.frr:.4f}",
                 f"({self.metrics.confidence_intervals.get('frr', (0, 0))[0]:.4f}, {self.metrics.confidence_intervals.get('frr', (0, 0))[1]:.4f})",
                 f"{self.paper_claims.frr:.4f}",
                 "✅" if abs(self.metrics.frr - self.paper_claims.frr) < 0.05 else "⚠️"],
                ["Accuracy", f"{self.metrics.accuracy:.4f}",
                 f"({self.metrics.confidence_intervals.get('accuracy', (0, 0))[0]:.4f}, {self.metrics.confidence_intervals.get('accuracy', (0, 0))[1]:.4f})",
                 f"{self.paper_claims.accuracy:.4f}",
                 "✅" if abs(self.metrics.accuracy - self.paper_claims.accuracy) < 0.05 else "⚠️"]
            ]
            
            details.append("```")
            details.append(tabulate(stats_data, headers="firstrow", tablefmt="grid"))
            details.append("```")
        else:
            details.append(f"- **FAR**: {self.metrics.far:.4f} (claimed: {self.paper_claims.far:.4f})")
            details.append(f"- **FRR**: {self.metrics.frr:.4f} (claimed: {self.paper_claims.frr:.4f})")
            details.append(f"- **Accuracy**: {self.metrics.accuracy:.4f} (claimed: {self.paper_claims.accuracy:.4f})")
        
        details.append("")
        
        # Query efficiency analysis
        details.append("## Query Efficiency Analysis")
        details.append("")
        details.append(f"- **Average Queries per Verification**: {self.metrics.avg_queries:.1f}")
        details.append(f"- **Total Queries Executed**: {self.metrics.total_queries}")
        details.append(f"- **Average Processing Time**: {self.metrics.processing_time:.3f}s")
        
        if self.metrics.avg_queries > 0 and self.paper_claims.average_queries > 0:
            efficiency = (self.paper_claims.average_queries - self.metrics.avg_queries) / self.paper_claims.average_queries
            details.append(f"- **Query Efficiency**: {efficiency:.1%} {'improvement' if efficiency > 0 else 'degradation'}")
        
        details.append("")
        
        # Raw data summary
        if self.data:
            details.append("## Raw Data Overview")
            details.append("")
            details.append(f"- **Total Result Records**: {len(self.data)}")
            
            # Identify unique experiments or challenge types
            experiment_types = set()
            challenge_families = set()
            
            for record in self.data:
                if 'experiment_type' in record:
                    experiment_types.add(record['experiment_type'])
                if 'challenge_family' in record:
                    challenge_families.add(record['challenge_family'])
            
            if experiment_types:
                details.append(f"- **Experiment Types**: {', '.join(sorted(experiment_types))}")
            if challenge_families:
                details.append(f"- **Challenge Families**: {', '.join(sorted(challenge_families))}")
        
        details.append("")
        return "\n".join(details)
    
    def generate_discrepancy_analysis(self) -> str:
        """Generate discrepancy analysis section."""
        analysis = []
        analysis.append("# Discrepancy Analysis")
        analysis.append("")
        
        if not self.discrepancies:
            analysis.append("✅ **No significant discrepancies detected.**")
            analysis.append("")
            analysis.append("All measured metrics align well with paper claims within acceptable tolerances.")
            return "\n".join(analysis)
        
        # Group by severity
        major = [d for d in self.discrepancies if d.severity == 'major']
        moderate = [d for d in self.discrepancies if d.severity == 'moderate']
        minor = [d for d in self.discrepancies if d.severity == 'minor']
        
        for severity, discrepancies in [('Major', major), ('Moderate', moderate), ('Minor', minor)]:
            if not discrepancies:
                continue
            
            emoji = "🔴" if severity == 'Major' else "🟡" if severity == 'Moderate' else "🟢"
            analysis.append(f"## {emoji} {severity} Discrepancies")
            analysis.append("")
            
            for d in discrepancies:
                analysis.append(f"### {d.metric.upper()}")
                analysis.append(f"- **Claimed**: {d.claimed:.4f}")
                analysis.append(f"- **Actual**: {d.actual:.4f}")
                analysis.append(f"- **Difference**: {d.difference:+.4f} ({d.relative_difference:.1%})")
                analysis.append(f"- **Recommendation**: {d.suggestion}")
                analysis.append("")
        
        return "\n".join(analysis)
    
    def generate_markdown_report(self) -> str:
        """Generate complete markdown report."""
        report = []
        
        # Title and metadata
        report.append("# PoT Experimental Results Report")
        report.append("")
        
        # Executive summary
        report.append(self.generate_executive_summary())
        
        # Detailed results
        report.append(self.generate_detailed_results())
        
        # Discrepancy analysis
        report.append(self.generate_discrepancy_analysis())
        
        # Visualization references
        report.append("# Visualizations")
        report.append("")
        report.append("The following plots have been generated:")
        report.append("")
        report.append("- `roc_curve.png` - ROC Curve Analysis")
        report.append("- `query_distribution.png` - Query Count Distribution")
        report.append("- `confidence_intervals.png` - Metric Confidence Intervals")
        report.append("- `performance_comparison.png` - Performance vs Claims Comparison")
        report.append("")
        
        # Save markdown report
        markdown_content = "\n".join(report)
        markdown_file = self.output_dir / "report.md"
        with open(markdown_file, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown report saved to {markdown_file}")
        return markdown_content
    
    def generate_plots(self) -> Dict[str, str]:
        """
        Generate comprehensive visualizations.
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        plots = {}
        
        # Set up matplotlib parameters
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        
        try:
            # 1. ROC Curve
            plots['roc_curve'] = self._generate_roc_curve()
            
            # 2. Query Distribution
            plots['query_distribution'] = self._generate_query_histogram()
            
            # 3. Confidence Intervals
            plots['confidence_intervals'] = self._generate_confidence_plot()
            
            # 4. Performance Comparison
            plots['performance_comparison'] = self._generate_comparison_plot()
            
            # 5. Challenge Difficulty Analysis
            plots['challenge_analysis'] = self._generate_challenge_analysis()
            
            # 6. Timeline Analysis (if timestamps available)
            plots['timeline_analysis'] = self._generate_timeline_plot()
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
        
        logger.info(f"Generated {len(plots)} visualization plots")
        return plots
    
    def _generate_roc_curve(self) -> str:
        """Generate ROC curve visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract FAR/FRR data points from results
        far_values = []
        frr_values = []
        thresholds = []
        
        for result in self.data:
            if 'far' in result and 'frr' in result:
                far_values.append(result['far'])
                frr_values.append(result['frr'])
                thresholds.append(result.get('threshold', 0.5))
        
        if len(far_values) > 1:
            # ROC Curve (TPR vs FPR)
            tpr = [1 - frr for frr in frr_values]
            ax1.plot(far_values, tpr, 'b-', linewidth=2, marker='o', markersize=4)
            ax1.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Random classifier')
            ax1.set_xlabel('False Accept Rate (FAR)')
            ax1.set_ylabel('True Positive Rate (1 - FRR)')
            ax1.set_title('ROC Curve')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # DET Curve (FAR vs FRR)
            ax2.plot(far_values, frr_values, 'g-', linewidth=2, marker='s', markersize=4)
            ax2.set_xlabel('False Accept Rate (FAR)')
            ax2.set_ylabel('False Reject Rate (FRR)')
            ax2.set_title('Detection Error Tradeoff (DET) Curve')
            ax2.grid(True, alpha=0.3)
        else:
            # Single point plot
            ax1.scatter([self.metrics.far], [1 - self.metrics.frr], c='blue', s=100, alpha=0.7)
            ax1.set_xlabel('False Accept Rate (FAR)')
            ax1.set_ylabel('True Positive Rate (1 - FRR)')
            ax1.set_title('ROC Point')
            ax1.grid(True, alpha=0.3)
            
            ax2.scatter([self.metrics.far], [self.metrics.frr], c='green', s=100, alpha=0.7)
            ax2.set_xlabel('False Accept Rate (FAR)')
            ax2.set_ylabel('False Reject Rate (FRR)')
            ax2.set_title('DET Point')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "roc_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _generate_query_histogram(self) -> str:
        """Generate query count distribution histogram."""
        query_counts = []
        
        for result in self.data:
            for key in ['queries', 'query_count', 'total_queries', 'n_queries']:
                if key in result:
                    query_counts.append(int(result[key]))
                    break
        
        if not query_counts:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No query data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Query Distribution')
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram
            ax1.hist(query_counts, bins=min(20, len(set(query_counts))), alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(np.mean(query_counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(query_counts):.1f}')
            ax1.axvline(np.median(query_counts), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(query_counts):.1f}')
            ax1.set_xlabel('Query Count')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Query Count Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(query_counts, vert=True)
            ax2.set_ylabel('Query Count')
            ax2.set_title('Query Count Box Plot')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "query_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _generate_confidence_plot(self) -> str:
        """Generate confidence interval visualization."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics_to_plot = ['far', 'frr', 'accuracy']
        values = []
        labels = []
        lower_errors = []
        upper_errors = []
        
        for i, metric in enumerate(metrics_to_plot):
            if hasattr(self.metrics, metric):
                value = getattr(self.metrics, metric)
                ci = self.metrics.confidence_intervals.get(metric, (value, value))
                
                values.append(value)
                labels.append(metric.upper())
                lower_errors.append(value - ci[0])
                upper_errors.append(ci[1] - value)
        
        if values:
            x_pos = np.arange(len(labels))
            ax.errorbar(x_pos, values, yerr=[lower_errors, upper_errors], 
                       fmt='o', capsize=5, capthick=2, markersize=8)
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Value')
            ax.set_title('Metrics with 95% Confidence Intervals')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.grid(True, alpha=0.3)
            
            # Add paper claims as horizontal lines
            paper_values = [self.paper_claims.far, self.paper_claims.frr, self.paper_claims.accuracy]
            for i, (paper_val, label) in enumerate(zip(paper_values, labels)):
                ax.axhline(y=paper_val, xmin=(i-0.3)/len(labels), xmax=(i+0.3)/len(labels), 
                          color='red', linestyle='--', alpha=0.7, linewidth=2)
        else:
            ax.text(0.5, 0.5, 'No confidence interval data available', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        output_path = self.output_dir / "confidence_intervals.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _generate_comparison_plot(self) -> str:
        """Generate performance comparison with paper claims."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_data = [
            ('FAR', self.paper_claims.far, self.metrics.far),
            ('FRR', self.paper_claims.frr, self.metrics.frr),
            ('Accuracy', self.paper_claims.accuracy, self.metrics.accuracy)
        ]
        
        metrics = [item[0] for item in metrics_data]
        claimed = [item[1] for item in metrics_data]
        actual = [item[2] for item in metrics_data]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, claimed, width, label='Paper Claims', alpha=0.8, color='lightcoral')
        bars2 = ax.bar(x + width/2, actual, width, label='Actual Results', alpha=0.8, color='skyblue')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Performance: Claims vs Actual Results')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        plt.tight_layout()
        output_path = self.output_dir / "performance_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _generate_challenge_analysis(self) -> str:
        """Generate challenge difficulty vs performance analysis."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract challenge-specific data if available
        challenge_families = {}
        
        for result in self.data:
            family = result.get('challenge_family', 'unknown')
            if family not in challenge_families:
                challenge_families[family] = {'accuracy': [], 'queries': []}
            
            if 'accuracy' in result:
                challenge_families[family]['accuracy'].append(result['accuracy'])
            
            for key in ['queries', 'query_count', 'total_queries', 'n_queries']:
                if key in result:
                    challenge_families[family]['queries'].append(result[key])
                    break
        
        if len(challenge_families) > 1:
            families = list(challenge_families.keys())
            avg_accuracies = [np.mean(challenge_families[f]['accuracy']) if challenge_families[f]['accuracy'] else 0 
                            for f in families]
            avg_queries = [np.mean(challenge_families[f]['queries']) if challenge_families[f]['queries'] else 0 
                          for f in families]
            
            scatter = ax.scatter(avg_queries, avg_accuracies, s=100, alpha=0.7)
            
            # Add labels for each point
            for i, family in enumerate(families):
                ax.annotate(family, (avg_queries[i], avg_accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel('Average Queries')
            ax.set_ylabel('Average Accuracy')
            ax.set_title('Challenge Family Performance Analysis')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient challenge family data for analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Challenge Analysis')
        
        plt.tight_layout()
        output_path = self.output_dir / "challenge_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _generate_timeline_plot(self) -> str:
        """Generate timeline analysis if timestamp data is available."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Look for timestamp data
        timestamps = []
        values = []
        
        for result in self.data:
            if 'timestamp' in result:
                try:
                    timestamp = pd.to_datetime(result['timestamp'])
                    timestamps.append(timestamp)
                    values.append(result.get('accuracy', 0))
                except:
                    continue
        
        if len(timestamps) > 1:
            # Sort by timestamp
            sorted_data = sorted(zip(timestamps, values))
            timestamps, values = zip(*sorted_data)
            
            ax.plot(timestamps, values, 'b-', marker='o', markersize=4, linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Accuracy')
            ax.set_title('Performance Over Time')
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
        else:
            ax.text(0.5, 0.5, 'No timestamp data available for timeline analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Timeline Analysis')
        
        plt.tight_layout()
        output_path = self.output_dir / "timeline_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def generate_latex_tables(self) -> str:
        """
        Generate LaTeX tables for paper-ready output.
        
        Returns:
            LaTeX table code as string
        """
        latex_content = []
        
        # Main results table
        latex_content.append("% Main Results Table")
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Experimental Results Summary}")
        latex_content.append("\\label{tab:results}")
        latex_content.append("\\begin{tabular}{lcccc}")
        latex_content.append("\\toprule")
        latex_content.append("Metric & Actual & Paper Claim & Difference & Status \\\\")
        latex_content.append("\\midrule")
        
        # Results data
        results_data = [
            ("FAR", self.metrics.far, self.paper_claims.far),
            ("FRR", self.metrics.frr, self.paper_claims.frr),
            ("Accuracy", self.metrics.accuracy, self.paper_claims.accuracy),
            ("Avg. Queries", self.metrics.avg_queries, self.paper_claims.average_queries)
        ]
        
        for metric, actual, claimed in results_data:
            if actual > 0 and claimed > 0:
                difference = actual - claimed
                rel_diff = abs(difference) / claimed
                status = "\\checkmark" if rel_diff < 0.1 else "\\times"
                latex_content.append(f"{metric} & {actual:.4f} & {claimed:.4f} & {difference:+.4f} & {status} \\\\")
        
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        latex_content.append("")
        
        # Confidence intervals table
        latex_content.append("% Confidence Intervals Table")
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{95\\% Confidence Intervals}")
        latex_content.append("\\label{tab:confidence}")
        latex_content.append("\\begin{tabular}{lccc}")
        latex_content.append("\\toprule")
        latex_content.append("Metric & Mean & Lower Bound & Upper Bound \\\\")
        latex_content.append("\\midrule")
        
        for metric in ['far', 'frr', 'accuracy']:
            if metric in self.metrics.confidence_intervals:
                value = getattr(self.metrics, metric)
                ci = self.metrics.confidence_intervals[metric]
                latex_content.append(f"{metric.upper()} & {value:.4f} & {ci[0]:.4f} & {ci[1]:.4f} \\\\")
        
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        latex_content.append("")
        
        # Discrepancies table if any exist
        if self.discrepancies:
            latex_content.append("% Discrepancies Table")
            latex_content.append("\\begin{table}[htbp]")
            latex_content.append("\\centering")
            latex_content.append("\\caption{Identified Discrepancies}")
            latex_content.append("\\label{tab:discrepancies}")
            latex_content.append("\\begin{tabular}{lccl}")
            latex_content.append("\\toprule")
            latex_content.append("Metric & Rel. Difference & Severity & Recommendation \\\\")
            latex_content.append("\\midrule")
            
            for d in self.discrepancies:
                severity_symbol = "\\textcolor{red}{Major}" if d.severity == 'major' else \
                                "\\textcolor{orange}{Moderate}" if d.severity == 'moderate' else \
                                "\\textcolor{blue}{Minor}"
                
                # Truncate recommendation for table
                rec = d.suggestion[:40] + "..." if len(d.suggestion) > 40 else d.suggestion
                latex_content.append(f"{d.metric.upper()} & {d.relative_difference:.1%} & {severity_symbol} & \\parbox{{3cm}}{{{rec}}} \\\\")
            
            latex_content.append("\\bottomrule")
            latex_content.append("\\end{tabular}")
            latex_content.append("\\end{table}")
        
        latex_code = "\n".join(latex_content)
        
        # Save LaTeX tables
        latex_file = self.output_dir / "tables.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_code)
        
        logger.info(f"LaTeX tables saved to {latex_file}")
        return latex_code
    
    def generate_html_report(self) -> str:
        """Generate HTML report with interactive plots."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PoT Experimental Results Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .status-good {{ color: #27ae60; }}
        .status-warning {{ color: #f39c12; }}
        .status-error {{ color: #e74c3c; }}
        .plot-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #34495e;
            color: white;
        }}
        .discrepancy-major {{ background-color: #fadbd8; }}
        .discrepancy-moderate {{ background-color: #fef9e7; }}
        .discrepancy-minor {{ background-color: #d5f4e6; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 PoT Experimental Results Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Results Path:</strong> {self.results_path}</p>
        
        <h2>📊 Key Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>False Accept Rate</h3>
                <div class="metric-value">{self.metrics.far:.4f}</div>
                <small>Target: {self.paper_claims.far:.4f}</small>
            </div>
            <div class="metric-card">
                <h3>False Reject Rate</h3>
                <div class="metric-value">{self.metrics.frr:.4f}</div>
                <small>Target: {self.paper_claims.frr:.4f}</small>
            </div>
            <div class="metric-card">
                <h3>Overall Accuracy</h3>
                <div class="metric-value">{self.metrics.accuracy:.4f}</div>
                <small>Target: {self.paper_claims.accuracy:.4f}</small>
            </div>
            <div class="metric-card">
                <h3>Average Queries</h3>
                <div class="metric-value">{self.metrics.avg_queries:.1f}</div>
                <small>Target: {self.paper_claims.average_queries:.1f}</small>
            </div>
        </div>
        
        <h2>📈 Visualizations</h2>
        <div class="plot-container">
            <h3>ROC Curve Analysis</h3>
            <img src="roc_curve.png" alt="ROC Curve">
        </div>
        
        <div class="plot-container">
            <h3>Query Distribution</h3>
            <img src="query_distribution.png" alt="Query Distribution">
        </div>
        
        <div class="plot-container">
            <h3>Confidence Intervals</h3>
            <img src="confidence_intervals.png" alt="Confidence Intervals">
        </div>
        
        <div class="plot-container">
            <h3>Performance Comparison</h3>
            <img src="performance_comparison.png" alt="Performance Comparison">
        </div>
"""
        
        # Add discrepancy table if any exist
        if self.discrepancies:
            html_content += """
        <h2>⚠️ Discrepancy Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Claimed</th>
                    <th>Actual</th>
                    <th>Difference</th>
                    <th>Severity</th>
                    <th>Recommendation</th>
                </tr>
            </thead>
            <tbody>"""
            
            for d in self.discrepancies:
                row_class = f"discrepancy-{d.severity}"
                html_content += f"""
                <tr class="{row_class}">
                    <td><strong>{d.metric.upper()}</strong></td>
                    <td>{d.claimed:.4f}</td>
                    <td>{d.actual:.4f}</td>
                    <td>{d.difference:+.4f} ({d.relative_difference:.1%})</td>
                    <td>{d.severity.title()}</td>
                    <td>{d.suggestion}</td>
                </tr>"""
            
            html_content += """
            </tbody>
        </table>"""
        
        html_content += """
    </div>
</body>
</html>"""
        
        # Save HTML report
        html_file = self.output_dir / "report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {html_file}")
        return html_content
    
    def generate_json_export(self) -> str:
        """Generate JSON export for programmatic access."""
        json_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "results_path": str(self.results_path),
                "total_experiments": len(self.data),
                "generator_version": "1.0"
            },
            "metrics": {
                "far": self.metrics.far,
                "frr": self.metrics.frr, 
                "accuracy": self.metrics.accuracy,
                "precision": self.metrics.precision,
                "recall": self.metrics.recall,
                "f1_score": self.metrics.f1_score,
                "total_queries": self.metrics.total_queries,
                "avg_queries": self.metrics.avg_queries,
                "processing_time": self.metrics.processing_time,
                "confidence_intervals": self.metrics.confidence_intervals
            },
            "paper_claims": {
                "far": self.paper_claims.far,
                "frr": self.paper_claims.frr,
                "accuracy": self.paper_claims.accuracy,
                "efficiency_gain": self.paper_claims.efficiency_gain,
                "average_queries": self.paper_claims.average_queries,
                "confidence_level": self.paper_claims.confidence_level
            },
            "discrepancies": [
                {
                    "metric": d.metric,
                    "claimed": d.claimed,
                    "actual": d.actual,
                    "difference": d.difference,
                    "relative_difference": d.relative_difference,
                    "severity": d.severity,
                    "suggestion": d.suggestion
                }
                for d in self.discrepancies
            ],
            "raw_data": self.data
        }
        
        # Save JSON export
        json_file = self.output_dir / "report_data.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info(f"JSON export saved to {json_file}")
        return json.dumps(json_data, indent=2, default=str)
    
    def generate_all_reports(self) -> Dict[str, str]:
        """
        Generate all report formats.
        
        Returns:
            Dictionary mapping format names to file paths
        """
        reports = {}
        
        logger.info("Generating comprehensive report suite...")
        
        # Generate all plots first
        plots = self.generate_plots()
        
        # Generate all report formats
        reports['markdown'] = self.generate_markdown_report()
        reports['latex'] = self.generate_latex_tables()
        reports['html'] = self.generate_html_report()
        reports['json'] = self.generate_json_export()
        
        # Add plot file paths
        reports.update(plots)
        
        # Create summary index
        index_content = f"""
# PoT Experimental Results Report Suite

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Available Reports

- **Markdown Report**: `report.md` - Complete textual analysis
- **HTML Report**: `report.html` - Interactive web-based report  
- **LaTeX Tables**: `tables.tex` - Paper-ready tables
- **JSON Export**: `report_data.json` - Machine-readable data

## Visualizations

- **ROC Curve**: `roc_curve.png` - Performance analysis
- **Query Distribution**: `query_distribution.png` - Efficiency analysis
- **Confidence Intervals**: `confidence_intervals.png` - Statistical reliability
- **Performance Comparison**: `performance_comparison.png` - Claims vs results
- **Challenge Analysis**: `challenge_analysis.png` - Per-challenge breakdown
- **Timeline Analysis**: `timeline_analysis.png` - Performance over time

## Quick Stats

- **Total Experiments**: {len(self.data)}
- **Overall Accuracy**: {self.metrics.accuracy:.4f}
- **Average Queries**: {self.metrics.avg_queries:.1f}
- **Major Discrepancies**: {sum(1 for d in self.discrepancies if d.severity == 'major')}

Open `report.html` for the best viewing experience.
"""
        
        index_file = self.output_dir / "README.md"
        with open(index_file, 'w') as f:
            f.write(index_content)
        
        reports['index'] = str(index_file)
        
        logger.info(f"Generated complete report suite in {self.output_dir}")
        logger.info(f"Total files: {len(reports)}")
        
        return reports


# Example usage and testing functions
def create_sample_results(output_path: str) -> None:
    """Create sample results for testing the report generator."""
    sample_data = [
        {
            "experiment_id": "exp_001",
            "far": 0.008,
            "frr": 0.012,
            "accuracy": 0.994,
            "queries": 8,
            "processing_time": 0.234,
            "challenge_family": "vision:freq",
            "timestamp": "2024-01-15T10:30:00"
        },
        {
            "experiment_id": "exp_002", 
            "far": 0.006,
            "frr": 0.015,
            "accuracy": 0.991,
            "queries": 12,
            "processing_time": 0.287,
            "challenge_family": "vision:texture",
            "timestamp": "2024-01-15T10:35:00"
        },
        {
            "experiment_id": "exp_003",
            "far": 0.011,
            "frr": 0.009,
            "accuracy": 0.996,
            "queries": 6,
            "processing_time": 0.198,
            "challenge_family": "lm:templates",
            "timestamp": "2024-01-15T10:40:00"
        }
    ]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample results created at {output_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        # Create sample data for testing
        sample_path = "test_results.json"
        create_sample_results(sample_path)
        results_path = sample_path
    
    # Generate reports
    generator = ReportGenerator(results_path)
    reports = generator.generate_all_reports()
    
    print(f"\nGenerated reports in: {generator.output_dir}")
    for report_type, path in reports.items():
        print(f"  {report_type}: {Path(path).name}")
    
    print(f"\nOpen {generator.output_dir}/report.html for best viewing experience.")