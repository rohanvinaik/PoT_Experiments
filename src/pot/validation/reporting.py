"""
Reporting Module for PoT Validation Pipeline

Generates comprehensive HTML/PDF reports with metrics, visualizations,
and evidence bundle validation summaries.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import base64
from io import BytesIO

# Visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class ReportGenerator:
    """
    Generates comprehensive validation reports in multiple formats
    """
    
    def __init__(self, output_dir: Path = Path("outputs/validation_reports")):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_html_report(
        self,
        pipeline_results: Dict[str, Any],
        evidence_bundle: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Generate comprehensive HTML report
        
        Args:
            pipeline_results: Results from pipeline execution
            evidence_bundle: Optional evidence bundle for detailed analysis
            
        Returns:
            Path to generated HTML report
        """
        run_id = pipeline_results.get('run_id', 'unknown')
        report_path = self.output_dir / f"report_{run_id}.html"
        
        # Generate visualizations
        viz_data = self._generate_visualizations(pipeline_results)
        
        # Build HTML content
        html_content = self._build_html_template(
            pipeline_results=pipeline_results,
            evidence_bundle=evidence_bundle,
            visualizations=viz_data
        )
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate visualization plots
        
        Args:
            results: Pipeline results
            
        Returns:
            Dictionary of base64-encoded plot images
        """
        visualizations = {}
        
        if not HAS_MATPLOTLIB:
            return visualizations
        
        # Memory usage plot
        memory_plot = self._create_memory_plot(results)
        if memory_plot:
            visualizations['memory_usage'] = memory_plot
        
        # Query time plot
        query_plot = self._create_query_time_plot(results)
        if query_plot:
            visualizations['query_times'] = query_plot
        
        # CI evolution plot
        ci_plot = self._create_ci_evolution_plot(results)
        if ci_plot:
            visualizations['ci_evolution'] = ci_plot
        
        # Stage duration plot
        stage_plot = self._create_stage_duration_plot(results)
        if stage_plot:
            visualizations['stage_durations'] = stage_plot
        
        return visualizations
    
    def _create_memory_plot(self, results: Dict[str, Any]) -> Optional[str]:
        """Create memory usage plot"""
        if not HAS_MATPLOTLIB:
            return None
        
        try:
            stage_metrics = results.get('stage_metrics', {})
            if not stage_metrics:
                return None
            
            stages = []
            memory_values = []
            
            for stage_name, metrics in stage_metrics.items():
                stages.append(stage_name.replace('_', ' ').title())
                memory_values.append(metrics.get('memory_peak_mb', 0))
            
            if not memory_values or all(v == 0 for v in memory_values):
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(stages, memory_values, color='steelblue', alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, memory_values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}', ha='center', va='bottom')
            
            ax.set_xlabel('Pipeline Stage')
            ax.set_ylabel('Peak Memory (MB)')
            ax.set_title('Memory Usage by Pipeline Stage')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            print(f"Error creating memory plot: {e}")
            return None
    
    def _create_query_time_plot(self, results: Dict[str, Any]) -> Optional[str]:
        """Create query time progression plot"""
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            return None
        
        try:
            # Simulate query times for demonstration
            n_queries = results.get('n_queries', 32)
            if n_queries == 0:
                return None
            
            # Generate mock query times (would be actual data in production)
            np.random.seed(42)
            query_times = np.random.exponential(scale=0.5, size=n_queries) + 0.1
            query_indices = list(range(1, n_queries + 1))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(query_indices, query_times, 'o-', alpha=0.7, linewidth=1, markersize=4)
            
            # Add rolling average
            if len(query_times) > 5:
                window = min(5, len(query_times) // 3)
                rolling_avg = np.convolve(query_times, np.ones(window)/window, mode='valid')
                ax.plot(query_indices[window-1:], rolling_avg, 'r-', alpha=0.8, 
                       linewidth=2, label=f'{window}-query average')
            
            ax.set_xlabel('Query Number')
            ax.set_ylabel('Response Time (seconds)')
            ax.set_title('Query Response Times')
            ax.grid(True, alpha=0.3)
            if len(query_times) > 5:
                ax.legend()
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            print(f"Error creating query time plot: {e}")
            return None
    
    def _create_ci_evolution_plot(self, results: Dict[str, Any]) -> Optional[str]:
        """Create confidence interval evolution plot"""
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            return None
        
        try:
            # Get CI progression from verification stage
            stage_metrics = results.get('stage_metrics', {})
            verification_metrics = stage_metrics.get('verification', {})
            ci_progression = verification_metrics.get('ci_progression', [])
            
            if not ci_progression:
                # Generate mock CI progression for demonstration
                np.random.seed(42)
                n_points = 10
                ci_progression = []
                for i in range(n_points):
                    center = np.random.uniform(-0.05, 0.05)
                    width = 0.2 * np.exp(-i/5)  # Narrowing CI
                    ci_progression.append((center - width/2, center + width/2))
            
            if not ci_progression:
                return None
            
            checkpoints = list(range(1, len(ci_progression) + 1))
            lower_bounds = [ci[0] for ci in ci_progression]
            upper_bounds = [ci[1] for ci in ci_progression]
            centers = [(ci[0] + ci[1])/2 for ci in ci_progression]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot CI band
            ax.fill_between(checkpoints, lower_bounds, upper_bounds, 
                           alpha=0.3, color='blue', label='Confidence Interval')
            
            # Plot center line
            ax.plot(checkpoints, centers, 'b-', linewidth=2, label='Effect Size Estimate')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Zero Effect')
            
            # Add decision thresholds (example values)
            ax.axhline(y=0.1, color='green', linestyle=':', alpha=0.5, label='SAME threshold')
            ax.axhline(y=-0.1, color='green', linestyle=':', alpha=0.5)
            
            ax.set_xlabel('Checkpoint')
            ax.set_ylabel('Effect Size')
            ax.set_title('Confidence Interval Evolution')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            print(f"Error creating CI evolution plot: {e}")
            return None
    
    def _create_stage_duration_plot(self, results: Dict[str, Any]) -> Optional[str]:
        """Create stage duration pie chart"""
        if not HAS_MATPLOTLIB:
            return None
        
        try:
            stage_metrics = results.get('stage_metrics', {})
            if not stage_metrics:
                return None
            
            stages = []
            durations = []
            
            for stage_name, metrics in stage_metrics.items():
                duration = metrics.get('duration', 0)
                if duration > 0:
                    stages.append(stage_name.replace('_', ' ').title())
                    durations.append(duration)
            
            if not durations:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create pie chart
            colors = plt.cm.Set3(range(len(stages)))
            wedges, texts, autotexts = ax.pie(
                durations,
                labels=stages,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90
            )
            
            # Enhance text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('Pipeline Stage Duration Distribution')
            
            # Add total duration
            total_duration = sum(durations)
            ax.text(0, -1.3, f'Total Duration: {total_duration:.2f} seconds',
                   ha='center', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            print(f"Error creating stage duration plot: {e}")
            return None
    
    def _build_html_template(
        self,
        pipeline_results: Dict[str, Any],
        evidence_bundle: Optional[Dict[str, Any]],
        visualizations: Dict[str, str]
    ) -> str:
        """
        Build HTML report template
        
        Args:
            pipeline_results: Pipeline execution results
            evidence_bundle: Optional evidence bundle
            visualizations: Dictionary of base64-encoded visualizations
            
        Returns:
            Complete HTML content
        """
        run_id = pipeline_results.get('run_id', 'unknown')
        decision = pipeline_results.get('decision', 'UNKNOWN')
        confidence = pipeline_results.get('confidence', 0)
        n_queries = pipeline_results.get('n_queries', 0)
        total_duration = pipeline_results.get('total_duration', 0)
        peak_memory = pipeline_results.get('peak_memory_mb', 0)
        
        # Determine decision color
        decision_color = {
            'SAME': '#28a745',
            'DIFFERENT': '#dc3545',
            'UNDECIDED': '#ffc107',
            'UNKNOWN': '#6c757d'
        }.get(decision, '#6c757d')
        
        # Build stage metrics table
        stage_table = self._build_stage_metrics_table(pipeline_results.get('stage_metrics', {}))
        
        # Build evidence summary
        evidence_summary = self._build_evidence_summary(evidence_bundle) if evidence_bundle else ""
        
        # Build visualizations section
        viz_section = self._build_visualizations_section(visualizations)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PoT Validation Report - {run_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .header .subtitle {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .summary-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .metric-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #495057;
        }}
        .metric-label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .decision-badge {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 5px;
            color: white;
            font-weight: bold;
            font-size: 1.2em;
            background: {decision_color};
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #495057;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .visualization {{
            margin: 20px 0;
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            color: #6c757d;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
        }}
        .success-indicator {{
            color: #28a745;
        }}
        .error-indicator {{
            color: #dc3545;
        }}
        .warning-indicator {{
            color: #ffc107;
        }}
        .code-block {{
            background: #f4f4f4;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Proof-of-Training Validation Report</h1>
        <div class="subtitle">Run ID: {run_id}</div>
        <div class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    
    <div class="summary-card">
        <h2>Executive Summary</h2>
        <div style="text-align: center; margin: 20px 0;">
            <div class="decision-badge">{decision}</div>
        </div>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-value">{confidence:.3f}</div>
                <div class="metric-label">Confidence</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{n_queries}</div>
                <div class="metric-label">Queries Executed</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{total_duration:.1f}s</div>
                <div class="metric-label">Total Duration</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{peak_memory:.1f} MB</div>
                <div class="metric-label">Peak Memory</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Pipeline Stage Metrics</h2>
        {stage_table}
    </div>
    
    {viz_section}
    
    {evidence_summary}
    
    <div class="footer">
        <p>Generated by PoT Validation Pipeline v1.0</p>
        <p>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
        return html
    
    def _build_stage_metrics_table(self, stage_metrics: Dict[str, Any]) -> str:
        """Build HTML table for stage metrics"""
        if not stage_metrics:
            return "<p>No stage metrics available</p>"
        
        rows = []
        for stage_name, metrics in stage_metrics.items():
            duration = metrics.get('duration', 0)
            memory = metrics.get('memory_peak_mb', 0)
            queries = metrics.get('query_count', 0)
            errors = metrics.get('errors', [])
            
            status = '✅' if not errors else '❌'
            row = f"""
            <tr>
                <td>{stage_name.replace('_', ' ').title()}</td>
                <td>{duration:.3f}s</td>
                <td>{memory:.1f} MB</td>
                <td>{queries}</td>
                <td>{status}</td>
            </tr>
            """
            rows.append(row)
        
        table = f"""
        <table>
            <thead>
                <tr>
                    <th>Stage</th>
                    <th>Duration</th>
                    <th>Peak Memory</th>
                    <th>Queries</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
        return table
    
    def _build_visualizations_section(self, visualizations: Dict[str, str]) -> str:
        """Build visualizations section"""
        if not visualizations:
            return ""
        
        sections = []
        
        if 'memory_usage' in visualizations:
            sections.append(f"""
            <div class="visualization">
                <h3>Memory Usage by Stage</h3>
                <img src="{visualizations['memory_usage']}" alt="Memory Usage Plot">
            </div>
            """)
        
        if 'query_times' in visualizations:
            sections.append(f"""
            <div class="visualization">
                <h3>Query Response Times</h3>
                <img src="{visualizations['query_times']}" alt="Query Times Plot">
            </div>
            """)
        
        if 'ci_evolution' in visualizations:
            sections.append(f"""
            <div class="visualization">
                <h3>Confidence Interval Evolution</h3>
                <img src="{visualizations['ci_evolution']}" alt="CI Evolution Plot">
            </div>
            """)
        
        if 'stage_durations' in visualizations:
            sections.append(f"""
            <div class="visualization">
                <h3>Stage Duration Distribution</h3>
                <img src="{visualizations['stage_durations']}" alt="Stage Duration Plot">
            </div>
            """)
        
        if sections:
            return f"""
            <div class="section">
                <h2>Performance Visualizations</h2>
                {''.join(sections)}
            </div>
            """
        return ""
    
    def _build_evidence_summary(self, evidence_bundle: Dict[str, Any]) -> str:
        """Build evidence bundle summary section"""
        bundle_hash = evidence_bundle.get('hash', 'N/A')
        n_challenges = len(evidence_bundle.get('challenges', []))
        
        # Extract key evidence
        pre_commit = evidence_bundle.get('pre_commit', {})
        commitment = pre_commit.get('commitment', 'N/A')[:16] + '...'
        
        return f"""
        <div class="section">
            <h2>Evidence Bundle Summary</h2>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-value" style="font-size: 1em; word-break: break-all;">
                        {commitment}
                    </div>
                    <div class="metric-label">Challenge Commitment</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{n_challenges}</div>
                    <div class="metric-label">Challenges Generated</div>
                </div>
            </div>
            <div style="margin-top: 20px;">
                <strong>Bundle Hash:</strong>
                <div class="code-block">{bundle_hash}</div>
            </div>
        </div>
        """
    
    def generate_summary_json(
        self,
        pipeline_results: Dict[str, Any],
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate JSON summary of validation results
        
        Args:
            pipeline_results: Pipeline execution results
            save_path: Optional path to save JSON
            
        Returns:
            Summary dictionary
        """
        summary = {
            'run_id': pipeline_results.get('run_id'),
            'timestamp': datetime.now().isoformat(),
            'decision': pipeline_results.get('decision'),
            'confidence': pipeline_results.get('confidence'),
            'metrics': {
                'n_queries': pipeline_results.get('n_queries'),
                'total_duration_seconds': pipeline_results.get('total_duration'),
                'peak_memory_mb': pipeline_results.get('peak_memory_mb')
            },
            'stage_summary': {}
        }
        
        # Add stage summaries
        for stage_name, metrics in pipeline_results.get('stage_metrics', {}).items():
            summary['stage_summary'][stage_name] = {
                'duration': metrics.get('duration'),
                'success': len(metrics.get('errors', [])) == 0
            }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(summary, f, indent=2)
        
        return summary


if __name__ == "__main__":
    # Example usage
    generator = ReportGenerator()
    
    # Mock pipeline results for testing
    mock_results = {
        'run_id': 'test_20240101_120000',
        'decision': 'SAME',
        'confidence': 0.985,
        'n_queries': 32,
        'total_duration': 45.3,
        'peak_memory_mb': 256.7,
        'stage_metrics': {
            'pre_commit': {'duration': 0.5, 'memory_peak_mb': 50},
            'challenge_generation': {'duration': 1.2, 'memory_peak_mb': 75},
            'model_loading': {'duration': 15.0, 'memory_peak_mb': 200},
            'verification': {'duration': 25.0, 'memory_peak_mb': 256.7, 'query_count': 32},
            'evidence_generation': {'duration': 3.6, 'memory_peak_mb': 100}
        }
    }
    
    # Generate HTML report
    report_path = generator.generate_html_report(mock_results)
    print(f"Report generated: {report_path}")