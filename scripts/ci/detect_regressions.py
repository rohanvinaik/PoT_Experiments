#!/usr/bin/env python3
"""
Performance Regression Detection

Detects performance regressions by comparing current benchmark results
with historical baselines using statistical analysis.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import statistics
import math

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class RegressionDetector:
    """Detects performance regressions using statistical analysis"""
    
    def __init__(self, threshold: float = 5.0, confidence_level: float = 0.95):
        self.threshold = threshold  # Percentage threshold for regression
        self.confidence_level = confidence_level
        self.z_score = self._calculate_z_score(confidence_level)
        
    def _calculate_z_score(self, confidence_level: float) -> float:
        """Calculate z-score for given confidence level"""
        # Approximate z-scores for common confidence levels
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        return z_scores.get(confidence_level, 1.96)
    
    def detect_regressions(self, current_file: str, historical_file: str) -> Dict[str, Any]:
        """Detect regressions by comparing current vs historical results"""
        
        # Load data files
        current_data = self._load_benchmark_data(current_file)
        historical_data = self._load_benchmark_data(historical_file)
        
        # Extract metrics
        current_metrics = self._extract_metrics(current_data)
        historical_metrics = self._extract_metrics(historical_data)
        
        # Perform regression analysis
        regression_analysis = {
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'current_file': current_file,
            'historical_file': historical_file,
            'threshold_percent': self.threshold,
            'confidence_level': self.confidence_level,
            'regression_detected': False,
            'regressions': [],
            'improvements': [],
            'stable_metrics': [],
            'summary': {}
        }
        
        # Compare common metrics
        common_metrics = set(current_metrics.keys()) & set(historical_metrics.keys())
        
        for metric_name in common_metrics:
            current_value = current_metrics[metric_name]
            historical_value = historical_metrics[metric_name]
            
            comparison = self._compare_metric(
                metric_name, current_value, historical_value
            )
            
            if comparison['is_regression']:
                regression_analysis['regressions'].append(comparison)
                regression_analysis['regression_detected'] = True
            elif comparison['is_improvement']:
                regression_analysis['improvements'].append(comparison)
            else:
                regression_analysis['stable_metrics'].append(comparison)
        
        # Generate summary
        regression_analysis['summary'] = self._generate_summary(regression_analysis)
        
        return regression_analysis
    
    def _load_benchmark_data(self, file_path: str) -> Dict[str, Any]:
        """Load benchmark data from JSON file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {file_path}")
        
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
    
    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric metrics from benchmark data"""
        metrics = {}
        
        def extract_recursive(obj: Any, prefix: str = ""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}_{key}" if prefix else key
                    if isinstance(value, (int, float)):
                        metrics[new_key] = float(value)
                    elif isinstance(value, dict):
                        extract_recursive(value, new_key)
                    elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                        # Aggregate list of numbers
                        metrics[f"{new_key}_mean"] = statistics.mean(value)
                        metrics[f"{new_key}_median"] = statistics.median(value)
                        if len(value) > 1:
                            metrics[f"{new_key}_std"] = statistics.stdev(value)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_recursive(item, f"{prefix}_{i}")
        
        extract_recursive(data)
        return metrics
    
    def _compare_metric(self, metric_name: str, current_value: float, 
                       historical_value: float) -> Dict[str, Any]:
        """Compare a single metric and determine if it's a regression"""
        
        # Calculate change percentage
        if historical_value == 0:
            change_percent = float('inf') if current_value > 0 else 0.0
        else:
            change_percent = ((current_value - historical_value) / historical_value) * 100
        
        # Determine if lower values are better for this metric
        lower_is_better = self._is_lower_better(metric_name)
        
        # Classify the change
        is_regression = False
        is_improvement = False
        significance = 'none'
        
        abs_change = abs(change_percent)
        
        if abs_change >= self.threshold:
            if lower_is_better:
                is_regression = change_percent > self.threshold
                is_improvement = change_percent < -self.threshold
            else:
                is_regression = change_percent < -self.threshold
                is_improvement = change_percent > self.threshold
            
            # Determine significance level
            if abs_change >= 50:
                significance = 'critical'
            elif abs_change >= 25:
                significance = 'high'
            elif abs_change >= 15:
                significance = 'medium'
            else:
                significance = 'low'
        
        return {
            'metric': metric_name,
            'current_value': current_value,
            'historical_value': historical_value,
            'change_percent': change_percent,
            'abs_change_percent': abs_change,
            'lower_is_better': lower_is_better,
            'is_regression': is_regression,
            'is_improvement': is_improvement,
            'significance': significance,
            'threshold': self.threshold
        }
    
    def _is_lower_better(self, metric_name: str) -> bool:
        """Determine if lower values are better for a metric"""
        metric_lower = metric_name.lower()
        
        # Lower is better keywords
        lower_keywords = [
            'time', 'duration', 'latency', 'delay', 'response_time',
            'memory', 'ram', 'cpu', 'usage', 'utilization',
            'size', 'bytes', 'mb', 'gb', 'kb',
            'error', 'failure', 'cost', 'overhead'
        ]
        
        # Higher is better keywords
        higher_keywords = [
            'accuracy', 'precision', 'recall', 'f1', 'score',
            'success', 'rate', 'percentage', 'ratio',
            'confidence', 'power', 'throughput', 'qps',
            'fps', 'bandwidth', 'coverage'
        ]
        
        # Check for exact or partial matches
        for keyword in lower_keywords:
            if keyword in metric_lower:
                return True
        
        for keyword in higher_keywords:
            if keyword in metric_lower:
                return False
        
        # Default assumption: lower is better
        return True
    
    def _generate_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics"""
        regressions = analysis['regressions']
        improvements = analysis['improvements']
        stable_metrics = analysis['stable_metrics']
        
        total_metrics = len(regressions) + len(improvements) + len(stable_metrics)
        
        summary = {
            'total_metrics_compared': total_metrics,
            'regressions_count': len(regressions),
            'improvements_count': len(improvements),
            'stable_metrics_count': len(stable_metrics),
            'regression_rate': len(regressions) / total_metrics if total_metrics > 0 else 0,
            'improvement_rate': len(improvements) / total_metrics if total_metrics > 0 else 0
        }
        
        # Regression severity breakdown
        if regressions:
            severity_counts = {}
            max_regression = 0
            worst_metric = None
            
            for reg in regressions:
                severity = reg['significance']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                if reg['abs_change_percent'] > max_regression:
                    max_regression = reg['abs_change_percent']
                    worst_metric = reg['metric']
            
            summary['regression_severity'] = severity_counts
            summary['worst_regression'] = {
                'metric': worst_metric,
                'change_percent': max_regression
            }
        
        # Improvement statistics
        if improvements:
            max_improvement = 0
            best_metric = None
            
            for imp in improvements:
                if imp['abs_change_percent'] > max_improvement:
                    max_improvement = imp['abs_change_percent']
                    best_metric = imp['metric']
            
            summary['best_improvement'] = {
                'metric': best_metric,
                'change_percent': max_improvement
            }
        
        return summary
    
    def generate_report(self, analysis: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate human-readable regression report"""
        lines = []
        
        # Header
        lines.append("# Performance Regression Analysis Report")
        lines.append(f"Generated: {analysis['analysis_timestamp']}")
        lines.append(f"Threshold: {analysis['threshold_percent']}%")
        lines.append(f"Confidence Level: {analysis['confidence_level']}")
        lines.append("")
        
        # Executive Summary
        summary = analysis['summary']
        status = "🚨 REGRESSIONS DETECTED" if analysis['regression_detected'] else "✅ NO REGRESSIONS"
        lines.append(f"## Status: {status}")
        lines.append("")
        
        lines.append("## Summary")
        lines.append(f"- **Total Metrics**: {summary['total_metrics_compared']}")
        lines.append(f"- **Regressions**: {summary['regressions_count']} ({summary['regression_rate']:.1%})")
        lines.append(f"- **Improvements**: {summary['improvements_count']} ({summary['improvement_rate']:.1%})")
        lines.append(f"- **Stable**: {summary['stable_metrics_count']}")
        lines.append("")
        
        # Regressions Detail
        if analysis['regressions']:
            lines.append("## 🚨 Performance Regressions")
            
            # Sort by severity and change magnitude
            regressions = sorted(
                analysis['regressions'],
                key=lambda x: (
                    {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(x['significance'], 0),
                    x['abs_change_percent']
                ),
                reverse=True
            )
            
            for reg in regressions:
                severity_emoji = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(reg['significance'], '⚪')
                
                lines.append(f"### {severity_emoji} {reg['metric']} ({reg['significance'].upper()})")
                lines.append(f"- **Change**: {reg['change_percent']:+.1f}%")
                lines.append(f"- **Current**: {reg['current_value']:.4f}")
                lines.append(f"- **Historical**: {reg['historical_value']:.4f}")
                
                # Add context about metric type
                direction = "↓" if reg['lower_is_better'] else "↑"
                metric_type = "lower is better" if reg['lower_is_better'] else "higher is better"
                lines.append(f"- **Type**: {metric_type} {direction}")
                lines.append("")
        
        # Improvements Detail
        if analysis['improvements']:
            lines.append("## 📈 Performance Improvements")
            
            improvements = sorted(
                analysis['improvements'],
                key=lambda x: x['abs_change_percent'],
                reverse=True
            )
            
            for imp in improvements[:5]:  # Show top 5 improvements
                lines.append(f"### ✅ {imp['metric']}")
                lines.append(f"- **Change**: {imp['change_percent']:+.1f}%")
                lines.append(f"- **Current**: {imp['current_value']:.4f}")
                lines.append(f"- **Historical**: {imp['historical_value']:.4f}")
                lines.append("")
        
        # Severity Breakdown
        if 'regression_severity' in summary:
            lines.append("## Regression Severity Breakdown")
            severity_counts = summary['regression_severity']
            for severity in ['critical', 'high', 'medium', 'low']:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    lines.append(f"- **{severity.title()}**: {count}")
            lines.append("")
        
        # Recommendations
        lines.append("## 🔧 Recommendations")
        
        if analysis['regression_detected']:
            lines.append("### Immediate Actions")
            if summary.get('worst_regression'):
                worst = summary['worst_regression']
                lines.append(f"1. **Priority**: Investigate {worst['metric']} regression ({worst['change_percent']:.1f}%)")
            
            critical_regressions = [r for r in analysis['regressions'] if r['significance'] == 'critical']
            if critical_regressions:
                lines.append("2. **Critical**: Address critical regressions before deployment")
            
            lines.append("3. **Analysis**: Review code changes that may have caused performance impact")
            lines.append("4. **Monitoring**: Set up alerts for performance-critical metrics")
        else:
            lines.append("- No regressions detected - performance is stable")
            if analysis['improvements']:
                lines.append("- Consider documenting performance improvements for future reference")
        
        report_text = "\n".join(lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {output_file}")
        
        return report_text


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Detect performance regressions in benchmark results'
    )
    parser.add_argument(
        '--current',
        type=str,
        required=True,
        help='Current benchmark results file (JSON)'
    )
    parser.add_argument(
        '--historical',
        type=str,
        required=True,
        help='Historical/baseline benchmark results file (JSON)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=5.0,
        help='Regression threshold percentage (default: 5.0)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.95,
        help='Statistical confidence level (default: 0.95)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output regression analysis to JSON file'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Generate human-readable report file (Markdown)'
    )
    parser.add_argument(
        '--check-regressions',
        action='store_true',
        help='Exit with code 1 if regressions are detected'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        # Create detector
        detector = RegressionDetector(
            threshold=args.threshold,
            confidence_level=args.confidence
        )
        
        # Detect regressions
        print(f"Analyzing regressions...")
        print(f"  Current: {args.current}")
        print(f"  Historical: {args.historical}")
        print(f"  Threshold: {args.threshold}%")
        
        analysis = detector.detect_regressions(args.current, args.historical)
        
        # Save analysis if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"📊 Analysis saved to: {args.output}")
        
        # Generate report if requested
        if args.report:
            report_text = detector.generate_report(analysis, args.report)
            if args.verbose:
                print("\n" + report_text)
        
        # Print summary
        summary = analysis['summary']
        regression_detected = analysis['regression_detected']
        
        print(f"\n{'='*60}")
        print(f"REGRESSION DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Status: {'🚨 REGRESSIONS DETECTED' if regression_detected else '✅ NO REGRESSIONS'}")
        print(f"Metrics analyzed: {summary['total_metrics_compared']}")
        print(f"Regressions found: {summary['regressions_count']}")
        print(f"Improvements found: {summary['improvements_count']}")
        
        if regression_detected:
            print(f"\n🚨 Regression Details:")
            if 'worst_regression' in summary:
                worst = summary['worst_regression']
                print(f"  Worst: {worst['metric']} ({worst['change_percent']:+.1f}%)")
            
            if 'regression_severity' in summary:
                severity_counts = summary['regression_severity']
                severity_list = []
                for severity in ['critical', 'high', 'medium', 'low']:
                    count = severity_counts.get(severity, 0)
                    if count > 0:
                        severity_list.append(f"{count} {severity}")
                if severity_list:
                    print(f"  Severity: {', '.join(severity_list)}")
        
        # Set exit code if regression checking is enabled
        if args.check_regressions and regression_detected:
            print(f"\n💥 Exiting with error code due to regressions")
            sys.exit(1)
        else:
            sys.exit(0)
        
    except Exception as e:
        print(f"❌ Error detecting regressions: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()