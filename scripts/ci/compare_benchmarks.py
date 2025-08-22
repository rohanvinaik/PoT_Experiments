#!/usr/bin/env python3
"""
Benchmark Comparison and Regression Detection

Compares current benchmark results with historical data to detect
performance regressions and improvements.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import statistics

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class BenchmarkComparison:
    """Represents a comparison between two benchmark results"""
    
    def __init__(self, metric_name: str, current_value: float, baseline_value: float):
        self.metric_name = metric_name
        self.current_value = current_value
        self.baseline_value = baseline_value
        self.change_percent = self._calculate_change_percent()
        self.change_type = self._determine_change_type()
    
    def _calculate_change_percent(self) -> float:
        """Calculate percentage change"""
        if self.baseline_value == 0:
            return float('inf') if self.current_value > 0 else 0.0
        return ((self.current_value - self.baseline_value) / self.baseline_value) * 100
    
    def _determine_change_type(self) -> str:
        """Determine type of change"""
        if abs(self.change_percent) < 0.1:
            return "no_change"
        elif self.change_percent > 0:
            return "increase"
        else:
            return "decrease"
    
    def is_regression(self, threshold: float, lower_is_better: bool = True) -> bool:
        """Check if this represents a performance regression"""
        if lower_is_better:
            return self.change_percent > threshold
        else:
            return self.change_percent < -threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'metric': self.metric_name,
            'current_value': self.current_value,
            'baseline_value': self.baseline_value,
            'change_percent': self.change_percent,
            'change_type': self.change_type
        }


class BenchmarkRegression:
    """Represents a detected performance regression"""
    
    def __init__(self, comparison: BenchmarkComparison, threshold: float, severity: str):
        self.comparison = comparison
        self.threshold = threshold
        self.severity = severity
        self.detected_at = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'metric': self.comparison.metric_name,
            'change_percent': self.comparison.change_percent,
            'threshold': self.threshold,
            'severity': self.severity,
            'detected_at': self.detected_at,
            'current_value': self.comparison.current_value,
            'baseline_value': self.comparison.baseline_value
        }


class BenchmarkComparator:
    """Compares benchmark results and detects regressions"""
    
    def __init__(self, regression_threshold: float = 10.0):
        self.regression_threshold = regression_threshold
        self.comparisons: List[BenchmarkComparison] = []
        self.regressions: List[BenchmarkRegression] = []
        self.improvements: List[BenchmarkComparison] = []
        
        # Metric configuration - defines which metrics are "lower is better"
        self.metric_config = {
            # Performance metrics (lower is better)
            'duration': True,
            'execution_time': True,
            'latency': True,
            'response_time': True,
            'build_time': True,
            'test_time': True,
            'memory_usage': True,
            'peak_memory': True,
            'cpu_usage': True,
            
            # Quality metrics (higher is better)
            'accuracy': False,
            'precision': False,
            'recall': False,
            'f1_score': False,
            'confidence': False,
            'success_rate': False,
            'coverage': False,
            'throughput': False,
            'qps': False,  # Queries per second
            
            # ZK proof metrics (lower is better for time/size, higher for verification)
            'proof_generation_time': True,
            'proof_size': True,
            'verification_time': True,
            'verification_success_rate': False,
            
            # Model verification metrics
            'statistical_power': False,
            'verification_confidence': False,
            'detection_accuracy': False
        }
    
    def load_benchmark_data(self, file_path: str) -> Dict[str, Any]:
        """Load benchmark data from file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {file_path}")
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in benchmark file {file_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading benchmark file {file_path}: {e}")
    
    def extract_metrics(self, benchmark_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric metrics from benchmark data"""
        metrics = {}
        
        def extract_recursive(data: Any, prefix: str = ""):
            """Recursively extract numeric values"""
            if isinstance(data, dict):
                for key, value in data.items():
                    new_key = f"{prefix}_{key}" if prefix else key
                    extract_recursive(value, new_key)
            elif isinstance(data, list):
                if data and isinstance(data[0], (int, float)):
                    # Calculate statistics for numeric lists
                    if len(data) > 1:
                        metrics[f"{prefix}_mean"] = statistics.mean(data)
                        metrics[f"{prefix}_median"] = statistics.median(data)
                        metrics[f"{prefix}_max"] = max(data)
                        metrics[f"{prefix}_min"] = min(data)
                        if len(data) > 2:
                            metrics[f"{prefix}_stdev"] = statistics.stdev(data)
                    else:
                        metrics[prefix] = data[0]
                else:
                    # Process list elements
                    for i, item in enumerate(data):
                        extract_recursive(item, f"{prefix}_{i}")
            elif isinstance(data, (int, float)):
                metrics[prefix] = float(data)
        
        extract_recursive(benchmark_data)
        return metrics
    
    def compare_metrics(self, current_metrics: Dict[str, float], 
                       baseline_metrics: Dict[str, float]) -> List[BenchmarkComparison]:
        """Compare current metrics with baseline"""
        comparisons = []
        
        # Find common metrics
        common_metrics = set(current_metrics.keys()) & set(baseline_metrics.keys())
        
        for metric_name in common_metrics:
            current_value = current_metrics[metric_name]
            baseline_value = baseline_metrics[metric_name]
            
            comparison = BenchmarkComparison(metric_name, current_value, baseline_value)
            comparisons.append(comparison)
        
        # Check for missing metrics
        missing_in_current = set(baseline_metrics.keys()) - set(current_metrics.keys())
        missing_in_baseline = set(current_metrics.keys()) - set(baseline_metrics.keys())
        
        if missing_in_current:
            print(f"⚠️  Metrics missing in current: {list(missing_in_current)}")
        if missing_in_baseline:
            print(f"ℹ️  New metrics in current: {list(missing_in_baseline)}")
        
        return comparisons
    
    def detect_regressions(self, comparisons: List[BenchmarkComparison]) -> List[BenchmarkRegression]:
        """Detect performance regressions from comparisons"""
        regressions = []
        
        for comparison in comparisons:
            # Determine if lower is better for this metric
            lower_is_better = self._is_lower_better(comparison.metric_name)
            
            # Check for regression
            if comparison.is_regression(self.regression_threshold, lower_is_better):
                severity = self._calculate_severity(comparison.change_percent)
                regression = BenchmarkRegression(comparison, self.regression_threshold, severity)
                regressions.append(regression)
        
        return regressions
    
    def detect_improvements(self, comparisons: List[BenchmarkComparison]) -> List[BenchmarkComparison]:
        """Detect performance improvements from comparisons"""
        improvements = []
        
        for comparison in comparisons:
            # Determine if lower is better for this metric
            lower_is_better = self._is_lower_better(comparison.metric_name)
            
            # Check for improvement (opposite of regression)
            change_percent = abs(comparison.change_percent)
            if change_percent > self.regression_threshold:
                is_improvement = (
                    (lower_is_better and comparison.change_percent < -self.regression_threshold) or
                    (not lower_is_better and comparison.change_percent > self.regression_threshold)
                )
                
                if is_improvement:
                    improvements.append(comparison)
        
        return improvements
    
    def _is_lower_better(self, metric_name: str) -> bool:
        """Determine if lower values are better for a metric"""
        # Check exact match first
        if metric_name in self.metric_config:
            return self.metric_config[metric_name]
        
        # Check for partial matches
        lower_keywords = ['time', 'duration', 'latency', 'memory', 'usage', 'size', 'cost']
        higher_keywords = ['accuracy', 'precision', 'recall', 'score', 'rate', 'confidence', 'throughput']
        
        metric_lower = metric_name.lower()
        
        for keyword in lower_keywords:
            if keyword in metric_lower:
                return True
        
        for keyword in higher_keywords:
            if keyword in metric_lower:
                return False
        
        # Default to lower is better for unknown metrics
        return True
    
    def _calculate_severity(self, change_percent: float) -> str:
        """Calculate regression severity based on change percentage"""
        abs_change = abs(change_percent)
        
        if abs_change >= 50:
            return "critical"
        elif abs_change >= 25:
            return "high"
        elif abs_change >= 15:
            return "medium"
        else:
            return "low"
    
    def compare_benchmarks(self, current_file: str, baseline_file: str) -> Dict[str, Any]:
        """Compare two benchmark files and return analysis results"""
        print(f"Loading current benchmarks from: {current_file}")
        current_data = self.load_benchmark_data(current_file)
        current_metrics = self.extract_metrics(current_data)
        
        print(f"Loading baseline benchmarks from: {baseline_file}")
        baseline_data = self.load_benchmark_data(baseline_file)
        baseline_metrics = self.extract_metrics(baseline_data)
        
        print(f"Comparing {len(current_metrics)} current vs {len(baseline_metrics)} baseline metrics")
        
        # Perform comparison
        self.comparisons = self.compare_metrics(current_metrics, baseline_metrics)
        self.regressions = self.detect_regressions(self.comparisons)
        self.improvements = self.detect_improvements(self.comparisons)
        
        # Generate summary
        results = {
            'comparison_timestamp': datetime.utcnow().isoformat(),
            'current_file': current_file,
            'baseline_file': baseline_file,
            'regression_threshold': self.regression_threshold,
            'summary': {
                'total_metrics_compared': len(self.comparisons),
                'regressions_detected': len(self.regressions),
                'improvements_detected': len(self.improvements),
                'regression_detected': len(self.regressions) > 0
            },
            'regressions': [r.to_dict() for r in self.regressions],
            'improvements': [i.to_dict() for i in self.improvements],
            'all_comparisons': [c.to_dict() for c in self.comparisons]
        }
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate human-readable comparison report"""
        report_lines = []
        
        # Header
        report_lines.append("# Benchmark Comparison Report")
        report_lines.append(f"Generated: {results['comparison_timestamp']}")
        report_lines.append(f"Threshold: {results['regression_threshold']}%")
        report_lines.append("")
        
        # Summary
        summary = results['summary']
        report_lines.append("## Summary")
        report_lines.append(f"- Metrics compared: {summary['total_metrics_compared']}")
        report_lines.append(f"- Regressions detected: {summary['regressions_detected']}")
        report_lines.append(f"- Improvements detected: {summary['improvements_detected']}")
        
        status = "❌ REGRESSION DETECTED" if summary['regression_detected'] else "✅ NO REGRESSIONS"
        report_lines.append(f"- Status: {status}")
        report_lines.append("")
        
        # Regressions
        if results['regressions']:
            report_lines.append("## 🚨 Performance Regressions")
            for regression in results['regressions']:
                severity_emoji = {
                    'critical': '🔴',
                    'high': '🟠', 
                    'medium': '🟡',
                    'low': '🟢'
                }.get(regression['severity'], '⚪')
                
                report_lines.append(f"### {severity_emoji} {regression['metric']} ({regression['severity'].upper()})")
                report_lines.append(f"- Change: {regression['change_percent']:+.1f}%")
                report_lines.append(f"- Current: {regression['current_value']:.4f}")
                report_lines.append(f"- Baseline: {regression['baseline_value']:.4f}")
                report_lines.append("")
        
        # Improvements
        if results['improvements']:
            report_lines.append("## 📈 Performance Improvements")
            for improvement in results['improvements']:
                report_lines.append(f"### ✅ {improvement['metric']}")
                report_lines.append(f"- Change: {improvement['change_percent']:+.1f}%")
                report_lines.append(f"- Current: {improvement['current_value']:.4f}")
                report_lines.append(f"- Baseline: {improvement['baseline_value']:.4f}")
                report_lines.append("")
        
        # Top changes
        all_comparisons = results['all_comparisons']
        if all_comparisons:
            # Sort by absolute change
            sorted_comparisons = sorted(
                all_comparisons, 
                key=lambda x: abs(x['change_percent']), 
                reverse=True
            )
            
            report_lines.append("## Top 10 Changes")
            for i, comp in enumerate(sorted_comparisons[:10]):
                direction = "📈" if comp['change_percent'] > 0 else "📉"
                report_lines.append(
                    f"{i+1}. {direction} {comp['metric']}: {comp['change_percent']:+.1f}%"
                )
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")
        
        return report_text


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Compare benchmark results and detect regressions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare current results with baseline
  %(prog)s --current benchmark_results.json --baseline baseline.json
  
  # Use custom regression threshold
  %(prog)s --current results.json --baseline baseline.json --threshold 5.0
  
  # Generate detailed report
  %(prog)s --current results.json --baseline baseline.json --report regression_report.md
        """
    )
    
    parser.add_argument(
        '--current',
        type=str,
        required=True,
        help='Current benchmark results file (JSON)'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        help='Baseline benchmark results file (JSON)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=10.0,
        help='Regression threshold percentage (default: 10.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output comparison results to JSON file'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Generate human-readable report file (Markdown)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        # If no baseline provided, look for common baseline files
        baseline_file = args.baseline
        if not baseline_file:
            baseline_candidates = [
                'baseline_benchmark.json',
                'benchmark_baseline.json', 
                'historical_benchmark.json',
                Path(args.current).parent / 'baseline.json'
            ]
            
            for candidate in baseline_candidates:
                if Path(candidate).exists():
                    baseline_file = str(candidate)
                    print(f"Using detected baseline: {baseline_file}")
                    break
            
            if not baseline_file:
                print("❌ No baseline file provided and none found automatically")
                print("   Use --baseline to specify baseline file")
                sys.exit(1)
        
        # Perform comparison
        comparator = BenchmarkComparator(regression_threshold=args.threshold)
        results = comparator.compare_benchmarks(args.current, baseline_file)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Comparison results saved to: {args.output}")
        
        # Generate report if requested
        if args.report:
            report_text = comparator.generate_report(results, args.report)
            if args.verbose:
                print("\n" + report_text)
        
        # Print summary
        summary = results['summary']
        print(f"\n{'='*60}")
        print(f"BENCHMARK COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Metrics compared: {summary['total_metrics_compared']}")
        print(f"Regressions: {summary['regressions_detected']}")
        print(f"Improvements: {summary['improvements_detected']}")
        
        if summary['regression_detected']:
            print(f"\n❌ REGRESSIONS DETECTED")
            for regression in results['regressions']:
                print(f"  • {regression['metric']}: {regression['change_percent']:+.1f}% ({regression['severity']})")
            sys.exit(1)
        else:
            print(f"\n✅ NO REGRESSIONS DETECTED")
            if summary['improvements_detected'] > 0:
                print(f"🎉 {summary['improvements_detected']} improvements found!")
            sys.exit(0)
    
    except Exception as e:
        print(f"❌ Error comparing benchmarks: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()