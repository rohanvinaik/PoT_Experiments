#!/usr/bin/env python3
"""
ZK Metrics Summary Script

Parses ZK metrics JSON and generates comprehensive performance analysis,
including regression detection and trend analysis.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import statistics


def load_metrics(filepath: str) -> Optional[Dict[str, Any]]:
    """Load metrics from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading metrics file: {e}")
        return None


def load_baseline_metrics(baseline_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load baseline metrics for comparison."""
    if not baseline_path:
        return None
    
    try:
        with open(baseline_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️  Could not load baseline metrics: {e}")
        return None


def format_duration(ms: float) -> str:
    """Format duration in human-readable format."""
    if ms < 1000:
        return f"{ms:.1f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}m"


def format_size(bytes_size: int) -> str:
    """Format size in human-readable format."""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024**2:
        return f"{bytes_size/1024:.1f} KB"
    elif bytes_size < 1024**3:
        return f"{bytes_size/(1024**2):.1f} MB"
    else:
        return f"{bytes_size/(1024**3):.1f} GB"


def calculate_percentile(data: List[float], percentile: float) -> float:
    """Calculate percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100.0)
    index = min(index, len(sorted_data) - 1)
    return sorted_data[index]


def print_performance_table(proof_type: str, stats: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None):
    """Print performance statistics table."""
    if stats.get('count', 0) == 0:
        print(f"   No {proof_type} proofs recorded")
        return
    
    print(f"\n   📊 {proof_type.upper()} Proof Performance:")
    print(f"   {'Metric':<20} {'Current':<15} {'Baseline':<15} {'Change':<10}")
    print(f"   {'-'*20} {'-'*15} {'-'*15} {'-'*10}")
    
    # Helper function to format comparison
    def format_comparison(current: float, baseline_val: Optional[float], unit: str = "ms", lower_is_better: bool = True) -> tuple:
        if baseline_val is None or baseline_val == 0:
            return f"{current:.1f}{unit}", "N/A", "N/A"
        
        change = ((current - baseline_val) / baseline_val) * 100
        change_sign = "↓" if (change < 0) == lower_is_better else "↑"
        change_color = "🟢" if (change < 0) == lower_is_better else "🔴"
        
        if abs(change) > 20:  # Significant change
            change_str = f"{change_color} {change_sign}{abs(change):.1f}%"
        else:
            change_str = f"{change_sign}{abs(change):.1f}%"
        
        return f"{current:.1f}{unit}", f"{baseline_val:.1f}{unit}", change_str
    
    baseline_stats = baseline.get('performance', {}).get(f'{proof_type}_stats', {}) if baseline else {}
    
    # Duration metrics
    metrics = [
        ('Count', stats.get('count', 0), baseline_stats.get('count', 0), '', False),
        ('Avg Duration', stats.get('avg_duration_ms', 0), baseline_stats.get('avg_duration_ms'), 'ms', True),
        ('Median Duration', stats.get('median_duration_ms', 0), baseline_stats.get('median_duration_ms'), 'ms', True),
        ('P95 Duration', stats.get('p95_duration_ms', 0), baseline_stats.get('p95_duration_ms'), 'ms', True),
        ('Success Rate', stats.get('success_rate', 1.0) * 100, (baseline_stats.get('success_rate', 1.0) * 100) if baseline_stats else None, '%', False),
    ]
    
    if 'avg_size_bytes' in stats:
        metrics.append(('Avg Proof Size', stats.get('avg_size_bytes', 0), baseline_stats.get('avg_size_bytes'), ' B', True))
    
    for metric_name, current_val, baseline_val, unit, lower_better in metrics:
        if metric_name == 'Count':
            current_str = str(int(current_val))
            baseline_str = str(int(baseline_val)) if baseline_val else "N/A"
            change_str = f"+{int(current_val - baseline_val)}" if baseline_val else "N/A"
        else:
            current_str, baseline_str, change_str = format_comparison(current_val, baseline_val, unit, lower_better)
        
        print(f"   {metric_name:<20} {current_str:<15} {baseline_str:<15} {change_str:<10}")


def analyze_performance_trends(metrics: Dict[str, Any]) -> List[str]:
    """Analyze performance trends and generate insights."""
    insights = []
    
    # Get performance stats
    sgd_stats = metrics.get('performance', {}).get('sgd_stats', {})
    lora_stats = metrics.get('performance', {}).get('lora_stats', {})
    verification_stats = metrics.get('performance', {}).get('verification_stats', {})
    
    # Compression analysis
    if sgd_stats.get('avg_size_bytes') and lora_stats.get('avg_size_bytes'):
        compression_ratio = sgd_stats['avg_size_bytes'] / lora_stats['avg_size_bytes']
        insights.append(f"📏 LoRA provides {compression_ratio:.1f}x compression vs SGD")
        
        if compression_ratio > 50:
            insights.append("🎯 Excellent compression achieved with LoRA")
        elif compression_ratio > 10:
            insights.append("✅ Good compression with LoRA")
        else:
            insights.append("⚠️  Low compression ratio - investigate LoRA efficiency")
    
    # Performance comparison
    if sgd_stats.get('avg_duration_ms') and lora_stats.get('avg_duration_ms'):
        sgd_time = sgd_stats['avg_duration_ms']
        lora_time = lora_stats['avg_duration_ms']
        
        if lora_time < sgd_time:
            speedup = sgd_time / lora_time
            insights.append(f"⚡ LoRA is {speedup:.1f}x faster than SGD")
        else:
            slowdown = lora_time / sgd_time
            insights.append(f"🐌 LoRA is {slowdown:.1f}x slower than SGD (unexpected)")
    
    # Success rate analysis
    overall_success = metrics.get('summary', {}).get('overall_success_rate', 1.0)
    if overall_success < 0.95:
        insights.append(f"🚨 Low success rate: {overall_success:.1%} - investigate failures")
    elif overall_success >= 0.99:
        insights.append(f"🎉 Excellent success rate: {overall_success:.1%}")
    
    # Error analysis
    errors = metrics.get('errors', {})
    if errors:
        total_errors = sum(errors.values())
        top_error = max(errors.items(), key=lambda x: x[1])
        insights.append(f"🔍 Top error: {top_error[0]} ({top_error[1]}/{total_errors} errors)")
    
    return insights


def detect_performance_regressions(current: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None) -> List[str]:
    """Detect performance regressions."""
    warnings = current.get('performance_warnings', [])
    
    if not baseline:
        return warnings
    
    # Additional regression checks with baseline
    current_sgd = current.get('performance', {}).get('sgd_stats', {})
    baseline_sgd = baseline.get('performance', {}).get('sgd_stats', {})
    
    current_lora = current.get('performance', {}).get('lora_stats', {})
    baseline_lora = baseline.get('performance', {}).get('lora_stats', {})
    
    # Check for significant regressions (>25%)
    if current_sgd.get('avg_duration_ms', 0) > baseline_sgd.get('avg_duration_ms', 0) * 1.25:
        regression = ((current_sgd['avg_duration_ms'] - baseline_sgd['avg_duration_ms']) / baseline_sgd['avg_duration_ms']) * 100
        warnings.append(f"🚨 SGD performance regression: +{regression:.1f}% slower")
    
    if current_lora.get('avg_duration_ms', 0) > baseline_lora.get('avg_duration_ms', 0) * 1.25:
        regression = ((current_lora['avg_duration_ms'] - baseline_lora['avg_duration_ms']) / baseline_lora['avg_duration_ms']) * 100
        warnings.append(f"🚨 LoRA performance regression: +{regression:.1f}% slower")
    
    return warnings


def generate_summary_report(metrics: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None) -> str:
    """Generate comprehensive summary report."""
    
    # Extract key information
    timestamp = metrics.get('timestamp', 'Unknown')
    session_info = metrics.get('session_info', {})
    summary = metrics.get('summary', {})
    performance = metrics.get('performance', {})
    
    # Calculate session duration
    if 'start_time' in session_info and 'end_time' in session_info:
        start = datetime.fromisoformat(session_info['start_time'].replace('Z', '+00:00'))
        end = datetime.fromisoformat(session_info['end_time'].replace('Z', '+00:00'))
        duration = end - start
        duration_str = f"{duration.total_seconds():.1f}s"
    else:
        duration_str = "Unknown"
    
    print("=" * 80)
    print("🔧 ZERO-KNOWLEDGE PROOF SYSTEM METRICS SUMMARY")
    print("=" * 80)
    
    print(f"\n📅 Report Generated: {timestamp}")
    print(f"⏱️  Session Duration: {duration_str}")
    print(f"🔢 Total Operations: {session_info.get('total_operations', 0)}")
    
    # Overall Summary
    print(f"\n📊 OVERALL SUMMARY")
    print(f"   Total Proofs: {summary.get('total_proofs', 0)}")
    print(f"   Total Verifications: {summary.get('total_verifications', 0)}")
    print(f"   Success Rate: {summary.get('overall_success_rate', 1.0):.1%}")
    
    if summary.get('compression_ratio'):
        print(f"   Compression Ratio: {summary['compression_ratio']:.1f}x (LoRA vs SGD)")
    
    # Performance Details
    print(f"\n🚀 PERFORMANCE DETAILS")
    
    sgd_stats = performance.get('sgd_stats', {})
    lora_stats = performance.get('lora_stats', {})
    verification_stats = performance.get('verification_stats', {})
    
    if sgd_stats:
        print_performance_table('sgd', sgd_stats, baseline)
    
    if lora_stats:
        print_performance_table('lora', lora_stats, baseline)
    
    if verification_stats and verification_stats.get('count', 0) > 0:
        print(f"\n   📊 VERIFICATION Performance:")
        print(f"   {'Metric':<20} {'Value':<15}")
        print(f"   {'-'*20} {'-'*15}")
        print(f"   {'Count':<20} {verification_stats.get('count', 0):<15}")
        print(f"   {'Avg Duration':<20} {format_duration(verification_stats.get('avg_duration_ms', 0)):<15}")
        print(f"   {'Success Rate':<20} {verification_stats.get('success_rate', 1.0):.1%:<15}")
    
    # Circuit Constraints
    constraints = metrics.get('circuit_constraints', {})
    if constraints:
        print(f"\n🔧 CIRCUIT CONSTRAINTS")
        for circuit_type, data in constraints.items():
            print(f"   {circuit_type.upper()}:")
            print(f"     Constraints: {data.get('constraint_count', 'N/A')}")
            print(f"     Witness Size: {format_size(data.get('witness_size', 0))}")
    
    # Performance Insights
    insights = analyze_performance_trends(metrics)
    if insights:
        print(f"\n💡 PERFORMANCE INSIGHTS")
        for insight in insights:
            print(f"   {insight}")
    
    # Regression Analysis
    regressions = detect_performance_regressions(metrics, baseline)
    if regressions:
        print(f"\n🚨 PERFORMANCE ALERTS")
        for warning in regressions:
            print(f"   {warning}")
    else:
        print(f"\n✅ NO PERFORMANCE REGRESSIONS DETECTED")
    
    # Error Analysis
    errors = metrics.get('errors', {})
    if errors:
        print(f"\n❌ ERROR BREAKDOWN")
        total_errors = sum(errors.values())
        for error_type, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_errors) * 100
            print(f"   {error_type}: {count} ({percentage:.1f}%)")
    else:
        print(f"\n✅ NO ERRORS RECORDED")
    
    # Recommendations
    print(f"\n💭 RECOMMENDATIONS")
    recommendations = []
    
    # Performance recommendations
    if sgd_stats.get('avg_duration_ms', 0) > 5000:
        recommendations.append("🔧 Consider optimizing SGD circuit for better performance")
    
    if lora_stats.get('avg_duration_ms', 0) > 2000:
        recommendations.append("🔧 LoRA performance below target - investigate bottlenecks")
    
    if summary.get('overall_success_rate', 1.0) < 0.95:
        recommendations.append("🔧 Investigate and fix proof generation failures")
    
    if summary.get('compression_ratio', 1) < 10:
        recommendations.append("🔧 Optimize LoRA compression ratio for better efficiency")
    
    if not recommendations:
        recommendations.append("🎉 System performing well - no immediate optimizations needed")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print("\n" + "=" * 80)
    
    return "Summary generated successfully"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Summarize ZK metrics and detect performance regressions")
    parser.add_argument("metrics_file", help="Path to ZK metrics JSON file")
    parser.add_argument("--baseline", help="Path to baseline metrics for comparison")
    parser.add_argument("--output", help="Output file for summary (default: stdout)")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    args = parser.parse_args()
    
    # Load current metrics
    metrics = load_metrics(args.metrics_file)
    if not metrics:
        sys.exit(1)
    
    # Load baseline if provided
    baseline = load_baseline_metrics(args.baseline) if args.baseline else None
    
    if args.json:
        # Output JSON format
        output = {
            'metrics': metrics,
            'baseline': baseline,
            'analysis': {
                'insights': analyze_performance_trends(metrics),
                'regressions': detect_performance_regressions(metrics, baseline)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2, default=str)
            print(f"✅ JSON summary saved to {args.output}")
        else:
            print(json.dumps(output, indent=2, default=str))
    else:
        # Output human-readable format
        if args.output:
            # Redirect stdout to file
            import sys
            original_stdout = sys.stdout
            with open(args.output, 'w') as f:
                sys.stdout = f
                generate_summary_report(metrics, baseline)
            sys.stdout = original_stdout
            print(f"✅ Summary saved to {args.output}")
        else:
            generate_summary_report(metrics, baseline)


if __name__ == "__main__":
    main()