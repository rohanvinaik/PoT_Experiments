#!/usr/bin/env python3
"""
Benchmark Analysis and Historical Tracking

Analyzes benchmark results, tracks performance over time,
and generates comprehensive performance reports.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import statistics
import tempfile

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class BenchmarkAnalyzer:
    """Analyzes benchmark results and tracks historical performance"""
    
    def __init__(self, historical_dir: Optional[str] = None):
        self.historical_dir = Path(historical_dir) if historical_dir else None
        self.analysis_results = {}
        
    def analyze_current_benchmarks(self, input_dir: str) -> Dict[str, Any]:
        """Analyze current benchmark results"""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        analysis = {
            'timestamp': datetime.utcnow().isoformat(),
            'input_directory': str(input_path),
            'benchmark_files': [],
            'metrics_summary': {},
            'performance_insights': [],
            'recommendations': []
        }
        
        # Find and analyze benchmark files
        benchmark_files = list(input_path.glob('**/*.json'))
        
        for bench_file in benchmark_files:
            if bench_file.name.startswith('benchmark'):
                try:
                    with open(bench_file, 'r') as f:
                        data = json.load(f)
                    
                    file_analysis = self._analyze_benchmark_file(bench_file, data)
                    analysis['benchmark_files'].append(file_analysis)
                    
                except Exception as e:
                    analysis['benchmark_files'].append({
                        'file': str(bench_file),
                        'error': str(e)
                    })
        
        # Generate summary metrics
        analysis['metrics_summary'] = self._generate_metrics_summary(analysis['benchmark_files'])
        
        # Generate insights and recommendations
        analysis['performance_insights'] = self._generate_performance_insights(analysis['metrics_summary'])
        analysis['recommendations'] = self._generate_recommendations(analysis['metrics_summary'])
        
        return analysis
    
    def _analyze_benchmark_file(self, file_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual benchmark file"""
        file_analysis = {
            'file': str(file_path),
            'file_size_bytes': file_path.stat().st_size,
            'metrics': {},
            'insights': []
        }
        
        # Extract metrics recursively
        metrics = self._extract_metrics_recursive(data)
        file_analysis['metrics'] = metrics
        
        # Analyze specific benchmark types
        if 'model' in str(file_path).lower():
            file_analysis['insights'].extend(self._analyze_model_benchmarks(metrics))
        elif 'zk' in str(file_path).lower() or 'proof' in str(file_path).lower():
            file_analysis['insights'].extend(self._analyze_zk_benchmarks(metrics))
        elif 'security' in str(file_path).lower():
            file_analysis['insights'].extend(self._analyze_security_benchmarks(metrics))
        
        return file_analysis
    
    def _extract_metrics_recursive(self, data: Any, prefix: str = "") -> Dict[str, float]:
        """Recursively extract numeric metrics"""
        metrics = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}_{key}" if prefix else key
                if isinstance(value, (int, float)):
                    metrics[new_prefix] = float(value)
                elif isinstance(value, (dict, list)):
                    metrics.update(self._extract_metrics_recursive(value, new_prefix))
        elif isinstance(data, list):
            if data and all(isinstance(x, (int, float)) for x in data):
                # Numeric list - calculate statistics
                if len(data) > 1:
                    metrics[f"{prefix}_mean"] = statistics.mean(data)
                    metrics[f"{prefix}_median"] = statistics.median(data)
                    metrics[f"{prefix}_std"] = statistics.stdev(data)
                    metrics[f"{prefix}_min"] = min(data)
                    metrics[f"{prefix}_max"] = max(data)
                else:
                    metrics[prefix] = float(data[0])
            else:
                # Non-numeric list
                for i, item in enumerate(data):
                    metrics.update(self._extract_metrics_recursive(item, f"{prefix}_{i}"))
        
        return metrics
    
    def _analyze_model_benchmarks(self, metrics: Dict[str, float]) -> List[str]:
        """Analyze model verification benchmarks"""
        insights = []
        
        # Check verification time
        verification_times = [v for k, v in metrics.items() if 'time' in k.lower() and 'verification' in k.lower()]
        if verification_times:
            avg_time = statistics.mean(verification_times)
            if avg_time > 10.0:
                insights.append(f"High verification time: {avg_time:.2f}s average")
            elif avg_time < 1.0:
                insights.append(f"Fast verification: {avg_time:.2f}s average")
        
        # Check statistical power
        power_metrics = [v for k, v in metrics.items() if 'power' in k.lower()]
        if power_metrics:
            avg_power = statistics.mean(power_metrics)
            if avg_power < 0.8:
                insights.append(f"Low statistical power: {avg_power:.3f}")
            elif avg_power > 0.95:
                insights.append(f"High statistical power: {avg_power:.3f}")
        
        # Check memory usage
        memory_metrics = [v for k, v in metrics.items() if 'memory' in k.lower()]
        if memory_metrics:
            max_memory = max(memory_metrics)
            if max_memory > 8000:  # 8GB in MB
                insights.append(f"High memory usage: {max_memory:.0f}MB")
        
        return insights
    
    def _analyze_zk_benchmarks(self, metrics: Dict[str, float]) -> List[str]:
        """Analyze zero-knowledge proof benchmarks"""
        insights = []
        
        # Check proof generation time
        proof_times = [v for k, v in metrics.items() if 'proof' in k.lower() and 'time' in k.lower()]
        if proof_times:
            avg_time = statistics.mean(proof_times)
            if avg_time > 60.0:
                insights.append(f"Slow proof generation: {avg_time:.1f}s average")
            elif avg_time < 1.0:
                insights.append(f"Fast proof generation: {avg_time:.1f}s average")
        
        # Check proof size
        proof_sizes = [v for k, v in metrics.items() if 'proof' in k.lower() and 'size' in k.lower()]
        if proof_sizes:
            avg_size = statistics.mean(proof_sizes)
            if avg_size > 1000000:  # 1MB
                insights.append(f"Large proof size: {avg_size/1000000:.1f}MB average")
            elif avg_size < 10000:  # 10KB
                insights.append(f"Compact proof size: {avg_size/1000:.1f}KB average")
        
        # Check verification time
        verify_times = [v for k, v in metrics.items() if 'verify' in k.lower() and 'time' in k.lower()]
        if verify_times:
            avg_time = statistics.mean(verify_times)
            if avg_time > 5.0:
                insights.append(f"Slow verification: {avg_time:.2f}s average")
        
        return insights
    
    def _analyze_security_benchmarks(self, metrics: Dict[str, float]) -> List[str]:
        """Analyze security test benchmarks"""
        insights = []
        
        # Check attack detection rates
        detection_rates = [v for k, v in metrics.items() if 'detection' in k.lower() and 'rate' in k.lower()]
        if detection_rates:
            avg_rate = statistics.mean(detection_rates)
            if avg_rate < 0.9:
                insights.append(f"Low attack detection rate: {avg_rate:.2f}")
            elif avg_rate > 0.99:
                insights.append(f"High attack detection rate: {avg_rate:.2f}")
        
        # Check false positive rates
        fp_rates = [v for k, v in metrics.items() if 'false' in k.lower() and 'positive' in k.lower()]
        if fp_rates:
            avg_fp = statistics.mean(fp_rates)
            if avg_fp > 0.1:
                insights.append(f"High false positive rate: {avg_fp:.3f}")
        
        return insights
    
    def _generate_metrics_summary(self, benchmark_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of all metrics"""
        summary = {
            'total_files': len(benchmark_files),
            'total_metrics': 0,
            'metric_categories': {},
            'performance_summary': {}
        }
        
        all_metrics = {}
        
        # Collect all metrics
        for file_info in benchmark_files:
            if 'metrics' in file_info:
                all_metrics.update(file_info['metrics'])
        
        summary['total_metrics'] = len(all_metrics)
        
        # Categorize metrics
        categories = {
            'time': [],
            'memory': [],
            'accuracy': [],
            'size': [],
            'rate': [],
            'other': []
        }
        
        for metric_name, value in all_metrics.items():
            categorized = False
            for category in categories:
                if category in metric_name.lower():
                    categories[category].append(value)
                    categorized = True
                    break
            if not categorized:
                categories['other'].append(value)
        
        # Calculate category statistics
        for category, values in categories.items():
            if values:
                summary['metric_categories'][category] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values)
                }
        
        return summary
    
    def _generate_performance_insights(self, metrics_summary: Dict[str, Any]) -> List[str]:
        """Generate performance insights"""
        insights = []
        
        categories = metrics_summary.get('metric_categories', {})
        
        # Time-based insights
        if 'time' in categories:
            time_stats = categories['time']
            if time_stats['mean'] > 10.0:
                insights.append(f"Performance concern: Average execution time is {time_stats['mean']:.1f}s")
            if time_stats['max'] > time_stats['mean'] * 3:
                insights.append(f"Performance inconsistency: Max time ({time_stats['max']:.1f}s) is much higher than average")
        
        # Memory-based insights
        if 'memory' in categories:
            memory_stats = categories['memory']
            if memory_stats['max'] > 4000:  # 4GB in MB
                insights.append(f"High memory usage detected: Peak {memory_stats['max']:.0f}MB")
        
        # Accuracy-based insights
        if 'accuracy' in categories:
            accuracy_stats = categories['accuracy']
            if accuracy_stats['min'] < 0.9:
                insights.append(f"Low accuracy detected: Minimum {accuracy_stats['min']:.3f}")
        
        return insights
    
    def _generate_recommendations(self, metrics_summary: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        categories = metrics_summary.get('metric_categories', {})
        
        # Time optimization recommendations
        if 'time' in categories:
            time_stats = categories['time']
            if time_stats['std'] > time_stats['mean'] * 0.5:
                recommendations.append("Consider investigating timing inconsistencies - high standard deviation")
            if time_stats['mean'] > 30.0:
                recommendations.append("Consider optimization for long-running operations")
        
        # Memory optimization recommendations
        if 'memory' in categories:
            memory_stats = categories['memory']
            if memory_stats['max'] > 8000:  # 8GB
                recommendations.append("Consider memory optimization or sharding for large memory usage")
        
        # General recommendations
        total_files = metrics_summary.get('total_files', 0)
        if total_files == 0:
            recommendations.append("No benchmark files found - ensure benchmarks are running correctly")
        elif total_files < 3:
            recommendations.append("Consider adding more comprehensive benchmarks")
        
        return recommendations
    
    def compare_with_historical(self, current_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with historical data"""
        if not self.historical_dir or not self.historical_dir.exists():
            return {'error': 'No historical data available'}
        
        # Load historical analyses
        historical_files = list(self.historical_dir.glob('analysis_*.json'))
        if not historical_files:
            return {'error': 'No historical analysis files found'}
        
        historical_analyses = []
        for hist_file in sorted(historical_files)[-10:]:  # Last 10 analyses
            try:
                with open(hist_file, 'r') as f:
                    historical_analyses.append(json.load(f))
            except Exception as e:
                print(f"⚠️  Error loading {hist_file}: {e}")
        
        if not historical_analyses:
            return {'error': 'No valid historical analyses loaded'}
        
        comparison = {
            'historical_count': len(historical_analyses),
            'trends': {},
            'regressions': [],
            'improvements': []
        }
        
        # Analyze trends for key metrics
        current_metrics = current_analysis.get('metrics_summary', {}).get('metric_categories', {})
        
        for category, current_stats in current_metrics.items():
            if category in ['time', 'memory', 'accuracy']:
                historical_values = []
                
                for hist_analysis in historical_analyses:
                    hist_categories = hist_analysis.get('metrics_summary', {}).get('metric_categories', {})
                    if category in hist_categories:
                        historical_values.append(hist_categories[category]['mean'])
                
                if len(historical_values) >= 3:  # Need at least 3 points for trend
                    comparison['trends'][category] = self._analyze_trend(
                        historical_values, current_stats['mean']
                    )
        
        return comparison
    
    def _analyze_trend(self, historical_values: List[float], current_value: float) -> Dict[str, Any]:
        """Analyze trend in metric values"""
        if len(historical_values) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate simple linear trend
        x_values = list(range(len(historical_values)))
        y_values = historical_values
        
        # Simple linear regression
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            trend_direction = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
            
            # Check current value against trend
            predicted_next = y_values[-1] + slope
            deviation_percent = abs((current_value - predicted_next) / predicted_next * 100) if predicted_next != 0 else 0
            
            return {
                'trend': trend_direction,
                'slope': slope,
                'predicted_value': predicted_next,
                'actual_value': current_value,
                'deviation_percent': deviation_percent,
                'historical_mean': statistics.mean(historical_values),
                'historical_std': statistics.stdev(historical_values) if len(historical_values) > 1 else 0
            }
        
        return {'trend': 'calculation_error'}
    
    def save_analysis(self, analysis: Dict[str, Any], output_file: str):
        """Save analysis results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✅ Analysis saved to: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Analyze benchmark results and track performance'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing benchmark results'
    )
    parser.add_argument(
        '--historical',
        type=str,
        help='Historical data directory for trend analysis'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = BenchmarkAnalyzer(args.historical)
        
        # Analyze current benchmarks
        print(f"Analyzing benchmarks in: {args.input}")
        current_analysis = analyzer.analyze_current_benchmarks(args.input)
        
        # Compare with historical data if available
        if args.historical:
            print(f"Comparing with historical data in: {args.historical}")
            historical_comparison = analyzer.compare_with_historical(current_analysis)
            current_analysis['historical_comparison'] = historical_comparison
        
        # Save analysis
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main analysis
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        analysis_file = output_dir / f'analysis_{timestamp}.json'
        analyzer.save_analysis(current_analysis, analysis_file)
        
        # Save summary
        summary_file = output_dir / 'summary.json'
        analyzer.save_analysis(current_analysis, summary_file)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"BENCHMARK ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        metrics_summary = current_analysis.get('metrics_summary', {})
        print(f"Files analyzed: {metrics_summary.get('total_files', 0)}")
        print(f"Metrics extracted: {metrics_summary.get('total_metrics', 0)}")
        
        insights = current_analysis.get('performance_insights', [])
        if insights:
            print(f"\n📊 Performance Insights:")
            for insight in insights[:5]:  # Show first 5
                print(f"  • {insight}")
        
        recommendations = current_analysis.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations[:3]:  # Show first 3
                print(f"  • {rec}")
        
        historical_comparison = current_analysis.get('historical_comparison', {})
        if 'trends' in historical_comparison:
            print(f"\n📈 Historical Trends:")
            for category, trend_info in historical_comparison['trends'].items():
                trend = trend_info.get('trend', 'unknown')
                print(f"  • {category}: {trend}")
        
        print(f"\n✅ Analysis completed successfully")
        
    except Exception as e:
        print(f"❌ Error analyzing benchmarks: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()