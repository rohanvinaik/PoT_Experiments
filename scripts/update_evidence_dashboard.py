#!/usr/bin/env python3
"""
Update Evidence Dashboard and Rolling Evidence Section in README
Auto-generates comprehensive performance metrics and evidence
"""

import json
import datetime
import pathlib
import sys
import os
import statistics
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pot.core.evidence_logger import EvidenceLogger


def update_readme_evidence_section():
    """Update the Live Performance Metrics section in README"""
    
    readme_path = pathlib.Path(__file__).parent.parent / "README.md"
    if not readme_path.exists():
        print("❌ README.md not found")
        return False
    
    # Initialize evidence logger to get latest metrics
    logger = EvidenceLogger()
    logger.update_evidence_dashboard()
    
    # Load rolling metrics
    metrics_file = pathlib.Path("experimental_results") / "rolling_metrics.json"
    if not metrics_file.exists():
        print("❌ No rolling metrics data available")
        return False
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    # Generate updated metrics section
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Calculate summary statistics
    success_rate = (metrics['successful_runs'] / metrics['total_runs'] * 100) if metrics['total_runs'] > 0 else 0
    zk_success_rate = (metrics['zk_pipeline_runs'] / metrics['total_runs'] * 100) if metrics['total_runs'] > 0 else 0
    
    # Timing statistics
    timing_stats = calculate_timing_stats(metrics.get('timing_samples', []))
    statistical_stats = calculate_statistical_stats(metrics.get('statistical_samples', []))
    
    new_metrics_section = f"""## 📈 Live Performance Metrics

*Auto-updated from latest validation runs:*

- **Total Validation Runs**: {metrics['total_runs']}
- **Overall Success Rate**: {success_rate:.1f}% ({metrics['successful_runs']}/{metrics['total_runs']})
- **ZK Pipeline Success Rate**: {zk_success_rate:.1f}% ({metrics['zk_pipeline_runs']} runs with ZK)
- **Interface Compliance Runs**: {metrics['interface_test_runs']} tests
{format_timing_summary(timing_stats)}
{format_statistical_summary(statistical_stats)}

*Last Updated: {timestamp}*

### 📊 Recent Performance Trends

{format_recent_trends(metrics)}

### 🎯 Quality Metrics

{format_quality_metrics(metrics)}
"""

    # Read current README
    with open(readme_path) as f:
        readme_content = f.read()
    
    # Find and replace the Live Performance Metrics section
    start_marker = "## 📈 Live Performance Metrics"
    end_marker = "## 🧪 Testing & Validation"
    
    start_idx = readme_content.find(start_marker)
    end_idx = readme_content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print("❌ Could not find Live Performance Metrics section markers in README")
        return False
    
    # Replace the section
    updated_readme = (
        readme_content[:start_idx] +
        new_metrics_section + "\n" +
        readme_content[end_idx:]
    )
    
    # Write updated README
    with open(readme_path, 'w') as f:
        f.write(updated_readme)
    
    print(f"✅ Updated README.md with latest metrics ({metrics['total_runs']} total runs)")
    return True


def calculate_timing_stats(samples: List[Dict]) -> Dict[str, float]:
    """Calculate timing performance statistics"""
    if not samples:
        return {}
        
    per_query_times = [s['t_per_query'] for s in samples if s.get('t_per_query', 0) > 0]
    total_times = [s['t_total'] for s in samples if s.get('t_total', 0) > 0]
    
    stats = {}
    if per_query_times:
        stats['avg_per_query'] = statistics.mean(per_query_times)
        stats['median_per_query'] = statistics.median(per_query_times)
        stats['min_per_query'] = min(per_query_times)
        stats['max_per_query'] = max(per_query_times)
        if len(per_query_times) > 1:
            stats['std_per_query'] = statistics.stdev(per_query_times)
    
    if total_times:
        stats['avg_total'] = statistics.mean(total_times)
        
    return stats


def calculate_statistical_stats(samples: List[Dict]) -> Dict:
    """Calculate statistical testing performance"""
    if not samples:
        return {}
        
    decisions = [s['decision'] for s in samples]
    confidences = [s['confidence'] for s in samples if s.get('confidence', 0) > 0]
    n_used_values = [s['n_used'] for s in samples if s.get('n_used', 0) > 0]
    
    stats = {
        'total_tests': len(samples),
        'decisions': {
            'SAME': decisions.count('SAME'),
            'DIFFERENT': decisions.count('DIFFERENT'),
            'UNDECIDED': decisions.count('UNDECIDED')
        }
    }
    
    if confidences:
        stats['avg_confidence'] = statistics.mean(confidences)
    
    if n_used_values:
        stats['avg_samples_used'] = statistics.mean(n_used_values)
    
    return stats


def format_timing_summary(stats: Dict) -> str:
    """Format timing statistics for README"""
    if not stats:
        return "- **Performance Data**: Collecting..."
        
    lines = []
    if 'avg_per_query' in stats:
        lines.append(f"- **Average Query Time**: {stats['avg_per_query']:.3f}s")
        lines.append(f"- **Performance Range**: {stats['min_per_query']:.3f}s - {stats['max_per_query']:.3f}s")
        if 'std_per_query' in stats:
            coefficient_var = (stats['std_per_query'] / stats['avg_per_query']) * 100
            lines.append(f"- **Performance Stability**: {coefficient_var:.1f}% coefficient of variation")
    
    return '\n'.join(lines)


def format_statistical_summary(stats: Dict) -> str:
    """Format statistical testing summary"""
    if not stats:
        return "- **Statistical Tests**: Collecting data..."
        
    lines = []
    if 'total_tests' in stats:
        lines.append(f"- **Statistical Tests Completed**: {stats['total_tests']}")
        
    if 'decisions' in stats:
        decisive = stats['decisions']['SAME'] + stats['decisions']['DIFFERENT']
        decisive_rate = (decisive / stats['total_tests'] * 100) if stats['total_tests'] > 0 else 0
        lines.append(f"- **Decisive Outcome Rate**: {decisive_rate:.1f}% ({decisive}/{stats['total_tests']})")
    
    if 'avg_confidence' in stats:
        lines.append(f"- **Average Confidence Level**: {stats['avg_confidence']:.1%}")
        
    if 'avg_samples_used' in stats:
        lines.append(f"- **Average Samples per Test**: {stats['avg_samples_used']:.1f}")
    
    return '\n'.join(lines)


def format_recent_trends(metrics: Dict) -> str:
    """Format recent performance trends"""
    recent_samples = metrics.get('timing_samples', [])[-10:]  # Last 10 runs
    
    if len(recent_samples) < 2:
        return "- **Trend Analysis**: Insufficient data (need ≥2 runs)"
    
    # Calculate trend in performance
    recent_times = [s.get('t_per_query', 0) for s in recent_samples if s.get('t_per_query', 0) > 0]
    
    if len(recent_times) < 2:
        return "- **Performance Trend**: Collecting timing data..."
    
    # Simple linear trend
    avg_first_half = statistics.mean(recent_times[:len(recent_times)//2])
    avg_second_half = statistics.mean(recent_times[len(recent_times)//2:])
    
    if avg_second_half < avg_first_half:
        trend = f"⬇️ Improving ({((avg_first_half - avg_second_half) / avg_first_half * 100):.1f}% faster)"
    elif avg_second_half > avg_first_half:
        trend = f"⬆️ Slight increase ({((avg_second_half - avg_first_half) / avg_first_half * 100):.1f}% slower)"
    else:
        trend = "➡️ Stable performance"
    
    return f"- **Performance Trend**: {trend} (last {len(recent_times)} runs)"


def format_quality_metrics(metrics: Dict) -> str:
    """Format quality and reliability metrics"""
    lines = []
    
    # Success rate trend
    if metrics['total_runs'] > 0:
        success_rate = metrics['successful_runs'] / metrics['total_runs']
        if success_rate >= 0.95:
            quality_status = "🟢 Excellent"
        elif success_rate >= 0.90:
            quality_status = "🟡 Good"
        else:
            quality_status = "🔴 Needs Improvement"
            
        lines.append(f"- **System Reliability**: {quality_status} ({success_rate:.1%} success rate)")
    
    # ZK pipeline health
    if metrics['zk_pipeline_runs'] > 0:
        zk_rate = metrics['zk_pipeline_runs'] / metrics['total_runs']
        lines.append(f"- **ZK Pipeline Health**: {zk_rate:.1%} of runs include ZK proofs")
    
    # Interface compliance
    if metrics['interface_test_runs'] > 0:
        interface_rate = metrics['interface_test_runs'] / metrics['total_runs']
        lines.append(f"- **Interface Compliance**: {interface_rate:.1%} of runs include interface tests")
    
    return '\n'.join(lines) if lines else "- **Quality Metrics**: Collecting data..."


def main():
    """Update evidence dashboard and README"""
    print("🔄 Updating Evidence Dashboard and README...")
    
    # Update evidence dashboard
    logger = EvidenceLogger()
    logger.update_evidence_dashboard()
    
    # Update README
    success = update_readme_evidence_section()
    
    if success:
        print("✅ Evidence dashboard and README updated successfully")
        print(f"📊 View dashboard: cat EVIDENCE_DASHBOARD.md")
    else:
        print("❌ Failed to update README")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())