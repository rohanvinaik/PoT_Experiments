#!/usr/bin/env python3
"""
Results History Tracker for PoT Framework Validation
Maintains a comprehensive history of all validation runs and computes rolling averages.
"""

import json
import os
import glob
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple
from statistics import mean, stdev
import numpy as np


class ValidationResultsHistory:
    """Track and analyze validation results history."""
    
    def __init__(self, history_file: str = "validation_results_history.json"):
        """Initialize results history tracker."""
        self.history_file = history_file
        self.history = self.load_history()
    
    def load_history(self) -> Dict[str, Any]:
        """Load existing results history."""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {
            'metadata': {
                'created': datetime.now().isoformat(),
                'total_runs': 0,
                'last_updated': None
            },
            'runs': [],
            'statistics': {}
        }
    
    def save_history(self):
        """Save results history to file."""
        self.history['metadata']['last_updated'] = datetime.now().isoformat()
        self.history['metadata']['total_runs'] = len(self.history['runs'])
        
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def collect_new_results(self) -> List[Dict[str, Any]]:
        """Collect new validation results from files."""
        new_results = []
        
        # Collect deterministic validation results
        deterministic_files = glob.glob("reliable_validation_results_*.json")
        for file_path in deterministic_files:
            if self.is_new_result(file_path):
                result = self.parse_deterministic_result(file_path)
                if result:
                    new_results.append(result)
        
        # Collect legacy validation results  
        legacy_files = glob.glob("experimental_results/validation_results_*.json")
        for file_path in legacy_files:
            if self.is_new_result(file_path):
                result = self.parse_legacy_result(file_path)
                if result:
                    new_results.append(result)
        
        return new_results
    
    def is_new_result(self, file_path: str) -> bool:
        """Check if result file is new (not in history)."""
        existing_files = {run.get('source_file') for run in self.history['runs']}
        return file_path not in existing_files
    
    def parse_deterministic_result(self, file_path: str) -> Dict[str, Any]:
        """Parse deterministic validation result file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            validation_run = data.get('validation_run', {})
            timestamp = validation_run.get('timestamp')
            config = validation_run.get('config', {})
            
            # Extract performance metrics
            verification_times = []
            success_rates = []
            confidence_scores = []
            
            for test in validation_run.get('tests', []):
                if test['test_name'] == 'reliable_verification':
                    for result in test.get('results', []):
                        for depth in result.get('depths', []):
                            verification_times.append(depth.get('duration', 0))
                            success_rates.append(1.0 if depth.get('verified') else 0.0)
                            confidence_scores.append(depth.get('confidence', 0))
                
                elif test['test_name'] == 'performance_benchmark':
                    for result in test.get('results', []):
                        if 'verification_time' in result:
                            batch_time = result.get('verification_time', 0)
                            model_count = result.get('model_count', 1)
                            if model_count > 0:
                                verification_times.append(batch_time / model_count)
                        success_rates.append(result.get('success_rate', 0))
            
            return {
                'timestamp': timestamp,
                'source_file': file_path,
                'validation_type': 'deterministic',
                'seed': config.get('model_seed', 42),
                'model_count': config.get('model_count', 0),
                'metrics': {
                    'success_rate': mean(success_rates) if success_rates else 0,
                    'avg_verification_time': mean(verification_times) if verification_times else 0,
                    'min_verification_time': min(verification_times) if verification_times else 0,
                    'max_verification_time': max(verification_times) if verification_times else 0,
                    'avg_confidence': mean(confidence_scores) if confidence_scores else 0,
                    'verification_count': len(verification_times)
                }
            }
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def parse_legacy_result(self, file_path: str) -> Dict[str, Any]:
        """Parse legacy validation result file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract timestamp from filename
            timestamp_match = re.search(r'(\d{8}_\d{6})', file_path)
            timestamp = timestamp_match.group(1) if timestamp_match else None
            
            experiments = data.get('experiments', [])
            success_count = sum(1 for exp in experiments if 'data' in exp and exp['data'])
            total_count = len(experiments)
            
            return {
                'timestamp': timestamp,
                'source_file': file_path,
                'validation_type': 'legacy',
                'seed': 'random',
                'model_count': total_count,
                'metrics': {
                    'success_rate': success_count / total_count if total_count > 0 else 0,
                    'avg_verification_time': None,  # Not available in legacy
                    'min_verification_time': None,
                    'max_verification_time': None,
                    'avg_confidence': None,
                    'verification_count': total_count
                }
            }
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def update_statistics(self):
        """Update rolling statistics from all runs."""
        if not self.history['runs']:
            return
        
        # Separate deterministic and legacy results
        deterministic_runs = [r for r in self.history['runs'] if r['validation_type'] == 'deterministic']
        legacy_runs = [r for r in self.history['runs'] if r['validation_type'] == 'legacy']
        
        # Calculate deterministic statistics
        if deterministic_runs:
            det_success_rates = [r['metrics']['success_rate'] for r in deterministic_runs]
            det_times = [r['metrics']['avg_verification_time'] for r in deterministic_runs 
                        if r['metrics']['avg_verification_time'] is not None]
            det_confidences = [r['metrics']['avg_confidence'] for r in deterministic_runs
                              if r['metrics']['avg_confidence'] is not None]
            
            self.history['statistics']['deterministic'] = {
                'total_runs': len(deterministic_runs),
                'avg_success_rate': mean(det_success_rates),
                'success_rate_std': stdev(det_success_rates) if len(det_success_rates) > 1 else 0,
                'avg_verification_time': mean(det_times) if det_times else None,
                'verification_time_std': stdev(det_times) if len(det_times) > 1 else 0,
                'avg_confidence': mean(det_confidences) if det_confidences else None,
                'min_verification_time': min(det_times) if det_times else None,
                'max_verification_time': max(det_times) if det_times else None,
                'recent_10_success_rate': mean(det_success_rates[-10:]) if len(det_success_rates) >= 10 else mean(det_success_rates)
            }
        
        # Calculate legacy statistics
        if legacy_runs:
            leg_success_rates = [r['metrics']['success_rate'] for r in legacy_runs]
            
            self.history['statistics']['legacy'] = {
                'total_runs': len(legacy_runs),
                'avg_success_rate': mean(leg_success_rates),
                'success_rate_std': stdev(leg_success_rates) if len(leg_success_rates) > 1 else 0,
                'recent_10_success_rate': mean(leg_success_rates[-10:]) if len(leg_success_rates) >= 10 else mean(leg_success_rates)
            }
        
        # Calculate overall statistics
        all_success_rates = [r['metrics']['success_rate'] for r in self.history['runs']]
        all_times = [r['metrics']['avg_verification_time'] for r in self.history['runs']
                    if r['metrics']['avg_verification_time'] is not None]
        
        self.history['statistics']['overall'] = {
            'total_runs': len(self.history['runs']),
            'avg_success_rate': mean(all_success_rates),
            'avg_verification_time': mean(all_times) if all_times else None,
            'date_range': {
                'earliest': min(r['timestamp'] for r in self.history['runs'] if r['timestamp']),
                'latest': max(r['timestamp'] for r in self.history['runs'] if r['timestamp'])
            }
        }
    
    def add_results(self, new_results: List[Dict[str, Any]]):
        """Add new results to history."""
        self.history['runs'].extend(new_results)
        # Sort by timestamp
        self.history['runs'].sort(key=lambda x: x['timestamp'] or '')
    
    def generate_summary_report(self) -> str:
        """Generate summary report of all validation results."""
        stats = self.history['statistics']
        
        report = []
        report.append("# PoT Framework Validation Results History")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Validation Runs:** {self.history['metadata']['total_runs']}")
        report.append("")
        
        # Deterministic results
        if 'deterministic' in stats:
            det = stats['deterministic']
            report.append("## 🎯 Current Deterministic Framework Results")
            report.append(f"- **Total Runs:** {det['total_runs']}")
            report.append(f"- **Average Success Rate:** {det['avg_success_rate']:.1%} ± {det['success_rate_std']:.1%}")
            report.append(f"- **Recent 10 Runs Success Rate:** {det['recent_10_success_rate']:.1%}")
            if det['avg_verification_time']:
                report.append(f"- **Average Verification Time:** {det['avg_verification_time']:.6f}s")
                report.append(f"- **Verification Time Range:** {det['min_verification_time']:.6f}s - {det['max_verification_time']:.6f}s")
            if det['avg_confidence']:
                report.append(f"- **Average Confidence:** {det['avg_confidence']:.1%}")
            report.append("")
        
        # Legacy results
        if 'legacy' in stats:
            leg = stats['legacy']
            report.append("## 📚 Legacy Random Model Results")
            report.append(f"- **Total Runs:** {leg['total_runs']}")
            report.append(f"- **Average Success Rate:** {leg['avg_success_rate']:.1%} ± {leg['success_rate_std']:.1%}")
            report.append(f"- **Recent 10 Runs Success Rate:** {leg['recent_10_success_rate']:.1%}")
            report.append("")
        
        # Overall summary
        if 'overall' in stats:
            overall = stats['overall']
            report.append("## 📊 Overall Validation Summary")
            report.append(f"- **Combined Success Rate:** {overall['avg_success_rate']:.1%}")
            if overall['avg_verification_time']:
                report.append(f"- **Overall Average Time:** {overall['avg_verification_time']:.6f}s")
            if overall['date_range']['earliest']:
                report.append(f"- **Validation Period:** {overall['date_range']['earliest']} to {overall['date_range']['latest']}")
            report.append("")
        
        return "\n".join(report)
    
    def get_readme_metrics(self) -> Dict[str, str]:
        """Get metrics formatted for README update."""
        stats = self.history['statistics']
        
        if 'deterministic' in stats:
            det = stats['deterministic']
            success_rate = f"{det['avg_success_rate']:.1%}"
            if det['total_runs'] > 1:
                success_rate += f" (±{det['success_rate_std']:.1%})"
            
            verification_time = "N/A"
            if det['avg_verification_time']:
                verification_time = f"{det['avg_verification_time']:.6f}s"
                if det['verification_time_std'] > 0:
                    verification_time += f" (±{det['verification_time_std']:.6f}s)"
            
            return {
                'validation_success': f"{success_rate} ({det['total_runs']} runs)",
                'verification_time': verification_time,
                'total_runs': str(det['total_runs']),
                'recent_success': f"{det['recent_10_success_rate']:.1%}"
            }
        
        return {
            'validation_success': "No data available",
            'verification_time': "N/A",
            'total_runs': "0",
            'recent_success': "N/A"
        }


def main():
    """Main function to update results history."""
    print("Updating PoT validation results history...")
    
    # Initialize history tracker
    tracker = ValidationResultsHistory()
    
    # Collect new results
    new_results = tracker.collect_new_results()
    
    if new_results:
        print(f"Found {len(new_results)} new validation results")
        tracker.add_results(new_results)
        tracker.update_statistics()
        tracker.save_history()
        
        # Generate summary report
        summary = tracker.generate_summary_report()
        with open("VALIDATION_RESULTS_SUMMARY.md", "w") as f:
            f.write(summary)
        
        print(f"Updated history with {len(new_results)} new results")
        print(f"Total runs in history: {tracker.history['metadata']['total_runs']}")
        
        # Print current metrics
        metrics = tracker.get_readme_metrics()
        print("\nCurrent Metrics for README:")
        print(f"- Validation Success: {metrics['validation_success']}")
        print(f"- Verification Time: {metrics['verification_time']}")
        print(f"- Total Runs: {metrics['total_runs']}")
        print(f"- Recent Success: {metrics['recent_success']}")
        
    else:
        print("No new validation results found")
    
    print(f"Results history saved to: {tracker.history_file}")
    print("Summary report saved to: VALIDATION_RESULTS_SUMMARY.md")


if __name__ == "__main__":
    main()