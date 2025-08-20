#!/usr/bin/env python3
"""
Statistical Analysis of PoT Framework Validation Results
Provides independent verification of paper claims with confidence intervals
"""

import json
import numpy as np
from scipy import stats
from datetime import datetime

def analyze_validation_history():
    """Analyze complete validation history and generate statistical report"""
    
    print("📊 STATISTICAL ANALYSIS OF POT FRAMEWORK VALIDATION")
    print("=" * 60)
    
    # Load validation history
    try:
        with open('validation_results_history.json') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ validation_results_history.json not found")
        print("   Please run this script from the PoT_Experiments directory")
        return
    
    # Extract deterministic validation data
    det_stats = data['statistics']['deterministic']
    runs = [run for run in data['runs'] if run.get('validation_type') == 'deterministic']
    
    print(f"📈 DATASET OVERVIEW")
    print(f"   Total Deterministic Runs: {len(runs)}")
    print(f"   Date Range: {data['statistics']['overall']['date_range']['earliest']} to {data['metadata']['last_updated']}")
    print(f"   Framework Type: Deterministic Test Models")
    print()
    
    # Success Rate Analysis
    success_rates = [run['metrics']['success_rate'] for run in runs]
    avg_success = np.mean(success_rates)
    success_ci = stats.norm.interval(0.95, loc=avg_success, scale=np.std(success_rates)/np.sqrt(len(success_rates)))
    
    print(f"🎯 SUCCESS RATE ANALYSIS")
    print(f"   Sample Size: n = {len(success_rates)}")
    print(f"   Mean Success Rate: {avg_success:.1%}")
    print(f"   Standard Deviation: {np.std(success_rates):.4f}")
    print(f"   95% Confidence Interval: [{success_ci[0]:.1%}, {success_ci[1]:.1%}]")
    print(f"   Paper Claim (>95%): {'✅ EXCEEDED' if avg_success > 0.95 else '❌ NOT MET'}")
    print()
    
    # Timing Analysis
    times = [run['metrics']['avg_verification_time'] for run in runs if run['metrics']['avg_verification_time']]
    times_ms = np.array(times) * 1000  # Convert to milliseconds
    
    mean_time = np.mean(times_ms)
    std_time = np.std(times_ms)
    timing_ci = stats.norm.interval(0.95, loc=mean_time, scale=std_time/np.sqrt(len(times)))
    
    print(f"⚡ PERFORMANCE TIMING ANALYSIS")
    print(f"   Sample Size: n = {len(times)}")
    print(f"   Mean Verification Time: {mean_time:.3f}ms")
    print(f"   Standard Deviation: {std_time:.3f}ms")
    print(f"   95% Confidence Interval: [{timing_ci[0]:.3f}ms, {timing_ci[1]:.3f}ms]")
    print(f"   Coefficient of Variation: {(std_time/mean_time)*100:.1f}%")
    print(f"   Paper Claim (<1000ms): {'✅ EXCEEDED' if mean_time < 1000 else '❌ NOT MET'} ({1000/mean_time:.0f}x faster)")
    print()
    
    # Consistency Analysis
    cv = (std_time / mean_time) * 100
    print(f"📊 CONSISTENCY ANALYSIS")
    print(f"   Coefficient of Variation: {cv:.1f}%")
    print(f"   Timing Consistency: {'✅ EXCELLENT' if cv < 20 else '⚠️ MODERATE' if cv < 40 else '❌ POOR'}")
    print(f"   Min Time: {np.min(times_ms):.3f}ms")
    print(f"   Max Time: {np.max(times_ms):.3f}ms")
    print(f"   Range: {np.max(times_ms) - np.min(times_ms):.3f}ms")
    print()
    
    # Throughput Calculation
    throughput = 1000 / mean_time  # verifications per second
    print(f"🚀 THROUGHPUT ANALYSIS")
    print(f"   Theoretical Throughput: {throughput:.0f} verifications/second")
    print(f"   Conservative Estimate: {throughput * 0.7:.0f} verifications/second")
    print(f"   Production Capacity: {'✅ HIGH PERFORMANCE' if throughput > 1000 else '⚠️ MODERATE'}")
    print()
    
    # Statistical Tests
    print(f"🧪 STATISTICAL TESTS")
    
    # Test for normal distribution
    shapiro_stat, shapiro_p = stats.shapiro(times_ms)
    print(f"   Normality Test (Shapiro-Wilk):")
    print(f"     Statistic: {shapiro_stat:.4f}")
    print(f"     p-value: {shapiro_p:.6f}")
    print(f"     Distribution: {'✅ NORMAL' if shapiro_p > 0.05 else '⚠️ NON-NORMAL'}")
    
    # Test against paper claim
    claim_time = 1000  # 1 second in milliseconds
    t_stat, t_p = stats.ttest_1samp(times_ms, claim_time)
    print(f"   One-sample t-test vs 1000ms:")
    print(f"     t-statistic: {t_stat:.4f}")
    print(f"     p-value: {t_p:.10f}")
    print(f"     Significantly faster: {'✅ YES' if t_p < 0.001 and t_stat < 0 else '❌ NO'}")
    print()
    
    # Recent Performance Trend
    recent_runs = runs[-10:] if len(runs) >= 10 else runs
    recent_times = [run['metrics']['avg_verification_time']*1000 for run in recent_runs if run['metrics']['avg_verification_time']]
    recent_success = [run['metrics']['success_rate'] for run in recent_runs]
    
    print(f"📈 RECENT PERFORMANCE TREND (Last {len(recent_runs)} runs)")
    print(f"   Recent Mean Time: {np.mean(recent_times):.3f}ms")
    print(f"   Recent Success Rate: {np.mean(recent_success):.1%}")
    print(f"   Performance Stability: {'✅ STABLE' if np.std(recent_times) < std_time else '⚠️ VARIABLE'}")
    print()
    
    # Paper Claims Summary
    print(f"📋 PAPER CLAIMS VALIDATION SUMMARY")
    print(f"   Claim 1 - Speed (<1s): {'✅ VALIDATED' if mean_time < 1000 else '❌ FAILED'} ({mean_time:.3f}ms measured)")
    print(f"   Claim 2 - Accuracy (>95%): {'✅ VALIDATED' if avg_success > 0.95 else '❌ FAILED'} ({avg_success:.1%} measured)")
    print(f"   Claim 3 - Consistency: {'✅ VALIDATED' if cv < 30 else '❌ FAILED'} ({cv:.1f}% CV)")
    print(f"   Claim 4 - Production Ready: {'✅ VALIDATED' if throughput > 1000 else '❌ FAILED'} ({throughput:.0f}/sec)")
    print()
    
    # Confidence Assessment
    all_claims_met = (mean_time < 1000 and avg_success > 0.95 and cv < 30 and throughput > 1000)
    print(f"🎯 OVERALL VALIDATION STATUS")
    print(f"   All Claims Met: {'✅ YES' if all_claims_met else '❌ NO'}")
    print(f"   Statistical Confidence: {'🔥 HIGH' if all_claims_met and len(runs) >= 20 else '⚠️ MODERATE'}")
    print(f"   Recommendation: {'✅ READY FOR PUBLICATION' if all_claims_met else '⚠️ REQUIRES INVESTIGATION'}")
    
    return {
        'success_rate': avg_success,
        'avg_time_ms': mean_time,
        'consistency_cv': cv,
        'throughput': throughput,
        'all_claims_met': all_claims_met,
        'sample_size': len(runs)
    }

def analyze_latest_validation():
    """Analyze the most recent validation run"""
    
    print("\n" + "=" * 60)
    print("🔍 LATEST VALIDATION RUN ANALYSIS")
    print("=" * 60)
    
    try:
        with open('latest_validation_results.json') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ latest_validation_results.json not found")
        return
    
    validation_run = data['validation_run']
    timestamp = validation_run['timestamp']
    
    print(f"📅 Run Details")
    print(f"   Timestamp: {timestamp}")
    print(f"   Seed: {validation_run['config']['model_seed']}")
    print(f"   Models Tested: {validation_run['config']['model_count']}")
    print()
    
    # Analyze verification test
    verification_test = next(test for test in validation_run['tests'] if test['test_name'] == 'reliable_verification')
    result = verification_test['results'][0]
    
    for depth_result in result['depths']:
        depth = depth_result['depth']
        verified = depth_result['verified']
        confidence = depth_result['confidence']
        duration = depth_result['duration'] * 1000  # Convert to ms
        
        print(f"🧪 {depth.title()} Verification")
        print(f"   Verified: {'✅ YES' if verified else '❌ NO'}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Duration: {duration:.3f}ms")
        print(f"   Challenges: {depth_result['challenges_passed']}/{depth_result['challenges_total']}")
        print()
    
    # Analyze performance test
    perf_test = next(test for test in validation_run['tests'] if test['test_name'] == 'performance_benchmark')
    batch_result = perf_test['results'][0]
    
    print(f"🚀 Batch Performance")
    print(f"   Models Processed: {batch_result['model_count']}")
    print(f"   Batch Time: {batch_result['verification_time']*1000:.3f}ms")
    print(f"   Per-Model Time: {(batch_result['verification_time']/batch_result['model_count'])*1000:.3f}ms")
    print(f"   Success Rate: {batch_result['success_rate']:.1%}")
    print(f"   Registration Time: {batch_result['registration_time']*1000:.3f}ms")

if __name__ == "__main__":
    # Run complete statistical analysis
    results = analyze_validation_history()
    
    # Analyze latest run
    analyze_latest_validation()
    
    print("\n" + "=" * 60)
    print("📊 VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("All data independently verifiable from source JSON files.")