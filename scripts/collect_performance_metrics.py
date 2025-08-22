#!/usr/bin/env python3
"""Collect detailed performance metrics for audit-grade reporting."""

import subprocess
import time
import psutil
import json
import sys
import os
from datetime import datetime

def get_system_metrics():
    """Collect current system metrics."""
    process = psutil.Process()
    
    # Memory metrics
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    
    # Page faults (on macOS, use resource module)
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        page_faults = {
            'minor': usage.ru_minflt,
            'major': usage.ru_majflt
        }
    except:
        page_faults = {'minor': 0, 'major': 0}
    
    # Disk I/O (system-wide)
    disk_io = psutil.disk_io_counters()
    
    return {
        'timestamp': time.time(),
        'rss_mb': rss_mb,
        'page_faults': page_faults,
        'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
        'disk_write_bytes': disk_io.write_bytes if disk_io else 0,
        'cpu_percent': process.cpu_percent(),
    }

def run_validation_with_metrics(ref_model='gpt2', cand_model='gpt2', mode='quick', max_queries=10):
    """Run validation while collecting metrics."""
    
    print(f"Starting performance monitoring for {ref_model} vs {cand_model}")
    
    # Collect baseline metrics
    baseline = get_system_metrics()
    
    # Start the validation process
    cmd = [
        'python', 'scripts/run_e2e_validation.py',
        '--ref-model', ref_model,
        '--cand-model', cand_model,
        '--mode', mode,
        '--max-queries', str(max_queries)
    ]
    
    metrics_timeline = []
    query_times = []
    
    # Start process
    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              universal_newlines=True, bufsize=1)
    
    # Monitor process
    query_count = 0
    last_query_time = start_time
    
    try:
        while True:
            # Check if process is still running
            if process.poll() is not None:
                break
                
            # Collect metrics every 0.5 seconds
            current_metrics = get_system_metrics()
            metrics_timeline.append(current_metrics)
            
            # Check for query completion in output
            line = process.stdout.readline()
            if line:
                print(line.strip())
                if 'Setting `pad_token_id`' in line:
                    query_count += 1
                    current_time = time.time()
                    query_duration = current_time - last_query_time
                    query_times.append({
                        'query_num': query_count,
                        'duration': query_duration,
                        'is_cold': query_count <= 2  # First 2 queries are "cold"
                    })
                    last_query_time = current_time
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        process.terminate()
        
    # Wait for process to complete
    process.wait()
    end_time = time.time()
    
    # Final metrics
    final = get_system_metrics()
    
    # Calculate statistics
    peak_rss = max(m['rss_mb'] for m in metrics_timeline) if metrics_timeline else final['rss_mb']
    total_page_faults = {
        'minor': final['page_faults']['minor'] - baseline['page_faults']['minor'],
        'major': final['page_faults']['major'] - baseline['page_faults']['major']
    }
    
    disk_read_throughput = (final['disk_read_bytes'] - baseline['disk_read_bytes']) / (end_time - start_time) / (1024 * 1024)  # MB/s
    
    # Separate cold and warm query times
    cold_queries = [q for q in query_times if q['is_cold']]
    warm_queries = [q for q in query_times if not q['is_cold']]
    
    avg_cold_time = sum(q['duration'] for q in cold_queries) / len(cold_queries) if cold_queries else 0
    avg_warm_time = sum(q['duration'] for q in warm_queries) / len(warm_queries) if warm_queries else 0
    
    report = {
        'test_config': {
            'ref_model': ref_model,
            'cand_model': cand_model,
            'mode': mode,
            'max_queries': max_queries
        },
        'performance_metrics': {
            'total_duration_seconds': end_time - start_time,
            'peak_rss_mb': peak_rss,
            'baseline_rss_mb': baseline['rss_mb'],
            'rss_growth_mb': final['rss_mb'] - baseline['rss_mb'],
            'page_faults': total_page_faults,
            'disk_read_throughput_mb_s': disk_read_throughput,
            'query_metrics': {
                'total_queries': len(query_times),
                'avg_cold_query_seconds': avg_cold_time,
                'avg_warm_query_seconds': avg_warm_time,
                'cold_warm_ratio': avg_cold_time / avg_warm_time if avg_warm_time > 0 else 0
            }
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Save report
    output_file = f'experimental_results/performance_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    os.makedirs('experimental_results', exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("PERFORMANCE METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"Peak RSS: {peak_rss:.2f} MB")
    print(f"Page Faults: Major={total_page_faults['major']}, Minor={total_page_faults['minor']}")
    print(f"Disk Read Throughput: {disk_read_throughput:.2f} MB/s")
    print(f"Avg Query Time: Cold={avg_cold_time:.3f}s, Warm={avg_warm_time:.3f}s")
    print(f"Cold/Warm Ratio: {avg_cold_time/avg_warm_time if avg_warm_time > 0 else 0:.2f}x")
    print(f"\nFull report saved to: {output_file}")
    
    return report

if __name__ == "__main__":
    # Run a quick test
    report = run_validation_with_metrics(
        ref_model='gpt2',
        cand_model='gpt2', 
        mode='quick',
        max_queries=10
    )