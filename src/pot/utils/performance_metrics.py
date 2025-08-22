"""
Enhanced Performance Metrics Collection for Evidence Bundles

This module provides comprehensive performance metric collection including:
- Memory usage (RSS, page faults)
- Query timing (cold vs warm)
- Disk I/O throughput
- CPU utilization
"""

import time
import psutil
import resource
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Metrics for individual query execution"""
    query_id: int
    start_time: float
    end_time: float
    duration: float
    is_cold: bool  # First N queries are "cold"
    memory_before: float  # MB
    memory_after: float  # MB
    
@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    page_faults_major: int
    page_faults_minor: int
    disk_read_bytes: int
    disk_write_bytes: int
    
@dataclass
class PerformanceSnapshot:
    """Complete performance snapshot at a point in time"""
    peak_rss_mb: float
    baseline_rss_mb: float
    rss_growth_mb: float
    page_faults: Dict[str, int]
    disk_throughput_mb_s: float
    query_metrics: Dict[str, Any]
    system_timeline: List[Dict[str, Any]]

class PerformanceMonitor:
    """Comprehensive performance monitoring for PoT validation"""
    
    def __init__(self, cold_query_count: int = 2):
        """Initialize performance monitor
        
        Args:
            cold_query_count: Number of initial queries considered "cold"
        """
        self.cold_query_count = cold_query_count
        self.query_metrics: List[QueryMetrics] = []
        self.system_timeline: List[SystemMetrics] = []
        self.process = psutil.Process()
        
        # Baseline metrics
        self.start_time = time.time()
        self.baseline_metrics = self._capture_system_metrics()
        self.peak_rss = 0.0
        
        # Query tracking
        self.query_count = 0
        self.current_query_start = None
        
    def _capture_system_metrics(self) -> SystemMetrics:
        """Capture current system metrics"""
        
        # Memory metrics
        mem_info = self.process.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)
        vms_mb = mem_info.vms / (1024 * 1024) if hasattr(mem_info, 'vms') else 0
        
        # Update peak RSS
        self.peak_rss = max(self.peak_rss, rss_mb)
        
        # Page faults (platform-specific)
        try:
            if hasattr(resource, 'getrusage'):
                usage = resource.getrusage(resource.RUSAGE_SELF)
                page_faults_major = usage.ru_majflt
                page_faults_minor = usage.ru_minflt
            else:
                # Windows or other platforms
                page_faults_major = 0
                page_faults_minor = 0
        except:
            page_faults_major = 0
            page_faults_minor = 0
        
        # Disk I/O (system-wide)
        try:
            disk_io = psutil.disk_io_counters()
            disk_read = disk_io.read_bytes if disk_io else 0
            disk_write = disk_io.write_bytes if disk_io else 0
        except:
            disk_read = 0
            disk_write = 0
        
        # CPU usage
        try:
            cpu_percent = self.process.cpu_percent()
        except:
            cpu_percent = 0.0
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_rss_mb=rss_mb,
            memory_vms_mb=vms_mb,
            page_faults_major=page_faults_major,
            page_faults_minor=page_faults_minor,
            disk_read_bytes=disk_read,
            disk_write_bytes=disk_write
        )
    
    def start_query(self) -> None:
        """Mark the start of a query execution"""
        self.current_query_start = time.time()
        self.query_count += 1
        
    def end_query(self) -> QueryMetrics:
        """Mark the end of a query execution and return metrics"""
        
        if self.current_query_start is None:
            logger.warning("end_query called without matching start_query")
            return None
        
        end_time = time.time()
        duration = end_time - self.current_query_start
        
        # Capture memory state
        current_metrics = self._capture_system_metrics()
        
        # Determine if this is a cold query
        is_cold = self.query_count <= self.cold_query_count
        
        query_metric = QueryMetrics(
            query_id=self.query_count,
            start_time=self.current_query_start,
            end_time=end_time,
            duration=duration,
            is_cold=is_cold,
            memory_before=self.baseline_metrics.memory_rss_mb if self.query_count == 1 
                         else self.system_timeline[-1].memory_rss_mb if self.system_timeline 
                         else self.baseline_metrics.memory_rss_mb,
            memory_after=current_metrics.memory_rss_mb
        )
        
        self.query_metrics.append(query_metric)
        self.system_timeline.append(current_metrics)
        self.current_query_start = None
        
        return query_metric
    
    def record_checkpoint(self, label: str = "checkpoint") -> None:
        """Record a system metrics checkpoint"""
        metrics = self._capture_system_metrics()
        self.system_timeline.append(metrics)
        logger.debug(f"Performance checkpoint '{label}': RSS={metrics.memory_rss_mb:.1f}MB")
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get aggregated query statistics"""
        
        if not self.query_metrics:
            return {
                "total_queries": 0,
                "avg_cold_query_seconds": 0,
                "avg_warm_query_seconds": 0,
                "cold_warm_ratio": 0
            }
        
        cold_queries = [q for q in self.query_metrics if q.is_cold]
        warm_queries = [q for q in self.query_metrics if not q.is_cold]
        
        avg_cold = sum(q.duration for q in cold_queries) / len(cold_queries) if cold_queries else 0
        avg_warm = sum(q.duration for q in warm_queries) / len(warm_queries) if warm_queries else 0
        
        # Per-query memory growth
        memory_growth_per_query = []
        for q in self.query_metrics:
            growth = q.memory_after - q.memory_before
            memory_growth_per_query.append(growth)
        
        avg_memory_growth = sum(memory_growth_per_query) / len(memory_growth_per_query) if memory_growth_per_query else 0
        
        return {
            "total_queries": len(self.query_metrics),
            "cold_queries": len(cold_queries),
            "warm_queries": len(warm_queries),
            "avg_cold_query_seconds": avg_cold,
            "avg_warm_query_seconds": avg_warm,
            "cold_warm_ratio": avg_cold / avg_warm if avg_warm > 0 else 0,
            "min_query_time": min(q.duration for q in self.query_metrics) if self.query_metrics else 0,
            "max_query_time": max(q.duration for q in self.query_metrics) if self.query_metrics else 0,
            "avg_memory_growth_mb": avg_memory_growth,
            "query_durations": [q.duration for q in self.query_metrics]
        }
    
    def get_performance_snapshot(self) -> PerformanceSnapshot:
        """Get complete performance snapshot"""
        
        # Calculate totals
        final_metrics = self._capture_system_metrics()
        elapsed_time = time.time() - self.start_time
        
        # Page fault deltas
        page_faults = {
            "major": final_metrics.page_faults_major - self.baseline_metrics.page_faults_major,
            "minor": final_metrics.page_faults_minor - self.baseline_metrics.page_faults_minor
        }
        
        # Disk throughput
        disk_read_delta = final_metrics.disk_read_bytes - self.baseline_metrics.disk_read_bytes
        disk_write_delta = final_metrics.disk_write_bytes - self.baseline_metrics.disk_write_bytes
        disk_throughput = (disk_read_delta + disk_write_delta) / (1024 * 1024) / elapsed_time if elapsed_time > 0 else 0
        
        return PerformanceSnapshot(
            peak_rss_mb=self.peak_rss,
            baseline_rss_mb=self.baseline_metrics.memory_rss_mb,
            rss_growth_mb=self.peak_rss - self.baseline_metrics.memory_rss_mb,
            page_faults=page_faults,
            disk_throughput_mb_s=disk_throughput,
            query_metrics=self.get_query_statistics(),
            system_timeline=[asdict(m) for m in self.system_timeline[-100:]]  # Keep last 100 samples
        )
    
    def export_metrics(self, filepath: Path) -> None:
        """Export performance metrics to JSON file"""
        
        snapshot = self.get_performance_snapshot()
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": time.time() - self.start_time,
            "performance_metrics": asdict(snapshot),
            "query_details": [asdict(q) for q in self.query_metrics],
            "system_info": {
                "platform": psutil.LINUX if hasattr(psutil, 'LINUX') else "unknown",
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Performance metrics exported to {filepath}")
    
    def get_evidence_metrics(self) -> Dict[str, Any]:
        """Get metrics formatted for evidence bundle"""
        
        snapshot = self.get_performance_snapshot()
        query_stats = snapshot.query_metrics
        
        return {
            "performance": {
                "peak_rss_mb": snapshot.peak_rss_mb,
                "baseline_rss_mb": snapshot.baseline_rss_mb,
                "rss_growth_mb": snapshot.rss_growth_mb,
                "page_faults_major": snapshot.page_faults.get("major", 0),
                "page_faults_minor": snapshot.page_faults.get("minor", 0),
                "disk_throughput_mb_s": snapshot.disk_throughput_mb_s,
                "total_queries": query_stats.get("total_queries", 0),
                "avg_cold_query_seconds": query_stats.get("avg_cold_query_seconds", 0),
                "avg_warm_query_seconds": query_stats.get("avg_warm_query_seconds", 0),
                "cold_warm_ratio": query_stats.get("cold_warm_ratio", 0),
                "query_time_range": {
                    "min": query_stats.get("min_query_time", 0),
                    "max": query_stats.get("max_query_time", 0)
                }
            },
            "system_timeline_samples": len(self.system_timeline),
            "measurement_duration": time.time() - self.start_time
        }


def create_performance_monitor() -> PerformanceMonitor:
    """Factory function to create performance monitor"""
    return PerformanceMonitor()


# Integration helper for existing pipeline
def monitor_validation_run(func):
    """Decorator to add performance monitoring to validation functions"""
    
    def wrapper(*args, **kwargs):
        monitor = create_performance_monitor()
        
        # Inject monitor into kwargs if function accepts it
        import inspect
        sig = inspect.signature(func)
        if 'performance_monitor' in sig.parameters:
            kwargs['performance_monitor'] = monitor
        
        try:
            # Run the function
            result = func(*args, **kwargs)
            
            # Add performance metrics to result if it's a dict
            if isinstance(result, dict):
                result['performance_metrics'] = monitor.get_evidence_metrics()
            
            return result
            
        finally:
            # Always export metrics
            output_dir = kwargs.get('output_dir', Path.cwd())
            if not isinstance(output_dir, Path):
                output_dir = Path(output_dir)
            
            metrics_file = output_dir / f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            monitor.export_metrics(metrics_file)
    
    return wrapper