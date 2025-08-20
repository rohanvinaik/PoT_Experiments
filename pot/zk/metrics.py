"""
Monitoring and metrics collection for ZK proof system.

This module provides comprehensive monitoring of proof generation performance,
verification success rates, and system health metrics.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class ProofMetric:
    """Single proof generation metric."""
    timestamp: float
    proof_type: str  # "sgd", "lora", "aggregated"
    generation_time_ms: float
    proof_size_bytes: int
    success: bool
    circuit_constraints: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationMetric:
    """Single proof verification metric."""
    timestamp: float
    proof_type: str
    verification_time_ms: float
    success: bool
    batch_size: int = 1
    error: Optional[str] = None


@dataclass
class SystemMetric:
    """System performance metric."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    disk_io_mb: float
    network_io_mb: float
    active_threads: int
    queue_size: int


class MetricsCollector:
    """
    Collects and aggregates metrics for the ZK proof system.
    """
    
    def __init__(self, 
                 window_size: int = 1000,
                 persist_to_file: bool = True,
                 metrics_dir: Optional[Path] = None):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Size of sliding window for metrics
            persist_to_file: Whether to persist metrics to disk
            metrics_dir: Directory for metrics files
        """
        self.window_size = window_size
        self.persist_to_file = persist_to_file
        self.metrics_dir = metrics_dir or Path("./zk_metrics")
        
        if persist_to_file:
            self.metrics_dir.mkdir(exist_ok=True)
        
        # Metrics storage (sliding windows)
        self.proof_metrics: deque = deque(maxlen=window_size)
        self.verification_metrics: deque = deque(maxlen=window_size)
        self.system_metrics: deque = deque(maxlen=window_size)
        
        # Aggregated statistics
        self.stats = {
            'total_proofs': 0,
            'successful_proofs': 0,
            'failed_proofs': 0,
            'total_verifications': 0,
            'successful_verifications': 0,
            'failed_verifications': 0,
            'total_proof_time_ms': 0,
            'total_verification_time_ms': 0,
            'proof_types': defaultdict(int),
            'errors': defaultdict(int)
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Start time
        self.start_time = time.time()
    
    def record_proof(self, metric: ProofMetric):
        """Record a proof generation metric."""
        with self.lock:
            self.proof_metrics.append(metric)
            self.stats['total_proofs'] += 1
            
            if metric.success:
                self.stats['successful_proofs'] += 1
            else:
                self.stats['failed_proofs'] += 1
                if metric.error:
                    self.stats['errors'][metric.error] += 1
            
            self.stats['total_proof_time_ms'] += metric.generation_time_ms
            self.stats['proof_types'][metric.proof_type] += 1
            
            # Persist if enabled
            if self.persist_to_file:
                self._persist_metric('proof', metric)
    
    def record_verification(self, metric: VerificationMetric):
        """Record a proof verification metric."""
        with self.lock:
            self.verification_metrics.append(metric)
            self.stats['total_verifications'] += metric.batch_size
            
            if metric.success:
                self.stats['successful_verifications'] += metric.batch_size
            else:
                self.stats['failed_verifications'] += metric.batch_size
                if metric.error:
                    self.stats['errors'][metric.error] += 1
            
            self.stats['total_verification_time_ms'] += metric.verification_time_ms
            
            # Persist if enabled
            if self.persist_to_file:
                self._persist_metric('verification', metric)
    
    def record_system(self, metric: SystemMetric):
        """Record a system performance metric."""
        with self.lock:
            self.system_metrics.append(metric)
            
            # Persist if enabled
            if self.persist_to_file:
                self._persist_metric('system', metric)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self.lock:
            uptime = time.time() - self.start_time
            
            summary = {
                'uptime_seconds': uptime,
                'total_proofs': self.stats['total_proofs'],
                'proof_success_rate': self._safe_divide(
                    self.stats['successful_proofs'], 
                    self.stats['total_proofs']
                ),
                'total_verifications': self.stats['total_verifications'],
                'verification_success_rate': self._safe_divide(
                    self.stats['successful_verifications'],
                    self.stats['total_verifications']
                ),
                'avg_proof_time_ms': self._safe_divide(
                    self.stats['total_proof_time_ms'],
                    self.stats['successful_proofs']
                ),
                'avg_verification_time_ms': self._safe_divide(
                    self.stats['total_verification_time_ms'],
                    self.stats['successful_verifications']
                ),
                'proofs_per_second': self._safe_divide(
                    self.stats['total_proofs'],
                    uptime
                ),
                'proof_types': dict(self.stats['proof_types']),
                'top_errors': self._get_top_errors()
            }
            
            # Add recent performance metrics
            if self.proof_metrics:
                recent_proofs = list(self.proof_metrics)[-100:]
                summary['recent_avg_proof_time_ms'] = np.mean([
                    m.generation_time_ms for m in recent_proofs
                ])
            
            if self.system_metrics:
                recent_system = list(self.system_metrics)[-100:]
                summary['avg_cpu_percent'] = np.mean([
                    m.cpu_percent for m in recent_system
                ])
                summary['avg_memory_mb'] = np.mean([
                    m.memory_mb for m in recent_system
                ])
            
            return summary
    
    def get_time_series(self, 
                       metric_type: str,
                       duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """
        Get time series data for a metric type.
        
        Args:
            metric_type: "proof", "verification", or "system"
            duration_minutes: Duration to retrieve
            
        Returns:
            List of metrics within the time window
        """
        cutoff_time = time.time() - (duration_minutes * 60)
        
        with self.lock:
            if metric_type == "proof":
                metrics = self.proof_metrics
            elif metric_type == "verification":
                metrics = self.verification_metrics
            elif metric_type == "system":
                metrics = self.system_metrics
            else:
                return []
            
            return [
                asdict(m) for m in metrics
                if m.timestamp >= cutoff_time
            ]
    
    def _persist_metric(self, metric_type: str, metric: Any):
        """Persist metric to file."""
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            file_path = self.metrics_dir / f"{metric_type}_{date_str}.jsonl"
            
            with open(file_path, 'a') as f:
                json.dump(asdict(metric), f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to persist metric: {e}")
    
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Safe division with zero check."""
        return numerator / denominator if denominator > 0 else 0.0
    
    def _get_top_errors(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get top errors by frequency."""
        return sorted(
            self.stats['errors'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]


class PerformanceMonitor:
    """
    Real-time performance monitoring for ZK proof system.
    """
    
    def __init__(self, 
                 update_interval: float = 1.0,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize performance monitor.
        
        Args:
            update_interval: Update interval in seconds
            alert_thresholds: Thresholds for alerts
        """
        self.update_interval = update_interval
        self.alert_thresholds = alert_thresholds or {
            'proof_time_ms': 10000,  # 10 seconds
            'verification_time_ms': 1000,  # 1 second
            'cpu_percent': 90,
            'memory_mb': 4096,
            'failure_rate': 0.1
        }
        
        self.metrics_collector = MetricsCollector()
        self.monitoring = False
        self.monitor_thread = None
        self.alerts: List[Dict[str, Any]] = []
    
    def start(self):
        """Start monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect system metrics
                system_metric = self._collect_system_metrics()
                self.metrics_collector.record_system(system_metric)
                
                # Check for alerts
                self._check_alerts()
                
                # Sleep
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
    
    def _collect_system_metrics(self) -> SystemMetric:
        """Collect current system metrics."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            
            # Get disk I/O if available
            try:
                disk_io = psutil.disk_io_counters()
                disk_io_mb = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)
            except:
                disk_io_mb = 0
            
            # Get network I/O if available
            try:
                net_io = psutil.net_io_counters()
                network_io_mb = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)
            except:
                network_io_mb = 0
            
            # Get thread count
            active_threads = threading.active_count()
            
            # Estimate queue size (would need actual queue reference)
            queue_size = 0
            
        except ImportError:
            # psutil not available, use dummy values
            cpu_percent = 0
            memory_mb = 0
            disk_io_mb = 0
            network_io_mb = 0
            active_threads = threading.active_count()
            queue_size = 0
        
        return SystemMetric(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            disk_io_mb=disk_io_mb,
            network_io_mb=network_io_mb,
            active_threads=active_threads,
            queue_size=queue_size
        )
    
    def _check_alerts(self):
        """Check for alert conditions."""
        summary = self.metrics_collector.get_summary()
        
        # Check proof time
        if summary.get('avg_proof_time_ms', 0) > self.alert_thresholds['proof_time_ms']:
            self._add_alert('high_proof_time', f"Avg proof time: {summary['avg_proof_time_ms']:.0f}ms")
        
        # Check verification time
        if summary.get('avg_verification_time_ms', 0) > self.alert_thresholds['verification_time_ms']:
            self._add_alert('high_verification_time', f"Avg verification time: {summary['avg_verification_time_ms']:.0f}ms")
        
        # Check failure rate
        failure_rate = 1.0 - summary.get('proof_success_rate', 1.0)
        if failure_rate > self.alert_thresholds['failure_rate']:
            self._add_alert('high_failure_rate', f"Failure rate: {failure_rate:.1%}")
        
        # Check system metrics
        if self.metrics_collector.system_metrics:
            latest_system = self.metrics_collector.system_metrics[-1]
            
            if latest_system.cpu_percent > self.alert_thresholds['cpu_percent']:
                self._add_alert('high_cpu', f"CPU: {latest_system.cpu_percent:.1f}%")
            
            if latest_system.memory_mb > self.alert_thresholds['memory_mb']:
                self._add_alert('high_memory', f"Memory: {latest_system.memory_mb:.0f}MB")
    
    def _add_alert(self, alert_type: str, message: str):
        """Add an alert."""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'message': message
        }
        
        self.alerts.append(alert)
        logger.warning(f"Alert: {alert_type} - {message}")
        
        # Keep only recent alerts
        cutoff = time.time() - 3600  # 1 hour
        self.alerts = [a for a in self.alerts if a['timestamp'] > cutoff]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        summary = self.metrics_collector.get_summary()
        
        return {
            'summary': summary,
            'recent_proofs': self.metrics_collector.get_time_series('proof', 10),
            'recent_verifications': self.metrics_collector.get_time_series('verification', 10),
            'system_metrics': self.metrics_collector.get_time_series('system', 10),
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'monitoring': self.monitoring
        }


# Global monitor instance
_global_monitor = None


def get_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
        _global_monitor.start()
    return _global_monitor


def record_proof_generation(proof_type: str,
                           generation_time_ms: float,
                           proof_size: int,
                           success: bool,
                           **kwargs):
    """Convenience function to record proof generation."""
    monitor = get_monitor()
    
    metric = ProofMetric(
        timestamp=time.time(),
        proof_type=proof_type,
        generation_time_ms=generation_time_ms,
        proof_size_bytes=proof_size,
        success=success,
        metadata=kwargs
    )
    
    monitor.metrics_collector.record_proof(metric)


def record_proof_verification(proof_type: str,
                             verification_time_ms: float,
                             success: bool,
                             batch_size: int = 1,
                             **kwargs):
    """Convenience function to record proof verification."""
    monitor = get_monitor()
    
    metric = VerificationMetric(
        timestamp=time.time(),
        proof_type=proof_type,
        verification_time_ms=verification_time_ms,
        success=success,
        batch_size=batch_size
    )
    
    monitor.metrics_collector.record_verification(metric)