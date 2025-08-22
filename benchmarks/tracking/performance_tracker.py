#!/usr/bin/env python3
"""
Performance Tracking System

Comprehensive performance tracking and monitoring system for the PoT framework.
Tracks metrics over time, detects regressions, and provides performance insights.
"""

import json
import os
import sys
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import statistics
import threading
import psutil

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class PerformanceMetric:
    """Represents a single performance metric"""
    
    def __init__(self, name: str, value: float, unit: str = "", 
                 category: str = "general", timestamp: Optional[datetime] = None):
        self.name = name
        self.value = value
        self.unit = unit
        self.category = category
        self.timestamp = timestamp or datetime.utcnow()
        self.metadata = {}
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the metric"""
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'category': self.category,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetric':
        """Create from dictionary"""
        metric = cls(
            name=data['name'],
            value=data['value'],
            unit=data.get('unit', ''),
            category=data.get('category', 'general'),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )
        metric.metadata = data.get('metadata', {})
        return metric


class PerformanceSession:
    """Represents a performance measurement session"""
    
    def __init__(self, session_id: str, test_name: str = ""):
        self.session_id = session_id
        self.test_name = test_name
        self.start_time = datetime.utcnow()
        self.end_time = None
        self.metrics: List[PerformanceMetric] = []
        self.system_info = self._collect_system_info()
        self.status = "active"
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version,
                'platform': sys.platform,
                'architecture': os.uname().machine if hasattr(os, 'uname') else 'unknown'
            }
        except Exception:
            return {'error': 'Could not collect system info'}
    
    def add_metric(self, metric: PerformanceMetric):
        """Add a metric to this session"""
        self.metrics.append(metric)
    
    def finish(self):
        """Mark session as finished"""
        self.end_time = datetime.utcnow()
        self.status = "completed"
    
    def get_duration(self) -> float:
        """Get session duration in seconds"""
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()
    
    def get_metrics_by_category(self, category: str) -> List[PerformanceMetric]:
        """Get all metrics in a category"""
        return [m for m in self.metrics if m.category == category]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'session_id': self.session_id,
            'test_name': self.test_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.get_duration(),
            'status': self.status,
            'system_info': self.system_info,
            'metrics': [m.to_dict() for m in self.metrics]
        }


class PerformanceTracker:
    """Main performance tracking system"""
    
    def __init__(self, db_path: str = "benchmarks/tracking/performance.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.current_session: Optional[PerformanceSession] = None
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    test_name TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    duration_seconds REAL,
                    status TEXT,
                    system_info TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    name TEXT,
                    value REAL,
                    unit TEXT,
                    category TEXT,
                    timestamp TEXT,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_session 
                ON metrics (session_id)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics (name, timestamp)
            ''')
    
    def start_session(self, test_name: str = "") -> str:
        """Start a new performance tracking session"""
        session_id = f"{test_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = PerformanceSession(session_id, test_name)
        return session_id
    
    def end_session(self) -> Optional[str]:
        """End the current session"""
        if not self.current_session:
            return None
        
        self.current_session.finish()
        session_id = self.current_session.session_id
        
        # Save to database
        self._save_session(self.current_session)
        
        # Clear current session
        self.current_session = None
        
        return session_id
    
    def record_metric(self, name: str, value: float, unit: str = "", 
                     category: str = "general", **metadata) -> bool:
        """Record a performance metric"""
        if not self.current_session:
            return False
        
        metric = PerformanceMetric(name, value, unit, category)
        for key, val in metadata.items():
            metric.add_metadata(key, val)
        
        self.current_session.add_metric(metric)
        return True
    
    def record_timing(self, operation_name: str, duration_ms: float, **metadata) -> bool:
        """Record timing metric"""
        return self.record_metric(
            name=f"{operation_name}_time", 
            value=duration_ms, 
            unit="ms", 
            category="timing",
            **metadata
        )
    
    def record_memory(self, operation_name: str, memory_mb: float, **metadata) -> bool:
        """Record memory usage metric"""
        return self.record_metric(
            name=f"{operation_name}_memory", 
            value=memory_mb, 
            unit="MB", 
            category="memory",
            **metadata
        )
    
    def record_accuracy(self, test_name: str, accuracy: float, **metadata) -> bool:
        """Record accuracy metric"""
        return self.record_metric(
            name=f"{test_name}_accuracy", 
            value=accuracy, 
            unit="ratio", 
            category="accuracy",
            **metadata
        )
    
    def record_throughput(self, operation_name: str, ops_per_second: float, **metadata) -> bool:
        """Record throughput metric"""
        return self.record_metric(
            name=f"{operation_name}_throughput", 
            value=ops_per_second, 
            unit="ops/sec", 
            category="throughput",
            **metadata
        )
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start continuous system monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_system, 
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous system monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_system(self, interval_seconds: float):
        """Monitor system resources continuously"""
        while self.monitoring_active and self.current_session:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.record_metric("system_cpu_usage", cpu_percent, "percent", "system")
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.record_metric("system_memory_usage", memory.percent, "percent", "system")
                self.record_metric("system_memory_available", memory.available / (1024**3), "GB", "system")
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.record_metric("system_disk_read_mb", disk_io.read_bytes / (1024**2), "MB", "system")
                    self.record_metric("system_disk_write_mb", disk_io.write_bytes / (1024**2), "MB", "system")
                
                # Network I/O
                net_io = psutil.net_io_counters()
                if net_io:
                    self.record_metric("system_net_sent_mb", net_io.bytes_sent / (1024**2), "MB", "system")
                    self.record_metric("system_net_recv_mb", net_io.bytes_recv / (1024**2), "MB", "system")
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                print(f"Warning: System monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def _save_session(self, session: PerformanceSession):
        """Save session to database"""
        with sqlite3.connect(self.db_path) as conn:
            # Save session
            conn.execute('''
                INSERT OR REPLACE INTO sessions 
                (session_id, test_name, start_time, end_time, duration_seconds, status, system_info)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session.session_id,
                session.test_name,
                session.start_time.isoformat(),
                session.end_time.isoformat() if session.end_time else None,
                session.get_duration(),
                session.status,
                json.dumps(session.system_info)
            ))
            
            # Save metrics
            for metric in session.metrics:
                conn.execute('''
                    INSERT INTO metrics 
                    (session_id, name, value, unit, category, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session.session_id,
                    metric.name,
                    metric.value,
                    metric.unit,
                    metric.category,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.metadata)
                ))
    
    def get_session(self, session_id: str) -> Optional[PerformanceSession]:
        """Retrieve a session from database"""
        with sqlite3.connect(self.db_path) as conn:
            # Get session data
            cursor = conn.execute('''
                SELECT session_id, test_name, start_time, end_time, status, system_info
                FROM sessions WHERE session_id = ?
            ''', (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            session = PerformanceSession(row[0], row[1])
            session.start_time = datetime.fromisoformat(row[2])
            if row[3]:
                session.end_time = datetime.fromisoformat(row[3])
            session.status = row[4]
            session.system_info = json.loads(row[5]) if row[5] else {}
            
            # Get metrics
            cursor = conn.execute('''
                SELECT name, value, unit, category, timestamp, metadata
                FROM metrics WHERE session_id = ? ORDER BY timestamp
            ''', (session_id,))
            
            for metric_row in cursor.fetchall():
                metric = PerformanceMetric(
                    name=metric_row[0],
                    value=metric_row[1],
                    unit=metric_row[2],
                    category=metric_row[3],
                    timestamp=datetime.fromisoformat(metric_row[4])
                )
                metric.metadata = json.loads(metric_row[5]) if metric_row[5] else {}
                session.add_metric(metric)
            
            return session
    
    def get_recent_sessions(self, limit: int = 10) -> List[PerformanceSession]:
        """Get recent sessions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT session_id FROM sessions 
                ORDER BY start_time DESC LIMIT ?
            ''', (limit,))
            
            sessions = []
            for row in cursor.fetchall():
                session = self.get_session(row[0])
                if session:
                    sessions.append(session)
            
            return sessions
    
    def get_metric_history(self, metric_name: str, days: int = 30) -> List[Tuple[datetime, float]]:
        """Get historical values for a metric"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT timestamp, value FROM metrics 
                WHERE name = ? AND timestamp >= ?
                ORDER BY timestamp
            ''', (metric_name, cutoff_date.isoformat()))
            
            return [(datetime.fromisoformat(row[0]), row[1]) for row in cursor.fetchall()]
    
    def analyze_metric_trends(self, metric_name: str, days: int = 30) -> Dict[str, Any]:
        """Analyze trends for a metric"""
        history = self.get_metric_history(metric_name, days)
        
        if len(history) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        values = [value for _, value in history]
        timestamps = [ts for ts, _ in history]
        
        analysis = {
            'metric_name': metric_name,
            'data_points': len(values),
            'time_range_days': (timestamps[-1] - timestamps[0]).days,
            'statistics': {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'latest': values[-1],
                'first': values[0]
            }
        }
        
        # Calculate trend
        if len(values) >= 3:
            # Simple linear trend
            x_values = list(range(len(values)))
            n = len(values)
            sum_x = sum(x_values)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(x_values, values))
            sum_x2 = sum(x * x for x in x_values)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                # Calculate percentage change
                if analysis['statistics']['first'] != 0:
                    percent_change = ((analysis['statistics']['latest'] - analysis['statistics']['first']) / 
                                    analysis['statistics']['first']) * 100
                else:
                    percent_change = 0
                
                analysis['trend'] = {
                    'slope': slope,
                    'direction': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable',
                    'percent_change': percent_change
                }
        
        return analysis
    
    def generate_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'report_timestamp': datetime.utcnow().isoformat(),
            'time_range_days': days,
            'sessions': [],
            'metric_trends': {},
            'summary': {}
        }
        
        # Get recent sessions
        recent_sessions = self.get_recent_sessions(limit=50)
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Filter sessions within time range
        filtered_sessions = [
            s for s in recent_sessions 
            if s.start_time >= cutoff_date
        ]
        
        report['sessions'] = [s.to_dict() for s in filtered_sessions]
        
        # Analyze key metrics
        key_metrics = [
            'verification_time_ms',
            'memory_usage_mb', 
            'accuracy',
            'throughput_ops_per_sec',
            'system_cpu_usage',
            'system_memory_usage'
        ]
        
        for metric in key_metrics:
            trend_analysis = self.analyze_metric_trends(metric, days)
            if 'error' not in trend_analysis:
                report['metric_trends'][metric] = trend_analysis
        
        # Generate summary
        report['summary'] = self._generate_report_summary(filtered_sessions, report['metric_trends'])
        
        return report
    
    def _generate_report_summary(self, sessions: List[PerformanceSession], 
                                trends: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report summary"""
        summary = {
            'total_sessions': len(sessions),
            'avg_session_duration': 0,
            'performance_insights': [],
            'recommendations': []
        }
        
        if sessions:
            durations = [s.get_duration() for s in sessions]
            summary['avg_session_duration'] = statistics.mean(durations)
        
        # Analyze trends for insights
        for metric_name, trend_data in trends.items():
            if 'trend' in trend_data:
                trend = trend_data['trend']
                direction = trend['direction']
                percent_change = abs(trend['percent_change'])
                
                if direction == 'increasing' and percent_change > 10:
                    if 'time' in metric_name.lower() or 'memory' in metric_name.lower():
                        summary['performance_insights'].append(
                            f"⚠️ {metric_name} has increased by {percent_change:.1f}%"
                        )
                        summary['recommendations'].append(
                            f"Investigate cause of {metric_name} increase"
                        )
                    elif 'accuracy' in metric_name.lower() or 'throughput' in metric_name.lower():
                        summary['performance_insights'].append(
                            f"✅ {metric_name} has improved by {percent_change:.1f}%"
                        )
                elif direction == 'decreasing' and percent_change > 10:
                    if 'accuracy' in metric_name.lower() or 'throughput' in metric_name.lower():
                        summary['performance_insights'].append(
                            f"⚠️ {metric_name} has decreased by {percent_change:.1f}%"
                        )
                        summary['recommendations'].append(
                            f"Address decline in {metric_name}"
                        )
        
        return summary
    
    def export_data(self, output_file: str, format: str = 'json', days: int = 30):
        """Export performance data"""
        if format == 'json':
            report = self.generate_performance_report(days)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        elif format == 'csv':
            self._export_csv(output_file, days)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_csv(self, output_file: str, days: int):
        """Export data as CSV"""
        import csv
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT s.session_id, s.test_name, s.start_time, s.duration_seconds,
                       m.name, m.value, m.unit, m.category, m.timestamp
                FROM sessions s
                JOIN metrics m ON s.session_id = m.session_id
                WHERE s.start_time >= ?
                ORDER BY s.start_time, m.timestamp
            ''', (cutoff_date.isoformat(),))
            
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'session_id', 'test_name', 'session_start', 'session_duration',
                    'metric_name', 'metric_value', 'metric_unit', 'metric_category', 'metric_timestamp'
                ])
                
                for row in cursor.fetchall():
                    writer.writerow(row)


# Context manager for easy performance tracking
class PerformanceContext:
    """Context manager for performance tracking"""
    
    def __init__(self, tracker: PerformanceTracker, test_name: str = "", 
                 auto_monitor: bool = True):
        self.tracker = tracker
        self.test_name = test_name
        self.auto_monitor = auto_monitor
        self.session_id = None
    
    def __enter__(self):
        self.session_id = self.tracker.start_session(self.test_name)
        if self.auto_monitor:
            self.tracker.start_monitoring()
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.auto_monitor:
            self.tracker.stop_monitoring()
        self.tracker.end_session()


# Decorators for automatic performance tracking
def track_performance(test_name: str = "", tracker: Optional[PerformanceTracker] = None):
    """Decorator to automatically track function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal tracker
            if tracker is None:
                tracker = PerformanceTracker()
            
            name = test_name or func.__name__
            
            with PerformanceContext(tracker, name, auto_monitor=False):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    tracker.record_timing(func.__name__, duration_ms, status='success')
                    return result
                    
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    tracker.record_timing(func.__name__, duration_ms, status='error', error=str(e))
                    raise
        
        return wrapper
    return decorator


if __name__ == '__main__':
    # Example usage
    tracker = PerformanceTracker()
    
    # Manual tracking
    session_id = tracker.start_session("example_test")
    tracker.record_timing("operation_1", 150.5)
    tracker.record_memory("operation_1", 256.0)
    tracker.record_accuracy("test_accuracy", 0.95)
    tracker.end_session()
    
    # Context manager usage
    with PerformanceContext(tracker, "context_test") as t:
        t.record_timing("context_operation", 200.0)
        time.sleep(0.1)  # Simulate work
    
    # Generate report
    report = tracker.generate_performance_report(days=1)
    print(f"Generated report with {len(report['sessions'])} sessions")
    
    # Export data
    tracker.export_data("performance_report.json", format='json', days=1)
    print("Performance data exported to performance_report.json")