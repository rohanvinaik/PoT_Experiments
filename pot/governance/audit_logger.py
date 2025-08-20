"""
Comprehensive Audit Logger for PoT Governance Framework
Implements tamper-evident logging, analysis, and retention management
"""

import json
import hashlib
import hmac
import logging
import gzip
import shutil
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import threading
import time
import base64
import os
import sqlite3
from collections import defaultdict, Counter
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend


class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"


class LogCategory(Enum):
    """Log categories for classification"""
    GOVERNANCE = "governance"
    MODEL_OPERATION = "model_operation"
    DATA_ACCESS = "data_access"
    SECURITY_EVENT = "security_event"
    POLICY_ENFORCEMENT = "policy_enforcement"
    SYSTEM_OPERATION = "system_operation"
    COMPLIANCE = "compliance"
    AUDIT = "audit"


class StorageBackend(Enum):
    """Storage backend types"""
    FILE = "file"
    DATABASE = "database"
    IMMUTABLE = "immutable"
    ARCHIVE = "archive"


@dataclass
class LogEntry:
    """Individual audit log entry"""
    entry_id: str
    timestamp: datetime
    category: LogCategory
    level: LogLevel
    actor: str  # User or system component
    action: str
    resource: str  # What was acted upon
    result: str  # Success, failure, etc.
    metadata: Dict[str, Any]
    hash: Optional[str] = None
    previous_hash: Optional[str] = None
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['category'] = self.category.value
        result['level'] = self.level.value
        return result
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
    
    def compute_hash(self, previous_hash: str = "") -> str:
        """Compute cryptographic hash of entry"""
        content = f"{self.entry_id}{self.timestamp.isoformat()}{self.category.value}"
        content += f"{self.level.value}{self.actor}{self.action}{self.resource}"
        content += f"{self.result}{json.dumps(self.metadata, sort_keys=True)}"
        content += previous_hash
        
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class AuditReport:
    """Audit report for a time period"""
    report_id: str
    start_date: datetime
    end_date: datetime
    total_entries: int
    entries_by_category: Dict[str, int]
    entries_by_level: Dict[str, int]
    top_actors: List[Tuple[str, int]]
    top_actions: List[Tuple[str, int]]
    anomalies_detected: List[Dict[str, Any]]
    policy_violations: List[Dict[str, Any]]
    compliance_gaps: List[Dict[str, Any]]
    integrity_verified: bool
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['start_date'] = self.start_date.isoformat()
        result['end_date'] = self.end_date.isoformat()
        return result


class AnomalyDetector:
    """Detect anomalies in audit logs"""
    
    def __init__(self):
        self.baseline = {}
        self.thresholds = {
            "access_rate": 100,  # Max accesses per hour
            "failure_rate": 0.1,  # Max failure ratio
            "unusual_time": True,  # Flag access outside business hours
            "repeated_failures": 5  # Max consecutive failures
        }
    
    def update_baseline(self, entries: List[LogEntry]):
        """Update baseline behavior from historical data"""
        # Access patterns by actor
        actor_patterns = defaultdict(lambda: {"count": 0, "hours": set(), "actions": set()})
        
        for entry in entries:
            actor = entry.actor
            hour = entry.timestamp.hour
            
            actor_patterns[actor]["count"] += 1
            actor_patterns[actor]["hours"].add(hour)
            actor_patterns[actor]["actions"].add(entry.action)
        
        # Convert sets to lists for serialization
        for actor in actor_patterns:
            actor_patterns[actor]["hours"] = list(actor_patterns[actor]["hours"])
            actor_patterns[actor]["actions"] = list(actor_patterns[actor]["actions"])
        
        self.baseline = dict(actor_patterns)
    
    def detect_anomalies(self, entries: List[LogEntry]) -> List[Dict[str, Any]]:
        """Detect anomalous patterns in log entries"""
        anomalies = []
        
        # Group entries by actor and time window
        actor_windows = defaultdict(list)
        for entry in entries:
            window = entry.timestamp.replace(minute=0, second=0, microsecond=0)
            actor_windows[(entry.actor, window)].append(entry)
        
        # Check for anomalies
        for (actor, window), window_entries in actor_windows.items():
            # High access rate
            if len(window_entries) > self.thresholds["access_rate"]:
                anomalies.append({
                    "type": "high_access_rate",
                    "actor": actor,
                    "timestamp": window.isoformat(),
                    "count": len(window_entries),
                    "threshold": self.thresholds["access_rate"]
                })
            
            # High failure rate
            failures = sum(1 for e in window_entries if "fail" in e.result.lower())
            if failures > 0:
                failure_rate = failures / len(window_entries)
                if failure_rate > self.thresholds["failure_rate"]:
                    anomalies.append({
                        "type": "high_failure_rate",
                        "actor": actor,
                        "timestamp": window.isoformat(),
                        "rate": failure_rate,
                        "threshold": self.thresholds["failure_rate"]
                    })
            
            # Unusual access time
            if self.thresholds["unusual_time"]:
                for entry in window_entries:
                    hour = entry.timestamp.hour
                    if hour < 6 or hour > 22:  # Outside business hours
                        if actor in self.baseline:
                            if hour not in self.baseline[actor].get("hours", []):
                                anomalies.append({
                                    "type": "unusual_time",
                                    "actor": actor,
                                    "timestamp": entry.timestamp.isoformat(),
                                    "hour": hour
                                })
        
        # Consecutive failures
        actor_failures = defaultdict(int)
        for entry in sorted(entries, key=lambda e: e.timestamp):
            if "fail" in entry.result.lower():
                actor_failures[entry.actor] += 1
                if actor_failures[entry.actor] >= self.thresholds["repeated_failures"]:
                    anomalies.append({
                        "type": "repeated_failures",
                        "actor": entry.actor,
                        "timestamp": entry.timestamp.isoformat(),
                        "count": actor_failures[entry.actor],
                        "threshold": self.thresholds["repeated_failures"]
                    })
            else:
                actor_failures[entry.actor] = 0
        
        return anomalies


class RetentionManager:
    """Manage log retention and archival"""
    
    def __init__(self, retention_days: int = 90, archive_days: int = 365):
        self.retention_days = retention_days
        self.archive_days = archive_days
        self.archive_dir = None
    
    def set_archive_dir(self, archive_dir: Path):
        """Set archive directory"""
        self.archive_dir = archive_dir
        self.archive_dir.mkdir(parents=True, exist_ok=True)
    
    def should_rotate(self, log_file: Path) -> bool:
        """Check if log file should be rotated"""
        # Rotate if file is larger than 100MB
        if log_file.exists():
            return log_file.stat().st_size > 100 * 1024 * 1024
        return False
    
    def should_archive(self, entry_date: datetime) -> bool:
        """Check if entry should be archived"""
        age = (datetime.now() - entry_date).days
        return age > self.retention_days and age <= self.archive_days
    
    def should_delete(self, entry_date: datetime) -> bool:
        """Check if entry should be deleted"""
        age = (datetime.now() - entry_date).days
        return age > self.archive_days
    
    def archive_logs(self, logs: List[LogEntry], archive_file: Path):
        """Archive logs to compressed file"""
        # Group by date
        logs_by_date = defaultdict(list)
        for log in logs:
            date_key = log.timestamp.date()
            logs_by_date[date_key].append(log)
        
        # Archive each date
        for date, date_logs in logs_by_date.items():
            archive_path = self.archive_dir / f"audit_{date.isoformat()}.jsonl.gz"
            
            with gzip.open(archive_path, 'at') as f:
                for log in date_logs:
                    f.write(log.to_json() + '\n')
    
    def cleanup_old_archives(self):
        """Remove archives older than retention period"""
        if not self.archive_dir:
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.archive_days)
        
        for archive_file in self.archive_dir.glob("audit_*.jsonl.gz"):
            # Extract date from filename
            try:
                date_str = archive_file.stem.replace("audit_", "").replace(".jsonl", "")
                file_date = datetime.fromisoformat(date_str)
                
                if file_date < cutoff_date:
                    archive_file.unlink()
                    logging.info(f"Deleted old archive: {archive_file}")
            except Exception as e:
                logging.error(f"Error processing archive {archive_file}: {e}")


class AuditLogger:
    """
    Comprehensive audit logger with tamper-evident features
    """
    
    def __init__(
        self,
        log_dir: str,
        encryption_key: Optional[str] = None,
        signing_key: Optional[str] = None,
        backend: StorageBackend = StorageBackend.FILE
    ):
        """
        Initialize audit logger
        
        Args:
            log_dir: Directory for log storage
            encryption_key: Optional encryption key for sensitive data
            signing_key: Optional signing key for digital signatures
            backend: Storage backend type
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Encryption setup
        self.encryption = encryption_key is not None
        if self.encryption:
            self.cipher = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
        else:
            self.cipher = None
        
        # Signing setup
        self.signing = signing_key is not None
        if self.signing:
            self._setup_signing(signing_key)
        else:
            self.signing_key = None
        
        # Storage backend
        self.backend = backend
        self._setup_backend()
        
        # Components
        self.anomaly_detector = AnomalyDetector()
        self.retention_manager = RetentionManager()
        self.retention_manager.set_archive_dir(self.log_dir / "archive")
        
        # State
        self.last_hash = self._get_last_hash()
        self.entry_count = 0
        self.lock = threading.Lock()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Start background tasks
        self._start_background_tasks()
    
    def _setup_signing(self, signing_key: str):
        """Setup digital signing"""
        try:
            # Try to load as PEM private key
            self.signing_key = serialization.load_pem_private_key(
                signing_key.encode(),
                password=None,
                backend=default_backend()
            )
        except:
            # Generate new key if not provided
            self.signing_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
    
    def _setup_backend(self):
        """Setup storage backend"""
        if self.backend == StorageBackend.DATABASE:
            self.db_path = self.log_dir / "audit.db"
            self._init_database()
        elif self.backend == StorageBackend.IMMUTABLE:
            # Could integrate with blockchain or other immutable storage
            self.immutable_store = []
        
        # Default to file backend
        self.current_log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    def _init_database(self):
        """Initialize SQLite database for log storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                entry_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                level TEXT NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                resource TEXT NOT NULL,
                result TEXT NOT NULL,
                metadata TEXT,
                hash TEXT,
                previous_hash TEXT,
                signature TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_logs(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_actor ON audit_logs(actor)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON audit_logs(category)
        """)
        
        conn.commit()
        conn.close()
    
    def _get_last_hash(self) -> str:
        """Get the hash of the last log entry"""
        if self.backend == StorageBackend.DATABASE:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT hash FROM audit_logs ORDER BY created_at DESC LIMIT 1")
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else ""
        else:
            # Read last line from current log file
            if self.current_log_file.exists():
                with open(self.current_log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        try:
                            last_entry = json.loads(lines[-1])
                            return last_entry.get('hash', '')
                        except:
                            pass
        return ""
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Rotation thread
        def rotation_worker():
            while True:
                time.sleep(3600)  # Check every hour
                try:
                    self._rotate_logs()
                    self._cleanup_old_logs()
                except Exception as e:
                    self.logger.error(f"Rotation error: {e}")
        
        rotation_thread = threading.Thread(target=rotation_worker, daemon=True)
        rotation_thread.start()
    
    def _rotate_logs(self):
        """Rotate log files if needed"""
        if self.backend == StorageBackend.FILE:
            if self.retention_manager.should_rotate(self.current_log_file):
                # Create new log file
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                rotated_file = self.log_dir / f"audit_{timestamp}_rotated.jsonl"
                
                with self.lock:
                    shutil.move(self.current_log_file, rotated_file)
                    self.current_log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
                
                # Compress rotated file
                with open(rotated_file, 'rb') as f_in:
                    with gzip.open(f"{rotated_file}.gz", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                rotated_file.unlink()
                self.logger.info(f"Rotated log file: {rotated_file}")
    
    def _cleanup_old_logs(self):
        """Clean up old log files based on retention policy"""
        # Archive old logs
        cutoff_archive = datetime.now() - timedelta(days=self.retention_manager.retention_days)
        cutoff_delete = datetime.now() - timedelta(days=self.retention_manager.archive_days)
        
        for log_file in self.log_dir.glob("audit_*.jsonl"):
            try:
                # Extract date from filename
                date_str = log_file.stem.replace("audit_", "").split("_")[0]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                
                if file_date < cutoff_delete:
                    # Delete old files
                    log_file.unlink()
                    self.logger.info(f"Deleted old log: {log_file}")
                elif file_date < cutoff_archive:
                    # Archive to cold storage
                    self._archive_log_file(log_file)
            except Exception as e:
                self.logger.error(f"Error processing log file {log_file}: {e}")
        
        # Clean old archives
        self.retention_manager.cleanup_old_archives()
    
    def _archive_log_file(self, log_file: Path):
        """Archive a log file to cold storage"""
        archive_path = self.log_dir / "archive" / f"{log_file.stem}.gz"
        archive_path.parent.mkdir(exist_ok=True)
        
        with open(log_file, 'rb') as f_in:
            with gzip.open(archive_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        log_file.unlink()
        self.logger.info(f"Archived log: {log_file} -> {archive_path}")
    
    def _create_entry(
        self,
        category: LogCategory,
        level: LogLevel,
        actor: str,
        action: str,
        resource: str,
        result: str,
        metadata: Dict[str, Any]
    ) -> LogEntry:
        """Create a log entry with tamper-evident features"""
        # Generate entry ID
        entry_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{self.entry_count:06d}"
        self.entry_count += 1
        
        # Get trusted timestamp (could use NTP or timestamp authority)
        timestamp = datetime.now()
        
        # Create entry
        entry = LogEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            category=category,
            level=level,
            actor=actor,
            action=action,
            resource=resource,
            result=result,
            metadata=metadata
        )
        
        # Add hash chain
        entry.previous_hash = self.last_hash
        entry.hash = entry.compute_hash(self.last_hash)
        self.last_hash = entry.hash
        
        # Add digital signature if available
        if self.signing_key:
            signature_data = entry.to_json().encode()
            signature = self.signing_key.sign(
                signature_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            entry.signature = base64.b64encode(signature).decode()
        
        return entry
    
    def _write_entry(self, entry: LogEntry):
        """Write entry to storage backend"""
        if self.backend == StorageBackend.DATABASE:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO audit_logs (
                    entry_id, timestamp, category, level, actor, action,
                    resource, result, metadata, hash, previous_hash, signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.entry_id, entry.timestamp.isoformat(), entry.category.value,
                entry.level.value, entry.actor, entry.action, entry.resource,
                entry.result, json.dumps(entry.metadata), entry.hash,
                entry.previous_hash, entry.signature
            ))
            
            conn.commit()
            conn.close()
        
        elif self.backend == StorageBackend.IMMUTABLE:
            # Add to immutable store (could be blockchain, etc.)
            self.immutable_store.append(entry)
        
        else:  # FILE backend
            with open(self.current_log_file, 'a') as f:
                # Encrypt sensitive fields if encryption is enabled
                entry_dict = entry.to_dict()
                if self.encryption and entry.level == LogLevel.SECURITY:
                    # Encrypt sensitive metadata
                    if 'metadata' in entry_dict:
                        encrypted_metadata = self.cipher.encrypt(
                            json.dumps(entry_dict['metadata']).encode()
                        )
                        entry_dict['metadata'] = base64.b64encode(encrypted_metadata).decode()
                        entry_dict['encrypted'] = True
                
                f.write(json.dumps(entry_dict) + '\n')
    
    def log_governance_decision(self, decision: Dict[str, Any]):
        """
        Log a governance decision with tamper-evident features
        
        Args:
            decision: Governance decision details
        """
        with self.lock:
            entry = self._create_entry(
                category=LogCategory.GOVERNANCE,
                level=LogLevel.INFO,
                actor=decision.get('decision_maker', 'system'),
                action='governance_decision',
                resource=decision.get('resource', 'policy'),
                result=decision.get('approved', 'unknown'),
                metadata=decision
            )
            
            self._write_entry(entry)
            
            # Check for anomalies
            if decision.get('override', False):
                self.logger.warning(f"Governance override by {entry.actor}")
    
    def log_model_action(self, action: str, model_id: str, metadata: Dict[str, Any]):
        """
        Track all model operations
        
        Args:
            action: Action performed (train, deploy, retrain, etc.)
            model_id: Model identifier
            metadata: Additional metadata
        """
        with self.lock:
            # Determine level based on action
            if action in ['deploy', 'production_update']:
                level = LogLevel.WARNING
            elif action in ['delete', 'rollback']:
                level = LogLevel.CRITICAL
            else:
                level = LogLevel.INFO
            
            entry = self._create_entry(
                category=LogCategory.MODEL_OPERATION,
                level=level,
                actor=metadata.get('user', 'system'),
                action=action,
                resource=f"model:{model_id}",
                result=metadata.get('result', 'success'),
                metadata=metadata
            )
            
            self._write_entry(entry)
    
    def log_data_access(self, accessor: str, data_id: str, purpose: str, metadata: Optional[Dict] = None):
        """
        Record data access for compliance
        
        Args:
            accessor: Who accessed the data
            data_id: Data identifier
            purpose: Purpose of access
            metadata: Additional metadata
        """
        with self.lock:
            access_metadata = metadata or {}
            access_metadata['purpose'] = purpose
            
            # Check if accessing sensitive data
            level = LogLevel.SECURITY if access_metadata.get('sensitive', False) else LogLevel.INFO
            
            entry = self._create_entry(
                category=LogCategory.DATA_ACCESS,
                level=level,
                actor=accessor,
                action='data_access',
                resource=f"data:{data_id}",
                result='granted',
                metadata=access_metadata
            )
            
            self._write_entry(entry)
            
            # Track access patterns for anomaly detection
            self._track_access_pattern(accessor, data_id)
    
    def log_security_event(self, event_type: str, actor: str, details: Dict[str, Any]):
        """
        Log security-related events
        
        Args:
            event_type: Type of security event
            actor: Actor involved
            details: Event details
        """
        with self.lock:
            entry = self._create_entry(
                category=LogCategory.SECURITY_EVENT,
                level=LogLevel.SECURITY,
                actor=actor,
                action=event_type,
                resource=details.get('resource', 'system'),
                result=details.get('result', 'detected'),
                metadata=details
            )
            
            self._write_entry(entry)
            
            # Immediate alert for critical security events
            if event_type in ['breach_attempt', 'unauthorized_access', 'data_exfiltration']:
                self._send_security_alert(entry)
    
    def _track_access_pattern(self, accessor: str, data_id: str):
        """Track access patterns for anomaly detection"""
        # Build access pattern for anomaly detection
        pattern_key = f"{accessor}:{data_id}"
        current_time = datetime.now()
        
        # Store in anomaly detector's internal state
        if not hasattr(self.anomaly_detector, 'access_patterns'):
            self.anomaly_detector.access_patterns = defaultdict(list)
        
        # Track access time and frequency
        self.anomaly_detector.access_patterns[pattern_key].append(current_time)
        
        # Keep only recent accesses (last 30 days)
        cutoff_time = current_time - timedelta(days=30)
        self.anomaly_detector.access_patterns[pattern_key] = [
            t for t in self.anomaly_detector.access_patterns[pattern_key]
            if t > cutoff_time
        ]
        
        # Check for anomalies
        access_times = self.anomaly_detector.access_patterns[pattern_key]
        if len(access_times) > 1:
            # Calculate access frequency
            time_diffs = [(access_times[i] - access_times[i-1]).total_seconds() 
                          for i in range(1, len(access_times))]
            
            if time_diffs:
                avg_interval = np.mean(time_diffs) if time_diffs else float('inf')
                std_interval = np.std(time_diffs) if len(time_diffs) > 1 else 0
                
                # Detect rapid successive access (potential data scraping)
                if len(access_times) > 10 and avg_interval < 1.0:  # More than 10 accesses with <1s interval
                    self.log_security_event(
                        event_type='rapid_access_detected',
                        actor=accessor,
                        details={
                            'data_id': data_id,
                            'access_count': len(access_times),
                            'avg_interval': avg_interval,
                            'pattern': 'potential_data_scraping'
                        }
                    )
                
                # Detect unusual access patterns (e.g., access at unusual times)
                hour_distribution = Counter(t.hour for t in access_times)
                if len(hour_distribution) > 5:  # Enough data to analyze
                    common_hours = set(h for h, count in hour_distribution.most_common(3))
                    current_hour = current_time.hour
                    
                    # Flag if accessing at unusual hour for this user
                    if current_hour not in common_hours and hour_distribution[current_hour] < 2:
                        self.log_security_event(
                            event_type='unusual_access_time',
                            actor=accessor,
                            details={
                                'data_id': data_id,
                                'access_hour': current_hour,
                                'common_hours': list(common_hours),
                                'pattern': 'temporal_anomaly'
                            }
                        )
    
    def _send_security_alert(self, entry: LogEntry):
        """Send immediate security alert"""
        alert = {
            "type": "SECURITY_ALERT",
            "timestamp": entry.timestamp.isoformat(),
            "actor": entry.actor,
            "action": entry.action,
            "severity": "CRITICAL"
        }
        # In production, this would send to SIEM or alert system
        self.logger.critical(f"SECURITY ALERT: {json.dumps(alert)}")
    
    def query_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        category: Optional[LogCategory] = None,
        actor: Optional[str] = None,
        limit: int = 1000
    ) -> List[LogEntry]:
        """
        Query audit logs
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            category: Filter by category
            actor: Filter by actor
            limit: Maximum results
            
        Returns:
            List of matching log entries
        """
        entries = []
        
        if self.backend == StorageBackend.DATABASE:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM audit_logs WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            if category:
                query += " AND category = ?"
                params.append(category.value)
            
            if actor:
                query += " AND actor = ?"
                params.append(actor)
            
            query += f" ORDER BY timestamp DESC LIMIT {limit}"
            
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                entry = LogEntry(
                    entry_id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    category=LogCategory(row[2]),
                    level=LogLevel(row[3]),
                    actor=row[4],
                    action=row[5],
                    resource=row[6],
                    result=row[7],
                    metadata=json.loads(row[8]) if row[8] else {},
                    hash=row[9],
                    previous_hash=row[10],
                    signature=row[11]
                )
                entries.append(entry)
            
            conn.close()
        
        else:  # FILE backend
            # Read from files
            for log_file in sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True):
                if len(entries) >= limit:
                    break
                
                with open(log_file, 'r') as f:
                    for line in f:
                        if len(entries) >= limit:
                            break
                        
                        try:
                            data = json.loads(line)
                            entry_time = datetime.fromisoformat(data['timestamp'])
                            
                            # Apply filters
                            if start_date and entry_time < start_date:
                                continue
                            if end_date and entry_time > end_date:
                                continue
                            if category and data['category'] != category.value:
                                continue
                            if actor and data['actor'] != actor:
                                continue
                            
                            # Decrypt if needed
                            if self.encryption and data.get('encrypted'):
                                encrypted_metadata = base64.b64decode(data['metadata'])
                                decrypted = self.cipher.decrypt(encrypted_metadata)
                                data['metadata'] = json.loads(decrypted)
                            
                            entry = LogEntry(
                                entry_id=data['entry_id'],
                                timestamp=entry_time,
                                category=LogCategory(data['category']),
                                level=LogLevel(data['level']),
                                actor=data['actor'],
                                action=data['action'],
                                resource=data['resource'],
                                result=data['result'],
                                metadata=data.get('metadata', {}),
                                hash=data.get('hash'),
                                previous_hash=data.get('previous_hash'),
                                signature=data.get('signature')
                            )
                            entries.append(entry)
                        
                        except Exception as e:
                            self.logger.error(f"Error parsing log entry: {e}")
        
        return entries
    
    def verify_integrity(self, start_date: Optional[datetime] = None) -> Tuple[bool, List[str]]:
        """
        Verify integrity of audit logs
        
        Args:
            start_date: Start verification from this date
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        entries = self.query_logs(start_date=start_date, limit=10000)
        
        if not entries:
            return True, []
        
        # Sort by timestamp
        entries.sort(key=lambda e: e.timestamp)
        
        # Verify hash chain
        previous_hash = ""
        for i, entry in enumerate(entries):
            # Verify hash chain
            if i > 0 and entry.previous_hash != previous_hash:
                issues.append(f"Hash chain broken at entry {entry.entry_id}")
            
            # Recompute hash
            computed_hash = entry.compute_hash(entry.previous_hash or "")
            if computed_hash != entry.hash:
                issues.append(f"Hash mismatch for entry {entry.entry_id}")
            
            # Verify signature if present
            if entry.signature and self.signing_key:
                try:
                    public_key = self.signing_key.public_key()
                    signature = base64.b64decode(entry.signature)
                    
                    # Remove signature from entry for verification
                    entry_copy = LogEntry(**{k: v for k, v in asdict(entry).items() if k != 'signature'})
                    
                    public_key.verify(
                        signature,
                        entry_copy.to_json().encode(),
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH
                        ),
                        hashes.SHA256()
                    )
                except Exception as e:
                    issues.append(f"Signature verification failed for {entry.entry_id}: {e}")
            
            previous_hash = entry.hash
        
        return len(issues) == 0, issues
    
    def generate_audit_report(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Produce audit report for period
        
        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            
        Returns:
            Comprehensive audit report
        """
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        # Query logs for period
        entries = self.query_logs(start_date=start, end_date=end, limit=100000)
        
        # Analyze entries
        entries_by_category = Counter(e.category.value for e in entries)
        entries_by_level = Counter(e.level.value for e in entries)
        actor_counts = Counter(e.actor for e in entries)
        action_counts = Counter(e.action for e in entries)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(entries)
        
        # Identify policy violations
        violations = self._identify_violations(entries)
        
        # Identify compliance gaps
        gaps = self._identify_compliance_gaps(entries)
        
        # Verify integrity
        integrity_valid, integrity_issues = self.verify_integrity(start)
        
        report = AuditReport(
            report_id=f"AUDIT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            start_date=start,
            end_date=end,
            total_entries=len(entries),
            entries_by_category=dict(entries_by_category),
            entries_by_level=dict(entries_by_level),
            top_actors=actor_counts.most_common(10),
            top_actions=action_counts.most_common(10),
            anomalies_detected=anomalies,
            policy_violations=violations,
            compliance_gaps=gaps,
            integrity_verified=integrity_valid
        )
        
        return report.to_dict()
    
    def _identify_violations(self, entries: List[LogEntry]) -> List[Dict[str, Any]]:
        """Identify policy violations in log entries"""
        violations = []
        
        for entry in entries:
            # Check for failed policy enforcements
            if entry.category == LogCategory.POLICY_ENFORCEMENT and "fail" in entry.result.lower():
                violations.append({
                    "timestamp": entry.timestamp.isoformat(),
                    "actor": entry.actor,
                    "policy": entry.resource,
                    "violation": entry.metadata.get("violation", "Unknown")
                })
            
            # Check for unauthorized access
            if entry.category == LogCategory.DATA_ACCESS and entry.result == "denied":
                violations.append({
                    "timestamp": entry.timestamp.isoformat(),
                    "actor": entry.actor,
                    "resource": entry.resource,
                    "violation": "Unauthorized access attempt"
                })
        
        return violations
    
    def _identify_compliance_gaps(self, entries: List[LogEntry]) -> List[Dict[str, Any]]:
        """Identify compliance gaps from audit logs"""
        gaps = []
        
        # Check for missing required logs
        required_actions = ['data_deletion', 'consent_withdrawal', 'data_export']
        logged_actions = set(e.action for e in entries)
        
        for required in required_actions:
            if required not in logged_actions:
                gaps.append({
                    "type": "missing_audit",
                    "requirement": required,
                    "description": f"No audit logs found for required action: {required}"
                })
        
        # Check for encryption compliance
        sensitive_access = [e for e in entries if e.category == LogCategory.DATA_ACCESS 
                          and e.metadata.get('sensitive', False)]
        unencrypted = [e for e in sensitive_access if not e.metadata.get('encrypted', False)]
        
        if unencrypted:
            gaps.append({
                "type": "encryption_gap",
                "count": len(unencrypted),
                "description": f"{len(unencrypted)} sensitive data accesses without encryption"
            })
        
        return gaps
    
    def export_siem_format(
        self,
        entries: List[LogEntry],
        format: str = "cef"
    ) -> str:
        """
        Export logs in SIEM format
        
        Args:
            entries: Log entries to export
            format: Export format (cef, leef, json)
            
        Returns:
            Formatted log data
        """
        if format == "cef":
            # Common Event Format
            output = []
            for entry in entries:
                cef = (
                    f"CEF:0|PoT|AuditLogger|1.0|{entry.action}|"
                    f"{entry.action}|{self._severity_to_cef(entry.level)}|"
                    f"act={entry.action} duser={entry.actor} "
                    f"msg={entry.resource} outcome={entry.result}"
                )
                output.append(cef)
            return '\n'.join(output)
        
        elif format == "leef":
            # Log Event Extended Format
            output = []
            for entry in entries:
                leef = (
                    f"LEEF:2.0|PoT|AuditLogger|1.0|{entry.action}|"
                    f"cat={entry.category.value}|devTime={entry.timestamp.isoformat()}|"
                    f"usrName={entry.actor}|action={entry.action}|"
                    f"resource={entry.resource}|result={entry.result}"
                )
                output.append(leef)
            return '\n'.join(output)
        
        else:  # JSON format
            return json.dumps([e.to_dict() for e in entries], indent=2)
    
    def _severity_to_cef(self, level: LogLevel) -> int:
        """Convert log level to CEF severity"""
        mapping = {
            LogLevel.DEBUG: 1,
            LogLevel.INFO: 3,
            LogLevel.WARNING: 5,
            LogLevel.ERROR: 7,
            LogLevel.CRITICAL: 9,
            LogLevel.SECURITY: 10
        }
        return mapping.get(level, 3)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logger statistics"""
        total_entries = 0
        
        if self.backend == StorageBackend.DATABASE:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM audit_logs")
            total_entries = cursor.fetchone()[0]
            conn.close()
        
        return {
            "total_entries": total_entries,
            "encryption_enabled": self.encryption,
            "signing_enabled": self.signing is not None,
            "backend": self.backend.value,
            "retention_days": self.retention_manager.retention_days,
            "archive_days": self.retention_manager.archive_days,
            "last_hash": self.last_hash[:8] + "..." if self.last_hash else None
        }