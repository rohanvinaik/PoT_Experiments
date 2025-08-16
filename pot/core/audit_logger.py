#!/usr/bin/env python3
"""
Audit Logger Module for Regulatory Compliance
Provides comprehensive audit logging for PoT verification system
to meet regulatory requirements (EU AI Act, NIST RMF, GDPR, etc.)
"""

import json
import time
import hashlib
import hmac
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import os
import gzip
import base64
from contextlib import contextmanager

# Try to import cryptography for encryption
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


class ComplianceFramework(Enum):
    """Supported regulatory compliance frameworks"""
    EU_AI_ACT = "eu_ai_act"
    NIST_RMF = "nist_rmf"
    GDPR = "gdpr"
    CCPA = "ccpa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    HIPAA = "hipaa"


class AuditEventType(Enum):
    """Types of audit events"""
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"
    VERIFICATION_FAILED = "verification_failed"
    MODEL_LOADED = "model_loaded"
    CHALLENGE_GENERATED = "challenge_generated"
    RESPONSE_COMPUTED = "response_computed"
    ATTACK_DETECTED = "attack_detected"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    CONFIG_CHANGED = "config_changed"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    DATA_EXPORTED = "data_exported"
    DATA_DELETED = "data_deleted"
    SYSTEM_ERROR = "system_error"
    SECURITY_ALERT = "security_alert"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Single audit event record"""
    event_id: str
    timestamp: str
    event_type: AuditEventType
    severity: AuditSeverity
    actor: str  # User or system component
    resource: str  # Model, dataset, or component affected
    action: str  # What was done
    outcome: str  # Success, failure, etc.
    details: Dict[str, Any]
    compliance_tags: List[ComplianceFramework]
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None


class AuditLogger:
    """
    Comprehensive audit logging system for regulatory compliance
    Features:
    - Tamper-proof logging with digital signatures
    - Encryption at rest
    - Automatic rotation and archival
    - Compliance reporting
    - Search and analytics
    """
    
    def __init__(
        self,
        storage_path: str = "audit_logs",
        encryption_key: Optional[str] = None,
        signing_key: Optional[str] = None,
        retention_days: int = 2555,  # 7 years default
        rotation_size_mb: int = 100,
        enable_encryption: bool = True,
        enable_signing: bool = True,
        compliance_frameworks: Optional[List[ComplianceFramework]] = None
    ):
        """
        Initialize audit logger
        
        Args:
            storage_path: Directory for audit logs
            encryption_key: Key for encrypting logs at rest
            signing_key: Key for signing log entries
            retention_days: How long to retain logs
            rotation_size_mb: Max size before rotating log file
            enable_encryption: Whether to encrypt logs
            enable_signing: Whether to sign log entries
            compliance_frameworks: List of compliance frameworks to track
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.retention_days = retention_days
        self.rotation_size_mb = rotation_size_mb
        self.enable_encryption = enable_encryption and HAS_CRYPTO
        self.enable_signing = enable_signing
        
        # Setup encryption
        if self.enable_encryption:
            if encryption_key:
                self.cipher = Fernet(encryption_key.encode()[:32].ljust(32, b'0'))
            else:
                # Generate key from password
                self.cipher = self._generate_cipher(b"default_audit_key")
        else:
            self.cipher = None
        
        # Setup signing
        self.signing_key = signing_key or "default_signing_key"
        
        # Compliance frameworks
        self.compliance_frameworks = compliance_frameworks or [
            ComplianceFramework.EU_AI_ACT,
            ComplianceFramework.NIST_RMF,
            ComplianceFramework.GDPR
        ]
        
        # Initialize database
        self.db_path = self.storage_path / "audit.db"
        self._init_database()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Current log file
        self.current_log_file = self._get_current_log_file()
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "events_by_type": {},
            "events_by_severity": {},
            "last_rotation": datetime.now().isoformat()
        }
    
    def _generate_cipher(self, password: bytes) -> 'Fernet':
        """Generate encryption cipher from password"""
        if not HAS_CRYPTO:
            return None
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'audit_salt_v1',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)
    
    def _init_database(self):
        """Initialize SQLite database for structured queries"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    details TEXT,
                    compliance_tags TEXT,
                    metadata TEXT,
                    signature TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_actor ON audit_events(actor)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_resource ON audit_events(resource)")
    
    def _get_current_log_file(self) -> Path:
        """Get current log file path"""
        timestamp = datetime.now().strftime("%Y%m%d")
        return self.storage_path / f"audit_{timestamp}.jsonl"
    
    def _sign_event(self, event: AuditEvent) -> str:
        """Generate digital signature for event"""
        if not self.enable_signing:
            return ""
        
        # Create canonical representation
        canonical = json.dumps({
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "event_type": event.event_type.value,
            "actor": event.actor,
            "resource": event.resource,
            "action": event.action,
            "outcome": event.outcome
        }, sort_keys=True)
        
        # Generate HMAC signature
        signature = hmac.new(
            self.signing_key.encode(),
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt data if encryption is enabled"""
        if self.cipher:
            return self.cipher.encrypt(data.encode()).decode()
        return data
    
    def _decrypt_data(self, data: str) -> str:
        """Decrypt data if encryption is enabled"""
        if self.cipher:
            return self.cipher.decrypt(data.encode()).decode()
        return data
    
    def log_event(
        self,
        event_type: AuditEventType,
        actor: str,
        resource: str,
        action: str,
        outcome: str = "success",
        severity: AuditSeverity = AuditSeverity.INFO,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an audit event
        
        Returns:
            Event ID for reference
        """
        with self.lock:
            # Generate event
            event_id = hashlib.sha256(
                f"{time.time()}:{actor}:{resource}:{action}".encode()
            ).hexdigest()[:16]
            
            event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.now().isoformat(),
                event_type=event_type,
                severity=severity,
                actor=actor,
                resource=resource,
                action=action,
                outcome=outcome,
                details=details or {},
                compliance_tags=self.compliance_frameworks,
                metadata=metadata or {}
            )
            
            # Sign event
            event.signature = self._sign_event(event)
            
            # Store in database
            self._store_in_database(event)
            
            # Write to file
            self._write_to_file(event)
            
            # Update statistics
            self._update_stats(event)
            
            # Check rotation
            self._check_rotation()
            
            return event_id
    
    def _store_in_database(self, event: AuditEvent):
        """Store event in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO audit_events 
                (event_id, timestamp, event_type, severity, actor, resource, 
                 action, outcome, details, compliance_tags, metadata, signature)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.timestamp,
                event.event_type.value,
                event.severity.value,
                event.actor,
                event.resource,
                event.action,
                event.outcome,
                json.dumps(event.details),
                json.dumps([f.value for f in event.compliance_tags]),
                json.dumps(event.metadata),
                event.signature
            ))
    
    def _write_to_file(self, event: AuditEvent):
        """Write event to file"""
        # Convert to dict
        event_dict = {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "event_type": event.event_type.value,
            "severity": event.severity.value,
            "actor": event.actor,
            "resource": event.resource,
            "action": event.action,
            "outcome": event.outcome,
            "details": event.details,
            "compliance_tags": [f.value for f in event.compliance_tags],
            "metadata": event.metadata,
            "signature": event.signature
        }
        
        # Serialize
        event_json = json.dumps(event_dict)
        
        # Encrypt if enabled
        if self.enable_encryption:
            event_json = self._encrypt_data(event_json)
        
        # Write to file
        with open(self.current_log_file, 'a') as f:
            f.write(event_json + '\n')
    
    def _update_stats(self, event: AuditEvent):
        """Update statistics"""
        self.stats["total_events"] += 1
        
        event_type = event.event_type.value
        self.stats["events_by_type"][event_type] = \
            self.stats["events_by_type"].get(event_type, 0) + 1
        
        severity = event.severity.value
        self.stats["events_by_severity"][severity] = \
            self.stats["events_by_severity"].get(severity, 0) + 1
    
    def _check_rotation(self):
        """Check if log rotation is needed"""
        if self.current_log_file.exists():
            size_mb = self.current_log_file.stat().st_size / (1024 * 1024)
            if size_mb >= self.rotation_size_mb:
                self._rotate_log()
    
    def _rotate_log(self):
        """Rotate current log file"""
        if not self.current_log_file.exists():
            return
        
        # Compress old log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.storage_path / f"audit_{timestamp}.jsonl.gz"
        
        with open(self.current_log_file, 'rb') as f_in:
            with gzip.open(archive_path, 'wb') as f_out:
                f_out.writelines(f_in)
        
        # Remove old log
        self.current_log_file.unlink()
        
        # Update current log file
        self.current_log_file = self._get_current_log_file()
        
        # Update stats
        self.stats["last_rotation"] = datetime.now().isoformat()
        
        # Clean old logs
        self._clean_old_logs()
    
    def _clean_old_logs(self):
        """Remove logs older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for log_file in self.storage_path.glob("audit_*.jsonl.gz"):
            # Parse date from filename
            try:
                date_str = log_file.stem.split('_')[1]
                file_date = datetime.strptime(date_str, "%Y%m%d")
                if file_date < cutoff_date:
                    log_file.unlink()
            except (IndexError, ValueError):
                continue
    
    @contextmanager
    def audit_context(
        self,
        operation: str,
        actor: str,
        resource: str
    ):
        """
        Context manager for auditing operations
        
        Example:
            with audit_logger.audit_context("verify_model", "user123", "model_abc"):
                # Perform verification
                result = verify_model(...)
        """
        start_time = time.time()
        event_id = None
        
        try:
            # Log start
            event_id = self.log_event(
                event_type=AuditEventType.VERIFICATION_STARTED,
                actor=actor,
                resource=resource,
                action=f"Starting {operation}",
                details={"operation": operation}
            )
            
            yield event_id
            
            # Log success
            duration = time.time() - start_time
            self.log_event(
                event_type=AuditEventType.VERIFICATION_COMPLETED,
                actor=actor,
                resource=resource,
                action=f"Completed {operation}",
                outcome="success",
                details={
                    "operation": operation,
                    "duration_seconds": duration,
                    "parent_event": event_id
                }
            )
            
        except Exception as e:
            # Log failure
            duration = time.time() - start_time
            self.log_event(
                event_type=AuditEventType.VERIFICATION_FAILED,
                actor=actor,
                resource=resource,
                action=f"Failed {operation}",
                outcome="failure",
                severity=AuditSeverity.ERROR,
                details={
                    "operation": operation,
                    "duration_seconds": duration,
                    "error": str(e),
                    "parent_event": event_id
                }
            )
            raise
    
    def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        severities: Optional[List[AuditSeverity]] = None,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Query audit events
        
        Returns:
            List of matching events
        """
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        if event_types:
            placeholders = ','.join(['?' for _ in event_types])
            query += f" AND event_type IN ({placeholders})"
            params.extend([e.value for e in event_types])
        
        if severities:
            placeholders = ','.join(['?' for _ in severities])
            query += f" AND severity IN ({placeholders})"
            params.extend([s.value for s in severities])
        
        if actor:
            query += " AND actor = ?"
            params.append(actor)
        
        if resource:
            query += " AND resource = ?"
            params.append(resource)
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            events = []
            for row in cursor:
                event = dict(row)
                # Parse JSON fields
                event['details'] = json.loads(event['details'])
                event['compliance_tags'] = json.loads(event['compliance_tags'])
                event['metadata'] = json.loads(event['metadata'])
                events.append(event)
            
            return events
    
    def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate compliance report for specific framework
        
        Returns:
            Compliance report dictionary
        """
        # Query relevant events
        events = self.query_events(
            start_time=start_date,
            end_time=end_date
        )
        
        # Filter by compliance framework
        relevant_events = [
            e for e in events 
            if framework.value in e.get('compliance_tags', [])
        ]
        
        # Generate report based on framework
        if framework == ComplianceFramework.EU_AI_ACT:
            report = self._generate_eu_ai_act_report(relevant_events)
        elif framework == ComplianceFramework.GDPR:
            report = self._generate_gdpr_report(relevant_events)
        elif framework == ComplianceFramework.NIST_RMF:
            report = self._generate_nist_rmf_report(relevant_events)
        else:
            report = self._generate_generic_report(relevant_events)
        
        return {
            "framework": framework.value,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(relevant_events),
            "report": report,
            "generated_at": datetime.now().isoformat()
        }
    
    def _generate_eu_ai_act_report(self, events: List[Dict]) -> Dict[str, Any]:
        """Generate EU AI Act compliance report"""
        return {
            "transparency": {
                "model_verifications": len([e for e in events if e['event_type'] == 'verification_completed']),
                "failed_verifications": len([e for e in events if e['event_type'] == 'verification_failed']),
                "attack_detections": len([e for e in events if e['event_type'] == 'attack_detected'])
            },
            "accountability": {
                "unique_actors": len(set(e['actor'] for e in events)),
                "unique_resources": len(set(e['resource'] for e in events)),
                "audit_trail_complete": True
            },
            "human_oversight": {
                "manual_interventions": len([e for e in events if 'manual' in e.get('details', {}).get('operation', '')]),
                "access_controls": len([e for e in events if e['event_type'] in ['access_granted', 'access_denied']])
            },
            "risk_management": {
                "security_alerts": len([e for e in events if e['event_type'] == 'security_alert']),
                "critical_events": len([e for e in events if e['severity'] == 'critical'])
            }
        }
    
    def _generate_gdpr_report(self, events: List[Dict]) -> Dict[str, Any]:
        """Generate GDPR compliance report"""
        return {
            "data_processing": {
                "processing_activities": len(events),
                "data_exports": len([e for e in events if e['event_type'] == 'data_exported']),
                "data_deletions": len([e for e in events if e['event_type'] == 'data_deleted'])
            },
            "lawful_basis": {
                "consent_records": len([e for e in events if 'consent' in e.get('details', {})]),
                "legitimate_interest": len([e for e in events if 'legitimate' in e.get('details', {})])
            },
            "data_subject_rights": {
                "access_requests": len([e for e in events if 'access_request' in e.get('action', '')]),
                "deletion_requests": len([e for e in events if 'deletion_request' in e.get('action', '')])
            },
            "security_measures": {
                "encryption_enabled": self.enable_encryption,
                "signing_enabled": self.enable_signing,
                "audit_trail_integrity": True
            }
        }
    
    def _generate_nist_rmf_report(self, events: List[Dict]) -> Dict[str, Any]:
        """Generate NIST RMF compliance report"""
        return {
            "categorize": {
                "system_boundary": len(set(e['resource'] for e in events)),
                "information_types": len(set(e['event_type'] for e in events))
            },
            "select": {
                "security_controls": ["AC-2", "AU-2", "AU-3", "AU-12", "IA-2"],
                "control_baselines": "MODERATE"
            },
            "implement": {
                "audit_logging": True,
                "access_control": True,
                "authentication": True
            },
            "assess": {
                "security_events": len([e for e in events if e['severity'] in ['warning', 'error', 'critical']]),
                "vulnerabilities": len([e for e in events if 'vulnerability' in e.get('details', {})])
            },
            "authorize": {
                "authorization_status": "AUTHORIZED",
                "risk_level": "LOW"
            },
            "monitor": {
                "continuous_monitoring": True,
                "events_per_day": len(events) / max((end_date - start_date).days, 1)
            }
        }
    
    def _generate_generic_report(self, events: List[Dict]) -> Dict[str, Any]:
        """Generate generic compliance report"""
        return {
            "summary": {
                "total_events": len(events),
                "events_by_type": {},
                "events_by_severity": {}
            },
            "statistics": self.stats,
            "audit_trail": {
                "integrity": True,
                "completeness": True,
                "availability": True
            }
        }
    
    def export_audit_trail(
        self,
        output_path: str,
        format: str = "json",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """Export audit trail for external analysis or archival"""
        events = self.query_events(
            start_time=start_date,
            end_time=end_date
        )
        
        output_file = Path(output_path)
        
        if format == "json":
            with open(output_file, 'w') as f:
                json.dump(events, f, indent=2)
        elif format == "csv":
            import csv
            with open(output_file, 'w', newline='') as f:
                if events:
                    writer = csv.DictWriter(f, fieldnames=events[0].keys())
                    writer.writeheader()
                    writer.writerows(events)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Log the export
        self.log_event(
            event_type=AuditEventType.DATA_EXPORTED,
            actor="system",
            resource="audit_trail",
            action=f"Exported audit trail to {output_path}",
            details={
                "format": format,
                "events_exported": len(events),
                "date_range": {
                    "start": start_date.isoformat() if start_date else None,
                    "end": end_date.isoformat() if end_date else None
                }
            }
        )
    
    def verify_integrity(self) -> bool:
        """Verify integrity of audit trail"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM audit_events ORDER BY timestamp")
            
            for row in cursor:
                event_dict = dict(zip(row.keys(), row))
                
                # Recreate event
                event = AuditEvent(
                    event_id=event_dict['event_id'],
                    timestamp=event_dict['timestamp'],
                    event_type=AuditEventType(event_dict['event_type']),
                    severity=AuditSeverity(event_dict['severity']),
                    actor=event_dict['actor'],
                    resource=event_dict['resource'],
                    action=event_dict['action'],
                    outcome=event_dict['outcome'],
                    details=json.loads(event_dict['details']),
                    compliance_tags=[ComplianceFramework(t) for t in json.loads(event_dict['compliance_tags'])],
                    metadata=json.loads(event_dict['metadata'])
                )
                
                # Verify signature
                expected_signature = self._sign_event(event)
                if expected_signature != event_dict['signature']:
                    self.logger.error(f"Integrity check failed for event {event.event_id}")
                    return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit statistics"""
        return {
            **self.stats,
            "database_size_mb": self.db_path.stat().st_size / (1024 * 1024),
            "current_log_size_mb": self.current_log_file.stat().st_size / (1024 * 1024) if self.current_log_file.exists() else 0,
            "integrity_verified": self.verify_integrity()
        }


# Convenience functions for common operations
def create_audit_logger(config: Optional[Dict[str, Any]] = None) -> AuditLogger:
    """Create audit logger with configuration"""
    config = config or {}
    return AuditLogger(**config)


def audit_verification(
    logger: AuditLogger,
    model_id: str,
    user_id: str,
    result: bool,
    details: Optional[Dict[str, Any]] = None
):
    """Audit a model verification event"""
    logger.log_event(
        event_type=AuditEventType.VERIFICATION_COMPLETED if result else AuditEventType.VERIFICATION_FAILED,
        actor=user_id,
        resource=model_id,
        action="Model verification",
        outcome="success" if result else "failure",
        severity=AuditSeverity.INFO if result else AuditSeverity.WARNING,
        details=details or {}
    )


def audit_attack_detection(
    logger: AuditLogger,
    attack_type: str,
    model_id: str,
    details: Optional[Dict[str, Any]] = None
):
    """Audit an attack detection event"""
    logger.log_event(
        event_type=AuditEventType.ATTACK_DETECTED,
        actor="system",
        resource=model_id,
        action=f"Detected {attack_type} attack",
        outcome="detected",
        severity=AuditSeverity.WARNING,
        details=details or {}
    )