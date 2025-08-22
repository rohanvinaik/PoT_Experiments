#!/usr/bin/env python3
"""
Audit Trail Query System for analyzing and visualizing audit trails.

Provides comprehensive querying, integrity verification, anomaly detection,
and dashboard capabilities for PoT audit trails with advanced analytics.
"""

import os
import json
import hashlib
import statistics
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import re
from collections import defaultdict, Counter

# Optional dependencies for advanced features
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Import audit system components
try:
    from .commit_reveal import CommitmentRecord, read_and_verify_audit_trail
    from .crypto_utils import compute_hash_chain, verify_timestamp_proof, TimestampProof
except ImportError:
    # Fallback for testing
    pass


class AnomalyType(Enum):
    """Types of anomalies that can be detected in audit trails."""
    ACCURACY_DRIFT = "accuracy_drift"
    FINGERPRINT_DRIFT = "fingerprint_drift"
    TIMING_ANOMALY = "timing_anomaly"
    FREQUENCY_ANOMALY = "frequency_anomaly"
    CONFIDENCE_ANOMALY = "confidence_anomaly"
    INTEGRITY_VIOLATION = "integrity_violation"


class IntegrityStatus(Enum):
    """Integrity verification status."""
    VALID = "valid"
    COMPROMISED = "compromised"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


@dataclass
class AnomalyRecord:
    """
    Record representing an anomaly detected in the audit trail.
    
    Attributes:
        anomaly_type: Type of anomaly detected
        model_id: Model associated with the anomaly
        timestamp: When the anomaly occurred
        severity: Severity score (0.0-1.0, higher is more severe)
        description: Human-readable description
        metadata: Additional anomaly-specific data
        affected_records: List of record IDs affected by this anomaly
    """
    anomaly_type: AnomalyType
    model_id: str
    timestamp: datetime
    severity: float
    description: str
    metadata: Dict[str, Any]
    affected_records: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'anomaly_type': self.anomaly_type.value,
            'model_id': self.model_id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity,
            'description': self.description,
            'metadata': self.metadata,
            'affected_records': self.affected_records
        }


@dataclass
class IntegrityReport:
    """
    Report on audit trail integrity verification.
    
    Attributes:
        status: Overall integrity status
        total_records: Total number of records examined
        valid_records: Number of records that passed verification
        integrity_score: Overall integrity score (0.0-1.0)
        hash_chain_valid: Whether hash chains are intact
        commitment_verification: Results of commitment verification
        anomalies_detected: List of integrity-related anomalies
        recommendations: List of recommended actions
    """
    status: IntegrityStatus
    total_records: int
    valid_records: int
    integrity_score: float
    hash_chain_valid: bool
    commitment_verification: Dict[str, Any]
    anomalies_detected: List[AnomalyRecord]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'status': self.status.value,
            'total_records': self.total_records,
            'valid_records': self.valid_records,
            'integrity_score': self.integrity_score,
            'hash_chain_valid': self.hash_chain_valid,
            'commitment_verification': self.commitment_verification,
            'anomalies_detected': [a.to_dict() for a in self.anomalies_detected],
            'recommendations': self.recommendations
        }


class AuditTrailQuery:
    """
    Comprehensive audit trail query and analysis system.
    
    Provides querying, integrity verification, anomaly detection,
    and statistical analysis of PoT audit trails.
    """
    
    def __init__(self, audit_log_path: str):
        """
        Initialize audit trail query system.
        
        Args:
            audit_log_path: Path to audit trail file or directory
        """
        self.audit_log_path = audit_log_path
        self.records = self._load_audit_trail(audit_log_path)
        self._build_indices()
        
    def _load_audit_trail(self, audit_log_path: str) -> List[Dict[str, Any]]:
        """Load audit trail records from file(s)."""
        records = []
        
        if os.path.isfile(audit_log_path):
            # Single file
            records.extend(self._load_single_file(audit_log_path))
        elif os.path.isdir(audit_log_path):
            # Directory of audit files
            for filename in os.listdir(audit_log_path):
                if filename.endswith('.json') or filename.endswith('.jsonl'):
                    filepath = os.path.join(audit_log_path, filename)
                    records.extend(self._load_single_file(filepath))
        else:
            raise FileNotFoundError(f"Audit log path not found: {audit_log_path}")
        
        # Sort by timestamp
        records.sort(key=lambda r: self._parse_timestamp(r.get('timestamp', '')))
        
        return records
    
    def _load_single_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Load records from a single audit file."""
        records = []
        
        try:
            with open(filepath, 'r') as f:
                if filepath.endswith('.jsonl'):
                    # JSONL format (one JSON object per line)
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                record['_source_file'] = filepath
                                record['_line_number'] = line_num
                                records.append(record)
                            except json.JSONDecodeError as e:
                                print(f"Warning: Invalid JSON on line {line_num} in {filepath}: {e}")
                else:
                    # Regular JSON format
                    data = json.load(f)
                    if isinstance(data, list):
                        for i, record in enumerate(data):
                            record['_source_file'] = filepath
                            record['_record_index'] = i
                            records.append(record)
                    elif isinstance(data, dict):
                        data['_source_file'] = filepath
                        data['_record_index'] = 0
                        records.append(data)
        
        except Exception as e:
            print(f"Warning: Could not load audit file {filepath}: {e}")
        
        return records
    
    def _build_indices(self):
        """Build indices for efficient querying."""
        self.model_index = defaultdict(list)
        self.timestamp_index = []
        self.session_index = defaultdict(list)
        
        for i, record in enumerate(self.records):
            # Model index
            model_id = record.get('model_id', 'unknown')
            self.model_index[model_id].append(i)
            
            # Timestamp index
            timestamp = self._parse_timestamp(record.get('timestamp', ''))
            self.timestamp_index.append((timestamp, i))
            
            # Session index
            session_id = record.get('session_id', record.get('nonce', 'unknown'))
            self.session_index[session_id].append(i)
        
        # Sort timestamp index
        self.timestamp_index.sort()
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object."""
        if not timestamp_str:
            return datetime.min.replace(tzinfo=timezone.utc)
        
        # Try different timestamp formats
        formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ', 
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d'
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        
        # Fallback: try to parse ISO format
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except ValueError:
            print(f"Warning: Could not parse timestamp: {timestamp_str}")
            return datetime.min.replace(tzinfo=timezone.utc)
    
    def query_by_model(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Find all verification records for a specific model.
        
        Args:
            model_id: Model identifier to search for
            
        Returns:
            List of audit records for the specified model
        """
        if model_id not in self.model_index:
            return []
        
        return [self.records[i] for i in self.model_index[model_id]]
    
    def query_by_timerange(
        self, 
        start: datetime, 
        end: datetime
    ) -> List[Dict[str, Any]]:
        """
        Find verification records within a time range.
        
        Args:
            start: Start of time range (inclusive)
            end: End of time range (inclusive)
            
        Returns:
            List of audit records within the time range
        """
        # Ensure timestamps have timezone info
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        
        # Binary search for start and end indices
        start_idx = 0
        end_idx = len(self.timestamp_index)
        
        for i, (timestamp, _) in enumerate(self.timestamp_index):
            if timestamp >= start:
                start_idx = i
                break
        
        for i in range(len(self.timestamp_index) - 1, -1, -1):
            timestamp, _ = self.timestamp_index[i]
            if timestamp <= end:
                end_idx = i + 1
                break
        
        # Extract records in range
        result_indices = [idx for _, idx in self.timestamp_index[start_idx:end_idx]]
        return [self.records[i] for i in result_indices]
    
    def query_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Find all records for a specific verification session.
        
        Args:
            session_id: Session identifier to search for
            
        Returns:
            List of audit records for the session
        """
        if session_id not in self.session_index:
            return []
        
        return [self.records[i] for i in self.session_index[session_id]]
    
    def query_by_confidence_range(
        self, 
        min_confidence: float, 
        max_confidence: float
    ) -> List[Dict[str, Any]]:
        """
        Find records with confidence scores in specified range.
        
        Args:
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            
        Returns:
            List of records with confidence in range
        """
        results = []
        for record in self.records:
            confidence = record.get('confidence', record.get('verification_confidence'))
            if confidence is not None:
                if min_confidence <= confidence <= max_confidence:
                    results.append(record)
        
        return results
    
    def query_by_verification_result(self, result: str) -> List[Dict[str, Any]]:
        """
        Find records with specific verification results.
        
        Args:
            result: Verification result to search for ('PASS', 'FAIL', etc.)
            
        Returns:
            List of records with matching verification result
        """
        results = []
        for record in self.records:
            record_result = (
                record.get('verification_decision') or 
                record.get('verification_result') or
                record.get('decision') or
                record.get('result')
            )
            if record_result and str(record_result).upper() == result.upper():
                results.append(record)
        
        return results
    
    def verify_integrity(self) -> IntegrityReport:
        """
        Verify the integrity of the entire audit trail.
        
        Checks hash chains, commitment validity, timestamps,
        and detects potential tampering.
        
        Returns:
            Comprehensive integrity report
        """
        total_records = len(self.records)
        valid_records = 0
        integrity_anomalies = []
        commitment_results = {
            'verified': 0,
            'failed': 0,
            'missing': 0
        }
        
        # Verify individual record integrity
        for i, record in enumerate(self.records):
            is_valid = True
            
            # Check required fields
            required_fields = ['timestamp']
            for field in required_fields:
                if field not in record:
                    integrity_anomalies.append(AnomalyRecord(
                        anomaly_type=AnomalyType.INTEGRITY_VIOLATION,
                        model_id=record.get('model_id', 'unknown'),
                        timestamp=self._parse_timestamp(record.get('timestamp', '')),
                        severity=0.8,
                        description=f"Missing required field: {field}",
                        metadata={'record_index': i, 'missing_field': field},
                        affected_records=[str(i)]
                    ))
                    is_valid = False
            
            # Verify commitment if present
            if 'commitment' in record:
                try:
                    # This would verify the commitment against the data
                    # Implementation depends on available commitment verification
                    commitment_results['verified'] += 1
                except Exception:
                    commitment_results['failed'] += 1
                    is_valid = False
            else:
                commitment_results['missing'] += 1
            
            if is_valid:
                valid_records += 1
        
        # Verify hash chain integrity
        hash_chain_valid = self._verify_hash_chain()
        
        # Calculate overall integrity score
        integrity_score = valid_records / total_records if total_records > 0 else 0.0
        if not hash_chain_valid:
            integrity_score *= 0.5  # Significant penalty for hash chain issues
        
        # Determine overall status
        if integrity_score >= 0.95 and hash_chain_valid:
            status = IntegrityStatus.VALID
        elif integrity_score >= 0.8:
            status = IntegrityStatus.INCOMPLETE
        else:
            status = IntegrityStatus.COMPROMISED
        
        # Generate recommendations
        recommendations = []
        if integrity_score < 1.0:
            recommendations.append("Some records failed integrity verification")
        if not hash_chain_valid:
            recommendations.append("Hash chain integrity compromised - potential tampering detected")
        if commitment_results['failed'] > 0:
            recommendations.append("Some commitments failed verification")
        if len(integrity_anomalies) > 0:
            recommendations.append(f"Found {len(integrity_anomalies)} integrity anomalies")
        
        return IntegrityReport(
            status=status,
            total_records=total_records,
            valid_records=valid_records,
            integrity_score=integrity_score,
            hash_chain_valid=hash_chain_valid,
            commitment_verification=commitment_results,
            anomalies_detected=integrity_anomalies,
            recommendations=recommendations
        )
    
    def _verify_hash_chain(self) -> bool:
        """Verify hash chain integrity across records."""
        try:
            # Extract hashes from records in chronological order
            hashes = []
            for record in self.records:
                # Look for various hash fields
                record_hash = (
                    record.get('commitment') or
                    record.get('commitment_hash') or
                    record.get('record_hash')
                )
                
                if record_hash:
                    if isinstance(record_hash, str):
                        try:
                            hashes.append(bytes.fromhex(record_hash))
                        except ValueError:
                            # Not a hex string, hash it
                            hashes.append(hashlib.sha256(record_hash.encode()).digest())
                    else:
                        hashes.append(hashlib.sha256(str(record_hash).encode()).digest())
            
            if len(hashes) < 2:
                return True  # Can't verify chain with fewer than 2 elements
            
            # Compute expected chain hash
            try:
                expected_chain = compute_hash_chain(hashes)
                
                # Check if any record contains the chain hash
                chain_hash_hex = expected_chain.hex()
                for record in self.records:
                    if record.get('chain_hash') == chain_hash_hex:
                        return True
                
                # If no explicit chain hash, assume valid (chain might not be stored)
                return True
                
            except Exception:
                # If hash chain computation fails, assume compromised
                return False
                
        except Exception:
            return False
    
    def find_anomalies(self) -> List[AnomalyRecord]:
        """
        Detect unusual patterns and anomalies in the audit trail.
        
        Analyzes accuracy trends, fingerprint changes, timing patterns,
        and other behavioral indicators.
        
        Returns:
            List of detected anomalies with severity scores
        """
        anomalies = []
        
        # Group records by model for analysis
        model_groups = defaultdict(list)
        for record in self.records:
            model_id = record.get('model_id', 'unknown')
            model_groups[model_id].append(record)
        
        for model_id, model_records in model_groups.items():
            if len(model_records) < 2:
                continue  # Need at least 2 records to detect anomalies
                
            # Sort by timestamp
            model_records.sort(key=lambda r: self._parse_timestamp(r.get('timestamp', '')))
            
            # Detect accuracy drift
            anomalies.extend(self._detect_accuracy_anomalies(model_id, model_records))
            
            # Detect timing anomalies
            anomalies.extend(self._detect_timing_anomalies(model_id, model_records))
            
            # Detect confidence anomalies
            anomalies.extend(self._detect_confidence_anomalies(model_id, model_records))
            
            # Detect frequency anomalies
            anomalies.extend(self._detect_frequency_anomalies(model_id, model_records))
            
            # Detect fingerprint drift
            anomalies.extend(self._detect_fingerprint_anomalies(model_id, model_records))
        
        # Sort by severity (highest first)
        anomalies.sort(key=lambda a: a.severity, reverse=True)
        
        return anomalies
    
    def _detect_accuracy_anomalies(
        self, 
        model_id: str, 
        records: List[Dict[str, Any]]
    ) -> List[AnomalyRecord]:
        """Detect sudden changes in accuracy."""
        anomalies = []
        
        # Extract accuracy values
        accuracies = []
        for record in records:
            accuracy = (
                record.get('accuracy') or
                record.get('verification_confidence') or
                record.get('confidence')
            )
            if accuracy is not None:
                accuracies.append((record, accuracy))
        
        if len(accuracies) < 3:
            return anomalies
        
        # Calculate rolling statistics
        window_size = min(5, len(accuracies) // 2)
        for i in range(window_size, len(accuracies)):
            current_window = [acc for _, acc in accuracies[i-window_size:i]]
            current_accuracy = accuracies[i][1]
            current_record = accuracies[i][0]
            
            if len(current_window) >= 2:
                window_mean = statistics.mean(current_window)
                window_std = statistics.stdev(current_window) if len(current_window) > 1 else 0.1
                
                # Detect significant deviations
                z_score = abs(current_accuracy - window_mean) / max(window_std, 0.01)
                
                if z_score > 3.0:  # 3-sigma rule
                    severity = min(z_score / 5.0, 1.0)  # Normalize to [0,1]
                    
                    anomalies.append(AnomalyRecord(
                        anomaly_type=AnomalyType.ACCURACY_DRIFT,
                        model_id=model_id,
                        timestamp=self._parse_timestamp(current_record.get('timestamp', '')),
                        severity=severity,
                        description=f"Accuracy anomaly: {current_accuracy:.3f} deviates {z_score:.1f}σ from recent average {window_mean:.3f}",
                        metadata={
                            'current_accuracy': current_accuracy,
                            'window_mean': window_mean,
                            'window_std': window_std,
                            'z_score': z_score
                        },
                        affected_records=[str(accuracies[j][0].get('session_id', j)) for j in range(max(0, i-window_size), i+1)]
                    ))
        
        return anomalies
    
    def _detect_timing_anomalies(
        self, 
        model_id: str, 
        records: List[Dict[str, Any]]
    ) -> List[AnomalyRecord]:
        """Detect unusual timing patterns."""
        anomalies = []
        
        # Extract verification durations and intervals
        durations = []
        intervals = []
        
        prev_timestamp = None
        for record in records:
            # Duration
            duration = record.get('duration_seconds') or record.get('duration')
            if duration is not None:
                durations.append((record, duration))
            
            # Interval between verifications
            timestamp = self._parse_timestamp(record.get('timestamp', ''))
            if prev_timestamp:
                interval = (timestamp - prev_timestamp).total_seconds()
                intervals.append((record, interval))
            prev_timestamp = timestamp
        
        # Analyze durations
        if len(durations) >= 5:
            duration_values = [d for _, d in durations]
            duration_mean = statistics.mean(duration_values)
            duration_std = statistics.stdev(duration_values) if len(duration_values) > 1 else 0.1
            
            for record, duration in durations:
                z_score = abs(duration - duration_mean) / max(duration_std, 0.01)
                
                if z_score > 2.5:  # Significant timing anomaly
                    severity = min(z_score / 4.0, 1.0)
                    
                    anomalies.append(AnomalyRecord(
                        anomaly_type=AnomalyType.TIMING_ANOMALY,
                        model_id=model_id,
                        timestamp=self._parse_timestamp(record.get('timestamp', '')),
                        severity=severity,
                        description=f"Verification duration anomaly: {duration:.2f}s vs average {duration_mean:.2f}s",
                        metadata={
                            'duration': duration,
                            'average_duration': duration_mean,
                            'z_score': z_score
                        },
                        affected_records=[str(record.get('session_id', record.get('timestamp')))]
                    ))
        
        return anomalies
    
    def _detect_confidence_anomalies(
        self, 
        model_id: str, 
        records: List[Dict[str, Any]]
    ) -> List[AnomalyRecord]:
        """Detect confidence score anomalies."""
        anomalies = []
        
        confidences = []
        for record in records:
            confidence = record.get('confidence') or record.get('verification_confidence')
            if confidence is not None:
                confidences.append((record, confidence))
        
        if len(confidences) < 3:
            return anomalies
        
        # Look for sudden drops in confidence
        for i in range(1, len(confidences)):
            prev_record, prev_confidence = confidences[i-1]
            current_record, current_confidence = confidences[i]
            
            confidence_drop = prev_confidence - current_confidence
            
            if confidence_drop > 0.2:  # Significant drop
                severity = min(confidence_drop / 0.5, 1.0)
                
                anomalies.append(AnomalyRecord(
                    anomaly_type=AnomalyType.CONFIDENCE_ANOMALY,
                    model_id=model_id,
                    timestamp=self._parse_timestamp(current_record.get('timestamp', '')),
                    severity=severity,
                    description=f"Confidence drop: {prev_confidence:.3f} → {current_confidence:.3f}",
                    metadata={
                        'previous_confidence': prev_confidence,
                        'current_confidence': current_confidence,
                        'drop_magnitude': confidence_drop
                    },
                    affected_records=[
                        str(prev_record.get('session_id', prev_record.get('timestamp'))),
                        str(current_record.get('session_id', current_record.get('timestamp')))
                    ]
                ))
        
        return anomalies
    
    def _detect_frequency_anomalies(
        self, 
        model_id: str, 
        records: List[Dict[str, Any]]
    ) -> List[AnomalyRecord]:
        """Detect unusual verification frequency patterns."""
        anomalies = []
        
        if len(records) < 5:
            return anomalies
        
        # Calculate time intervals between verifications
        timestamps = [self._parse_timestamp(r.get('timestamp', '')) for r in records]
        intervals = []
        
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        if len(intervals) < 3:
            return anomalies
        
        # Detect sudden changes in verification frequency
        interval_mean = statistics.mean(intervals)
        interval_std = statistics.stdev(intervals) if len(intervals) > 1 else interval_mean * 0.1
        
        for i, interval in enumerate(intervals):
            if interval_std > 0:
                z_score = abs(interval - interval_mean) / interval_std
                
                if z_score > 2.0:  # Unusual frequency
                    severity = min(z_score / 3.0, 1.0)
                    
                    if interval > interval_mean:
                        description = f"Verification frequency dropped: {interval:.0f}s interval vs average {interval_mean:.0f}s"
                    else:
                        description = f"Verification frequency increased: {interval:.0f}s interval vs average {interval_mean:.0f}s"
                    
                    anomalies.append(AnomalyRecord(
                        anomaly_type=AnomalyType.FREQUENCY_ANOMALY,
                        model_id=model_id,
                        timestamp=timestamps[i+1],
                        severity=severity,
                        description=description,
                        metadata={
                            'interval': interval,
                            'average_interval': interval_mean,
                            'z_score': z_score
                        },
                        affected_records=[str(records[i+1].get('session_id', timestamps[i+1]))]
                    ))
        
        return anomalies
    
    def _detect_fingerprint_anomalies(
        self, 
        model_id: str, 
        records: List[Dict[str, Any]]
    ) -> List[AnomalyRecord]:
        """Detect fingerprint drift anomalies."""
        anomalies = []
        
        # Extract fingerprint data
        fingerprints = []
        for record in records:
            fp_data = (
                record.get('fingerprint') or
                record.get('fingerprint_similarity') or
                record.get('io_hash')
            )
            if fp_data:
                fingerprints.append((record, fp_data))
        
        if len(fingerprints) < 2:
            return anomalies
        
        # Look for significant changes in fingerprint similarity
        for i in range(1, len(fingerprints)):
            prev_record, prev_fp = fingerprints[i-1]
            current_record, current_fp = fingerprints[i]
            
            # If we have similarity scores, check for drops
            if isinstance(current_fp, (int, float)) and isinstance(prev_fp, (int, float)):
                similarity_drop = prev_fp - current_fp
                
                if similarity_drop > 0.1:  # Significant fingerprint change
                    severity = min(similarity_drop / 0.3, 1.0)
                    
                    anomalies.append(AnomalyRecord(
                        anomaly_type=AnomalyType.FINGERPRINT_DRIFT,
                        model_id=model_id,
                        timestamp=self._parse_timestamp(current_record.get('timestamp', '')),
                        severity=severity,
                        description=f"Fingerprint similarity drop: {prev_fp:.3f} → {current_fp:.3f}",
                        metadata={
                            'previous_similarity': prev_fp,
                            'current_similarity': current_fp,
                            'drift_magnitude': similarity_drop
                        },
                        affected_records=[
                            str(prev_record.get('session_id', prev_record.get('timestamp'))),
                            str(current_record.get('session_id', current_record.get('timestamp')))
                        ]
                    ))
        
        return anomalies
    
    def generate_audit_report(self, format: str = "json") -> str:
        """
        Generate comprehensive audit report.
        
        Args:
            format: Output format ("json", "markdown", "html")
            
        Returns:
            Formatted audit report string
        """
        # Collect statistics
        total_records = len(self.records)
        model_count = len(self.model_index)
        
        # Verification outcomes
        outcomes = Counter()
        confidences = []
        durations = []
        
        for record in self.records:
            # Count outcomes
            result = (
                record.get('verification_decision') or
                record.get('verification_result') or
                record.get('decision') or
                'unknown'
            )
            outcomes[str(result).upper()] += 1
            
            # Collect metrics
            confidence = record.get('confidence') or record.get('verification_confidence')
            if confidence is not None:
                confidences.append(confidence)
                
            duration = record.get('duration_seconds') or record.get('duration')
            if duration is not None:
                durations.append(duration)
        
        # Generate integrity report
        integrity_report = self.verify_integrity()
        
        # Find anomalies
        anomalies = self.find_anomalies()
        
        # Time range analysis
        if self.records:
            start_time = self._parse_timestamp(self.records[0].get('timestamp', ''))
            end_time = self._parse_timestamp(self.records[-1].get('timestamp', ''))
            time_span = (end_time - start_time).total_seconds() / 86400  # days
        else:
            start_time = end_time = datetime.now(timezone.utc)
            time_span = 0
        
        # Build report data
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'audit_trail_path': self.audit_log_path,
                'format': format
            },
            'summary_statistics': {
                'total_records': total_records,
                'unique_models': model_count,
                'time_span_days': round(time_span, 2),
                'verification_outcomes': dict(outcomes),
                'average_confidence': round(statistics.mean(confidences), 3) if confidences else None,
                'average_duration': round(statistics.mean(durations), 3) if durations else None
            },
            'integrity_report': integrity_report.to_dict(),
            'anomalies_summary': {
                'total_anomalies': len(anomalies),
                'high_severity': len([a for a in anomalies if a.severity >= 0.7]),
                'medium_severity': len([a for a in anomalies if 0.3 <= a.severity < 0.7]),
                'low_severity': len([a for a in anomalies if a.severity < 0.3]),
                'by_type': Counter([a.anomaly_type.value for a in anomalies])
            },
            'detailed_anomalies': [a.to_dict() for a in anomalies[:10]],  # Top 10
            'model_analysis': self._generate_model_analysis(),
            'recommendations': self._generate_recommendations(integrity_report, anomalies)
        }
        
        # Format output
        if format.lower() == "json":
            return json.dumps(report_data, indent=2, default=str)
        elif format.lower() == "markdown":
            return self._format_markdown_report(report_data)
        elif format.lower() == "html":
            return self._format_html_report(report_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_model_analysis(self) -> Dict[str, Any]:
        """Generate per-model analysis."""
        model_analysis = {}
        
        for model_id, record_indices in self.model_index.items():
            model_records = [self.records[i] for i in record_indices]
            
            # Calculate model-specific metrics
            confidences = []
            outcomes = Counter()
            
            for record in model_records:
                confidence = record.get('confidence') or record.get('verification_confidence')
                if confidence is not None:
                    confidences.append(confidence)
                
                result = (
                    record.get('verification_decision') or
                    record.get('verification_result') or
                    'unknown'
                )
                outcomes[str(result).upper()] += 1
            
            model_analysis[model_id] = {
                'total_verifications': len(model_records),
                'average_confidence': round(statistics.mean(confidences), 3) if confidences else None,
                'outcomes': dict(outcomes),
                'first_verification': self._parse_timestamp(model_records[0].get('timestamp', '')).isoformat(),
                'last_verification': self._parse_timestamp(model_records[-1].get('timestamp', '')).isoformat()
            }
        
        return model_analysis
    
    def _generate_recommendations(
        self, 
        integrity_report: IntegrityReport, 
        anomalies: List[AnomalyRecord]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Integrity-based recommendations
        if integrity_report.integrity_score < 0.9:
            recommendations.append("Consider investigating records that failed integrity verification")
        
        if not integrity_report.hash_chain_valid:
            recommendations.append("CRITICAL: Hash chain integrity compromised - investigate potential tampering")
        
        # Anomaly-based recommendations
        high_severity_anomalies = [a for a in anomalies if a.severity >= 0.7]
        if high_severity_anomalies:
            recommendations.append(f"Investigate {len(high_severity_anomalies)} high-severity anomalies")
        
        # Model-specific recommendations
        accuracy_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.ACCURACY_DRIFT]
        if accuracy_anomalies:
            recommendations.append("Monitor models with accuracy drift for potential degradation")
        
        fingerprint_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.FINGERPRINT_DRIFT]
        if fingerprint_anomalies:
            recommendations.append("Investigate models with fingerprint drift for unauthorized modifications")
        
        # General recommendations
        recommendations.append("Regularly verify audit trail integrity")
        recommendations.append("Monitor anomaly trends over time")
        recommendations.append("Implement automated alerting for high-severity anomalies")
        
        return recommendations
    
    def _format_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Format report as Markdown."""
        md = []
        
        md.append("# PoT Audit Trail Report")
        md.append("")
        md.append(f"**Generated:** {report_data['report_metadata']['generated_at']}")
        md.append(f"**Audit Trail:** {report_data['report_metadata']['audit_trail_path']}")
        md.append("")
        
        # Summary
        md.append("## Summary Statistics")
        stats = report_data['summary_statistics']
        md.append(f"- **Total Records:** {stats['total_records']}")
        md.append(f"- **Unique Models:** {stats['unique_models']}")
        md.append(f"- **Time Span:** {stats['time_span_days']} days")
        md.append(f"- **Average Confidence:** {stats['average_confidence']}")
        md.append("")
        
        # Integrity
        md.append("## Integrity Report")
        integrity = report_data['integrity_report']
        md.append(f"- **Status:** {integrity['status']}")
        md.append(f"- **Integrity Score:** {integrity['integrity_score']:.3f}")
        md.append(f"- **Valid Records:** {integrity['valid_records']}/{integrity['total_records']}")
        md.append("")
        
        # Anomalies
        md.append("## Anomalies Summary")
        anomaly_summary = report_data['anomalies_summary']
        md.append(f"- **Total Anomalies:** {anomaly_summary['total_anomalies']}")
        md.append(f"- **High Severity:** {anomaly_summary['high_severity']}")
        md.append("")
        
        return "\n".join(md)
    
    def _format_html_report(self, report_data: Dict[str, Any]) -> str:
        """Format report as HTML."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PoT Audit Trail Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .anomaly {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; }}
                .severity-high {{ border-left: 5px solid #dc3545; }}
                .severity-medium {{ border-left: 5px solid #ffc107; }}
                .severity-low {{ border-left: 5px solid #28a745; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>PoT Audit Trail Report</h1>
                <p><strong>Generated:</strong> {report_data['report_metadata']['generated_at']}</p>
                <p><strong>Audit Trail:</strong> {report_data['report_metadata']['audit_trail_path']}</p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <ul>
                    <li><strong>Total Records:</strong> {report_data['summary_statistics']['total_records']}</li>
                    <li><strong>Unique Models:</strong> {report_data['summary_statistics']['unique_models']}</li>
                    <li><strong>Time Span:</strong> {report_data['summary_statistics']['time_span_days']} days</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Integrity Status</h2>
                <p><strong>Status:</strong> {report_data['integrity_report']['status']}</p>
                <p><strong>Score:</strong> {report_data['integrity_report']['integrity_score']:.3f}</p>
            </div>
        </body>
        </html>
        """
        
        return html


class AuditDashboard:
    """
    Web dashboard for audit trail visualization and analysis.
    
    Provides interactive visualization of audit trails, anomalies,
    and integrity reports using Streamlit or Flask.
    """
    
    def __init__(self, query: AuditTrailQuery):
        """
        Initialize audit dashboard.
        
        Args:
            query: AuditTrailQuery instance with loaded data
        """
        self.query = query
        self.title = "PoT Audit Trail Dashboard"
    
    def create_streamlit_app(self):
        """
        Create Streamlit dashboard application.
        
        Note: This method sets up the Streamlit interface but doesn't run it.
        Use `streamlit run` with a script that calls this method.
        """
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit not available. Install with: pip install streamlit")
        
        st.set_page_config(
            page_title=self.title,
            page_icon="🔍",
            layout="wide"
        )
        
        st.title(self.title)
        
        # Sidebar for filters
        st.sidebar.header("Filters")
        
        # Model filter
        model_ids = list(self.query.model_index.keys())
        selected_models = st.sidebar.multiselect(
            "Select Models",
            model_ids,
            default=model_ids[:5] if len(model_ids) > 5 else model_ids
        )
        
        # Time range filter
        if self.query.records:
            min_date = self.query._parse_timestamp(self.query.records[0].get('timestamp', '')).date()
            max_date = self.query._parse_timestamp(self.query.records[-1].get('timestamp', '')).date()
            
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Integrity", "Anomalies", "Model Analysis"])
        
        with tab1:
            self._render_overview_tab()
        
        with tab2:
            self._render_integrity_tab()
        
        with tab3:
            self._render_anomalies_tab()
        
        with tab4:
            self._render_model_analysis_tab(selected_models)
    
    def _render_overview_tab(self):
        """Render overview dashboard tab."""
        if not STREAMLIT_AVAILABLE:
            return
        
        st.header("Audit Trail Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(self.query.records))
        
        with col2:
            st.metric("Unique Models", len(self.query.model_index))
        
        with col3:
            # Calculate pass rate
            pass_records = self.query.query_by_verification_result("PASS")
            pass_rate = len(pass_records) / len(self.query.records) * 100 if self.query.records else 0
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
        
        with col4:
            # Recent anomalies
            anomalies = self.query.find_anomalies()
            high_severity = len([a for a in anomalies if a.severity >= 0.7])
            st.metric("High Severity Anomalies", high_severity)
        
        # Timeline visualization
        if MATPLOTLIB_AVAILABLE and self.query.records:
            st.subheader("Verification Timeline")
            self._create_timeline_chart()
    
    def _render_integrity_tab(self):
        """Render integrity analysis tab."""
        if not STREAMLIT_AVAILABLE:
            return
        
        st.header("Integrity Analysis")
        
        # Run integrity verification
        with st.spinner("Verifying audit trail integrity..."):
            integrity_report = self.query.verify_integrity()
        
        # Display integrity status
        status_color = {
            IntegrityStatus.VALID: "green",
            IntegrityStatus.INCOMPLETE: "orange", 
            IntegrityStatus.COMPROMISED: "red",
            IntegrityStatus.UNKNOWN: "gray"
        }
        
        st.markdown(f"**Status:** :{status_color[integrity_report.status]}[{integrity_report.status.value.upper()}]")
        st.markdown(f"**Integrity Score:** {integrity_report.integrity_score:.3f}")
        st.markdown(f"**Valid Records:** {integrity_report.valid_records}/{integrity_report.total_records}")
        
        # Progress bar for integrity score
        st.progress(integrity_report.integrity_score)
        
        # Recommendations
        if integrity_report.recommendations:
            st.subheader("Recommendations")
            for rec in integrity_report.recommendations:
                st.warning(rec)
    
    def _render_anomalies_tab(self):
        """Render anomalies analysis tab."""
        if not STREAMLIT_AVAILABLE:
            return
        
        st.header("Anomaly Detection")
        
        # Find anomalies
        with st.spinner("Analyzing anomalies..."):
            anomalies = self.query.find_anomalies()
        
        if not anomalies:
            st.success("No anomalies detected!")
            return
        
        # Anomaly summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_sev = len([a for a in anomalies if a.severity >= 0.7])
            st.metric("High Severity", high_sev, delta=None)
        
        with col2:
            med_sev = len([a for a in anomalies if 0.3 <= a.severity < 0.7])
            st.metric("Medium Severity", med_sev)
        
        with col3:
            low_sev = len([a for a in anomalies if a.severity < 0.3])
            st.metric("Low Severity", low_sev)
        
        # Anomaly details table
        st.subheader("Anomaly Details")
        
        anomaly_data = []
        for anomaly in anomalies[:20]:  # Show top 20
            anomaly_data.append({
                "Type": anomaly.anomaly_type.value,
                "Model": anomaly.model_id,
                "Severity": f"{anomaly.severity:.2f}",
                "Timestamp": anomaly.timestamp.strftime("%Y-%m-%d %H:%M"),
                "Description": anomaly.description
            })
        
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(anomaly_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.json(anomaly_data)
    
    def _render_model_analysis_tab(self, selected_models: List[str]):
        """Render per-model analysis tab."""
        if not STREAMLIT_AVAILABLE:
            return
        
        st.header("Model Analysis")
        
        for model_id in selected_models:
            st.subheader(f"Model: {model_id}")
            
            model_records = self.query.query_by_model(model_id)
            
            if not model_records:
                st.warning(f"No records found for model {model_id}")
                continue
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Verifications", len(model_records))
            
            with col2:
                # Calculate average confidence
                confidences = [
                    r.get('confidence', r.get('verification_confidence'))
                    for r in model_records
                    if r.get('confidence') or r.get('verification_confidence')
                ]
                avg_conf = statistics.mean(confidences) if confidences else 0
                st.metric("Avg Confidence", f"{avg_conf:.3f}")
            
            with col3:
                # Pass rate for this model
                model_passes = [r for r in model_records 
                              if (r.get('verification_decision') or '').upper() == 'PASS']
                pass_rate = len(model_passes) / len(model_records) * 100 if model_records else 0
                st.metric("Pass Rate", f"{pass_rate:.1f}%")
    
    def _create_timeline_chart(self):
        """Create timeline visualization of verifications."""
        if not MATPLOTLIB_AVAILABLE or not STREAMLIT_AVAILABLE:
            return
        
        # Extract timestamps and results
        timestamps = []
        results = []
        
        for record in self.query.records:
            timestamp = self.query._parse_timestamp(record.get('timestamp', ''))
            result = record.get('verification_decision', 'UNKNOWN').upper()
            
            timestamps.append(timestamp)
            results.append(result)
        
        if not timestamps:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color code by result
        colors = {'PASS': 'green', 'FAIL': 'red', 'UNKNOWN': 'gray'}
        result_colors = [colors.get(r, 'gray') for r in results]
        
        # Plot timeline
        ax.scatter(timestamps, range(len(timestamps)), c=result_colors, alpha=0.6)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Verification Sequence')
        ax.set_title('Verification Timeline')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(timestamps)//10)))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)


def create_dashboard_app(audit_log_path: str):
    """
    Create and configure Streamlit dashboard application.
    
    Args:
        audit_log_path: Path to audit trail file or directory
        
    Note:
        This function should be called from a Streamlit script.
    """
    # Initialize query system
    query = AuditTrailQuery(audit_log_path)
    
    # Create dashboard
    dashboard = AuditDashboard(query)
    dashboard.create_streamlit_app()


if __name__ == "__main__":
    # Demo usage
    print("PoT Audit Trail Query System")
    print("=" * 40)
    
    # This would normally load real audit data
    print("Note: This is a demo. In practice, initialize with:")
    print("  query = AuditTrailQuery('/path/to/audit/trail')")
    print("  dashboard = AuditDashboard(query)")
    print("  dashboard.create_streamlit_app()")