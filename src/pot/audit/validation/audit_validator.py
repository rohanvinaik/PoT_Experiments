"""
Audit Trail Validator

Provides comprehensive validation of audit trails, ensuring integrity,
consistency, and completeness of recorded operations.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Represents a single audit log entry"""
    timestamp: datetime
    operation: str
    actor: str
    resource: str
    outcome: str
    metadata: Dict[str, Any]
    hash: str
    previous_hash: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of audit validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]
    integrity_score: float


class AuditValidator:
    """
    Validates audit trails for integrity, consistency, and completeness.
    
    Features:
    - Cryptographic hash chain verification
    - Temporal consistency checking  
    - Completeness analysis
    - Anomaly detection
    - Cross-reference validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the audit validator.
        
        Args:
            config: Configuration dictionary with validation parameters
        """
        self.config = config or {}
        self.hash_algorithm = self.config.get('hash_algorithm', 'sha256')
        self.max_time_gap = self.config.get('max_time_gap', 3600)  # seconds
        self.require_signatures = self.config.get('require_signatures', False)
        self.entries: List[AuditEntry] = []
        self.validation_cache = {}
        
    def load_audit_trail(self, filepath: Path) -> List[AuditEntry]:
        """
        Load audit trail from file.
        
        Args:
            filepath: Path to audit log file
            
        Returns:
            List of parsed audit entries
        """
        entries = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        entry_data = json.loads(line)
                        entry = self._parse_entry(entry_data)
                        entries.append(entry)
        except Exception as e:
            logger.error(f"Failed to load audit trail: {e}")
            raise
            
        self.entries = entries
        return entries
    
    def _parse_entry(self, data: Dict[str, Any]) -> AuditEntry:
        """Parse raw audit data into AuditEntry"""
        return AuditEntry(
            timestamp=datetime.fromisoformat(data['timestamp']),
            operation=data['operation'],
            actor=data['actor'],
            resource=data['resource'],
            outcome=data['outcome'],
            metadata=data.get('metadata', {}),
            hash=data['hash'],
            previous_hash=data.get('previous_hash')
        )
    
    def validate(self, entries: Optional[List[AuditEntry]] = None) -> ValidationResult:
        """
        Perform comprehensive validation of audit trail.
        
        Args:
            entries: Audit entries to validate (uses loaded entries if None)
            
        Returns:
            ValidationResult with detailed findings
        """
        if entries is None:
            entries = self.entries
            
        if not entries:
            return ValidationResult(
                is_valid=False,
                errors=["No audit entries to validate"],
                warnings=[],
                statistics={},
                integrity_score=0.0
            )
        
        errors = []
        warnings = []
        
        # Perform various validation checks
        hash_valid, hash_errors = self._validate_hash_chain(entries)
        temporal_valid, temporal_errors = self._validate_temporal_consistency(entries)
        complete_valid, complete_warnings = self._check_completeness(entries)
        anomalies = self._detect_anomalies(entries)
        
        errors.extend(hash_errors)
        errors.extend(temporal_errors)
        warnings.extend(complete_warnings)
        warnings.extend(anomalies)
        
        # Calculate statistics
        statistics = self._calculate_statistics(entries)
        
        # Calculate integrity score
        integrity_score = self._calculate_integrity_score(
            hash_valid, temporal_valid, complete_valid, len(anomalies)
        )
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            statistics=statistics,
            integrity_score=integrity_score
        )
    
    def _validate_hash_chain(self, entries: List[AuditEntry]) -> Tuple[bool, List[str]]:
        """
        Validate cryptographic hash chain integrity.
        
        Args:
            entries: Audit entries to validate
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        for i, entry in enumerate(entries):
            # Verify hash of current entry
            computed_hash = self._compute_entry_hash(entry)
            if computed_hash != entry.hash:
                errors.append(f"Hash mismatch at entry {i}: expected {entry.hash}, got {computed_hash}")
            
            # Verify chain linkage
            if i > 0 and entry.previous_hash != entries[i-1].hash:
                errors.append(f"Chain broken at entry {i}: previous_hash doesn't match")
        
        return len(errors) == 0, errors
    
    def _compute_entry_hash(self, entry: AuditEntry) -> str:
        """Compute hash for an audit entry"""
        hasher = hashlib.new(self.hash_algorithm)
        
        # Create canonical representation
        canonical = {
            'timestamp': entry.timestamp.isoformat(),
            'operation': entry.operation,
            'actor': entry.actor,
            'resource': entry.resource,
            'outcome': entry.outcome,
            'metadata': entry.metadata,
            'previous_hash': entry.previous_hash
        }
        
        hasher.update(json.dumps(canonical, sort_keys=True).encode())
        return hasher.hexdigest()
    
    def _validate_temporal_consistency(self, entries: List[AuditEntry]) -> Tuple[bool, List[str]]:
        """
        Check temporal consistency of audit entries.
        
        Args:
            entries: Audit entries to validate
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        for i in range(1, len(entries)):
            time_delta = (entries[i].timestamp - entries[i-1].timestamp).total_seconds()
            
            # Check for backwards time travel
            if time_delta < 0:
                errors.append(f"Temporal inconsistency at entry {i}: timestamp goes backwards")
            
            # Check for suspicious gaps
            if time_delta > self.max_time_gap:
                errors.append(f"Suspicious time gap at entry {i}: {time_delta}s")
        
        return len(errors) == 0, errors
    
    def _check_completeness(self, entries: List[AuditEntry]) -> Tuple[bool, List[str]]:
        """
        Check for completeness of audit trail.
        
        Args:
            entries: Audit entries to check
            
        Returns:
            Tuple of (is_complete, warnings)
        """
        warnings = []
        
        # Check for required operations
        operations = {entry.operation for entry in entries}
        required_ops = self.config.get('required_operations', [])
        
        for op in required_ops:
            if op not in operations:
                warnings.append(f"Missing required operation: {op}")
        
        # Check for balanced operations (e.g., start/end pairs)
        operation_counts = {}
        for entry in entries:
            operation_counts[entry.operation] = operation_counts.get(entry.operation, 0) + 1
        
        # Check for unbalanced operations
        paired_ops = self.config.get('paired_operations', {})
        for start_op, end_op in paired_ops.items():
            start_count = operation_counts.get(start_op, 0)
            end_count = operation_counts.get(end_op, 0)
            if start_count != end_count:
                warnings.append(f"Unbalanced operations: {start_op}({start_count}) vs {end_op}({end_count})")
        
        return len(warnings) == 0, warnings
    
    def _detect_anomalies(self, entries: List[AuditEntry]) -> List[str]:
        """
        Detect anomalies in audit trail using statistical methods.
        
        Args:
            entries: Audit entries to analyze
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if len(entries) < 10:
            return anomalies  # Need sufficient data for statistical analysis
        
        # Analyze operation frequency
        operation_times = {}
        for entry in entries:
            if entry.operation not in operation_times:
                operation_times[entry.operation] = []
            operation_times[entry.operation].append(entry.timestamp)
        
        # Detect unusual patterns
        for operation, times in operation_times.items():
            if len(times) > 2:
                # Calculate inter-arrival times
                deltas = [(times[i+1] - times[i]).total_seconds() 
                         for i in range(len(times)-1)]
                
                if deltas:
                    mean_delta = np.mean(deltas)
                    std_delta = np.std(deltas)
                    
                    # Flag outliers (>3 sigma)
                    for i, delta in enumerate(deltas):
                        if abs(delta - mean_delta) > 3 * std_delta:
                            anomalies.append(
                                f"Anomalous timing for {operation} at index {i}: "
                                f"{delta}s (mean: {mean_delta:.2f}s)"
                            )
        
        # Check for unusual actors
        actor_counts = {}
        for entry in entries:
            actor_counts[entry.actor] = actor_counts.get(entry.actor, 0) + 1
        
        total_entries = len(entries)
        for actor, count in actor_counts.items():
            ratio = count / total_entries
            if ratio > 0.9:  # Single actor dominates
                anomalies.append(f"Single actor dominance: {actor} ({ratio:.2%})")
        
        return anomalies
    
    def _calculate_statistics(self, entries: List[AuditEntry]) -> Dict[str, Any]:
        """Calculate audit trail statistics"""
        if not entries:
            return {}
        
        operation_counts = {}
        actor_counts = {}
        outcome_counts = {'success': 0, 'failure': 0, 'partial': 0}
        
        for entry in entries:
            operation_counts[entry.operation] = operation_counts.get(entry.operation, 0) + 1
            actor_counts[entry.actor] = actor_counts.get(entry.actor, 0) + 1
            if entry.outcome in outcome_counts:
                outcome_counts[entry.outcome] += 1
        
        duration = (entries[-1].timestamp - entries[0].timestamp).total_seconds()
        
        return {
            'total_entries': len(entries),
            'duration_seconds': duration,
            'unique_operations': len(operation_counts),
            'unique_actors': len(actor_counts),
            'operation_counts': operation_counts,
            'actor_counts': actor_counts,
            'outcome_counts': outcome_counts,
            'entries_per_minute': len(entries) / (duration / 60) if duration > 0 else 0
        }
    
    def _calculate_integrity_score(
        self, 
        hash_valid: bool, 
        temporal_valid: bool,
        complete: bool,
        anomaly_count: int
    ) -> float:
        """
        Calculate overall integrity score.
        
        Args:
            hash_valid: Whether hash chain is valid
            temporal_valid: Whether temporal consistency is maintained
            complete: Whether audit trail is complete
            anomaly_count: Number of detected anomalies
            
        Returns:
            Integrity score between 0.0 and 1.0
        """
        score = 0.0
        
        # Hash chain integrity (40% weight)
        if hash_valid:
            score += 0.4
        
        # Temporal consistency (30% weight)
        if temporal_valid:
            score += 0.3
        
        # Completeness (20% weight)
        if complete:
            score += 0.2
        
        # Anomaly penalty (10% weight)
        anomaly_penalty = min(anomaly_count * 0.02, 0.1)
        score += (0.1 - anomaly_penalty)
        
        return min(max(score, 0.0), 1.0)
    
    def cross_validate(
        self, 
        primary_trail: List[AuditEntry],
        secondary_trail: List[AuditEntry]
    ) -> ValidationResult:
        """
        Cross-validate two audit trails for consistency.
        
        Args:
            primary_trail: Primary audit trail
            secondary_trail: Secondary audit trail for cross-validation
            
        Returns:
            ValidationResult with cross-validation findings
        """
        errors = []
        warnings = []
        
        # Create lookup maps
        primary_map = {(e.timestamp, e.operation): e for e in primary_trail}
        secondary_map = {(e.timestamp, e.operation): e for e in secondary_trail}
        
        # Check for missing entries
        for key in primary_map:
            if key not in secondary_map:
                warnings.append(f"Entry missing in secondary trail: {key}")
        
        for key in secondary_map:
            if key not in primary_map:
                warnings.append(f"Entry missing in primary trail: {key}")
        
        # Check for inconsistencies in common entries
        for key in primary_map:
            if key in secondary_map:
                p_entry = primary_map[key]
                s_entry = secondary_map[key]
                
                if p_entry.outcome != s_entry.outcome:
                    errors.append(f"Outcome mismatch at {key}: {p_entry.outcome} vs {s_entry.outcome}")
                
                if p_entry.actor != s_entry.actor:
                    errors.append(f"Actor mismatch at {key}: {p_entry.actor} vs {s_entry.actor}")
        
        # Calculate correlation score
        common_keys = set(primary_map.keys()) & set(secondary_map.keys())
        if primary_map and secondary_map:
            correlation = len(common_keys) / max(len(primary_map), len(secondary_map))
        else:
            correlation = 0.0
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            statistics={'correlation': correlation, 'common_entries': len(common_keys)},
            integrity_score=correlation
        )