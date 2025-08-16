#!/usr/bin/env python3
"""
Test suite for audit logger module
"""

import unittest
import tempfile
import shutil
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

from audit_logger import (
    AuditLogger,
    AuditEventType,
    AuditSeverity,
    ComplianceFramework,
    create_audit_logger,
    audit_verification,
    audit_attack_detection
)


class TestAuditLogger(unittest.TestCase):
    """Test audit logger functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.logger = AuditLogger(
            storage_path=self.test_dir,
            enable_encryption=False,  # Disable for testing
            enable_signing=True,
            retention_days=7
        )
    
    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_log_event(self):
        """Test basic event logging"""
        event_id = self.logger.log_event(
            event_type=AuditEventType.VERIFICATION_STARTED,
            actor="test_user",
            resource="test_model",
            action="Starting verification",
            details={"test": True}
        )
        
        self.assertIsNotNone(event_id)
        self.assertEqual(len(event_id), 16)
        
        # Check statistics
        stats = self.logger.get_statistics()
        self.assertEqual(stats["total_events"], 1)
        self.assertEqual(stats["events_by_type"]["verification_started"], 1)
    
    def test_audit_context(self):
        """Test audit context manager"""
        with self.logger.audit_context("test_operation", "test_user", "test_resource") as event_id:
            self.assertIsNotNone(event_id)
            # Simulate some work
            time.sleep(0.1)
        
        # Check that both start and complete events were logged
        stats = self.logger.get_statistics()
        self.assertEqual(stats["total_events"], 2)
        self.assertEqual(stats["events_by_type"]["verification_started"], 1)
        self.assertEqual(stats["events_by_type"]["verification_completed"], 1)
    
    def test_audit_context_with_error(self):
        """Test audit context with error"""
        try:
            with self.logger.audit_context("failing_operation", "test_user", "test_resource"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Check that start and failed events were logged
        stats = self.logger.get_statistics()
        self.assertEqual(stats["total_events"], 2)
        self.assertEqual(stats["events_by_type"]["verification_started"], 1)
        self.assertEqual(stats["events_by_type"]["verification_failed"], 1)
    
    def test_query_events(self):
        """Test querying events"""
        # Log multiple events
        for i in range(5):
            self.logger.log_event(
                event_type=AuditEventType.MODEL_LOADED,
                actor=f"user_{i}",
                resource=f"model_{i}",
                action="Loading model",
                severity=AuditSeverity.INFO if i % 2 == 0 else AuditSeverity.WARNING
            )
        
        # Query all events
        events = self.logger.query_events()
        self.assertEqual(len(events), 5)
        
        # Query by severity
        info_events = self.logger.query_events(severities=[AuditSeverity.INFO])
        self.assertEqual(len(info_events), 3)
        
        # Query by actor
        user_0_events = self.logger.query_events(actor="user_0")
        self.assertEqual(len(user_0_events), 1)
    
    def test_compliance_report_eu_ai_act(self):
        """Test EU AI Act compliance report generation"""
        # Log various events
        self.logger.log_event(
            event_type=AuditEventType.VERIFICATION_COMPLETED,
            actor="user1",
            resource="model1",
            action="Verification"
        )
        self.logger.log_event(
            event_type=AuditEventType.VERIFICATION_FAILED,
            actor="user2",
            resource="model2",
            action="Verification"
        )
        self.logger.log_event(
            event_type=AuditEventType.ATTACK_DETECTED,
            actor="system",
            resource="model3",
            action="Attack detection"
        )
        
        # Generate report
        report = self.logger.generate_compliance_report(
            ComplianceFramework.EU_AI_ACT,
            datetime.now() - timedelta(days=1),
            datetime.now()
        )
        
        self.assertEqual(report["framework"], "eu_ai_act")
        self.assertEqual(report["total_events"], 3)
        self.assertIn("transparency", report["report"])
        self.assertEqual(report["report"]["transparency"]["model_verifications"], 1)
        self.assertEqual(report["report"]["transparency"]["failed_verifications"], 1)
        self.assertEqual(report["report"]["transparency"]["attack_detections"], 1)
    
    def test_compliance_report_gdpr(self):
        """Test GDPR compliance report generation"""
        # Log GDPR-relevant events
        self.logger.log_event(
            event_type=AuditEventType.DATA_EXPORTED,
            actor="user1",
            resource="dataset1",
            action="Export data",
            details={"format": "csv"}
        )
        self.logger.log_event(
            event_type=AuditEventType.DATA_DELETED,
            actor="user2",
            resource="user_data",
            action="Delete user data",
            details={"reason": "user_request"}
        )
        
        # Generate report
        report = self.logger.generate_compliance_report(
            ComplianceFramework.GDPR,
            datetime.now() - timedelta(days=1),
            datetime.now()
        )
        
        self.assertEqual(report["framework"], "gdpr")
        self.assertIn("data_processing", report["report"])
        self.assertEqual(report["report"]["data_processing"]["data_exports"], 1)
        self.assertEqual(report["report"]["data_processing"]["data_deletions"], 1)
    
    def test_signature_verification(self):
        """Test event signature verification"""
        # Log event with signing enabled
        event_id = self.logger.log_event(
            event_type=AuditEventType.SECURITY_ALERT,
            actor="system",
            resource="critical_resource",
            action="Security alert",
            severity=AuditSeverity.CRITICAL
        )
        
        # Verify integrity
        self.assertTrue(self.logger.verify_integrity())
        
        # Query and check signature exists
        events = self.logger.query_events()
        self.assertEqual(len(events), 1)
        self.assertIsNotNone(events[0]["signature"])
        self.assertNotEqual(events[0]["signature"], "")
    
    def test_export_audit_trail(self):
        """Test exporting audit trail"""
        # Log some events
        for i in range(3):
            self.logger.log_event(
                event_type=AuditEventType.ACCESS_GRANTED,
                actor=f"user_{i}",
                resource=f"resource_{i}",
                action="Access granted"
            )
        
        # Export as JSON
        export_path = Path(self.test_dir) / "export.json"
        self.logger.export_audit_trail(str(export_path), format="json")
        
        self.assertTrue(export_path.exists())
        
        # Load and verify
        with open(export_path) as f:
            exported_data = json.load(f)
        
        # Should have 3 original events + 1 export event
        events = self.logger.query_events()
        self.assertEqual(len(events), 4)
        
        # Exported data should have the 3 original events
        self.assertEqual(len(exported_data), 3)
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        # Test audit_verification
        audit_verification(
            self.logger,
            model_id="test_model",
            user_id="test_user",
            result=True,
            details={"accuracy": 0.95}
        )
        
        # Test audit_attack_detection
        audit_attack_detection(
            self.logger,
            attack_type="wrapper",
            model_id="test_model",
            details={"confidence": 0.99}
        )
        
        # Check events were logged
        events = self.logger.query_events()
        self.assertEqual(len(events), 2)
        
        verification_events = [e for e in events if e["event_type"] == "verification_completed"]
        self.assertEqual(len(verification_events), 1)
        
        attack_events = [e for e in events if e["event_type"] == "attack_detected"]
        self.assertEqual(len(attack_events), 1)
    
    def test_rotation(self):
        """Test log rotation"""
        # Set small rotation size for testing
        self.logger.rotation_size_mb = 0.0001  # Very small to trigger rotation
        
        # Log many events to trigger rotation
        for i in range(100):
            self.logger.log_event(
                event_type=AuditEventType.MODEL_LOADED,
                actor=f"user_{i}",
                resource=f"model_{i}",
                action="Loading model with lots of details to increase size",
                details={
                    "large_data": "x" * 1000,
                    "index": i
                }
            )
        
        # Check that rotation occurred (compressed file should exist)
        compressed_files = list(Path(self.test_dir).glob("audit_*.jsonl.gz"))
        self.assertGreater(len(compressed_files), 0)
    
    def test_statistics(self):
        """Test statistics tracking"""
        # Log events of different types and severities
        self.logger.log_event(
            event_type=AuditEventType.VERIFICATION_STARTED,
            actor="user1",
            resource="model1",
            action="Start",
            severity=AuditSeverity.INFO
        )
        self.logger.log_event(
            event_type=AuditEventType.ATTACK_DETECTED,
            actor="system",
            resource="model2",
            action="Attack",
            severity=AuditSeverity.WARNING
        )
        self.logger.log_event(
            event_type=AuditEventType.SYSTEM_ERROR,
            actor="system",
            resource="system",
            action="Error",
            severity=AuditSeverity.ERROR
        )
        
        stats = self.logger.get_statistics()
        
        self.assertEqual(stats["total_events"], 3)
        self.assertEqual(stats["events_by_type"]["verification_started"], 1)
        self.assertEqual(stats["events_by_type"]["attack_detected"], 1)
        self.assertEqual(stats["events_by_type"]["system_error"], 1)
        self.assertEqual(stats["events_by_severity"]["info"], 1)
        self.assertEqual(stats["events_by_severity"]["warning"], 1)
        self.assertEqual(stats["events_by_severity"]["error"], 1)
        self.assertTrue(stats["integrity_verified"])


if __name__ == "__main__":
    unittest.main()