#!/usr/bin/env python3
"""
Demonstration of audit logging capabilities for Proof-of-Training framework.
Shows how to generate, verify, and analyze regulatory-compliant audit logs.
"""

import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pot.core.audit_logger import (
    AuditLogger,
    AuditEventType,
    AuditSeverity,
    ComplianceFramework,
    create_audit_logger,
    audit_verification,
    audit_attack_detection
)

def demonstrate_basic_logging():
    """Demonstrate basic audit logging functionality"""
    print("=" * 60)
    print("BASIC AUDIT LOGGING DEMONSTRATION")
    print("=" * 60)
    
    # Initialize audit logger with compliance frameworks
    config = {
        'storage_path': './demo_audit_logs',
        'encryption_key': None,  # Use default for demo
        'signing_key': 'demo_signing_key',
        'retention_days': 2555,  # 7 years for FDA
        'enable_encryption': False,  # Disable for demo visibility
        'enable_signing': True,
        'compliance_frameworks': [
            ComplianceFramework.EU_AI_ACT,
            ComplianceFramework.NIST_RMF,
            ComplianceFramework.GDPR
        ]
    }
    
    audit_logger = create_audit_logger(config)
    
    # Log model registration/loading
    event1_id = audit_logger.log_event(
        event_type=AuditEventType.MODEL_LOADED,
        actor="admin-001",
        resource="model-gpt2-xl",
        action="Register and load GPT-2 XL model",
        outcome="success",
        severity=AuditSeverity.INFO,
        details={
            "model_type": "language_model",
            "parameters": 1500000000,
            "version": "1.0.0",
            "training_date": "2024-01-01",
            "verification_method": "fuzzy_hash"
        }
    )
    
    print(f"\n✓ Logged model registration: {event1_id}")
    
    # Log challenge generation
    event2_id = audit_logger.log_event(
        event_type=AuditEventType.CHALLENGE_GENERATED,
        actor="verifier-001",
        resource="model-gpt2-xl",
        action="Generate verification challenge",
        outcome="success",
        severity=AuditSeverity.INFO,
        details={
            "challenge_type": "text_comprehension",
            "challenge_id": "chal-001",
            "dimensions": 512,
            "complexity": "high",
            "kdf_iterations": 100000
        }
    )
    
    print(f"✓ Logged challenge generation: {event2_id}")
    
    # Log response computation
    event3_id = audit_logger.log_event(
        event_type=AuditEventType.RESPONSE_COMPUTED,
        actor="model-gpt2-xl",
        resource="chal-001",
        action="Compute response to challenge",
        outcome="success",
        severity=AuditSeverity.INFO,
        details={
            "response_hash": "sha256:abcd1234...",
            "computation_time_ms": 234.5,
            "confidence": 0.95,
            "gpu_used": True
        }
    )
    
    print(f"✓ Logged response: {event3_id}")
    
    # Simulate verification using helper function
    verification_passed = True  # Simulate success
    
    audit_verification(
        audit_logger,
        model_id="model-gpt2-xl",
        user_id="verifier-001",
        result=verification_passed,
        details={
            "challenge_id": "chal-001",
            "verification_method": "fuzzy_hash",
            "similarity_score": 0.94 if verification_passed else 0.45,
            "threshold": 0.85,
            "time_taken_ms": 156.3
        }
    )
    
    print(f"✓ Logged verification: {'PASSED' if verification_passed else 'FAILED'}")
    
    return audit_logger

def demonstrate_attack_detection(audit_logger):
    """Demonstrate attack detection and logging"""
    print("\n" + "=" * 60)
    print("ATTACK DETECTION SIMULATION")
    print("=" * 60)
    
    print("\nSimulating wrapper attack detection...")
    
    # Log wrapper attack detection
    audit_attack_detection(
        audit_logger,
        attack_type="wrapper",
        model_id="model-suspicious",
        details={
            "detection_method": "timing_analysis",
            "confidence": 0.92,
            "timing_deviation_ms": 45.3,
            "statistical_anomaly": True,
            "recommended_action": "block_verification"
        }
    )
    
    print("✓ Logged wrapper attack detection")
    
    # Simulate multiple failed attempts (potential attack)
    print("\nSimulating repeated verification failures...")
    
    for i in range(5):
        audit_logger.log_event(
            event_type=AuditEventType.VERIFICATION_FAILED,
            actor="suspicious-actor",
            resource="model-unknown",
            action=f"Verification attempt {i+1}",
            outcome="failure",
            severity=AuditSeverity.WARNING,
            details={
                "attempt": i + 1,
                "reason": "invalid_response",
                "similarity_score": 0.2 + i * 0.05,
                "ip_address": "192.168.1.100",
                "user_agent": "suspicious-client/1.0"
            }
        )
        time.sleep(0.1)
    
    print("✓ Logged 5 suspicious verification failures")
    
    # Log security alert
    audit_logger.log_event(
        event_type=AuditEventType.SECURITY_ALERT,
        actor="security-system",
        resource="suspicious-actor",
        action="Anomaly detected - repeated failures",
        outcome="detected",
        severity=AuditSeverity.CRITICAL,
        details={
            "anomaly_type": "repeated_failures",
            "failure_count": 5,
            "time_window_seconds": 60,
            "risk_assessment": "HIGH",
            "action_taken": "actor_blocked",
            "notification_sent": True
        }
    )
    
    print("✓ Security alert logged")

def demonstrate_querying(audit_logger):
    """Demonstrate log querying capabilities"""
    print("\n" + "=" * 60)
    print("LOG QUERYING DEMONSTRATION")
    print("=" * 60)
    
    # Query verification events
    print("\nQuerying verification events...")
    verification_events = audit_logger.query_events(
        event_types=[AuditEventType.VERIFICATION_COMPLETED, 
                    AuditEventType.VERIFICATION_FAILED],
        limit=10
    )
    
    print(f"Found {len(verification_events)} verification events")
    for event in verification_events[:3]:
        print(f"  - {event['event_id'][:8]}... : {event['action']} ({event['outcome']})")
    
    # Query by severity
    print("\nQuerying high-severity events...")
    critical_events = audit_logger.query_events(
        severities=[AuditSeverity.ERROR, AuditSeverity.CRITICAL],
        limit=10
    )
    
    print(f"Found {len(critical_events)} critical events")
    for event in critical_events[:3]:
        print(f"  - Severity {event['severity']}: {event['action']}")
    
    # Query by actor
    print("\nQuerying events by actor...")
    admin_events = audit_logger.query_events(
        actor="admin-001",
        limit=10
    )
    
    print(f"Found {len(admin_events)} admin actions")
    
    # Query by time range
    print("\nQuerying recent events (last hour)...")
    recent_events = audit_logger.query_events(
        start_time=datetime.now() - timedelta(hours=1),
        end_time=datetime.now()
    )
    
    print(f"Found {len(recent_events)} events in the last hour")

def demonstrate_compliance_reporting(audit_logger):
    """Demonstrate compliance report generation"""
    print("\n" + "=" * 60)
    print("COMPLIANCE REPORTING")
    print("=" * 60)
    
    # Generate EU AI Act report
    print("\nGenerating EU AI Act compliance report...")
    
    eu_report = audit_logger.generate_compliance_report(
        framework=ComplianceFramework.EU_AI_ACT,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    print(f"Report period: {eu_report['period']['start']} to {eu_report['period']['end']}")
    print(f"Total events: {eu_report['total_events']}")
    
    if 'transparency' in eu_report['report']:
        trans = eu_report['report']['transparency']
        print(f"\nTransparency metrics:")
        print(f"  - Model verifications: {trans.get('model_verifications', 0)}")
        print(f"  - Failed verifications: {trans.get('failed_verifications', 0)}")
        print(f"  - Attack detections: {trans.get('attack_detections', 0)}")
    
    # Save report
    with open("eu_ai_act_report.json", "w") as f:
        json.dump(eu_report, f, indent=2, default=str)
    print("\n✓ EU AI Act report saved to eu_ai_act_report.json")
    
    # Generate GDPR report
    print("\nGenerating GDPR compliance report...")
    
    gdpr_report = audit_logger.generate_compliance_report(
        framework=ComplianceFramework.GDPR,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    if 'data_processing' in gdpr_report['report']:
        dp = gdpr_report['report']['data_processing']
        print(f"\nData processing activities:")
        print(f"  - Total activities: {dp.get('processing_activities', 0)}")
        print(f"  - Data exports: {dp.get('data_exports', 0)}")
        print(f"  - Data deletions: {dp.get('data_deletions', 0)}")
    
    # Save report
    with open("gdpr_report.json", "w") as f:
        json.dump(gdpr_report, f, indent=2, default=str)
    print("\n✓ GDPR report saved to gdpr_report.json")
    
    # Generate NIST RMF report
    print("\nGenerating NIST RMF compliance report...")
    
    nist_report = audit_logger.generate_compliance_report(
        framework=ComplianceFramework.NIST_RMF,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    if 'monitor' in nist_report['report']:
        mon = nist_report['report']['monitor']
        print(f"\nMonitoring metrics:")
        print(f"  - Continuous monitoring: {mon.get('continuous_monitoring', False)}")
        print(f"  - Events per day: {mon.get('events_per_day', 0):.1f}")
    
    # Save report
    with open("nist_rmf_report.json", "w") as f:
        json.dump(nist_report, f, indent=2, default=str)
    print("\n✓ NIST RMF report saved to nist_rmf_report.json")

def demonstrate_export(audit_logger):
    """Demonstrate audit log export"""
    print("\n" + "=" * 60)
    print("AUDIT LOG EXPORT")
    print("=" * 60)
    
    print("\nExporting audit trail...")
    
    # Export as JSON
    json_file = "audit_export.json"
    audit_logger.export_audit_trail(
        output_path=json_file,
        format="json",
        start_date=datetime.now() - timedelta(days=7)
    )
    print(f"✓ JSON export saved to {json_file}")
    
    # Export as CSV
    csv_file = "audit_export.csv"
    audit_logger.export_audit_trail(
        output_path=csv_file,
        format="csv",
        start_date=datetime.now() - timedelta(days=7)
    )
    print(f"✓ CSV export saved to {csv_file}")
    
    # Show export summary
    with open(json_file, 'r') as f:
        export_data = json.load(f)
    
    if isinstance(export_data, list):
        print(f"\nExported {len(export_data)} events")
    else:
        print(f"\nExport contains audit trail data")

def demonstrate_integrity_verification(audit_logger):
    """Demonstrate audit trail integrity verification"""
    print("\n" + "=" * 60)
    print("INTEGRITY VERIFICATION")
    print("=" * 60)
    
    print("\nVerifying audit trail integrity...")
    
    is_valid = audit_logger.verify_integrity()
    
    if is_valid:
        print("✓ Audit trail integrity verified - all signatures valid")
    else:
        print("✗ Integrity check failed - potential tampering detected!")
    
    # Get statistics
    stats = audit_logger.get_statistics()
    
    print(f"\nAudit trail statistics:")
    print(f"  - Total events: {stats['total_events']}")
    print(f"  - Database size: {stats.get('database_size_mb', 0):.2f} MB")
    print(f"  - Integrity verified: {stats.get('integrity_verified', False)}")
    
    if 'events_by_type' in stats:
        print(f"\nEvents by type:")
        for event_type, count in list(stats['events_by_type'].items())[:5]:
            print(f"  - {event_type}: {count}")
    
    if 'events_by_severity' in stats:
        print(f"\nEvents by severity:")
        for severity, count in stats['events_by_severity'].items():
            print(f"  - {severity}: {count}")

def demonstrate_audit_context(audit_logger):
    """Demonstrate audit context manager"""
    print("\n" + "=" * 60)
    print("AUDIT CONTEXT DEMONSTRATION")
    print("=" * 60)
    
    print("\nUsing audit context for operation tracking...")
    
    # Use context manager for automatic logging
    try:
        with audit_logger.audit_context("model_training", "trainer-001", "model-new"):
            print("  Simulating model training operation...")
            time.sleep(0.5)  # Simulate work
            
            # Simulate some training metrics
            audit_logger.log_event(
                event_type=AuditEventType.MODEL_LOADED,
                actor="trainer-001",
                resource="model-new",
                action="Training epoch completed",
                details={
                    "epoch": 1,
                    "loss": 0.234,
                    "accuracy": 0.89
                }
            )
            
            print("  Training operation completed")
    except Exception as e:
        print(f"  Operation failed: {e}")
    
    print("✓ Operation automatically logged with start/complete events")

def main():
    """Main demonstration flow"""
    print("\n" + "=" * 60)
    print("PROOF-OF-TRAINING AUDIT LOGGING DEMONSTRATION")
    print("=" * 60)
    print("\nThis demo shows regulatory-compliant audit logging")
    print("for the Proof-of-Training verification framework.\n")
    
    try:
        # Run demonstrations
        audit_logger = demonstrate_basic_logging()
        demonstrate_attack_detection(audit_logger)
        demonstrate_querying(audit_logger)
        demonstrate_audit_context(audit_logger)
        demonstrate_compliance_reporting(audit_logger)
        demonstrate_export(audit_logger)
        demonstrate_integrity_verification(audit_logger)
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        
        print("\nGenerated files:")
        print("  - demo_audit_logs/ (audit database and logs)")
        print("  - eu_ai_act_report.json")
        print("  - gdpr_report.json")
        print("  - nist_rmf_report.json")
        print("  - audit_export.json")
        print("  - audit_export.csv")
        
        print("\n✓ All regulatory compliance features demonstrated successfully!")
        
        # Show compliance summary
        print("\n" + "=" * 60)
        print("COMPLIANCE SUMMARY")
        print("=" * 60)
        
        frameworks = [
            ComplianceFramework.EU_AI_ACT,
            ComplianceFramework.GDPR,
            ComplianceFramework.NIST_RMF
        ]
        
        for framework in frameworks:
            report = audit_logger.generate_compliance_report(
                framework=framework,
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
            print(f"\n{framework.value}:")
            print(f"  Events logged: {report['total_events']}")
            print(f"  Report generated: {report['generated_at']}")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())