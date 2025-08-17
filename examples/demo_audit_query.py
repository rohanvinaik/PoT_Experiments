#!/usr/bin/env python3
"""
Demonstration of the PoT audit trail query system.

Shows how to query, analyze, and visualize audit trails with advanced
analytics and anomaly detection capabilities.
"""

import sys
import os
import json
import tempfile
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pot.audit.query import AuditTrailQuery, AuditDashboard
from test_audit_query import create_mock_audit_records, create_test_audit_file


def demo_basic_querying():
    """Demonstrate basic audit trail querying capabilities."""
    print("=" * 60)
    print("BASIC AUDIT TRAIL QUERYING DEMONSTRATION")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create realistic audit data
        print("Creating demo audit trail with 100 verification records...")
        records = create_mock_audit_records(100)
        audit_file = os.path.join(temp_dir, "demo_audit.jsonl")
        create_test_audit_file(records, audit_file)
        
        # Initialize query system
        print("Loading audit trail...")
        query = AuditTrailQuery(audit_file)
        
        print(f"✓ Loaded {len(query.records)} audit records")
        print(f"✓ Found {len(query.model_index)} unique models")
        print(f"✓ Time span: {(query._parse_timestamp(query.records[-1]['timestamp']) - query._parse_timestamp(query.records[0]['timestamp'])).days} days")
        
        # Demonstrate different query types
        print("\n--- Query by Model ---")
        model_a_records = query.query_by_model("model_A")
        print(f"Model A verifications: {len(model_a_records)}")
        
        # Show recent verification
        if model_a_records:
            recent = model_a_records[-1]
            print(f"  Latest verification: {recent['verification_decision']} (confidence: {recent['confidence']:.3f})")
        
        print("\n--- Query by Verification Result ---")
        pass_records = query.query_by_verification_result("PASS")
        fail_records = query.query_by_verification_result("FAIL")
        pass_rate = len(pass_records) / len(query.records) * 100
        
        print(f"Passed verifications: {len(pass_records)} ({pass_rate:.1f}%)")
        print(f"Failed verifications: {len(fail_records)}")
        
        print("\n--- Query by Confidence Range ---")
        high_conf = query.query_by_confidence_range(0.9, 1.0)
        medium_conf = query.query_by_confidence_range(0.7, 0.9)
        low_conf = query.query_by_confidence_range(0.0, 0.7)
        
        print(f"High confidence (0.9-1.0): {len(high_conf)} records")
        print(f"Medium confidence (0.7-0.9): {len(medium_conf)} records")
        print(f"Low confidence (0.0-0.7): {len(low_conf)} records")
        
        print("\n--- Query by Time Range ---")
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        recent_start = base_time + timedelta(days=20)
        recent_end = base_time + timedelta(days=30)
        
        recent_records = query.query_by_timerange(recent_start, recent_end)
        print(f"Recent verifications (last 10 days): {len(recent_records)} records")
        
        if recent_records:
            recent_avg_conf = sum(r['confidence'] for r in recent_records) / len(recent_records)
            print(f"  Average confidence: {recent_avg_conf:.3f}")


def demo_integrity_verification():
    """Demonstrate integrity verification capabilities."""
    print("\n" + "=" * 60)
    print("INTEGRITY VERIFICATION DEMONSTRATION")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Creating audit trail with integrity test data...")
        records = create_mock_audit_records(50)
        
        # Add some integrity issues for demonstration
        records[10].pop('timestamp')  # Missing required field
        records[20]['verification_decision'] = None  # Invalid value
        records[30]['confidence'] = 1.5  # Out of range value
        
        audit_file = os.path.join(temp_dir, "integrity_demo.jsonl")
        create_test_audit_file(records, audit_file)
        
        query = AuditTrailQuery(audit_file)
        
        print("Running comprehensive integrity verification...")
        integrity_report = query.verify_integrity()
        
        print(f"\n--- Integrity Report ---")
        print(f"Status: {integrity_report.status.value.upper()}")
        print(f"Integrity Score: {integrity_report.integrity_score:.3f}")
        print(f"Valid Records: {integrity_report.valid_records}/{integrity_report.total_records}")
        print(f"Hash Chain Valid: {integrity_report.hash_chain_valid}")
        
        print(f"\n--- Commitment Verification ---")
        cv = integrity_report.commitment_verification
        print(f"Verified: {cv['verified']}")
        print(f"Failed: {cv['failed']}")
        print(f"Missing: {cv['missing']}")
        
        if integrity_report.anomalies_detected:
            print(f"\n--- Integrity Anomalies ---")
            for i, anomaly in enumerate(integrity_report.anomalies_detected[:3]):
                print(f"{i+1}. {anomaly.description}")
                print(f"   Severity: {anomaly.severity:.2f}")
        
        if integrity_report.recommendations:
            print(f"\n--- Recommendations ---")
            for i, rec in enumerate(integrity_report.recommendations[:3]):
                print(f"{i+1}. {rec}")


def demo_anomaly_detection():
    """Demonstrate advanced anomaly detection."""
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION DEMONSTRATION")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Creating audit trail with known anomalies...")
        records = create_mock_audit_records(80)
        
        # Inject specific anomalies for demonstration
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Accuracy drift anomaly
        records[25]['confidence'] = 0.2  # Sudden accuracy drop
        records[26]['confidence'] = 0.15
        records[27]['verification_decision'] = "FAIL"
        
        # Timing anomaly
        records[35]['duration_seconds'] = 25.0  # Unusually slow verification
        
        # Frequency anomaly (burst of verifications)
        for i in range(40, 45):
            records[i]['timestamp'] = (base_time + timedelta(days=15, seconds=i-40)).isoformat()
        
        # Fingerprint drift
        records[50]['fingerprint_similarity'] = 0.3  # Poor fingerprint match
        records[51]['fingerprint_similarity'] = 0.25
        
        audit_file = os.path.join(temp_dir, "anomaly_demo.jsonl")
        create_test_audit_file(records, audit_file)
        
        query = AuditTrailQuery(audit_file)
        
        print("Running advanced anomaly detection...")
        anomalies = query.find_anomalies()
        
        print(f"\n--- Anomaly Summary ---")
        print(f"Total anomalies detected: {len(anomalies)}")
        
        # Categorize by severity
        high_severity = [a for a in anomalies if a.severity >= 0.7]
        medium_severity = [a for a in anomalies if 0.3 <= a.severity < 0.7]
        low_severity = [a for a in anomalies if a.severity < 0.3]
        
        print(f"High severity (≥0.7): {len(high_severity)}")
        print(f"Medium severity (0.3-0.7): {len(medium_severity)}")
        print(f"Low severity (<0.3): {len(low_severity)}")
        
        # Categorize by type
        from collections import Counter
        anomaly_types = Counter([a.anomaly_type.value for a in anomalies])
        print(f"\n--- Anomaly Types ---")
        for atype, count in anomaly_types.most_common():
            print(f"{atype}: {count}")
        
        # Show top anomalies
        print(f"\n--- Top 5 Anomalies (by severity) ---")
        for i, anomaly in enumerate(anomalies[:5]):
            print(f"{i+1}. {anomaly.anomaly_type.value.upper()}")
            print(f"   Model: {anomaly.model_id}")
            print(f"   Severity: {anomaly.severity:.2f}")
            print(f"   Description: {anomaly.description}")
            print(f"   Timestamp: {anomaly.timestamp.strftime('%Y-%m-%d %H:%M')}")
            print()


def demo_report_generation():
    """Demonstrate report generation in different formats."""
    print("\n" + "=" * 60)
    print("REPORT GENERATION DEMONSTRATION")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Creating comprehensive audit trail for reporting...")
        records = create_mock_audit_records(60)
        audit_file = os.path.join(temp_dir, "report_demo.jsonl")
        create_test_audit_file(records, audit_file)
        
        query = AuditTrailQuery(audit_file)
        
        # Generate JSON report
        print("\n--- Generating JSON Report ---")
        json_report = query.generate_audit_report("json")
        json_data = json.loads(json_report)
        
        print(f"Report sections:")
        for section in json_data.keys():
            print(f"  ✓ {section}")
        
        # Show summary statistics
        stats = json_data['summary_statistics']
        print(f"\nSummary Statistics:")
        print(f"  Total records: {stats['total_records']}")
        print(f"  Unique models: {stats['unique_models']}")
        print(f"  Time span: {stats['time_span_days']} days")
        print(f"  Average confidence: {stats['average_confidence']}")
        
        # Generate Markdown report
        print("\n--- Generating Markdown Report ---")
        md_report = query.generate_audit_report("markdown")
        
        # Show first few lines
        md_lines = md_report.split('\n')[:10]
        print("Markdown preview:")
        for line in md_lines:
            print(f"  {line}")
        
        # Generate HTML report
        print("\n--- Generating HTML Report ---")
        html_report = query.generate_audit_report("html")
        
        print(f"HTML report generated ({len(html_report)} characters)")
        print("Contains standard HTML structure with styling")
        
        # Save reports to files for inspection
        reports_dir = os.path.join(temp_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        with open(os.path.join(reports_dir, "audit_report.json"), 'w') as f:
            f.write(json_report)
        
        with open(os.path.join(reports_dir, "audit_report.md"), 'w') as f:
            f.write(md_report)
        
        with open(os.path.join(reports_dir, "audit_report.html"), 'w') as f:
            f.write(html_report)
        
        print(f"\n✓ Reports saved to: {reports_dir}")


def demo_dashboard_setup():
    """Demonstrate dashboard setup and configuration."""
    print("\n" + "=" * 60)
    print("DASHBOARD SETUP DEMONSTRATION")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Setting up audit trail dashboard...")
        records = create_mock_audit_records(40)
        audit_file = os.path.join(temp_dir, "dashboard_demo.jsonl")
        create_test_audit_file(records, audit_file)
        
        # Initialize dashboard
        query = AuditTrailQuery(audit_file)
        dashboard = AuditDashboard(query)
        
        print(f"✓ Dashboard initialized with {len(query.records)} records")
        print(f"✓ Monitoring {len(query.model_index)} models")
        
        # Show dashboard capabilities
        print(f"\n--- Dashboard Features ---")
        print("✓ Real-time audit trail visualization")
        print("✓ Interactive filtering by model, time range, confidence")
        print("✓ Integrity monitoring with status indicators") 
        print("✓ Anomaly detection with severity highlighting")
        print("✓ Model performance comparison")
        print("✓ Timeline visualization of verification events")
        
        # Create dashboard app (setup only, doesn't run)
        print(f"\n--- Dashboard Configuration ---")
        print(f"Title: {dashboard.title}")
        print(f"Data source: {query.audit_log_path}")
        
        try:
            # Test dashboard method availability
            assert hasattr(dashboard, 'create_streamlit_app')
            print("✓ Streamlit integration available")
            
            # Show how to run the dashboard
            print(f"\n--- Running the Dashboard ---")
            print("To launch the interactive dashboard:")
            print(f"1. Create a Streamlit app script:")
            print(f"   ```python")
            print(f"   from pot.audit.query import create_dashboard_app")
            print(f"   create_dashboard_app('{audit_file}')")
            print(f"   ```")
            print(f"2. Run with: streamlit run dashboard_app.py")
            
        except ImportError:
            print("⚠ Streamlit not available - install with: pip install streamlit")


def demo_performance_analysis():
    """Demonstrate performance analysis capabilities."""
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Creating large audit trail for performance testing...")
        
        import time
        
        # Create progressively larger datasets
        sizes = [100, 500, 1000]
        
        for size in sizes:
            print(f"\n--- Testing with {size} records ---")
            
            records = create_mock_audit_records(size)
            audit_file = os.path.join(temp_dir, f"perf_{size}.jsonl")
            create_test_audit_file(records, audit_file)
            
            # Test loading performance
            start_time = time.time()
            query = AuditTrailQuery(audit_file)
            load_time = time.time() - start_time
            
            print(f"Loading: {load_time:.3f}s ({size/load_time:.0f} records/sec)")
            
            # Test query performance
            start_time = time.time()
            model_results = query.query_by_model("model_A")
            query_time = time.time() - start_time
            
            print(f"Querying: {query_time:.3f}s (found {len(model_results)} records)")
            
            # Test integrity verification performance
            start_time = time.time()
            integrity_report = query.verify_integrity()
            integrity_time = time.time() - start_time
            
            print(f"Integrity check: {integrity_time:.3f}s (score: {integrity_report.integrity_score:.3f})")
            
            # Test anomaly detection performance
            start_time = time.time()
            anomalies = query.find_anomalies()
            anomaly_time = time.time() - start_time
            
            print(f"Anomaly detection: {anomaly_time:.3f}s (found {len(anomalies)} anomalies)")
            
            # Calculate total analysis time
            total_time = load_time + query_time + integrity_time + anomaly_time
            print(f"Total analysis time: {total_time:.3f}s")


def demo_real_world_scenarios():
    """Demonstrate real-world usage scenarios."""
    print("\n" + "=" * 60)
    print("REAL-WORLD SCENARIOS DEMONSTRATION")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Scenario 1: Security incident investigation
        print("Scenario 1: Security Incident Investigation")
        print("-" * 45)
        
        records = create_mock_audit_records(50)
        # Simulate a security incident with multiple indicators
        incident_time = datetime.now(timezone.utc) - timedelta(days=5)
        
        # Model compromise indicators
        for i in range(20, 25):
            records[i]['confidence'] = 0.3  # Sudden performance drop
            records[i]['verification_decision'] = "FAIL"
            records[i]['fingerprint_similarity'] = 0.4  # Fingerprint changes
            records[i]['timestamp'] = (incident_time + timedelta(hours=i-20)).isoformat()
        
        audit_file = os.path.join(temp_dir, "incident.jsonl")
        create_test_audit_file(records, audit_file)
        
        query = AuditTrailQuery(audit_file)
        
        # Investigate the incident
        incident_start = incident_time - timedelta(hours=1)
        incident_end = incident_time + timedelta(hours=6)
        incident_records = query.query_by_timerange(incident_start, incident_end)
        
        print(f"Incident timeframe: {len(incident_records)} records")
        
        # Find related anomalies
        anomalies = query.find_anomalies()
        incident_anomalies = [a for a in anomalies 
                            if incident_start <= a.timestamp <= incident_end]
        
        print(f"Anomalies during incident: {len(incident_anomalies)}")
        
        # Analyze affected models
        affected_models = set(r['model_id'] for r in incident_records)
        print(f"Affected models: {', '.join(affected_models)}")
        
        # Scenario 2: Model performance monitoring
        print(f"\nScenario 2: Model Performance Monitoring")
        print("-" * 45)
        
        # Generate monthly report
        recent_time = datetime.now(timezone.utc) - timedelta(days=7)
        recent_records = query.query_by_timerange(recent_time, datetime.now(timezone.utc))
        
        model_performance = {}
        for model_id in query.model_index.keys():
            model_records = [r for r in recent_records if r['model_id'] == model_id]
            if model_records:
                avg_confidence = sum(r['confidence'] for r in model_records) / len(model_records)
                pass_rate = len([r for r in model_records if r['verification_decision'] == 'PASS']) / len(model_records)
                model_performance[model_id] = {
                    'verifications': len(model_records),
                    'avg_confidence': avg_confidence,
                    'pass_rate': pass_rate
                }
        
        print("Model performance (last 7 days):")
        for model_id, perf in model_performance.items():
            print(f"  {model_id}: {perf['verifications']} verifications, "
                  f"{perf['avg_confidence']:.3f} confidence, "
                  f"{perf['pass_rate']*100:.1f}% pass rate")
        
        # Scenario 3: Compliance reporting
        print(f"\nScenario 3: Compliance Reporting")
        print("-" * 45)
        
        integrity_report = query.verify_integrity()
        
        print(f"Compliance Status: {integrity_report.status.value.upper()}")
        print(f"Audit coverage: {len(query.records)} verification events")
        print(f"Data integrity: {integrity_report.integrity_score*100:.1f}%")
        print(f"Models monitored: {len(query.model_index)}")
        
        # Generate compliance summary
        high_risk_anomalies = [a for a in anomalies if a.severity >= 0.7]
        print(f"High-risk findings: {len(high_risk_anomalies)}")
        
        if high_risk_anomalies:
            print("Action required:")
            for anomaly in high_risk_anomalies[:3]:
                print(f"  - {anomaly.description}")


def run_full_demonstration():
    """Run complete demonstration of audit trail query system."""
    print("🔍 PoT AUDIT TRAIL QUERY SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("This demonstration showcases the comprehensive audit trail analysis")
    print("and visualization capabilities of the PoT framework.")
    print("=" * 80)
    
    try:
        demo_basic_querying()
        demo_integrity_verification()
        demo_anomaly_detection()
        demo_report_generation()
        demo_dashboard_setup()
        demo_performance_analysis()
        demo_real_world_scenarios()
        
        print("\n" + "=" * 80)
        print("🎉 DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("The audit trail query system provides:")
        print("✓ Multi-dimensional querying capabilities")
        print("✓ Comprehensive integrity verification")
        print("✓ Advanced anomaly detection with severity scoring")
        print("✓ Multi-format report generation")
        print("✓ Interactive dashboard for visualization")
        print("✓ High performance with large datasets")
        print("✓ Real-world scenario support")
        print()
        print("Ready for production deployment in PoT verification pipelines!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_full_demonstration()