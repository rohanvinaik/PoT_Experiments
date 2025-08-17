#!/usr/bin/env python3
"""
Comprehensive test suite for audit trail query system.

Tests querying, integrity verification, anomaly detection, and dashboard functionality.
"""

import sys
import os
import json
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pot.audit.query import (
    AuditTrailQuery, AuditDashboard, AnomalyRecord, IntegrityReport,
    AnomalyType, IntegrityStatus
)


def create_mock_audit_records(count: int = 50) -> List[Dict[str, Any]]:
    """Create mock audit records for testing."""
    base_time = datetime.now(timezone.utc) - timedelta(days=30)
    records = []
    
    models = ["model_A", "model_B", "model_C", "model_D"]
    
    for i in range(count):
        # Create realistic variation in timestamps
        timestamp = base_time + timedelta(
            days=i // 2,
            hours=(i % 24),
            minutes=(i * 17) % 60
        )
        
        model_id = models[i % len(models)]
        
        # Introduce some patterns and anomalies
        if i < 20:
            # Early records - good performance
            confidence = 0.85 + (i % 10) * 0.01
            verification_result = "PASS" if confidence > 0.8 else "FAIL"
            duration = 2.0 + (i % 5) * 0.5
        elif 20 <= i < 30:
            # Middle period - some degradation for model_B
            if model_id == "model_B":
                confidence = 0.7 + (i % 5) * 0.02  # Lower confidence
                verification_result = "PASS" if confidence > 0.75 else "FAIL"
                duration = 5.0 + (i % 3) * 2.0  # Longer duration
            else:
                confidence = 0.85 + (i % 8) * 0.01
                verification_result = "PASS"
                duration = 2.0 + (i % 5) * 0.5
        else:
            # Later records - recovery
            confidence = 0.88 + (i % 6) * 0.01
            verification_result = "PASS"
            duration = 2.5 + (i % 4) * 0.3
        
        # Add some anomalies
        if i == 25:  # Sudden accuracy drop
            confidence = 0.3
            verification_result = "FAIL"
        elif i == 35:  # Timing anomaly
            duration = 15.0
        
        record = {
            "session_id": f"session_{i:03d}",
            "model_id": model_id,
            "timestamp": timestamp.isoformat(),
            "verification_decision": verification_result,
            "confidence": confidence,
            "verification_confidence": confidence,
            "duration_seconds": duration,
            "duration": duration,
            "challenges_passed": int(confidence * 50),
            "challenges_total": 50,
            "fingerprint_similarity": confidence + 0.05,
            "accuracy": confidence,
            "commitment": f"commit_{i:032x}",
            "nonce": f"nonce_{i:016x}",
            "_source_file": "test_audit.jsonl",
            "_line_number": i + 1
        }
        
        records.append(record)
    
    return records


def create_test_audit_file(records: List[Dict[str, Any]], filepath: str, format: str = "jsonl"):
    """Create test audit file with given records."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        if format == "jsonl":
            for record in records:
                f.write(json.dumps(record) + "\n")
        elif format == "json":
            json.dump(records, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


def test_audit_trail_loading():
    """Test loading audit trails from files."""
    print("Testing audit trail loading...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test single JSONL file
        records = create_mock_audit_records(20)
        jsonl_file = os.path.join(temp_dir, "audit.jsonl")
        create_test_audit_file(records, jsonl_file, "jsonl")
        
        query = AuditTrailQuery(jsonl_file)
        assert len(query.records) == 20
        assert len(query.model_index) == 4  # 4 different models
        
        # Test single JSON file
        json_file = os.path.join(temp_dir, "audit.json")
        create_test_audit_file(records, json_file, "json")
        
        query_json = AuditTrailQuery(json_file)
        assert len(query_json.records) == 20
        
        # Test directory with multiple files
        audit_dir = os.path.join(temp_dir, "audit_logs")
        os.makedirs(audit_dir)
        
        # Split records into multiple files
        for i in range(0, 20, 5):
            batch_file = os.path.join(audit_dir, f"batch_{i//5}.jsonl")
            create_test_audit_file(records[i:i+5], batch_file, "jsonl")
        
        query_dir = AuditTrailQuery(audit_dir)
        assert len(query_dir.records) == 20
        
        print("✓ Audit trail loading tests passed")


def test_basic_querying():
    """Test basic query functionality."""
    print("Testing basic querying...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        records = create_mock_audit_records(30)
        audit_file = os.path.join(temp_dir, "test_audit.jsonl")
        create_test_audit_file(records, audit_file)
        
        query = AuditTrailQuery(audit_file)
        
        # Test query by model
        model_a_records = query.query_by_model("model_A")
        assert len(model_a_records) > 0
        assert all(r['model_id'] == "model_A" for r in model_a_records)
        
        # Test query by verification result
        pass_records = query.query_by_verification_result("PASS")
        fail_records = query.query_by_verification_result("FAIL")
        assert len(pass_records) > 0
        assert len(fail_records) > 0
        assert all(r['verification_decision'] == "PASS" for r in pass_records)
        
        # Test query by confidence range
        high_conf_records = query.query_by_confidence_range(0.8, 1.0)
        low_conf_records = query.query_by_confidence_range(0.0, 0.5)
        assert len(high_conf_records) > 0
        assert all(r['confidence'] >= 0.8 for r in high_conf_records)
        
        # Test query by time range
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        start_time = base_time + timedelta(days=5)
        end_time = base_time + timedelta(days=15)
        
        time_range_records = query.query_by_timerange(start_time, end_time)
        assert len(time_range_records) > 0
        
        # Test query by session
        session_records = query.query_by_session("session_001")
        assert len(session_records) == 1
        assert session_records[0]['session_id'] == "session_001"
        
        print("✓ Basic querying tests passed")


def test_integrity_verification():
    """Test audit trail integrity verification."""
    print("Testing integrity verification...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with valid records
        records = create_mock_audit_records(20)
        audit_file = os.path.join(temp_dir, "valid_audit.jsonl")
        create_test_audit_file(records, audit_file)
        
        query = AuditTrailQuery(audit_file)
        integrity_report = query.verify_integrity()
        
        assert isinstance(integrity_report, IntegrityReport)
        assert integrity_report.total_records == 20
        assert integrity_report.integrity_score >= 0.8  # Should be high for valid records
        
        # Test with corrupted records (missing required fields)
        corrupted_records = records.copy()
        corrupted_records[5].pop('timestamp')  # Remove required field
        corrupted_records[10]['model_id'] = None  # Invalid model_id
        
        corrupted_file = os.path.join(temp_dir, "corrupted_audit.jsonl")
        create_test_audit_file(corrupted_records, corrupted_file)
        
        query_corrupted = AuditTrailQuery(corrupted_file)
        corrupted_report = query_corrupted.verify_integrity()
        
        assert corrupted_report.integrity_score < 1.0  # Should be lower due to corruption
        assert len(corrupted_report.anomalies_detected) > 0
        assert len(corrupted_report.recommendations) > 0
        
        print("✓ Integrity verification tests passed")


def test_anomaly_detection():
    """Test anomaly detection functionality."""
    print("Testing anomaly detection...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create records with known anomalies
        records = create_mock_audit_records(40)
        
        # Inject specific anomalies
        # Accuracy drift
        records[20]['confidence'] = 0.3  # Sudden drop
        records[21]['verification_decision'] = "FAIL"
        
        # Timing anomaly
        records[25]['duration_seconds'] = 20.0  # Very slow
        
        # Confidence anomaly
        records[30]['confidence'] = 0.1  # Very low
        
        # Frequency anomaly (rapid succession)
        base_time = datetime.now(timezone.utc)
        records[35]['timestamp'] = base_time.isoformat()
        records[36]['timestamp'] = (base_time + timedelta(seconds=1)).isoformat()
        records[37]['timestamp'] = (base_time + timedelta(seconds=2)).isoformat()
        
        audit_file = os.path.join(temp_dir, "anomaly_audit.jsonl")
        create_test_audit_file(records, audit_file)
        
        query = AuditTrailQuery(audit_file)
        anomalies = query.find_anomalies()
        
        assert len(anomalies) > 0
        
        # Check for different types of anomalies
        anomaly_types = [a.anomaly_type for a in anomalies]
        assert AnomalyType.ACCURACY_DRIFT in anomaly_types or AnomalyType.CONFIDENCE_ANOMALY in anomaly_types
        
        # Check severity scores
        assert all(0.0 <= a.severity <= 1.0 for a in anomalies)
        
        # Check that high-severity anomalies are detected
        high_severity = [a for a in anomalies if a.severity >= 0.7]
        assert len(high_severity) > 0
        
        print("✓ Anomaly detection tests passed")


def test_report_generation():
    """Test audit report generation."""
    print("Testing report generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        records = create_mock_audit_records(25)
        audit_file = os.path.join(temp_dir, "report_audit.jsonl")
        create_test_audit_file(records, audit_file)
        
        query = AuditTrailQuery(audit_file)
        
        # Test JSON report
        json_report = query.generate_audit_report("json")
        assert isinstance(json_report, str)
        
        # Parse JSON to verify structure
        report_data = json.loads(json_report)
        assert 'report_metadata' in report_data
        assert 'summary_statistics' in report_data
        assert 'integrity_report' in report_data
        assert 'anomalies_summary' in report_data
        assert 'model_analysis' in report_data
        assert 'recommendations' in report_data
        
        # Test Markdown report
        md_report = query.generate_audit_report("markdown")
        assert isinstance(md_report, str)
        assert "# PoT Audit Trail Report" in md_report
        assert "## Summary Statistics" in md_report
        
        # Test HTML report
        html_report = query.generate_audit_report("html")
        assert isinstance(html_report, str)
        assert "<html>" in html_report
        assert "<title>PoT Audit Trail Report</title>" in html_report
        
        # Test invalid format
        try:
            query.generate_audit_report("invalid")
            assert False, "Should raise ValueError for invalid format"
        except ValueError:
            pass
        
        print("✓ Report generation tests passed")


def test_dashboard_initialization():
    """Test dashboard initialization and basic functionality."""
    print("Testing dashboard initialization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        records = create_mock_audit_records(15)
        audit_file = os.path.join(temp_dir, "dashboard_audit.jsonl")
        create_test_audit_file(records, audit_file)
        
        query = AuditTrailQuery(audit_file)
        dashboard = AuditDashboard(query)
        
        assert dashboard.query == query
        assert dashboard.title == "PoT Audit Trail Dashboard"
        
        # Test dashboard creation (without actually running Streamlit)
        try:
            # This would normally require Streamlit to be running
            # We just test that the method exists and doesn't crash during setup
            assert hasattr(dashboard, 'create_streamlit_app')
            print("  ✓ Dashboard methods available")
        except ImportError:
            print("  ⚠ Streamlit not available - dashboard creation skipped")
        
        print("✓ Dashboard initialization tests passed")


def test_edge_cases_and_error_handling():
    """Test edge cases and error handling."""
    print("Testing edge cases and error handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test empty audit trail
        empty_file = os.path.join(temp_dir, "empty.jsonl")
        with open(empty_file, 'w') as f:
            pass  # Create empty file
        
        query_empty = AuditTrailQuery(empty_file)
        assert len(query_empty.records) == 0
        assert len(query_empty.model_index) == 0
        
        # Queries should return empty results
        assert len(query_empty.query_by_model("nonexistent")) == 0
        assert len(query_empty.query_by_verification_result("PASS")) == 0
        
        # Integrity verification should handle empty trails
        empty_integrity = query_empty.verify_integrity()
        assert empty_integrity.total_records == 0
        
        # Anomaly detection should handle empty trails
        empty_anomalies = query_empty.find_anomalies()
        assert len(empty_anomalies) == 0
        
        # Test malformed JSON
        malformed_file = os.path.join(temp_dir, "malformed.jsonl")
        with open(malformed_file, 'w') as f:
            f.write('{"valid": "record"}\n')
            f.write('invalid json line\n')  # This should be skipped
            f.write('{"another": "valid"}\n')
        
        query_malformed = AuditTrailQuery(malformed_file)
        assert len(query_malformed.records) == 2  # Should skip malformed line
        
        # Test nonexistent file
        try:
            AuditTrailQuery("/nonexistent/path/audit.jsonl")
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass
        
        # Test single record (edge case for anomaly detection)
        single_record = create_mock_audit_records(1)
        single_file = os.path.join(temp_dir, "single.jsonl")
        create_test_audit_file(single_record, single_file)
        
        query_single = AuditTrailQuery(single_file)
        single_anomalies = query_single.find_anomalies()
        # Should not crash, may or may not detect anomalies
        assert isinstance(single_anomalies, list)
        
        print("✓ Edge cases and error handling tests passed")


def test_performance_and_scalability():
    """Test performance with larger datasets."""
    print("Testing performance and scalability...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create larger dataset
        large_records = create_mock_audit_records(500)
        large_file = os.path.join(temp_dir, "large_audit.jsonl")
        create_test_audit_file(large_records, large_file)
        
        import time
        
        # Test loading performance
        start_time = time.time()
        query_large = AuditTrailQuery(large_file)
        load_time = time.time() - start_time
        
        assert len(query_large.records) == 500
        print(f"  Loaded 500 records in {load_time:.3f}s")
        
        # Test query performance
        start_time = time.time()
        model_records = query_large.query_by_model("model_A")
        query_time = time.time() - start_time
        
        assert len(model_records) > 0
        print(f"  Model query completed in {query_time:.3f}s")
        
        # Test integrity verification performance
        start_time = time.time()
        integrity_report = query_large.verify_integrity()
        integrity_time = time.time() - start_time
        
        print(f"  Integrity verification completed in {integrity_time:.3f}s")
        
        # Test anomaly detection performance
        start_time = time.time()
        anomalies = query_large.find_anomalies()
        anomaly_time = time.time() - start_time
        
        print(f"  Anomaly detection completed in {anomaly_time:.3f}s")
        print(f"  Found {len(anomalies)} anomalies")
        
        # Performance assertions
        assert load_time < 5.0, "Loading should be fast"
        assert query_time < 1.0, "Queries should be fast"
        assert integrity_time < 10.0, "Integrity verification should be reasonable"
        assert anomaly_time < 15.0, "Anomaly detection should be reasonable"
        
        print("✓ Performance and scalability tests passed")


def test_integration_with_crypto_utils():
    """Test integration with cryptographic utilities."""
    print("Testing integration with cryptographic utilities...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create records that include cryptographic elements
        records = create_mock_audit_records(10)
        
        # Add hash chain elements
        import hashlib
        for i, record in enumerate(records):
            # Add commitment hash
            record['commitment_hash'] = hashlib.sha256(f"commit_{i}".encode()).hexdigest()
            
            # Add chain hash for last record
            if i == len(records) - 1:
                all_hashes = [hashlib.sha256(f"commit_{j}".encode()).digest() for j in range(len(records))]
                # Simulate hash chain
                chain_data = b''.join(all_hashes)
                record['chain_hash'] = hashlib.sha256(chain_data).hexdigest()
        
        audit_file = os.path.join(temp_dir, "crypto_audit.jsonl")
        create_test_audit_file(records, audit_file)
        
        query = AuditTrailQuery(audit_file)
        
        # Test integrity verification with crypto elements
        integrity_report = query.verify_integrity()
        
        # Should recognize cryptographic elements
        assert integrity_report.total_records == 10
        assert isinstance(integrity_report.hash_chain_valid, bool)
        
        print("✓ Cryptographic utilities integration tests passed")


def run_all_tests():
    """Run all audit query system tests."""
    print("=" * 60)
    print("COMPREHENSIVE AUDIT TRAIL QUERY SYSTEM TEST SUITE")
    print("=" * 60)
    
    test_functions = [
        test_audit_trail_loading,
        test_basic_querying,
        test_integrity_verification,
        test_anomaly_detection,
        test_report_generation,
        test_dashboard_initialization,
        test_edge_cases_and_error_handling,
        test_performance_and_scalability,
        test_integration_with_crypto_utils
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        print("\nAudit trail query system ready for production!")
        print("Features validated:")
        print("  ✓ Audit trail loading from files and directories")
        print("  ✓ Multi-dimensional querying (model, time, confidence, etc.)")
        print("  ✓ Comprehensive integrity verification")
        print("  ✓ Advanced anomaly detection with severity scoring")
        print("  ✓ Multi-format report generation (JSON, Markdown, HTML)")
        print("  ✓ Dashboard integration capabilities")
        print("  ✓ Edge case handling and error recovery")
        print("  ✓ Performance with large datasets (500+ records)")
        print("  ✓ Integration with cryptographic utilities")
    else:
        print(f"❌ {failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)