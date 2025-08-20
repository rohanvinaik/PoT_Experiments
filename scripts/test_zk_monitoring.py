#!/usr/bin/env python3
"""
Test script for ZK monitoring and health check components.
Validates that all monitoring tools work correctly.
"""

import sys
import json
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_diagnostic_system():
    """Test ZK diagnostic system"""
    print("Testing ZK diagnostic system...")
    try:
        from pot.zk.diagnostic import ZKDiagnostic
        
        runner = ZKDiagnostic()
        
        # Test diagnostic
        print("  Running ZK system diagnostic...")
        results = runner.diagnose_zk_system()
        assert 'health_score' in results
        print(f"    ✓ Diagnostic completed (Score: {results['health_score']:.1f})")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Diagnostic system test failed: {e}")
        traceback.print_exc()
        return False


def test_version_info():
    """Test version information system"""
    print("Testing version information system...")
    try:
        from pot.zk.version_info import VersionManager, get_system_version
        
        manager = VersionManager()
        
        # Test basic version info
        print("  Getting system version...")
        system_version = get_system_version()
        assert system_version.system_version
        assert system_version.python_version
        print(f"    ✓ System version: {system_version.system_version}")
        
        # Test binary scanning
        print("  Scanning binaries...")
        binaries = manager.scan_all_binaries()
        print(f"    ✓ Found {len(binaries)} binaries")
        
        # Test git info
        git_commit = manager.get_git_commit_hash()
        if git_commit:
            print(f"    ✓ Git commit: {git_commit[:8]}...")
        else:
            print("    - No git commit info available")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Version info test failed: {e}")
        traceback.print_exc()
        return False


def test_metrics_collection():
    """Test ZK metrics collection"""
    print("Testing ZK metrics collection...")
    try:
        from pot.zk.metrics import get_zk_metrics_collector, ZKMetricsCollector
        
        # Test collector initialization
        print("  Getting metrics collector...")
        collector = get_zk_metrics_collector()
        assert isinstance(collector, ZKMetricsCollector)
        print("    ✓ Metrics collector initialized")
        
        # Test report generation
        print("  Generating metrics report...")
        report = collector.generate_report()
        assert isinstance(report, dict)
        print(f"    ✓ Metrics report generated ({len(report)} sections)")
        
        # Test metric recording (simulated)
        print("  Testing metric recording...")
        collector.record_proof_generation('sgd', 1000, 1024, True)
        collector.record_verification('sgd', 500, True)
        
        updated_report = collector.generate_report()
        assert len(updated_report.get('sgd_proofs', [])) >= 1
        print("    ✓ Metric recording works")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Metrics collection test failed: {e}")
        traceback.print_exc()
        return False


def test_healthcheck_system():
    """Test health check system (without starting server)"""
    print("Testing health check system...")
    try:
        from pot.zk.healthcheck import HealthMonitor
        
        # Test monitor initialization
        print("  Initializing health monitor...")
        monitor = HealthMonitor(check_interval=60)
        assert monitor.start_time > 0
        print("    ✓ Health monitor initialized")
        
        # Test quick status (without background monitoring)
        print("  Getting quick status...")
        monitor._perform_health_check()  # Force one check
        status = monitor.get_quick_status()
        assert 'status' in status
        assert 'timestamp' in status
        print(f"    ✓ Quick status: {status['status']}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Health check test failed: {e}")
        traceback.print_exc()
        return False


def test_monitoring_system():
    """Test monitoring and alerting system"""
    print("Testing monitoring and alerting system...")
    try:
        from pot.zk.monitoring import AlertManager, ZKSystemMonitor, MonitoringMetrics
        from datetime import datetime, timezone
        
        # Test alert manager
        print("  Testing alert manager...")
        alert_manager = AlertManager()
        assert len(alert_manager.rules) > 0
        print(f"    ✓ Alert manager loaded {len(alert_manager.rules)} rules")
        
        # Test alert evaluation with dummy metrics
        print("  Testing alert evaluation...")
        dummy_metrics = MonitoringMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            health_score=75.0,
            binary_availability={'prove_sgd_stdin': True, 'verify_sgd_stdin': True},
            proof_success_rates={'overall': 95.0},
            verification_success_rates={'overall': 98.0}, 
            average_proof_times={'sgd_proofs': 1500.0},
            system_resources={'cpu_percent': 45.0, 'memory_percent': 60.0, 'disk_percent': 30.0},
            alert_count=0,
            uptime_seconds=3600
        )
        
        alerts = alert_manager.evaluate_metrics(dummy_metrics)
        print(f"    ✓ Alert evaluation completed ({len(alerts)} alerts triggered)")
        
        # Test system monitor initialization
        print("  Testing system monitor...")
        monitor = ZKSystemMonitor(check_interval=300)
        status = monitor.get_monitoring_status()
        assert 'monitoring_active' in status
        print("    ✓ System monitor initialized")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Monitoring system test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all monitoring system tests"""
    print("ZK Monitoring System Test Suite")
    print("=" * 40)
    
    tests = [
        ("Diagnostic System", test_diagnostic_system),
        ("Version Information", test_version_info),
        ("Metrics Collection", test_metrics_collection),
        ("Health Check System", test_healthcheck_system),
        ("Monitoring & Alerts", test_monitoring_system)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            if test_func():
                print(f"✅ {test_name} - PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} - FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*40}")
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All monitoring system tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed - check output above")
        return 1


if __name__ == "__main__":
    exit(main())