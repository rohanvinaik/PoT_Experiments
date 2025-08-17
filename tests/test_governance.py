"""Integration tests for governance framework."""

import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
import yaml

from pot.core.governance import GovernanceFramework
from pot.governance.audit_logger import AuditLogger
from pot.governance.compliance_dashboard import ComplianceDashboard
from pot.governance.eu_ai_act_compliance import EUAIActCompliance
from pot.governance.nist_ai_rmf_compliance import NISTAIRMFCompliance
from pot.governance.policy_engine import PolicyEngine, PolicyRule
from pot.governance.risk_assessment import AIRiskAssessment


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def governance_config():
    """Basic governance configuration."""
    return {
        'organization': 'TestOrg',
        'contact': 'test@example.com',
        'policies': {
            'data_retention_days': 90,
            'model_retraining_frequency': 30,
            'access_control_level': 'strict'
        },
        'compliance': {
            'frameworks': ['EU_AI_Act', 'NIST_AI_RMF'],
            'risk_appetite': 'low'
        }
    }


@pytest.fixture
def sample_model():
    """Create sample model for testing."""
    return SimpleModel()


# ============================================================================
# Governance Framework Tests
# ============================================================================

def test_governance_framework_initialization(governance_config):
    """Test framework setup with various configs."""
    framework = GovernanceFramework(governance_config)
    
    assert framework.config == governance_config
    assert len(framework.policies) > 0
    assert len(framework.compliance_checks) == 0
    assert len(framework.decisions) == 0


def test_policy_enforcement(governance_config):
    """Verify policies are correctly enforced."""
    framework = GovernanceFramework(governance_config)
    
    # Test data retention policy
    old_data = {
        'timestamp': (datetime.now() - timedelta(days=100)).isoformat(),
        'data': 'old_data'
    }
    
    result = framework.check_compliance('data_retention', old_data)
    assert not result['compliant']
    assert 'exceeds retention' in result['details'].lower()
    
    # Test access control policy
    result = framework.check_compliance('access_control', {'level': 'public'})
    assert not result['compliant']


def test_compliance_checking(governance_config, sample_model):
    """Test EU AI Act and NIST RMF compliance."""
    framework = GovernanceFramework(governance_config)
    
    # Add model metadata
    model_info = {
        'model': sample_model,
        'purpose': 'classification',
        'risk_category': 'limited',
        'training_data': 'public_dataset'
    }
    
    # Check compliance
    result = framework.check_compliance('model_deployment', model_info)
    assert 'compliant' in result
    
    # Log decision
    decision = framework.log_decision(
        'deploy_model',
        'approved',
        {'model_id': 'test_model_001'}
    )
    assert decision in framework.decisions


def test_audit_logging(temp_dir):
    """Verify audit trail integrity."""
    log_file = temp_dir / 'audit.log'
    logger = AuditLogger(str(log_file))
    
    # Log events
    events = []
    for i in range(10):
        event = logger.log_event(
            'test_event',
            'INFO',
            {'index': i, 'data': f'test_{i}'}
        )
        events.append(event)
    
    # Verify integrity
    assert logger.verify_integrity()
    
    # Query logs
    results = logger.query_logs(event_type='test_event')
    assert len(results) == 10
    
    # Test anomaly detection
    anomalies = logger.detect_anomalies()
    assert isinstance(anomalies, list)


# ============================================================================
# Regulatory Compliance Tests
# ============================================================================

def test_eu_ai_act_risk_categorization():
    """Test EU AI Act risk categorization."""
    compliance = EUAIActCompliance()
    
    # Test different risk categories
    test_cases = [
        ('social_scoring', 'unacceptable'),
        ('biometric_identification', 'high'),
        ('chatbot', 'limited'),
        ('spam_filter', 'minimal')
    ]
    
    for use_case, expected_risk in test_cases:
        risk = compliance.categorize_risk(use_case)
        assert risk == expected_risk


def test_nist_rmf_function_implementation():
    """Test NIST RMF function implementation."""
    nist = NISTAIRMFCompliance()
    
    # Test GOVERN function
    govern_result = nist.assess_govern_function({
        'policies': ['data_governance', 'model_governance'],
        'accountability': True,
        'culture': 'risk-aware'
    })
    assert govern_result['implemented']
    assert govern_result['score'] > 0
    
    # Test MAP function
    map_result = nist.assess_map_function({
        'context': 'healthcare',
        'stakeholders': ['patients', 'doctors', 'regulators'],
        'risks_identified': True
    })
    assert map_result['implemented']
    
    # Test MEASURE function
    measure_result = nist.assess_measure_function({
        'metrics': ['accuracy', 'fairness', 'robustness'],
        'testing': True,
        'monitoring': True
    })
    assert measure_result['implemented']
    
    # Test MANAGE function
    manage_result = nist.assess_manage_function({
        'risk_treatment': 'mitigate',
        'monitoring': True,
        'communication': True
    })
    assert manage_result['implemented']


def test_documentation_generation(temp_dir):
    """Test compliance documentation generation."""
    compliance = EUAIActCompliance()
    
    # Generate technical documentation
    tech_doc = compliance.generate_technical_documentation({
        'system_name': 'TestAI',
        'version': '1.0',
        'description': 'Test AI system',
        'capabilities': ['classification', 'prediction']
    })
    
    assert 'system_name' in tech_doc
    assert tech_doc['compliance']['eu_ai_act']
    
    # Save documentation
    doc_file = temp_dir / 'tech_doc.json'
    with open(doc_file, 'w') as f:
        json.dump(tech_doc, f)
    
    assert doc_file.exists()


def test_compliance_scoring_accuracy():
    """Test accuracy of compliance scoring."""
    eu_compliance = EUAIActCompliance()
    nist_compliance = NISTAIRMFCompliance()
    
    # Full compliance scenario
    full_compliance_data = {
        'transparency': True,
        'human_oversight': True,
        'robustness': True,
        'documentation': True,
        'risk_assessment': True
    }
    
    eu_score = eu_compliance.calculate_compliance_score(full_compliance_data)
    assert eu_score > 0.8
    
    # Partial compliance scenario
    partial_compliance_data = {
        'transparency': True,
        'human_oversight': False,
        'robustness': True,
        'documentation': False,
        'risk_assessment': True
    }
    
    partial_score = eu_compliance.calculate_compliance_score(partial_compliance_data)
    assert 0.4 < partial_score < 0.7


# ============================================================================
# Policy Engine Tests
# ============================================================================

def test_policy_parsing_and_validation(temp_dir):
    """Test policy parsing from YAML/JSON."""
    policy_engine = PolicyEngine()
    
    # Create YAML policy
    yaml_policy = """
    policies:
      - name: data_quality
        type: threshold
        rules:
          - field: accuracy
            operator: gte
            value: 0.95
        enforcement: strict
      - name: model_size
        type: range
        rules:
          - field: parameters
            operator: between
            min: 1000
            max: 1000000
        enforcement: advisory
    """
    
    policy_file = temp_dir / 'policies.yaml'
    with open(policy_file, 'w') as f:
        f.write(yaml_policy)
    
    # Load policies
    with open(policy_file, 'r') as f:
        policies_data = yaml.safe_load(f)
    
    for policy_data in policies_data['policies']:
        policy_engine.add_policy(policy_data)
    
    assert len(policy_engine.policies) >= 2


def test_conflict_resolution():
    """Test policy conflict resolution."""
    engine = PolicyEngine()
    
    # Add conflicting policies
    engine.add_policy({
        'name': 'strict_accuracy',
        'type': 'threshold',
        'rules': [{'field': 'accuracy', 'operator': 'gte', 'value': 0.99}],
        'priority': 1
    })
    
    engine.add_policy({
        'name': 'relaxed_accuracy',
        'type': 'threshold',
        'rules': [{'field': 'accuracy', 'operator': 'gte', 'value': 0.90}],
        'priority': 2
    })
    
    # Evaluate with conflict
    result = engine.evaluate({'accuracy': 0.95})
    
    # Higher priority (strict) should win
    assert not result.compliant  # 0.95 < 0.99


def test_dynamic_policy_updates():
    """Test dynamic policy addition and removal."""
    engine = PolicyEngine()
    
    # Add policy
    policy_id = engine.add_policy({
        'name': 'test_policy',
        'type': 'threshold',
        'rules': [{'field': 'score', 'operator': 'gte', 'value': 0.5}]
    })
    
    assert policy_id in engine.policies
    
    # Update policy
    engine.update_policy(policy_id, {
        'rules': [{'field': 'score', 'operator': 'gte', 'value': 0.7}]
    })
    
    # Remove policy
    engine.remove_policy(policy_id)
    assert policy_id not in engine.policies


def test_performance_under_load():
    """Test policy engine performance with many policies."""
    engine = PolicyEngine()
    
    # Add many policies
    for i in range(100):
        engine.add_policy({
            'name': f'policy_{i}',
            'type': 'threshold',
            'rules': [{'field': f'metric_{i}', 'operator': 'gte', 'value': i/100}]
        })
    
    # Create test data
    test_data = {f'metric_{i}': i/100 + 0.1 for i in range(100)}
    
    # Measure evaluation time
    start = time.time()
    result = engine.evaluate(test_data)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should complete within 1 second
    assert result.compliant


# ============================================================================
# Audit System Tests
# ============================================================================

def test_log_tamper_detection(temp_dir):
    """Test detection of tampered logs."""
    log_file = temp_dir / 'audit.log'
    logger = AuditLogger(str(log_file))
    
    # Log events
    logger.log_event('test', 'INFO', {'data': 'original'})
    logger.log_event('test', 'INFO', {'data': 'second'})
    
    # Tamper with log file
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) > 0:
        # Modify first log entry
        log_entry = json.loads(lines[0])
        log_entry['details']['data'] = 'tampered'
        lines[0] = json.dumps(log_entry) + '\n'
    
    with open(log_file, 'w') as f:
        f.writelines(lines)
    
    # Verify should detect tampering
    assert not logger.verify_integrity()


def test_query_performance(temp_dir):
    """Test audit log query performance."""
    log_file = temp_dir / 'audit.log'
    logger = AuditLogger(str(log_file))
    
    # Log many events
    for i in range(1000):
        logger.log_event(
            f'event_type_{i % 10}',
            'INFO' if i % 2 == 0 else 'WARNING',
            {'index': i}
        )
    
    # Test query performance
    start = time.time()
    results = logger.query_logs(
        event_type='event_type_0',
        severity='INFO'
    )
    elapsed = time.time() - start
    
    assert elapsed < 0.5  # Should be fast
    assert len(results) > 0


def test_retention_management(temp_dir):
    """Test log retention policies."""
    log_file = temp_dir / 'audit.log'
    logger = AuditLogger(str(log_file), retention_days=1)
    
    # Log old events (simulate by modifying timestamps)
    old_timestamp = (datetime.now() - timedelta(days=2)).isoformat()
    
    # Log current events
    logger.log_event('current', 'INFO', {'data': 'new'})
    
    # Manually add old event
    with open(log_file, 'a') as f:
        old_event = {
            'timestamp': old_timestamp,
            'event_type': 'old',
            'severity': 'INFO',
            'details': {'data': 'old'}
        }
        f.write(json.dumps(old_event) + '\n')
    
    # Apply retention
    logger.apply_retention_policy()
    
    # Check old events are removed
    results = logger.query_logs()
    for event in results:
        event_time = datetime.fromisoformat(event['timestamp'])
        assert (datetime.now() - event_time).days <= 1


def test_export_formats(temp_dir):
    """Test SIEM export formats."""
    log_file = temp_dir / 'audit.log'
    logger = AuditLogger(str(log_file))
    
    # Log events
    logger.log_event('security', 'HIGH', {
        'user': 'admin',
        'action': 'modify_policy',
        'resource': 'governance_config'
    })
    
    # Export to CEF
    cef_file = temp_dir / 'export.cef'
    logger.export_to_siem(str(cef_file), format='CEF')
    assert cef_file.exists()
    
    # Export to LEEF
    leef_file = temp_dir / 'export.leef'
    logger.export_to_siem(str(leef_file), format='LEEF')
    assert leef_file.exists()


# ============================================================================
# Integration Scenarios
# ============================================================================

def test_end_to_end_governance_workflow(governance_config, sample_model, temp_dir):
    """Test complete governance workflow."""
    # Initialize components
    framework = GovernanceFramework(governance_config)
    eu_compliance = EUAIActCompliance()
    nist_compliance = NISTAIRMFCompliance()
    policy_engine = PolicyEngine()
    risk_assessment = AIRiskAssessment()
    audit_logger = AuditLogger(str(temp_dir / 'audit.log'))
    
    # Step 1: Risk Assessment
    risks = risk_assessment.assess_ai_risks({
        'model_type': 'classification',
        'data_sensitivity': 'medium',
        'deployment_scale': 'enterprise'
    })
    
    audit_logger.log_event('risk_assessment', 'INFO', risks)
    
    # Step 2: Policy Evaluation
    policy_result = policy_engine.evaluate({
        'risk_score': risks['overall_score'],
        'compliance_required': True
    })
    
    audit_logger.log_event('policy_evaluation', 'INFO', {
        'compliant': policy_result.compliant
    })
    
    # Step 3: Compliance Checking
    eu_result = eu_compliance.check_compliance({
        'risk_category': 'limited',
        'transparency': True,
        'human_oversight': True
    })
    
    nist_result = nist_compliance.assess_lifecycle_phase('deployment', {
        'testing_complete': True,
        'monitoring_enabled': True
    })
    
    # Step 4: Decision Logging
    decision = framework.log_decision(
        'model_deployment',
        'approved' if eu_result['compliant'] else 'rejected',
        {
            'eu_compliant': eu_result['compliant'],
            'nist_compliant': nist_result['compliant'],
            'risk_score': risks['overall_score']
        }
    )
    
    # Step 5: Generate Report
    report = framework.generate_compliance_report()
    
    # Verify workflow completion
    assert len(framework.decisions) > 0
    assert audit_logger.verify_integrity()
    assert 'compliance_status' in report


def test_multi_regulation_compliance(sample_model):
    """Test compliance with multiple regulations."""
    # Initialize compliance modules
    eu_compliance = EUAIActCompliance()
    nist_compliance = NISTAIRMFCompliance()
    
    # System metadata
    system_info = {
        'name': 'MultiRegSystem',
        'purpose': 'healthcare_diagnosis',
        'risk_category': 'high',
        'transparency': True,
        'human_oversight': True,
        'robustness': True,
        'documentation': True
    }
    
    # Check EU AI Act
    eu_result = eu_compliance.check_compliance(system_info)
    eu_score = eu_compliance.calculate_compliance_score(system_info)
    
    # Check NIST RMF
    nist_govern = nist_compliance.assess_govern_function(system_info)
    nist_map = nist_compliance.assess_map_function(system_info)
    nist_measure = nist_compliance.assess_measure_function(system_info)
    nist_manage = nist_compliance.assess_manage_function(system_info)
    
    # Aggregate compliance
    multi_compliance = {
        'eu_ai_act': {
            'compliant': eu_result['compliant'],
            'score': eu_score
        },
        'nist_rmf': {
            'govern': nist_govern['score'],
            'map': nist_map['score'],
            'measure': nist_measure['score'],
            'manage': nist_manage['score']
        }
    }
    
    # Verify multi-regulation compliance
    assert multi_compliance['eu_ai_act']['score'] > 0
    assert all(v > 0 for k, v in multi_compliance['nist_rmf'].items())


def test_policy_violation_handling(temp_dir):
    """Test handling of policy violations."""
    policy_engine = PolicyEngine(enforcement_mode='strict')
    audit_logger = AuditLogger(str(temp_dir / 'audit.log'))
    
    # Add strict policy
    policy_engine.add_policy({
        'name': 'accuracy_requirement',
        'type': 'threshold',
        'rules': [{'field': 'accuracy', 'operator': 'gte', 'value': 0.95}],
        'enforcement': 'strict'
    })
    
    # Test violation
    test_data = {'accuracy': 0.90}
    result = policy_engine.evaluate(test_data)
    
    if not result.compliant:
        # Log violation
        audit_logger.log_event(
            'policy_violation',
            'WARNING',
            {
                'policy': 'accuracy_requirement',
                'violations': result.violations,
                'data': test_data
            }
        )
        
        # Trigger remediation
        remediation = {
            'action': 'retrain_model',
            'reason': 'accuracy_below_threshold',
            'current_value': test_data['accuracy'],
            'required_value': 0.95
        }
        
        audit_logger.log_event('remediation_triggered', 'INFO', remediation)
    
    # Verify violation handling
    violations = audit_logger.query_logs(event_type='policy_violation')
    assert len(violations) > 0
    
    remediations = audit_logger.query_logs(event_type='remediation_triggered')
    assert len(remediations) > 0


def test_risk_assessment_pipeline(sample_model):
    """Test complete risk assessment pipeline."""
    risk_assessment = AIRiskAssessment()
    
    # Define test scenarios
    scenarios = [
        {
            'name': 'low_risk_scenario',
            'model_type': 'spam_filter',
            'data_sensitivity': 'low',
            'deployment_scale': 'small',
            'expected_risk': 'low'
        },
        {
            'name': 'high_risk_scenario',
            'model_type': 'medical_diagnosis',
            'data_sensitivity': 'high',
            'deployment_scale': 'large',
            'expected_risk': 'high'
        }
    ]
    
    for scenario in scenarios:
        # Assess risks
        risks = risk_assessment.assess_ai_risks({
            'model_type': scenario['model_type'],
            'data_sensitivity': scenario['data_sensitivity'],
            'deployment_scale': scenario['deployment_scale']
        })
        
        # Verify risk categorization
        overall_risk = 'low' if risks['overall_score'] < 3 else 'high'
        assert overall_risk == scenario['expected_risk']
        
        # Get mitigation recommendations
        mitigations = risk_assessment.recommend_mitigations(risks)
        assert len(mitigations) > 0
        
        # Generate risk report
        report = risk_assessment.generate_risk_report(risks)
        assert 'risk_matrix' in report
        assert 'recommendations' in report


# ============================================================================
# Dashboard and Visualization Tests
# ============================================================================

def test_compliance_dashboard_generation(temp_dir):
    """Test dashboard HTML generation."""
    dashboard = ComplianceDashboard()
    
    # Add sample metrics
    dashboard.add_metric('compliance_score', 0.85, {'framework': 'EU_AI_Act'})
    dashboard.add_metric('risk_score', 2.5, {'category': 'limited'})
    dashboard.add_metric('policy_violations', 3, {'severity': 'low'})
    
    # Generate dashboard
    html_file = temp_dir / 'dashboard.html'
    dashboard.generate_dashboard(str(html_file))
    
    assert html_file.exists()
    
    # Verify content
    with open(html_file, 'r') as f:
        content = f.read()
        assert 'Compliance Dashboard' in content
        assert 'Chart.js' in content


def test_real_time_monitoring():
    """Test real-time metric updates."""
    dashboard = ComplianceDashboard()
    
    # Simulate real-time updates
    for i in range(10):
        dashboard.add_metric(
            'accuracy',
            0.9 + i * 0.01,
            {'timestamp': datetime.now().isoformat()}
        )
        time.sleep(0.1)
    
    # Get latest metrics
    latest = dashboard.get_latest_metrics()
    assert 'accuracy' in latest
    assert latest['accuracy']['value'] >= 0.99


# ============================================================================
# Performance and Load Tests
# ============================================================================

def test_governance_framework_performance():
    """Test framework performance under load."""
    config = {
        'organization': 'PerfTest',
        'policies': {
            'data_retention_days': 90
        }
    }
    
    framework = GovernanceFramework(config)
    
    # Log many decisions
    start = time.time()
    for i in range(1000):
        framework.log_decision(
            f'decision_{i}',
            'approved' if i % 2 == 0 else 'rejected',
            {'index': i}
        )
    elapsed = time.time() - start
    
    assert elapsed < 5.0  # Should handle 1000 decisions in < 5 seconds
    assert len(framework.decisions) == 1000


def test_concurrent_policy_evaluation():
    """Test concurrent policy evaluations."""
    import threading
    
    engine = PolicyEngine()
    results = []
    
    # Add policies
    for i in range(10):
        engine.add_policy({
            'name': f'policy_{i}',
            'type': 'threshold',
            'rules': [{'field': f'metric_{i}', 'operator': 'gte', 'value': 0.5}]
        })
    
    def evaluate_policies():
        test_data = {f'metric_{i}': 0.7 for i in range(10)}
        result = engine.evaluate(test_data)
        results.append(result)
    
    # Run concurrent evaluations
    threads = []
    for _ in range(100):
        t = threading.Thread(target=evaluate_policies)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    # Verify all evaluations completed
    assert len(results) == 100
    assert all(r.compliant for r in results)


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_invalid_policy_handling():
    """Test handling of invalid policies."""
    engine = PolicyEngine()
    
    # Try to add invalid policy
    with pytest.raises(ValueError):
        engine.add_policy({
            'name': 'invalid',
            'type': 'unknown_type',
            'rules': []
        })


def test_corrupted_log_recovery(temp_dir):
    """Test recovery from corrupted audit logs."""
    log_file = temp_dir / 'audit.log'
    logger = AuditLogger(str(log_file))
    
    # Log some events
    logger.log_event('test', 'INFO', {'data': 'test'})
    
    # Corrupt log file
    with open(log_file, 'a') as f:
        f.write('corrupted_line_not_json\n')
    
    # Logger should handle corruption gracefully
    try:
        results = logger.query_logs()
        # Should return valid entries despite corruption
        assert isinstance(results, list)
    except Exception as e:
        pytest.fail(f"Failed to handle corrupted log: {e}")


def test_missing_configuration_handling():
    """Test handling of missing configuration."""
    # Create framework with minimal config
    minimal_config = {'organization': 'Test'}
    
    try:
        framework = GovernanceFramework(minimal_config)
        # Should initialize with defaults
        assert framework.config['organization'] == 'Test'
    except Exception as e:
        pytest.fail(f"Failed to handle minimal config: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])