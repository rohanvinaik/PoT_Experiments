"""
Complete governance workflow demonstration for PoT framework.

This example demonstrates:
- Full governance workflow integration
- Compliance checking across multiple regulations
- Policy violation handling
- Report generation and export
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pot.core.governance import GovernanceFramework
from pot.governance.audit_logger import AuditLogger
from pot.governance.compliance_dashboard import ComplianceDashboard
from pot.governance.eu_ai_act_compliance import EUAIActCompliance
from pot.governance.nist_ai_rmf_compliance import NISTAIRMFCompliance
from pot.governance.policy_engine import PolicyEngine
from pot.governance.risk_assessment import AIRiskAssessment
from pot.security.proof_of_training import ProofOfTraining

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoModel(nn.Module):
    """Simple model for demonstration."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_governance_config():
    """Create comprehensive governance configuration."""
    return {
        'organization': 'PoT Demo Organization',
        'contact': 'governance@pot-demo.org',
        'version': '1.0.0',
        
        'policies': {
            'data_retention_days': 365,
            'model_retraining_frequency': 30,
            'access_control_level': 'strict',
            'verification_required': True,
            'min_verification_confidence': 0.85
        },
        
        'compliance': {
            'frameworks': ['EU_AI_Act', 'NIST_AI_RMF'],
            'risk_appetite': 'low',
            'audit_enabled': True
        },
        
        'monitoring': {
            'real_time': True,
            'alert_on_violation': True,
            'dashboard_enabled': True
        }
    }


def create_pot_config():
    """Create PoT verification configuration."""
    return {
        'model_id': 'demo_model_001',
        'challenge_types': ['frequency', 'texture'],
        'num_challenges': 5,
        'confidence_threshold': 0.85,
        'security_level': 'high'
    }


def demonstrate_governance_workflow():
    """Demonstrate complete governance workflow."""
    
    print("\n" + "="*60)
    print("PoT GOVERNANCE WORKFLOW DEMONSTRATION")
    print("="*60)
    
    # Initialize components
    print("\n[1] Initializing Governance Components...")
    
    governance_config = create_governance_config()
    pot_config = create_pot_config()
    
    governance = GovernanceFramework(governance_config)
    eu_compliance = EUAIActCompliance()
    nist_compliance = NISTAIRMFCompliance()
    policy_engine = PolicyEngine()
    risk_assessment = AIRiskAssessment()
    audit_logger = AuditLogger('demo_audit.log')
    dashboard = ComplianceDashboard()
    
    print("✓ All governance components initialized")
    
    # Create and verify model
    print("\n[2] Creating and Verifying Model...")
    
    model = DemoModel()
    
    # Simulate PoT verification (in practice, use actual ProofOfTraining)
    verification_result = {
        'verified': True,
        'confidence': 0.92,
        'challenges_passed': 5,
        'challenges_total': 5,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"✓ Model verified with confidence: {verification_result['confidence']:.2%}")
    
    # Risk Assessment
    print("\n[3] Performing Risk Assessment...")
    
    risks = risk_assessment.assess_ai_risks({
        'model_type': 'classification',
        'purpose': 'medical_diagnosis',
        'data_sensitivity': 'high',
        'deployment_scale': 'enterprise',
        'user_base': 'healthcare_professionals'
    })
    
    print(f"✓ Risk assessment complete:")
    print(f"  - Overall risk score: {risks['overall_score']:.2f}")
    print(f"  - Risk level: {risks['risk_level']}")
    
    # Log risk assessment
    audit_logger.log_event(
        'risk_assessment',
        'INFO',
        {
            'model_id': pot_config['model_id'],
            'risks': risks
        }
    )
    
    # EU AI Act Compliance Check
    print("\n[4] Checking EU AI Act Compliance...")
    
    # Categorize risk
    risk_category = eu_compliance.categorize_risk('medical_diagnosis')
    print(f"  - Risk category: {risk_category}")
    
    # Check compliance
    eu_result = eu_compliance.check_compliance({
        'risk_category': risk_category,
        'transparency': True,
        'human_oversight': True,
        'robustness': True,
        'accuracy': True,
        'documentation': True
    })
    
    eu_score = eu_compliance.calculate_compliance_score({
        'transparency': True,
        'human_oversight': True,
        'robustness': True,
        'accuracy': True,
        'documentation': True
    })
    
    print(f"✓ EU AI Act compliance:")
    print(f"  - Compliant: {eu_result['compliant']}")
    print(f"  - Score: {eu_score:.2%}")
    
    # NIST AI RMF Assessment
    print("\n[5] Assessing NIST AI RMF Compliance...")
    
    # Assess core functions
    govern_result = nist_compliance.assess_govern_function({
        'policies': ['ai_governance', 'risk_management'],
        'accountability': True,
        'culture': 'risk-aware',
        'resources': 'adequate'
    })
    
    map_result = nist_compliance.assess_map_function({
        'context': 'healthcare',
        'stakeholders': ['patients', 'doctors', 'regulators'],
        'risks_identified': True,
        'impacts_assessed': True
    })
    
    measure_result = nist_compliance.assess_measure_function({
        'metrics': ['accuracy', 'fairness', 'robustness'],
        'testing': True,
        'monitoring': True,
        'documentation': True
    })
    
    manage_result = nist_compliance.assess_manage_function({
        'risk_treatment': 'mitigate',
        'monitoring': True,
        'communication': True,
        'improvement': True
    })
    
    print("✓ NIST AI RMF assessment:")
    print(f"  - GOVERN score: {govern_result['score']:.2f}/5")
    print(f"  - MAP score: {map_result['score']:.2f}/5")
    print(f"  - MEASURE score: {measure_result['score']:.2f}/5")
    print(f"  - MANAGE score: {manage_result['score']:.2f}/5")
    
    # Policy Evaluation
    print("\n[6] Evaluating Policies...")
    
    # Add policies
    policy_engine.add_policy({
        'name': 'verification_confidence',
        'type': 'threshold',
        'rules': [{
            'field': 'confidence',
            'operator': 'gte',
            'value': governance_config['policies']['min_verification_confidence']
        }],
        'enforcement': 'strict'
    })
    
    policy_engine.add_policy({
        'name': 'risk_threshold',
        'type': 'threshold',
        'rules': [{
            'field': 'risk_score',
            'operator': 'lte',
            'value': 3.0  # Medium risk threshold
        }],
        'enforcement': 'advisory'
    })
    
    # Evaluate policies
    policy_result = policy_engine.evaluate({
        'confidence': verification_result['confidence'],
        'risk_score': risks['overall_score'],
        'eu_compliant': eu_result['compliant'],
        'nist_compliant': all([
            govern_result['implemented'],
            map_result['implemented'],
            measure_result['implemented'],
            manage_result['implemented']
        ])
    })
    
    print(f"✓ Policy evaluation:")
    print(f"  - Compliant: {policy_result.compliant}")
    if policy_result.violations:
        print(f"  - Violations: {policy_result.violations}")
    else:
        print("  - No violations detected")
    
    # Log policy evaluation
    audit_logger.log_event(
        'policy_evaluation',
        'INFO' if policy_result.compliant else 'WARNING',
        {
            'compliant': policy_result.compliant,
            'violations': policy_result.violations
        }
    )
    
    # Handle Policy Violations (if any)
    if not policy_result.compliant:
        print("\n[7] Handling Policy Violations...")
        
        for violation in policy_result.violations:
            print(f"  ⚠ Violation: {violation}")
            
            # Log violation
            audit_logger.log_event(
                'policy_violation',
                'WARNING',
                {
                    'policy': violation['policy'],
                    'rule': violation['rule'],
                    'severity': 'medium'
                }
            )
            
            # Trigger remediation
            remediation = {
                'action': 'require_review',
                'reason': violation,
                'reviewer': 'compliance_team'
            }
            
            audit_logger.log_event(
                'remediation_triggered',
                'INFO',
                remediation
            )
            
            print(f"  → Remediation: {remediation['action']}")
    
    # Make Deployment Decision
    print("\n[8] Making Deployment Decision...")
    
    # Aggregate compliance status
    overall_compliant = all([
        verification_result['verified'],
        eu_result['compliant'],
        policy_result.compliant,
        risks['risk_level'] in ['low', 'medium']
    ])
    
    deployment_decision = 'approved' if overall_compliant else 'requires_review'
    
    # Log decision
    decision = governance.log_decision(
        'model_deployment',
        deployment_decision,
        {
            'model_id': pot_config['model_id'],
            'verification_confidence': verification_result['confidence'],
            'eu_compliance_score': eu_score,
            'risk_level': risks['risk_level'],
            'policy_compliant': policy_result.compliant
        }
    )
    
    print(f"✓ Deployment decision: {deployment_decision.upper()}")
    print(f"  - Decision ID: {decision['id']}")
    print(f"  - Timestamp: {decision['timestamp']}")
    
    # Update Dashboard Metrics
    print("\n[9] Updating Compliance Dashboard...")
    
    dashboard.add_metric('verification_confidence', verification_result['confidence'], {
        'model_id': pot_config['model_id']
    })
    
    dashboard.add_metric('eu_compliance_score', eu_score, {
        'framework': 'EU_AI_Act'
    })
    
    dashboard.add_metric('nist_rmf_score', 
        (govern_result['score'] + map_result['score'] + 
         measure_result['score'] + manage_result['score']) / 4,
        {'framework': 'NIST_AI_RMF'}
    )
    
    dashboard.add_metric('risk_score', risks['overall_score'], {
        'category': risks['risk_level']
    })
    
    dashboard.add_metric('policy_violations', 
        len(policy_result.violations) if policy_result.violations else 0,
        {'severity': 'medium'}
    )
    
    print("✓ Dashboard metrics updated")
    
    # Generate Reports
    print("\n[10] Generating Compliance Reports...")
    
    # Generate governance report
    governance_report = governance.generate_compliance_report()
    
    # Generate EU AI Act documentation
    if risk_category in ['high', 'limited']:
        tech_doc = eu_compliance.generate_technical_documentation({
            'system_name': 'PoT Demo Medical Diagnosis System',
            'version': '1.0.0',
            'description': 'AI system for medical diagnosis assistance',
            'capabilities': ['image_classification', 'risk_assessment'],
            'model_id': pot_config['model_id'],
            'verification_result': verification_result
        })
        
        print("✓ Technical documentation generated")
        
        # Save documentation
        with open('demo_technical_documentation.json', 'w') as f:
            json.dump(tech_doc, f, indent=2)
    
    # Generate risk report
    risk_report = risk_assessment.generate_risk_report(risks)
    print("✓ Risk assessment report generated")
    
    # Generate HTML dashboard
    dashboard.generate_dashboard('demo_dashboard.html')
    print("✓ HTML dashboard generated: demo_dashboard.html")
    
    # Export metrics
    dashboard.export_metrics('demo_metrics.json', format='json')
    dashboard.export_metrics('demo_metrics.csv', format='csv')
    print("✓ Metrics exported to JSON and CSV")
    
    # Verify Audit Trail Integrity
    print("\n[11] Verifying Audit Trail...")
    
    integrity_check = audit_logger.verify_integrity()
    print(f"✓ Audit trail integrity: {'VALID' if integrity_check else 'COMPROMISED'}")
    
    # Query audit logs
    recent_events = audit_logger.query_logs(
        start_time=(datetime.now() - timedelta(hours=1)).isoformat()
    )
    print(f"✓ Audit events logged: {len(recent_events)}")
    
    # Summary
    print("\n" + "="*60)
    print("GOVERNANCE WORKFLOW SUMMARY")
    print("="*60)
    
    summary = {
        'Model ID': pot_config['model_id'],
        'Verification Confidence': f"{verification_result['confidence']:.2%}",
        'Risk Level': risks['risk_level'],
        'EU AI Act Compliant': '✓' if eu_result['compliant'] else '✗',
        'NIST RMF Compliant': '✓' if all([
            govern_result['implemented'],
            map_result['implemented'],
            measure_result['implemented'],
            manage_result['implemented']
        ]) else '✗',
        'Policy Violations': len(policy_result.violations) if policy_result.violations else 0,
        'Deployment Decision': deployment_decision.upper(),
        'Audit Events': len(recent_events)
    }
    
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Governance workflow completed successfully!")
    
    return {
        'success': True,
        'decision': deployment_decision,
        'summary': summary
    }


def demonstrate_policy_violation_scenario():
    """Demonstrate handling of policy violations."""
    
    print("\n" + "="*60)
    print("POLICY VIOLATION SCENARIO")
    print("="*60)
    
    # Initialize components
    policy_engine = PolicyEngine(enforcement_mode='strict')
    audit_logger = AuditLogger('violation_demo.log')
    
    # Add strict policy
    policy_engine.add_policy({
        'name': 'high_accuracy_requirement',
        'type': 'threshold',
        'rules': [{
            'field': 'accuracy',
            'operator': 'gte',
            'value': 0.95
        }],
        'enforcement': 'strict',
        'priority': 1
    })
    
    print("\n[1] Testing model with insufficient accuracy...")
    
    # Test with violation
    test_data = {
        'accuracy': 0.88,  # Below threshold
        'model_id': 'test_model_002'
    }
    
    result = policy_engine.evaluate(test_data)
    
    if not result.compliant:
        print(f"✗ Policy violation detected!")
        print(f"  - Policy: {result.violations[0]['policy']}")
        print(f"  - Current accuracy: {test_data['accuracy']:.2%}")
        print(f"  - Required accuracy: ≥95%")
        
        # Log violation
        audit_logger.log_event(
            'policy_violation',
            'HIGH',
            {
                'policy': 'high_accuracy_requirement',
                'model_id': test_data['model_id'],
                'violations': result.violations,
                'data': test_data
            }
        )
        
        print("\n[2] Triggering automated remediation...")
        
        # Define remediation actions
        remediation_actions = [
            {
                'action': 'block_deployment',
                'reason': 'accuracy_below_threshold',
                'immediate': True
            },
            {
                'action': 'notify_ml_team',
                'message': 'Model requires retraining to meet accuracy requirements',
                'priority': 'high'
            },
            {
                'action': 'schedule_retraining',
                'deadline': (datetime.now() + timedelta(days=7)).isoformat(),
                'requirements': {'min_accuracy': 0.95}
            }
        ]
        
        for action in remediation_actions:
            audit_logger.log_event('remediation_action', 'INFO', action)
            print(f"  → {action['action']}: {action.get('reason', action.get('message', ''))}")
        
        print("\n✓ Remediation actions logged and initiated")
    
    # Demonstrate successful compliance
    print("\n[3] Testing model with compliant accuracy...")
    
    compliant_data = {
        'accuracy': 0.97,  # Above threshold
        'model_id': 'test_model_003'
    }
    
    result = policy_engine.evaluate(compliant_data)
    
    if result.compliant:
        print(f"✓ Model complies with all policies")
        print(f"  - Accuracy: {compliant_data['accuracy']:.2%}")
        print(f"  - Status: APPROVED for deployment")
        
        audit_logger.log_event(
            'policy_compliance',
            'INFO',
            {
                'model_id': compliant_data['model_id'],
                'policies_checked': ['high_accuracy_requirement'],
                'result': 'compliant'
            }
        )


def demonstrate_multi_regulation_compliance():
    """Demonstrate compliance across multiple regulations."""
    
    print("\n" + "="*60)
    print("MULTI-REGULATION COMPLIANCE CHECK")
    print("="*60)
    
    # Initialize compliance modules
    eu_compliance = EUAIActCompliance()
    nist_compliance = NISTAIRMFCompliance()
    
    # Define AI system
    system_info = {
        'name': 'Financial Risk Assessment AI',
        'purpose': 'credit_scoring',
        'deployment': 'production',
        'user_base': 'financial_institutions',
        'data_types': ['financial_history', 'personal_data'],
        
        # Compliance measures
        'transparency': True,
        'human_oversight': True,
        'robustness': True,
        'documentation': True,
        'fairness_testing': True,
        'audit_trail': True
    }
    
    print(f"\nSystem: {system_info['name']}")
    print(f"Purpose: {system_info['purpose']}")
    
    # EU AI Act Assessment
    print("\n[1] EU AI Act Compliance:")
    
    risk_category = eu_compliance.categorize_risk(system_info['purpose'])
    print(f"  - Risk Category: {risk_category}")
    
    eu_requirements = eu_compliance.get_requirements(risk_category)
    print(f"  - Requirements: {', '.join(eu_requirements[:3])}...")
    
    eu_result = eu_compliance.check_compliance(system_info)
    eu_score = eu_compliance.calculate_compliance_score(system_info)
    
    print(f"  - Compliance Status: {'✓ COMPLIANT' if eu_result['compliant'] else '✗ NON-COMPLIANT'}")
    print(f"  - Compliance Score: {eu_score:.2%}")
    
    # NIST AI RMF Assessment
    print("\n[2] NIST AI Risk Management Framework:")
    
    # Assess all four functions
    functions = {
        'GOVERN': nist_compliance.assess_govern_function({
            'policies': ['ai_ethics', 'risk_management', 'data_governance'],
            'accountability': True,
            'culture': 'risk-aware'
        }),
        'MAP': nist_compliance.assess_map_function({
            'context': 'financial_services',
            'stakeholders': ['customers', 'regulators', 'auditors'],
            'risks_identified': True
        }),
        'MEASURE': nist_compliance.assess_measure_function({
            'metrics': ['accuracy', 'fairness', 'explainability'],
            'testing': True,
            'monitoring': True
        }),
        'MANAGE': nist_compliance.assess_manage_function({
            'risk_treatment': 'mitigate',
            'monitoring': True,
            'communication': True
        })
    }
    
    for func_name, result in functions.items():
        status = '✓' if result['implemented'] else '✗'
        print(f"  - {func_name}: {status} (Score: {result['score']:.1f}/5)")
    
    # Overall Compliance Status
    print("\n[3] Overall Multi-Regulation Compliance:")
    
    overall_compliant = eu_result['compliant'] and all(
        f['implemented'] for f in functions.values()
    )
    
    print(f"\n  {'✓' if overall_compliant else '✗'} System is " +
          f"{'COMPLIANT' if overall_compliant else 'NON-COMPLIANT'} " +
          "with all regulations")
    
    if overall_compliant:
        print("\n  System is approved for deployment with full regulatory compliance!")
    else:
        print("\n  System requires additional compliance measures before deployment.")


def main():
    """Run all governance demonstrations."""
    
    try:
        # Run main governance workflow
        result = demonstrate_governance_workflow()
        
        # Demonstrate policy violation handling
        demonstrate_policy_violation_scenario()
        
        # Demonstrate multi-regulation compliance
        demonstrate_multi_regulation_compliance()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*60)
        
        print("\nGenerated Files:")
        print("  - demo_audit.log (Audit trail)")
        print("  - demo_dashboard.html (Compliance dashboard)")
        print("  - demo_metrics.json (Metrics export)")
        print("  - demo_metrics.csv (Metrics export)")
        print("  - demo_technical_documentation.json (EU AI Act documentation)")
        print("  - violation_demo.log (Policy violation audit)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())