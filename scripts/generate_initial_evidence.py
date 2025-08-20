#!/usr/bin/env python3
"""
Generate Initial Evidence Data for Dashboard
Creates sample validation run data to test the evidence logging system
"""

import sys
import pathlib
import datetime
import time

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pot.core.evidence_logger import EvidenceLogger, log_enhanced_diff_test, log_zk_integration_test, log_interface_tests


def generate_sample_validation_data():
    """Generate sample validation data for testing the evidence system"""
    
    print("🔄 Generating sample evidence data...")
    
    # Sample enhanced difference test
    log_enhanced_diff_test({
        'statistical_results': {
            'decision': 'SAME',
            'confidence': 0.975,
            'n_used': 25,
            'n_max': 120,
            'mean_diff': 0.001234,
            'ci_lower': -0.015432,
            'ci_upper': 0.017900,
            'half_width': 0.016666,
            'effect_size': 0.001234,
            'rule_fired': 'SAME criteria met within gamma bounds'
        },
        'timing': {
            't_per_query': 0.852,
            't_total': 21.3,
            'hardware': 'mps'
        },
        'models': {
            'model_a': 'gpt2',
            'model_b': 'gpt2'
        },
        'success': True,
        'test_type': 'enhanced_diff_quick_gate'
    })
    
    # Sample different models test
    log_enhanced_diff_test({
        'statistical_results': {
            'decision': 'DIFFERENT',
            'confidence': 0.99,
            'n_used': 32,
            'n_max': 400,
            'mean_diff': -8.75421,
            'ci_lower': -12.3421,
            'ci_upper': -5.1663,
            'half_width': 3.5879,
            'effect_size': 8.75421,
            'rule_fired': 'DIFFERENT criteria met with sufficient effect size'
        },
        'timing': {
            't_per_query': 0.734,
            't_total': 23.5,
            'hardware': 'mps'
        },
        'models': {
            'model_a': 'gpt2',
            'model_b': 'distilgpt2'
        },
        'success': True,
        'test_type': 'enhanced_diff_audit_grade'
    })
    
    # Sample ZK integration test
    log_zk_integration_test({
        'zk_proofs': [
            {
                'proof_generated': True,
                'proof_type': 'sgd',
                'proof_size_bytes': 924,
                'generation_time': 0.456,
                'verification_result': True
            },
            {
                'proof_generated': True,
                'proof_type': 'lora',
                'proof_size_bytes': 632,
                'generation_time': 0.287,
                'verification_result': True
            }
        ],
        'success': True,
        'models': {
            'zk_system': 'Halo2System'
        },
        'test_type': 'zk_proof_generation'
    })
    
    # Sample interface compliance test
    log_interface_tests({
        'interface_tests': {
            'total_tests': 28,
            'passed_tests': 28,
            'failed_tests': 0,
            'test_names': [
                'test_interface_inheritance',
                'test_log_event_implementation',
                'test_generate_proof_implementation',
                'test_verify_proof_implementation',
                'test_get_merkle_root_implementation'
            ],
            'failure_details': {},
            'compliance_rate': 1.0
        },
        'success': True,
        'models': {
            'auditor': 'TrainingProvenanceAuditor'
        },
        'test_type': 'interface_compliance_full'
    })
    
    # Sample runtime validation with timing variations
    for i in range(3):
        log_enhanced_diff_test({
            'statistical_results': {
                'decision': 'UNDECIDED',
                'confidence': 0.95,
                'n_used': 15 + i*2,
                'n_max': 50,
                'mean_diff': 0.012345 * (i+1),
                'ci_lower': -0.8 - i*0.1,
                'ci_upper': 0.9 + i*0.1,
                'half_width': 0.85 + i*0.1,
                'effect_size': 0.012345 * (i+1),
                'rule_fired': f'Neither SAME nor DIFFERENT criteria met at n={15 + i*2}'
            },
            'timing': {
                't_per_query': 0.890 - i*0.05,  # Show performance variation
                't_total': 15.4 + i*2.1,
                'hardware': 'mps'
            },
            'models': {
                'model_a': 'gpt2',
                'model_b': 'gpt2-medium'
            },
            'success': True,
            'test_type': f'runtime_validation_run_{i+1}'
        })
    
    print("✅ Generated 7 sample validation runs")
    print("📊 Evidence data ready for dashboard generation")


def main():
    """Generate initial evidence data"""
    generate_sample_validation_data()
    
    # Now update the dashboard
    print("\n🔄 Updating evidence dashboard...")
    from scripts.update_evidence_dashboard import main as update_dashboard
    return update_dashboard()


if __name__ == "__main__":
    sys.exit(main())