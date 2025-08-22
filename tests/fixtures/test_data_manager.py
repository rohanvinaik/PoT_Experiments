#!/usr/bin/env python3
"""
Test Data Manager

Manages test data, fixtures, and mock objects for comprehensive testing.
Provides reproducible test environments and data generation.
"""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import random
import string
import hashlib
from datetime import datetime, timedelta
import zipfile

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestDataManager:
    """Manages test data and fixtures for CI testing"""
    
    def __init__(self, fixtures_dir: Optional[str] = None):
        self.fixtures_dir = Path(fixtures_dir) if fixtures_dir else Path(__file__).parent
        self.temp_dirs = []
        
        # Ensure fixtures directory exists
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        
        # Test data categories
        self.data_categories = {
            'models': self._generate_model_fixtures,
            'benchmarks': self._generate_benchmark_fixtures,
            'security': self._generate_security_fixtures,
            'zk_proofs': self._generate_zk_fixtures,
            'audit_trails': self._generate_audit_fixtures,
            'evidence_bundles': self._generate_evidence_fixtures
        }
    
    def create_test_environment(self, test_name: str, categories: List[str] = None) -> Dict[str, Any]:
        """Create a complete test environment with fixtures"""
        if categories is None:
            categories = list(self.data_categories.keys())
        
        # Create temporary directory for this test
        temp_dir = Path(tempfile.mkdtemp(prefix=f'test_{test_name}_'))
        self.temp_dirs.append(temp_dir)
        
        test_env = {
            'test_name': test_name,
            'temp_dir': str(temp_dir),
            'created_at': datetime.utcnow().isoformat(),
            'fixtures': {}
        }
        
        # Generate fixtures for each category
        for category in categories:
            if category in self.data_categories:
                generator = self.data_categories[category]
                fixtures = generator(temp_dir / category)
                test_env['fixtures'][category] = fixtures
        
        return test_env
    
    def _generate_model_fixtures(self, output_dir: Path) -> Dict[str, Any]:
        """Generate model-related test fixtures"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fixtures = {
            'mock_models': [],
            'model_configs': [],
            'tokenizer_data': []
        }
        
        # Generate mock model configurations
        model_types = ['gpt2', 'distilgpt2', 'pythia-70m', 'pythia-160m']
        
        for i, model_type in enumerate(model_types):
            model_config = {
                'model_name': f'test_{model_type}',
                'model_type': model_type,
                'vocab_size': 50257 if 'gpt2' in model_type else 50432,
                'hidden_size': 768 + (i * 256),
                'num_layers': 12 + (i * 2),
                'num_heads': 12,
                'max_position_embeddings': 1024,
                'created_for_test': True
            }
            
            config_file = output_dir / f'{model_type}_config.json'
            with open(config_file, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            fixtures['model_configs'].append({
                'model_type': model_type,
                'config_file': str(config_file),
                'config': model_config
            })
        
        # Generate mock tokenizer data
        vocab_file = output_dir / 'test_vocab.json'
        vocab = {f'token_{i}': i for i in range(1000)}
        with open(vocab_file, 'w') as f:
            json.dump(vocab, f)
        
        fixtures['tokenizer_data'].append({
            'vocab_file': str(vocab_file),
            'vocab_size': len(vocab)
        })
        
        return fixtures
    
    def _generate_benchmark_fixtures(self, output_dir: Path) -> Dict[str, Any]:
        """Generate benchmark test fixtures"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fixtures = {
            'performance_data': [],
            'historical_baselines': [],
            'regression_test_data': []
        }
        
        # Generate performance benchmark data
        for i in range(5):
            benchmark_data = {
                'timestamp': (datetime.utcnow() - timedelta(days=i)).isoformat(),
                'test_run_id': f'test_run_{i:03d}',
                'metrics': {
                    'verification_time_ms': random.uniform(100, 2000),
                    'memory_usage_mb': random.uniform(100, 1000),
                    'cpu_usage_percent': random.uniform(10, 80),
                    'accuracy': random.uniform(0.85, 0.99),
                    'throughput_qps': random.uniform(1, 50),
                    'proof_generation_time_ms': random.uniform(500, 5000),
                    'proof_size_bytes': random.randint(1000, 50000),
                    'verification_success_rate': random.uniform(0.95, 1.0)
                },
                'system_info': {
                    'cpu_cores': 8,
                    'memory_gb': 16,
                    'os': 'linux'
                }
            }
            
            bench_file = output_dir / f'benchmark_{i:03d}.json'
            with open(bench_file, 'w') as f:
                json.dump(benchmark_data, f, indent=2)
            
            fixtures['performance_data'].append({
                'file': str(bench_file),
                'data': benchmark_data
            })
        
        # Generate historical baseline
        baseline_data = {
            'baseline_version': '1.0.0',
            'established_date': (datetime.utcnow() - timedelta(days=30)).isoformat(),
            'metrics': {
                'verification_time_ms': 500.0,
                'memory_usage_mb': 200.0,
                'cpu_usage_percent': 30.0,
                'accuracy': 0.95,
                'throughput_qps': 10.0
            }
        }
        
        baseline_file = output_dir / 'baseline.json'
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        fixtures['historical_baselines'].append({
            'file': str(baseline_file),
            'data': baseline_data
        })
        
        # Generate regression test scenarios
        regression_scenarios = [
            {
                'name': 'memory_regression',
                'description': 'Simulated memory usage increase',
                'current_metrics': {'memory_usage_mb': 800.0},
                'baseline_metrics': {'memory_usage_mb': 200.0},
                'expected_regression': True
            },
            {
                'name': 'performance_improvement',
                'description': 'Simulated performance improvement',
                'current_metrics': {'verification_time_ms': 300.0},
                'baseline_metrics': {'verification_time_ms': 500.0},
                'expected_regression': False
            },
            {
                'name': 'accuracy_regression',
                'description': 'Simulated accuracy decrease',
                'current_metrics': {'accuracy': 0.85},
                'baseline_metrics': {'accuracy': 0.95},
                'expected_regression': True
            }
        ]
        
        for scenario in regression_scenarios:
            scenario_file = output_dir / f'regression_{scenario["name"]}.json'
            with open(scenario_file, 'w') as f:
                json.dump(scenario, f, indent=2)
            
            fixtures['regression_test_data'].append({
                'scenario': scenario['name'],
                'file': str(scenario_file),
                'data': scenario
            })
        
        return fixtures
    
    def _generate_security_fixtures(self, output_dir: Path) -> Dict[str, Any]:
        """Generate security test fixtures"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fixtures = {
            'attack_scenarios': [],
            'vulnerability_reports': [],
            'security_configs': []
        }
        
        # Generate attack scenarios
        attack_types = [
            'replay_attack',
            'timing_attack',
            'injection_attack',
            'evasion_attack',
            'model_extraction'
        ]
        
        for attack_type in attack_types:
            scenario = {
                'attack_type': attack_type,
                'description': f'Test scenario for {attack_type}',
                'severity': random.choice(['low', 'medium', 'high', 'critical']),
                'test_vectors': [
                    f'test_input_{i}' for i in range(random.randint(3, 10))
                ],
                'expected_detection': random.choice([True, False]),
                'mitigation_strategy': f'mitigation_for_{attack_type}'
            }
            
            scenario_file = output_dir / f'{attack_type}_scenario.json'
            with open(scenario_file, 'w') as f:
                json.dump(scenario, f, indent=2)
            
            fixtures['attack_scenarios'].append({
                'attack_type': attack_type,
                'file': str(scenario_file),
                'data': scenario
            })
        
        # Generate vulnerability reports
        vuln_report = {
            'scan_timestamp': datetime.utcnow().isoformat(),
            'scanner_version': '1.0.0',
            'vulnerabilities': [
                {
                    'id': f'VULN-{i:04d}',
                    'severity': random.choice(['info', 'low', 'medium', 'high']),
                    'description': f'Test vulnerability {i}',
                    'file': f'test_file_{i}.py',
                    'line': random.randint(1, 100),
                    'recommendation': f'Fix vulnerability {i}'
                }
                for i in range(random.randint(0, 5))
            ]
        }
        
        vuln_file = output_dir / 'vulnerability_report.json'
        with open(vuln_file, 'w') as f:
            json.dump(vuln_report, f, indent=2)
        
        fixtures['vulnerability_reports'].append({
            'file': str(vuln_file),
            'data': vuln_report
        })
        
        return fixtures
    
    def _generate_zk_fixtures(self, output_dir: Path) -> Dict[str, Any]:
        """Generate zero-knowledge proof test fixtures"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fixtures = {
            'proof_data': [],
            'circuit_configs': [],
            'verification_keys': []
        }
        
        # Generate proof data
        circuit_sizes = ['tiny', 'small', 'medium', 'large']
        
        for size in circuit_sizes:
            proof_data = {
                'circuit_size': size,
                'constraints': random.randint(100, 1000000),
                'proof_generation_time_ms': random.uniform(10, 10000),
                'proof_size_bytes': random.randint(1000, 100000),
                'verification_time_ms': random.uniform(1, 100),
                'memory_usage_mb': random.uniform(10, 1000),
                'success': True,
                'proof_hex': ''.join(random.choices(string.hexdigits.lower(), k=128))
            }
            
            proof_file = output_dir / f'proof_{size}.json'
            with open(proof_file, 'w') as f:
                json.dump(proof_data, f, indent=2)
            
            fixtures['proof_data'].append({
                'circuit_size': size,
                'file': str(proof_file),
                'data': proof_data
            })
        
        # Generate circuit configurations
        circuit_config = {
            'version': '1.0.0',
            'supported_circuits': circuit_sizes,
            'default_parameters': {
                'security_level': 128,
                'curve': 'bn254',
                'hash_function': 'poseidon'
            },
            'optimization_flags': ['parallel_witness', 'batch_proving']
        }
        
        config_file = output_dir / 'circuit_config.json'
        with open(config_file, 'w') as f:
            json.dump(circuit_config, f, indent=2)
        
        fixtures['circuit_configs'].append({
            'file': str(config_file),
            'data': circuit_config
        })
        
        return fixtures
    
    def _generate_audit_fixtures(self, output_dir: Path) -> Dict[str, Any]:
        """Generate audit trail test fixtures"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fixtures = {
            'audit_logs': [],
            'trail_validation': [],
            'compliance_reports': []
        }
        
        # Generate audit log entries
        audit_entries = []
        for i in range(20):
            entry = {
                'timestamp': (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
                'operation': random.choice(['verify', 'generate_proof', 'validate', 'compare']),
                'actor': f'test_user_{random.randint(1, 5)}',
                'resource': f'model_{random.randint(1, 10)}',
                'outcome': random.choice(['success', 'failure']),
                'hash': hashlib.sha256(f'entry_{i}'.encode()).hexdigest(),
                'previous_hash': hashlib.sha256(f'entry_{i-1}'.encode()).hexdigest() if i > 0 else None,
                'metadata': {
                    'duration_ms': random.randint(100, 5000),
                    'ip_address': f'192.168.1.{random.randint(1, 255)}',
                    'user_agent': 'test_client/1.0'
                }
            }
            audit_entries.append(entry)
        
        audit_file = output_dir / 'audit_trail.ndjson'
        with open(audit_file, 'w') as f:
            for entry in audit_entries:
                f.write(json.dumps(entry) + '\n')
        
        fixtures['audit_logs'].append({
            'file': str(audit_file),
            'entries_count': len(audit_entries)
        })
        
        # Generate validation report
        validation_report = {
            'validation_timestamp': datetime.utcnow().isoformat(),
            'total_entries': len(audit_entries),
            'validation_results': {
                'hash_chain_valid': True,
                'temporal_consistency': True,
                'completeness_check': True,
                'integrity_verified': True
            },
            'warnings': [],
            'errors': []
        }
        
        validation_file = output_dir / 'validation_report.json'
        with open(validation_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        fixtures['trail_validation'].append({
            'file': str(validation_file),
            'data': validation_report
        })
        
        return fixtures
    
    def _generate_evidence_fixtures(self, output_dir: Path) -> Dict[str, Any]:
        """Generate evidence bundle test fixtures"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fixtures = {
            'evidence_bundles': [],
            'verification_reports': []
        }
        
        # Create a sample evidence bundle
        bundle_dir = output_dir / 'sample_bundle'
        bundle_dir.mkdir(exist_ok=True)
        
        # Evidence metadata
        evidence_metadata = {
            'bundle_id': f'test_bundle_{random.randint(1000, 9999)}',
            'created_at': datetime.utcnow().isoformat(),
            'generator_version': '1.0.0',
            'evidence_type': 'verification_evidence',
            'components': {
                'verification_results': True,
                'audit_trail': True,
                'performance_metrics': True,
                'security_analysis': True
            },
            'cryptographic_hash': hashlib.sha256(b'test_evidence').hexdigest()
        }
        
        metadata_file = bundle_dir / 'evidence_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(evidence_metadata, f, indent=2)
        
        # Sample verification results
        verification_results = {
            'decision': 'SAME',
            'confidence': 0.99,
            'n_queries': 32,
            'statistical_test': 'enhanced_sequential',
            'p_value': 0.001,
            'effect_size': 0.02
        }
        
        results_file = bundle_dir / 'verification_results.json'
        with open(results_file, 'w') as f:
            json.dump(verification_results, f, indent=2)
        
        # Create ZIP bundle
        bundle_zip = output_dir / 'test_evidence_bundle.zip'
        with zipfile.ZipFile(bundle_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in bundle_dir.rglob('*'):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(bundle_dir))
        
        fixtures['evidence_bundles'].append({
            'bundle_file': str(bundle_zip),
            'bundle_dir': str(bundle_dir),
            'metadata': evidence_metadata
        })
        
        return fixtures
    
    def create_mock_model_responses(self, model_name: str, num_responses: int = 10) -> List[str]:
        """Generate mock model responses for testing"""
        responses = []
        
        if 'gpt2' in model_name.lower():
            templates = [
                "The quick brown fox jumps over the lazy dog.",
                "In a hole in the ground there lived a hobbit.",
                "It was the best of times, it was the worst of times.",
                "To be or not to be, that is the question.",
                "All happy families are alike; each unhappy family is unhappy in its own way."
            ]
        elif 'pythia' in model_name.lower():
            templates = [
                "Python is a high-level programming language.",
                "Machine learning algorithms require training data.",
                "Neural networks consist of interconnected nodes.",
                "Deep learning is a subset of machine learning.",
                "Artificial intelligence aims to create intelligent machines."
            ]
        else:
            templates = [
                "This is a test response from a language model.",
                "Language models generate text based on patterns.",
                "Testing requires diverse and representative data.",
                "Model verification ensures consistent behavior.",
                "Automated testing improves software quality."
            ]
        
        for i in range(num_responses):
            base_response = random.choice(templates)
            # Add some variation
            variation_num = random.randint(1, 100)
            response = f"{base_response} (variation {variation_num})"
            responses.append(response)
        
        return responses
    
    def cleanup_test_environment(self, test_env: Dict[str, Any] = None):
        """Clean up test environment and temporary files"""
        if test_env and 'temp_dir' in test_env:
            temp_path = Path(test_env['temp_dir'])
            if temp_path.exists():
                shutil.rmtree(temp_path)
        
        # Clean up all tracked temp directories
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Could not clean up {temp_dir}: {e}")
        
        self.temp_dirs.clear()
    
    def get_fixture_path(self, category: str, fixture_name: str) -> Optional[Path]:
        """Get path to a specific fixture file"""
        fixture_path = self.fixtures_dir / category / fixture_name
        return fixture_path if fixture_path.exists() else None
    
    def list_available_fixtures(self) -> Dict[str, List[str]]:
        """List all available fixtures by category"""
        fixtures = {}
        
        for category in self.data_categories.keys():
            category_dir = self.fixtures_dir / category
            if category_dir.exists():
                fixtures[category] = [
                    f.name for f in category_dir.iterdir() 
                    if f.is_file() and f.suffix in ['.json', '.zip', '.ndjson']
                ]
            else:
                fixtures[category] = []
        
        return fixtures


# Convenience functions for common test scenarios
def create_model_verification_test_data(test_name: str = "model_verification") -> Dict[str, Any]:
    """Create test data specifically for model verification tests"""
    manager = TestDataManager()
    return manager.create_test_environment(test_name, ['models', 'benchmarks', 'audit_trails'])


def create_security_test_data(test_name: str = "security_testing") -> Dict[str, Any]:
    """Create test data specifically for security tests"""
    manager = TestDataManager()
    return manager.create_test_environment(test_name, ['security', 'audit_trails', 'evidence_bundles'])


def create_performance_test_data(test_name: str = "performance_testing") -> Dict[str, Any]:
    """Create test data specifically for performance tests"""
    manager = TestDataManager()
    return manager.create_test_environment(test_name, ['benchmarks', 'models', 'zk_proofs'])


def create_zk_test_data(test_name: str = "zk_testing") -> Dict[str, Any]:
    """Create test data specifically for ZK proof tests"""
    manager = TestDataManager()
    return manager.create_test_environment(test_name, ['zk_proofs', 'benchmarks', 'audit_trails'])


if __name__ == '__main__':
    # Example usage
    manager = TestDataManager()
    
    # Create test environment
    test_env = manager.create_test_environment('example_test')
    print(f"Created test environment: {test_env['temp_dir']}")
    
    # List available fixtures
    fixtures = manager.list_available_fixtures()
    print(f"Available fixtures: {fixtures}")
    
    # Clean up
    manager.cleanup_test_environment(test_env)