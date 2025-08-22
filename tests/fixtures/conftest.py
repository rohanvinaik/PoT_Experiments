#!/usr/bin/env python3
"""
Pytest Configuration and Fixtures

Provides shared fixtures and configuration for the test suite.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.fixtures.test_data_manager import TestDataManager


@pytest.fixture(scope="session")
def test_data_manager() -> TestDataManager:
    """Provide a test data manager for the test session"""
    return TestDataManager()


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for each test function"""
    temp_path = Path(tempfile.mkdtemp(prefix='pot_test_'))
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


@pytest.fixture(scope="function")
def model_test_data(test_data_manager: TestDataManager) -> Generator[Dict[str, Any], None, None]:
    """Provide model verification test data"""
    test_env = test_data_manager.create_test_environment('model_test', ['models', 'benchmarks'])
    try:
        yield test_env
    finally:
        test_data_manager.cleanup_test_environment(test_env)


@pytest.fixture(scope="function")
def security_test_data(test_data_manager: TestDataManager) -> Generator[Dict[str, Any], None, None]:
    """Provide security test data"""
    test_env = test_data_manager.create_test_environment('security_test', ['security', 'audit_trails'])
    try:
        yield test_env
    finally:
        test_data_manager.cleanup_test_environment(test_env)


@pytest.fixture(scope="function")
def benchmark_test_data(test_data_manager: TestDataManager) -> Generator[Dict[str, Any], None, None]:
    """Provide benchmark test data"""
    test_env = test_data_manager.create_test_environment('benchmark_test', ['benchmarks'])
    try:
        yield test_env
    finally:
        test_data_manager.cleanup_test_environment(test_env)


@pytest.fixture(scope="function")
def zk_test_data(test_data_manager: TestDataManager) -> Generator[Dict[str, Any], None, None]:
    """Provide ZK proof test data"""
    test_env = test_data_manager.create_test_environment('zk_test', ['zk_proofs', 'benchmarks'])
    try:
        yield test_env
    finally:
        test_data_manager.cleanup_test_environment(test_env)


@pytest.fixture(scope="function")
def evidence_test_data(test_data_manager: TestDataManager) -> Generator[Dict[str, Any], None, None]:
    """Provide evidence bundle test data"""
    test_env = test_data_manager.create_test_environment('evidence_test', ['evidence_bundles', 'audit_trails'])
    try:
        yield test_env
    finally:
        test_data_manager.cleanup_test_environment(test_env)


@pytest.fixture(scope="function")
def full_test_environment(test_data_manager: TestDataManager) -> Generator[Dict[str, Any], None, None]:
    """Provide complete test environment with all categories"""
    test_env = test_data_manager.create_test_environment('full_test')
    try:
        yield test_env
    finally:
        test_data_manager.cleanup_test_environment(test_env)


# Mock response fixtures
@pytest.fixture
def mock_gpt2_responses():
    """Mock GPT-2 responses for testing"""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times.",
        "To be or not to be, that is the question.",
        "Space: the final frontier."
    ]


@pytest.fixture
def mock_distilgpt2_responses():
    """Mock DistilGPT-2 responses for testing"""
    return [
        "The quick brown fox leaps over the lazy dog.",
        "In a burrow in the earth there dwelt a hobbit.",
        "It was the greatest of times, it was the most difficult of times.",
        "To exist or not to exist, that is the inquiry.",
        "Space: the ultimate boundary."
    ]


@pytest.fixture
def mock_benchmark_data():
    """Mock benchmark data for testing"""
    return {
        'timestamp': '2024-01-01T00:00:00Z',
        'test_run_id': 'test_run_001',
        'metrics': {
            'verification_time_ms': 1500.0,
            'memory_usage_mb': 512.0,
            'cpu_usage_percent': 45.0,
            'accuracy': 0.95,
            'throughput_qps': 5.0
        },
        'system_info': {
            'cpu_cores': 8,
            'memory_gb': 16,
            'os': 'linux'
        }
    }


@pytest.fixture
def mock_security_scan_results():
    """Mock security scan results for testing"""
    return {
        'scan_timestamp': '2024-01-01T00:00:00Z',
        'scanner_version': '1.0.0',
        'vulnerabilities': [
            {
                'id': 'VULN-0001',
                'severity': 'medium',
                'description': 'Test vulnerability 1',
                'file': 'test_file_1.py',
                'line': 42,
                'recommendation': 'Fix vulnerability 1'
            },
            {
                'id': 'VULN-0002',
                'severity': 'low',
                'description': 'Test vulnerability 2',
                'file': 'test_file_2.py',
                'line': 123,
                'recommendation': 'Fix vulnerability 2'
            }
        ]
    }


@pytest.fixture
def mock_zk_proof_data():
    """Mock ZK proof data for testing"""
    return {
        'circuit_size': 'small',
        'constraints': 10000,
        'proof_generation_time_ms': 1500.0,
        'proof_size_bytes': 5432,
        'verification_time_ms': 25.0,
        'memory_usage_mb': 128.0,
        'success': True,
        'proof_hex': '0123456789abcdef' * 8
    }


# Configuration fixtures
@pytest.fixture
def test_config():
    """Provide test configuration"""
    return {
        'timeout_seconds': 30,
        'max_retries': 3,
        'test_models': ['gpt2', 'distilgpt2'],
        'benchmark_iterations': 5,
        'regression_threshold': 10.0
    }


# Environment setup
def pytest_configure(config):
    """Configure pytest environment"""
    # Set test environment variables
    os.environ['POT_TEST_MODE'] = 'true'
    os.environ['POT_CI_MODE'] = 'true'
    
    # Create test output directory
    test_output_dir = Path('test_outputs')
    test_output_dir.mkdir(exist_ok=True)


def pytest_unconfigure(config):
    """Clean up after pytest run"""
    # Clean up environment variables
    os.environ.pop('POT_TEST_MODE', None)
    os.environ.pop('POT_CI_MODE', None)


# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "zk: marks tests as zero-knowledge proof tests"
    )


# Pytest collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add markers based on test file location
        test_file = str(item.fspath)
        
        if 'test_security' in test_file or '/security/' in test_file:
            item.add_marker(pytest.mark.security)
        
        if 'test_performance' in test_file or '/benchmarks/' in test_file:
            item.add_marker(pytest.mark.performance)
        
        if 'test_zk' in test_file or '/zk/' in test_file:
            item.add_marker(pytest.mark.zk)
        
        if 'test_integration' in test_file or '/integration/' in test_file:
            item.add_marker(pytest.mark.integration)
        elif 'test_unit' in test_file or '/unit/' in test_file:
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ['slow', 'benchmark', 'e2e', 'integration']):
            item.add_marker(pytest.mark.slow)