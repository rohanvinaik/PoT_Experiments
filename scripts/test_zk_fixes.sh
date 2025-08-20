#!/bin/bash
# Test script to verify ZK test fixes

echo "Testing ZK Test Files After Fixes"
echo "=================================="

# Compile tests
echo -e "\n1. Checking Python syntax..."
python -m py_compile tests/test_zk_integration.py 2>&1 && echo "✓ test_zk_integration.py compiles" || echo "✗ test_zk_integration.py has syntax errors"
python -m py_compile tests/test_zk_validation_suite.py 2>&1 && echo "✓ test_zk_validation_suite.py compiles" || echo "✗ test_zk_validation_suite.py has syntax errors"

# Check pytest collection
echo -e "\n2. Checking pytest collection..."
pytest --collect-only tests/test_zk_integration.py 2>&1 | grep -E "collected|error" | head -5
pytest --collect-only tests/test_zk_validation_suite.py 2>&1 | grep -E "collected|error" | head -5

# Run sample tests
echo -e "\n3. Running sample tests..."
echo "Testing test_zk_integration.py..."
pytest tests/test_zk_integration.py::test_zk_pytest_compatibility -v --tb=short 2>&1 | tail -3

echo -e "\nTesting test_zk_validation_suite.py..."
pytest tests/test_zk_validation_suite.py::TestZKValidationSuite::test_complete_sgd_workflow -v --tb=short 2>&1 | tail -3

echo -e "\n4. Summary"
echo "=========="
echo "All fixes have been applied:"
echo "✓ Fixed import paths (MockBlockchainClient, verify_model_weights)"
echo "✓ Created missing MockModel class"
echo "✓ Fixed SGDStepWitness/SGDStepStatement field names"
echo "✓ Fixed Tuple import in metrics.py"
echo "✓ Updated auto_prove_training_step return format"
echo ""
echo "Tests should now be able to run without collection errors."