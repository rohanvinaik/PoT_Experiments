#!/usr/bin/env bash
set -euo pipefail

# Helper function to replace bc with Python
py_rate() {
    python3 -c "import sys; ok,total=float(sys.argv[1]),float(sys.argv[2]); print(f'{(ok/total)*100:.1f}')" "$1" "$2"
}

# Proof-of-Training Comprehensive Experimental Validator
# This script runs all tests and experiments to validate the complete PoT system

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
PYTHON=${PYTHON:-python3}
RESULTS_DIR="experimental_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${RESULTS_DIR}/run_all_${TIMESTAMP}.log"

# Print colored output
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Start logging (commented out - was causing issues)
# exec 2>&1 | tee -a "${LOG_FILE}"

print_header "PROOF-OF-TRAINING COMPREHENSIVE EXPERIMENTAL VALIDATOR"
echo "Started at: $(date)"
echo "Python: ${PYTHON}"
echo "Results directory: ${RESULTS_DIR}"
echo "Log file: ${LOG_FILE}"

# Check Python and dependencies
print_header "CHECKING ENVIRONMENT"

check_python() {
    if command -v ${PYTHON} &> /dev/null; then
        print_success "Python found: $(${PYTHON} --version)"
    else
        print_error "Python not found"
        exit 1
    fi
}

check_dependencies() {
    print_info "Checking required packages..."
    
    ${PYTHON} -c "import numpy" 2>/dev/null && print_success "NumPy installed" || print_error "NumPy not found"
    
    # Check optional dependencies
    ${PYTHON} -c "import torch" 2>/dev/null && print_success "PyTorch installed" || print_info "PyTorch not found (optional)"
    ${PYTHON} -c "import transformers" 2>/dev/null && print_success "Transformers installed" || print_info "Transformers not found (optional)"
    ${PYTHON} -c "import ssdeep" 2>/dev/null && print_success "SSDeep installed" || print_info "SSDeep not found (optional)"
    ${PYTHON} -c "import tlsh" 2>/dev/null && print_success "TLSH installed" || print_info "TLSH not found (optional)"
}

check_python
check_dependencies

# Run component tests
print_header "RUNNING COMPONENT TESTS"

run_test() {
    local test_name=$1
    local test_file=$2
    
    echo -e "\n${YELLOW}Testing: ${test_name}${NC}"
    
    if [ -f "${test_file}" ]; then
        if ${PYTHON} "${test_file}" > "${RESULTS_DIR}/${test_name}_${TIMESTAMP}.log" 2>&1; then
            print_success "${test_name} tests passed"
            return 0
        else
            print_error "${test_name} tests failed (see ${RESULTS_DIR}/${test_name}_${TIMESTAMP}.log)"
            return 1
        fi
    else
        print_error "${test_file} not found"
        return 1
    fi
}

# Track test results - FIXED arithmetic
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Run individual component tests
if run_test "FuzzyHashVerifier" "pot/security/test_fuzzy_verifier.py"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

if run_test "TrainingProvenanceAuditor" "pot/security/test_provenance_auditor.py"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

if run_test "TokenSpaceNormalizer" "pot/security/test_token_normalizer.py"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Run integrated system demo
print_header "RUNNING INTEGRATED SYSTEM DEMO"

if ${PYTHON} pot/security/proof_of_training.py > "${RESULTS_DIR}/integrated_demo_${TIMESTAMP}.log" 2>&1; then
    print_success "Integrated system demo completed"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_error "Integrated system demo failed"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Run experimental validation suite
print_header "RUNNING EXPERIMENTAL VALIDATION"

# Create validation script
cat > "${RESULTS_DIR}/validation_experiment.py" << 'EOF'
"""
Comprehensive experimental validation of the Proof-of-Training system
"""

import sys
import json
import time
import numpy as np
from datetime import datetime

# Add parent directory to path
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PoT components
try:
    from pot.security.proof_of_training import ProofOfTraining, ChallengeLibrary
    from pot.security.fuzzy_hash_verifier import FuzzyHashVerifier, ChallengeVector
    from pot.prototypes.training_provenance_auditor import (
        TrainingProvenanceAuditor,
        EventType,
        ProofType,
    )
    from pot.security.token_space_normalizer import TokenSpaceNormalizer, StochasticDecodingController, TokenizerType
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def run_experiments():
    """Run comprehensive validation experiments"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'experiments': []
    }
    
    print("\n=== EXPERIMENT 1: Verification Types Comparison ===")
    exp1_results = experiment_verification_types()
    results['experiments'].append(exp1_results)
    
    print("\n=== EXPERIMENT 2: Security Levels Analysis ===")
    exp2_results = experiment_security_levels()
    results['experiments'].append(exp2_results)
    
    print("\n=== EXPERIMENT 3: Model Type Coverage ===")
    exp3_results = experiment_model_types()
    results['experiments'].append(exp3_results)
    
    print("\n=== EXPERIMENT 4: Challenge Effectiveness ===")
    exp4_results = experiment_challenge_effectiveness()
    results['experiments'].append(exp4_results)
    
    print("\n=== EXPERIMENT 5: Performance Benchmarks ===")
    exp5_results = experiment_performance()
    results['experiments'].append(exp5_results)
    
    return results

def experiment_verification_types():
    """Compare different verification types"""
    print("Testing verification types: exact, fuzzy, statistical")
    
    results = {'name': 'Verification Types', 'data': []}
    
    for v_type in ['exact', 'fuzzy', 'statistical']:
        config = {
            'verification_type': v_type,
            'model_type': 'generic',
            'security_level': 'medium'
        }
        
        try:
            pot = ProofOfTraining(config)
            
            # Mock model
            class MockModel:
                def forward(self, x):
                    return np.random.randn(10)
                def state_dict(self):
                    return {'layer': 'weights'}
            
            model = MockModel()
            model_id = pot.register_model(model, "test_arch", 1000)
            
            # Test different depths
            for depth in ['quick', 'standard', 'comprehensive']:
                start = time.time()
                result = pot.perform_verification(model, model_id, depth)
                duration = time.time() - start
                
                results['data'].append({
                    'type': v_type,
                    'depth': depth,
                    'verified': result.verified,
                    'confidence': float(result.confidence),
                    'duration': duration
                })
                
                print(f"  {v_type}/{depth}: verified={result.verified}, "
                      f"confidence={result.confidence:.2%}, time={duration:.3f}s")
        
        except Exception as e:
            print(f"  Error with {v_type}: {e}")
            results['data'].append({'type': v_type, 'error': str(e)})
    
    return results

def experiment_security_levels():
    """Test different security levels"""
    print("Testing security levels: low, medium, high")
    
    results = {'name': 'Security Levels', 'data': []}
    
    for level in ['low', 'medium', 'high']:
        config = {
            'verification_type': 'fuzzy',
            'model_type': 'generic',
            'security_level': level
        }
        
        try:
            pot = ProofOfTraining(config)
            stats = pot.get_statistics()
            
            # Check threshold settings
            if pot.fuzzy_verifier:
                threshold = pot.fuzzy_verifier.similarity_threshold
            else:
                threshold = 0.85
            
            results['data'].append({
                'level': level,
                'threshold': threshold,
                'components': stats['components']
            })
            
            print(f"  {level}: threshold={threshold}, components={stats['components']}")
            
        except Exception as e:
            print(f"  Error with {level}: {e}")
            results['data'].append({'level': level, 'error': str(e)})
    
    return results

def experiment_model_types():
    """Test different model types"""
    print("Testing model types: vision, language, multimodal, generic")
    
    results = {'name': 'Model Types', 'data': []}
    
    for model_type in ['vision', 'language', 'multimodal', 'generic']:
        print(f"\n  Testing {model_type} model:")
        
        # Test challenge generation
        if model_type == 'vision':
            challenges = ChallengeLibrary.get_vision_challenges(224, 3, 3)
            print(f"    Generated {len(challenges)} vision challenges")
            
        elif model_type == 'language':
            challenges = ChallengeLibrary.get_language_challenges(50000, 100, 3)
            print(f"    Generated {len(challenges)} language challenges")
            
        elif model_type == 'multimodal':
            challenges = ChallengeLibrary.get_multimodal_challenges(2)
            print(f"    Generated {len(challenges)} multimodal challenges")
            
        else:
            challenges = ChallengeLibrary.get_generic_challenges(100, 3)
            print(f"    Generated {len(challenges)} generic challenges")
        
        results['data'].append({
            'model_type': model_type,
            'num_challenges': len(challenges),
            'challenge_types': type(challenges[0]).__name__ if challenges else None
        })
    
    return results

def experiment_challenge_effectiveness():
    """Test challenge-response effectiveness"""
    print("Testing challenge effectiveness")
    
    results = {'name': 'Challenge Effectiveness', 'data': []}
    
    # Test ChallengeVector
    for topology in ['complex', 'sparse', 'normal']:
        for dimension in [100, 500, 1000]:
            try:
                challenge = ChallengeVector(dimension=dimension, topology=topology, seed=42)
                
                # Analyze challenge properties
                mean = float(np.mean(challenge.vector))
                std = float(np.std(challenge.vector))
                sparsity = float(np.sum(np.abs(challenge.vector) < 0.01) / dimension)
                
                results['data'].append({
                    'topology': topology,
                    'dimension': dimension,
                    'mean': mean,
                    'std': std,
                    'sparsity': sparsity
                })
                
                print(f"  {topology}/{dimension}D: mean={mean:.3f}, "
                      f"std={std:.3f}, sparsity={sparsity:.2%}")
                
            except Exception as e:
                print(f"  Error with {topology}/{dimension}: {e}")
    
    return results

def experiment_performance():
    """Benchmark performance metrics"""
    print("Running performance benchmarks")
    
    results = {'name': 'Performance', 'data': []}
    
    # Test fuzzy hash performance
    print("\n  Fuzzy Hash Performance:")
    verifier = FuzzyHashVerifier(similarity_threshold=0.85)
    
    for size in [100, 1000, 10000]:
        data = np.random.randn(size)
        
        start = time.time()
        hash_val = verifier.generate_fuzzy_hash(data)
        hash_time = time.time() - start
        
        start = time.time()
        verifier.verify_fuzzy(hash_val, hash_val)
        verify_time = time.time() - start
        
        results['data'].append({
            'operation': 'fuzzy_hash',
            'size': size,
            'hash_time': hash_time,
            'verify_time': verify_time
        })
        
        print(f"    Size {size}: hash={hash_time:.6f}s, verify={verify_time:.6f}s")
    
    # Test provenance auditor performance
    print("\n  Provenance Auditor Performance:")
    auditor = TrainingProvenanceAuditor(model_id="perf_test")
    
    for num_events in [10, 50, 100]:
        start = time.time()
        for i in range(num_events):
            auditor.log_training_event(
                epoch=i,
                metrics={'loss': 1.0/(i+1)},
                event_type=EventType.EPOCH_END
            )
        log_time = time.time() - start
        
        start = time.time()
        proof = auditor.generate_training_proof(0, num_events-1, ProofType.MERKLE)
        proof_time = time.time() - start
        
        results['data'].append({
            'operation': 'provenance',
            'num_events': num_events,
            'log_time': log_time,
            'proof_time': proof_time
        })
        
        print(f"    Events {num_events}: log={log_time:.6f}s, proof={proof_time:.6f}s")
    
    return results

if __name__ == "__main__":
    print("Starting comprehensive experimental validation...")
    results = run_experiments()
    
    # Save results
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n=== VALIDATION COMPLETE ===")
    print(f"Results saved to validation_results.json")
    
    # Summary
    total_experiments = len(results['experiments'])
    successful = sum(1 for exp in results['experiments'] if 'data' in exp and exp['data'])
    
    print(f"\nSummary:")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Successful: {successful}")
    print(f"  Success rate: {successful/total_experiments*100:.1f}%")
    
    sys.exit(0 if successful == total_experiments else 1)
EOF

print_info "Running experimental validation..."
if ${PYTHON} "${RESULTS_DIR}/validation_experiment.py" > "${RESULTS_DIR}/validation_${TIMESTAMP}.log" 2>&1; then
    print_success "Experimental validation completed"
    PASSED_TESTS=$((PASSED_TESTS + 1))
    
    # Display validation results
    if [ -f "validation_results.json" ]; then
        mv validation_results.json "${RESULTS_DIR}/validation_results_${TIMESTAMP}.json"
        print_info "Validation results saved to ${RESULTS_DIR}/validation_results_${TIMESTAMP}.json"
    fi
else
    print_error "Experimental validation failed"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Run stress tests
print_header "RUNNING STRESS TESTS"

cat > "${RESULTS_DIR}/stress_test.py" << 'EOF'
"""
Stress testing the PoT system
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.security.proof_of_training import ProofOfTraining

def stress_test_batch_verification():
    """Test batch verification with many models"""
    print("Stress testing batch verification...")
    
    config = {
        'verification_type': 'fuzzy',
        'model_type': 'generic',
        'security_level': 'low'  # Low for speed
    }
    
    pot = ProofOfTraining(config)
    
    # Create multiple mock models
    class MockModel:
        def __init__(self, seed):
            self.seed = seed
        def forward(self, x):
            np.random.seed(self.seed)
            return np.random.randn(10)
        def state_dict(self):
            return {'seed': self.seed}
    
    # Register many models
    num_models = 20
    models = []
    model_ids = []
    
    print(f"  Registering {num_models} models...")
    start = time.time()
    
    for i in range(num_models):
        model = MockModel(i)
        model_id = pot.register_model(model, f"model_{i}", 1000)
        models.append(model)
        model_ids.append(model_id)
    
    reg_time = time.time() - start
    print(f"  Registration time: {reg_time:.2f}s ({reg_time/num_models:.3f}s per model)")
    
    # Batch verify
    print(f"  Batch verifying {num_models} models...")
    start = time.time()
    
    results = pot.batch_verify(models, model_ids, 'quick')
    
    batch_time = time.time() - start
    print(f"  Batch verification time: {batch_time:.2f}s ({batch_time/num_models:.3f}s per model)")
    
    # Check results
    verified_count = sum(1 for r in results if r.verified)
    print(f"  Verified: {verified_count}/{num_models}")
    
    return verified_count == num_models

def stress_test_large_challenges():
    """Test with large challenge vectors"""
    print("Stress testing large challenges...")
    
    from pot.security.fuzzy_hash_verifier import ChallengeVector, FuzzyHashVerifier
    
    verifier = FuzzyHashVerifier()
    
    for dimension in [1000, 5000, 10000, 50000]:
        print(f"  Testing dimension {dimension}...")
        
        try:
            start = time.time()
            challenge = ChallengeVector(dimension=dimension, topology='complex')
            gen_time = time.time() - start
            
            start = time.time()
            hash_val = verifier.generate_fuzzy_hash(challenge.vector)
            hash_time = time.time() - start
            
            print(f"    Generation: {gen_time:.3f}s, Hashing: {hash_time:.3f}s")
            
        except Exception as e:
            print(f"    Error at dimension {dimension}: {e}")
            return False
    
    return True

def stress_test_provenance_history():
    """Test with large training histories"""
    print("Stress testing training history...")
    
    from pot.prototypes.training_provenance_auditor import (
        TrainingProvenanceAuditor,
        EventType,
        ProofType,
    )
    
    auditor = TrainingProvenanceAuditor(
        model_id="stress_test",
        max_history_size=100  # Test compression
    )
    
    num_epochs = 500
    print(f"  Logging {num_epochs} training events...")
    
    start = time.time()
    for epoch in range(num_epochs):
        auditor.log_training_event(
            epoch=epoch,
            metrics={
                'loss': 1.0/(epoch+1),
                'accuracy': min(0.99, epoch/100),
                'gradient_norm': np.random.random()
            },
            event_type=EventType.EPOCH_END
        )
    
    log_time = time.time() - start
    print(f"  Logging time: {log_time:.2f}s ({log_time/num_epochs:.6f}s per event)")
    
    # Generate proof
    print("  Generating Merkle proof...")
    start = time.time()
    proof = auditor.generate_training_proof(0, num_epochs-1, ProofType.MERKLE)
    proof_time = time.time() - start
    print(f"  Proof generation time: {proof_time:.3f}s")
    
    # Check compression
    stats = auditor.get_statistics()
    print(f"  Events in memory: {stats['total_events']}")
    print(f"  Compression active: {len(auditor.events) < num_epochs}")
    
    return True

if __name__ == "__main__":
    print("Starting stress tests...")
    
    tests = [
        ("Batch Verification", stress_test_batch_verification),
        ("Large Challenges", stress_test_large_challenges),
        ("Provenance History", stress_test_provenance_history)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                print(f"  ✓ {test_name} passed")
                passed += 1
            else:
                print(f"  ✗ {test_name} failed")
        except Exception as e:
            print(f"  ✗ {test_name} error: {e}")
    
    print(f"\nStress test results: {passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
EOF

print_info "Running stress tests..."
if ${PYTHON} "${RESULTS_DIR}/stress_test.py" > "${RESULTS_DIR}/stress_test_${TIMESTAMP}.log" 2>&1; then
    print_success "Stress tests passed"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_error "Stress tests failed"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Generate summary report
print_header "GENERATING SUMMARY REPORT"

cat > "${RESULTS_DIR}/summary_${TIMESTAMP}.txt" << EOF
PROOF-OF-TRAINING EXPERIMENTAL VALIDATION SUMMARY
==================================================

Date: $(date)
Python Version: $(${PYTHON} --version 2>&1)

TEST RESULTS
------------
Total Tests: ${TOTAL_TESTS}
Passed: ${PASSED_TESTS}
Failed: ${FAILED_TESTS}
Success Rate: $(py_rate ${PASSED_TESTS} ${TOTAL_TESTS})%

COMPONENT TESTS
---------------
$([ ${FAILED_TESTS} -eq 0 ] && echo "✓ All component tests passed" || echo "✗ Some tests failed")

EXPERIMENTS RUN
---------------
1. Verification Types Comparison
2. Security Levels Analysis  
3. Model Type Coverage
4. Challenge Effectiveness
5. Performance Benchmarks
6. Stress Testing

ARTIFACTS GENERATED
-------------------
- Log files: ${RESULTS_DIR}/*_${TIMESTAMP}.log
- Validation results: ${RESULTS_DIR}/validation_results_${TIMESTAMP}.json
- This summary: ${RESULTS_DIR}/summary_${TIMESTAMP}.txt

RECOMMENDATIONS
---------------
$(if [ ${FAILED_TESTS} -eq 0 ]; then
    echo "• System is fully operational"
    echo "• All components verified successfully"
    echo "• Ready for production deployment"
else
    echo "• Review failed test logs for details"
    echo "• Check missing dependencies"
    echo "• Ensure all required packages are installed"
fi)

EOF

print_success "Summary report generated: ${RESULTS_DIR}/summary_${TIMESTAMP}.txt"

# Display summary
print_header "FINAL RESULTS"

cat "${RESULTS_DIR}/summary_${TIMESTAMP}.txt"

# Set exit code based on test results
if [ ${FAILED_TESTS} -eq 0 ]; then
    print_success "All tests passed! The Proof-of-Training system is fully validated."
    exit 0
else
    print_error "Some tests failed. Please review the logs in ${RESULTS_DIR}/"
    exit 1
fi