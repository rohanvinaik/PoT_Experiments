#!/usr/bin/env python3
"""
Integration Test for API Security

Demonstrates the full flow of TEE attestation and API binding
with attack simulation and performance benchmarking.
"""

import json
import time
import uuid
import hashlib
import sys
import os
from typing import List, Dict, Any
import statistics

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.pot.security.tee_attestation import (
    AttestationType,
    ModelIdentity,
    create_attestation_provider
)
from src.pot.security.attestation_verifier import (
    AttestationVerifier,
    AttestationPolicy,
    VerificationStatus
)
from src.pot.api.secure_binding import (
    APITranscript,
    SecureAPIBinder,
    BindingPolicy,
    BindingMode
)


def test_full_tee_flow(provider_type: AttestationType):
    """Test complete TEE attestation and binding flow"""
    print(f"\n{'='*60}")
    print(f"Testing {provider_type.value.upper()} Provider")
    print(f"{'='*60}")
    
    # 1. Setup
    provider = create_attestation_provider(provider_type, {})
    
    model_identity = ModelIdentity(
        model_hash=hashlib.sha256(f"model_{provider_type.value}".encode()).hexdigest(),
        model_name=f"test-model-{provider_type.value}",
        version="1.0.0",
        provider="test",
        architecture="transformer",
        parameter_count=1000000
    )
    
    # 2. Generate attestation
    nonce = str(uuid.uuid4())
    start_time = time.time()
    attestation = provider.generate_attestation(model_identity, nonce)
    attestation_time = (time.time() - start_time) * 1000
    
    print(f"1. Attestation Generation: {attestation_time:.2f} ms")
    
    # 3. Verify attestation
    policy = AttestationPolicy.relaxed_policy()
    policy.allowed_providers.add(provider_type)
    verifier = AttestationVerifier(policy)
    
    start_time = time.time()
    result = verifier.verify_attestation(attestation, model_identity, nonce)
    verification_time = (time.time() - start_time) * 1000
    
    print(f"2. Attestation Verification: {verification_time:.2f} ms")
    print(f"   Status: {result.status.value}")
    print(f"   Valid: {result.is_valid()}")
    
    # 4. Create and bind API transcript
    binder = SecureAPIBinder(BindingPolicy.relaxed_policy())
    
    transcript = APITranscript(
        transcript_id=str(uuid.uuid4()),
        timestamp=time.time(),
        endpoint='/api/inference',
        method='POST',
        request={'prompt': 'test'},
        response={'text': 'response'},
        latency_ms=100.0
    )
    
    start_time = time.time()
    bound_transcript = binder.bind_transcript(
        transcript,
        model_identity,
        attestation,
        verify_immediately=False
    )
    binding_time = (time.time() - start_time) * 1000
    
    print(f"3. Transcript Binding: {binding_time:.2f} ms")
    
    # 5. Create evidence bundle
    start_time = time.time()
    bundle = binder.create_evidence_bundle([transcript.transcript_id])
    bundle_time = (time.time() - start_time) * 1000
    
    print(f"4. Evidence Bundle Creation: {bundle_time:.2f} ms")
    print(f"   Bundle Size: {len(json.dumps(bundle))} bytes")
    
    total_time = attestation_time + verification_time + binding_time + bundle_time
    print(f"\nTotal Processing Time: {total_time:.2f} ms")
    
    return {
        'provider': provider_type.value,
        'attestation_time': attestation_time,
        'verification_time': verification_time,
        'binding_time': binding_time,
        'bundle_time': bundle_time,
        'total_time': total_time,
        'valid': result.is_valid()
    }


def test_replay_attack_detection():
    """Test replay attack detection"""
    print(f"\n{'='*60}")
    print("REPLAY ATTACK DETECTION TEST")
    print(f"{'='*60}")
    
    provider = create_attestation_provider(AttestationType.MOCK, {})
    model_identity = ModelIdentity(
        model_hash="test_hash",
        model_name="test-model",
        version="1.0.0",
        provider="test",
        architecture="transformer",
        parameter_count=1000000
    )
    
    # Generate attestation
    nonce = str(uuid.uuid4())
    attestation = provider.generate_attestation(model_identity, nonce)
    
    # Create verifier
    verifier = AttestationVerifier(AttestationPolicy.relaxed_policy())
    
    # First verification should succeed
    result1 = verifier.verify_attestation(attestation, model_identity, nonce)
    print(f"1. First verification: {result1.status.value}")
    
    # Second verification with same attestation should fail (replay)
    result2 = verifier.verify_attestation(attestation, model_identity, nonce)
    print(f"2. Replay verification: {result2.status.value}")
    
    # Check for replay attack detection
    from src.pot.security.attestation_verifier import VerificationFailureReason
    replay_detected = VerificationFailureReason.REPLAY_ATTACK in result2.failures
    print(f"3. Replay attack detected: {replay_detected}")
    
    return replay_detected


def test_substitution_attack_detection():
    """Test model substitution attack detection"""
    print(f"\n{'='*60}")
    print("SUBSTITUTION ATTACK DETECTION TEST")
    print(f"{'='*60}")
    
    provider = create_attestation_provider(AttestationType.MOCK, {})
    
    # Original model
    model1 = ModelIdentity(
        model_hash="model1_hash",
        model_name="legitimate-model",
        version="1.0.0",
        provider="trusted",
        architecture="transformer",
        parameter_count=1000000
    )
    
    # Substituted model (attacker's model)
    model2 = ModelIdentity(
        model_hash="model2_hash",
        model_name="malicious-model",
        version="1.0.0",
        provider="attacker",
        architecture="transformer",
        parameter_count=1000000
    )
    
    # Generate attestation for model1
    nonce = str(uuid.uuid4())
    attestation = provider.generate_attestation(model1, nonce)
    
    # Try to verify with model2 (substitution attack)
    verifier = AttestationVerifier(AttestationPolicy.relaxed_policy())
    result = verifier.verify_attestation(attestation, model2, nonce)
    
    print(f"1. Attestation generated for: {model1.model_name}")
    print(f"2. Verification attempted with: {model2.model_name}")
    print(f"3. Verification result: {result.status.value}")
    print(f"4. Attack detected: {not result.is_valid()}")
    
    return not result.is_valid()


def benchmark_providers():
    """Benchmark different attestation providers"""
    print(f"\n{'='*60}")
    print("PERFORMANCE BENCHMARK")
    print(f"{'='*60}")
    
    providers_to_test = [
        AttestationType.SGX,
        AttestationType.SEV,
        AttestationType.NITRO,
        AttestationType.VENDOR,
        AttestationType.MOCK
    ]
    
    results = []
    iterations = 10
    
    for provider_type in providers_to_test:
        print(f"\nBenchmarking {provider_type.value}...")
        
        times = []
        for i in range(iterations):
            result = test_full_tee_flow(provider_type)
            times.append(result['total_time'])
        
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        
        results.append({
            'provider': provider_type.value,
            'avg_time': avg_time,
            'std_dev': std_dev,
            'min_time': min(times),
            'max_time': max(times)
        })
    
    # Display results
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"{'Provider':<15} {'Avg (ms)':<12} {'Std Dev':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-" * 60)
    
    for result in sorted(results, key=lambda x: x['avg_time']):
        print(f"{result['provider']:<15} {result['avg_time']:<12.2f} "
              f"{result['std_dev']:<12.2f} {result['min_time']:<12.2f} "
              f"{result['max_time']:<12.2f}")
    
    return results


def test_batch_verification():
    """Test batch attestation verification"""
    print(f"\n{'='*60}")
    print("BATCH VERIFICATION TEST")
    print(f"{'='*60}")
    
    batch_size = 100
    provider = create_attestation_provider(AttestationType.MOCK, {})
    verifier = AttestationVerifier(AttestationPolicy.relaxed_policy())
    
    attestations = []
    model_identities = []
    nonces = []
    
    # Generate batch
    print(f"1. Generating {batch_size} attestations...")
    start_time = time.time()
    
    for i in range(batch_size):
        model = ModelIdentity(
            model_hash=f"model_{i}",
            model_name=f"model-{i}",
            version="1.0.0",
            provider="test",
            architecture="transformer",
            parameter_count=1000000
        )
        nonce = str(uuid.uuid4())
        attestation = provider.generate_attestation(model, nonce)
        
        attestations.append(attestation)
        model_identities.append(model)
        nonces.append(nonce)
    
    generation_time = time.time() - start_time
    print(f"   Generation time: {generation_time:.2f} seconds")
    print(f"   Per attestation: {(generation_time/batch_size)*1000:.2f} ms")
    
    # Batch verify
    print(f"\n2. Batch verifying {batch_size} attestations...")
    start_time = time.time()
    
    results = verifier.batch_verify(attestations, model_identities, nonces)
    
    verification_time = time.time() - start_time
    print(f"   Verification time: {verification_time:.2f} seconds")
    print(f"   Per attestation: {(verification_time/batch_size)*1000:.2f} ms")
    
    # Summary
    summary = verifier.get_verification_summary(results)
    print(f"\n3. Verification Summary:")
    print(f"   Total: {summary['total']}")
    print(f"   Valid: {summary['valid']}")
    print(f"   Success Rate: {summary['success_rate']*100:.1f}%")
    
    return summary


def test_evidence_bundle_integrity():
    """Test evidence bundle integrity validation"""
    print(f"\n{'='*60}")
    print("EVIDENCE BUNDLE INTEGRITY TEST")
    print(f"{'='*60}")
    
    # Setup
    provider = create_attestation_provider(AttestationType.VENDOR, {})
    model_identity = ModelIdentity(
        model_hash="test_model",
        model_name="test-model",
        version="1.0.0",
        provider="test",
        architecture="transformer",
        parameter_count=1000000
    )
    
    binder = SecureAPIBinder(BindingPolicy.relaxed_policy())
    
    # Create multiple transcripts
    print("1. Creating transcripts and bindings...")
    transcript_ids = []
    
    for i in range(5):
        transcript = APITranscript(
            transcript_id=str(uuid.uuid4()),
            timestamp=time.time(),
            endpoint=f'/api/endpoint_{i}',
            method='POST',
            request={'data': f'request_{i}'},
            response={'data': f'response_{i}'},
            latency_ms=100.0 + i * 10
        )
        
        nonce = str(uuid.uuid4())
        attestation = provider.generate_attestation(model_identity, nonce)
        
        bound = binder.bind_transcript(
            transcript,
            model_identity,
            attestation,
            verify_immediately=False
        )
        
        transcript_ids.append(transcript.transcript_id)
    
    # Create bundle
    print("2. Creating evidence bundle...")
    bundle = binder.create_evidence_bundle(transcript_ids)
    
    # Validate bundle
    print("3. Validating bundle integrity...")
    valid, errors = binder.validate_evidence_bundle(bundle)
    print(f"   Valid: {valid}")
    if errors:
        print(f"   Errors: {errors}")
    
    # Tamper with bundle
    print("\n4. Testing tamper detection...")
    tampered_bundle = json.loads(json.dumps(bundle))
    tampered_bundle['transcripts'][0]['status'] = 'tampered'
    
    valid_tampered, errors_tampered = binder.validate_evidence_bundle(tampered_bundle)
    print(f"   Tampered bundle valid: {valid_tampered}")
    print(f"   Tamper detected: {not valid_tampered}")
    
    return valid and not valid_tampered


def main():
    """Run integration tests"""
    print("\n" + "🔒" * 30)
    print("API SECURITY INTEGRATION TESTS")
    print("🔒" * 30)
    
    test_results = {}
    
    # Test 1: Full TEE flow for each provider
    print("\n[TEST 1] Full TEE Flow")
    providers = [AttestationType.SGX, AttestationType.SEV, 
                AttestationType.NITRO, AttestationType.VENDOR, 
                AttestationType.MOCK]
    
    for provider in providers:
        result = test_full_tee_flow(provider)
        test_results[f"tee_flow_{provider.value}"] = result['valid']
    
    # Test 2: Attack detection
    print("\n[TEST 2] Security Tests")
    test_results['replay_attack_detection'] = test_replay_attack_detection()
    test_results['substitution_attack_detection'] = test_substitution_attack_detection()
    
    # Test 3: Performance benchmark
    print("\n[TEST 3] Performance Benchmark")
    benchmark_results = benchmark_providers()
    test_results['benchmark_completed'] = len(benchmark_results) > 0
    
    # Test 4: Batch verification
    print("\n[TEST 4] Batch Operations")
    batch_summary = test_batch_verification()
    test_results['batch_verification'] = batch_summary['success_rate'] > 0.95
    
    # Test 5: Evidence bundle integrity
    print("\n[TEST 5] Evidence Bundle Integrity")
    test_results['bundle_integrity'] = test_evidence_bundle_integrity()
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for v in test_results.values() if v)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<40} {status}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*60}")
    
    if passed == total:
        print("\n🎉 All integration tests passed successfully!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the results.")
    
    return test_results


if __name__ == '__main__':
    main()