#!/usr/bin/env python3
"""
TEE Attestation Validation Script

Quick script to test TEE attestation and API security features
"""

import sys
import os
import argparse
import json
import time
import uuid
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pot.security.tee_attestation import (
    AttestationType,
    ModelIdentity,
    create_attestation_provider
)
from src.pot.security.attestation_verifier import (
    AttestationVerifier,
    AttestationPolicy
)
from src.pot.api.secure_binding import (
    SecureAPIBinder,
    BindingPolicy,
    APITranscript
)
from src.pot.config.api_security import (
    create_default_config,
    create_test_config
)


def test_tee_attestation(provider_type: str = 'mock', policy_level: str = 'moderate'):
    """Test TEE attestation with specified provider"""
    
    print("=" * 60)
    print(f"TEE ATTESTATION TEST - Provider: {provider_type.upper()}")
    print("=" * 60)
    
    # Create provider
    provider = create_attestation_provider(AttestationType(provider_type), {})
    print(f"✓ Created {provider_type} attestation provider")
    
    # Create model identity
    model = ModelIdentity(
        model_hash="test_model_hash_123",
        model_name="test-model",
        version="1.0.0",
        provider="test-provider",
        architecture="transformer",
        parameter_count=175000000
    )
    print(f"✓ Created model identity: {model.model_name}")
    
    # Generate attestation
    nonce = str(uuid.uuid4())
    start_time = time.time()
    attestation = provider.generate_attestation(model, nonce)
    gen_time = (time.time() - start_time) * 1000
    print(f"✓ Generated attestation in {gen_time:.2f} ms")
    
    # Create verifier with appropriate policy
    if policy_level == 'strict':
        policy = AttestationPolicy.default_policy()
    elif policy_level == 'relaxed':
        policy = AttestationPolicy.relaxed_policy()
    else:
        policy = AttestationPolicy.relaxed_policy()
        policy.required_security_level = attestation.security_level
    
    policy.allowed_providers.add(AttestationType(provider_type))
    verifier = AttestationVerifier(policy)
    print(f"✓ Created verifier with {policy_level} policy")
    
    # Verify attestation
    start_time = time.time()
    result = verifier.verify_attestation(attestation, model, nonce)
    verify_time = (time.time() - start_time) * 1000
    
    print(f"✓ Verification completed in {verify_time:.2f} ms")
    print(f"  Status: {result.status.value}")
    print(f"  Valid: {result.is_valid()}")
    
    if result.failures:
        print("  Failures:")
        for failure in result.failures:
            print(f"    - {failure.value}")
    
    # Test API binding
    print("\n" + "-" * 60)
    print("API BINDING TEST")
    print("-" * 60)
    
    binder = SecureAPIBinder(BindingPolicy.relaxed_policy())
    
    # Create sample transcript
    transcript = APITranscript(
        transcript_id=str(uuid.uuid4()),
        timestamp=time.time(),
        endpoint='/api/inference',
        method='POST',
        request={'prompt': 'test prompt'},
        response={'text': 'test response'},
        latency_ms=100.0
    )
    
    # Bind transcript
    start_time = time.time()
    bound = binder.bind_transcript(
        transcript,
        model,
        attestation,
        verify_immediately=False
    )
    bind_time = (time.time() - start_time) * 1000
    print(f"✓ Bound transcript in {bind_time:.2f} ms")
    
    # Verify binding
    status = binder.verify_binding(bound)
    print(f"✓ Binding status: {status.value}")
    
    # Create evidence bundle
    bundle = binder.create_evidence_bundle([transcript.transcript_id])
    print(f"✓ Created evidence bundle with {len(bundle['transcripts'])} transcript(s)")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Provider: {provider_type}")
    print(f"Security Level: {attestation.security_level.value}")
    print(f"Attestation Valid: {result.is_valid()}")
    print(f"Binding Valid: {status.value == 'valid'}")
    print(f"Total Time: {(gen_time + verify_time + bind_time):.2f} ms")
    
    return result.is_valid() and status.value == 'valid'


def test_all_providers():
    """Test all available TEE providers"""
    providers = ['mock', 'sgx', 'sev', 'nitro', 'vendor']
    results = {}
    
    print("\n" + "🔒" * 30)
    print("TESTING ALL TEE PROVIDERS")
    print("🔒" * 30 + "\n")
    
    for provider in providers:
        try:
            success = test_tee_attestation(provider, 'relaxed')
            results[provider] = 'PASSED' if success else 'FAILED'
        except Exception as e:
            print(f"Error testing {provider}: {e}")
            results[provider] = 'ERROR'
        print()
    
    # Print summary
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    for provider, status in results.items():
        icon = "✅" if status == 'PASSED' else "❌"
        print(f"{icon} {provider.upper():<10} {status}")
    
    passed = sum(1 for s in results.values() if s == 'PASSED')
    print(f"\nTotal: {passed}/{len(results)} providers passed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Test TEE attestation and API security features'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        choices=['sgx', 'sev', 'nitro', 'vendor', 'mock', 'all'],
        default='mock',
        help='TEE provider to test (default: mock)'
    )
    
    parser.add_argument(
        '--policy',
        type=str,
        choices=['strict', 'moderate', 'relaxed'],
        default='moderate',
        help='Attestation policy level (default: moderate)'
    )
    
    parser.add_argument(
        '--run-examples',
        action='store_true',
        help='Run example scripts'
    )
    
    parser.add_argument(
        '--integration-test',
        action='store_true',
        help='Run full integration test'
    )
    
    args = parser.parse_args()
    
    if args.run_examples:
        # Run example scripts
        print("Running vendor commitment example...")
        os.system('python examples/api_security/vendor_commitment_example.py')
        print("\nRunning integration tests...")
        os.system('python examples/api_security/integration_test.py')
    elif args.integration_test:
        # Run integration test
        os.system('python examples/api_security/integration_test.py')
    elif args.provider == 'all':
        # Test all providers
        test_all_providers()
    else:
        # Test specific provider
        success = test_tee_attestation(args.provider, args.policy)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()