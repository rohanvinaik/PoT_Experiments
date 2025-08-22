#!/usr/bin/env python3
"""
Vendor Commitment Example

Demonstrates how to use vendor commitments for API-based model verification
when hardware TEE is not available.
"""

import json
import time
import uuid
import hashlib
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.pot.security.tee_attestation import (
    AttestationType,
    ModelIdentity,
    VendorCommitmentProvider
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
from src.pot.config.api_security import (
    APISecurityConfig,
    VendorCommitmentSchema,
    create_default_config
)


def create_vendor_commitment_example():
    """Create example vendor commitment attestation"""
    print("=" * 60)
    print("VENDOR COMMITMENT ATTESTATION EXAMPLE")
    print("=" * 60)
    
    # 1. Create vendor configuration
    vendor_config = {
        'vendor_name': 'ExampleAI',
        'vendor_key_id': 'example_key_2024',
        'api_endpoint': 'https://api.example-ai.com',
        'api_version': 'v2'
    }
    
    # 2. Create vendor commitment provider
    provider = VendorCommitmentProvider(vendor_config)
    
    print(f"\n1. Vendor Provider Initialized:")
    print(f"   Vendor: {vendor_config['vendor_name']}")
    print(f"   Endpoint: {vendor_config['api_endpoint']}")
    print(f"   API Version: {vendor_config['api_version']}")
    
    # 3. Create model identity
    model_identity = ModelIdentity(
        model_hash=hashlib.sha256(b"llama-2-7b-weights").hexdigest(),
        model_name="llama-2-7b-chat",
        version="2.0.1",
        provider="ExampleAI",
        architecture="transformer",
        parameter_count=7_000_000_000,
        training_hash=hashlib.sha256(b"training_data_v2").hexdigest(),
        metadata={
            'fine_tuned': True,
            'base_model': 'llama-2-7b',
            'training_date': '2024-01-15'
        }
    )
    
    print(f"\n2. Model Identity Created:")
    print(f"   Model: {model_identity.model_name} v{model_identity.version}")
    print(f"   Parameters: {model_identity.parameter_count:,}")
    print(f"   Identity Hash: {model_identity.compute_identity_hash()[:16]}...")
    
    # 4. Generate vendor attestation
    nonce = str(uuid.uuid4())
    attestation = provider.generate_attestation(
        model_identity,
        nonce,
        additional_data={
            'deployment_region': 'us-east-1',
            'compliance': ['SOC2', 'ISO27001']
        }
    )
    
    print(f"\n3. Attestation Generated:")
    print(f"   Type: {attestation.provider_type.value}")
    print(f"   Security Level: {attestation.security_level.value}")
    print(f"   Nonce: {nonce[:8]}...")
    
    # 5. Display commitment details
    commitment_doc = json.loads(attestation.quote)
    print(f"\n4. Vendor Commitment Details:")
    print(f"   Issuer: {commitment_doc['issuer']['name']}")
    print(f"   Valid Until: {time.ctime(commitment_doc['expires_at'])}")
    print(f"   Signature Algorithm: {commitment_doc['algorithm']}")
    
    commitment = commitment_doc['commitment']
    guarantees = commitment.get('guarantees', {})
    print(f"\n   Guarantees:")
    for key, value in guarantees.items():
        print(f"     - {key}: {value}")
    
    return attestation, model_identity, nonce


def verify_vendor_commitment(attestation, model_identity, nonce):
    """Verify vendor commitment attestation"""
    print("\n" + "=" * 60)
    print("VERIFICATION PROCESS")
    print("=" * 60)
    
    # Create verifier with relaxed policy for vendor commitments
    policy = AttestationPolicy.relaxed_policy()
    policy.allowed_providers.add(AttestationType.VENDOR)
    verifier = AttestationVerifier(policy)
    
    print("\n1. Verification Policy:")
    print(f"   Security Level: {policy.required_security_level.value}")
    print(f"   Max Age: {policy.max_attestation_age} seconds")
    print(f"   Certificate Validation: {policy.enforce_certificate_validation}")
    
    # Verify attestation
    result = verifier.verify_attestation(
        attestation,
        model_identity,
        nonce
    )
    
    print(f"\n2. Verification Result:")
    print(f"   Status: {result.status.value}")
    print(f"   Valid: {result.is_valid()}")
    
    if result.failures:
        print(f"   Failures:")
        for failure in result.failures:
            print(f"     - {failure.value}")
    
    if result.warnings:
        print(f"   Warnings:")
        for warning in result.warnings:
            print(f"     - {warning}")
    
    return result


def demonstrate_api_binding(attestation, model_identity):
    """Demonstrate API transcript binding with vendor commitment"""
    print("\n" + "=" * 60)
    print("API TRANSCRIPT BINDING")
    print("=" * 60)
    
    # Create binder with vendor commitment policy
    policy = BindingPolicy(
        mode=BindingMode.VENDOR_COMMITMENT,
        require_attestation=True,
        max_binding_age=7200,
        verify_on_bind=False,
        store_full_transcript=True,
        hash_sensitive_data=False
    )
    
    binder = SecureAPIBinder(policy)
    
    print("\n1. Binding Policy:")
    print(f"   Mode: {policy.mode.value}")
    print(f"   Require Attestation: {policy.require_attestation}")
    print(f"   Max Binding Age: {policy.max_binding_age} seconds")
    
    # Create sample API transcript
    transcript = APITranscript(
        transcript_id=str(uuid.uuid4()),
        timestamp=time.time(),
        endpoint='/v2/chat/completions',
        method='POST',
        request={
            'model': model_identity.model_name,
            'messages': [
                {'role': 'user', 'content': 'What is the capital of France?'}
            ],
            'temperature': 0.7
        },
        response={
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': 'The capital of France is Paris.'
                }
            }],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 7,
                'total_tokens': 17
            }
        },
        latency_ms=245.3,
        metadata={
            'region': 'us-east-1',
            'model_version': model_identity.version
        }
    )
    
    print(f"\n2. API Transcript Created:")
    print(f"   ID: {transcript.transcript_id[:8]}...")
    print(f"   Endpoint: {transcript.endpoint}")
    print(f"   Latency: {transcript.latency_ms:.1f} ms")
    
    # Bind transcript
    bound_transcript = binder.bind_transcript(
        transcript,
        model_identity,
        attestation,
        verify_immediately=False
    )
    
    print(f"\n3. Transcript Bound:")
    print(f"   Binding Signature: {bound_transcript.binding_signature[:16]}...")
    print(f"   Binding Nonce: {bound_transcript.binding_nonce[:8]}...")
    print(f"   Timestamp: {time.ctime(bound_transcript.binding_timestamp)}")
    
    # Verify binding
    status = binder.verify_binding(bound_transcript)
    print(f"\n4. Binding Verification:")
    print(f"   Status: {status.value}")
    
    # Create evidence bundle
    bundle = binder.create_evidence_bundle(
        [transcript.transcript_id],
        include_full_transcripts=False
    )
    
    print(f"\n5. Evidence Bundle Created:")
    print(f"   Transcripts: {bundle['summary']['total_transcripts']}")
    print(f"   Valid: {bundle['summary']['valid_bindings']}")
    print(f"   Bundle Signature: {bundle['signature'][:16]}...")
    
    return binder, bundle


def demonstrate_vendor_schema_validation():
    """Demonstrate vendor commitment schema validation"""
    print("\n" + "=" * 60)
    print("VENDOR SCHEMA VALIDATION")
    print("=" * 60)
    
    # Create vendor schema
    schema = VendorCommitmentSchema(
        vendor_name='ExampleAI',
        api_version='v2',
        endpoint='https://api.example-ai.com',
        key_id='example_key_2024',
        required_fields=['model', 'timestamp', 'signature', 'guarantees'],
        optional_fields=['metadata', 'compliance'],
        signature_algorithm='HMAC-SHA256',
        certificate_url='https://api.example-ai.com/certs',
        validation_rules={
            'timestamp': {'type': float, 'min': 0},
            'model': {'type': str}
        }
    )
    
    print("\n1. Vendor Schema:")
    print(f"   Vendor: {schema.vendor_name}")
    print(f"   Required Fields: {', '.join(schema.required_fields)}")
    print(f"   Signature Algorithm: {schema.signature_algorithm}")
    
    # Create sample commitment
    commitment = {
        'vendor': 'ExampleAI',
        'timestamp': time.time(),
        'model': {
            'name': 'llama-2-7b-chat',
            'version': '2.0.1'
        },
        'signature': 'abc123...',
        'guarantees': {
            'model_immutability': True,
            'audit_trail': True
        }
    }
    
    # Validate commitment
    valid, errors = schema.validate_commitment(commitment)
    
    print(f"\n2. Validation Result:")
    print(f"   Valid: {valid}")
    if errors:
        print(f"   Errors:")
        for error in errors:
            print(f"     - {error}")
    else:
        print(f"   No errors found")
    
    return schema


def save_configuration_example():
    """Save API security configuration example"""
    print("\n" + "=" * 60)
    print("CONFIGURATION MANAGEMENT")
    print("=" * 60)
    
    # Create configuration
    config = create_default_config()
    
    # Customize for vendor commitments
    config.default_provider = AttestationType.VENDOR
    config.binding_policy.mode = BindingMode.VENDOR_COMMITMENT
    
    # Save to file
    config_path = Path('examples/api_security/api_config_example.json')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(config_path)
    
    print(f"\n1. Configuration Saved:")
    print(f"   Path: {config_path}")
    print(f"   Default Provider: {config.default_provider.value}")
    print(f"   Binding Mode: {config.binding_policy.mode.value}")
    
    # Load and validate
    loaded_config = APISecurityConfig.load(config_path)
    valid, errors = validate_config(loaded_config)
    
    print(f"\n2. Configuration Validation:")
    print(f"   Valid: {valid}")
    if errors:
        print(f"   Errors:")
        for error in errors:
            print(f"     - {error}")
    
    return config_path


def main():
    """Run vendor commitment examples"""
    print("\n" + "🔐" * 30)
    print("VENDOR COMMITMENT API SECURITY EXAMPLES")
    print("🔐" * 30)
    
    # 1. Create vendor commitment
    attestation, model_identity, nonce = create_vendor_commitment_example()
    
    # 2. Verify commitment
    verification_result = verify_vendor_commitment(attestation, model_identity, nonce)
    
    # 3. Demonstrate API binding
    binder, bundle = demonstrate_api_binding(attestation, model_identity)
    
    # 4. Demonstrate schema validation
    schema = demonstrate_vendor_schema_validation()
    
    # 5. Save configuration
    config_path = save_configuration_example()
    
    # Summary
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print("\nSummary:")
    print(f"  ✅ Vendor commitment created and verified")
    print(f"  ✅ API transcript bound to model identity")
    print(f"  ✅ Evidence bundle generated")
    print(f"  ✅ Vendor schema validated")
    print(f"  ✅ Configuration saved to {config_path}")
    
    print("\nThis example demonstrates:")
    print("  1. Creating vendor commitments for model attestation")
    print("  2. Verifying vendor commitments")
    print("  3. Binding API transcripts to attested models")
    print("  4. Generating cryptographic evidence bundles")
    print("  5. Managing security configurations")


if __name__ == '__main__':
    # Ensure we can import Flask if running server example
    try:
        import flask
    except ImportError:
        print("Note: Flask not installed. Run 'pip install flask' to enable mock server.")
    
    main()