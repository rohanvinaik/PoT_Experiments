# TEE Attestation and API Security Guide

## Overview

This guide provides comprehensive documentation for setting up and integrating Trusted Execution Environment (TEE) attestation with the PoT framework for secure API-based model verification.

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Supported TEE Platforms](#supported-tee-platforms)
4. [Setup Instructions](#setup-instructions)
5. [API Integration](#api-integration)
6. [Configuration](#configuration)
7. [Examples](#examples)
8. [Security Considerations](#security-considerations)
9. [Troubleshooting](#troubleshooting)

## Introduction

The TEE attestation system provides cryptographic proof of model identity and execution environment integrity for API-based model verification. It supports hardware-based TEEs (Intel SGX, AMD SEV, AWS Nitro) and vendor commitment schemes.

### Key Features

- **Multiple TEE Support**: Intel SGX, AMD SEV, AWS Nitro Enclaves
- **Vendor Commitments**: Software-based attestation for cloud APIs
- **Secure Binding**: Cryptographic binding of API transcripts to model identities
- **Evidence Bundles**: Tamper-proof audit trails with integrity verification
- **Attack Prevention**: Replay attack and model substitution detection

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
├─────────────────────────────────────────────────────────┤
│                   API Secure Binding                     │
│  - Transcript binding to model identity                  │
│  - Evidence bundle generation                            │
│  - Integrity validation                                  │
├─────────────────────────────────────────────────────────┤
│                Attestation Verification                  │
│  - Quote verification                                    │
│  - Certificate chain validation                          │
│  - Measurement checks                                    │
│  - Freshness validation                                  │
├─────────────────────────────────────────────────────────┤
│                  TEE Attestation Layer                   │
│  ┌──────┐ ┌──────┐ ┌───────┐ ┌────────┐ ┌──────────┐  │
│  │ SGX  │ │ SEV  │ │ Nitro │ │ Vendor │ │   Mock   │  │
│  └──────┘ └──────┘ └───────┘ └────────┘ └──────────┘  │
├─────────────────────────────────────────────────────────┤
│                    Hardware/Platform                     │
└─────────────────────────────────────────────────────────┘
```

## Supported TEE Platforms

### Intel SGX (Software Guard Extensions)

**Security Level**: HIGH  
**Use Cases**: On-premises deployments, high-security environments

```python
from src.pot.security.tee_attestation import SGXAttestationProvider

provider = SGXAttestationProvider({
    'enclave_id': 'your_enclave_id',
    'mrenclave': 'measurement_hash',  # Optional, auto-generated if not provided
    'mrsigner': 'signer_hash',        # Optional
    'isv_prod_id': 1,
    'isv_svn': 1
})
```

### AMD SEV (Secure Encrypted Virtualization)

**Security Level**: HIGH  
**Use Cases**: Cloud VMs, confidential computing

```python
from src.pot.security.tee_attestation import SEVAttestationProvider

provider = SEVAttestationProvider({
    'vm_id': 'your_vm_id',
    'launch_measurement': 'measurement',  # Optional
    'policy': 0x01  # SEV policy flags
})
```

### AWS Nitro Enclaves

**Security Level**: HIGH  
**Use Cases**: AWS deployments, serverless functions

```python
from src.pot.security.tee_attestation import NitroAttestationProvider

provider = NitroAttestationProvider({
    'enclave_id': 'enclave_id',
    'instance_id': 'i-1234567890abcdef',
    'region': 'us-east-1'
})
```

### Vendor Commitments

**Security Level**: MEDIUM  
**Use Cases**: API providers without TEE hardware

```python
from src.pot.security.tee_attestation import VendorCommitmentProvider

provider = VendorCommitmentProvider({
    'vendor_name': 'YourCompany',
    'vendor_key_id': 'key_2024',
    'api_endpoint': 'https://api.yourcompany.com',
    'api_version': 'v1'
})
```

## Setup Instructions

### 1. Installation

```bash
# Install required dependencies
pip install cryptography pycose

# For Intel SGX support
# Install Intel SGX SDK: https://github.com/intel/linux-sgx

# For AMD SEV support
# Requires SEV-enabled hardware and kernel

# For AWS Nitro
# Deploy within AWS Nitro Enclaves environment
```

### 2. Basic Setup

```python
from src.pot.security.tee_attestation import create_attestation_provider, ModelIdentity
from src.pot.security.attestation_verifier import AttestationVerifier
from src.pot.api.secure_binding import SecureAPIBinder

# 1. Create attestation provider
provider = create_attestation_provider(AttestationType.SGX, {
    'enclave_id': 'your_enclave'
})

# 2. Define model identity
model = ModelIdentity(
    model_hash="sha256_of_weights",
    model_name="gpt-2",
    version="1.0.0",
    provider="openai",
    architecture="transformer",
    parameter_count=117000000
)

# 3. Generate attestation
nonce = str(uuid.uuid4())
attestation = provider.generate_attestation(model, nonce)

# 4. Verify attestation
verifier = AttestationVerifier()
result = verifier.verify_attestation(attestation, model, nonce)
```

### 3. Cloud Provider Setup

#### Intel SGX on Azure

```bash
# Create confidential computing VM
az vm create \
  --resource-group myResourceGroup \
  --name myVM \
  --image UbuntuServer \
  --size Standard_DC2s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys
```

#### AMD SEV on Google Cloud

```bash
# Create confidential VM
gcloud compute instances create my-sev-vm \
  --zone=us-central1-a \
  --machine-type=n2d-standard-2 \
  --confidential-compute \
  --maintenance-policy=TERMINATE
```

#### AWS Nitro Enclaves

```bash
# Launch Nitro-enabled instance
aws ec2 run-instances \
  --image-id ami-12345678 \
  --instance-type m5.xlarge \
  --enclave-options 'Enabled=true'
```

## API Integration

### 1. Mock TEE Server

Start the mock server for testing:

```bash
python examples/api_security/mock_tee_server.py \
  --provider sgx \
  --model-name gpt-2 \
  --port 8080
```

### 2. Client Integration

```python
import requests

# Create attestation session
response = requests.post('http://localhost:8080/attestation/create', json={
    'nonce': str(uuid.uuid4())
})
session_data = response.json()

# Perform inference with attestation
response = requests.post('http://localhost:8080/inference', json={
    'session_id': session_data['session_id'],
    'prompt': 'Hello, world!',
    'max_tokens': 50
})
result = response.json()

# Get evidence bundle
response = requests.get('http://localhost:8080/evidence/bundle')
evidence = response.json()
```

### 3. Production API Integration

```python
from src.pot.api.secure_binding import SecureAPIBinder, APITranscript

# Initialize binder
binder = SecureAPIBinder(policy=BindingPolicy.strict_policy())

# For each API call:
transcript = APITranscript(
    transcript_id=str(uuid.uuid4()),
    timestamp=time.time(),
    endpoint='/v1/completions',
    method='POST',
    request={
        'model': 'gpt-3.5-turbo',
        'prompt': 'User query',
        'max_tokens': 100
    },
    response={
        'choices': [{'text': 'Model response'}],
        'usage': {'total_tokens': 150}
    },
    latency_ms=245.3
)

# Bind to attestation
bound = binder.bind_transcript(transcript, model_identity, attestation)

# Generate evidence for audit
evidence = binder.create_evidence_bundle([transcript.transcript_id])
```

## Configuration

### 1. Create Configuration File

```python
from src.pot.config.api_security import APISecurityConfig, create_default_config

# Create and customize configuration
config = create_default_config()
config.default_provider = AttestationType.SGX
config.security_thresholds.min_security_level = SecurityLevel.HIGH
config.binding_policy.max_binding_age = 3600  # 1 hour

# Save configuration
config.save(Path('config/api_security.json'))
```

### 2. Configuration File Format

```json
{
  "tee_providers": {
    "sgx": {
      "provider_type": "sgx",
      "enabled": true,
      "config": {
        "enclave_id": "production_enclave",
        "mrenclave": null,
        "mrsigner": null,
        "isv_prod_id": 1,
        "isv_svn": 1
      },
      "priority": 10,
      "fallback_providers": ["vendor"]
    }
  },
  "default_provider": "sgx",
  "attestation_policy": {
    "required_security_level": "high",
    "max_attestation_age": 3600,
    "allowed_providers": ["sgx", "sev", "nitro"],
    "enforce_certificate_validation": true,
    "enforce_tcb_updates": true
  },
  "binding_policy": {
    "mode": "tee_attestation",
    "require_attestation": true,
    "max_binding_age": 3600,
    "verify_on_bind": true,
    "store_full_transcript": false,
    "hash_sensitive_data": true
  },
  "security_thresholds": {
    "min_security_level": "high",
    "max_attestation_age_seconds": 3600,
    "max_verification_retries": 3,
    "confidence_threshold": 0.99
  }
}
```

### 3. Load and Use Configuration

```python
from src.pot.config.api_security import APISecurityConfig

# Load configuration
config = APISecurityConfig.load(Path('config/api_security.json'))

# Create provider from config
provider_config = config.tee_providers['sgx']
provider = create_attestation_provider(
    provider_config.provider_type,
    provider_config.config
)

# Use configured policies
verifier = AttestationVerifier(config.attestation_policy)
binder = SecureAPIBinder(config.binding_policy)
```

## Examples

### Complete E2E Example

```python
#!/usr/bin/env python3
"""Complete TEE attestation example"""

from src.pot.security.tee_attestation import (
    AttestationType, ModelIdentity, create_attestation_provider
)
from src.pot.security.attestation_verifier import AttestationVerifier
from src.pot.api.secure_binding import SecureAPIBinder, APITranscript
from src.pot.config.api_security import create_default_config

def main():
    # 1. Setup
    config = create_default_config()
    provider = create_attestation_provider(AttestationType.SGX, {})
    verifier = AttestationVerifier(config.attestation_policy)
    binder = SecureAPIBinder(config.binding_policy)
    
    # 2. Model identity
    model = ModelIdentity(
        model_hash="abc123",
        model_name="llama-2-7b",
        version="1.0.0",
        provider="meta",
        architecture="transformer",
        parameter_count=7000000000
    )
    
    # 3. Generate attestation
    nonce = str(uuid.uuid4())
    attestation = provider.generate_attestation(model, nonce)
    
    # 4. Verify
    result = verifier.verify_attestation(attestation, model, nonce)
    print(f"Attestation valid: {result.is_valid()}")
    
    # 5. Bind API call
    transcript = APITranscript(
        transcript_id=str(uuid.uuid4()),
        timestamp=time.time(),
        endpoint='/chat/completions',
        method='POST',
        request={'messages': [{'role': 'user', 'content': 'Hi'}]},
        response={'choices': [{'message': {'content': 'Hello!'}}]},
        latency_ms=150.0
    )
    
    bound = binder.bind_transcript(transcript, model, attestation)
    
    # 6. Generate evidence
    evidence = binder.create_evidence_bundle([transcript.transcript_id])
    print(f"Evidence bundle created with {len(evidence['transcripts'])} transcripts")

if __name__ == '__main__':
    main()
```

### Running Examples

```bash
# Run vendor commitment example
python examples/api_security/vendor_commitment_example.py

# Run integration tests
python examples/api_security/integration_test.py

# Start mock TEE server
python examples/api_security/mock_tee_server.py --provider sgx

# Run comprehensive tests
python -m pytest tests/test_api_security.py -v
```

## Security Considerations

### 1. Key Management

- **Never hardcode keys**: Use secure key management services
- **Rotate keys regularly**: Implement key rotation policies
- **Use HSMs**: Hardware Security Modules for key storage

### 2. Network Security

- **Use TLS**: Always encrypt API communications
- **Certificate pinning**: Pin TEE provider certificates
- **Rate limiting**: Implement rate limits for attestation requests

### 3. Attestation Best Practices

- **Fresh nonces**: Always use fresh nonces for each attestation
- **Time bounds**: Set appropriate expiration times
- **Measurement validation**: Verify expected measurements
- **TCB updates**: Keep Trusted Computing Base updated

### 4. Attack Mitigation

```python
# Enable all security checks
policy = AttestationPolicy(
    required_security_level=SecurityLevel.HIGH,
    max_attestation_age=300,  # 5 minutes
    enforce_certificate_validation=True,
    enforce_tcb_updates=True,
    allow_test_mode=False  # Disable in production
)

# Detect replay attacks
verifier = AttestationVerifier(policy)
# Verifier automatically tracks used nonces

# Prevent substitution attacks
# Always verify model identity matches attestation
```

## Troubleshooting

### Common Issues

#### 1. "Attestation verification failed"

```python
# Check failure reasons
result = verifier.verify_attestation(attestation, model, nonce)
for failure in result.failures:
    print(f"Failure: {failure.value}")

# Common causes:
# - Expired attestation (check timestamps)
# - Wrong nonce
# - Model identity mismatch
```

#### 2. "TEE not available"

```bash
# Check SGX support
cpuid | grep -i sgx

# Check SEV support
dmesg | grep -i sev

# Check Nitro enclave
nitro-cli describe-enclaves
```

#### 3. "Certificate validation failed"

```python
# Disable certificate validation for testing
policy.enforce_certificate_validation = False

# In production, ensure certificates are properly configured
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Verbose attestation generation
attestation = provider.generate_attestation(
    model, nonce, 
    additional_data={'debug': True}
)

# Detailed verification
verifier = AttestationVerifier(policy)
result = verifier.verify_attestation(attestation, model, nonce)
print(f"Verification details: {result.to_dict()}")
```

### Performance Optimization

```python
# 1. Cache attestations
attestation_cache = {}
cache_key = f"{model.compute_identity_hash()}_{provider.provider_type.value}"
if cache_key not in attestation_cache:
    attestation_cache[cache_key] = provider.generate_attestation(model, nonce)

# 2. Batch verification
results = verifier.batch_verify(attestations, models, nonces)

# 3. Async operations
import asyncio

async def async_attestation():
    return await asyncio.to_thread(
        provider.generate_attestation, model, nonce
    )
```

## API Reference

### Core Classes

- `AttestationProvider`: Base class for TEE providers
- `AttestationReport`: Attestation data container
- `ModelIdentity`: Model identification information
- `AttestationVerifier`: Verification engine
- `SecureAPIBinder`: API transcript binding manager
- `APISecurityConfig`: Configuration management

### Key Methods

```python
# Generate attestation
attestation = provider.generate_attestation(model_identity, nonce)

# Verify attestation
result = verifier.verify_attestation(attestation, model_identity, nonce)

# Bind transcript
bound = binder.bind_transcript(transcript, model_identity, attestation)

# Create evidence
bundle = binder.create_evidence_bundle(transcript_ids)

# Validate bundle
valid, errors = binder.validate_evidence_bundle(bundle)
```

## References

- [Intel SGX Documentation](https://software.intel.com/content/www/us/en/develop/topics/software-guard-extensions.html)
- [AMD SEV Documentation](https://developer.amd.com/sev/)
- [AWS Nitro Enclaves](https://aws.amazon.com/ec2/nitro/nitro-enclaves/)
- [PoT Paper](https://arxiv.org/abs/2403.09399)

## Support

For issues and questions:
- GitHub Issues: [PoT_Experiments/issues](https://github.com/PoT_Experiments/issues)
- Documentation: [PoT_Experiments/docs](https://github.com/PoT_Experiments/docs)