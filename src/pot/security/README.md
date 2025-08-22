# TEE Attestation and API Security Module

## Overview

This module provides comprehensive TEE (Trusted Execution Environment) attestation and API security features for the PoT framework, enabling secure verification of models served through APIs.

## Quick Start

### Basic Usage

```bash
# Run E2E validation with TEE attestation
python scripts/run_e2e_validation.py \
    --ref-model gpt2 \
    --cand-model distilgpt2 \
    --tee-provider sgx \
    --enable-api-binding \
    --attestation-policy strict

# Use the wrapper script for easier execution
bash scripts/run_validation_with_tee.sh \
    --tee-provider mock \
    --enable-api-binding \
    --dry-run
```

### Python API

```python
from src.pot.security.tee_attestation import create_attestation_provider, ModelIdentity
from src.pot.security.attestation_verifier import AttestationVerifier
from src.pot.api.secure_binding import SecureAPIBinder

# Create provider
provider = create_attestation_provider(AttestationType.SGX, config)

# Generate attestation
model = ModelIdentity(...)
attestation = provider.generate_attestation(model, nonce)

# Verify attestation
verifier = AttestationVerifier()
result = verifier.verify_attestation(attestation, model, nonce)

# Bind API transcript
binder = SecureAPIBinder()
bound = binder.bind_transcript(transcript, model, attestation)
```

## Features

### TEE Providers

- **Intel SGX**: Hardware-based enclave attestation
- **AMD SEV**: Secure virtualization attestation
- **AWS Nitro**: Cloud enclave attestation
- **Vendor Commitment**: Software-based attestation for APIs
- **Mock**: Testing provider for development

### Security Features

- **Attestation Generation**: Platform-specific quote generation
- **Quote Verification**: Cryptographic verification of attestations
- **Certificate Chain Validation**: PKI-based trust establishment
- **Replay Attack Prevention**: Nonce tracking and freshness checks
- **Model Substitution Detection**: Identity binding verification
- **API Transcript Binding**: Cryptographic binding of API calls to models
- **Evidence Bundle Generation**: Tamper-proof audit trails

### Configuration

- **Policy-based Verification**: Strict, moderate, and relaxed policies
- **Security Thresholds**: Configurable security requirements
- **Provider Selection**: Automatic fallback and priority handling
- **Vendor Schemas**: Customizable commitment validation

## Module Structure

```
src/pot/security/
├── tee_attestation.py      # TEE provider implementations
├── attestation_verifier.py # Attestation verification engine
└── README.md               # This file

src/pot/api/
└── secure_binding.py       # API transcript binding

src/pot/config/
└── api_security.py         # Configuration management

examples/api_security/
├── mock_tee_server.py      # Mock TEE server for testing
├── vendor_commitment_example.py # Vendor commitment demo
└── integration_test.py    # Full integration tests

tests/
└── test_api_security.py   # Comprehensive test suite

docs/
└── TEE_ATTESTATION_GUIDE.md # Complete documentation
```

## Running Tests

```bash
# Run unit tests
python -m pytest tests/test_api_security.py -v

# Run TEE validation tests
python scripts/run_tee_validation.py --provider all

# Run integration tests
python examples/api_security/integration_test.py

# Start mock TEE server
python examples/api_security/mock_tee_server.py --provider sgx --port 8080
```

## Configuration Examples

### Strict Policy (Production)

```python
config = APISecurityConfig(
    default_provider=AttestationType.SGX,
    security_thresholds=SecurityThresholds.strict(),
    attestation_policy=AttestationPolicy.default_policy(),
    binding_policy=BindingPolicy.strict_policy()
)
```

### Relaxed Policy (Development)

```python
config = create_test_config()  # Pre-configured for testing
```

### Custom Configuration

```json
{
  "tee_providers": {
    "sgx": {
      "enabled": true,
      "priority": 10,
      "config": {
        "enclave_id": "production_enclave"
      }
    }
  },
  "security_thresholds": {
    "min_security_level": "high",
    "max_attestation_age_seconds": 3600
  }
}
```

## API Endpoints (Mock Server)

- `POST /attestation/create` - Generate attestation
- `POST /inference` - Perform inference with binding
- `GET /evidence/bundle` - Get evidence bundle
- `POST /evidence/verify` - Verify evidence bundle
- `GET /stats` - Get server statistics

## Security Considerations

1. **Always use fresh nonces** for attestation generation
2. **Verify attestations** before trusting model outputs
3. **Enable certificate validation** in production
4. **Monitor for replay attacks** using the built-in detection
5. **Store evidence bundles** for audit trails
6. **Use appropriate security levels** based on requirements

## Performance

- Attestation generation: ~1-5ms (mock), ~10-50ms (hardware TEE)
- Verification: ~1-10ms per attestation
- Binding: <1ms per transcript
- Bundle generation: ~5-20ms for 100 transcripts

## See Also

- [TEE Attestation Guide](../../docs/TEE_ATTESTATION_GUIDE.md) - Complete documentation
- [PoT Paper](https://arxiv.org/abs/2403.09399) - Academic paper
- [E2E Validation](../validation/README.md) - Main validation pipeline