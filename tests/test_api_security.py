"""
Comprehensive Test Suite for API Security

Tests TEE attestation, verification, binding, and attack scenarios.
"""

import unittest
import json
import time
import uuid
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pot.security.tee_attestation import (
    AttestationType,
    AttestationReport,
    ModelIdentity,
    SecurityLevel,
    create_attestation_provider,
    SGXAttestationProvider,
    SEVAttestationProvider,
    NitroAttestationProvider,
    VendorCommitmentProvider,
    MockTEEProvider
)
from src.pot.security.attestation_verifier import (
    AttestationVerifier,
    AttestationPolicy,
    VerificationResult,
    VerificationStatus,
    VerificationFailureReason
)
from src.pot.api.secure_binding import (
    APITranscript,
    BoundTranscript,
    SecureAPIBinder,
    BindingPolicy,
    BindingMode,
    BindingStatus
)
from src.pot.config.api_security import (
    APISecurityConfig,
    TEEProviderConfig,
    VendorCommitmentSchema,
    SecurityThresholds,
    create_default_config,
    create_test_config,
    validate_config
)


class TestTEEAttestation(unittest.TestCase):
    """Test TEE attestation providers"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_identity = ModelIdentity(
            model_hash="test_hash_123",
            model_name="test-model",
            version="1.0.0",
            provider="test-provider",
            architecture="transformer",
            parameter_count=1000000
        )
        self.nonce = str(uuid.uuid4())
    
    def test_sgx_attestation_generation(self):
        """Test SGX attestation generation"""
        provider = SGXAttestationProvider({
            'enclave_id': 'test_enclave',
            'mrenclave': 'a' * 64,
            'mrsigner': 'b' * 64
        })
        
        attestation = provider.generate_attestation(
            self.model_identity,
            self.nonce
        )
        
        self.assertEqual(attestation.provider_type, AttestationType.SGX)
        self.assertEqual(attestation.security_level, SecurityLevel.HIGH)
        self.assertEqual(attestation.nonce, self.nonce)
        self.assertIn('mrenclave', attestation.measurements)
        self.assertIn('mrsigner', attestation.measurements)
        self.assertTrue(len(attestation.quote) > 432)  # Minimum SGX quote size
    
    def test_sev_attestation_generation(self):
        """Test SEV attestation generation"""
        provider = SEVAttestationProvider({
            'vm_id': 'test_vm',
            'policy': 0x01
        })
        
        attestation = provider.generate_attestation(
            self.model_identity,
            self.nonce
        )
        
        self.assertEqual(attestation.provider_type, AttestationType.SEV)
        self.assertEqual(attestation.security_level, SecurityLevel.HIGH)
        self.assertIn('launch_measurement', attestation.measurements)
        self.assertIn('vm_id', attestation.measurements)
    
    def test_nitro_attestation_generation(self):
        """Test Nitro attestation generation"""
        provider = NitroAttestationProvider({
            'region': 'us-east-1',
            'instance_id': 'i-12345'
        })
        
        attestation = provider.generate_attestation(
            self.model_identity,
            self.nonce
        )
        
        self.assertEqual(attestation.provider_type, AttestationType.NITRO)
        self.assertEqual(attestation.security_level, SecurityLevel.HIGH)
        self.assertIn('pcr0', attestation.measurements)
        self.assertIn('enclave_id', attestation.measurements)
        
        # Check COSE structure
        cose_data = json.loads(attestation.quote)
        self.assertIn('payload', cose_data)
        self.assertIn('signature', cose_data)
    
    def test_vendor_commitment_generation(self):
        """Test vendor commitment generation"""
        provider = VendorCommitmentProvider({
            'vendor_name': 'TestVendor',
            'api_endpoint': 'https://api.test.com',
            'api_version': 'v1'
        })
        
        attestation = provider.generate_attestation(
            self.model_identity,
            self.nonce
        )
        
        self.assertEqual(attestation.provider_type, AttestationType.VENDOR)
        self.assertEqual(attestation.security_level, SecurityLevel.MEDIUM)
        
        # Check commitment document
        document = json.loads(attestation.quote)
        self.assertEqual(document['type'], 'vendor_commitment')
        self.assertIn('commitment', document)
        self.assertIn('signature', document)
    
    def test_attestation_report_serialization(self):
        """Test attestation report serialization"""
        provider = MockTEEProvider({})
        attestation = provider.generate_attestation(
            self.model_identity,
            self.nonce
        )
        
        # Serialize
        data = attestation.to_dict()
        
        # Deserialize
        restored = AttestationReport.from_dict(data)
        
        self.assertEqual(restored.provider_type, attestation.provider_type)
        self.assertEqual(restored.nonce, attestation.nonce)
        self.assertEqual(restored.measurements, attestation.measurements)
    
    def test_model_identity_hash(self):
        """Test model identity hash computation"""
        hash1 = self.model_identity.compute_identity_hash()
        
        # Same model should produce same hash
        model2 = ModelIdentity(
            model_hash=self.model_identity.model_hash,
            model_name=self.model_identity.model_name,
            version=self.model_identity.version,
            provider=self.model_identity.provider,
            architecture=self.model_identity.architecture,
            parameter_count=self.model_identity.parameter_count
        )
        hash2 = model2.compute_identity_hash()
        
        self.assertEqual(hash1, hash2)
        
        # Different model should produce different hash
        model3 = ModelIdentity(
            model_hash="different_hash",
            model_name=self.model_identity.model_name,
            version=self.model_identity.version,
            provider=self.model_identity.provider,
            architecture=self.model_identity.architecture,
            parameter_count=self.model_identity.parameter_count
        )
        hash3 = model3.compute_identity_hash()
        
        self.assertNotEqual(hash1, hash3)


class TestAttestationVerification(unittest.TestCase):
    """Test attestation verification"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_identity = ModelIdentity(
            model_hash="test_hash",
            model_name="test-model",
            version="1.0.0",
            provider="test",
            architecture="transformer",
            parameter_count=1000000
        )
        self.nonce = str(uuid.uuid4())
        self.policy = AttestationPolicy.relaxed_policy()
        self.verifier = AttestationVerifier(self.policy)
    
    def test_valid_attestation_verification(self):
        """Test verification of valid attestation"""
        provider = MockTEEProvider({})
        attestation = provider.generate_attestation(
            self.model_identity,
            self.nonce
        )
        
        result = self.verifier.verify_attestation(
            attestation,
            self.model_identity,
            self.nonce
        )
        
        self.assertEqual(result.status, VerificationStatus.VALID)
        self.assertTrue(result.is_valid())
        self.assertTrue(result.freshness_valid)
        self.assertEqual(len(result.failures), 0)
    
    def test_nonce_mismatch_detection(self):
        """Test nonce mismatch detection"""
        provider = MockTEEProvider({})
        attestation = provider.generate_attestation(
            self.model_identity,
            self.nonce
        )
        
        wrong_nonce = str(uuid.uuid4())
        result = self.verifier.verify_attestation(
            attestation,
            self.model_identity,
            wrong_nonce
        )
        
        self.assertFalse(result.is_valid())
        self.assertIn(VerificationFailureReason.NONCE_MISMATCH, result.failures)
    
    def test_replay_attack_detection(self):
        """Test replay attack detection"""
        provider = MockTEEProvider({})
        attestation = provider.generate_attestation(
            self.model_identity,
            self.nonce
        )
        
        # First verification
        result1 = self.verifier.verify_attestation(
            attestation,
            self.model_identity,
            self.nonce
        )
        self.assertTrue(result1.is_valid())
        
        # Replay attempt
        result2 = self.verifier.verify_attestation(
            attestation,
            self.model_identity,
            self.nonce
        )
        self.assertFalse(result2.is_valid())
        self.assertIn(VerificationFailureReason.REPLAY_ATTACK, result2.failures)
    
    def test_expired_attestation_detection(self):
        """Test expired attestation detection"""
        # Create policy with short expiration
        policy = AttestationPolicy(
            required_security_level=SecurityLevel.LOW,
            max_attestation_age=0.001,  # 1ms
            allowed_providers={AttestationType.MOCK},
            required_measurements={},
            allow_test_mode=True,
            enforce_certificate_validation=False,
            enforce_tcb_updates=False,
            custom_validators=[]
        )
        verifier = AttestationVerifier(policy)
        
        provider = MockTEEProvider({})
        attestation = provider.generate_attestation(
            self.model_identity,
            self.nonce
        )
        
        # Modify timestamp to make it old
        attestation.timestamp = time.time() - 10  # 10 seconds ago
        
        result = verifier.verify_attestation(
            attestation,
            self.model_identity,
            self.nonce
        )
        
        self.assertFalse(result.is_valid())
        self.assertIn(VerificationFailureReason.EXPIRED_ATTESTATION, result.failures)
    
    def test_model_substitution_detection(self):
        """Test model substitution attack detection"""
        provider = MockTEEProvider({})
        attestation = provider.generate_attestation(
            self.model_identity,
            self.nonce
        )
        
        # Different model
        wrong_model = ModelIdentity(
            model_hash="different_hash",
            model_name="malicious-model",
            version="1.0.0",
            provider="attacker",
            architecture="transformer",
            parameter_count=1000000
        )
        
        result = self.verifier.verify_attestation(
            attestation,
            wrong_model,
            self.nonce
        )
        
        self.assertFalse(result.is_valid())
        self.assertIn(VerificationFailureReason.INVALID_QUOTE, result.failures)
    
    def test_batch_verification(self):
        """Test batch attestation verification"""
        provider = MockTEEProvider({})
        
        attestations = []
        models = []
        nonces = []
        
        for i in range(10):
            model = ModelIdentity(
                model_hash=f"hash_{i}",
                model_name=f"model_{i}",
                version="1.0.0",
                provider="test",
                architecture="transformer",
                parameter_count=1000000
            )
            nonce = str(uuid.uuid4())
            attestation = provider.generate_attestation(model, nonce)
            
            attestations.append(attestation)
            models.append(model)
            nonces.append(nonce)
        
        results = self.verifier.batch_verify(attestations, models, nonces)
        
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertTrue(result.is_valid())
        
        # Get summary
        summary = self.verifier.get_verification_summary(results)
        self.assertEqual(summary['total'], 10)
        self.assertEqual(summary['valid'], 10)
        self.assertEqual(summary['success_rate'], 1.0)


class TestAPIBinding(unittest.TestCase):
    """Test API transcript binding"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_identity = ModelIdentity(
            model_hash="test_hash",
            model_name="test-model",
            version="1.0.0",
            provider="test",
            architecture="transformer",
            parameter_count=1000000
        )
        self.nonce = str(uuid.uuid4())
        self.policy = BindingPolicy.relaxed_policy()
        self.binder = SecureAPIBinder(self.policy)
    
    def test_transcript_binding(self):
        """Test binding API transcript to attestation"""
        # Create attestation
        provider = MockTEEProvider({})
        attestation = provider.generate_attestation(
            self.model_identity,
            self.nonce
        )
        
        # Create transcript
        transcript = APITranscript(
            transcript_id=str(uuid.uuid4()),
            timestamp=time.time(),
            endpoint='/api/test',
            method='POST',
            request={'data': 'test'},
            response={'result': 'success'},
            latency_ms=100.0
        )
        
        # Bind
        bound = self.binder.bind_transcript(
            transcript,
            self.model_identity,
            attestation,
            verify_immediately=False
        )
        
        self.assertIsInstance(bound, BoundTranscript)
        self.assertEqual(bound.transcript.transcript_id, transcript.transcript_id)
        self.assertIsNotNone(bound.binding_signature)
        self.assertIsNotNone(bound.binding_nonce)
    
    def test_binding_verification(self):
        """Test binding integrity verification"""
        provider = MockTEEProvider({})
        attestation = provider.generate_attestation(
            self.model_identity,
            self.nonce
        )
        
        transcript = APITranscript(
            transcript_id=str(uuid.uuid4()),
            timestamp=time.time(),
            endpoint='/api/test',
            method='POST',
            request={'data': 'test'},
            response={'result': 'success'},
            latency_ms=100.0
        )
        
        bound = self.binder.bind_transcript(
            transcript,
            self.model_identity,
            attestation,
            verify_immediately=False
        )
        
        # Verify binding
        status = self.binder.verify_binding(bound)
        self.assertEqual(status, BindingStatus.VALID)
    
    def test_evidence_bundle_creation(self):
        """Test evidence bundle creation"""
        provider = MockTEEProvider({})
        
        # Create multiple bound transcripts
        transcript_ids = []
        for i in range(3):
            attestation = provider.generate_attestation(
                self.model_identity,
                str(uuid.uuid4())
            )
            
            transcript = APITranscript(
                transcript_id=str(uuid.uuid4()),
                timestamp=time.time(),
                endpoint=f'/api/test_{i}',
                method='POST',
                request={'data': f'test_{i}'},
                response={'result': f'success_{i}'},
                latency_ms=100.0 + i * 10
            )
            
            bound = self.binder.bind_transcript(
                transcript,
                self.model_identity,
                attestation,
                verify_immediately=False
            )
            
            transcript_ids.append(transcript.transcript_id)
        
        # Create bundle
        bundle = self.binder.create_evidence_bundle(transcript_ids)
        
        self.assertIn('version', bundle)
        self.assertIn('transcripts', bundle)
        self.assertIn('summary', bundle)
        self.assertIn('signature', bundle)
        self.assertEqual(len(bundle['transcripts']), 3)
        self.assertEqual(bundle['summary']['total_transcripts'], 3)
    
    def test_bundle_validation(self):
        """Test evidence bundle validation"""
        provider = MockTEEProvider({})
        attestation = provider.generate_attestation(
            self.model_identity,
            self.nonce
        )
        
        transcript = APITranscript(
            transcript_id=str(uuid.uuid4()),
            timestamp=time.time(),
            endpoint='/api/test',
            method='POST',
            request={'data': 'test'},
            response={'result': 'success'},
            latency_ms=100.0
        )
        
        bound = self.binder.bind_transcript(
            transcript,
            self.model_identity,
            attestation,
            verify_immediately=False
        )
        
        # Create bundle
        bundle = self.binder.create_evidence_bundle([transcript.transcript_id])
        
        # Validate
        valid, errors = self.binder.validate_evidence_bundle(bundle)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)
        
        # Tamper with bundle
        bundle['summary']['total_transcripts'] = 999
        valid_tampered, errors_tampered = self.binder.validate_evidence_bundle(bundle)
        self.assertFalse(valid_tampered)
        self.assertGreater(len(errors_tampered), 0)
    
    def test_binding_export_import(self):
        """Test exporting and importing bindings"""
        provider = MockTEEProvider({})
        attestation = provider.generate_attestation(
            self.model_identity,
            self.nonce
        )
        
        transcript = APITranscript(
            transcript_id=str(uuid.uuid4()),
            timestamp=time.time(),
            endpoint='/api/test',
            method='POST',
            request={'data': 'test'},
            response={'result': 'success'},
            latency_ms=100.0
        )
        
        bound = self.binder.bind_transcript(
            transcript,
            self.model_identity,
            attestation,
            verify_immediately=False
        )
        
        # Export
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            export_path = Path(f.name)
        
        self.binder.export_bindings(export_path, [transcript.transcript_id])
        self.assertTrue(export_path.exists())
        
        # Import into new binder
        new_binder = SecureAPIBinder(self.policy)
        imported, errors = new_binder.import_bindings(export_path)
        
        self.assertEqual(imported, 1)
        self.assertEqual(len(errors), 0)
        self.assertIn(transcript.transcript_id, new_binder.bindings)
        
        # Clean up
        export_path.unlink()


class TestAPISecurityConfig(unittest.TestCase):
    """Test API security configuration"""
    
    def test_default_config_creation(self):
        """Test default configuration creation"""
        config = create_default_config()
        
        self.assertIsInstance(config, APISecurityConfig)
        self.assertEqual(config.default_provider, AttestationType.SGX)
        self.assertIn('sgx', config.tee_providers)
        self.assertIn('openai', config.vendor_schemas)
    
    def test_test_config_creation(self):
        """Test test configuration creation"""
        config = create_test_config()
        
        self.assertEqual(config.default_provider, AttestationType.MOCK)
        self.assertIn('mock', config.tee_providers)
        self.assertEqual(
            config.security_thresholds.min_security_level,
            SecurityLevel.LOW
        )
    
    def test_config_serialization(self):
        """Test configuration serialization"""
        config = create_default_config()
        
        # Convert to dict
        data = config.to_dict()
        
        # Recreate from dict
        restored = APISecurityConfig.from_dict(data)
        
        self.assertEqual(restored.default_provider, config.default_provider)
        self.assertEqual(
            len(restored.tee_providers),
            len(config.tee_providers)
        )
    
    def test_config_save_load(self):
        """Test configuration save and load"""
        config = create_default_config()
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = Path(f.name)
        
        # Save
        config.save(config_path)
        self.assertTrue(config_path.exists())
        
        # Load
        loaded = APISecurityConfig.load(config_path)
        
        self.assertEqual(loaded.default_provider, config.default_provider)
        self.assertEqual(
            loaded.binding_policy.mode,
            config.binding_policy.mode
        )
        
        # Clean up
        config_path.unlink()
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = create_default_config()
        valid, errors = validate_config(config)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)
        
        # Invalid config - default provider not configured
        config.default_provider = AttestationType.MOCK
        config.tee_providers = {}  # No providers
        valid, errors = validate_config(config)
        self.assertFalse(valid)
        self.assertGreater(len(errors), 0)
    
    def test_vendor_schema_validation(self):
        """Test vendor commitment schema validation"""
        schema = VendorCommitmentSchema(
            vendor_name='TestVendor',
            api_version='v1',
            endpoint='https://api.test.com',
            key_id='test_key',
            required_fields=['model', 'timestamp'],
            optional_fields=['metadata'],
            signature_algorithm='HMAC-SHA256',
            validation_rules={
                'timestamp': {'type': float, 'min': 0}
            }
        )
        
        # Valid commitment
        commitment = {
            'model': 'test-model',
            'timestamp': time.time()
        }
        valid, errors = schema.validate_commitment(commitment)
        self.assertTrue(valid)
        
        # Invalid commitment - missing required field
        commitment_invalid = {'timestamp': time.time()}
        valid, errors = schema.validate_commitment(commitment_invalid)
        self.assertFalse(valid)
        self.assertIn('Missing required field: model', errors[0])


class TestPerformance(unittest.TestCase):
    """Performance benchmarks for API security"""
    
    def test_attestation_generation_performance(self):
        """Benchmark attestation generation"""
        providers = [
            MockTEEProvider({}),
            SGXAttestationProvider({}),
            SEVAttestationProvider({}),
            NitroAttestationProvider({}),
            VendorCommitmentProvider({})
        ]
        
        model = ModelIdentity(
            model_hash="perf_test",
            model_name="perf-model",
            version="1.0.0",
            provider="test",
            architecture="transformer",
            parameter_count=1000000
        )
        
        for provider in providers:
            start = time.time()
            for _ in range(10):
                nonce = str(uuid.uuid4())
                attestation = provider.generate_attestation(model, nonce)
            elapsed = time.time() - start
            
            avg_time = (elapsed / 10) * 1000  # ms
            self.assertLess(avg_time, 100, 
                          f"{provider.provider_type.value} too slow: {avg_time:.2f}ms")
    
    def test_verification_performance(self):
        """Benchmark attestation verification"""
        provider = MockTEEProvider({})
        verifier = AttestationVerifier(AttestationPolicy.relaxed_policy())
        
        model = ModelIdentity(
            model_hash="perf_test",
            model_name="perf-model",
            version="1.0.0",
            provider="test",
            architecture="transformer",
            parameter_count=1000000
        )
        
        # Generate attestations
        attestations = []
        nonces = []
        for _ in range(100):
            nonce = str(uuid.uuid4())
            attestation = provider.generate_attestation(model, nonce)
            attestations.append(attestation)
            nonces.append(nonce)
        
        # Benchmark verification
        start = time.time()
        for attestation, nonce in zip(attestations, nonces):
            result = verifier.verify_attestation(attestation, model, nonce)
        elapsed = time.time() - start
        
        avg_time = (elapsed / 100) * 1000  # ms
        self.assertLess(avg_time, 10, 
                       f"Verification too slow: {avg_time:.2f}ms per attestation")
    
    def test_binding_performance(self):
        """Benchmark API transcript binding"""
        provider = MockTEEProvider({})
        binder = SecureAPIBinder(BindingPolicy.relaxed_policy())
        
        model = ModelIdentity(
            model_hash="perf_test",
            model_name="perf-model",
            version="1.0.0",
            provider="test",
            architecture="transformer",
            parameter_count=1000000
        )
        
        attestation = provider.generate_attestation(model, str(uuid.uuid4()))
        
        # Benchmark binding
        start = time.time()
        for i in range(100):
            transcript = APITranscript(
                transcript_id=str(uuid.uuid4()),
                timestamp=time.time(),
                endpoint='/api/test',
                method='POST',
                request={'data': f'test_{i}'},
                response={'result': f'success_{i}'},
                latency_ms=100.0
            )
            
            bound = binder.bind_transcript(
                transcript,
                model,
                attestation,
                verify_immediately=False
            )
        elapsed = time.time() - start
        
        avg_time = (elapsed / 100) * 1000  # ms
        self.assertLess(avg_time, 5, 
                       f"Binding too slow: {avg_time:.2f}ms per transcript")


if __name__ == '__main__':
    unittest.main()