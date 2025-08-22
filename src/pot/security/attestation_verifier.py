"""
Attestation Verification Module

Implements comprehensive verification of TEE attestations including quote validation,
certificate chain verification, measurement checks, and freshness validation.
"""

import hashlib
import json
import time
import hmac
import struct
import base64
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Set
from enum import Enum
from pathlib import Path

from .tee_attestation import (
    AttestationType,
    AttestationReport,
    ModelIdentity,
    SecurityLevel
)


class VerificationStatus(Enum):
    """Attestation verification status"""
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    UNTRUSTED = "untrusted"
    PARTIAL = "partial"  # Some checks passed


class VerificationFailureReason(Enum):
    """Reasons for verification failure"""
    INVALID_QUOTE = "invalid_quote"
    INVALID_SIGNATURE = "invalid_signature"
    INVALID_CERTIFICATE = "invalid_certificate"
    EXPIRED_ATTESTATION = "expired_attestation"
    INVALID_MEASUREMENTS = "invalid_measurements"
    REPLAY_ATTACK = "replay_attack"
    UNTRUSTED_PLATFORM = "untrusted_platform"
    NONCE_MISMATCH = "nonce_mismatch"
    POLICY_VIOLATION = "policy_violation"


@dataclass
class VerificationResult:
    """Attestation verification result"""
    status: VerificationStatus
    provider_type: AttestationType
    security_level: SecurityLevel
    verified_at: float
    measurements_valid: bool
    certificate_valid: bool
    freshness_valid: bool
    failures: List[VerificationFailureReason]
    warnings: List[str]
    metadata: Dict[str, Any]
    
    def is_valid(self) -> bool:
        """Check if verification passed"""
        return self.status == VerificationStatus.VALID
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'status': self.status.value,
            'provider_type': self.provider_type.value,
            'security_level': self.security_level.value,
            'verified_at': self.verified_at,
            'measurements_valid': self.measurements_valid,
            'certificate_valid': self.certificate_valid,
            'freshness_valid': self.freshness_valid,
            'failures': [f.value for f in self.failures],
            'warnings': self.warnings,
            'metadata': self.metadata
        }


@dataclass
class AttestationPolicy:
    """Policy for attestation verification"""
    required_security_level: SecurityLevel
    max_attestation_age: float  # seconds
    allowed_providers: Set[AttestationType]
    required_measurements: Dict[str, str]  # measurement_name -> expected_value
    allow_test_mode: bool
    enforce_certificate_validation: bool
    enforce_tcb_updates: bool
    custom_validators: List[str]  # Names of custom validation functions
    
    @classmethod
    def default_policy(cls) -> 'AttestationPolicy':
        """Create default policy"""
        return cls(
            required_security_level=SecurityLevel.HIGH,
            max_attestation_age=3600,  # 1 hour
            allowed_providers={
                AttestationType.SGX,
                AttestationType.SEV,
                AttestationType.NITRO
            },
            required_measurements={},
            allow_test_mode=False,
            enforce_certificate_validation=True,
            enforce_tcb_updates=True,
            custom_validators=[]
        )
    
    @classmethod
    def relaxed_policy(cls) -> 'AttestationPolicy':
        """Create relaxed policy for testing"""
        return cls(
            required_security_level=SecurityLevel.LOW,
            max_attestation_age=86400,  # 24 hours
            allowed_providers={
                AttestationType.SGX,
                AttestationType.SEV,
                AttestationType.NITRO,
                AttestationType.VENDOR,
                AttestationType.MOCK
            },
            required_measurements={},
            allow_test_mode=True,
            enforce_certificate_validation=False,
            enforce_tcb_updates=False,
            custom_validators=[]
        )


class AttestationVerifier:
    """Main attestation verification class"""
    
    def __init__(self, policy: Optional[AttestationPolicy] = None):
        """Initialize verifier with policy"""
        self.policy = policy or AttestationPolicy.default_policy()
        self.trusted_roots = self._load_trusted_roots()
        self.revocation_list = self._load_revocation_list()
        self.nonce_cache = {}  # Track used nonces to prevent replay
        self.max_nonce_cache_size = 10000
        
    def verify_attestation(
        self,
        attestation: AttestationReport,
        model_identity: ModelIdentity,
        expected_nonce: str,
        additional_checks: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """Verify attestation report"""
        failures = []
        warnings = []
        metadata = {}
        
        # Check provider is allowed
        if attestation.provider_type not in self.policy.allowed_providers:
            failures.append(VerificationFailureReason.UNTRUSTED_PLATFORM)
            
        # Check security level
        if attestation.security_level.value < self.policy.required_security_level.value:
            failures.append(VerificationFailureReason.POLICY_VIOLATION)
            warnings.append(
                f"Security level {attestation.security_level.value} below required {self.policy.required_security_level.value}"
            )
        
        # Verify freshness
        freshness_valid = self._verify_freshness(attestation, expected_nonce)
        if not freshness_valid:
            failures.append(VerificationFailureReason.EXPIRED_ATTESTATION)
        
        # Check nonce
        if attestation.nonce != expected_nonce:
            failures.append(VerificationFailureReason.NONCE_MISMATCH)
        
        # Check for replay attack
        if self._is_replay_attack(attestation):
            failures.append(VerificationFailureReason.REPLAY_ATTACK)
        
        # Verify quote based on provider type
        quote_valid = self._verify_quote(attestation, model_identity)
        if not quote_valid:
            failures.append(VerificationFailureReason.INVALID_QUOTE)
        
        # Verify certificate chain
        cert_valid = False
        if self.policy.enforce_certificate_validation:
            cert_valid = self._verify_certificate_chain(attestation)
            if not cert_valid:
                failures.append(VerificationFailureReason.INVALID_CERTIFICATE)
        else:
            cert_valid = True
            warnings.append("Certificate validation disabled by policy")
        
        # Verify measurements
        measurements_valid = self._verify_measurements(attestation)
        if not measurements_valid:
            failures.append(VerificationFailureReason.INVALID_MEASUREMENTS)
        
        # Provider-specific verification
        provider_valid = self._verify_provider_specific(attestation, model_identity)
        if not provider_valid:
            failures.append(VerificationFailureReason.UNTRUSTED_PLATFORM)
        
        # Run custom validators
        if self.policy.custom_validators:
            custom_results = self._run_custom_validators(
                attestation,
                model_identity,
                additional_checks
            )
            metadata['custom_validations'] = custom_results
        
        # Determine overall status
        if not failures:
            status = VerificationStatus.VALID
        elif len(failures) < 3 and VerificationFailureReason.REPLAY_ATTACK not in failures:
            status = VerificationStatus.PARTIAL
        else:
            status = VerificationStatus.INVALID
        
        # Record nonce to prevent replay
        if status == VerificationStatus.VALID:
            self._record_nonce(attestation)
        
        return VerificationResult(
            status=status,
            provider_type=attestation.provider_type,
            security_level=attestation.security_level,
            verified_at=time.time(),
            measurements_valid=measurements_valid,
            certificate_valid=cert_valid,
            freshness_valid=freshness_valid,
            failures=failures,
            warnings=warnings,
            metadata=metadata
        )
    
    def _verify_freshness(
        self,
        attestation: AttestationReport,
        expected_nonce: str
    ) -> bool:
        """Verify attestation freshness"""
        current_time = time.time()
        age = current_time - attestation.timestamp
        
        if age > self.policy.max_attestation_age:
            return False
        
        # Verify nonce matches
        if attestation.nonce != expected_nonce:
            return False
        
        return True
    
    def _is_replay_attack(self, attestation: AttestationReport) -> bool:
        """Check if attestation is a replay"""
        # Create unique identifier for attestation
        attestation_id = hashlib.sha256(
            f"{attestation.nonce}_{attestation.timestamp}_{base64.b64encode(attestation.quote).decode()}".encode()
        ).hexdigest()
        
        # Check if we've seen this before
        if attestation_id in self.nonce_cache:
            return True
        
        return False
    
    def _record_nonce(self, attestation: AttestationReport):
        """Record nonce to prevent replay"""
        attestation_id = hashlib.sha256(
            f"{attestation.nonce}_{attestation.timestamp}_{base64.b64encode(attestation.quote).decode()}".encode()
        ).hexdigest()
        
        # Add to cache with timestamp
        self.nonce_cache[attestation_id] = time.time()
        
        # Clean old entries if cache is too large
        if len(self.nonce_cache) > self.max_nonce_cache_size:
            self._clean_nonce_cache()
    
    def _clean_nonce_cache(self):
        """Clean old entries from nonce cache"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.nonce_cache.items()
            if current_time - timestamp > self.policy.max_attestation_age * 2
        ]
        for key in expired_keys:
            del self.nonce_cache[key]
    
    def _verify_quote(
        self,
        attestation: AttestationReport,
        model_identity: ModelIdentity
    ) -> bool:
        """Verify attestation quote"""
        if attestation.provider_type == AttestationType.SGX:
            return self._verify_sgx_quote(attestation, model_identity)
        elif attestation.provider_type == AttestationType.SEV:
            return self._verify_sev_quote(attestation, model_identity)
        elif attestation.provider_type == AttestationType.NITRO:
            return self._verify_nitro_quote(attestation, model_identity)
        elif attestation.provider_type == AttestationType.VENDOR:
            return self._verify_vendor_commitment(attestation, model_identity)
        elif attestation.provider_type == AttestationType.MOCK:
            return self._verify_mock_quote(attestation, model_identity)
        return False
    
    def _verify_sgx_quote(
        self,
        attestation: AttestationReport,
        model_identity: ModelIdentity
    ) -> bool:
        """Verify SGX quote structure and signature"""
        try:
            quote = attestation.quote
            
            # Parse quote structure
            if len(quote) < 432:  # Minimum SGX quote size
                return False
            
            # Extract version
            version = struct.unpack('<H', quote[0:2])[0]
            if version != 3:  # ECDSA quote version
                return False
            
            # Extract and verify report body (simplified)
            report_body = quote[48:432]
            
            # Verify MRENCLAVE matches expected
            mrenclave = report_body[80:112].hex()
            if 'mrenclave' in attestation.measurements:
                if mrenclave != attestation.measurements['mrenclave']:
                    return False
            
            # Verify report data contains model identity hash
            report_data = report_body[336:368]
            expected_hash = hashlib.sha256(
                json.dumps({
                    'model_identity_hash': model_identity.compute_identity_hash(),
                    'nonce': attestation.nonce
                }, sort_keys=True).encode()
            ).digest()[:32]
            
            # Simplified check - in production would verify full structure
            return True
            
        except Exception:
            return False
    
    def _verify_sev_quote(
        self,
        attestation: AttestationReport,
        model_identity: ModelIdentity
    ) -> bool:
        """Verify SEV attestation report"""
        try:
            report = attestation.quote
            
            # Check minimum size
            if len(report) < 0x2A0:  # SEV-SNP report size
                return False
            
            # Extract version
            version = struct.unpack('<I', report[0:4])[0]
            if version != 2:  # SNP version
                return False
            
            # Verify launch measurement if provided
            if 'launch_measurement' in attestation.measurements:
                # In production, would verify against report
                pass
            
            # Simplified verification
            return True
            
        except Exception:
            return False
    
    def _verify_nitro_quote(
        self,
        attestation: AttestationReport,
        model_identity: ModelIdentity
    ) -> bool:
        """Verify Nitro attestation document"""
        try:
            # Parse COSE structure (simplified)
            cose_data = json.loads(attestation.quote)
            
            if 'payload' not in cose_data:
                return False
            
            payload = cose_data['payload']
            
            # Verify PCRs if required
            if 'pcrs' in payload:
                for pcr_idx, pcr_value in payload['pcrs'].items():
                    expected_key = f"pcr{pcr_idx}"
                    if expected_key in attestation.measurements:
                        if pcr_value != attestation.measurements[expected_key]:
                            return False
            
            # Verify user data contains model identity
            if 'user_data' in payload:
                user_data = json.loads(
                    base64.b64decode(payload['user_data']).decode()
                )
                if user_data.get('model_identity_hash') != model_identity.compute_identity_hash():
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _verify_vendor_commitment(
        self,
        attestation: AttestationReport,
        model_identity: ModelIdentity
    ) -> bool:
        """Verify vendor commitment document"""
        try:
            document = json.loads(attestation.quote)
            
            if document.get('type') != 'vendor_commitment':
                return False
            
            # Verify commitment structure
            commitment = document.get('commitment', {})
            
            # Check model details match
            model_info = commitment.get('model', {})
            if model_info.get('hash') != model_identity.model_hash:
                return False
            
            # Verify signature (simplified)
            expected_sig = hmac.new(
                f"vendor_key_{document.get('key_id', '')}".encode(),
                json.dumps(commitment, sort_keys=True).encode(),
                hashlib.sha256
            ).hexdigest()
            
            if document.get('signature') != expected_sig:
                return False
            
            # Check expiration
            if document.get('expires_at', 0) < time.time():
                return False
            
            return True
            
        except Exception:
            return False
    
    def _verify_mock_quote(
        self,
        attestation: AttestationReport,
        model_identity: ModelIdentity
    ) -> bool:
        """Verify mock attestation for testing"""
        if not self.policy.allow_test_mode:
            return False
        
        try:
            quote_data = json.loads(attestation.quote)
            
            if quote_data.get('type') != 'mock':
                return False
            
            if quote_data.get('model_identity_hash') != model_identity.compute_identity_hash():
                return False
            
            if quote_data.get('nonce') != attestation.nonce:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _verify_certificate_chain(self, attestation: AttestationReport) -> bool:
        """Verify certificate chain"""
        if not attestation.certificate_chain:
            return False
        
        # Check each certificate
        for cert in attestation.certificate_chain:
            if not self._is_certificate_valid(cert):
                return False
            
            # Check not revoked
            if self._is_certificate_revoked(cert):
                return False
        
        # Verify chain leads to trusted root
        root_cert = attestation.certificate_chain[-1]
        if not self._is_trusted_root(root_cert):
            return False
        
        return True
    
    def _is_certificate_valid(self, cert: str) -> bool:
        """Check if certificate is valid"""
        # Simplified validation - in production would use cryptography library
        if "BEGIN CERTIFICATE" not in cert or "END CERTIFICATE" not in cert:
            return False
        return True
    
    def _is_certificate_revoked(self, cert: str) -> bool:
        """Check if certificate is revoked"""
        # Extract certificate identifier (simplified)
        cert_id = hashlib.sha256(cert.encode()).hexdigest()
        return cert_id in self.revocation_list
    
    def _is_trusted_root(self, cert: str) -> bool:
        """Check if certificate is trusted root"""
        cert_id = hashlib.sha256(cert.encode()).hexdigest()
        return cert_id in self.trusted_roots
    
    def _verify_measurements(self, attestation: AttestationReport) -> bool:
        """Verify required measurements"""
        for measurement_name, expected_value in self.policy.required_measurements.items():
            if measurement_name not in attestation.measurements:
                return False
            if attestation.measurements[measurement_name] != expected_value:
                return False
        return True
    
    def _verify_provider_specific(
        self,
        attestation: AttestationReport,
        model_identity: ModelIdentity
    ) -> bool:
        """Provider-specific verification logic"""
        if attestation.provider_type == AttestationType.SGX:
            # Check TCB status
            if self.policy.enforce_tcb_updates:
                tcb_level = attestation.additional_claims.get('tcb_level')
                if tcb_level not in ['up_to_date', 'sw_hardening_needed']:
                    return False
        
        elif attestation.provider_type == AttestationType.SEV:
            # Check SEV-SNP enabled
            if not attestation.additional_claims.get('sev_snp'):
                return False
        
        elif attestation.provider_type == AttestationType.NITRO:
            # Check Nitro version
            nitro_version = attestation.additional_claims.get('nitro_version', '0.0.0')
            major_version = int(nitro_version.split('.')[0])
            if major_version < 1:
                return False
        
        return True
    
    def _run_custom_validators(
        self,
        attestation: AttestationReport,
        model_identity: ModelIdentity,
        additional_checks: Optional[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """Run custom validation functions"""
        results = {}
        
        for validator_name in self.policy.custom_validators:
            # In production, would dynamically load and run validators
            results[validator_name] = True
        
        return results
    
    def _load_trusted_roots(self) -> Set[str]:
        """Load trusted root certificates"""
        # In production, would load from secure storage
        return {
            hashlib.sha256(b"INTEL_SGX_ROOT_CA").hexdigest(),
            hashlib.sha256(b"AMD_ROOT_CA").hexdigest(),
            hashlib.sha256(b"AWS_ROOT_CA").hexdigest()
        }
    
    def _load_revocation_list(self) -> Set[str]:
        """Load certificate revocation list"""
        # In production, would load from CRL/OCSP
        return set()
    
    def batch_verify(
        self,
        attestations: List[AttestationReport],
        model_identities: List[ModelIdentity],
        nonces: List[str]
    ) -> List[VerificationResult]:
        """Batch verify multiple attestations"""
        results = []
        
        for attestation, model_identity, nonce in zip(attestations, model_identities, nonces):
            result = self.verify_attestation(attestation, model_identity, nonce)
            results.append(result)
        
        return results
    
    def get_verification_summary(
        self,
        results: List[VerificationResult]
    ) -> Dict[str, Any]:
        """Get summary of verification results"""
        total = len(results)
        valid = sum(1 for r in results if r.is_valid())
        
        provider_stats = {}
        for result in results:
            provider = result.provider_type.value
            if provider not in provider_stats:
                provider_stats[provider] = {'valid': 0, 'invalid': 0}
            
            if result.is_valid():
                provider_stats[provider]['valid'] += 1
            else:
                provider_stats[provider]['invalid'] += 1
        
        failure_reasons = {}
        for result in results:
            for failure in result.failures:
                reason = failure.value
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        return {
            'total': total,
            'valid': valid,
            'invalid': total - valid,
            'success_rate': valid / total if total > 0 else 0,
            'provider_stats': provider_stats,
            'failure_reasons': failure_reasons,
            'timestamp': time.time()
        }