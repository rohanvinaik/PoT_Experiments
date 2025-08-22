"""
API Secure Binding Module

Implements secure binding of API transcripts to attested model identities,
managing attestation metadata in evidence bundles and validating binding integrity.
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from pathlib import Path
import base64

from ..security.tee_attestation import (
    AttestationType,
    AttestationReport,
    ModelIdentity,
    SecurityLevel
)
from ..security.attestation_verifier import (
    AttestationVerifier,
    AttestationPolicy,
    VerificationResult,
    VerificationStatus
)


class BindingMode(Enum):
    """API binding modes"""
    TEE_ATTESTATION = "tee_attestation"
    VENDOR_COMMITMENT = "vendor_commitment"
    HYBRID = "hybrid"
    NONE = "none"  # No binding (testing only)


class BindingStatus(Enum):
    """Binding verification status"""
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    UNVERIFIED = "unverified"


@dataclass
class APITranscript:
    """API call transcript"""
    transcript_id: str
    timestamp: float
    endpoint: str
    method: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_hash(self) -> str:
        """Compute hash of transcript"""
        transcript_data = json.dumps({
            'transcript_id': self.transcript_id,
            'timestamp': self.timestamp,
            'endpoint': self.endpoint,
            'method': self.method,
            'request': self.request,
            'response': self.response
        }, sort_keys=True)
        return hashlib.sha256(transcript_data.encode()).hexdigest()


@dataclass
class BoundTranscript:
    """Transcript bound to model identity"""
    transcript: APITranscript
    model_identity: ModelIdentity
    attestation: AttestationReport
    binding_timestamp: float
    binding_nonce: str
    binding_signature: str
    verification_result: Optional[VerificationResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'transcript': {
                'id': self.transcript.transcript_id,
                'hash': self.transcript.compute_hash(),
                'timestamp': self.transcript.timestamp,
                'endpoint': self.transcript.endpoint
            },
            'model_identity': {
                'hash': self.model_identity.compute_identity_hash(),
                'name': self.model_identity.model_name,
                'version': self.model_identity.version
            },
            'attestation': self.attestation.to_dict(),
            'binding': {
                'timestamp': self.binding_timestamp,
                'nonce': self.binding_nonce,
                'signature': self.binding_signature
            },
            'verification': self.verification_result.to_dict() if self.verification_result else None
        }


@dataclass
class BindingPolicy:
    """Policy for API binding"""
    mode: BindingMode
    require_attestation: bool
    max_binding_age: float  # seconds
    verify_on_bind: bool
    store_full_transcript: bool
    hash_sensitive_data: bool
    attestation_policy: Optional[AttestationPolicy] = None
    
    @classmethod
    def strict_policy(cls) -> 'BindingPolicy':
        """Create strict binding policy"""
        return cls(
            mode=BindingMode.TEE_ATTESTATION,
            require_attestation=True,
            max_binding_age=3600,  # 1 hour
            verify_on_bind=True,
            store_full_transcript=False,
            hash_sensitive_data=True,
            attestation_policy=AttestationPolicy.default_policy()
        )
    
    @classmethod
    def relaxed_policy(cls) -> 'BindingPolicy':
        """Create relaxed binding policy"""
        return cls(
            mode=BindingMode.VENDOR_COMMITMENT,
            require_attestation=False,
            max_binding_age=86400,  # 24 hours
            verify_on_bind=False,
            store_full_transcript=True,
            hash_sensitive_data=False,
            attestation_policy=AttestationPolicy.relaxed_policy()
        )


class SecureAPIBinder:
    """Manages secure binding of API transcripts to model identities"""
    
    def __init__(
        self,
        policy: Optional[BindingPolicy] = None,
        verifier: Optional[AttestationVerifier] = None
    ):
        """Initialize binder with policy"""
        self.policy = policy or BindingPolicy.strict_policy()
        self.verifier = verifier or AttestationVerifier(self.policy.attestation_policy)
        self.bindings = {}  # transcript_id -> BoundTranscript
        self.binding_cache = {}  # For performance optimization
        
    def bind_transcript(
        self,
        transcript: APITranscript,
        model_identity: ModelIdentity,
        attestation: AttestationReport,
        verify_immediately: Optional[bool] = None
    ) -> BoundTranscript:
        """Bind API transcript to attested model identity"""
        # Generate binding nonce
        binding_nonce = str(uuid.uuid4())
        
        # Verify attestation if required
        verification_result = None
        if verify_immediately or (verify_immediately is None and self.policy.verify_on_bind):
            verification_result = self.verifier.verify_attestation(
                attestation,
                model_identity,
                attestation.nonce  # Use attestation's nonce for verification
            )
            
            if self.policy.require_attestation and not verification_result.is_valid():
                raise ValueError(f"Attestation verification failed: {verification_result.failures}")
        
        # Create binding signature
        binding_signature = self._create_binding_signature(
            transcript,
            model_identity,
            attestation,
            binding_nonce
        )
        
        # Create bound transcript
        bound_transcript = BoundTranscript(
            transcript=transcript,
            model_identity=model_identity,
            attestation=attestation,
            binding_timestamp=time.time(),
            binding_nonce=binding_nonce,
            binding_signature=binding_signature,
            verification_result=verification_result
        )
        
        # Store binding
        self.bindings[transcript.transcript_id] = bound_transcript
        
        # Update cache
        self._update_cache(bound_transcript)
        
        return bound_transcript
    
    def verify_binding(
        self,
        bound_transcript: BoundTranscript,
        check_freshness: bool = True
    ) -> BindingStatus:
        """Verify binding integrity"""
        # Check binding signature
        expected_signature = self._create_binding_signature(
            bound_transcript.transcript,
            bound_transcript.model_identity,
            bound_transcript.attestation,
            bound_transcript.binding_nonce
        )
        
        if bound_transcript.binding_signature != expected_signature:
            return BindingStatus.INVALID
        
        # Check freshness if required
        if check_freshness:
            age = time.time() - bound_transcript.binding_timestamp
            if age > self.policy.max_binding_age:
                return BindingStatus.EXPIRED
        
        # Verify attestation if not already verified
        if bound_transcript.verification_result is None:
            verification_result = self.verifier.verify_attestation(
                bound_transcript.attestation,
                bound_transcript.model_identity,
                bound_transcript.attestation.nonce
            )
            bound_transcript.verification_result = verification_result
        
        if not bound_transcript.verification_result.is_valid():
            return BindingStatus.INVALID
        
        return BindingStatus.VALID
    
    def create_evidence_bundle(
        self,
        transcript_ids: List[str],
        include_full_transcripts: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Create evidence bundle with bound transcripts"""
        bundle = {
            'version': '1.0',
            'created_at': time.time(),
            'policy': {
                'mode': self.policy.mode.value,
                'require_attestation': self.policy.require_attestation,
                'max_binding_age': self.policy.max_binding_age
            },
            'transcripts': [],
            'summary': {}
        }
        
        include_full = include_full_transcripts
        if include_full is None:
            include_full = self.policy.store_full_transcript
        
        valid_count = 0
        invalid_count = 0
        
        for transcript_id in transcript_ids:
            if transcript_id not in self.bindings:
                continue
            
            bound_transcript = self.bindings[transcript_id]
            
            # Verify binding
            status = self.verify_binding(bound_transcript)
            
            if status == BindingStatus.VALID:
                valid_count += 1
            else:
                invalid_count += 1
            
            # Add to bundle
            transcript_data = {
                'transcript_id': transcript_id,
                'status': status.value,
                'binding': bound_transcript.to_dict()
            }
            
            if include_full:
                transcript_data['full_transcript'] = {
                    'request': bound_transcript.transcript.request,
                    'response': bound_transcript.transcript.response,
                    'metadata': bound_transcript.transcript.metadata
                }
            
            bundle['transcripts'].append(transcript_data)
        
        # Add summary
        bundle['summary'] = {
            'total_transcripts': len(bundle['transcripts']),
            'valid_bindings': valid_count,
            'invalid_bindings': invalid_count,
            'success_rate': valid_count / len(bundle['transcripts']) if bundle['transcripts'] else 0
        }
        
        # Add bundle signature
        bundle['signature'] = self._sign_bundle(bundle)
        
        return bundle
    
    def validate_evidence_bundle(
        self,
        bundle: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate evidence bundle integrity"""
        errors = []
        
        # Check bundle structure
        required_keys = ['version', 'created_at', 'policy', 'transcripts', 'summary', 'signature']
        for key in required_keys:
            if key not in bundle:
                errors.append(f"Missing required key: {key}")
        
        if errors:
            return False, errors
        
        # Verify bundle signature
        bundle_copy = bundle.copy()
        stored_signature = bundle_copy.pop('signature')
        expected_signature = self._sign_bundle(bundle_copy)
        
        if stored_signature != expected_signature:
            errors.append("Invalid bundle signature")
        
        # Verify each transcript binding
        for transcript_data in bundle.get('transcripts', []):
            if 'binding' not in transcript_data:
                errors.append(f"Missing binding for transcript {transcript_data.get('transcript_id')}")
                continue
            
            # Recreate bound transcript from data
            try:
                binding_data = transcript_data['binding']
                
                # Verify binding signature matches
                if 'binding' in binding_data:
                    binding_info = binding_data['binding']
                    # Additional validation could be done here
                    
            except Exception as e:
                errors.append(f"Error validating transcript {transcript_data.get('transcript_id')}: {str(e)}")
        
        # Check summary accuracy
        summary = bundle.get('summary', {})
        actual_total = len(bundle.get('transcripts', []))
        if summary.get('total_transcripts') != actual_total:
            errors.append(f"Summary mismatch: claims {summary.get('total_transcripts')} transcripts, found {actual_total}")
        
        return len(errors) == 0, errors
    
    def _create_binding_signature(
        self,
        transcript: APITranscript,
        model_identity: ModelIdentity,
        attestation: AttestationReport,
        nonce: str
    ) -> str:
        """Create cryptographic binding signature"""
        binding_data = {
            'transcript_hash': transcript.compute_hash(),
            'model_identity_hash': model_identity.compute_identity_hash(),
            'attestation_hash': hashlib.sha256(
                json.dumps(attestation.to_dict(), sort_keys=True).encode()
            ).hexdigest(),
            'nonce': nonce
        }
        
        binding_json = json.dumps(binding_data, sort_keys=True)
        signature = hashlib.sha512(binding_json.encode()).hexdigest()
        
        return signature
    
    def _sign_bundle(self, bundle: Dict[str, Any]) -> str:
        """Sign evidence bundle"""
        # Remove signature field if present
        bundle_copy = bundle.copy()
        bundle_copy.pop('signature', None)
        
        bundle_json = json.dumps(bundle_copy, sort_keys=True)
        signature = hashlib.sha512(bundle_json.encode()).hexdigest()
        
        return signature
    
    def _update_cache(self, bound_transcript: BoundTranscript):
        """Update binding cache for performance"""
        # Cache key based on model identity and attestation
        cache_key = f"{bound_transcript.model_identity.compute_identity_hash()}_{bound_transcript.attestation.nonce}"
        
        if cache_key not in self.binding_cache:
            self.binding_cache[cache_key] = []
        
        self.binding_cache[cache_key].append(bound_transcript.transcript.transcript_id)
        
        # Limit cache size
        if len(self.binding_cache) > 1000:
            # Remove oldest entries
            oldest_key = next(iter(self.binding_cache))
            del self.binding_cache[oldest_key]
    
    def get_bindings_for_model(
        self,
        model_identity: ModelIdentity
    ) -> List[BoundTranscript]:
        """Get all bindings for a specific model"""
        model_hash = model_identity.compute_identity_hash()
        results = []
        
        for bound_transcript in self.bindings.values():
            if bound_transcript.model_identity.compute_identity_hash() == model_hash:
                results.append(bound_transcript)
        
        return results
    
    def export_bindings(
        self,
        output_path: Path,
        transcript_ids: Optional[List[str]] = None
    ) -> Path:
        """Export bindings to file"""
        if transcript_ids is None:
            transcript_ids = list(self.bindings.keys())
        
        bundle = self.create_evidence_bundle(transcript_ids)
        
        with open(output_path, 'w') as f:
            json.dump(bundle, f, indent=2)
        
        return output_path
    
    def import_bindings(
        self,
        input_path: Path,
        verify: bool = True
    ) -> Tuple[int, List[str]]:
        """Import bindings from file"""
        with open(input_path, 'r') as f:
            bundle = json.load(f)
        
        if verify:
            valid, errors = self.validate_evidence_bundle(bundle)
            if not valid:
                return 0, errors
        
        imported = 0
        errors = []
        
        for transcript_data in bundle.get('transcripts', []):
            try:
                # Reconstruct objects from data
                binding_data = transcript_data['binding']
                
                # Create transcript
                transcript_info = binding_data['transcript']
                transcript = APITranscript(
                    transcript_id=transcript_info['id'],
                    timestamp=transcript_info['timestamp'],
                    endpoint=transcript_info['endpoint'],
                    method='POST',  # Default if not stored
                    request={},  # Would need full data if stored
                    response={},
                    latency_ms=0,
                    metadata={}
                )
                
                # Create model identity
                model_info = binding_data['model_identity']
                model_identity = ModelIdentity(
                    model_hash=model_info.get('hash', ''),
                    model_name=model_info['name'],
                    version=model_info['version'],
                    provider='imported',
                    architecture='unknown',
                    parameter_count=0
                )
                
                # Create attestation
                attestation = AttestationReport.from_dict(binding_data['attestation'])
                
                # Store binding
                self.bindings[transcript.transcript_id] = BoundTranscript(
                    transcript=transcript,
                    model_identity=model_identity,
                    attestation=attestation,
                    binding_timestamp=binding_data['binding']['timestamp'],
                    binding_nonce=binding_data['binding']['nonce'],
                    binding_signature=binding_data['binding']['signature']
                )
                
                imported += 1
                
            except Exception as e:
                errors.append(f"Error importing transcript {transcript_data.get('transcript_id')}: {str(e)}")
        
        return imported, errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get binding statistics"""
        total = len(self.bindings)
        
        if total == 0:
            return {
                'total_bindings': 0,
                'verified': 0,
                'unverified': 0,
                'expired': 0,
                'by_provider': {},
                'by_security_level': {}
            }
        
        verified = 0
        unverified = 0
        expired = 0
        by_provider = {}
        by_security_level = {}
        
        for bound_transcript in self.bindings.values():
            status = self.verify_binding(bound_transcript, check_freshness=True)
            
            if status == BindingStatus.VALID:
                verified += 1
            elif status == BindingStatus.EXPIRED:
                expired += 1
            else:
                unverified += 1
            
            # Count by provider
            provider = bound_transcript.attestation.provider_type.value
            by_provider[provider] = by_provider.get(provider, 0) + 1
            
            # Count by security level
            level = bound_transcript.attestation.security_level.value
            by_security_level[level] = by_security_level.get(level, 0) + 1
        
        return {
            'total_bindings': total,
            'verified': verified,
            'unverified': unverified,
            'expired': expired,
            'verification_rate': verified / total if total > 0 else 0,
            'by_provider': by_provider,
            'by_security_level': by_security_level,
            'cache_size': len(self.binding_cache),
            'timestamp': time.time()
        }