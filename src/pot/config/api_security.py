"""
API Security Configuration Module

Provides configuration management for TEE providers, attestation policies,
vendor commitments, and security thresholds.
"""

import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Set, Tuple
from enum import Enum
from pathlib import Path

from ..security.tee_attestation import AttestationType, SecurityLevel
from ..security.attestation_verifier import AttestationPolicy
from ..api.secure_binding import BindingMode, BindingPolicy


class ConfigFormat(Enum):
    """Configuration file formats"""
    JSON = "json"
    YAML = "yaml"


@dataclass
class TEEProviderConfig:
    """Configuration for TEE providers"""
    provider_type: AttestationType
    enabled: bool
    config: Dict[str, Any]
    priority: int = 0  # Higher priority providers are tried first
    fallback_providers: List[AttestationType] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'provider_type': self.provider_type.value,
            'enabled': self.enabled,
            'config': self.config,
            'priority': self.priority,
            'fallback_providers': [p.value for p in self.fallback_providers]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TEEProviderConfig':
        """Create from dictionary"""
        return cls(
            provider_type=AttestationType(data['provider_type']),
            enabled=data['enabled'],
            config=data['config'],
            priority=data.get('priority', 0),
            fallback_providers=[
                AttestationType(p) for p in data.get('fallback_providers', [])
            ]
        )


@dataclass
class VendorCommitmentSchema:
    """Schema for vendor commitments"""
    vendor_name: str
    api_version: str
    endpoint: str
    key_id: str
    required_fields: List[str]
    optional_fields: List[str]
    signature_algorithm: str
    certificate_url: Optional[str] = None
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    def validate_commitment(self, commitment: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate commitment against schema"""
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in commitment:
                errors.append(f"Missing required field: {field}")
        
        # Apply validation rules
        for field, rule in self.validation_rules.items():
            if field in commitment:
                value = commitment[field]
                if 'type' in rule and not isinstance(value, rule['type']):
                    errors.append(f"Field {field} has wrong type")
                if 'min' in rule and value < rule['min']:
                    errors.append(f"Field {field} below minimum value")
                if 'max' in rule and value > rule['max']:
                    errors.append(f"Field {field} above maximum value")
        
        return len(errors) == 0, errors


@dataclass
class SecurityThresholds:
    """Security threshold settings"""
    min_security_level: SecurityLevel
    max_attestation_age_seconds: float
    max_verification_retries: int
    require_certificate_validation: bool
    require_tcb_updates: bool
    allow_test_mode: bool
    confidence_threshold: float  # 0.0 to 1.0
    
    @classmethod
    def strict(cls) -> 'SecurityThresholds':
        """Create strict security thresholds"""
        return cls(
            min_security_level=SecurityLevel.HIGH,
            max_attestation_age_seconds=3600,
            max_verification_retries=3,
            require_certificate_validation=True,
            require_tcb_updates=True,
            allow_test_mode=False,
            confidence_threshold=0.99
        )
    
    @classmethod
    def moderate(cls) -> 'SecurityThresholds':
        """Create moderate security thresholds"""
        return cls(
            min_security_level=SecurityLevel.MEDIUM,
            max_attestation_age_seconds=7200,
            max_verification_retries=5,
            require_certificate_validation=True,
            require_tcb_updates=False,
            allow_test_mode=False,
            confidence_threshold=0.95
        )
    
    @classmethod
    def relaxed(cls) -> 'SecurityThresholds':
        """Create relaxed security thresholds"""
        return cls(
            min_security_level=SecurityLevel.LOW,
            max_attestation_age_seconds=86400,
            max_verification_retries=10,
            require_certificate_validation=False,
            require_tcb_updates=False,
            allow_test_mode=True,
            confidence_threshold=0.90
        )


@dataclass
class APISecurityConfig:
    """Main API security configuration"""
    # TEE Configuration
    tee_providers: Dict[str, TEEProviderConfig]
    default_provider: AttestationType
    
    # Attestation Policy
    attestation_policy: AttestationPolicy
    
    # Binding Policy
    binding_policy: BindingPolicy
    
    # Vendor Commitments
    vendor_schemas: Dict[str, VendorCommitmentSchema]
    
    # Security Thresholds
    security_thresholds: SecurityThresholds
    
    # Advanced Settings
    enable_caching: bool = True
    cache_ttl_seconds: float = 3600
    enable_audit_logging: bool = True
    audit_log_path: Optional[Path] = None
    enable_metrics: bool = True
    metrics_endpoint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'tee_providers': {
                name: provider.to_dict()
                for name, provider in self.tee_providers.items()
            },
            'default_provider': self.default_provider.value,
            'attestation_policy': {
                'required_security_level': self.attestation_policy.required_security_level.value,
                'max_attestation_age': self.attestation_policy.max_attestation_age,
                'allowed_providers': [p.value for p in self.attestation_policy.allowed_providers],
                'required_measurements': self.attestation_policy.required_measurements,
                'allow_test_mode': self.attestation_policy.allow_test_mode,
                'enforce_certificate_validation': self.attestation_policy.enforce_certificate_validation,
                'enforce_tcb_updates': self.attestation_policy.enforce_tcb_updates,
                'custom_validators': self.attestation_policy.custom_validators
            },
            'binding_policy': {
                'mode': self.binding_policy.mode.value,
                'require_attestation': self.binding_policy.require_attestation,
                'max_binding_age': self.binding_policy.max_binding_age,
                'verify_on_bind': self.binding_policy.verify_on_bind,
                'store_full_transcript': self.binding_policy.store_full_transcript,
                'hash_sensitive_data': self.binding_policy.hash_sensitive_data
            },
            'vendor_schemas': {
                name: asdict(schema)
                for name, schema in self.vendor_schemas.items()
            },
            'security_thresholds': {
                'min_security_level': self.security_thresholds.min_security_level.value,
                'max_attestation_age_seconds': self.security_thresholds.max_attestation_age_seconds,
                'max_verification_retries': self.security_thresholds.max_verification_retries,
                'require_certificate_validation': self.security_thresholds.require_certificate_validation,
                'require_tcb_updates': self.security_thresholds.require_tcb_updates,
                'allow_test_mode': self.security_thresholds.allow_test_mode,
                'confidence_threshold': self.security_thresholds.confidence_threshold
            },
            'advanced': {
                'enable_caching': self.enable_caching,
                'cache_ttl_seconds': self.cache_ttl_seconds,
                'enable_audit_logging': self.enable_audit_logging,
                'audit_log_path': str(self.audit_log_path) if self.audit_log_path else None,
                'enable_metrics': self.enable_metrics,
                'metrics_endpoint': self.metrics_endpoint
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APISecurityConfig':
        """Create from dictionary"""
        # Parse TEE providers
        tee_providers = {}
        for name, provider_data in data.get('tee_providers', {}).items():
            tee_providers[name] = TEEProviderConfig.from_dict(provider_data)
        
        # Parse attestation policy
        policy_data = data.get('attestation_policy', {})
        attestation_policy = AttestationPolicy(
            required_security_level=SecurityLevel(policy_data.get('required_security_level', 'high')),
            max_attestation_age=policy_data.get('max_attestation_age', 3600),
            allowed_providers={
                AttestationType(p) for p in policy_data.get('allowed_providers', [])
            },
            required_measurements=policy_data.get('required_measurements', {}),
            allow_test_mode=policy_data.get('allow_test_mode', False),
            enforce_certificate_validation=policy_data.get('enforce_certificate_validation', True),
            enforce_tcb_updates=policy_data.get('enforce_tcb_updates', True),
            custom_validators=policy_data.get('custom_validators', [])
        )
        
        # Parse binding policy
        binding_data = data.get('binding_policy', {})
        binding_policy = BindingPolicy(
            mode=BindingMode(binding_data.get('mode', 'tee_attestation')),
            require_attestation=binding_data.get('require_attestation', True),
            max_binding_age=binding_data.get('max_binding_age', 3600),
            verify_on_bind=binding_data.get('verify_on_bind', True),
            store_full_transcript=binding_data.get('store_full_transcript', False),
            hash_sensitive_data=binding_data.get('hash_sensitive_data', True),
            attestation_policy=attestation_policy
        )
        
        # Parse vendor schemas
        vendor_schemas = {}
        for name, schema_data in data.get('vendor_schemas', {}).items():
            vendor_schemas[name] = VendorCommitmentSchema(**schema_data)
        
        # Parse security thresholds
        threshold_data = data.get('security_thresholds', {})
        security_thresholds = SecurityThresholds(
            min_security_level=SecurityLevel(threshold_data.get('min_security_level', 'high')),
            max_attestation_age_seconds=threshold_data.get('max_attestation_age_seconds', 3600),
            max_verification_retries=threshold_data.get('max_verification_retries', 3),
            require_certificate_validation=threshold_data.get('require_certificate_validation', True),
            require_tcb_updates=threshold_data.get('require_tcb_updates', True),
            allow_test_mode=threshold_data.get('allow_test_mode', False),
            confidence_threshold=threshold_data.get('confidence_threshold', 0.99)
        )
        
        # Parse advanced settings
        advanced = data.get('advanced', {})
        audit_log_path = advanced.get('audit_log_path')
        if audit_log_path:
            audit_log_path = Path(audit_log_path)
        
        return cls(
            tee_providers=tee_providers,
            default_provider=AttestationType(data.get('default_provider', 'sgx')),
            attestation_policy=attestation_policy,
            binding_policy=binding_policy,
            vendor_schemas=vendor_schemas,
            security_thresholds=security_thresholds,
            enable_caching=advanced.get('enable_caching', True),
            cache_ttl_seconds=advanced.get('cache_ttl_seconds', 3600),
            enable_audit_logging=advanced.get('enable_audit_logging', True),
            audit_log_path=audit_log_path,
            enable_metrics=advanced.get('enable_metrics', True),
            metrics_endpoint=advanced.get('metrics_endpoint')
        )
    
    def save(self, path: Path, format: ConfigFormat = ConfigFormat.JSON):
        """Save configuration to file"""
        data = self.to_dict()
        
        with open(path, 'w') as f:
            if format == ConfigFormat.JSON:
                json.dump(data, f, indent=2)
            elif format == ConfigFormat.YAML:
                yaml.dump(data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, path: Path) -> 'APISecurityConfig':
        """Load configuration from file"""
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.json']:
                data = json.load(f)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                # Try to detect format
                content = f.read()
                f.seek(0)
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    data = yaml.safe_load(content)
        
        return cls.from_dict(data)


def create_default_config() -> APISecurityConfig:
    """Create default API security configuration"""
    # Default TEE providers
    tee_providers = {
        'sgx': TEEProviderConfig(
            provider_type=AttestationType.SGX,
            enabled=True,
            config={
                'enclave_id': 'default_enclave',
                'mrenclave': None,  # Will be generated
                'mrsigner': None,
                'isv_prod_id': 1,
                'isv_svn': 1
            },
            priority=10,
            fallback_providers=[AttestationType.VENDOR]
        ),
        'sev': TEEProviderConfig(
            provider_type=AttestationType.SEV,
            enabled=True,
            config={
                'vm_id': 'default_vm',
                'policy': 0x01
            },
            priority=9,
            fallback_providers=[AttestationType.VENDOR]
        ),
        'nitro': TEEProviderConfig(
            provider_type=AttestationType.NITRO,
            enabled=True,
            config={
                'region': 'us-east-1',
                'instance_id': None  # Will be auto-detected
            },
            priority=8,
            fallback_providers=[AttestationType.VENDOR]
        ),
        'vendor': TEEProviderConfig(
            provider_type=AttestationType.VENDOR,
            enabled=True,
            config={
                'vendor_name': 'DefaultVendor',
                'api_endpoint': 'https://api.example.com',
                'api_version': 'v1'
            },
            priority=5,
            fallback_providers=[]
        )
    }
    
    # Default attestation policy
    attestation_policy = AttestationPolicy.default_policy()
    
    # Default binding policy
    binding_policy = BindingPolicy.strict_policy()
    
    # Default vendor schemas
    vendor_schemas = {
        'openai': VendorCommitmentSchema(
            vendor_name='OpenAI',
            api_version='v1',
            endpoint='https://api.openai.com',
            key_id='openai_key_1',
            required_fields=['model', 'timestamp', 'signature'],
            optional_fields=['usage', 'metadata'],
            signature_algorithm='HMAC-SHA256',
            certificate_url='https://api.openai.com/certs',
            validation_rules={
                'timestamp': {'type': float, 'min': 0},
                'model': {'type': str}
            }
        ),
        'anthropic': VendorCommitmentSchema(
            vendor_name='Anthropic',
            api_version='v1',
            endpoint='https://api.anthropic.com',
            key_id='anthropic_key_1',
            required_fields=['model', 'timestamp', 'signature'],
            optional_fields=['usage', 'metadata'],
            signature_algorithm='HMAC-SHA256',
            certificate_url='https://api.anthropic.com/certs',
            validation_rules={
                'timestamp': {'type': float, 'min': 0},
                'model': {'type': str}
            }
        )
    }
    
    # Default security thresholds
    security_thresholds = SecurityThresholds.moderate()
    
    return APISecurityConfig(
        tee_providers=tee_providers,
        default_provider=AttestationType.SGX,
        attestation_policy=attestation_policy,
        binding_policy=binding_policy,
        vendor_schemas=vendor_schemas,
        security_thresholds=security_thresholds,
        enable_caching=True,
        cache_ttl_seconds=3600,
        enable_audit_logging=True,
        audit_log_path=Path('logs/api_security_audit.log'),
        enable_metrics=True,
        metrics_endpoint='http://localhost:9090/metrics'
    )


def create_test_config() -> APISecurityConfig:
    """Create test configuration with relaxed settings"""
    config = create_default_config()
    
    # Add mock provider for testing
    config.tee_providers['mock'] = TEEProviderConfig(
        provider_type=AttestationType.MOCK,
        enabled=True,
        config={'mock_id': 'test_mock'},
        priority=100,  # Highest priority for testing
        fallback_providers=[]
    )
    
    # Use relaxed policies
    config.attestation_policy = AttestationPolicy.relaxed_policy()
    config.binding_policy = BindingPolicy.relaxed_policy()
    config.security_thresholds = SecurityThresholds.relaxed()
    
    # Set mock as default
    config.default_provider = AttestationType.MOCK
    
    return config


def validate_config(config: APISecurityConfig) -> tuple[bool, list[str]]:
    """Validate configuration for consistency and completeness"""
    errors = []
    
    # Check default provider is configured
    if config.default_provider.value not in [p.provider_type.value for p in config.tee_providers.values()]:
        errors.append(f"Default provider {config.default_provider.value} not in configured providers")
    
    # Check default provider is enabled
    for name, provider in config.tee_providers.items():
        if provider.provider_type == config.default_provider and not provider.enabled:
            errors.append(f"Default provider {config.default_provider.value} is not enabled")
    
    # Check attestation policy consistency
    if config.attestation_policy.required_security_level.value > config.security_thresholds.min_security_level.value:
        errors.append("Attestation policy requires higher security level than thresholds allow")
    
    # Check binding policy consistency
    if config.binding_policy.require_attestation and not config.tee_providers:
        errors.append("Binding policy requires attestation but no TEE providers configured")
    
    # Check vendor schemas have required fields
    for name, schema in config.vendor_schemas.items():
        if not schema.required_fields:
            errors.append(f"Vendor schema {name} has no required fields")
    
    # Check paths exist if specified
    if config.audit_log_path and not config.audit_log_path.parent.exists():
        errors.append(f"Audit log directory {config.audit_log_path.parent} does not exist")
    
    return len(errors) == 0, errors