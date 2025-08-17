"""JSON schema definitions for POT audit records."""

import copy
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

# Legacy audit schema for backward compatibility
LEGACY_AUDIT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "POT Audit Record",
    "description": "Schema for Proof-of-Training verification audit records",
    "type": "object",
    "required": [
        "session_id",
        "model_id",
        "family",
        "alpha",
        "beta",
        "boundary",
        "nonce",
        "commitment",
        "prf_info",
        "reuse_policy",
        "env",
        "timestamp"
    ],
    "properties": {
        "session_id": {
            "type": "string",
            "description": "Unique identifier for the verification session",
            "pattern": "^session_[a-f0-9]{16}$"
        },
        "model_id": {
            "type": "string",
            "description": "Identifier of the model being verified",
            "minLength": 1
        },
        "family": {
            "type": "string",
            "description": "Model family (e.g., resnet, vit, bert)",
            "minLength": 1
        },
        "alpha": {
            "type": "number",
            "description": "Type I error rate (false acceptance)",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "beta": {
            "type": "number",
            "description": "Type II error rate (false rejection)",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "boundary": {
            "type": "number",
            "description": "Decision boundary threshold",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "nonce": {
            "type": "string",
            "description": "Random nonce for commitment (hex encoded)",
            "pattern": "^[a-f0-9]+$"
        },
        "commitment": {
            "type": "string",
            "description": "HMAC-SHA256 commitment hash (hex encoded)",
            "pattern": "^[a-f0-9]{64}$"
        },
        "prf_info": {
            "type": "object",
            "description": "PRF configuration information",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "description": "PRF algorithm used"
                },
                "key_derivation": {
                    "type": "string",
                    "description": "Key derivation method"
                },
                "seed_length": {
                    "type": "integer",
                    "description": "Length of PRF seed in bytes",
                    "minimum": 16
                }
            },
            "additionalProperties": True
        },
        "reuse_policy": {
            "type": "string",
            "description": "Challenge reuse policy",
            "enum": ["never", "session", "model", "family", "always"]
        },
        "env": {
            "type": "object",
            "description": "Environment information",
            "properties": {
                "python_version": {
                    "type": "string",
                    "description": "Python version used"
                },
                "torch_version": {
                    "type": "string",
                    "description": "PyTorch version (if applicable)"
                },
                "cuda_version": {
                    "type": "string",
                    "description": "CUDA version (if applicable)"
                },
                "platform": {
                    "type": "string",
                    "description": "Operating system platform"
                },
                "hostname": {
                    "type": "string",
                    "description": "Hostname of verification machine"
                },
                "timestamp_utc": {
                    "type": "string",
                    "description": "UTC timestamp of environment capture"
                }
            },
            "additionalProperties": True
        },
        "artifacts": {
            "type": "object",
            "description": "Optional additional artifacts",
            "properties": {
                "challenges_hash": {
                    "type": "string",
                    "description": "Hash of all challenges used"
                },
                "ranges": {
                    "type": "array",
                    "description": "Challenge index ranges used",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        },
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "verification_result": {
                    "type": "object",
                    "description": "Verification outcome details",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": ["accept", "reject", "inconclusive"]
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "num_challenges": {
                            "type": "integer",
                            "minimum": 1
                        },
                        "duration_seconds": {
                            "type": "number",
                            "minimum": 0.0
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata",
                    "additionalProperties": True
                }
            },
            "additionalProperties": True
        },
        "timestamp": {
            "type": "string",
            "description": "ISO 8601 timestamp of audit record creation",
            "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?Z$"
        }
    },
    "additionalProperties": False
}


# Enhanced audit schema for commit-reveal protocol (UPDATED 2025-08-17)
AUDIT_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "POT Enhanced Audit Record",
    "description": "Comprehensive schema for Proof-of-Training verification audit records with commit-reveal support",
    "type": "object",
    "required": ["commitment", "timestamp", "verification_result", "metadata"],
    "properties": {
        "commitment": {
            "type": "object",
            "description": "Cryptographic commitment information",
            "required": ["hash", "algorithm", "salt_length"],
            "properties": {
                "hash": {
                    "type": "string",
                    "description": "Commitment hash (hex encoded)",
                    "pattern": "^[a-f0-9]{64}$"
                },
                "algorithm": {
                    "type": "string",
                    "description": "Hash algorithm used for commitment",
                    "enum": ["SHA256", "SHA3-256"]
                },
                "salt_length": {
                    "type": "integer",
                    "description": "Length of salt in bytes",
                    "minimum": 16,
                    "maximum": 64
                },
                "version": {
                    "type": "string",
                    "description": "Commitment protocol version",
                    "pattern": "^\\d+\\.\\d+$"
                },
                "nonce": {
                    "type": "string",
                    "description": "Random nonce (hex encoded)",
                    "pattern": "^[a-f0-9]+$",
                    "minLength": 32
                }
            },
            "additionalProperties": False
        },
        "timestamp": {
            "type": "string",
            "description": "ISO 8601 timestamp with timezone",
            "format": "date-time",
            "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?Z$"
        },
        "verification_result": {
            "type": "object",
            "description": "Verification outcome and metrics",
            "required": ["decision", "confidence", "samples_used"],
            "properties": {
                "decision": {
                    "type": "string",
                    "description": "Verification decision",
                    "enum": ["PASS", "FAIL", "INCONCLUSIVE"]
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence level (0-1)",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "samples_used": {
                    "type": "integer",
                    "description": "Number of verification samples",
                    "minimum": 1
                },
                "early_stopping": {
                    "type": "boolean",
                    "description": "Whether sequential testing stopped early"
                },
                "p_value": {
                    "type": "number",
                    "description": "Statistical p-value (if computed)",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "test_statistic": {
                    "type": "number",
                    "description": "Test statistic value"
                },
                "effect_size": {
                    "type": "number",
                    "description": "Observed effect size"
                },
                "duration_seconds": {
                    "type": "number",
                    "description": "Verification duration",
                    "minimum": 0.0
                },
                "sequential_trajectory": {
                    "type": "array",
                    "description": "Sequential test trajectory",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sample": {"type": "integer"},
                            "mean": {"type": "number"},
                            "variance": {"type": "number"},
                            "confidence_radius": {"type": "number"}
                        }
                    }
                }
            },
            "additionalProperties": False
        },
        "metadata": {
            "type": "object",
            "description": "Verification metadata and context",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Model identifier",
                    "minLength": 1
                },
                "verifier_version": {
                    "type": "string",
                    "description": "Verifier software version",
                    "pattern": "^\\d+\\.\\d+\\.\\d+.*$"
                },
                "expected_ranges": {
                    "type": "object",
                    "description": "Expected value ranges for verification",
                    "properties": {
                        "alpha": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "beta": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "tau": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "additionalProperties": False
                },
                "challenge_info": {
                    "type": "object",
                    "description": "Challenge generation metadata",
                    "properties": {
                        "family": {
                            "type": "string",
                            "enum": ["vision:freq", "vision:texture", "lm:templates"]
                        },
                        "count": {
                            "type": "integer",
                            "minimum": 1
                        },
                        "generation_seed": {
                            "type": "string",
                            "pattern": "^[a-f0-9]+$"
                        },
                        "parameters": {
                            "type": "object",
                            "additionalProperties": True
                        }
                    }
                },
                "environment": {
                    "type": "object",
                    "description": "Execution environment information",
                    "properties": {
                        "python_version": {"type": "string"},
                        "framework_version": {"type": "string"},
                        "platform": {"type": "string"},
                        "hostname": {"type": "string"},
                        "gpu_info": {"type": "string"},
                        "memory_gb": {"type": "number", "minimum": 0}
                    },
                    "additionalProperties": True
                },
                "fingerprint": {
                    "type": "object",
                    "description": "Behavioral fingerprint metadata",
                    "properties": {
                        "io_hash": {
                            "type": "string",
                            "pattern": "^[a-f0-9]{64}$"
                        },
                        "jacobian_sketch": {
                            "type": "string",
                            "pattern": "^[a-f0-9]+$"
                        },
                        "similarity_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "timing_ms": {
                            "type": "number",
                            "minimum": 0.0
                        }
                    },
                    "additionalProperties": False
                }
            },
            "additionalProperties": True
        },
        "session_info": {
            "type": "object",
            "description": "Session and audit trail information",
            "properties": {
                "session_id": {
                    "type": "string",
                    "pattern": "^session_[a-f0-9]{16}$"
                },
                "audit_version": {
                    "type": "string",
                    "description": "Audit schema version"
                },
                "created_by": {
                    "type": "string",
                    "description": "Entity that created the audit record"
                },
                "integrity_hash": {
                    "type": "string",
                    "description": "Hash for integrity verification",
                    "pattern": "^[a-f0-9]{16,64}$"
                }
            }
        }
    },
    "additionalProperties": False
}

# Commitment record schema for commit-reveal protocol
COMMITMENT_RECORD_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "POT Commitment Record",
    "description": "Schema for cryptographic commitment records",
    "type": "object",
    "required": ["commitment_hash", "timestamp", "salt", "version"],
    "properties": {
        "commitment_hash": {
            "type": "string",
            "description": "SHA256 commitment hash (hex encoded)",
            "pattern": "^[a-f0-9]{64}$"
        },
        "timestamp": {
            "type": "string",
            "description": "ISO 8601 timestamp",
            "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?Z$"
        },
        "salt": {
            "type": "string",
            "description": "Cryptographic salt (hex encoded)",
            "pattern": "^[a-f0-9]{64}$"
        },
        "version": {
            "type": "string",
            "description": "Protocol version",
            "pattern": "^\\d+\\.\\d+$"
        },
        "metadata": {
            "type": "object",
            "description": "Additional commitment metadata",
            "properties": {
                "data_size": {
                    "type": "integer",
                    "minimum": 0
                },
                "salt_size": {
                    "type": "integer",
                    "minimum": 16
                },
                "hash_algorithm": {
                    "type": "string",
                    "enum": ["SHA256", "SHA3-256"]
                }
            },
            "additionalProperties": True
        },
        "integrity": {
            "type": "object",
            "description": "File integrity metadata",
            "properties": {
                "format_version": {"type": "string"},
                "written_at": {"type": "string"},
                "checksum": {
                    "type": "string",
                    "pattern": "^[a-f0-9]{16}$"
                }
            }
        }
    },
    "additionalProperties": False
}


def validate_audit_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate an audit record against the appropriate schema.
    
    Automatically detects record type (legacy, enhanced, or commitment)
    and validates against the appropriate schema.
    
    Args:
        record: Audit record dictionary to validate
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    try:
        import jsonschema
        
        # Detect record type and choose appropriate schema
        schema = _detect_and_get_schema(record)
        
        # Validate against detected schema
        jsonschema.validate(instance=record, schema=schema)
        
        # Additional custom validations
        custom_errors = _custom_validations(record)
        if custom_errors:
            return False, custom_errors
            
        return True, []
    
    except ImportError:
        # Fallback validation without jsonschema
        return _fallback_validation(record)
    
    except jsonschema.exceptions.ValidationError as e:
        return False, [f"Schema validation failed: {str(e)}"]
    except Exception as e:
        return False, [f"Validation error: {str(e)}"]


def _detect_and_get_schema(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect record type and return appropriate schema.
    
    Args:
        record: Record to analyze
        
    Returns:
        Appropriate JSON schema
    """
    # Check for commitment record
    if 'commitment_hash' in record and 'salt' in record:
        return COMMITMENT_RECORD_SCHEMA
    
    # Check for enhanced audit record
    elif 'commitment' in record and isinstance(record.get('commitment'), dict):
        return AUDIT_JSON_SCHEMA
    
    # Check for legacy audit record  
    elif 'session_id' in record and 'nonce' in record:
        return LEGACY_AUDIT_SCHEMA
    
    # Default to enhanced schema
    else:
        return AUDIT_JSON_SCHEMA


def _custom_validations(record: Dict[str, Any]) -> List[str]:
    """
    Perform custom validations beyond JSON schema.
    
    Args:
        record: Record to validate
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Validate timestamp format and recency
    if 'timestamp' in record:
        timestamp_errors = _validate_timestamp(record['timestamp'])
        errors.extend(timestamp_errors)
    
    # Validate hex strings
    hex_fields = ['commitment_hash', 'salt']
    for field in hex_fields:
        if field in record:
            if not _is_valid_hex(record[field]):
                errors.append(f"{field} must be valid hex string")
    
    # Validate nested commitment object
    if 'commitment' in record and isinstance(record['commitment'], dict):
        commitment_errors = _validate_commitment_object(record['commitment'])
        errors.extend(commitment_errors)
    
    # Validate verification result consistency
    if 'verification_result' in record:
        result_errors = _validate_verification_result(record['verification_result'])
        errors.extend(result_errors)
    
    return errors


def _validate_timestamp(timestamp: str) -> List[str]:
    """Validate timestamp format and recency."""
    errors = []
    
    try:
        # Parse ISO 8601 timestamp
        dt = datetime.fromisoformat(timestamp.rstrip('Z'))
        
        # Check if timestamp is reasonable (not too far in future/past)
        now = datetime.utcnow()
        diff = abs((now - dt).total_seconds())
        
        # Allow 24 hours of clock skew
        if diff > 86400:
            errors.append(f"Timestamp too far from current time: {timestamp}")
            
    except ValueError:
        errors.append(f"Invalid timestamp format: {timestamp}")
    
    return errors


def _is_valid_hex(value: str) -> bool:
    """Check if string is valid hexadecimal."""
    if not isinstance(value, str):
        return False
    try:
        int(value, 16)
        return True
    except ValueError:
        return False


def _validate_commitment_object(commitment: Dict[str, Any]) -> List[str]:
    """Validate commitment object structure."""
    errors = []
    
    # Check required fields
    required = ['hash', 'algorithm', 'salt_length']
    for field in required:
        if field not in commitment:
            errors.append(f"Commitment missing required field: {field}")
    
    # Validate hash length for algorithm
    if 'hash' in commitment and 'algorithm' in commitment:
        expected_length = 64 if commitment['algorithm'] in ['SHA256', 'SHA3-256'] else None
        if expected_length and len(commitment['hash']) != expected_length:
            errors.append(f"Hash length {len(commitment['hash'])} doesn't match algorithm {commitment['algorithm']}")
    
    # Validate salt length
    if 'salt_length' in commitment:
        if not isinstance(commitment['salt_length'], int) or commitment['salt_length'] < 16:
            errors.append("Salt length must be integer >= 16")
    
    return errors


def _validate_verification_result(result: Dict[str, Any]) -> List[str]:
    """Validate verification result consistency."""
    errors = []
    
    # Check confidence vs decision consistency
    if 'confidence' in result and 'decision' in result:
        confidence = result['confidence']
        decision = result['decision']
        
        # High confidence should not lead to INCONCLUSIVE
        if confidence > 0.9 and decision == 'INCONCLUSIVE':
            errors.append("High confidence with INCONCLUSIVE decision is inconsistent")
        
        # Very low confidence should not lead to definitive decisions
        if confidence < 0.1 and decision in ['PASS', 'FAIL']:
            errors.append("Very low confidence with definitive decision is suspicious")
    
    # Check samples used vs duration consistency
    if 'samples_used' in result and 'duration_seconds' in result:
        samples = result['samples_used']
        duration = result['duration_seconds']
        
        # Sanity check: too many samples in too little time
        if samples > 1000 and duration < 1.0:
            errors.append("Suspicious: too many samples processed too quickly")
    
    return errors


def _fallback_validation(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Fallback validation when jsonschema is not available.
    
    Args:
        record: Record to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Detect record type
    if 'commitment_hash' in record and 'salt' in record:
        # Commitment record validation
        required_fields = ['commitment_hash', 'timestamp', 'salt', 'version']
        for field in required_fields:
            if field not in record:
                errors.append(f"Missing required field: {field}")
        
        # Validate hex strings
        if 'commitment_hash' in record and len(record['commitment_hash']) != 64:
            errors.append("commitment_hash must be 64 hex characters")
        if 'salt' in record and len(record['salt']) != 64:
            errors.append("salt must be 64 hex characters")
            
    elif 'commitment' in record and isinstance(record.get('commitment'), dict):
        # Enhanced audit record validation
        required_fields = ['commitment', 'timestamp', 'verification_result', 'metadata']
        for field in required_fields:
            if field not in record:
                errors.append(f"Missing required field: {field}")
        
        # Validate commitment object
        commitment = record.get('commitment', {})
        commitment_required = ['hash', 'algorithm', 'salt_length']
        for field in commitment_required:
            if field not in commitment:
                errors.append(f"Commitment missing required field: {field}")
    
    else:
        # Legacy audit record validation
        required_fields = ['session_id', 'model_id', 'family', 'alpha', 'beta', 'boundary', 'nonce', 'commitment', 'prf_info', 'reuse_policy', 'env', 'timestamp']
        for field in required_fields:
            if field not in record:
                errors.append(f"Missing required field: {field}")
        
        # Basic type checks
        if "alpha" in record and not (0 <= record["alpha"] <= 1):
            errors.append("alpha must be between 0 and 1")
        if "beta" in record and not (0 <= record["beta"] <= 1):
            errors.append("beta must be between 0 and 1")
        if "boundary" in record and not (0 <= record["boundary"] <= 1):
            errors.append("boundary must be between 0 and 1")
    
    # Common validations
    custom_errors = _custom_validations(record)
    errors.extend(custom_errors)
    
    return len(errors) == 0, errors


def sanitize_for_audit(data: Any) -> Dict[str, Any]:
    """
    Sanitize data for audit record creation.
    
    Removes sensitive information, ensures JSON serializable,
    and maintains audit integrity.
    
    Args:
        data: Raw data to be included in audit record
        
    Returns:
        Sanitized dictionary suitable for audit records
    """
    if data is None:
        return {}
    
    # Convert to dictionary if not already
    if not isinstance(data, dict):
        data = {'value': data}
    
    # Deep copy to avoid modifying original
    sanitized = copy.deepcopy(data)
    
    # Remove sensitive fields
    sensitive_fields = [
        'password', 'secret', 'key', 'token', 'api_key',
        'private_key', 'master_key', 'session_key', 'auth_token',
        'credentials', 'authentication', 'authorization'
    ]
    
    _remove_sensitive_fields(sanitized, sensitive_fields)
    
    # Ensure JSON serializable
    sanitized = _make_json_serializable(sanitized)
    
    # Truncate large values
    sanitized = _truncate_large_values(sanitized)
    
    # Add sanitization metadata
    sanitized['_audit_metadata'] = {
        'sanitized_at': datetime.utcnow().isoformat() + 'Z',
        'sanitization_version': '1.0',
        'original_type': type(data).__name__
    }
    
    return sanitized


def _remove_sensitive_fields(data: Dict[str, Any], sensitive_fields: List[str]) -> None:
    """Remove sensitive fields from nested dictionary."""
    if not isinstance(data, dict):
        return
    
    # Check each key
    keys_to_remove = []
    for key in data.keys():
        if isinstance(key, str):
            # Check if key contains sensitive terms
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_fields):
                keys_to_remove.append(key)
    
    # Remove sensitive keys
    for key in keys_to_remove:
        data[key] = '[REDACTED]'
    
    # Recursively process nested dictionaries and lists
    for key, value in data.items():
        if isinstance(value, dict):
            _remove_sensitive_fields(value, sensitive_fields)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _remove_sensitive_fields(item, sensitive_fields)


def _make_json_serializable(obj: Any) -> Any:
    """Convert object to JSON serializable format."""
    if obj is None:
        return None
    elif isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        # Convert object with attributes to dict
        return _make_json_serializable(obj.__dict__)
    elif hasattr(obj, 'tolist'):
        # Handle numpy arrays
        return _make_json_serializable(obj.tolist())
    elif hasattr(obj, 'item'):
        # Handle numpy scalars
        return _make_json_serializable(obj.item())
    else:
        # Convert to string as fallback
        return str(obj)


def _truncate_large_values(data: Dict[str, Any], max_length: int = 1000) -> Dict[str, Any]:
    """Truncate large string values to prevent excessive audit record size."""
    if not isinstance(data, dict):
        return data
    
    result = {}
    for key, value in data.items():
        if isinstance(value, str) and len(value) > max_length:
            result[key] = value[:max_length] + f'... [TRUNCATED {len(value)-max_length} chars]'
        elif isinstance(value, dict):
            result[key] = _truncate_large_values(value, max_length)
        elif isinstance(value, list):
            result[key] = [
                _truncate_large_values(item, max_length) if isinstance(item, dict)
                else (item[:max_length] + f'... [TRUNCATED]' if isinstance(item, str) and len(item) > max_length else item)
                for item in value
            ]
        else:
            result[key] = value
    
    return result


def validate_commitment_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a commitment record specifically.
    
    Args:
        record: Commitment record to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    try:
        import jsonschema
        jsonschema.validate(instance=record, schema=COMMITMENT_RECORD_SCHEMA)
        return True, []
    except ImportError:
        return _fallback_commitment_validation(record)
    except jsonschema.exceptions.ValidationError as e:
        return False, [f"Commitment validation failed: {str(e)}"]
    except Exception as e:
        return False, [f"Validation error: {str(e)}"]


def _fallback_commitment_validation(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Fallback validation for commitment records."""
    errors = []
    
    required_fields = ['commitment_hash', 'timestamp', 'salt', 'version']
    for field in required_fields:
        if field not in record:
            errors.append(f"Missing required field: {field}")
    
    # Validate hex strings
    if 'commitment_hash' in record:
        if not _is_valid_hex(record['commitment_hash']) or len(record['commitment_hash']) != 64:
            errors.append("commitment_hash must be 64 hex characters")
    
    if 'salt' in record:
        if not _is_valid_hex(record['salt']) or len(record['salt']) != 64:
            errors.append("salt must be 64 hex characters")
    
    # Validate version format
    if 'version' in record:
        if not re.match(r'^\d+\.\d+$', record['version']):
            errors.append("version must be in format 'X.Y'")
    
    # Validate timestamp
    if 'timestamp' in record:
        timestamp_errors = _validate_timestamp(record['timestamp'])
        errors.extend(timestamp_errors)
    
    return len(errors) == 0, errors


def create_enhanced_audit_record(
    commitment_hash: str,
    commitment_algorithm: str,
    salt_length: int,
    verification_decision: str,
    verification_confidence: float,
    samples_used: int,
    model_id: str,
    verifier_version: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a properly formatted enhanced audit record.
    
    Args:
        commitment_hash: Cryptographic commitment hash
        commitment_algorithm: Hash algorithm used
        salt_length: Length of salt in bytes
        verification_decision: PASS, FAIL, or INCONCLUSIVE
        verification_confidence: Confidence level (0-1)
        samples_used: Number of verification samples
        model_id: Model identifier
        verifier_version: Version of verification software
        metadata: Additional metadata
        
    Returns:
        Enhanced audit record dictionary
    """
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    record = {
        'commitment': {
            'hash': commitment_hash,
            'algorithm': commitment_algorithm,
            'salt_length': salt_length,
            'version': '1.0'
        },
        'timestamp': timestamp,
        'verification_result': {
            'decision': verification_decision,
            'confidence': verification_confidence,
            'samples_used': samples_used
        },
        'metadata': {
            'model_id': model_id,
            'verifier_version': verifier_version,
            **(metadata or {})
        }
    }
    
    return record


def get_schema_version() -> str:
    """Get the current audit schema version."""
    return "2.0"  # Enhanced schema with commit-reveal support


def get_supported_algorithms() -> List[str]:
    """Get list of supported hash algorithms."""
    return ["SHA256", "SHA3-256"]


def get_supported_decisions() -> List[str]:
    """Get list of supported verification decisions."""
    return ["PASS", "FAIL", "INCONCLUSIVE"]