"""JSON schema definitions for POT audit records."""

AUDIT_JSON_SCHEMA = {
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


def validate_audit_record(record: dict) -> tuple[bool, list[str]]:
    """
    Validate an audit record against the schema.
    
    Args:
        record: Audit record dictionary to validate
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    try:
        import jsonschema
        
        # Validate against schema
        jsonschema.validate(instance=record, schema=AUDIT_JSON_SCHEMA)
        return True, []
    
    except ImportError:
        # If jsonschema is not installed, do basic validation
        errors = []
        required_fields = AUDIT_JSON_SCHEMA["required"]
        
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
        
        # Check hex encoding
        if "nonce" in record and not all(c in "0123456789abcdef" for c in record["nonce"]):
            errors.append("nonce must be hex encoded")
        if "commitment" in record:
            if not all(c in "0123456789abcdef" for c in record["commitment"]):
                errors.append("commitment must be hex encoded")
            if len(record["commitment"]) != 64:
                errors.append("commitment must be 64 hex characters (32 bytes)")
        
        # Check reuse policy
        valid_policies = ["never", "session", "model", "family", "always"]
        if "reuse_policy" in record and record["reuse_policy"] not in valid_policies:
            errors.append(f"reuse_policy must be one of: {valid_policies}")
        
        return len(errors) == 0, errors
    
    except jsonschema.exceptions.ValidationError as e:
        return False, [str(e)]