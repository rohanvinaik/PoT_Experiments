"""Audit and commit-reveal infrastructure for POT verification protocol."""

from .commit_reveal import (
    serialize_for_commit,
    make_commitment,
    verify_commitment,
    write_audit_record,
    load_audit_record,
    generate_session_id,
    generate_nonce
)

from .schema import (
    AUDIT_JSON_SCHEMA,
    validate_audit_record
)

__all__ = [
    # Commit-reveal functions
    'serialize_for_commit',
    'make_commitment',
    'verify_commitment',
    'write_audit_record',
    'load_audit_record',
    'generate_session_id',
    'generate_nonce',
    # Schema definitions
    'AUDIT_JSON_SCHEMA',
    'validate_audit_record'
]