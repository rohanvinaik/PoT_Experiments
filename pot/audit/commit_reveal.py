"""Commit-reveal protocol for POT verification audit trail."""

import hashlib
import hmac
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def serialize_for_commit(
    challenges: List[Any],
    ranges: List[Tuple[int, int]],
    context: Dict[str, Any]
) -> bytes:
    """
    Serialize challenges, ranges, and context deterministically for commitment.
    
    Args:
        challenges: List of challenge data (can be various types)
        ranges: List of (start, end) tuples for challenge ranges
        context: Dictionary of additional context data
    
    Returns:
        Deterministically serialized bytes
    """
    data = {
        'challenges': challenges,
        'ranges': ranges,
        'context': context
    }
    
    # Use sort_keys=True for deterministic JSON serialization
    json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return json_str.encode('utf-8')


def make_commitment(
    master_commit_key: bytes,
    nonce: bytes,
    data_to_commit: bytes
) -> bytes:
    """
    Create HMAC-SHA256 commitment of data.
    
    Args:
        master_commit_key: Master key for commitment
        nonce: Random nonce for this commitment
        data_to_commit: Serialized data to commit to
    
    Returns:
        32-byte commitment hash
    """
    # Combine master key with nonce to create session key
    session_key = master_commit_key + nonce
    
    # Create HMAC-SHA256 commitment
    commitment = hmac.new(
        session_key,
        data_to_commit,
        hashlib.sha256
    ).digest()
    
    return commitment


def verify_commitment(
    master_commit_key: bytes,
    nonce: bytes,
    revealed_data: bytes,
    commitment: bytes
) -> bool:
    """
    Verify that revealed data matches a commitment.
    
    Args:
        master_commit_key: Master key used for commitment
        nonce: Nonce used for this commitment
        revealed_data: The revealed data to verify
        commitment: The commitment to verify against
    
    Returns:
        True if commitment is valid, False otherwise
    """
    # Recompute commitment with revealed data
    expected_commitment = make_commitment(
        master_commit_key,
        nonce,
        revealed_data
    )
    
    # Use constant-time comparison to prevent timing attacks
    return hmac.compare_digest(expected_commitment, commitment)


def write_audit_record(
    session_id: str,
    model_id: str,
    family: str,
    alpha: float,
    beta: float,
    boundary: float,
    nonce: bytes,
    commitment: bytes,
    prf_info: Dict[str, Any],
    reuse_policy: str,
    env: Dict[str, Any],
    artifacts: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Write timestamped JSON audit record.
    
    Args:
        session_id: Unique session identifier
        model_id: Model identifier
        family: Model family (e.g., "resnet", "vit")
        alpha: Type I error rate
        beta: Type II error rate
        boundary: Decision boundary threshold
        nonce: Random nonce used for commitment
        commitment: The commitment hash
        prf_info: PRF configuration information
        reuse_policy: Challenge reuse policy
        env: Environment information
        artifacts: Optional additional artifacts
        output_dir: Directory to write audit record (default: audit_logs/)
    
    Returns:
        The complete audit record that was written
    """
    if output_dir is None:
        output_dir = Path("audit_logs")
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create audit record with timestamp
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    audit_record = {
        "session_id": session_id,
        "model_id": model_id,
        "family": family,
        "alpha": alpha,
        "beta": beta,
        "boundary": boundary,
        "nonce": nonce.hex() if isinstance(nonce, bytes) else nonce,
        "commitment": commitment.hex() if isinstance(commitment, bytes) else commitment,
        "prf_info": prf_info,
        "reuse_policy": reuse_policy,
        "env": env,
        "artifacts": artifacts or {},
        "timestamp": timestamp
    }
    
    # Generate filename with timestamp and session_id
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"audit_{timestamp_str}_{session_id}.json"
    filepath = output_dir / filename
    
    # Write audit record
    with open(filepath, 'w') as f:
        json.dump(audit_record, f, indent=2, sort_keys=True)
    
    return audit_record


def load_audit_record(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load an audit record from a JSON file.
    
    Args:
        filepath: Path to the audit record file
    
    Returns:
        The audit record dictionary
    """
    with open(filepath, 'r') as f:
        record = json.load(f)
    
    # Convert hex strings back to bytes for nonce and commitment if present
    if 'nonce' in record and isinstance(record['nonce'], str):
        record['nonce_bytes'] = bytes.fromhex(record['nonce'])
    if 'commitment' in record and isinstance(record['commitment'], str):
        record['commitment_bytes'] = bytes.fromhex(record['commitment'])
    
    return record


def generate_session_id() -> str:
    """
    Generate a unique session ID for audit records.
    
    Returns:
        A unique session identifier
    """
    # Use timestamp and random bytes for uniqueness
    timestamp = int(time.time() * 1000000)  # Microsecond precision
    random_bytes = os.urandom(8)
    session_data = f"{timestamp}_{random_bytes.hex()}"
    
    # Create a shorter hash for the session ID
    session_hash = hashlib.sha256(session_data.encode()).hexdigest()[:16]
    return f"session_{session_hash}"


def generate_nonce(size: int = 32) -> bytes:
    """
    Generate a cryptographically secure random nonce.
    
    Args:
        size: Size of nonce in bytes (default: 32)
    
    Returns:
        Random nonce bytes
    """
    return os.urandom(size)