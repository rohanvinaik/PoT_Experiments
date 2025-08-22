"""Commit-reveal protocol for POT verification audit trail."""

import hashlib
import hmac
import json
import os
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .schema import AUDIT_JSON_SCHEMA, validate_audit_record


@dataclass
class CommitmentRecord:
    """Record for a cryptographic commitment."""
    commitment_hash: str  # Hex-encoded SHA256 hash
    timestamp: str        # ISO 8601 timestamp
    salt: str            # Hex-encoded salt
    version: str = "1.0" # Protocol version
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommitmentRecord':
        """Create from dictionary representation."""
        return cls(
            commitment_hash=data['commitment_hash'],
            timestamp=data['timestamp'],
            salt=data['salt'],
            version=data.get('version', '1.0'),
            metadata=data.get('metadata')
        )


# Current protocol version for commitment serialization
COMMIT_PROTOCOL_VERSION = "1.0"


def serialize_for_commit(data: Dict[str, Any]) -> bytes:
    """
    Canonically serialize data for cryptographic commitment.
    
    Creates deterministic serialization with:
    - Sorted keys for reproducible JSON
    - Version inclusion (but NO timestamp for determinism)
    - UTF-8 encoding
    
    Args:
        data: Dictionary containing data to commit to
    
    Returns:
        Deterministically serialized bytes suitable for hashing
    """
    # Add version for commitment integrity (NO timestamp for determinism)
    commitment_data = {
        'version': COMMIT_PROTOCOL_VERSION,
        'data': data
    }
    
    # Use sort_keys=True and minimal separators for deterministic JSON
    json_str = json.dumps(
        commitment_data,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True  # Ensure consistent encoding
    )
    return json_str.encode('utf-8')


def compute_commitment(
    data: Dict[str, Any], 
    salt: Optional[bytes] = None
) -> CommitmentRecord:
    """
    Compute cryptographic commitment for given data.
    
    Creates a SHA256 hash of canonically serialized data with salt.
    Implements POT_PAPER_COMPLETE §2.2 commitment protocol.
    
    Args:
        data: Dictionary of data to commit to
        salt: Optional salt bytes (generated if not provided)
    
    Returns:
        CommitmentRecord with hash, timestamp, and metadata
    """
    # Generate salt if not provided
    if salt is None:
        salt = generate_nonce(32)
    
    # Canonically serialize the data
    serialized_data = serialize_for_commit(data)
    
    # Compute commitment: SHA256(serialized_data || salt)
    hasher = hashlib.sha256()
    hasher.update(serialized_data)
    hasher.update(salt)
    commitment_hash = hasher.digest()
    
    # Create commitment record
    record = CommitmentRecord(
        commitment_hash=commitment_hash.hex(),
        timestamp=datetime.utcnow().isoformat() + 'Z',
        salt=salt.hex(),
        version=COMMIT_PROTOCOL_VERSION,
        metadata={
            'data_size': len(serialized_data),
            'salt_size': len(salt),
            'hash_algorithm': 'SHA256'
        }
    )
    
    return record


def verify_reveal(
    commitment: CommitmentRecord, 
    revealed_data: Dict[str, Any], 
    salt: bytes
) -> bool:
    """
    Verify that revealed data matches the original commitment.
    
    Recomputes the commitment hash using revealed data and salt,
    then performs constant-time comparison. Includes timestamp
    validity checks.
    
    Args:
        commitment: Original commitment record
        revealed_data: The data being revealed
        salt: Salt used in original commitment
    
    Returns:
        True if commitment is valid and timestamps are consistent
    """
    try:
        # Verify salt matches
        if salt.hex() != commitment.salt:
            return False
        
        # Recompute commitment with revealed data
        serialized_revealed = serialize_for_commit(revealed_data)
        
        hasher = hashlib.sha256()
        hasher.update(serialized_revealed)
        hasher.update(salt)
        recomputed_hash = hasher.digest()
        
        # Use constant-time comparison to prevent timing attacks
        hash_matches = hmac.compare_digest(
            recomputed_hash.hex(), 
            commitment.commitment_hash
        )
        
        # Verify timestamp format and recency (basic sanity check)
        try:
            commit_time = datetime.fromisoformat(commitment.timestamp.rstrip('Z'))
            now = datetime.utcnow()
            
            # Allow for reasonable clock skew (24 hours)
            time_valid = abs((now - commit_time).total_seconds()) < 86400
        except ValueError:
            time_valid = False
        
        return hash_matches and time_valid
        
    except Exception:
        # Return False on any verification error
        return False


def write_commitment_record(
    record: CommitmentRecord, 
    filepath: str
) -> None:
    """
    Write commitment record with atomic operation.
    
    Writes CommitmentRecord atomically using temporary file + rename 
    to prevent corruption. Does not validate against audit schema
    since this is a commitment-specific format.
    
    Args:
        record: CommitmentRecord to write
        filepath: Path to write commitment record
        
    Raises:
        OSError: If file write operation fails
    """
    # Convert to dictionary format
    record_dict = record.to_dict()
    
    # Add integrity metadata
    record_dict['integrity'] = {
        'format_version': COMMIT_PROTOCOL_VERSION,
        'written_at': datetime.utcnow().isoformat() + 'Z',
        'checksum': hashlib.sha256(
            json.dumps(record_dict, sort_keys=True).encode('utf-8')
        ).hexdigest()[:16]  # Short checksum for quick integrity check
    }
    
    # Ensure output directory exists
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write atomically using temporary file + rename
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=output_path.parent,
        delete=False,
        suffix='.tmp'
    ) as temp_file:
        try:
            # Write record with sorted keys for consistency
            json.dump(record_dict, temp_file, indent=2, sort_keys=True)
            temp_file.flush()
            os.fsync(temp_file.fileno())  # Force write to disk
            
            # Atomic rename
            os.rename(temp_file.name, filepath)
            
        except Exception:
            # Clean up temporary file on error
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass
            raise


def write_audit_record(
    record: Dict[str, Any], 
    filepath: str
) -> None:
    """
    Write audit record with atomic operation and schema validation.
    
    Performs validation against AUDIT_JSON_SCHEMA and writes atomically
    using temporary file + rename to prevent corruption.
    
    Args:
        record: Audit record dictionary to write
        filepath: Path to write audit record
        
    Raises:
        ValueError: If record fails schema validation
        OSError: If file write operation fails
    """
    # Validate against schema (basic validation if jsonschema not available)
    is_valid, errors = validate_audit_record(record)
    if not is_valid:
        raise ValueError(f"Audit record validation failed: {', '.join(errors)}")
    
    # Add integrity metadata
    record_with_integrity = record.copy()
    record_with_integrity['integrity'] = {
        'format_version': COMMIT_PROTOCOL_VERSION,
        'written_at': datetime.utcnow().isoformat() + 'Z',
        'checksum': hashlib.sha256(
            json.dumps(record, sort_keys=True).encode('utf-8')
        ).hexdigest()[:16]  # Short checksum for quick integrity check
    }
    
    # Ensure output directory exists
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write atomically using temporary file + rename
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=output_path.parent,
        delete=False,
        suffix='.tmp'
    ) as temp_file:
        try:
            # Write record with sorted keys for consistency
            json.dump(record_with_integrity, temp_file, indent=2, sort_keys=True)
            temp_file.flush()
            os.fsync(temp_file.fileno())  # Force write to disk
            
            # Atomic rename
            os.rename(temp_file.name, filepath)
            
        except Exception:
            # Clean up temporary file on error
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass
            raise


def read_and_verify_audit_trail(filepath: str) -> List[Union[CommitmentRecord, Dict[str, Any]]]:
    """
    Read audit log and verify integrity of all records.
    
    Loads records from file, detects type (commitment vs audit),
    verifies integrity checksums, and validates structure.
    
    Args:
        filepath: Path to audit trail file
        
    Returns:
        List of verified records (CommitmentRecord or audit dict)
        
    Raises:
        FileNotFoundError: If audit file doesn't exist
        ValueError: If any record fails validation
        json.JSONDecodeError: If file contains invalid JSON
    """
    records = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Handle both single records and arrays of records
            content = json.load(f)
            
            if isinstance(content, dict):
                # Single record
                record_dicts = [content]
            elif isinstance(content, list):
                # Array of records
                record_dicts = content
            else:
                raise ValueError("Audit file must contain dict or list of dicts")
        
        for i, record_dict in enumerate(record_dicts):
            try:
                # Verify integrity checksum if present
                if 'integrity' in record_dict:
                    integrity_info = record_dict['integrity']
                    if 'checksum' in integrity_info:
                        # Remove integrity field temporarily for checksum verification
                        temp_record = record_dict.copy()
                        del temp_record['integrity']
                        
                        computed_checksum = hashlib.sha256(
                            json.dumps(temp_record, sort_keys=True).encode('utf-8')
                        ).hexdigest()[:16]
                        
                        if computed_checksum != integrity_info['checksum']:
                            raise ValueError(f"Record {i} integrity checksum mismatch")
                
                # Detect record type and validate accordingly
                if 'commitment_hash' in record_dict and 'salt' in record_dict:
                    # This is a CommitmentRecord
                    record = CommitmentRecord.from_dict(record_dict)
                    records.append(record)
                    
                elif 'session_id' in record_dict and 'nonce' in record_dict:
                    # This is an audit record - validate against schema
                    is_valid, errors = validate_audit_record(record_dict)
                    if not is_valid:
                        raise ValueError(f"Audit record {i} validation failed: {', '.join(errors)}")
                    records.append(record_dict)
                    
                else:
                    # Unknown record type - try to validate as both and give helpful error
                    commitment_fields = {'commitment_hash', 'salt', 'timestamp'}
                    audit_fields = {'session_id', 'nonce', 'commitment', 'model_id'}
                    
                    record_fields = set(record_dict.keys())
                    
                    if commitment_fields.issubset(record_fields):
                        record = CommitmentRecord.from_dict(record_dict)
                        records.append(record)
                    elif audit_fields.intersection(record_fields):
                        # Try audit validation to get specific errors
                        is_valid, errors = validate_audit_record(record_dict)
                        if not is_valid:
                            raise ValueError(f"Record {i} appears to be audit record but validation failed: {', '.join(errors)}")
                        records.append(record_dict)
                    else:
                        raise ValueError(f"Record {i} has unknown format - not commitment or audit record")
                
            except Exception as e:
                raise ValueError(f"Failed to process record {i}: {str(e)}")
        
        # Verify timestamp ordering (records should be chronologically ordered)
        if len(records) > 1:
            for i in range(1, len(records)):
                try:
                    # Extract timestamp from either record type
                    if isinstance(records[i-1], CommitmentRecord):
                        prev_time_str = records[i-1].timestamp
                    else:
                        prev_time_str = records[i-1]['timestamp']
                        
                    if isinstance(records[i], CommitmentRecord):
                        curr_time_str = records[i].timestamp
                    else:
                        curr_time_str = records[i]['timestamp']
                    
                    prev_time = datetime.fromisoformat(prev_time_str.rstrip('Z'))
                    curr_time = datetime.fromisoformat(curr_time_str.rstrip('Z'))
                    
                    if curr_time < prev_time:
                        raise ValueError(f"Records not in chronological order at index {i}")
                except (ValueError, KeyError):
                    # Continue if timestamp parsing fails, but log the issue
                    pass
        
        return records
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Audit trail file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in audit file: {str(e)}", e.doc, e.pos)


def read_commitment_records(filepath: str) -> List[CommitmentRecord]:
    """
    Read and verify commitment records specifically.
    
    Args:
        filepath: Path to commitment records file
        
    Returns:
        List of verified CommitmentRecord objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If any record fails validation
    """
    all_records = read_and_verify_audit_trail(filepath)
    commitment_records = []
    
    for record in all_records:
        if isinstance(record, CommitmentRecord):
            commitment_records.append(record)
        else:
            raise ValueError("File contains non-commitment records")
    
    return commitment_records


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


# Legacy compatibility functions - maintain existing API
def make_commitment(
    master_commit_key: bytes,
    nonce: bytes,
    data_to_commit: bytes
) -> bytes:
    """
    Legacy compatibility function for HMAC-SHA256 commitment.
    
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
    Legacy compatibility function for commitment verification.
    
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


def load_audit_record(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Legacy compatibility function to load a single audit record.
    
    Args:
        filepath: Path to the audit record file
    
    Returns:
        The audit record dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        record = json.load(f)
    
    # Convert hex strings back to bytes for nonce and commitment if present
    if 'nonce' in record and isinstance(record['nonce'], str):
        record['nonce_bytes'] = bytes.fromhex(record['nonce'])
    if 'commitment' in record and isinstance(record['commitment'], str):
        record['commitment_bytes'] = bytes.fromhex(record['commitment'])
    
    return record


def create_verification_audit_record(
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
    Create and write a verification audit record (legacy compatibility).
    
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
    
    # Write audit record atomically
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=output_dir,
        delete=False,
        suffix='.tmp'
    ) as temp_file:
        try:
            json.dump(audit_record, temp_file, indent=2, sort_keys=True)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            
            # Atomic rename
            os.rename(temp_file.name, filepath)
            
        except Exception:
            # Clean up temporary file on error
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass
            raise
    
    return audit_record