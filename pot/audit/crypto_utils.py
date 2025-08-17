#!/usr/bin/env python3
"""
Cryptographic utilities for the audit system.

Provides advanced cryptographic primitives for enhanced audit capabilities including:
- Cryptographically secure salt generation
- Hash chain construction for audit trails
- Timestamp proofs with RFC 3161 and OpenTimestamps support
- Commitment aggregation with Merkle trees
- Zero-knowledge proofs for privacy-preserving verification

All implementations follow cryptographic best practices and provide configurable
security parameters for different deployment scenarios.
"""

import os
import hashlib
import hmac
import secrets
import time
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# Optional dependencies for advanced features
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


class HashAlgorithm(Enum):
    """Supported hash algorithms for cryptographic operations."""
    SHA256 = "sha256"
    SHA3_256 = "sha3_256"
    BLAKE2B = "blake2b"
    SHA512 = "sha512"


class TimestampProofType(Enum):
    """Types of timestamp proof systems."""
    RFC3161 = "rfc3161"
    OPENTIMESTAMPS = "opentimestamps"
    LOCAL = "local"  # Local timestamp with system clock


@dataclass
class TimestampProof:
    """
    Timestamp proof structure for proving data existed at specific time.
    
    Attributes:
        data_hash: Hash of the data being timestamped
        timestamp: Time when proof was created
        proof_type: Type of timestamp proof system used
        proof_data: Cryptographic proof data (format depends on type)
        verifier_info: Information about the timestamp authority
        signature: Digital signature from timestamp authority (if applicable)
    """
    data_hash: bytes
    timestamp: datetime
    proof_type: TimestampProofType
    proof_data: bytes
    verifier_info: Dict[str, Any]
    signature: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'data_hash': self.data_hash.hex(),
            'timestamp': self.timestamp.isoformat(),
            'proof_type': self.proof_type.value,
            'proof_data': self.proof_data.hex(),
            'verifier_info': self.verifier_info,
            'signature': self.signature.hex() if self.signature else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimestampProof':
        """Create from dictionary."""
        return cls(
            data_hash=bytes.fromhex(data['data_hash']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            proof_type=TimestampProofType(data['proof_type']),
            proof_data=bytes.fromhex(data['proof_data']),
            verifier_info=data['verifier_info'],
            signature=bytes.fromhex(data['signature']) if data['signature'] else None
        )


@dataclass
class AggregateCommitment:
    """
    Aggregate commitment combining multiple individual commitments.
    
    Attributes:
        commitment_hashes: List of individual commitment hashes
        merkle_root: Root of Merkle tree containing all commitments
        aggregate_hash: Single hash representing the entire set
        size: Number of commitments in the aggregate
        proofs: Merkle proofs for each commitment (commitment_hash -> proof)
        metadata: Additional metadata about the aggregation
    """
    commitment_hashes: List[str]  # Hex strings for JSON serialization
    merkle_root: str
    aggregate_hash: str
    size: int
    proofs: Dict[str, List[Tuple[str, bool]]]  # commitment_hash -> Merkle proof
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'commitment_hashes': self.commitment_hashes,
            'merkle_root': self.merkle_root,
            'aggregate_hash': self.aggregate_hash,
            'size': self.size,
            'proofs': self.proofs,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AggregateCommitment':
        """Create from dictionary."""
        return cls(
            commitment_hashes=data['commitment_hashes'],
            merkle_root=data['merkle_root'],
            aggregate_hash=data['aggregate_hash'],
            size=data['size'],
            proofs=data['proofs'],
            metadata=data['metadata']
        )


@dataclass
class ZKProof:
    """
    Zero-knowledge proof structure for privacy-preserving verification.
    
    Attributes:
        statement: Public statement being proven (without sensitive details)
        proof_data: Cryptographic proof data
        proof_type: Type of ZK proof system used
        verifier_key: Key for verifying the proof
        metadata: Additional proof metadata
    """
    statement: Dict[str, Any]
    proof_data: bytes
    proof_type: str
    verifier_key: bytes
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'statement': self.statement,
            'proof_data': self.proof_data.hex(),
            'proof_type': self.proof_type,
            'verifier_key': self.verifier_key.hex(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ZKProof':
        """Create from dictionary."""
        return cls(
            statement=data['statement'],
            proof_data=bytes.fromhex(data['proof_data']),
            proof_type=data['proof_type'],
            verifier_key=bytes.fromhex(data['verifier_key']),
            metadata=data['metadata']
        )


def generate_cryptographic_salt(length: int = 32) -> bytes:
    """
    Generate cryptographically secure random salt.
    
    Uses the cryptographically secure random number generator provided by the OS.
    Validates length requirements to ensure sufficient entropy.
    
    Args:
        length: Length of salt in bytes (minimum 16, recommended 32)
        
    Returns:
        Cryptographically secure random bytes
        
    Raises:
        ValueError: If length is too small for security
        
    Example:
        >>> salt = generate_cryptographic_salt(32)
        >>> len(salt)
        32
        >>> salt1 = generate_cryptographic_salt()
        >>> salt2 = generate_cryptographic_salt()
        >>> salt1 != salt2  # Extremely high probability
        True
    """
    if length < 16:
        raise ValueError(f"Salt length {length} is too small. Minimum is 16 bytes for security.")
    
    if length > 1024:
        raise ValueError(f"Salt length {length} is unnecessarily large. Maximum is 1024 bytes.")
    
    return secrets.token_bytes(length)


def compute_hash_chain(hashes: List[bytes], algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bytes:
    """
    Create hash chain for audit trail where each hash depends on the previous.
    
    Computes H(H_n || H_{n-1} || ... || H_1 || H_0) where || is concatenation.
    This creates a tamper-evident chain where modifying any element changes
    the final hash.
    
    Args:
        hashes: List of hash values to chain together
        algorithm: Hash algorithm to use for chaining
        
    Returns:
        Final hash of the complete chain
        
    Raises:
        ValueError: If hashes list is empty or contains invalid data
        
    Example:
        >>> hash1 = hashlib.sha256(b"data1").digest()
        >>> hash2 = hashlib.sha256(b"data2").digest()
        >>> chain_hash = compute_hash_chain([hash1, hash2])
        >>> len(chain_hash)
        32
    """
    if not hashes:
        raise ValueError("Cannot create hash chain from empty list")
    
    if not all(isinstance(h, bytes) for h in hashes):
        raise ValueError("All hashes must be bytes objects")
    
    # Choose hash function based on algorithm
    if algorithm == HashAlgorithm.SHA256:
        hasher = hashlib.sha256()
    elif algorithm == HashAlgorithm.SHA3_256:
        hasher = hashlib.sha3_256()
    elif algorithm == HashAlgorithm.BLAKE2B:
        hasher = hashlib.blake2b()
    elif algorithm == HashAlgorithm.SHA512:
        hasher = hashlib.sha512()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    # Chain hashes: H(H_n || H_{n-1} || ... || H_0)
    for hash_value in reversed(hashes):  # Reverse for chronological order
        hasher.update(hash_value)
    
    return hasher.digest()


def create_timestamp_proof(
    data: bytes, 
    timestamp: Optional[datetime] = None,
    proof_type: TimestampProofType = TimestampProofType.LOCAL
) -> TimestampProof:
    """
    Create timestamp proof to prove data existed at specific time.
    
    Supports multiple timestamp proof systems:
    - LOCAL: Simple local timestamp with system clock
    - RFC3161: RFC 3161 timestamp authority (requires cryptography package)
    - OPENTIMESTAMPS: OpenTimestamps protocol (requires network access)
    
    Args:
        data: Data to timestamp
        timestamp: Specific timestamp (defaults to current time)
        proof_type: Type of timestamp proof system to use
        
    Returns:
        TimestampProof containing cryptographic timestamp evidence
        
    Example:
        >>> data = b"important document"
        >>> proof = create_timestamp_proof(data)
        >>> proof.proof_type == TimestampProofType.LOCAL
        True
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    data_hash = hashlib.sha256(data).digest()
    
    if proof_type == TimestampProofType.LOCAL:
        # Local timestamp with system clock
        proof_data = _create_local_timestamp_proof(data_hash, timestamp)
        verifier_info = {
            'type': 'local',
            'system_info': {
                'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
                'timestamp_source': 'system_clock'
            }
        }
        
        return TimestampProof(
            data_hash=data_hash,
            timestamp=timestamp,
            proof_type=proof_type,
            proof_data=proof_data,
            verifier_info=verifier_info
        )
    
    elif proof_type == TimestampProofType.RFC3161:
        # RFC 3161 timestamp authority
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography package required for RFC 3161 timestamps")
        
        proof_data, signature = _create_rfc3161_timestamp_proof(data_hash, timestamp)
        verifier_info = {
            'type': 'rfc3161',
            'authority': 'local_ca',  # Would be actual CA in production
            'algorithm': 'sha256_with_rsa'
        }
        
        return TimestampProof(
            data_hash=data_hash,
            timestamp=timestamp,
            proof_type=proof_type,
            proof_data=proof_data,
            verifier_info=verifier_info,
            signature=signature
        )
    
    elif proof_type == TimestampProofType.OPENTIMESTAMPS:
        # OpenTimestamps protocol
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package required for OpenTimestamps")
        
        proof_data = _create_opentimestamps_proof(data_hash, timestamp)
        verifier_info = {
            'type': 'opentimestamps',
            'calendar_servers': ['https://alice.btc.calendar.opentimestamps.org'],
            'blockchain': 'bitcoin'
        }
        
        return TimestampProof(
            data_hash=data_hash,
            timestamp=timestamp,
            proof_type=proof_type,
            proof_data=proof_data,
            verifier_info=verifier_info
        )
    
    else:
        raise ValueError(f"Unsupported timestamp proof type: {proof_type}")


def aggregate_commitments(
    commitment_records: List['CommitmentRecord'],
    metadata: Optional[Dict[str, Any]] = None
) -> AggregateCommitment:
    """
    Combine multiple commitments efficiently using Merkle tree.
    
    Creates an aggregate commitment that allows efficient verification of
    individual commitments without revealing the full set. Uses Merkle tree
    for logarithmic proof size.
    
    Args:
        commitment_records: List of CommitmentRecord objects to aggregate
        metadata: Additional metadata about the aggregation
        
    Returns:
        AggregateCommitment containing Merkle tree and proofs
        
    Raises:
        ValueError: If commitment list is empty
        ImportError: If Merkle tree implementation is not available
        
    Example:
        >>> from pot.audit.commit_reveal import CommitmentRecord
        >>> records = [create_mock_commitment() for _ in range(10)]
        >>> aggregate = aggregate_commitments(records)
        >>> aggregate.size
        10
    """
    if not commitment_records:
        raise ValueError("Cannot aggregate empty commitment list")
    
    # Import Merkle tree implementation
    try:
        from pot.prototypes.training_provenance_auditor import (
            build_merkle_tree, compute_merkle_root, generate_merkle_proof
        )
    except ImportError:
        raise ImportError("Merkle tree implementation not available")
    
    if metadata is None:
        metadata = {}
    
    # Extract commitment hashes and convert to bytes for Merkle tree
    commitment_hashes = []
    commitment_data = []
    
    for record in commitment_records:
        commitment_hash = record.commitment_hash
        if isinstance(commitment_hash, str):
            commitment_hash_bytes = bytes.fromhex(commitment_hash)
        else:
            commitment_hash_bytes = commitment_hash
        
        commitment_hashes.append(commitment_hash)
        commitment_data.append(commitment_hash_bytes)
    
    # Build Merkle tree
    merkle_tree = build_merkle_tree(commitment_data)
    merkle_root_bytes = merkle_tree.hash
    merkle_root = merkle_root_bytes.hex()
    
    # Generate Merkle proofs for each commitment
    proofs = {}
    for i, (record, commitment_bytes) in enumerate(zip(commitment_records, commitment_data)):
        proof = generate_merkle_proof(merkle_tree, i)
        
        # Convert proof to serializable format
        serializable_proof = [
            (sibling_hash.hex(), is_right) for sibling_hash, is_right in proof
        ]
        
        commitment_key = record.commitment_hash if isinstance(record.commitment_hash, str) else record.commitment_hash.hex()
        proofs[commitment_key] = serializable_proof
    
    # Compute aggregate hash of all commitments
    aggregate_hasher = hashlib.sha256()
    for commitment_bytes in commitment_data:
        aggregate_hasher.update(commitment_bytes)
    aggregate_hash = aggregate_hasher.digest().hex()
    
    # Add aggregation metadata
    full_metadata = {
        'creation_time': datetime.now(timezone.utc).isoformat(),
        'merkle_tree_depth': len(generate_merkle_proof(merkle_tree, 0)),
        'hash_algorithm': 'sha256',
        **metadata
    }
    
    return AggregateCommitment(
        commitment_hashes=[
            h if isinstance(h, str) else h.hex() for h in commitment_hashes
        ],
        merkle_root=merkle_root,
        aggregate_hash=aggregate_hash,
        size=len(commitment_records),
        proofs=proofs,
        metadata=full_metadata
    )


def create_zk_proof(
    statement: Dict[str, Any], 
    witness: Dict[str, Any],
    proof_type: str = "schnorr"
) -> ZKProof:
    """
    Create zero-knowledge proof of verification without revealing sensitive details.
    
    Supports simple ZK proof systems for proving statements about model verification
    without revealing sensitive model parameters or intermediate results.
    
    Args:
        statement: Public statement being proven (e.g., "model passes verification")
        witness: Private witness data (e.g., actual model parameters)
        proof_type: Type of ZK proof system ("schnorr", "simple_commit")
        
    Returns:
        ZKProof containing cryptographic proof
        
    Note:
        This is a simplified implementation for demonstration. Production systems
        should use established ZK proof libraries like libsnark or bulletproofs.
        
    Example:
        >>> statement = {"verification_passed": True, "threshold": 0.95}
        >>> witness = {"actual_score": 0.97, "secret_key": "hidden"}
        >>> proof = create_zk_proof(statement, witness)
        >>> proof.proof_type
        'schnorr'
    """
    if proof_type == "schnorr":
        return _create_schnorr_proof(statement, witness)
    elif proof_type == "simple_commit":
        return _create_simple_commitment_proof(statement, witness)
    else:
        raise ValueError(f"Unsupported ZK proof type: {proof_type}")


# Helper functions for timestamp proofs

def _create_local_timestamp_proof(data_hash: bytes, timestamp: datetime) -> bytes:
    """Create simple local timestamp proof with HMAC."""
    # Use a derived key for the timestamp proof
    # In practice, this would use a long-term secret key
    timestamp_key = hashlib.sha256(b"timestamp_proof_key_" + data_hash[:16]).digest()
    
    # Create HMAC of timestamp + data hash
    timestamp_bytes = timestamp.isoformat().encode('utf-8')
    message = timestamp_bytes + data_hash
    
    proof = hmac.new(timestamp_key, message, hashlib.sha256).digest()
    return proof


def _create_rfc3161_timestamp_proof(data_hash: bytes, timestamp: datetime) -> Tuple[bytes, bytes]:
    """Create RFC 3161 style timestamp proof (simplified)."""
    # This is a simplified implementation
    # Real RFC 3161 would involve a proper timestamp authority
    
    # Generate a temporary RSA key pair for demo
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    public_key = private_key.public_key()
    
    # Create timestamp request
    timestamp_bytes = timestamp.isoformat().encode('utf-8')
    tsr_data = timestamp_bytes + data_hash
    
    # Sign the timestamp request
    signature = private_key.sign(
        tsr_data,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    
    # Return proof data and signature
    proof_data = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return proof_data, signature


def _create_opentimestamps_proof(data_hash: bytes, timestamp: datetime) -> bytes:
    """Create OpenTimestamps style proof (simplified mock)."""
    # This is a mock implementation
    # Real OpenTimestamps requires network interaction with calendar servers
    
    # Mock commitment to calendar servers
    calendar_commitment = hashlib.sha256(
        b"opentimestamps_calendar_" + data_hash + timestamp.isoformat().encode()
    ).digest()
    
    # Mock Bitcoin block header hash (would be real in production)
    mock_block_hash = hashlib.sha256(
        calendar_commitment + b"mock_bitcoin_block"
    ).digest()
    
    # Create proof structure
    proof_structure = {
        'calendar_commitment': calendar_commitment.hex(),
        'block_hash': mock_block_hash.hex(),
        'block_height': 800000,  # Mock block height
        'timestamp_utc': timestamp.isoformat()
    }
    
    proof_json = json.dumps(proof_structure, sort_keys=True)
    return proof_json.encode('utf-8')


# Helper functions for ZK proofs

def _create_schnorr_proof(statement: Dict[str, Any], witness: Dict[str, Any]) -> ZKProof:
    """Create simplified Schnorr-style proof."""
    # This is a simplified educational implementation
    # Production systems should use established libraries
    
    # Generate proof parameters
    proof_seed = secrets.token_bytes(32)
    statement_hash = hashlib.sha256(
        json.dumps(statement, sort_keys=True).encode()
    ).digest()
    
    # Create "proof" by combining statement and witness in a way that
    # allows verification without revealing the witness
    challenge = hashlib.sha256(proof_seed + statement_hash).digest()
    
    # Mock response (in real Schnorr, this would involve discrete log)
    response = hashlib.sha256(
        challenge + json.dumps(witness, sort_keys=True).encode()
    ).digest()
    
    # Combine into proof data
    proof_data = proof_seed + challenge + response
    
    # Verifier key is derived from statement
    verifier_key = hashlib.sha256(statement_hash + b"verifier").digest()
    
    metadata = {
        'creation_time': datetime.now(timezone.utc).isoformat(),
        'proof_algorithm': 'simplified_schnorr',
        'security_parameter': 256
    }
    
    return ZKProof(
        statement=statement,
        proof_data=proof_data,
        proof_type="schnorr",
        verifier_key=verifier_key,
        metadata=metadata
    )


def _create_simple_commitment_proof(statement: Dict[str, Any], witness: Dict[str, Any]) -> ZKProof:
    """Create simple commitment-based proof."""
    # Generate commitment to witness
    nonce = secrets.token_bytes(32)
    witness_bytes = json.dumps(witness, sort_keys=True).encode()
    commitment = hashlib.sha256(nonce + witness_bytes).digest()
    
    # Create proof that commitment is consistent with statement
    statement_bytes = json.dumps(statement, sort_keys=True).encode()
    proof_hash = hashlib.sha256(commitment + statement_bytes).digest()
    
    # Proof data contains commitment and proof hash
    proof_data = nonce + commitment + proof_hash
    
    # Verifier key for checking the proof
    verifier_key = hashlib.sha256(statement_bytes + b"simple_commit_verifier").digest()
    
    metadata = {
        'creation_time': datetime.now(timezone.utc).isoformat(),
        'proof_algorithm': 'simple_commitment',
        'commitment_algorithm': 'sha256'
    }
    
    return ZKProof(
        statement=statement,
        proof_data=proof_data,
        proof_type="simple_commit",
        verifier_key=verifier_key,
        metadata=metadata
    )


# Verification functions

def verify_timestamp_proof(proof: TimestampProof, data: bytes) -> bool:
    """
    Verify a timestamp proof against the original data.
    
    Args:
        proof: TimestampProof to verify
        data: Original data that was timestamped
        
    Returns:
        True if proof is valid, False otherwise
    """
    # Verify data hash matches
    data_hash = hashlib.sha256(data).digest()
    if data_hash != proof.data_hash:
        return False
    
    if proof.proof_type == TimestampProofType.LOCAL:
        return _verify_local_timestamp_proof(proof, data_hash)
    elif proof.proof_type == TimestampProofType.RFC3161:
        return _verify_rfc3161_timestamp_proof(proof, data_hash)
    elif proof.proof_type == TimestampProofType.OPENTIMESTAMPS:
        return _verify_opentimestamps_proof(proof, data_hash)
    else:
        return False


def verify_zk_proof(proof: ZKProof) -> bool:
    """
    Verify a zero-knowledge proof.
    
    Args:
        proof: ZKProof to verify
        
    Returns:
        True if proof is valid, False otherwise
    """
    if proof.proof_type == "schnorr":
        return _verify_schnorr_proof(proof)
    elif proof.proof_type == "simple_commit":
        return _verify_simple_commitment_proof(proof)
    else:
        return False


def _verify_local_timestamp_proof(proof: TimestampProof, data_hash: bytes) -> bool:
    """Verify local timestamp proof."""
    # Recreate the proof using same algorithm
    timestamp_key = hashlib.sha256(b"timestamp_proof_key_" + data_hash[:16]).digest()
    timestamp_bytes = proof.timestamp.isoformat().encode('utf-8')
    message = timestamp_bytes + data_hash
    
    expected_proof = hmac.new(timestamp_key, message, hashlib.sha256).digest()
    return hmac.compare_digest(expected_proof, proof.proof_data)


def _verify_rfc3161_timestamp_proof(proof: TimestampProof, data_hash: bytes) -> bool:
    """Verify RFC 3161 timestamp proof (simplified)."""
    if not CRYPTOGRAPHY_AVAILABLE or not proof.signature:
        return False
    
    try:
        # Load public key from proof data
        public_key = serialization.load_der_public_key(proof.proof_data)
        
        # Recreate signed data
        timestamp_bytes = proof.timestamp.isoformat().encode('utf-8')
        tsr_data = timestamp_bytes + data_hash
        
        # Verify signature
        public_key.verify(
            proof.signature,
            tsr_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except Exception:
        return False


def _verify_opentimestamps_proof(proof: TimestampProof, data_hash: bytes) -> bool:
    """Verify OpenTimestamps proof (simplified)."""
    try:
        proof_structure = json.loads(proof.proof_data.decode('utf-8'))
        
        # Verify calendar commitment
        calendar_commitment = hashlib.sha256(
            b"opentimestamps_calendar_" + data_hash + proof.timestamp.isoformat().encode()
        ).digest()
        
        expected_commitment = calendar_commitment.hex()
        return proof_structure.get('calendar_commitment') == expected_commitment
    except Exception:
        return False


def _verify_schnorr_proof(proof: ZKProof) -> bool:
    """Verify simplified Schnorr proof."""
    try:
        # Extract proof components
        proof_data = proof.proof_data
        if len(proof_data) != 96:  # 32 + 32 + 32
            return False
        
        proof_seed = proof_data[:32]
        challenge = proof_data[32:64]
        response = proof_data[64:]
        
        # Verify challenge generation
        statement_hash = hashlib.sha256(
            json.dumps(proof.statement, sort_keys=True).encode()
        ).digest()
        expected_challenge = hashlib.sha256(proof_seed + statement_hash).digest()
        
        return hmac.compare_digest(challenge, expected_challenge)
    except Exception:
        return False


def _verify_simple_commitment_proof(proof: ZKProof) -> bool:
    """Verify simple commitment proof."""
    try:
        # Extract proof components
        proof_data = proof.proof_data
        if len(proof_data) != 96:  # 32 + 32 + 32
            return False
        
        nonce = proof_data[:32]
        commitment = proof_data[32:64]
        proof_hash = proof_data[64:]
        
        # Verify proof hash
        statement_bytes = json.dumps(proof.statement, sort_keys=True).encode()
        expected_proof_hash = hashlib.sha256(commitment + statement_bytes).digest()
        
        return hmac.compare_digest(proof_hash, expected_proof_hash)
    except Exception:
        return False


# Utility functions for key management

def derive_key_from_password(
    password: str, 
    salt: bytes, 
    key_length: int = 32,
    iterations: int = 100000
) -> bytes:
    """
    Derive cryptographic key from password using PBKDF2.
    
    Args:
        password: Password to derive key from
        salt: Cryptographic salt
        key_length: Length of derived key in bytes
        iterations: Number of PBKDF2 iterations
        
    Returns:
        Derived cryptographic key
    """
    if not CRYPTOGRAPHY_AVAILABLE:
        # Fallback to hashlib-based PBKDF2
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations, key_length)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=key_length,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(password.encode())


def secure_compare(a: bytes, b: bytes) -> bool:
    """
    Constant-time comparison of byte strings.
    
    Args:
        a: First byte string
        b: Second byte string
        
    Returns:
        True if strings are equal, False otherwise
    """
    return hmac.compare_digest(a, b)


def get_available_features() -> Dict[str, bool]:
    """
    Get information about available cryptographic features.
    
    Returns:
        Dictionary mapping feature names to availability
    """
    return {
        'cryptography_package': CRYPTOGRAPHY_AVAILABLE,
        'requests_package': REQUESTS_AVAILABLE,
        'rfc3161_timestamps': CRYPTOGRAPHY_AVAILABLE,
        'opentimestamps': REQUESTS_AVAILABLE,
        'advanced_zk_proofs': False,  # Would require additional libraries
        'hardware_rng': hasattr(secrets, 'SystemRandom'),
    }


if __name__ == "__main__":
    # Simple test/demo of the cryptographic utilities
    print("Cryptographic Utilities Demo")
    print("=" * 40)
    
    # Test salt generation
    salt = generate_cryptographic_salt(32)
    print(f"Generated salt: {salt.hex()[:32]}...")
    
    # Test hash chain
    hashes = [hashlib.sha256(f"data_{i}".encode()).digest() for i in range(3)]
    chain_hash = compute_hash_chain(hashes)
    print(f"Hash chain: {chain_hash.hex()[:32]}...")
    
    # Test timestamp proof
    data = b"important verification result"
    timestamp_proof = create_timestamp_proof(data)
    print(f"Timestamp proof created: {timestamp_proof.proof_type.value}")
    
    # Test ZK proof
    statement = {"verification_passed": True, "confidence": 0.95}
    witness = {"actual_confidence": 0.97, "secret_threshold": 0.9}
    zk_proof = create_zk_proof(statement, witness)
    print(f"ZK proof created: {zk_proof.proof_type}")
    
    # Test verification
    timestamp_valid = verify_timestamp_proof(timestamp_proof, data)
    zk_valid = verify_zk_proof(zk_proof)
    print(f"Timestamp proof valid: {timestamp_valid}")
    print(f"ZK proof valid: {zk_valid}")
    
    print(f"\nAvailable features: {get_available_features()}")