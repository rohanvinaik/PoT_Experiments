import hashlib
import hmac
import secrets
from typing import Optional, List
import struct

def kdf(master_key: bytes, label: str, context: bytes = b'') -> bytes:
    """
    Key Derivation Function (KDF) for cryptographic challenge derivation
    Based on HKDF-Extract-and-Expand (RFC 5869) simplified version
    
    From paper Section 6.2 Algorithm 3
    """
    # Use HMAC-SHA256 as the PRF
    info = label.encode() + context
    return hmac.new(master_key, info, hashlib.sha256).digest()

def derive_challenge_key(master_key_hex: str, epoch: int, session: str) -> bytes:
    """
    Cryptographic challenge key derivation with rotation
    Algorithm 3 from paper Section 6.2
    
    Args:
        master_key_hex: Master key in hexadecimal
        epoch: Epoch number for key rotation
        session: Session identifier
        
    Returns:
        Derived challenge key
    """
    master_key = bytes.fromhex(master_key_hex)
    
    # Step 1: k_epoch = KDF(k, "epoch" || e)
    epoch_bytes = struct.pack('>I', epoch)  # Big-endian 4-byte integer
    k_epoch = kdf(master_key, "epoch", epoch_bytes)
    
    # Step 2: k_session = KDF(k_epoch, "session" || s)
    k_session = kdf(k_epoch, "session", session.encode())
    
    # Step 3: seed = KDF(k_session, "challenge")
    seed = kdf(k_session, "challenge")
    
    return seed

def commit_reveal_protocol(challenge_id: str, salt_hex: str) -> dict:
    """
    Commit-reveal protocol for challenge governance
    
    Returns:
        dict with 'commitment' and 'reveal' phases
    """
    # Commitment phase: hash(challenge_id || salt)
    commitment = hashlib.sha256((challenge_id + salt_hex).encode()).hexdigest()
    
    # Reveal phase data
    reveal = {
        'challenge_id': challenge_id,
        'salt': salt_hex,
        'commitment': commitment
    }
    
    return {
        'commitment': commitment,
        'reveal': reveal
    }

def verify_commitment(commitment: str, challenge_id: str, salt_hex: str) -> bool:
    """
    Verify a commitment in the commit-reveal protocol
    
    Returns:
        True if the commitment is valid
    """
    expected = hashlib.sha256((challenge_id + salt_hex).encode()).hexdigest()
    return hmac.compare_digest(commitment, expected)

def rotate_epoch_key(master_key_hex: str, current_epoch: int) -> dict:
    """
    Implement epoch-based key rotation for leakage resilience
    From paper Section 6.2
    
    Returns:
        Dictionary with old and new epoch keys
    """
    master_key = bytes.fromhex(master_key_hex)
    
    old_epoch_key = kdf(master_key, "epoch", struct.pack('>I', current_epoch))
    new_epoch_key = kdf(master_key, "epoch", struct.pack('>I', current_epoch + 1))
    
    return {
        'old_epoch': current_epoch,
        'old_key': old_epoch_key.hex(),
        'new_epoch': current_epoch + 1,
        'new_key': new_epoch_key.hex()
    }

def commit_message(challenge_id: str, salt_hex: str) -> str:
    """Legacy function - use commit_reveal_protocol instead"""
    return hashlib.sha256((challenge_id + salt_hex).encode()).hexdigest()

def new_session_nonce() -> str:
    """Generate a new session nonce"""
    return secrets.token_hex(16)

class ChallengeGovernance:
    """
    Complete challenge governance system with cryptographic guarantees
    Implements leakage resilience from paper Section 6.2
    """
    
    def __init__(self, master_key_hex: str):
        self.master_key_hex = master_key_hex
        self.current_epoch = 0
        self.sessions = {}
        
    def new_epoch(self) -> int:
        """Start a new epoch for key rotation"""
        self.current_epoch += 1
        return self.current_epoch
    
    def new_session(self, session_id: Optional[str] = None) -> str:
        """Create a new verification session"""
        if session_id is None:
            session_id = secrets.token_hex(16)
        
        # Derive session key
        session_key = derive_challenge_key(
            self.master_key_hex, 
            self.current_epoch, 
            session_id
        )
        
        self.sessions[session_id] = {
            'epoch': self.current_epoch,
            'key': session_key.hex(),
            'challenges': []
        }
        
        return session_id
    
    def generate_challenge_seed(self, session_id: str) -> bytes:
        """Generate deterministic seed for challenge sampling"""
        if session_id not in self.sessions:
            raise ValueError(f"Unknown session: {session_id}")
        
        session = self.sessions[session_id]
        return bytes.fromhex(session['key'])
    
    def is_session_valid(self, session_id: str, max_age_epochs: int = 2) -> bool:
        """Check if a session is still valid (not too old)"""
        if session_id not in self.sessions:
            return False
        
        session_epoch = self.sessions[session_id]['epoch']
        return (self.current_epoch - session_epoch) <= max_age_epochs