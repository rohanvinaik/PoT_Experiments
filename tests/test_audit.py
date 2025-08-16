"""
Comprehensive unit tests for audit and commit-reveal protocol.

Tests:
- Commitment determinism
- Commitment verification with bit flips
- Audit record schema compliance
- JSON serialization/deserialization
"""

import pytest
import json
import hashlib
import hmac
import time
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pot.audit.commit_reveal import (
    serialize_for_commit,
    make_commitment,
    verify_commitment,
    write_audit_record,
    generate_session_id,
    generate_nonce
)


class TestSerialization:
    """Test deterministic serialization for commitments."""
    
    def test_serialize_deterministic(self):
        """Test that serialization is deterministic."""
        challenge_ids = ["chal_1", "chal_2", "chal_3"]
        ranges = [(0, 2), (1, 3)]
        context = {
            "session": "test_123",
            "tau": 0.05,
            "alpha": 0.01,
            "nested": {
                "key1": "value1",
                "key2": 42
            }
        }
        
        # Serialize multiple times
        data1 = serialize_for_commit(challenge_ids, ranges, context)
        data2 = serialize_for_commit(challenge_ids, ranges, context)
        data3 = serialize_for_commit(challenge_ids, ranges, context)
        
        # Should be identical
        assert data1 == data2 == data3
        
        # Should be bytes
        assert isinstance(data1, bytes)
        
        # Should contain expected content
        data_str = data1.decode('utf-8')
        assert "chal_1" in data_str
        assert "chal_2" in data_str
        assert "chal_3" in data_str
        assert "0.05" in data_str
        assert "0.01" in data_str
    
    def test_serialize_order_independent(self):
        """Test that dict key order doesn't affect serialization."""
        challenge_ids = ["id1", "id2"]
        ranges = [(0, 2)]
        
        # Same context, different key order
        context1 = {"b": 2, "a": 1, "c": 3}
        context2 = {"a": 1, "c": 3, "b": 2}
        context3 = {"c": 3, "a": 1, "b": 2}
        
        data1 = serialize_for_commit(challenge_ids, ranges, context1)
        data2 = serialize_for_commit(challenge_ids, ranges, context2)
        data3 = serialize_for_commit(challenge_ids, ranges, context3)
        
        # Should all be identical despite different dict key order
        assert data1 == data2 == data3
    
    def test_serialize_nested_structures(self):
        """Test serialization of nested data structures."""
        challenge_ids = ["test"]
        ranges = [(0, 1)]
        
        context = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": 42,
                        "list": [1, 2, 3],
                        "float": 3.14159
                    }
                }
            },
            "array": [
                {"key": "val1"},
                {"key": "val2"}
            ]
        }
        
        data = serialize_for_commit(challenge_ids, ranges, context)
        
        # Should serialize without error
        assert isinstance(data, bytes)
        assert len(data) > 0
        
        # Should be deterministic
        data2 = serialize_for_commit(challenge_ids, ranges, context)
        assert data == data2


class TestCommitment:
    """Test commitment generation and verification."""
    
    def test_commitment_deterministic(self):
        """Test that commitments are deterministic."""
        master_key = b"test_master_key_32_bytes_padded!"
        nonce = b"test_nonce_32_bytes_padded_here!"
        data = b"test data to commit"
        
        # Generate multiple commitments
        commitment1 = make_commitment(master_key, nonce, data)
        commitment2 = make_commitment(master_key, nonce, data)
        commitment3 = make_commitment(master_key, nonce, data)
        
        # Should all be identical
        assert commitment1 == commitment2 == commitment3
        
        # Should be 32 bytes (SHA256)
        assert len(commitment1) == 32
    
    def test_commitment_sensitivity(self):
        """Test that commitments are sensitive to input changes."""
        master_key = b"test_master_key_32_bytes_padded!"
        nonce = b"test_nonce_32_bytes_padded_here!"
        data = b"test data to commit"
        
        # Base commitment
        base_commitment = make_commitment(master_key, nonce, data)
        
        # Change master key
        diff_key = b"diff_master_key_32_bytes_padded!"
        key_commitment = make_commitment(diff_key, nonce, data)
        assert base_commitment != key_commitment
        
        # Change nonce
        diff_nonce = b"diff_nonce_32_bytes_padded_here!"
        nonce_commitment = make_commitment(master_key, diff_nonce, data)
        assert base_commitment != nonce_commitment
        
        # Change data (single bit flip)
        diff_data = b"test data to commiT"  # Changed last 't' to 'T'
        data_commitment = make_commitment(master_key, nonce, diff_data)
        assert base_commitment != data_commitment
    
    def test_verify_commitment_success(self):
        """Test successful commitment verification."""
        master_key = b"test_master_key_32_bytes_padded!"
        nonce = b"test_nonce_32_bytes_padded_here!"
        data = b"test data to commit"
        
        # Make commitment
        commitment = make_commitment(master_key, nonce, data)
        
        # Verify should succeed
        assert verify_commitment(master_key, nonce, data, commitment) == True
    
    def test_verify_commitment_failure(self):
        """Test commitment verification with bit flips."""
        master_key = b"test_master_key_32_bytes_padded!"
        nonce = b"test_nonce_32_bytes_padded_here!"
        data = b"test data to commit"
        
        # Make commitment
        commitment = make_commitment(master_key, nonce, data)
        
        # Flip a bit in commitment
        corrupted = bytearray(commitment)
        corrupted[0] ^= 0x01  # Flip first bit
        
        # Verification should fail
        assert verify_commitment(master_key, nonce, data, bytes(corrupted)) == False
        
        # Wrong data should also fail
        wrong_data = b"wrong data to commit"
        assert verify_commitment(master_key, nonce, wrong_data, commitment) == False
        
        # Wrong key should fail
        wrong_key = b"wrong_master_key_32_bytes_padded"
        assert verify_commitment(wrong_key, nonce, data, commitment) == False
        
        # Wrong nonce should fail
        wrong_nonce = b"wrong_nonce_32_bytes_padded_her!"
        assert verify_commitment(master_key, wrong_nonce, data, commitment) == False
    
    def test_commitment_with_serialized_data(self):
        """Test commitment with serialized challenge data."""
        master_key = os.urandom(32)
        nonce = os.urandom(32)
        
        # Create realistic data
        challenge_ids = [f"challenge_{i}" for i in range(100)]
        ranges = [(i, i+10) for i in range(0, 100, 10)]
        context = {
            "session_id": generate_session_id(),
            "tau": 0.05,
            "alpha": 0.01,
            "beta": 0.01,
            "timestamp": time.time()
        }
        
        # Serialize and commit
        data = serialize_for_commit(challenge_ids, ranges, context)
        commitment = make_commitment(master_key, nonce, data)
        
        # Verify
        assert verify_commitment(master_key, nonce, data, commitment) == True
        
        # Modify one challenge ID and verify it fails
        challenge_ids[50] = "modified_challenge"
        modified_data = serialize_for_commit(challenge_ids, ranges, context)
        assert verify_commitment(master_key, nonce, modified_data, commitment) == False


class TestAuditRecord:
    """Test audit record generation and schema compliance."""
    
    def test_audit_record_schema(self):
        """Test that audit records comply with expected schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = generate_session_id()
            
            # Write audit record
            write_audit_record(
                session_id=session_id,
                model_id="test_model_v1",
                family="vision:freq",
                alpha=0.01,
                beta=0.05,
                boundary=0.1,
                nonce=os.urandom(32),
                commitment=os.urandom(32),
                prf_info={
                    "algorithm": "HMAC-SHA256",
                    "key_length": 32
                },
                reuse_policy="u=5, rho_max=0.3",
                env={
                    "python": "3.9",
                    "platform": "linux"
                },
                artifacts={
                    "decision": "accept_id",
                    "stopping_time": 42
                },
                output_dir=tmpdir
            )
            
            # Find and load the audit file
            audit_files = list(Path(tmpdir).glob("audit_*.json"))
            assert len(audit_files) == 1
            
            with open(audit_files[0]) as f:
                audit = json.load(f)
            
            # Check required fields
            assert "session_id" in audit
            assert audit["session_id"] == session_id
            assert "timestamp" in audit
            assert isinstance(audit["timestamp"], str)  # ISO format string
            assert "model_id" in audit
            assert audit["model_id"] == "test_model_v1"
            assert "verification_params" in audit
            assert "alpha" in audit["verification_params"]
            assert audit["verification_params"]["alpha"] == 0.01
            assert "beta" in audit["verification_params"]
            assert audit["verification_params"]["beta"] == 0.05
            assert "boundary" in audit["verification_params"]
            assert audit["verification_params"]["boundary"] == 0.1
            assert "commitment" in audit
            assert "nonce" in audit
            assert "prf_info" in audit
            assert audit["prf_info"]["algorithm"] == "HMAC-SHA256"
            assert "reuse_policy" in audit
            assert "env" in audit
            assert "artifacts" in audit
            assert audit["artifacts"]["decision"] == "accept_id"
    
    def test_audit_record_json_serializable(self):
        """Test that audit records are properly JSON serializable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create audit with various data types
            write_audit_record(
                session_id=generate_session_id(),
                model_id="model_123",
                family="lm:templates",
                alpha=0.001,
                beta=0.999,
                boundary=0.5,
                nonce=os.urandom(32),
                commitment=os.urandom(32),
                prf_info={
                    "nested": {
                        "deeply": {
                            "nested": "value"
                        }
                    },
                    "list": [1, 2, 3],
                    "float": 3.14159,
                    "bool": True,
                    "null": None
                },
                reuse_policy="complex policy",
                env={"key": "value"},
                artifacts={"data": [1, 2, 3]},
                output_dir=tmpdir
            )
            
            # Should be able to load and re-serialize
            audit_files = list(Path(tmpdir).glob("audit_*.json"))
            with open(audit_files[0]) as f:
                audit = json.load(f)
            
            # Should be able to re-serialize
            json_str = json.dumps(audit, indent=2)
            assert isinstance(json_str, str)
            
            # Re-parse should match
            reparsed = json.loads(json_str)
            assert reparsed == audit
    
    def test_audit_record_multiple_sessions(self):
        """Test that multiple sessions create separate audit records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_ids = []
            
            # Create multiple audit records
            for i in range(5):
                session_id = generate_session_id()
                session_ids.append(session_id)
                
                write_audit_record(
                    session_id=session_id,
                    model_id=f"model_{i}",
                    family="vision:freq",
                    alpha=0.01 * (i + 1),
                    beta=0.01 * (i + 1),
                    boundary=0.1,
                    nonce=os.urandom(32),
                    commitment=os.urandom(32),
                    prf_info={},
                    reuse_policy="",
                    env={},
                    artifacts={},
                    output_dir=tmpdir
                )
            
            # Should have 5 separate audit files
            audit_files = list(Path(tmpdir).glob("audit_*.json"))
            assert len(audit_files) == 5
            
            # Each should have unique session ID
            found_sessions = set()
            for audit_file in audit_files:
                with open(audit_file) as f:
                    audit = json.load(f)
                    found_sessions.add(audit["session_id"])
            
            assert found_sessions == set(session_ids)


class TestSessionManagement:
    """Test session ID and nonce generation."""
    
    def test_session_id_generation(self):
        """Test that session IDs are unique and properly formatted."""
        ids = set()
        
        # Generate many IDs
        for _ in range(1000):
            sid = generate_session_id()
            
            # Should be string
            assert isinstance(sid, str)
            
            # Should have reasonable length
            assert 10 <= len(sid) <= 50
            
            # Should be unique
            assert sid not in ids
            ids.add(sid)
            
            # Should contain only valid characters
            # Assuming alphanumeric, underscore, hyphen
            assert all(c.isalnum() or c in "_-" for c in sid)
    
    def test_nonce_generation(self):
        """Test that nonces are unique and properly sized."""
        nonces = set()
        
        for _ in range(100):
            nonce = generate_nonce()
            
            # Should be bytes
            assert isinstance(nonce, bytes)
            
            # Should be 32 bytes
            assert len(nonce) == 32
            
            # Should be unique (with very high probability)
            assert nonce not in nonces
            nonces.add(nonce)
    
    def test_nonce_entropy(self):
        """Test that nonces have sufficient entropy."""
        nonces = [generate_nonce() for _ in range(100)]
        
        # Check that nonces are not predictable
        # Each byte position should have variation
        for byte_pos in range(32):
            byte_values = [nonce[byte_pos] for nonce in nonces]
            unique_values = len(set(byte_values))
            
            # Should see many different values at each position
            # With 100 samples, expect at least 20 unique values
            assert unique_values >= 20, f"Low entropy at byte {byte_pos}"


class TestIntegration:
    """Integration tests for complete commit-reveal flow."""
    
    def test_complete_commit_reveal_flow(self):
        """Test a complete commit-reveal workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            master_key = os.urandom(32)
            nonce = generate_nonce()
            session_id = generate_session_id()
            
            # Phase 1: Prepare data and commit
            challenge_ids = [f"challenge_{i:04d}" for i in range(50)]
            ranges = [(0, 10), (10, 30), (30, 50)]
            context = {
                "session_id": session_id,
                "model": "test_model",
                "tau": 0.05,
                "alpha": 0.01,
                "beta": 0.01,
                "timestamp": time.time()
            }
            
            # Serialize and commit
            data_to_commit = serialize_for_commit(challenge_ids, ranges, context)
            commitment = make_commitment(master_key, nonce, data_to_commit)
            
            # Save commitment (public)
            commitment_public = {
                "session_id": session_id,
                "commitment": commitment.hex(),
                "timestamp": time.time()
            }
            
            with open(Path(tmpdir) / "commitment.json", "w") as f:
                json.dump(commitment_public, f)
            
            # Phase 2: Verification (would happen after model evaluation)
            # ...
            
            # Phase 3: Reveal
            reveal_data = {
                "session_id": session_id,
                "challenge_ids": challenge_ids,
                "ranges": ranges,
                "context": context,
                "nonce": nonce.hex()
            }
            
            with open(Path(tmpdir) / "reveal.json", "w") as f:
                json.dump(reveal_data, f)
            
            # Phase 4: Verify commitment
            # Reconstruct data from reveal
            reconstructed_data = serialize_for_commit(
                reveal_data["challenge_ids"],
                reveal_data["ranges"],
                reveal_data["context"]
            )
            
            # Verify commitment matches
            assert verify_commitment(
                master_key,
                bytes.fromhex(reveal_data["nonce"]),
                reconstructed_data,
                bytes.fromhex(commitment_public["commitment"])
            )
            
            # Phase 5: Generate audit record
            write_audit_record(
                session_id=session_id,
                model_id=context["model"],
                family="test",
                alpha=context["alpha"],
                beta=context["beta"],
                boundary=context["tau"],
                nonce=nonce,
                commitment=commitment,
                prf_info={"test": "data"},
                reuse_policy="test_policy",
                env={"test": "env"},
                artifacts={"reveal": reveal_data},
                output_dir=tmpdir
            )
            
            # Verify all artifacts exist
            assert (Path(tmpdir) / "commitment.json").exists()
            assert (Path(tmpdir) / "reveal.json").exists()
            assert len(list(Path(tmpdir).glob("audit_*.json"))) == 1


def test_hmac_properties():
    """Test HMAC properties used in commitment."""
    key = b"test_key"
    message = b"test_message"
    
    # HMAC should be deterministic
    mac1 = hmac.new(key, message, hashlib.sha256).digest()
    mac2 = hmac.new(key, message, hashlib.sha256).digest()
    assert mac1 == mac2
    
    # HMAC should be 32 bytes for SHA256
    assert len(mac1) == 32
    
    # Different keys should produce different MACs
    mac_diff_key = hmac.new(b"different_key", message, hashlib.sha256).digest()
    assert mac1 != mac_diff_key
    
    # Different messages should produce different MACs
    mac_diff_msg = hmac.new(key, b"different_message", hashlib.sha256).digest()
    assert mac1 != mac_diff_msg


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])