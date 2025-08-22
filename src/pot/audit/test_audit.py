"""Tests for POT audit and commit-reveal infrastructure."""

import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pot.audit import (
    serialize_for_commit,
    make_commitment,
    verify_commitment,
    write_audit_record,
    load_audit_record,
    generate_session_id,
    generate_nonce,
    validate_audit_record,
    AUDIT_JSON_SCHEMA
)


def test_commitment_determinism():
    """Test that commitments are deterministic for same inputs."""
    print("Testing commitment determinism...")
    
    # Test data
    challenges = [
        {"type": "image", "data": [1, 2, 3]},
        {"type": "text", "data": "test challenge"}
    ]
    ranges = [(0, 10), (10, 20)]
    context = {"model": "test_model", "version": "1.0"}
    
    # Fixed key and nonce for testing
    master_key = b"test_master_key_32_bytes_long!!!"
    nonce = b"test_nonce_16_bytes!"
    
    # Serialize data
    data1 = serialize_for_commit(challenges, ranges, context)
    data2 = serialize_for_commit(challenges, ranges, context)
    
    # Should produce identical serialization
    assert data1 == data2, "Serialization is not deterministic"
    
    # Create commitments
    commitment1 = make_commitment(master_key, nonce, data1)
    commitment2 = make_commitment(master_key, nonce, data2)
    
    # Should produce identical commitments
    assert commitment1 == commitment2, "Commitments are not deterministic"
    
    print("✓ Commitments are deterministic for same inputs")
    return True


def test_commitment_verification():
    """Test commitment verification with bit flips."""
    print("Testing commitment verification...")
    
    # Test data
    challenges = [{"id": 1, "value": "test"}]
    ranges = [(0, 5)]
    context = {"test": "data"}
    
    master_key = b"secret_key_for_testing_32_bytes!"
    nonce = generate_nonce()
    
    # Create commitment
    original_data = serialize_for_commit(challenges, ranges, context)
    commitment = make_commitment(master_key, nonce, original_data)
    
    # Verify original data
    assert verify_commitment(master_key, nonce, original_data, commitment), \
        "Failed to verify valid commitment"
    
    print("✓ Valid commitment verified successfully")
    
    # Test with bit flips in data
    print("Testing bit flips in revealed data...")
    flipped_data = bytearray(original_data)
    
    # Flip a single bit
    if len(flipped_data) > 0:
        flipped_data[0] ^= 1  # Flip the least significant bit of first byte
        
        # Should fail verification
        assert not verify_commitment(master_key, nonce, bytes(flipped_data), commitment), \
            "Verification should fail with bit flip in data"
        
        print("✓ Commitment verification correctly fails with bit flip in data")
    
    # Test with wrong key
    wrong_key = b"wrong_key_for_testing_32_bytes!!"
    assert not verify_commitment(wrong_key, nonce, original_data, commitment), \
        "Verification should fail with wrong key"
    
    print("✓ Commitment verification correctly fails with wrong key")
    
    # Test with wrong nonce
    wrong_nonce = generate_nonce()
    assert not verify_commitment(master_key, wrong_nonce, original_data, commitment), \
        "Verification should fail with wrong nonce"
    
    print("✓ Commitment verification correctly fails with wrong nonce")
    
    # Test with bit flip in commitment
    flipped_commitment = bytearray(commitment)
    flipped_commitment[0] ^= 1
    assert not verify_commitment(master_key, nonce, original_data, bytes(flipped_commitment)), \
        "Verification should fail with bit flip in commitment"
    
    print("✓ Commitment verification correctly fails with bit flip in commitment")
    
    return True


def test_audit_record_structure():
    """Test audit record structure and timestamps."""
    print("Testing audit record structure...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate test data
        session_id = generate_session_id()
        nonce = generate_nonce()
        commitment = make_commitment(b"test_key", nonce, b"test_data")
        
        # Write audit record
        record = write_audit_record(
            session_id=session_id,
            model_id="test_model_v1",
            family="resnet",
            alpha=0.05,
            beta=0.10,
            boundary=0.85,
            nonce=nonce,
            commitment=commitment,
            prf_info={
                "algorithm": "HMAC-SHA256",
                "key_derivation": "HKDF",
                "seed_length": 32
            },
            reuse_policy="session",
            env={
                "python_version": "3.9.0",
                "torch_version": "1.13.0",
                "platform": "linux",
                "hostname": "test-host"
            },
            artifacts={
                "challenges_hash": "abc123",
                "ranges": [[0, 10], [10, 20]],
                "verification_result": {
                    "decision": "accept",
                    "confidence": 0.95,
                    "num_challenges": 20,
                    "duration_seconds": 5.2
                }
            },
            output_dir=tmpdir
        )
        
        # Verify structure
        assert "session_id" in record
        assert "timestamp" in record
        assert record["session_id"] == session_id
        assert record["alpha"] == 0.05
        assert record["beta"] == 0.10
        assert record["boundary"] == 0.85
        
        # Check hex encoding
        assert isinstance(record["nonce"], str)
        assert isinstance(record["commitment"], str)
        assert len(record["commitment"]) == 64  # 32 bytes as hex
        
        print("✓ Audit record has correct structure")
        
        # Verify timestamp format (ISO 8601)
        timestamp = record["timestamp"]
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            print(f"✓ Timestamp is valid ISO 8601: {timestamp}")
        except ValueError:
            raise AssertionError(f"Invalid timestamp format: {timestamp}")
        
        # Load and verify the written file
        audit_files = list(Path(tmpdir).glob("audit_*.json"))
        assert len(audit_files) == 1, f"Expected 1 audit file, found {len(audit_files)}"
        
        loaded_record = load_audit_record(audit_files[0])
        assert loaded_record["session_id"] == session_id
        assert "nonce_bytes" in loaded_record  # Added by load function
        assert "commitment_bytes" in loaded_record
        
        print("✓ Audit record successfully written and loaded from file")
        
        # Validate against schema
        is_valid, errors = validate_audit_record(record)
        assert is_valid, f"Audit record validation failed: {errors}"
        
        print("✓ Audit record validates against schema")
        
    return True


def test_session_id_generation():
    """Test that session IDs are unique."""
    print("Testing session ID generation...")
    
    # Generate multiple session IDs
    ids = set()
    for _ in range(100):
        session_id = generate_session_id()
        assert session_id.startswith("session_"), f"Invalid session ID format: {session_id}"
        assert len(session_id) == 24, f"Invalid session ID length: {len(session_id)}"
        ids.add(session_id)
        time.sleep(0.001)  # Small delay to ensure timestamp differences
    
    # All should be unique
    assert len(ids) == 100, f"Generated {100 - len(ids)} duplicate session IDs"
    
    print(f"✓ Generated 100 unique session IDs")
    return True


def test_nonce_generation():
    """Test nonce generation."""
    print("Testing nonce generation...")
    
    # Generate nonces of different sizes
    nonce1 = generate_nonce()
    assert len(nonce1) == 32, f"Default nonce should be 32 bytes, got {len(nonce1)}"
    
    nonce2 = generate_nonce(16)
    assert len(nonce2) == 16, f"Expected 16 byte nonce, got {len(nonce2)}"
    
    # Should be different
    nonce3 = generate_nonce()
    assert nonce1 != nonce3, "Nonces should be unique"
    
    print("✓ Nonce generation works correctly")
    return True


def test_serialization_types():
    """Test serialization with various data types."""
    print("Testing serialization with various types...")
    
    test_cases = [
        # Simple types
        ([1, 2, 3], [(0, 1)], {"key": "value"}),
        # Nested structures
        ([{"nested": {"deep": "value"}}], [(1, 2)], {"list": [1, 2, 3]}),
        # Mixed types
        ([1, "string", True, None], [(0, 10)], {"mixed": [1, "two", None]}),
        # Empty values
        ([], [], {}),
    ]
    
    for challenges, ranges, context in test_cases:
        data = serialize_for_commit(challenges, ranges, context)
        assert isinstance(data, bytes), "Serialization should return bytes"
        
        # Verify determinism
        data2 = serialize_for_commit(challenges, ranges, context)
        assert data == data2, "Serialization should be deterministic"
    
    print("✓ Serialization handles various data types correctly")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running POT Audit Infrastructure Tests")
    print("=" * 60)
    
    tests = [
        test_commitment_determinism,
        test_commitment_verification,
        test_audit_record_structure,
        test_session_id_generation,
        test_nonce_generation,
        test_serialization_types
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All tests passed successfully!")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)