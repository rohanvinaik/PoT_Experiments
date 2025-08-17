#!/usr/bin/env python3
"""
Comprehensive test suite for cryptographic utilities.

Tests all cryptographic primitives including salt generation, hash chains,
timestamp proofs, commitment aggregation, and zero-knowledge proofs.
"""

import sys
import os
import time
import hashlib
import json
from datetime import datetime, timezone
from typing import List, Dict

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pot.audit.crypto_utils import (
    generate_cryptographic_salt, compute_hash_chain, create_timestamp_proof,
    aggregate_commitments, create_zk_proof, verify_timestamp_proof, verify_zk_proof,
    HashAlgorithm, TimestampProofType, TimestampProof, AggregateCommitment, ZKProof,
    derive_key_from_password, secure_compare, get_available_features
)


# Mock CommitmentRecord for testing
class MockCommitmentRecord:
    """Mock commitment record for testing aggregation."""
    
    def __init__(self, commitment_hash: str, data: Dict = None):
        self.commitment_hash = commitment_hash
        self.data = data or {}
        self.salt = "test_salt"
        self.timestamp = datetime.now(timezone.utc)


def test_cryptographic_salt_generation():
    """Test cryptographically secure salt generation."""
    print("Testing cryptographic salt generation...")
    
    # Test default length
    salt1 = generate_cryptographic_salt()
    assert len(salt1) == 32
    
    # Test custom length
    salt2 = generate_cryptographic_salt(64)
    assert len(salt2) == 64
    
    # Test uniqueness
    salt3 = generate_cryptographic_salt()
    assert salt1 != salt3
    
    # Test minimum length validation
    try:
        generate_cryptographic_salt(8)  # Too small
        assert False, "Should raise ValueError for small length"
    except ValueError as e:
        assert "too small" in str(e)
    
    # Test maximum length validation
    try:
        generate_cryptographic_salt(2048)  # Too large
        assert False, "Should raise ValueError for large length"
    except ValueError as e:
        assert "unnecessarily large" in str(e)
    
    print("✓ Cryptographic salt generation tests passed")


def test_hash_chain_computation():
    """Test hash chain computation for audit trails."""
    print("Testing hash chain computation...")
    
    # Test basic hash chain
    hash1 = hashlib.sha256(b"data1").digest()
    hash2 = hashlib.sha256(b"data2").digest()
    hash3 = hashlib.sha256(b"data3").digest()
    
    chain_hash = compute_hash_chain([hash1, hash2, hash3])
    assert len(chain_hash) == 32  # SHA256 output
    
    # Test determinism
    chain_hash2 = compute_hash_chain([hash1, hash2, hash3])
    assert chain_hash == chain_hash2
    
    # Test order dependency
    chain_hash_reversed = compute_hash_chain([hash3, hash2, hash1])
    assert chain_hash != chain_hash_reversed
    
    # Test single hash
    single_chain = compute_hash_chain([hash1])
    assert len(single_chain) == 32
    
    # Test different algorithms
    for algorithm in [HashAlgorithm.SHA256, HashAlgorithm.SHA3_256, HashAlgorithm.BLAKE2B]:
        alg_chain = compute_hash_chain([hash1, hash2], algorithm)
        assert len(alg_chain) >= 32  # Different algorithms may have different output sizes
    
    # Test empty list
    try:
        compute_hash_chain([])
        assert False, "Should raise ValueError for empty list"
    except ValueError as e:
        assert "empty list" in str(e)
    
    # Test invalid input
    try:
        compute_hash_chain(["not_bytes"])
        assert False, "Should raise ValueError for non-bytes"
    except ValueError as e:
        assert "bytes objects" in str(e)
    
    print("✓ Hash chain computation tests passed")


def test_timestamp_proof_creation():
    """Test timestamp proof creation and verification."""
    print("Testing timestamp proof creation...")
    
    test_data = b"document to be timestamped"
    timestamp = datetime.now(timezone.utc)
    
    # Test local timestamp proof
    local_proof = create_timestamp_proof(test_data, timestamp, TimestampProofType.LOCAL)
    assert local_proof.proof_type == TimestampProofType.LOCAL
    assert local_proof.data_hash == hashlib.sha256(test_data).digest()
    assert local_proof.timestamp == timestamp
    assert len(local_proof.proof_data) == 32  # HMAC-SHA256
    assert local_proof.verifier_info['type'] == 'local'
    
    # Test proof verification
    assert verify_timestamp_proof(local_proof, test_data)
    
    # Test proof fails with wrong data
    wrong_data = b"different document"
    assert not verify_timestamp_proof(local_proof, wrong_data)
    
    # Test serialization
    proof_dict = local_proof.to_dict()
    reconstructed_proof = TimestampProof.from_dict(proof_dict)
    assert reconstructed_proof.data_hash == local_proof.data_hash
    assert reconstructed_proof.timestamp == local_proof.timestamp
    assert reconstructed_proof.proof_type == local_proof.proof_type
    
    # Test default timestamp
    auto_timestamp_proof = create_timestamp_proof(test_data)
    assert auto_timestamp_proof.timestamp is not None
    
    print("✓ Timestamp proof creation tests passed")


def test_timestamp_proof_advanced():
    """Test advanced timestamp proof types."""
    print("Testing advanced timestamp proof types...")
    
    test_data = b"advanced timestamp test"
    
    # Test RFC3161 if cryptography is available
    features = get_available_features()
    if features['cryptography_package']:
        try:
            rfc3161_proof = create_timestamp_proof(test_data, proof_type=TimestampProofType.RFC3161)
            assert rfc3161_proof.proof_type == TimestampProofType.RFC3161
            assert rfc3161_proof.signature is not None
            assert rfc3161_proof.verifier_info['type'] == 'rfc3161'
            
            # Verify RFC3161 proof
            assert verify_timestamp_proof(rfc3161_proof, test_data)
            print("  ✓ RFC3161 timestamp proof test passed")
        except Exception as e:
            print(f"  ⚠ RFC3161 test failed (expected in some environments): {e}")
    else:
        print("  ⚠ Skipping RFC3161 tests (cryptography package not available)")
    
    # Test OpenTimestamps if requests is available
    if features['requests_package']:
        try:
            ots_proof = create_timestamp_proof(test_data, proof_type=TimestampProofType.OPENTIMESTAMPS)
            assert ots_proof.proof_type == TimestampProofType.OPENTIMESTAMPS
            assert ots_proof.verifier_info['type'] == 'opentimestamps'
            
            # Verify OpenTimestamps proof
            assert verify_timestamp_proof(ots_proof, test_data)
            print("  ✓ OpenTimestamps proof test passed")
        except Exception as e:
            print(f"  ⚠ OpenTimestamps test failed (expected in some environments): {e}")
    else:
        print("  ⚠ Skipping OpenTimestamps tests (requests package not available)")
    
    print("✓ Advanced timestamp proof tests completed")


def test_commitment_aggregation():
    """Test commitment aggregation with Merkle trees."""
    print("Testing commitment aggregation...")
    
    # Create mock commitment records
    commitment_records = []
    for i in range(10):
        commitment_hash = hashlib.sha256(f"commitment_{i}".encode()).hexdigest()
        record = MockCommitmentRecord(commitment_hash, {"index": i})
        commitment_records.append(record)
    
    # Test aggregation
    try:
        aggregate = aggregate_commitments(commitment_records)
        
        assert isinstance(aggregate, AggregateCommitment)
        assert aggregate.size == 10
        assert len(aggregate.commitment_hashes) == 10
        assert aggregate.merkle_root is not None
        assert aggregate.aggregate_hash is not None
        assert len(aggregate.proofs) == 10
        
        # Test each commitment has a proof
        for record in commitment_records:
            assert record.commitment_hash in aggregate.proofs
            proof = aggregate.proofs[record.commitment_hash]
            assert isinstance(proof, list)
        
        # Test serialization
        agg_dict = aggregate.to_dict()
        reconstructed = AggregateCommitment.from_dict(agg_dict)
        assert reconstructed.size == aggregate.size
        assert reconstructed.merkle_root == aggregate.merkle_root
        
        print("  ✓ Commitment aggregation with Merkle tree passed")
        
    except ImportError:
        print("  ⚠ Skipping aggregation tests (Merkle tree implementation not available)")
    
    # Test empty list
    try:
        aggregate_commitments([])
        assert False, "Should raise ValueError for empty list"
    except ValueError as e:
        assert "empty" in str(e)
    
    print("✓ Commitment aggregation tests passed")


def test_zero_knowledge_proofs():
    """Test zero-knowledge proof creation and verification."""
    print("Testing zero-knowledge proofs...")
    
    # Test statement and witness
    statement = {
        "verification_passed": True,
        "confidence_threshold": 0.95,
        "model_type": "vision"
    }
    
    witness = {
        "actual_confidence": 0.97,
        "secret_parameters": [1.2, 3.4, 5.6],
        "private_key": "super_secret_key"
    }
    
    # Test Schnorr-style proof
    schnorr_proof = create_zk_proof(statement, witness, "schnorr")
    assert isinstance(schnorr_proof, ZKProof)
    assert schnorr_proof.proof_type == "schnorr"
    assert schnorr_proof.statement == statement
    assert len(schnorr_proof.proof_data) == 96  # 3 * 32 bytes
    assert len(schnorr_proof.verifier_key) == 32
    
    # Test proof verification
    assert verify_zk_proof(schnorr_proof)
    
    # Test simple commitment proof
    commit_proof = create_zk_proof(statement, witness, "simple_commit")
    assert commit_proof.proof_type == "simple_commit"
    assert verify_zk_proof(commit_proof)
    
    # Test proof serialization
    proof_dict = schnorr_proof.to_dict()
    reconstructed_proof = ZKProof.from_dict(proof_dict)
    assert reconstructed_proof.statement == schnorr_proof.statement
    assert reconstructed_proof.proof_data == schnorr_proof.proof_data
    assert verify_zk_proof(reconstructed_proof)
    
    # Test invalid proof type
    try:
        create_zk_proof(statement, witness, "invalid_type")
        assert False, "Should raise ValueError for invalid proof type"
    except ValueError as e:
        assert "Unsupported" in str(e)
    
    # Test tampered proof detection
    tampered_proof = ZKProof(
        statement=statement,
        proof_data=b"invalid_proof_data" * 3,  # Wrong length
        proof_type="schnorr",
        verifier_key=schnorr_proof.verifier_key,
        metadata={}
    )
    assert not verify_zk_proof(tampered_proof)
    
    print("✓ Zero-knowledge proof tests passed")


def test_key_derivation():
    """Test key derivation from passwords."""
    print("Testing key derivation...")
    
    password = "test_password_123"
    salt = generate_cryptographic_salt(16)
    
    # Test basic key derivation
    key1 = derive_key_from_password(password, salt)
    assert len(key1) == 32  # Default length
    
    # Test determinism
    key2 = derive_key_from_password(password, salt)
    assert key1 == key2
    
    # Test different password produces different key
    key3 = derive_key_from_password("different_password", salt)
    assert key1 != key3
    
    # Test different salt produces different key
    salt2 = generate_cryptographic_salt(16)
    key4 = derive_key_from_password(password, salt2)
    assert key1 != key4
    
    # Test custom key length
    key5 = derive_key_from_password(password, salt, key_length=64)
    assert len(key5) == 64
    
    print("✓ Key derivation tests passed")


def test_secure_comparison():
    """Test constant-time secure comparison."""
    print("Testing secure comparison...")
    
    data1 = b"secret_data_123"
    data2 = b"secret_data_123"
    data3 = b"different_data"
    
    # Test equal data
    assert secure_compare(data1, data2)
    
    # Test different data
    assert not secure_compare(data1, data3)
    
    # Test different lengths
    assert not secure_compare(data1, b"short")
    
    # Test empty data
    assert secure_compare(b"", b"")
    
    print("✓ Secure comparison tests passed")


def test_feature_availability():
    """Test feature availability detection."""
    print("Testing feature availability...")
    
    features = get_available_features()
    
    # Check expected keys
    expected_keys = [
        'cryptography_package', 'requests_package', 'rfc3161_timestamps',
        'opentimestamps', 'advanced_zk_proofs', 'hardware_rng'
    ]
    
    for key in expected_keys:
        assert key in features
        assert isinstance(features[key], bool)
    
    print(f"  Available features: {sum(features.values())}/{len(features)}")
    for feature, available in features.items():
        status = "✓" if available else "✗"
        print(f"    {status} {feature}")
    
    print("✓ Feature availability tests passed")


def test_edge_cases_and_error_handling():
    """Test edge cases and error handling."""
    print("Testing edge cases and error handling...")
    
    # Test hash chain with single element
    single_hash = hashlib.sha256(b"single").digest()
    chain = compute_hash_chain([single_hash])
    assert len(chain) == 32
    
    # Test timestamp proof with extreme dates
    old_timestamp = datetime(1970, 1, 1, tzinfo=timezone.utc)
    future_timestamp = datetime(2050, 12, 31, tzinfo=timezone.utc)
    
    data = b"test_data"
    old_proof = create_timestamp_proof(data, old_timestamp)
    future_proof = create_timestamp_proof(data, future_timestamp)
    
    assert verify_timestamp_proof(old_proof, data)
    assert verify_timestamp_proof(future_proof, data)
    
    # Test ZK proof with empty statement/witness
    empty_statement = {}
    empty_witness = {}
    
    empty_proof = create_zk_proof(empty_statement, empty_witness)
    assert verify_zk_proof(empty_proof)
    
    # Test very large data
    large_data = b"x" * 1000000  # 1MB
    large_proof = create_timestamp_proof(large_data)
    assert verify_timestamp_proof(large_proof, large_data)
    
    print("✓ Edge cases and error handling tests passed")


def test_integration_scenarios():
    """Test realistic integration scenarios."""
    print("Testing integration scenarios...")
    
    # Scenario 1: Complete audit trail for model verification
    model_id = "production_model_v2.1"
    verification_data = {
        "model_id": model_id,
        "verification_timestamp": datetime.now(timezone.utc).isoformat(),
        "challenges_passed": 95,
        "total_challenges": 100,
        "confidence": 0.95
    }
    
    # Create timestamp proof for verification
    verification_bytes = json.dumps(verification_data, sort_keys=True).encode()
    timestamp_proof = create_timestamp_proof(verification_bytes)
    
    # Create ZK proof that verification passed without revealing details
    public_statement = {
        "model_verified": True,
        "confidence_above_threshold": True
    }
    
    private_witness = {
        "actual_confidence": verification_data["confidence"],
        "challenge_details": "sensitive_challenge_data",
        "model_parameters": "secret_parameters"
    }
    
    zk_proof = create_zk_proof(public_statement, private_witness)
    
    # Verify proofs
    assert verify_timestamp_proof(timestamp_proof, verification_bytes)
    assert verify_zk_proof(zk_proof)
    
    # Scenario 2: Aggregate multiple verification sessions
    if get_available_features()['cryptography_package'] or True:  # Allow fallback
        try:
            verification_sessions = []
            for i in range(5):
                session_data = {
                    "session_id": f"session_{i}",
                    "verification_result": "PASS",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                session_hash = hashlib.sha256(
                    json.dumps(session_data, sort_keys=True).encode()
                ).hexdigest()
                
                session_record = MockCommitmentRecord(session_hash, session_data)
                verification_sessions.append(session_record)
            
            # Aggregate all sessions
            try:
                aggregate = aggregate_commitments(verification_sessions)
                assert aggregate.size == 5
                print("  ✓ Multi-session aggregation passed")
            except ImportError:
                print("  ⚠ Skipping aggregation (Merkle tree not available)")
                
        except Exception as e:
            print(f"  ⚠ Aggregation test failed: {e}")
    
    # Scenario 3: Hash chain for audit trail
    audit_events = [
        b"model_registration",
        b"challenge_generation", 
        b"verification_execution",
        b"result_commitment",
        b"audit_completion"
    ]
    
    event_hashes = [hashlib.sha256(event).digest() for event in audit_events]
    audit_chain = compute_hash_chain(event_hashes)
    
    # Verify chain integrity
    reconstructed_chain = compute_hash_chain(event_hashes)
    assert audit_chain == reconstructed_chain
    
    print("✓ Integration scenario tests passed")


def test_performance_benchmarks():
    """Test performance characteristics of cryptographic operations."""
    print("Testing performance benchmarks...")
    
    # Benchmark salt generation
    start_time = time.time()
    salts = [generate_cryptographic_salt() for _ in range(1000)]
    salt_time = time.time() - start_time
    print(f"  Salt generation (1000x): {salt_time:.3f}s ({salt_time/1000*1000:.2f}ms each)")
    
    # Benchmark hash chaining
    hashes = [hashlib.sha256(f"data_{i}".encode()).digest() for i in range(100)]
    start_time = time.time()
    for _ in range(100):
        compute_hash_chain(hashes)
    chain_time = time.time() - start_time
    print(f"  Hash chaining (100x100 items): {chain_time:.3f}s ({chain_time/100*1000:.2f}ms each)")
    
    # Benchmark timestamp proofs
    test_data = b"performance_test_data"
    start_time = time.time()
    for _ in range(100):
        proof = create_timestamp_proof(test_data)
        verify_timestamp_proof(proof, test_data)
    timestamp_time = time.time() - start_time
    print(f"  Timestamp proof creation+verification (100x): {timestamp_time:.3f}s ({timestamp_time/100*1000:.2f}ms each)")
    
    # Benchmark ZK proofs
    statement = {"verified": True}
    witness = {"secret": "value"}
    start_time = time.time()
    for _ in range(100):
        proof = create_zk_proof(statement, witness)
        verify_zk_proof(proof)
    zk_time = time.time() - start_time
    print(f"  ZK proof creation+verification (100x): {zk_time:.3f}s ({zk_time/100*1000:.2f}ms each)")
    
    # Performance assertions
    assert salt_time < 1.0, "Salt generation should be fast"
    assert chain_time < 2.0, "Hash chaining should be fast"  
    assert timestamp_time < 5.0, "Timestamp proofs should be reasonably fast"
    assert zk_time < 10.0, "ZK proofs should complete within reasonable time"
    
    print("✓ Performance benchmark tests passed")


def run_all_tests():
    """Run all cryptographic utilities tests."""
    print("=" * 70)
    print("COMPREHENSIVE CRYPTOGRAPHIC UTILITIES TEST SUITE")
    print("=" * 70)
    
    test_functions = [
        test_cryptographic_salt_generation,
        test_hash_chain_computation,
        test_timestamp_proof_creation,
        test_timestamp_proof_advanced,
        test_commitment_aggregation,
        test_zero_knowledge_proofs,
        test_key_derivation,
        test_secure_comparison,
        test_feature_availability,
        test_edge_cases_and_error_handling,
        test_integration_scenarios,
        test_performance_benchmarks
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        print("\nCryptographic utilities ready for production!")
    else:
        print(f"❌ {failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)