#!/usr/bin/env python3
"""
Integration test demonstrating cryptographic utilities with existing audit system.

Shows how the new crypto utilities integrate with the commit-reveal protocol
and other audit infrastructure for enhanced security.
"""

import sys
import os
import hashlib
import json
from datetime import datetime, timezone

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pot.audit.crypto_utils import (
    generate_cryptographic_salt, compute_hash_chain, create_timestamp_proof,
    create_zk_proof, verify_timestamp_proof, verify_zk_proof,
    TimestampProofType
)
from pot.audit.commit_reveal import (
    compute_commitment, verify_reveal, serialize_for_commit
)


def test_crypto_audit_integration():
    """Test integration of cryptographic utilities with audit system."""
    print("Testing cryptographic utilities integration with audit system...")
    
    # 1. Enhanced commit-reveal with cryptographic salt
    verification_data = {
        "model_id": "integration_test_model",
        "verification_result": "PASS",
        "confidence": 0.95,
        "challenges_passed": 48,
        "challenges_total": 50
    }
    
    # Generate cryptographically secure salt instead of default
    crypto_salt = generate_cryptographic_salt(32)
    
    # Create commitment using crypto salt
    commitment = compute_commitment(verification_data, salt=crypto_salt)
    print(f"Enhanced commitment created: {commitment.commitment_hash[:16]}...")
    
    # 2. Create timestamp proof for the commitment
    commitment_bytes = serialize_for_commit(verification_data)
    timestamp_proof = create_timestamp_proof(commitment_bytes)
    
    # Verify timestamp proof
    timestamp_valid = verify_timestamp_proof(timestamp_proof, commitment_bytes)
    print(f"Timestamp proof valid: {timestamp_valid}")
    
    # 3. Create audit trail with hash chain
    audit_events = [
        commitment.commitment_hash.encode(),
        timestamp_proof.data_hash,
        b"verification_completed",
        b"audit_trail_finalized"
    ]
    
    # Hash each event
    event_hashes = [hashlib.sha256(event).digest() for event in audit_events]
    
    # Create tamper-evident audit chain
    audit_chain = compute_hash_chain(event_hashes)
    print(f"Audit chain created: {audit_chain.hex()[:16]}...")
    
    # 4. Create zero-knowledge proof of verification success
    public_statement = {
        "verification_completed": True,
        "confidence_above_threshold": True,
        "audit_trail_intact": True
    }
    
    private_witness = {
        "actual_verification_data": verification_data,
        "crypto_salt": crypto_salt.hex(),
        "audit_chain_hash": audit_chain.hex(),
        "timestamp_proof": timestamp_proof.to_dict()
    }
    
    zk_proof = create_zk_proof(public_statement, private_witness)
    zk_valid = verify_zk_proof(zk_proof)
    print(f"ZK proof valid: {zk_valid}")
    
    # 5. Verify the complete commit-reveal with enhanced crypto
    reveal_valid = verify_reveal(commitment, verification_data, crypto_salt)
    print(f"Enhanced commit-reveal valid: {reveal_valid}")
    
    # 6. Create complete audit package
    enhanced_audit_package = {
        "commitment_record": {
            "commitment_hash": commitment.commitment_hash,
            "salt": crypto_salt.hex(),
            "timestamp": commitment.timestamp
        },
        "timestamp_proof": timestamp_proof.to_dict(),
        "audit_chain": audit_chain.hex(),
        "zero_knowledge_proof": zk_proof.to_dict(),
        "public_verifiable_claims": public_statement,
        "cryptographic_guarantees": [
            "commitment_integrity",
            "timestamp_authenticity", 
            "audit_trail_immutability",
            "privacy_preservation"
        ]
    }
    
    print(f"\nEnhanced audit package created:")
    print(f"  Size: {len(json.dumps(enhanced_audit_package))} bytes")
    print(f"  Cryptographic components: {len(enhanced_audit_package['cryptographic_guarantees'])}")
    print(f"  All verifications passed: {all([timestamp_valid, zk_valid, reveal_valid])}")
    
    return all([timestamp_valid, zk_valid, reveal_valid])


def test_crypto_performance_integration():
    """Test performance characteristics of integrated crypto operations."""
    print("\nTesting performance of integrated cryptographic operations...")
    
    import time
    
    # Benchmark integrated workflow
    start_time = time.time()
    
    for i in range(100):
        # Quick integrated workflow
        salt = generate_cryptographic_salt()
        data = {"test": f"data_{i}"}
        commitment = compute_commitment(data, salt)
        timestamp_proof = create_timestamp_proof(commitment.commitment_hash.encode())
        
        # Verify everything
        assert verify_reveal(commitment, data, salt)
        assert verify_timestamp_proof(timestamp_proof, commitment.commitment_hash.encode())
    
    total_time = time.time() - start_time
    per_operation = total_time / 100
    
    print(f"  100 integrated operations: {total_time:.3f}s")
    print(f"  Per operation: {per_operation*1000:.2f}ms")
    print(f"  Operations/second: {100/total_time:.1f}")
    
    # Performance should be reasonable for production use
    assert per_operation < 0.1, "Integrated operations should be fast enough for production"
    print("  ✓ Performance meets production requirements")
    
    return True


def test_crypto_security_properties():
    """Test security properties of integrated system."""
    print("\nTesting security properties of integrated cryptographic system...")
    
    # 1. Test salt uniqueness prevents replay attacks
    data = {"sensitive": "verification_result"}
    
    salt1 = generate_cryptographic_salt()
    salt2 = generate_cryptographic_salt()
    
    commitment1 = compute_commitment(data, salt1)
    commitment2 = compute_commitment(data, salt2)
    
    # Same data with different salts should produce different commitments
    assert commitment1.commitment_hash != commitment2.commitment_hash
    print("  ✓ Salt uniqueness prevents replay attacks")
    
    # 2. Test hash chain tamper detection
    events = [b"event1", b"event2", b"event3"]
    event_hashes = [hashlib.sha256(e).digest() for e in events]
    
    original_chain = compute_hash_chain(event_hashes)
    
    # Tamper with middle event
    tampered_hashes = event_hashes.copy()
    tampered_hashes[1] = hashlib.sha256(b"tampered_event").digest()
    tampered_chain = compute_hash_chain(tampered_hashes)
    
    assert original_chain != tampered_chain
    print("  ✓ Hash chain detects tampering")
    
    # 3. Test ZK proof privacy preservation
    sensitive_data = {"secret_key": "super_secret", "private_score": 0.97}
    public_claim = {"score_above_threshold": True}
    
    zk_proof = create_zk_proof(public_claim, sensitive_data)
    
    # Verifier should only see public statement, not witness
    assert zk_proof.statement == public_claim
    assert "secret_key" not in str(zk_proof.proof_data)
    assert "private_score" not in str(zk_proof.proof_data)
    print("  ✓ ZK proof preserves privacy")
    
    # 4. Test timestamp proof integrity
    important_data = b"critical_verification_result"
    timestamp_proof = create_timestamp_proof(important_data)
    
    # Proof should fail with wrong data
    wrong_data = b"modified_verification_result"
    assert not verify_timestamp_proof(timestamp_proof, wrong_data)
    print("  ✓ Timestamp proof ensures data integrity")
    
    return True


def run_integration_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("CRYPTOGRAPHIC UTILITIES INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_crypto_audit_integration,
        test_crypto_performance_integration, 
        test_crypto_security_properties
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✓ {test_func.__name__} PASSED")
            else:
                failed += 1
                print(f"✗ {test_func.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("\nCryptographic utilities successfully integrated with audit system!")
        print("Enhanced security features:")
        print("  ✓ Cryptographically secure randomness")
        print("  ✓ Tamper-evident audit trails")
        print("  ✓ Timestamp proof authenticity")
        print("  ✓ Privacy-preserving verification")
        print("  ✓ Production-ready performance")
    else:
        print(f"❌ {failed} integration tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)