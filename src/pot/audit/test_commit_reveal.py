#!/usr/bin/env python3
"""
Comprehensive test suite for commit-reveal protocol in the PoT audit system.

Tests all aspects of cryptographic commitments, reveals, audit trail integrity,
and schema validation for the commit-reveal protocol.
"""

import sys
import os
import json
import tempfile
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pot.audit.commit_reveal import (
    CommitmentRecord, serialize_for_commit, compute_commitment,
    verify_reveal, write_commitment_record, write_audit_record,
    read_and_verify_audit_trail, read_commitment_records,
    make_commitment, verify_commitment  # Legacy compatibility
)
from pot.audit.schema import (
    validate_audit_record, sanitize_for_audit, create_enhanced_audit_record,
    validate_commitment_record, get_schema_version
)


class TestCommitRevealProtocol:
    """Comprehensive test suite for commit-reveal protocol."""
    
    def test_commitment_generation(self):
        """Test commitment generation with various data types."""
        print("Testing commitment generation...")
        
        # Test with basic verification data
        verification_data = {
            'model_id': 'test_model_v1',
            'verification_decision': 'PASS',
            'confidence': 0.95,
            'challenges_passed': 48,
            'challenges_total': 50,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Test automatic commitment generation
        commitment = compute_commitment(verification_data)
        assert isinstance(commitment, CommitmentRecord)
        assert len(commitment.commitment_hash) == 64  # SHA256 hex
        assert len(commitment.salt) == 64  # 32 bytes hex
        assert commitment.timestamp is not None
        assert commitment.version == '1.0'
        
        # Test determinism with same salt
        salt_bytes = bytes.fromhex(commitment.salt)
        commitment2 = compute_commitment(verification_data, salt=salt_bytes)
        assert commitment.commitment_hash == commitment2.commitment_hash
        assert commitment.salt == commitment2.salt
        
        # Test different data produces different commitment
        different_data = verification_data.copy()
        different_data['confidence'] = 0.85
        commitment3 = compute_commitment(different_data)
        assert commitment.commitment_hash != commitment3.commitment_hash
        
        print("  ✓ Commitment generation tests passed")
    
    def test_reveal_protocol_correct_data(self):
        """Test reveal protocol with correct data."""
        print("Testing reveal protocol with correct data...")
        
        original_data = {
            'session_id': 'test_session_123',
            'model_verification': {
                'model_id': 'secure_model_v2',
                'verification_result': 'PASS',
                'confidence_score': 0.97,
                'verification_timestamp': datetime.now(timezone.utc).isoformat()
            },
            'challenge_results': [
                {'challenge_id': 'c1', 'passed': True, 'score': 0.98},
                {'challenge_id': 'c2', 'passed': True, 'score': 0.96},
                {'challenge_id': 'c3', 'passed': True, 'score': 0.97}
            ]
        }
        
        # Create commitment
        commitment = compute_commitment(original_data)
        
        # Test successful reveal
        salt_bytes = bytes.fromhex(commitment.salt)
        reveal_result = verify_reveal(commitment, original_data, salt_bytes)
        assert reveal_result is True
        
        # Test reveal with exact same data (should pass)
        reveal_result2 = verify_reveal(commitment, original_data, salt_bytes)
        assert reveal_result2 is True
        
        print("  ✓ Reveal protocol with correct data tests passed")
    
    def test_reveal_protocol_incorrect_data(self):
        """Test reveal protocol with incorrect/tampered data."""
        print("Testing reveal protocol with incorrect data...")
        
        original_data = {
            'verification_decision': 'PASS',
            'confidence': 0.95,
            'model_id': 'original_model',
            'session_id': 'session_abc123'
        }
        
        # Create commitment
        commitment = compute_commitment(original_data)
        salt_bytes = bytes.fromhex(commitment.salt)
        
        # Test 1: Modified confidence score
        tampered_data1 = original_data.copy()
        tampered_data1['confidence'] = 0.85  # Changed value
        assert verify_reveal(commitment, tampered_data1, salt_bytes) is False
        
        # Test 2: Modified decision
        tampered_data2 = original_data.copy()
        tampered_data2['verification_decision'] = 'FAIL'
        assert verify_reveal(commitment, tampered_data2, salt_bytes) is False
        
        # Test 3: Added extra field
        tampered_data3 = original_data.copy()
        tampered_data3['extra_field'] = 'malicious_addition'
        assert verify_reveal(commitment, tampered_data3, salt_bytes) is False
        
        # Test 4: Removed field
        tampered_data4 = original_data.copy()
        del tampered_data4['model_id']
        assert verify_reveal(commitment, tampered_data4, salt_bytes) is False
        
        # Test 5: Wrong salt
        wrong_salt = b'wrong_salt_32_bytes_long_exactly!!'
        assert len(wrong_salt) == 32
        assert verify_reveal(commitment, original_data, wrong_salt) is False
        
        # Test 6: Completely different data
        different_data = {
            'completely': 'different',
            'data': 'structure',
            'nothing': 'in_common'
        }
        assert verify_reveal(commitment, different_data, salt_bytes) is False
        
        print("  ✓ Reveal protocol with incorrect data tests passed")
    
    def test_audit_trail_integrity(self):
        """Test audit trail integrity and tamper detection."""
        print("Testing audit trail integrity...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_file = os.path.join(temp_dir, "test_audit_trail.json")
            
            # Create sequence of verification events
            verification_events = []
            commitments = []
            
            for i in range(5):
                event_data = {
                    'event_id': f'event_{i:03d}',
                    'model_id': f'model_{chr(65 + i % 3)}',  # A, B, C
                    'verification_decision': 'PASS' if i % 4 != 0 else 'FAIL',
                    'confidence': 0.8 + (i % 5) * 0.04,
                    'timestamp': (datetime.now(timezone.utc) + timedelta(minutes=i)).isoformat(),
                    'challenge_count': 20 + i * 5
                }
                
                # Create commitment for this event
                commitment = compute_commitment(event_data)
                commitments.append(commitment)
                
                # Create audit record using the proper schema
                audit_record = create_enhanced_audit_record(
                    commitment_hash=commitment.commitment_hash,
                    commitment_algorithm='SHA256',
                    salt_length=32,
                    verification_decision=event_data['verification_decision'],
                    verification_confidence=event_data['confidence'],
                    samples_used=event_data['challenge_count'],
                    model_id=event_data['model_id'],
                    verifier_version='2.1.0',
                    metadata={
                        'sequence_number': i,
                        'event_data': event_data,
                        'commitment_record': {
                            'commitment_hash': commitment.commitment_hash,
                            'salt': commitment.salt,
                            'timestamp': commitment.timestamp,
                            'version': commitment.version
                        }
                    }
                )
                
                verification_events.append(audit_record)
                
                # Write each record to audit trail
                write_audit_record(audit_record, audit_file)
            
            # Test 1: Read and verify complete audit trail
            loaded_trail = read_and_verify_audit_trail(audit_file)
            assert len(loaded_trail) == 5
            
            # Verify each commitment in the trail
            for i, record in enumerate(loaded_trail):
                original_event = verification_events[i]['metadata']['event_data']
                commitment_info = record['metadata']['commitment_record']
                
                # Reconstruct commitment record
                commitment = CommitmentRecord(
                    commitment_hash=commitment_info['commitment_hash'],
                    salt=commitment_info['salt'],
                    timestamp=commitment_info['timestamp'],
                    version=commitment_info['version']
                )
                
                # Verify reveal
                salt_bytes = bytes.fromhex(commitment.salt)
                assert verify_reveal(commitment, original_event, salt_bytes)
            
            # Test 2: Detect tampered audit trail
            tampered_file = os.path.join(temp_dir, "tampered_audit.json")
            
            # Copy audit trail and tamper with one record
            with open(audit_file, 'r') as f:
                lines = f.readlines()
            
            # Tamper with middle record
            if len(lines) >= 3:
                middle_record = json.loads(lines[2])
                middle_record['event_data']['confidence'] = 0.99  # Changed value
                lines[2] = json.dumps(middle_record) + '\n'
            
            with open(tampered_file, 'w') as f:
                f.writelines(lines)
            
            # Load tampered trail
            tampered_trail = read_and_verify_audit_trail(tampered_file)
            
            # Verification should fail for tampered record
            tampered_record = tampered_trail[2]
            tampered_event = tampered_record['event_data']
            tampered_commitment_info = tampered_record['commitment_record']
            
            tampered_commitment = CommitmentRecord(
                commitment_hash=tampered_commitment_info['commitment_hash'],
                salt=tampered_commitment_info['salt'],
                timestamp=tampered_commitment_info['timestamp'],
                version=tampered_commitment_info['version']
            )
            
            salt_bytes = bytes.fromhex(tampered_commitment.salt)
            tamper_detected = not verify_reveal(tampered_commitment, tampered_event, salt_bytes)
            assert tamper_detected, "Should detect tampering in audit trail"
            
            # Test 3: Commitment record isolation
            commitment_file = os.path.join(temp_dir, "commitments_only.json")
            
            # Write only commitment records
            for commitment in commitments:
                write_commitment_record(commitment, commitment_file)
            
            # Read commitment records
            loaded_commitments = read_commitment_records(commitment_file)
            assert len(loaded_commitments) == 5
            
            # Verify commitment structure
            for loaded_commitment in loaded_commitments:
                assert isinstance(loaded_commitment, CommitmentRecord)
                assert len(loaded_commitment.commitment_hash) == 64
                assert len(loaded_commitment.salt) == 64
                assert loaded_commitment.version == '1.0'
        
        print("  ✓ Audit trail integrity tests passed")
    
    def test_schema_validation(self):
        """Test schema validation for commitments and audit records."""
        print("Testing schema validation...")
        
        # Test 1: Valid commitment record schema
        timestamp_str = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        valid_commitment_data = {
            'commitment_hash': 'a' * 64,  # Valid 64-char hex
            'salt': 'b' * 64,  # Valid 64-char hex
            'timestamp': timestamp_str,
            'version': '1.0'
        }
        
        is_valid, errors = validate_commitment_record(valid_commitment_data)
        assert is_valid, f"Valid commitment should pass validation: {errors}"
        
        # Test 2: Invalid commitment record schemas
        invalid_cases = [
            # Missing required field
            {'salt': 'b' * 64, 'timestamp': timestamp_str, 'version': '1.0'},
            # Invalid hash length
            {'commitment_hash': 'short', 'salt': 'b' * 64, 'timestamp': timestamp_str, 'version': '1.0'},
            # Invalid timestamp format
            {'commitment_hash': 'a' * 64, 'salt': 'b' * 64, 'timestamp': 'not_a_timestamp', 'version': '1.0'},
            # Invalid version
            {'commitment_hash': 'a' * 64, 'salt': 'b' * 64, 'timestamp': timestamp_str, 'version': 'invalid'}
        ]
        
        for invalid_data in invalid_cases:
            is_valid, errors = validate_commitment_record(invalid_data)
            assert not is_valid, f"Invalid commitment should fail validation: {invalid_data}"
            assert len(errors) > 0
        
        # Test 3: Valid enhanced audit record
        valid_audit_record = create_enhanced_audit_record(
            commitment_hash='c' * 64,
            commitment_algorithm='SHA256',
            salt_length=32,
            verification_decision='PASS',
            verification_confidence=0.95,
            samples_used=50,
            model_id='test_model_v1',
            verifier_version='2.1.0',
            metadata={'test': 'data'}
        )
        
        is_valid, errors = validate_audit_record(valid_audit_record)
        assert is_valid, f"Valid audit record should pass validation: {errors}"
        
        # Test 4: Data sanitization
        sensitive_data = {
            'public_info': 'this_is_ok',
            'api_key': 'secret_key_12345',
            'password': 'super_secret_password',
            'verification_result': 'PASS',
            'confidence': 0.95,
            'model_weights': list(range(1000)),  # Large data
            'private_key': 'very_secret_key'
        }
        
        sanitized = sanitize_for_audit(sensitive_data)
        
        # Sensitive fields should be redacted
        assert sanitized['api_key'] == '[REDACTED]'
        assert sanitized['password'] == '[REDACTED]'
        assert sanitized['private_key'] == '[REDACTED]'
        
        # Public fields should remain
        assert sanitized['public_info'] == 'this_is_ok'
        assert sanitized['verification_result'] == 'PASS'
        assert sanitized['confidence'] == 0.95
        
        # Large data should be truncated
        assert '[TRUNCATED' in str(sanitized['model_weights'])
        
        print("  ✓ Schema validation tests passed")
    
    def test_legacy_compatibility(self):
        """Test compatibility with legacy HMAC-based commit-reveal."""
        print("Testing legacy compatibility...")
        
        # Test legacy commitment generation
        data = {'test': 'legacy_data', 'value': 123}
        key = b'test_key_32_bytes_long_exactly!!!'
        nonce = b'nonce_16_bytes!!'
        data_bytes = json.dumps(data, sort_keys=True).encode()
        
        legacy_commitment = make_commitment(key, nonce, data_bytes)
        assert len(legacy_commitment) == 64  # HMAC-SHA256 hex
        
        # Test legacy verification
        is_valid = verify_commitment(legacy_commitment, key, nonce, data_bytes)
        assert is_valid
        
        # Test legacy verification with wrong data
        wrong_data_bytes = json.dumps({'test': 'different_data', 'value': 456}, sort_keys=True).encode()
        is_valid_wrong = verify_commitment(legacy_commitment, key, nonce, wrong_data_bytes)
        assert not is_valid_wrong
        
        # Test legacy verification with wrong key
        wrong_key = b'wrong_key_32_bytes_long_exactly!!'
        is_valid_wrong_key = verify_commitment(legacy_commitment, wrong_key, nonce, data_bytes)
        assert not is_valid_wrong_key
        
        print("  ✓ Legacy compatibility tests passed")
    
    def test_serialization_and_canonicalization(self):
        """Test data serialization and canonicalization."""
        print("Testing serialization and canonicalization...")
        
        # Test canonical serialization with different field orders
        data1 = {'b': 2, 'a': 1, 'c': 3}
        data2 = {'a': 1, 'c': 3, 'b': 2}
        data3 = {'c': 3, 'a': 1, 'b': 2}
        
        serialized1 = serialize_for_commit(data1)
        serialized2 = serialize_for_commit(data2)
        serialized3 = serialize_for_commit(data3)
        
        # All should produce identical canonical serialization
        assert serialized1 == serialized2 == serialized3
        
        # Test with nested data
        nested_data1 = {
            'outer': {'z': 26, 'a': 1, 'm': 13},
            'list': [3, 1, 2],
            'simple': 'value'
        }
        
        nested_data2 = {
            'simple': 'value',
            'list': [3, 1, 2],
            'outer': {'a': 1, 'm': 13, 'z': 26}
        }
        
        nested_serialized1 = serialize_for_commit(nested_data1)
        nested_serialized2 = serialize_for_commit(nested_data2)
        
        assert nested_serialized1 == nested_serialized2
        
        # Test timestamp handling
        timestamp_str = datetime.now(timezone.utc).isoformat()
        data_with_timestamp = {
            'timestamp': timestamp_str,
            'other_data': 'test'
        }
        
        # Should not raise errors and produce consistent output
        serialized_timestamp = serialize_for_commit(data_with_timestamp)
        assert isinstance(serialized_timestamp, bytes)
        assert timestamp_str.encode() in serialized_timestamp
        
        print("  ✓ Serialization and canonicalization tests passed")
    
    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases."""
        print("Testing error handling and edge cases...")
        
        # Test 1: Empty data
        empty_data = {}
        commitment_empty = compute_commitment(empty_data)
        assert isinstance(commitment_empty, CommitmentRecord)
        assert len(commitment_empty.commitment_hash) == 64
        
        # Verify empty data
        salt_bytes = bytes.fromhex(commitment_empty.salt)
        assert verify_reveal(commitment_empty, empty_data, salt_bytes)
        
        # Test 2: Very large data
        large_data = {
            'large_list': list(range(10000)),
            'large_string': 'x' * 100000,
            'nested_large': {
                'inner': list(range(5000))
            }
        }
        
        commitment_large = compute_commitment(large_data)
        salt_bytes_large = bytes.fromhex(commitment_large.salt)
        assert verify_reveal(commitment_large, large_data, salt_bytes_large)
        
        # Test 3: Unicode and special characters
        unicode_data = {
            'unicode': '测试数据 🔒 🛡️',
            'special_chars': '!@#$%^&*()_+-=[]{}|;:,.<>?',
            'emoji': '🚀 🔐 ✨ 💯',
            'mixed': 'ASCII + 中文 + العربية + 🌍'
        }
        
        commitment_unicode = compute_commitment(unicode_data)
        salt_bytes_unicode = bytes.fromhex(commitment_unicode.salt)
        assert verify_reveal(commitment_unicode, unicode_data, salt_bytes_unicode)
        
        # Test 4: None and null values
        null_data = {
            'null_value': None,
            'empty_string': '',
            'zero': 0,
            'false': False,
            'empty_list': [],
            'empty_dict': {}
        }
        
        commitment_null = compute_commitment(null_data)
        salt_bytes_null = bytes.fromhex(commitment_null.salt)
        assert verify_reveal(commitment_null, null_data, salt_bytes_null)
        
        # Test 5: Numeric precision
        numeric_data = {
            'float': 3.141592653589793,
            'large_int': 12345678901234567890,
            'scientific': 1.23e-10,
            'negative': -987.654321
        }
        
        commitment_numeric = compute_commitment(numeric_data)
        salt_bytes_numeric = bytes.fromhex(commitment_numeric.salt)
        assert verify_reveal(commitment_numeric, numeric_data, salt_bytes_numeric)
        
        # Test 6: Invalid salt sizes
        test_data = {'test': 'data'}
        commitment_test = compute_commitment(test_data)
        
        # Wrong salt lengths should fail
        short_salt = b'short'
        long_salt = b'this_salt_is_way_too_long_and_should_fail_validation'
        
        assert not verify_reveal(commitment_test, test_data, short_salt)
        assert not verify_reveal(commitment_test, test_data, long_salt)
        
        print("  ✓ Error handling and edge cases tests passed")
    
    def test_performance_characteristics(self):
        """Test performance characteristics of commit-reveal protocol."""
        print("Testing performance characteristics...")
        
        import time
        
        # Test commitment generation performance
        test_data = {
            'model_id': 'performance_test_model',
            'verification_data': list(range(1000)),
            'metadata': {'test': True}
        }
        
        # Benchmark commitment generation
        start_time = time.time()
        commitments = []
        for i in range(100):
            data_variant = test_data.copy()
            data_variant['iteration'] = i
            commitment = compute_commitment(data_variant)
            commitments.append(commitment)
        
        commit_time = time.time() - start_time
        print(f"  Commitment generation (100x): {commit_time:.3f}s ({commit_time/100*1000:.2f}ms each)")
        
        # Benchmark verification
        start_time = time.time()
        for i, commitment in enumerate(commitments):
            data_variant = test_data.copy()
            data_variant['iteration'] = i
            salt_bytes = bytes.fromhex(commitment.salt)
            result = verify_reveal(commitment, data_variant, salt_bytes)
            assert result  # All should verify
        
        verify_time = time.time() - start_time
        print(f"  Verification (100x): {verify_time:.3f}s ({verify_time/100*1000:.2f}ms each)")
        
        # Performance assertions
        assert commit_time < 5.0, "Commitment generation should be fast"
        assert verify_time < 5.0, "Verification should be fast"
        assert commit_time / 100 < 0.05, "Individual commitment should be under 50ms"
        assert verify_time / 100 < 0.05, "Individual verification should be under 50ms"
        
        print("  ✓ Performance characteristics tests passed")


def run_all_tests():
    """Run all commit-reveal protocol tests."""
    print("=" * 70)
    print("COMPREHENSIVE COMMIT-REVEAL PROTOCOL TEST SUITE")
    print("=" * 70)
    
    test_suite = TestCommitRevealProtocol()
    
    test_methods = [
        test_suite.test_commitment_generation,
        test_suite.test_reveal_protocol_correct_data,
        test_suite.test_reveal_protocol_incorrect_data,
        test_suite.test_audit_trail_integrity,
        test_suite.test_schema_validation,
        test_suite.test_legacy_compatibility,
        test_suite.test_serialization_and_canonicalization,
        test_suite.test_error_handling_and_edge_cases,
        test_suite.test_performance_characteristics
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_method()
            passed += 1
        except Exception as e:
            print(f"✗ {test_method.__name__} FAILED: {e}")
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
        print("🎉 ALL COMMIT-REVEAL PROTOCOL TESTS PASSED!")
        print("\nCommit-reveal protocol ready for production!")
        print("Features validated:")
        print("  ✓ Cryptographic commitment generation")
        print("  ✓ Secure reveal protocol with tamper detection")
        print("  ✓ Audit trail integrity verification")
        print("  ✓ Comprehensive schema validation")
        print("  ✓ Legacy HMAC compatibility")
        print("  ✓ Canonical serialization")
        print("  ✓ Error handling and edge cases")
        print("  ✓ Production-grade performance")
    else:
        print(f"❌ {failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)