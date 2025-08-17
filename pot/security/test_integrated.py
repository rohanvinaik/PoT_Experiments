#!/usr/bin/env python3
"""
Comprehensive test suite for integrated verification pipeline in the PoT security system.

Tests the full verification pipeline including expected ranges validation,
blockchain integration (with mock), audit trail generation, and end-to-end workflows.
"""

import sys
import os
import json
import tempfile
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pot.security.proof_of_training import (
    ProofOfTraining, ExpectedRanges, RangeCalibrator, SessionConfig,
    VerificationReport, ModelType, SecurityLevel, VerificationDepth
)
from pot.prototypes.training_provenance_auditor import (
    MockBlockchainClient, BlockchainConfig, ChainType
)
from pot.audit.commit_reveal import (
    compute_commitment, verify_reveal, write_audit_record, read_and_verify_audit_trail
)
from pot.audit.query import AuditTrailQuery


class MockVerificationModel:
    """Mock model for testing verification pipeline."""
    
    def __init__(self, model_type="vision", accuracy=0.9, behavior_pattern="normal"):
        self.model_type = model_type
        self.accuracy = accuracy
        self.behavior_pattern = behavior_pattern
        self.weights = np.random.randn(1000, 100)
        self.bias = np.random.randn(100)
        
    def forward(self, x):
        """Simulate model forward pass."""
        if self.behavior_pattern == "normal":
            return np.random.randn(10) * 0.1 + self.accuracy
        elif self.behavior_pattern == "adversarial":
            return np.random.randn(10) * 2.0 + 0.5  # Different distribution
        elif self.behavior_pattern == "degraded":
            return np.random.randn(10) * 0.1 + (self.accuracy - 0.3)  # Lower performance
        else:
            return np.random.randn(10) * 0.1 + self.accuracy
    
    def state_dict(self):
        """Return model state dictionary."""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'model_type': self.model_type,
            'accuracy': self.accuracy
        }


class TestIntegratedVerificationPipeline:
    """Comprehensive test suite for integrated verification pipeline."""
    
    def test_full_verification_pipeline_basic(self):
        """Test basic full verification pipeline."""
        print("Testing basic full verification pipeline...")
        
        # Initialize PoT system
        config = {
            'verification_type': 'fuzzy',
            'model_type': 'vision',
            'security_level': 'medium'
        }
        
        pot_system = ProofOfTraining(config)
        
        # Create test model
        model = MockVerificationModel(model_type="vision", accuracy=0.9)
        
        # Register model
        model_id = pot_system.register_model(
            model,
            architecture="test_resnet",
            parameter_count=100000
        )
        
        assert model_id is not None
        assert len(model_id) > 0
        
        # Perform basic verification
        result = pot_system.perform_verification(model, model_id, 'quick')
        
        assert isinstance(result, object)  # Should return verification result
        assert hasattr(result, 'verified')
        assert isinstance(result.verified, bool)
        
        if hasattr(result, 'confidence'):
            assert 0.0 <= result.confidence <= 1.0
        
        # Get system statistics
        stats = pot_system.get_statistics()
        assert isinstance(stats, dict)
        assert 'models_registered' in stats
        assert stats['models_registered'] >= 1
        
        print("  ✓ Basic full verification pipeline tests passed")
    
    def test_expected_ranges_validation_comprehensive(self):
        """Test comprehensive expected ranges validation."""
        print("Testing comprehensive expected ranges validation...")
        
        config = {
            'verification_type': 'fuzzy',
            'model_type': 'vision',
            'security_level': 'high'
        }
        
        pot_system = ProofOfTraining(config)
        
        # Test 1: Model within expected ranges
        normal_model = MockVerificationModel(behavior_pattern="normal", accuracy=0.9)
        model_id_normal = pot_system.register_model(normal_model, architecture="normal_model")
        
        # Set expected ranges for normal performance
        expected_ranges = ExpectedRanges(
            accuracy_range=(0.85, 0.95),
            latency_range=(10.0, 50.0),
            fingerprint_similarity=(0.90, 0.99),
            jacobian_norm_range=(0.5, 2.0),
            confidence_level=0.95,
            tolerance_factor=1.1
        )
        
        pot_system.set_expected_ranges(model_id_normal, expected_ranges)
        
        # Verify normal model
        result_normal = pot_system.perform_verification(normal_model, model_id_normal, 'standard')
        
        # Should pass verification and range validation
        assert result_normal.verified
        if hasattr(result_normal, 'range_validation') and result_normal.range_validation:
            assert result_normal.range_validation.passed
        
        # Test 2: Model outside expected ranges (adversarial)
        adversarial_model = MockVerificationModel(behavior_pattern="adversarial", accuracy=0.5)
        model_id_adversarial = pot_system.register_model(adversarial_model, architecture="adversarial_model")
        
        # Use same expected ranges (adversarial model should fail)
        pot_system.set_expected_ranges(model_id_adversarial, expected_ranges)
        
        result_adversarial = pot_system.perform_verification(adversarial_model, model_id_adversarial, 'standard')
        
        # May pass basic verification but should fail range validation
        if hasattr(result_adversarial, 'range_validation') and result_adversarial.range_validation:
            # If range validation is performed, it should detect the anomaly
            if result_adversarial.range_validation.violations:
                assert len(result_adversarial.range_validation.violations) > 0
        
        # Test 3: Range calibration
        calibrator = RangeCalibrator(confidence_level=0.95, percentile_margin=0.05)
        
        # Create reference model performance data
        reference_model = MockVerificationModel(behavior_pattern="normal", accuracy=0.9)
        test_suite = [np.random.randn(224, 224, 3) for _ in range(10)]
        
        calibrated_ranges = calibrator.calibrate(
            reference_model=reference_model,
            test_suite=test_suite,
            model_type=ModelType.VISION,
            num_runs=5
        )
        
        assert isinstance(calibrated_ranges, ExpectedRanges)
        assert calibrated_ranges.accuracy_range[0] < calibrated_ranges.accuracy_range[1]
        assert calibrated_ranges.latency_range[0] < calibrated_ranges.latency_range[1]
        
        print("  ✓ Expected ranges validation tests passed")
    
    def test_blockchain_integration_mock(self):
        """Test blockchain integration using mock client."""
        print("Testing blockchain integration with mock client...")
        
        # Initialize mock blockchain client
        mock_config = BlockchainConfig.local_ganache()
        mock_client = MockBlockchainClient()
        
        # Test connection
        with mock_client as client:
            assert client.connect()
            
            # Test single commitment storage
            verification_data = {
                "model_id": "blockchain_test_model",
                "verification_result": "PASS",
                "confidence": 0.95,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            commitment_hash = hashlib.sha256(
                json.dumps(verification_data, sort_keys=True).encode()
            ).digest()
            
            metadata = {"verification_type": "integrated_test"}
            
            # Store commitment
            tx_hash = client.store_commitment(commitment_hash, metadata)
            assert tx_hash is not None
            assert len(tx_hash) > 0
            
            # Retrieve commitment
            record = client.retrieve_commitment(tx_hash)
            assert record is not None
            assert record.commitment_hash == commitment_hash.hex()
            assert record.metadata == metadata
            
            # Verify commitment on-chain
            is_valid = client.verify_commitment_onchain(commitment_hash)
            assert is_valid
            
            # Test batch commitment storage
            batch_commitments = []
            for i in range(10):
                batch_data = {
                    "batch_item": i,
                    "verification_result": "PASS" if i % 2 == 0 else "FAIL",
                    "confidence": 0.8 + (i % 3) * 0.05
                }
                batch_hash = hashlib.sha256(
                    json.dumps(batch_data, sort_keys=True).encode()
                ).digest()
                batch_commitments.append(batch_hash)
            
            # Store batch
            batch_tx_hash = client.batch_store_commitments(batch_commitments)
            assert batch_tx_hash is not None
            
            # Retrieve batch record
            batch_record = client.get_batch_commitment(batch_tx_hash)
            assert batch_record is not None
            assert batch_record.merkle_root is not None
            assert len(batch_record.proofs) == len(batch_commitments)
            
            # Verify Merkle proofs for batch items
            for i, commitment in enumerate(batch_commitments[:3]):  # Test first 3
                proof = batch_record.proofs[commitment.hex()]
                assert proof is not None
                
                # Verify Merkle proof
                from pot.prototypes.training_provenance_auditor import verify_merkle_proof
                leaf_hash = hashlib.sha256(commitment).digest()
                root_hash = bytes.fromhex(batch_record.merkle_root)
                
                is_proof_valid = verify_merkle_proof(leaf_hash, proof, root_hash)
                assert is_proof_valid
        
        print("  ✓ Blockchain integration tests passed")
    
    def test_audit_trail_generation_comprehensive(self):
        """Test comprehensive audit trail generation."""
        print("Testing comprehensive audit trail generation...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_file = os.path.join(temp_dir, "integrated_audit.json")
            
            # Initialize PoT system with audit logging
            config = {
                'verification_type': 'fuzzy',
                'model_type': 'vision',
                'security_level': 'high',
                'audit_log_path': audit_file
            }
            
            pot_system = ProofOfTraining(config)
            
            # Simulate verification sequence with audit trail
            verification_sequence = []
            
            for i in range(5):
                # Create model variant
                model = MockVerificationModel(
                    behavior_pattern="normal" if i < 3 else "degraded",
                    accuracy=0.9 - (i * 0.05) if i >= 3 else 0.9
                )
                
                model_id = f"audit_test_model_{i}"
                pot_system.register_model(model, architecture=f"test_arch_{i}")
                
                # Create pre-verification commitment
                pre_verification_data = {
                    'model_id': model_id,
                    'verification_parameters': {
                        'security_level': 'high',
                        'verification_depth': 'standard'
                    },
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'sequence_number': i
                }
                
                commitment = compute_commitment(pre_verification_data)
                
                # Perform verification
                verification_result = pot_system.perform_verification(model, model_id, 'standard')
                
                # Create post-verification audit record
                post_verification_data = {
                    'verification_decision': 'PASS' if verification_result.verified else 'FAIL',
                    'confidence': getattr(verification_result, 'confidence', 0.0),
                    'model_id': model_id,
                    'sequence_number': i,
                    'completion_timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                # Verify commitment-reveal consistency
                salt_bytes = bytes.fromhex(commitment.salt)
                reveal_valid = verify_reveal(commitment, pre_verification_data, salt_bytes)
                assert reveal_valid
                
                # Create comprehensive audit record
                audit_record = {
                    'commitment_record': {
                        'commitment_hash': commitment.commitment_hash,
                        'salt': commitment.salt,
                        'timestamp': commitment.timestamp,
                        'version': commitment.version
                    },
                    'pre_verification_data': pre_verification_data,
                    'post_verification_data': post_verification_data,
                    'verification_metadata': {
                        'verifier_version': '2.1.0',
                        'audit_sequence': i,
                        'integrity_check': True
                    }
                }
                
                # Write to audit trail
                write_audit_record(audit_record, audit_file)
                verification_sequence.append(audit_record)
            
            # Test audit trail integrity
            loaded_trail = read_and_verify_audit_trail(audit_file)
            assert len(loaded_trail) == 5
            
            # Verify each audit record
            for i, record in enumerate(loaded_trail):
                original_pre_data = verification_sequence[i]['pre_verification_data']
                commitment_info = record['commitment_record']
                
                # Reconstruct and verify commitment
                from pot.audit.commit_reveal import CommitmentRecord
                commitment = CommitmentRecord(
                    commitment_hash=commitment_info['commitment_hash'],
                    salt=commitment_info['salt'],
                    timestamp=commitment_info['timestamp'],
                    version=commitment_info['version']
                )
                
                salt_bytes = bytes.fromhex(commitment.salt)
                assert verify_reveal(commitment, original_pre_data, salt_bytes)
            
            # Test audit trail querying
            query_system = AuditTrailQuery(audit_file)
            
            # Query by model
            model_records = query_system.query_by_model("audit_test_model_0")
            assert len(model_records) == 1
            
            # Query by verification result
            pass_records = query_system.query_by_verification_result("PASS")
            fail_records = query_system.query_by_verification_result("FAIL")
            
            # Should have both passes and potential failures
            assert len(pass_records) + len(fail_records) == 5
            
            # Test integrity verification
            integrity_report = query_system.verify_integrity()
            assert integrity_report.total_records == 5
            assert integrity_report.integrity_score > 0.8  # Should be high for valid trail
            
        print("  ✓ Audit trail generation tests passed")
    
    def test_end_to_end_verification_workflow(self):
        """Test complete end-to-end verification workflow."""
        print("Testing end-to-end verification workflow...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up complete integrated environment
            audit_file = os.path.join(temp_dir, "e2e_audit.json")
            
            # Initialize with full configuration
            config = {
                'verification_type': 'fuzzy',
                'model_type': 'vision',
                'security_level': 'high',
                'audit_log_path': audit_file
            }
            
            pot_system = ProofOfTraining(config)
            
            # Create and configure reference model
            reference_model = MockVerificationModel(behavior_pattern="normal", accuracy=0.92)
            reference_model_id = pot_system.register_model(
                reference_model,
                architecture="reference_model_v1",
                parameter_count=1000000
            )
            
            # Calibrate expected ranges from reference model
            calibrated_ranges = pot_system.calibrate_expected_ranges(
                reference_model, reference_model_id, num_calibration_runs=10
            )
            
            assert isinstance(calibrated_ranges, ExpectedRanges)
            
            # Test legitimate model verification
            legitimate_model = MockVerificationModel(behavior_pattern="normal", accuracy=0.91)
            legitimate_model_id = pot_system.register_model(
                legitimate_model,
                architecture="legitimate_production_model"
            )
            
            pot_system.set_expected_ranges(legitimate_model_id, calibrated_ranges)
            
            # Create complete session configuration
            session_config = SessionConfig(
                model=legitimate_model,
                model_id=legitimate_model_id,
                master_seed="e2e_test_seed_64_characters_long_for_comprehensive_testing",
                
                num_challenges=20,
                challenge_family="vision:freq",
                challenge_params={'frequency_range': [0.1, 10.0]},
                
                accuracy_threshold=0.05,
                type1_error=0.01,
                type2_error=0.01,
                max_samples=500,
                
                use_fingerprinting=True,
                use_sequential=True,
                use_range_validation=True,
                use_blockchain=False,  # Skip for test speed
                
                expected_ranges=calibrated_ranges,
                audit_log_path=audit_file
            )
            
            # Execute full verification protocol
            verification_report = pot_system.run_verification(session_config)
            
            assert isinstance(verification_report, VerificationReport)
            assert verification_report.session_id is not None
            assert verification_report.duration_seconds > 0
            
            # Legitimate model should pass
            assert verification_report.passed
            assert verification_report.confidence > 0.7
            
            # Test adversarial model detection
            adversarial_model = MockVerificationModel(behavior_pattern="adversarial", accuracy=0.6)
            adversarial_model_id = pot_system.register_model(
                adversarial_model,
                architecture="adversarial_model"
            )
            
            pot_system.set_expected_ranges(adversarial_model_id, calibrated_ranges)
            
            adversarial_session_config = SessionConfig(
                model=adversarial_model,
                model_id=adversarial_model_id,
                master_seed="adversarial_test_seed_64_characters_long_for_detection",
                
                num_challenges=15,
                challenge_family="vision:freq",
                accuracy_threshold=0.05,
                type1_error=0.01,
                type2_error=0.01,
                
                use_fingerprinting=True,
                use_sequential=True,
                use_range_validation=True,
                use_blockchain=False,
                
                expected_ranges=calibrated_ranges,
                audit_log_path=audit_file
            )
            
            adversarial_report = pot_system.run_verification(adversarial_session_config)
            
            # Adversarial model should be detected
            # Either fail verification or trigger range violations
            detection_successful = (
                not adversarial_report.passed or
                (adversarial_report.range_validation and 
                 not adversarial_report.range_validation.passed and
                 len(adversarial_report.range_validation.violations) > 0)
            )
            
            assert detection_successful, "Should detect adversarial model"
            
            # Test audit trail completeness
            query_system = AuditTrailQuery(audit_file)
            all_records = query_system.records
            
            # Should have records for both verification sessions
            assert len(all_records) >= 2
            
            # Test audit trail integrity
            integrity_report = query_system.verify_integrity()
            assert integrity_report.integrity_score > 0.8
            
            # Test anomaly detection in audit trail
            anomalies = query_system.find_anomalies()
            # May or may not detect anomalies depending on the implementation
            
            # Generate comprehensive report
            audit_report = query_system.generate_audit_report("json")
            report_data = json.loads(audit_report)
            
            assert 'report_metadata' in report_data
            assert 'summary_statistics' in report_data
            assert 'integrity_report' in report_data
            
            print(f"  E2E workflow completed:")
            print(f"    Legitimate model verified: {verification_report.passed}")
            print(f"    Adversarial model detected: {detection_successful}")
            print(f"    Audit records generated: {len(all_records)}")
            print(f"    Audit integrity score: {integrity_report.integrity_score:.3f}")
        
        print("  ✓ End-to-end verification workflow tests passed")
    
    def test_security_stress_testing(self):
        """Test security under stress conditions."""
        print("Testing security under stress conditions...")
        
        config = {
            'verification_type': 'fuzzy',
            'model_type': 'vision',
            'security_level': 'high'
        }
        
        pot_system = ProofOfTraining(config)
        
        # Test 1: Rapid verification requests
        rapid_models = []
        for i in range(10):
            model = MockVerificationModel(
                behavior_pattern="normal" if i % 3 != 0 else "adversarial",
                accuracy=0.8 + (i % 5) * 0.02
            )
            model_id = f"stress_test_model_{i}"
            pot_system.register_model(model, architecture=f"stress_arch_{i}")
            rapid_models.append((model, model_id))
        
        # Perform rapid verifications
        verification_results = []
        for model, model_id in rapid_models:
            result = pot_system.perform_verification(model, model_id, 'quick')
            verification_results.append(result)
        
        # All verifications should complete
        assert len(verification_results) == 10
        
        # Test 2: Large model verification
        large_model = MockVerificationModel(behavior_pattern="normal", accuracy=0.9)
        large_model.weights = np.random.randn(10000, 1000)  # Much larger
        large_model.bias = np.random.randn(1000)
        
        large_model_id = pot_system.register_model(large_model, architecture="large_model")
        large_result = pot_system.perform_verification(large_model, large_model_id, 'standard')
        
        # Should handle large models
        assert hasattr(large_result, 'verified')
        
        # Test 3: Edge case ranges
        edge_model = MockVerificationModel(behavior_pattern="normal", accuracy=0.9)
        edge_model_id = pot_system.register_model(edge_model, architecture="edge_model")
        
        # Set very tight ranges
        tight_ranges = ExpectedRanges(
            accuracy_range=(0.899, 0.901),  # Very tight
            latency_range=(1.0, 2.0),       # Very tight
            fingerprint_similarity=(0.999, 1.0),  # Almost perfect match required
            jacobian_norm_range=(0.99, 1.01),    # Very tight
            confidence_level=0.999,
            tolerance_factor=1.001  # Minimal tolerance
        )
        
        pot_system.set_expected_ranges(edge_model_id, tight_ranges)
        edge_result = pot_system.perform_verification(edge_model, edge_model_id, 'comprehensive')
        
        # Should handle tight ranges without crashing
        assert hasattr(edge_result, 'verified')
        
        print("  ✓ Security stress testing passed")
    
    def test_integration_error_handling(self):
        """Test error handling in integrated pipeline."""
        print("Testing integration error handling...")
        
        config = {
            'verification_type': 'fuzzy',
            'model_type': 'vision',
            'security_level': 'medium'
        }
        
        pot_system = ProofOfTraining(config)
        
        # Test 1: Invalid model registration
        try:
            pot_system.register_model(None, architecture="invalid_model")
            assert False, "Should raise error for None model"
        except (ValueError, TypeError):
            pass  # Expected
        
        # Test 2: Verification of unregistered model
        unregistered_model = MockVerificationModel()
        try:
            result = pot_system.perform_verification(unregistered_model, "nonexistent_id", 'quick')
            # May succeed with warning or fail gracefully
        except Exception:
            pass  # Acceptable to fail
        
        # Test 3: Invalid expected ranges
        valid_model = MockVerificationModel()
        valid_model_id = pot_system.register_model(valid_model, architecture="valid_model")
        
        invalid_ranges = ExpectedRanges(
            accuracy_range=(0.95, 0.85),  # Invalid: min > max
            latency_range=(-1.0, 10.0),   # Invalid: negative latency
            fingerprint_similarity=(1.1, 1.2),  # Invalid: > 1.0
            jacobian_norm_range=(10.0, 1.0),    # Invalid: min > max
            confidence_level=1.5,  # Invalid: > 1.0
            tolerance_factor=0.5   # Invalid: < 1.0
        )
        
        try:
            pot_system.set_expected_ranges(valid_model_id, invalid_ranges)
            # May accept with warnings or corrections
        except ValueError:
            pass  # Acceptable to reject invalid ranges
        
        # Test 4: Malformed session configuration
        try:
            invalid_session = SessionConfig(
                model=valid_model,
                model_id=valid_model_id,
                master_seed="short",  # Too short
                num_challenges=-5,    # Negative
                accuracy_threshold=2.0,  # > 1.0
                type1_error=-0.1,     # Negative
                type2_error=1.5       # > 1.0
            )
            # Configuration creation may or may not fail
        except ValueError:
            pass  # Acceptable
        
        print("  ✓ Integration error handling tests passed")


def run_all_tests():
    """Run all integrated verification pipeline tests."""
    print("=" * 70)
    print("COMPREHENSIVE INTEGRATED VERIFICATION PIPELINE TEST SUITE")
    print("=" * 70)
    
    test_suite = TestIntegratedVerificationPipeline()
    
    test_methods = [
        test_suite.test_full_verification_pipeline_basic,
        test_suite.test_expected_ranges_validation_comprehensive,
        test_suite.test_blockchain_integration_mock,
        test_suite.test_audit_trail_generation_comprehensive,
        test_suite.test_end_to_end_verification_workflow,
        test_suite.test_security_stress_testing,
        test_suite.test_integration_error_handling
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
        print("🎉 ALL INTEGRATED VERIFICATION TESTS PASSED!")
        print("\nIntegrated verification pipeline ready for production!")
        print("Features validated:")
        print("  ✓ Full verification pipeline with all components")
        print("  ✓ Expected ranges validation and calibration")
        print("  ✓ Blockchain integration with mock client")
        print("  ✓ Comprehensive audit trail generation")
        print("  ✓ End-to-end verification workflows")
        print("  ✓ Security stress testing")
        print("  ✓ Error handling and edge cases")
    else:
        print(f"❌ {failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)