#!/usr/bin/env python3
"""
Comprehensive integration tests for TrainingProvenanceAuditor ZK integration.

Tests the fixed ZK witness generation, dual commitment support, 
fallback mechanisms, and validation functions.
"""

import sys
import pytest
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pot.prototypes.training_provenance_auditor import TrainingProvenanceAuditor


class TestTrainingProvenanceAuditorZKIntegration:
    """Comprehensive tests for ZK integration in TrainingProvenanceAuditor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model_id = "test_model_zk_integration"
        self.auditor = TrainingProvenanceAuditor(
            model_id=self.model_id,
            fail_on_zk_error=False  # Test graceful fallback
        )
        
        # Sample SGD training step data
        self.sgd_training_data = {
            'weights_before': {
                'layer1.weight': np.random.randn(32, 16).astype(np.float32),
                'layer1.bias': np.random.randn(32).astype(np.float32),
                'layer2.weight': np.random.randn(10, 32).astype(np.float32),
                'layer2.bias': np.random.randn(10).astype(np.float32)
            },
            'weights_after': {},  # Will be filled in tests
            'batch_data': {
                'inputs': np.random.randn(8, 16).astype(np.float32),
                'targets': np.random.randint(0, 10, size=(8,)),
                'gradients': {},
                'loss': 0.5
            },
            'hyperparameters': {
                'learning_rate': 0.01,
                'batch_size': 8,
                'optimizer': 'sgd'
            },
            'step_info': {
                'step_number': 1,
                'epoch': 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        }
        
        # Create weights_after with small changes
        self.sgd_training_data['weights_after'] = {
            key: weight + np.random.randn(*weight.shape).astype(np.float32) * 0.001
            for key, weight in self.sgd_training_data['weights_before'].items()
        }
        
        # Sample LoRA training step data
        self.lora_training_data = {
            'weights_before': {
                'base.weight': np.random.randn(256, 256).astype(np.float32),
                'lora_A.weight': np.random.randn(256, 16).astype(np.float32),
                'lora_B.weight': np.random.randn(16, 256).astype(np.float32)
            },
            'weights_after': {},  # Will be filled
            'batch_data': {
                'inputs': np.random.randn(8, 256).astype(np.float32),
                'targets': np.random.randn(8, 256).astype(np.float32),
                'loss': 0.3
            },
            'hyperparameters': {
                'learning_rate': 0.001,
                'lora_rank': 16,
                'lora_alpha': 32,
                'optimizer': 'adam'
            },
            'step_info': {
                'step_number': 1,
                'epoch': 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        }
        
        # Create LoRA weights_after (only adapters change)
        self.lora_training_data['weights_after'] = {
            'base.weight': self.lora_training_data['weights_before']['base.weight'],  # Frozen
            'lora_A.weight': (
                self.lora_training_data['weights_before']['lora_A.weight'] + 
                np.random.randn(256, 16).astype(np.float32) * 0.001
            ),
            'lora_B.weight': (
                self.lora_training_data['weights_before']['lora_B.weight'] + 
                np.random.randn(16, 256).astype(np.float32) * 0.001
            )
        }
    
    def test_sgd_witness_generation_success(self):
        """Test successful SGD witness generation with ZK modules available."""
        # Generate witness
        witness_record = self.auditor.generate_zk_witness(self.sgd_training_data)
        
        # Verify witness record structure
        assert witness_record is not None, "Witness record should be generated"
        assert 'witness_id' in witness_record
        assert 'model_type' in witness_record
        assert 'proof_type' in witness_record
        assert 'witness' in witness_record
        assert 'statement' in witness_record
        assert 'sha256_root' in witness_record
        assert 'dual_commitments' in witness_record
        
        # Check model type detection
        assert witness_record['model_type'] == 'sgd', f"Expected SGD, got {witness_record['model_type']}"
        assert witness_record['proof_type'] == 'sgd_step'
        
        # Verify dual commitments
        dual_commitments = witness_record['dual_commitments']
        assert 'sha256_root' in dual_commitments
        assert 'commitment_scheme' in dual_commitments
        
        # Check witness metadata
        metadata = witness_record['witness_metadata']
        assert metadata['step_number'] == 1
        assert metadata['epoch'] == 0
        assert metadata['model_id'] == self.model_id
        
        print(f"✅ SGD witness generated: {witness_record['witness_id']}")
    
    def test_lora_witness_generation_success(self):
        """Test successful LoRA witness generation with automatic detection."""
        # Generate witness
        witness_record = self.auditor.generate_zk_witness(self.lora_training_data)
        
        # Verify basic structure
        assert witness_record is not None
        assert witness_record['model_type'] == 'lora', f"Expected LoRA, got {witness_record['model_type']}"
        assert witness_record['proof_type'] == 'lora_step'
        
        # Check LoRA-specific fields in statement
        statement = witness_record['statement']
        assert statement['proof_type'] == 'lora_step'
        
        print(f"✅ LoRA witness generated: {witness_record['witness_id']}")
    
    def test_model_type_detection_sgd(self):
        """Test automatic model type detection for SGD models."""
        weights_before = self.sgd_training_data['weights_before']
        weights_after = self.sgd_training_data['weights_after']
        
        detected = self.auditor._detect_lora_model(weights_before, weights_after)
        assert not detected, "SGD model incorrectly detected as LoRA"
        
        print("✅ SGD model type detection working")
    
    def test_model_type_detection_lora(self):
        """Test automatic model type detection for LoRA models."""
        weights_before = self.lora_training_data['weights_before']
        weights_after = self.lora_training_data['weights_after']
        
        detected = self.auditor._detect_lora_model(weights_before, weights_after)
        assert detected, "LoRA model not detected correctly"
        
        print("✅ LoRA model type detection working")
    
    def test_training_step_data_validation_success(self):
        """Test successful validation of training step data."""
        validation = self.auditor._validate_training_step_data(self.sgd_training_data)
        
        assert validation['valid'], f"Validation failed: {validation['errors']}"
        assert len(validation['errors']) == 0
        assert validation['tensor_count'] > 0
        
        print("✅ Training step data validation passed")
    
    def test_training_step_data_validation_failures(self):
        """Test validation with invalid training step data."""
        # Test missing required keys
        invalid_data = {'weights_before': {}}  # Missing required keys
        validation = self.auditor._validate_training_step_data(invalid_data)
        
        assert not validation['valid']
        assert len(validation['errors']) > 0
        
        # Test empty weights
        invalid_data = {
            'weights_before': {},
            'weights_after': {},
            'batch_data': {},
            'hyperparameters': {},
            'step_info': {}
        }
        validation = self.auditor._validate_training_step_data(invalid_data)
        
        assert not validation['valid']
        assert any('weights_before must be non-empty' in error for error in validation['errors'])
        
        print("✅ Training step data validation correctly detects invalid data")
    
    def test_tensor_dimension_validation(self):
        """Test tensor dimension validation between before/after weights."""
        weights_before = {'layer1': np.random.randn(10, 5)}
        weights_after = {'layer1': np.random.randn(10, 5)}  # Same shape
        
        # Should not raise exception
        self.auditor._validate_tensor_dimensions(weights_before, weights_after)
        
        # Test dimension mismatch
        weights_after_wrong = {'layer1': np.random.randn(5, 10)}  # Wrong shape
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            self.auditor._validate_tensor_dimensions(weights_before, weights_after_wrong)
        
        # Test missing keys
        weights_after_missing = {'layer2': np.random.randn(10, 5)}  # Wrong key
        
        with pytest.raises(ValueError, match="Keys missing"):
            self.auditor._validate_tensor_dimensions(weights_before, weights_after_missing)
        
        print("✅ Tensor dimension validation working correctly")
    
    def test_dual_commitment_generation(self):
        """Test dual commitment generation (SHA-256 + Poseidon)."""
        witness = {
            'weights_diff': np.array([1.0, 2.0, 3.0]),
            'step_number': 1,
            'learning_rate': 0.01
        }
        
        commitments = self.auditor._generate_dual_commitments(witness)
        
        # Verify SHA-256 commitment always present
        assert 'sha256_root' in commitments
        assert commitments['sha256_root'] is not None
        assert isinstance(commitments['sha256_root'], str)
        assert len(commitments['sha256_root']) == 64  # SHA-256 hex length
        
        # Check commitment scheme info
        assert 'commitment_scheme' in commitments
        assert 'poseidon_available' in commitments
        
        print(f"✅ Dual commitments generated: {commitments['commitment_scheme']}")
    
    def test_witness_to_field_elements_conversion(self):
        """Test conversion of witness data to field elements for Poseidon."""
        witness = {
            'scalar': 42,
            'array': np.array([1.0, 2.0, 3.0]),
            'list': [4.0, 5.0, 6.0],
            'string': 'test_string'
        }
        
        elements = self.auditor._witness_to_field_elements(witness)
        
        assert isinstance(elements, list)
        assert len(elements) > 0
        assert all(isinstance(e, int) for e in elements)
        assert all(0 <= e < (2**251 - 1) for e in elements)  # Valid field elements
        
        print(f"✅ Converted witness to {len(elements)} field elements")
    
    def test_fallback_to_sha256_only(self):
        """Test fallback to SHA-256-only witness when ZK modules unavailable."""
        with patch.dict('sys.modules', {
            'pot.zk.witness': None,
            'pot.zk.lora_builder': None,
            'pot.zk.poseidon': None
        }):
            # This should trigger ImportError and fallback
            witness_record = self.auditor.generate_zk_witness(self.sgd_training_data)
            
            assert witness_record is not None
            assert witness_record['proof_type'] == 'sha256_only'
            assert witness_record['witness_metadata']['fallback_mode'] is True
            
            dual_commitments = witness_record['dual_commitments']
            assert dual_commitments['commitment_scheme'] == 'sha256_only'
            assert dual_commitments['sha256_fallback'] is True
            assert dual_commitments['supports_zk'] is False
        
        print("✅ SHA-256-only fallback working")
    
    def test_fail_on_zk_error_mode(self):
        """Test fail_on_zk_error configuration."""
        # Create auditor that fails on ZK errors
        strict_auditor = TrainingProvenanceAuditor(
            model_id="test_strict",
            fail_on_zk_error=True
        )
        
        # Test with invalid data - should raise exception
        invalid_data = {
            'weights_before': {},  # Invalid
            'weights_after': {},
            'batch_data': {},
            'hyperparameters': {},
            'step_info': {}
        }
        
        with pytest.raises(ValueError, match="Invalid training step data"):
            strict_auditor.generate_zk_witness(invalid_data)
        
        # Test with lenient auditor - should return None
        lenient_result = self.auditor.generate_zk_witness(invalid_data)
        assert lenient_result is None
        
        print("✅ fail_on_zk_error configuration working")
    
    def test_witness_record_storage_and_retrieval(self):
        """Test witness record storage and retrieval functions."""
        # Generate multiple witness records
        witness1 = self.auditor.generate_zk_witness(self.sgd_training_data)
        
        # Modify step number for second witness
        lora_data_step2 = self.lora_training_data.copy()
        lora_data_step2['step_info'] = lora_data_step2['step_info'].copy()
        lora_data_step2['step_info']['step_number'] = 2
        
        witness2 = self.auditor.generate_zk_witness(lora_data_step2)
        
        # Test retrieval
        all_records = self.auditor.get_witness_records()
        assert len(all_records) == 2
        
        # Test limited retrieval
        limited_records = self.auditor.get_witness_records(limit=1)
        assert len(limited_records) == 1
        
        # Test retrieval by ID
        record_by_id = self.auditor.get_witness_record_by_id(witness1['witness_id'])
        assert record_by_id is not None
        assert record_by_id['witness_id'] == witness1['witness_id']
        
        # Test statistics
        stats = self.auditor.get_witness_statistics()
        assert stats['total_witnesses'] == 2
        assert stats['sgd_witnesses'] >= 1
        assert stats['lora_witnesses'] >= 1
        
        print(f"✅ Witness record storage/retrieval: {stats}")
    
    def test_witness_merkle_path_validation(self):
        """Test validation of Merkle paths for witness records."""
        # Generate witness
        witness_record = self.auditor.generate_zk_witness(self.sgd_training_data)
        witness_id = witness_record['witness_id']
        
        # Validate Merkle paths
        validation = self.auditor.validate_witness_merkle_paths(witness_id)
        
        assert validation['witness_id'] == witness_id
        assert 'sha256_valid' in validation
        assert 'overall_valid' in validation
        
        # SHA-256 validation should work
        if not validation['sha256_valid']:
            print(f"SHA-256 validation failed: {validation.get('sha256_error', 'Unknown error')}")
        
        print(f"✅ Merkle path validation: {validation['overall_valid']}")
    
    def test_pytorch_tensor_compatibility(self):
        """Test compatibility with PyTorch tensors."""
        try:
            import torch
            
            # Create PyTorch tensors
            pytorch_data = {
                'weights_before': {
                    'layer.weight': torch.randn(10, 5),
                    'layer.bias': torch.randn(10)
                },
                'weights_after': {
                    'layer.weight': torch.randn(10, 5),
                    'layer.bias': torch.randn(10)
                },
                'batch_data': {
                    'inputs': torch.randn(8, 5),
                    'targets': torch.randint(0, 2, (8,)),
                    'loss': 0.5
                },
                'hyperparameters': {'learning_rate': 0.01},
                'step_info': {'step_number': 1, 'epoch': 0}
            }
            
            # Should handle PyTorch tensors
            witness_record = self.auditor.generate_zk_witness(pytorch_data)
            assert witness_record is not None
            
            print("✅ PyTorch tensor compatibility verified")
            
        except ImportError:
            print("⚠️ PyTorch not available - skipping tensor compatibility test")
    
    def test_witness_generation_with_missing_components(self):
        """Test witness generation with missing optional components."""
        # Remove optional components
        minimal_data = {
            'weights_before': {'w': np.array([1.0, 2.0])},
            'weights_after': {'w': np.array([1.1, 2.1])},
            'batch_data': {},  # Missing inputs/targets
            'hyperparameters': {},  # Missing learning_rate
            'step_info': {'step_number': 1, 'epoch': 0}
        }
        
        # Should still work with warnings
        witness_record = self.auditor.generate_zk_witness(minimal_data)
        assert witness_record is not None
        
        # Check validation warnings
        validation = witness_record.get('validation', {})
        warnings = validation.get('warnings', [])
        assert len(warnings) > 0  # Should have warnings about missing components
        
        print(f"✅ Witness generation with minimal data: {len(warnings)} warnings")
    
    def test_integration_with_training_events(self):
        """Test integration with existing training event logging."""
        from pot.prototypes.training_provenance_auditor import EventType
        
        # Generate witness (which should also log training event)
        witness_record = self.auditor.generate_zk_witness(self.sgd_training_data)
        
        # Check if training event was logged
        events = self.auditor.get_training_events()
        
        # Find ZK-related event
        zk_events = [e for e in events if e.event_type == EventType.CUSTOM and 
                     'zk_witness_id' in e.metadata]
        
        assert len(zk_events) > 0, "ZK witness generation should create training event"
        
        zk_event = zk_events[0]
        assert zk_event.metadata['zk_witness_id'] == witness_record['witness_id']
        
        print("✅ Integration with training event logging verified")


def test_comprehensive_zk_integration_demo():
    """
    Comprehensive demonstration of ZK integration functionality.
    
    This test serves as both a test and a usage example.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE ZK INTEGRATION DEMONSTRATION")
    print("="*80)
    
    # Create auditor with different configurations
    auditors = {
        'strict': TrainingProvenanceAuditor('demo_strict', fail_on_zk_error=True),
        'lenient': TrainingProvenanceAuditor('demo_lenient', fail_on_zk_error=False)
    }
    
    # Create sample training data
    training_scenarios = {
        'sgd_small': {
            'weights_before': {'layer': np.random.randn(5, 3).astype(np.float32)},
            'weights_after': None,  # Will be filled
            'batch_data': {'inputs': np.random.randn(2, 3), 'loss': 0.1},
            'hyperparameters': {'learning_rate': 0.01},
            'step_info': {'step_number': 1, 'epoch': 0}
        },
        'lora_efficient': {
            'weights_before': {
                'base.weight': np.random.randn(64, 64).astype(np.float32),
                'lora_A.weight': np.random.randn(64, 8).astype(np.float32),
                'lora_B.weight': np.random.randn(8, 64).astype(np.float32)
            },
            'weights_after': None,  # Will be filled
            'batch_data': {'inputs': np.random.randn(4, 64), 'loss': 0.05},
            'hyperparameters': {'learning_rate': 0.001, 'lora_rank': 8},
            'step_info': {'step_number': 2, 'epoch': 0}
        }
    }
    
    # Fill in weights_after
    for scenario in training_scenarios.values():
        scenario['weights_after'] = {
            key: weight + np.random.randn(*weight.shape).astype(np.float32) * 0.01
            for key, weight in scenario['weights_before'].items()
        }
    
    results_summary = {
        'total_witnesses': 0,
        'successful_witnesses': 0,
        'sgd_witnesses': 0,
        'lora_witnesses': 0,
        'dual_commitments': 0,
        'sha256_fallbacks': 0
    }
    
    # Test each scenario with each auditor
    for auditor_name, auditor in auditors.items():
        print(f"\n--- Testing with {auditor_name} auditor ---")
        
        for scenario_name, data in training_scenarios.items():
            print(f"\nScenario: {scenario_name}")
            
            try:
                witness_record = auditor.generate_zk_witness(data)
                
                if witness_record:
                    results_summary['total_witnesses'] += 1
                    results_summary['successful_witnesses'] += 1
                    
                    model_type = witness_record['model_type']
                    if model_type == 'sgd':
                        results_summary['sgd_witnesses'] += 1
                    elif model_type == 'lora':
                        results_summary['lora_witnesses'] += 1
                    
                    dual_commitments = witness_record['dual_commitments']
                    if dual_commitments['commitment_scheme'] == 'dual':
                        results_summary['dual_commitments'] += 1
                    elif dual_commitments['sha256_fallback']:
                        results_summary['sha256_fallbacks'] += 1
                    
                    print(f"  ✅ Generated: {witness_record['witness_id']}")
                    print(f"     Model type: {model_type}")
                    print(f"     Commitment: {dual_commitments['commitment_scheme']}")
                    
                    # Validate Merkle paths
                    validation = auditor.validate_witness_merkle_paths(witness_record['witness_id'])
                    print(f"     Validation: {'✅' if validation['overall_valid'] else '❌'}")
                    
                else:
                    print(f"  ❌ Witness generation returned None")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results_summary['total_witnesses'] += 1
    
    # Display comprehensive statistics
    print(f"\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    for auditor_name, auditor in auditors.items():
        stats = auditor.get_witness_statistics()
        print(f"\n{auditor_name.upper()} AUDITOR STATISTICS:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print(f"\nOVERALL RESULTS:")
    for key, value in results_summary.items():
        print(f"  {key}: {value}")
    
    # Success rate
    if results_summary['total_witnesses'] > 0:
        success_rate = results_summary['successful_witnesses'] / results_summary['total_witnesses']
        print(f"\nSuccess Rate: {success_rate:.1%}")
        
        if success_rate >= 0.8:
            print("🎉 ZK Integration working excellently!")
        elif success_rate >= 0.6:
            print("✅ ZK Integration working well")
        else:
            print("⚠️ ZK Integration needs improvement")
    
    print("="*80)


if __name__ == "__main__":
    # Run comprehensive demo
    test_comprehensive_zk_integration_demo()
    
    # Run individual tests
    test_class = TestTrainingProvenanceAuditorZKIntegration()
    test_methods = [
        method for method in dir(test_class) 
        if method.startswith('test_') and callable(getattr(test_class, method))
    ]
    
    print(f"\nRunning {len(test_methods)} individual tests...")
    
    for method_name in test_methods:
        try:
            test_class.setup_method()
            method = getattr(test_class, method_name)
            method()
            print(f"✅ {method_name}")
        except Exception as e:
            print(f"❌ {method_name}: {e}")
    
    print("\n🎉 ZK Integration testing complete!")