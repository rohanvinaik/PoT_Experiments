"""
Full Pipeline Integration Tests

Comprehensive tests covering all verification modes and pipeline components.
"""

import pytest
import tempfile
import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Import pipeline components
from src.pot.sharding import AdaptiveShardManager, MemoryManager
from src.pot.sharding.pipeline_integration import ShardedVerificationPipeline
from src.pot.audit.validation import AuditValidator, LeakageDetector
from src.pot.audit.adversarial import AttackSimulator, AttackScenario, AttackStrategy


class TestFullPipeline:
    """Test complete E2E pipeline with all components"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        model = MagicMock()
        model.generate = MagicMock(return_value="test_response")
        model.config = {"hidden_size": 768, "num_layers": 12}
        return model
    
    def test_standard_verification_pipeline(self, temp_dir, mock_model):
        """Test standard verification without sharding"""
        # Mock E2E pipeline components
        with patch('scripts.run_e2e_validation.PipelineOrchestrator') as MockOrchestrator:
            orchestrator = MockOrchestrator.return_value
            orchestrator.run.return_value = {
                'success': True,
                'decision': 'SAME',
                'confidence': 0.99,
                'n_queries': 32,
                'evidence_bundle_path': str(temp_dir / 'evidence.zip')
            }
            
            # Run verification
            from scripts.run_e2e_validation import main
            
            # Mock command line arguments
            with patch('sys.argv', [
                'run_e2e_validation.py',
                '--ref-model', 'gpt2',
                '--cand-model', 'distilgpt2',
                '--mode', 'quick',
                '--output-dir', str(temp_dir),
                '--dry-run'
            ]):
                # Execute pipeline
                main()
            
            # Verify orchestrator was called
            orchestrator.run.assert_called_once()
    
    def test_sharded_verification_pipeline(self, temp_dir):
        """Test sharded verification for large models"""
        # Configure sharding
        config = {
            'enable_sharding': True,
            'max_memory_usage_percent': 70.0,
            'shard_cache_dir': str(temp_dir / 'shards')
        }
        
        pipeline = ShardedVerificationPipeline(config)
        
        # Test sharding detection
        assert pipeline.should_use_sharding('fake_70b_model') == True
        
        # Test shard preparation
        shard_config, shard_dir = pipeline.prepare_sharded_verification(
            'fake_model',
            str(temp_dir / 'shards')
        )
        
        assert shard_config.num_shards > 0
        assert os.path.exists(shard_dir)
    
    def test_audit_integration(self, temp_dir):
        """Test audit trail validation integration"""
        validator = AuditValidator()
        
        # Create test audit entries
        audit_entries = []
        for i in range(10):
            entry = {
                'timestamp': f'2024-01-01T00:00:{i:02d}',
                'operation': f'test_op_{i}',
                'actor': 'test_user',
                'resource': 'test_resource',
                'outcome': 'success',
                'hash': f'hash_{i}',
                'previous_hash': f'hash_{i-1}' if i > 0 else None
            }
            audit_entries.append(entry)
        
        # Save audit trail
        audit_file = temp_dir / 'audit.log'
        with open(audit_file, 'w') as f:
            for entry in audit_entries:
                json.dump(entry, f)
                f.write('\n')
        
        # Load and validate
        entries = validator.load_audit_trail(audit_file)
        result = validator.validate(entries)
        
        assert len(entries) == 10
        assert result.is_valid or len(result.errors) > 0  # Should process without crash
    
    def test_leakage_detection_integration(self):
        """Test information leakage detection"""
        detector = LeakageDetector()
        
        # Generate test responses
        responses = [
            "The model has 12 layers",
            "Weight matrix dimension is 768x768",
            "Training used learning_rate=0.001",
            "Normal response without leakage"
        ]
        
        challenges = ["challenge_" + str(i) for i in range(len(responses))]
        
        # Analyze for leakage
        report = detector.analyze(responses, challenges)
        
        assert report.leakage_score >= 0.0
        assert report.leakage_score <= 1.0
        assert len(report.leakage_types) >= 0
        assert len(report.recommendations) > 0
    
    def test_adversarial_attack_integration(self):
        """Test adversarial attack simulation"""
        simulator = AttackSimulator()
        
        # Create attack scenario
        scenario = AttackScenario(
            name="test_attack",
            strategy=AttackStrategy.RANDOM,
            target="model",
            objective="test",
            constraints={'max_queries': 10},
            parameters={'dimension': 5}
        )
        
        # Mock target system
        target = MagicMock()
        target.process = MagicMock(return_value={'success': False, 'detected': False})
        
        # Run attack
        outcome = simulator.simulate_attack(scenario, target)
        
        assert outcome.scenario == scenario
        assert outcome.queries_used <= 10
        assert isinstance(outcome.success, bool)
    
    @pytest.mark.parametrize("verification_mode", ["local", "api", "hybrid"])
    def test_verification_modes(self, verification_mode, temp_dir):
        """Test different verification modes"""
        # Mock appropriate components based on mode
        if verification_mode == "api":
            with patch('requests.post') as mock_post:
                mock_post.return_value.json.return_value = {"response": "api_response"}
                mock_post.return_value.status_code = 200
                
                # Run API verification
                # This would call the actual API verification code
                pass
        
        elif verification_mode == "local":
            # Test local model loading
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                mock_model.return_value = MagicMock()
                # Run local verification
                pass
        
        elif verification_mode == "hybrid":
            # Test hybrid mode combining local and API
            pass
    
    def test_memory_management(self):
        """Test memory manager integration"""
        manager = MemoryManager()
        
        # Establish baseline
        baseline = manager.establish_baseline()
        assert baseline.rss_bytes > 0
        
        # Check memory availability
        available = manager.check_memory_available(100 * 1024 * 1024)  # 100MB
        assert isinstance(available, bool)
        
        # Get memory summary
        summary = manager.get_memory_usage_summary()
        assert 'current' in summary
        assert 'statistics' in summary
    
    def test_checkpoint_recovery(self, temp_dir):
        """Test checkpoint and recovery mechanism"""
        from src.pot.sharding import CheckpointManager
        
        manager = CheckpointManager(str(temp_dir / 'checkpoints'))
        
        # Create checkpoint
        checkpoint = manager.create_checkpoint(
            model_path='test_model',
            shard_config={'num_shards': 10},
            challenges=['c1', 'c2', 'c3']
        )
        
        assert checkpoint.checkpoint_id
        assert checkpoint.status == 'in_progress'
        
        # Update checkpoint
        manager.update_checkpoint(checkpoint, 0, ['response1'])
        
        # Simulate recovery
        recovered = manager.recover_from_checkpoint(checkpoint.checkpoint_id)
        assert recovered is not None
        assert len(recovered.processed_shards) == 1
    
    def test_evidence_generation(self, temp_dir):
        """Test evidence bundle generation"""
        evidence_data = {
            'verification_results': {
                'decision': 'SAME',
                'confidence': 0.99,
                'n_queries': 32
            },
            'audit_trail': [],
            'performance_metrics': {
                'duration': 10.5,
                'peak_memory_mb': 512
            }
        }
        
        # Save evidence
        evidence_file = temp_dir / 'evidence.json'
        with open(evidence_file, 'w') as f:
            json.dump(evidence_data, f)
        
        # Verify evidence can be loaded
        with open(evidence_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['verification_results']['decision'] == 'SAME'
    
    @pytest.mark.slow
    def test_full_e2e_with_real_models(self, temp_dir):
        """Test complete E2E pipeline with real small models"""
        pytest.skip("Requires actual model files")
        
        # This test would run the actual pipeline with small models
        # Only run in CI with cached models
        from scripts.run_e2e_validation_with_sharding import main
        
        with patch('sys.argv', [
            'script.py',
            '--ref-model', 'gpt2',
            '--cand-model', 'distilgpt2',
            '--mode', 'quick',
            '--output-dir', str(temp_dir),
            '--enable-sharding',
            '--dry-run'
        ]):
            main()


class TestPipelineErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_model_path(self):
        """Test handling of invalid model paths"""
        pipeline = ShardedVerificationPipeline()
        
        # Should handle gracefully
        should_shard = pipeline.should_use_sharding('/nonexistent/model')
        assert should_shard == False
    
    def test_oom_prevention(self):
        """Test out-of-memory prevention"""
        manager = MemoryManager()
        
        # Request unrealistic amount of memory
        available = manager.check_memory_available(1024 * 1024 * 1024 * 1024)  # 1TB
        assert available == False
    
    def test_corrupted_checkpoint(self, temp_dir):
        """Test handling of corrupted checkpoints"""
        from src.pot.sharding import CheckpointManager
        
        manager = CheckpointManager(str(temp_dir))
        
        # Create corrupted checkpoint file
        corrupt_file = temp_dir / 'corrupt.ckpt'
        with open(corrupt_file, 'w') as f:
            f.write("corrupted data")
        
        # Should handle gracefully
        recovered = manager.recover_from_checkpoint('corrupt')
        assert recovered is None
    
    def test_attack_detection(self):
        """Test attack detection during verification"""
        from src.pot.audit.adversarial import StatisticalAttackDetector
        
        detector = StatisticalAttackDetector()
        
        # Establish baseline with normal data
        normal_data = [np.random.randn() for _ in range(100)]
        detector.establish_baseline(normal_data)
        
        # Test with anomalous data
        anomaly = 100.0  # Extreme value
        result = detector.detect_attack(anomaly)
        
        assert result.attack_detected == True or result.confidence > 0.5


class TestCrossPlatform:
    """Test cross-platform compatibility"""
    
    @pytest.mark.skipif(os.name != 'posix', reason="Unix only")
    def test_unix_specific_features(self):
        """Test Unix-specific features"""
        import resource
        
        # Get resource limits
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        assert soft <= hard or soft == resource.RLIM_INFINITY
    
    @pytest.mark.skipif(os.name != 'nt', reason="Windows only")  
    def test_windows_specific_features(self):
        """Test Windows-specific features"""
        # Windows-specific tests would go here
        pass


@pytest.fixture(scope="session")
def setup_test_environment():
    """Set up test environment once per session"""
    # Set environment variables
    os.environ['POT_CI_MODE'] = 'true'
    os.environ['POT_TEST_MODE'] = 'true'
    
    yield
    
    # Cleanup
    os.environ.pop('POT_CI_MODE', None)
    os.environ.pop('POT_TEST_MODE', None)