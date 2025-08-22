"""
Integration tests for the End-to-End Validation Pipeline

Tests pipeline stage transitions, metric collection, report generation,
and evidence bundle integrity.
"""

import unittest
import tempfile
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pot.validation.e2e_pipeline import (
    PipelineOrchestrator,
    PipelineConfig,
    PipelineStage,
    StageMetrics,
    TestingMode,
    VerificationMode
)
from src.pot.validation.reporting import ReportGenerator


class TestPipelineOrchestrator(unittest.TestCase):
    """Test cases for PipelineOrchestrator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = PipelineConfig(
            testing_mode=TestingMode.QUICK_GATE,
            verification_mode=VerificationMode.LOCAL_WEIGHTS,
            output_dir=Path(self.temp_dir),
            dry_run=True,
            verbose=False
        )
        self.orchestrator = PipelineOrchestrator(self.config)
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test orchestrator initialization"""
        self.assertEqual(self.orchestrator.current_stage, PipelineStage.INITIALIZATION)
        self.assertIsNotNone(self.orchestrator.run_id)
        self.assertEqual(self.orchestrator.config.testing_mode, TestingMode.QUICK_GATE)
        self.assertTrue(self.orchestrator.config.dry_run)
    
    def test_stage_transitions(self):
        """Test pipeline stage transitions"""
        # Test pre-commit stage
        result = self.orchestrator.pre_commit_challenges(n_challenges=5)
        
        self.assertIn('commitment', result)
        self.assertIn('seeds', result)
        self.assertEqual(len(result['seeds']), 2)  # Limited in dry-run
        self.assertIn(PipelineStage.PRE_COMMIT, self.orchestrator.stage_metrics)
        
        # Check stage metrics
        pre_commit_metrics = self.orchestrator.stage_metrics[PipelineStage.PRE_COMMIT]
        self.assertIsInstance(pre_commit_metrics, StageMetrics)
        self.assertGreater(pre_commit_metrics.duration, 0)
    
    def test_challenge_generation(self):
        """Test challenge generation from seeds"""
        seeds = ['seed1', 'seed2', 'seed3']
        challenges = self.orchestrator.generate_challenges(seeds)
        
        self.assertEqual(len(challenges), 3)
        for i, challenge in enumerate(challenges):
            self.assertIn('id', challenge)
            self.assertIn('seed', challenge)
            self.assertIn('prompt', challenge)
            self.assertEqual(challenge['seed'], seeds[i])
    
    def test_model_loading_dry_run(self):
        """Test model loading in dry-run mode"""
        ref_model, cand_model = self.orchestrator.load_models(
            ref_model_path='mock_ref_model',
            cand_model_path='mock_cand_model'
        )
        
        self.assertIsNotNone(ref_model)
        self.assertIsNotNone(cand_model)
        self.assertIn(PipelineStage.MODEL_LOADING, self.orchestrator.stage_metrics)
    
    def test_verification_dry_run(self):
        """Test verification in dry-run mode"""
        mock_ref = Mock()
        mock_cand = Mock()
        challenges = [
            {'id': f'ch_{i}', 'prompt': f'prompt_{i}'}
            for i in range(5)
        ]
        
        result = self.orchestrator.run_verification(
            ref_model=mock_ref,
            cand_model=mock_cand,
            challenges=challenges
        )
        
        self.assertIn('decision', result)
        self.assertIn('confidence', result)
        self.assertIn('n_queries', result)
        self.assertIn('ci_progression', result)
        self.assertEqual(result['decision'], 'SAME')
    
    def test_evidence_bundle_generation(self):
        """Test evidence bundle generation"""
        # Populate some evidence
        self.orchestrator.evidence_bundle['pre_commit'] = {
            'commitment': 'test_commitment',
            'seeds': ['seed1', 'seed2']
        }
        self.orchestrator.evidence_bundle['verification'] = {
            'decision': 'SAME',
            'confidence': 0.99
        }
        
        # Generate bundle
        bundle = self.orchestrator.generate_evidence_bundle()
        
        self.assertIn('run_id', bundle)
        self.assertIn('timestamp', bundle)
        self.assertIn('hash', bundle)
        self.assertIn('pre_commit', bundle)
        self.assertIn('verification', bundle)
        self.assertIn('environment', bundle)
        
        # Check file was saved
        bundle_files = list(Path(self.temp_dir).glob('evidence_bundle_*.json'))
        self.assertEqual(len(bundle_files), 1)
    
    def test_complete_pipeline_dry_run(self):
        """Test complete pipeline execution in dry-run mode"""
        results = self.orchestrator.run_complete_pipeline(
            ref_model_path='test_ref',
            cand_model_path='test_cand',
            n_challenges=10
        )
        
        self.assertTrue(results['success'])
        self.assertIn('decision', results)
        self.assertIn('confidence', results)
        self.assertIn('n_queries', results)
        self.assertIn('total_duration', results)
        self.assertIn('peak_memory_mb', results)
        self.assertIn('stage_metrics', results)
        
        # Check all stages were executed
        expected_stages = [
            PipelineStage.PRE_COMMIT,
            PipelineStage.CHALLENGE_GENERATION,
            PipelineStage.MODEL_LOADING,
            PipelineStage.VERIFICATION,
            PipelineStage.EVIDENCE_GENERATION
        ]
        
        for stage in expected_stages:
            self.assertIn(stage, self.orchestrator.stage_metrics)
    
    def test_benchmark_mode(self):
        """Test benchmark mode execution"""
        results = self.orchestrator.benchmark_pipeline(
            ref_model_path='test_ref',
            cand_model_path='test_cand',
            n_runs=2
        )
        
        self.assertEqual(results['n_runs'], 2)
        self.assertIn('n_successful', results)
        self.assertIn('individual_runs', results)
        
        if results['n_successful'] > 0:
            self.assertIn('avg_duration_seconds', results)
            self.assertIn('avg_peak_memory_mb', results)
            self.assertIn('avg_queries', results)
    
    def test_metric_collection_accuracy(self):
        """Test accuracy of metric collection"""
        # Start a stage
        metrics = self.orchestrator._start_stage(PipelineStage.VERIFICATION)
        
        # Simulate some processing
        time.sleep(0.1)
        metrics.query_count = 10
        metrics.ci_progression = [(0.1, 0.2), (0.05, 0.15)]
        
        # End the stage
        self.orchestrator._end_stage(metrics)
        
        # Check metrics
        self.assertGreater(metrics.duration, 0.09)  # Should be at least 0.1s
        self.assertEqual(metrics.query_count, 10)
        self.assertEqual(len(metrics.ci_progression), 2)
        
        if self.config.enable_memory_tracking:
            self.assertGreaterEqual(metrics.memory_peak_mb, 0)
    
    def test_error_handling(self):
        """Test error handling in pipeline"""
        # Create orchestrator that will fail
        config = PipelineConfig(
            output_dir=Path('/invalid/path/that/does/not/exist'),
            dry_run=False,
            verbose=False
        )
        
        with self.assertRaises(Exception):
            orchestrator = PipelineOrchestrator(config)
            # This should fail due to invalid path
            orchestrator.generate_evidence_bundle()


class TestStageMetrics(unittest.TestCase):
    """Test cases for StageMetrics class"""
    
    def test_stage_metrics_initialization(self):
        """Test StageMetrics initialization"""
        metrics = StageMetrics(
            stage=PipelineStage.VERIFICATION,
            start_time=100.0
        )
        
        self.assertEqual(metrics.stage, PipelineStage.VERIFICATION)
        self.assertEqual(metrics.start_time, 100.0)
        self.assertEqual(metrics.duration, 0.0)
        self.assertEqual(metrics.query_count, 0)
        self.assertEqual(len(metrics.errors), 0)
    
    def test_stage_metrics_duration(self):
        """Test duration calculation"""
        metrics = StageMetrics(
            stage=PipelineStage.VERIFICATION,
            start_time=100.0,
            end_time=125.5
        )
        
        self.assertEqual(metrics.duration, 25.5)
    
    def test_stage_metrics_to_dict(self):
        """Test conversion to dictionary"""
        metrics = StageMetrics(
            stage=PipelineStage.VERIFICATION,
            start_time=100.0,
            end_time=110.0,
            query_count=50,
            memory_peak_mb=256.5
        )
        
        metrics_dict = metrics.to_dict()
        
        self.assertEqual(metrics_dict['stage'], 'verification')
        self.assertEqual(metrics_dict['duration'], 10.0)
        self.assertEqual(metrics_dict['query_count'], 50)
        self.assertEqual(metrics_dict['memory_peak_mb'], 256.5)


class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ReportGenerator(output_dir=Path(self.temp_dir))
        
        # Mock pipeline results
        self.mock_results = {
            'run_id': 'test_run_123',
            'success': True,
            'decision': 'SAME',
            'confidence': 0.99,
            'n_queries': 32,
            'total_duration': 45.0,
            'peak_memory_mb': 256.0,
            'stage_metrics': {
                'pre_commit': {
                    'duration': 1.0,
                    'memory_peak_mb': 50.0
                },
                'verification': {
                    'duration': 30.0,
                    'memory_peak_mb': 256.0,
                    'query_count': 32,
                    'ci_progression': [(0.1, 0.2), (0.05, 0.15)]
                }
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_html_report_generation(self):
        """Test HTML report generation"""
        report_path = self.generator.generate_html_report(self.mock_results)
        
        self.assertTrue(report_path.exists())
        self.assertTrue(report_path.suffix == '.html')
        
        # Check content
        with open(report_path, 'r') as f:
            content = f.read()
        
        self.assertIn('test_run_123', content)
        self.assertIn('SAME', content)
        self.assertIn('0.99', content)
        self.assertIn('32', content)
    
    def test_summary_json_generation(self):
        """Test JSON summary generation"""
        summary_path = Path(self.temp_dir) / 'summary.json'
        summary = self.generator.generate_summary_json(
            self.mock_results,
            save_path=summary_path
        )
        
        self.assertIn('run_id', summary)
        self.assertIn('decision', summary)
        self.assertIn('metrics', summary)
        self.assertEqual(summary['decision'], 'SAME')
        self.assertEqual(summary['metrics']['n_queries'], 32)
        
        # Check file was saved
        self.assertTrue(summary_path.exists())
        with open(summary_path, 'r') as f:
            saved_summary = json.load(f)
        self.assertEqual(saved_summary['decision'], 'SAME')
    
    def test_visualization_generation(self):
        """Test visualization generation (if matplotlib available)"""
        try:
            import matplotlib
            visualizations = self.generator._generate_visualizations(self.mock_results)
            
            # Some visualizations should be generated
            self.assertIsInstance(visualizations, dict)
            # At least stage duration should work
            if visualizations:
                for key, value in visualizations.items():
                    self.assertTrue(value.startswith('data:image/png;base64,'))
        except ImportError:
            # Skip if matplotlib not available
            pass
    
    def test_evidence_bundle_summary(self):
        """Test evidence bundle summary in report"""
        evidence_bundle = {
            'hash': 'test_hash_123456',
            'challenges': [{'id': f'ch_{i}'} for i in range(10)],
            'pre_commit': {
                'commitment': 'commitment_hash_abcdef'
            }
        }
        
        report_path = self.generator.generate_html_report(
            self.mock_results,
            evidence_bundle
        )
        
        with open(report_path, 'r') as f:
            content = f.read()
        
        self.assertIn('Evidence Bundle Summary', content)
        self.assertIn('test_hash_123456', content)
        self.assertIn('10', content)  # Number of challenges


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for complete pipeline flow"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_flow(self):
        """Test complete end-to-end flow"""
        # Create pipeline
        config = PipelineConfig(
            testing_mode=TestingMode.QUICK_GATE,
            output_dir=Path(self.temp_dir),
            dry_run=True,
            verbose=False
        )
        
        orchestrator = PipelineOrchestrator(config)
        
        # Run pipeline
        results = orchestrator.run_complete_pipeline(
            ref_model_path='test_ref',
            cand_model_path='test_cand',
            n_challenges=5
        )
        
        # Verify results
        self.assertTrue(results['success'])
        
        # Generate report
        generator = ReportGenerator(output_dir=Path(self.temp_dir))
        report_path = generator.generate_html_report(results)
        
        # Check outputs exist
        self.assertTrue(report_path.exists())
        
        # Check evidence bundle exists
        bundle_files = list(Path(self.temp_dir).glob('evidence_bundle_*.json'))
        self.assertGreater(len(bundle_files), 0)
        
        # Check pipeline results exist
        results_files = list(Path(self.temp_dir).glob('pipeline_results_*.json'))
        self.assertGreater(len(results_files), 0)
    
    def test_evidence_bundle_integrity(self):
        """Test evidence bundle integrity and completeness"""
        config = PipelineConfig(
            output_dir=Path(self.temp_dir),
            dry_run=True,
            verbose=False
        )
        
        orchestrator = PipelineOrchestrator(config)
        results = orchestrator.run_complete_pipeline(
            ref_model_path='test_ref',
            cand_model_path='test_cand',
            n_challenges=5
        )
        
        # Load evidence bundle
        bundle_path = Path(results['evidence_bundle_path'])
        self.assertTrue(bundle_path.exists())
        
        with open(bundle_path, 'r') as f:
            bundle = json.load(f)
        
        # Verify bundle structure
        required_keys = ['run_id', 'timestamp', 'config', 'hash', 
                        'pre_commit', 'challenges', 'verification', 
                        'metrics', 'environment']
        
        for key in required_keys:
            self.assertIn(key, bundle)
        
        # Verify hash integrity
        import hashlib
        bundle_copy = bundle.copy()
        stored_hash = bundle_copy.pop('hash')
        
        # Recalculate hash
        bundle_json = json.dumps(bundle_copy, sort_keys=True, indent=2)
        calculated_hash = hashlib.sha256(bundle_json.encode()).hexdigest()
        
        self.assertEqual(stored_hash, calculated_hash)


if __name__ == '__main__':
    unittest.main()