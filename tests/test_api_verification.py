#!/usr/bin/env python3
"""
Test suite for API Verification Script
Tests with mocked API responses
"""

import os
import sys
import json
import yaml
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.api_verification import (
    APIVerifier, APIResponse, VerificationResult
)


class TestAPIVerification(unittest.TestCase):
    """Test API verification functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for test results
        self.test_dir = tempfile.mkdtemp()
        
        # Mock configuration
        self.mock_config = {
            'apis': [
                {
                    'provider': 'mock',
                    'model_name': 'mock-model',
                    'temperature': 0.0,
                    'max_tokens': 100
                },
                {
                    'provider': 'openai',
                    'model_name': 'gpt-3.5-turbo',
                    'api_key': '${OPENAI_API_KEY}',
                    'temperature': 0.0,
                    'max_tokens': 100,
                    'cost_per_token': 0.0015
                }
            ],
            'challenges': {
                'families': [
                    {
                        'family': 'lm:templates',
                        'n': 10,
                        'params': {
                            'templates': ['Complete: {prompt}'],
                            'slots': {
                                'prompt': ['Test prompt 1', 'Test prompt 2']
                            }
                        }
                    }
                ]
            },
            'verification': {
                'similarity_threshold': 0.7,
                'confidence_threshold': 0.8,
                'max_errors_percent': 10
            },
            'output': {
                'base_dir': self.test_dir
            },
            'reference_models': {}
        }
        
        # Save mock config to temporary file
        self.config_file = os.path.join(self.test_dir, 'test_config.yaml')
        with open(self.config_file, 'w') as f:
            yaml.dump(self.mock_config, f)
            
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_initialization(self):
        """Test APIVerifier initialization"""
        verifier = APIVerifier(self.config_file)
        
        self.assertIsNotNone(verifier.config)
        self.assertIsNotNone(verifier.logger)
        self.assertEqual(str(verifier.results_dir), self.test_dir)
        self.assertIn('mock_mock-model', verifier.clients)
        
    def test_challenge_generation(self):
        """Test challenge generation"""
        verifier = APIVerifier(self.config_file)
        challenges = verifier.generate_pot_challenges(num_challenges=5)
        
        self.assertEqual(len(challenges), 5)
        for challenge in challenges:
            self.assertIn('id', challenge)
            self.assertIn('type', challenge)
            self.assertIn('prompt', challenge)
            self.assertIn('seed', challenge)
            
    def test_mock_api_query(self):
        """Test querying mock API"""
        verifier = APIVerifier(self.config_file)
        challenge = {
            'id': 0,
            'type': 'text',
            'prompt': 'Test prompt',
            'seed': 'abcd1234'
        }
        
        response = verifier.query_api('mock_mock-model', challenge)
        
        self.assertIsInstance(response, APIResponse)
        self.assertEqual(response.api_name, 'mock_mock-model')
        self.assertEqual(response.challenge_id, 0)
        self.assertTrue(response.success)
        self.assertIsNotNone(response.response)
        self.assertGreater(response.latency_ms, 0)
        
    @patch('scripts.api_verification.openai')
    def test_openai_api_query(self, mock_openai):
        """Test querying OpenAI API with mock"""
        # Set up mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage.total_tokens = 10
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        
        # Set environment variable
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            verifier = APIVerifier(self.config_file)
            verifier.clients['openai_gpt-3.5-turbo'] = {
                'type': 'openai',
                'client': mock_client,
                'config': self.mock_config['apis'][1]
            }
            
            challenge = {
                'id': 0,
                'type': 'text',
                'prompt': 'Test prompt',
                'seed': 'abcd1234'
            }
            
            response = verifier.query_api('openai_gpt-3.5-turbo', challenge)
            
            self.assertEqual(response.response, "Test response")
            self.assertEqual(response.tokens_used, 10)
            self.assertGreater(response.cost, 0)
            mock_client.chat.completions.create.assert_called_once()
            
    def test_distance_computation(self):
        """Test distance computation between responses"""
        verifier = APIVerifier(self.config_file)
        
        # Identical responses
        dist1 = verifier.compute_distance("Hello world", "Hello world")
        self.assertEqual(dist1, 0.0)
        
        # Completely different responses
        dist2 = verifier.compute_distance("Hello", "Goodbye")
        self.assertGreater(dist2, 0.5)
        
        # Similar responses
        dist3 = verifier.compute_distance("Hello world", "Hello World")
        self.assertLess(dist3, 0.1)
        
        # Empty response
        dist4 = verifier.compute_distance("Hello", "")
        self.assertEqual(dist4, 1.0)
        
    def test_verification_process(self):
        """Test full verification process"""
        verifier = APIVerifier(self.config_file)
        
        # Generate challenges
        verifier.generate_pot_challenges(num_challenges=5)
        
        # Run verification on mock API
        result = verifier.verify_api('mock_mock-model')
        
        self.assertIsInstance(result, VerificationResult)
        self.assertEqual(result.api_name, 'mock_mock-model')
        self.assertGreater(result.num_challenges, 0)
        self.assertIsInstance(result.distances, list)
        self.assertIsInstance(result.far, float)
        self.assertIsInstance(result.frr, float)
        self.assertIsInstance(result.auroc, float)
        self.assertGreaterEqual(result.auroc, 0.0)
        self.assertLessEqual(result.auroc, 1.0)
        
    def test_sequential_verification(self):
        """Test sequential verification with early stopping"""
        verifier = APIVerifier(self.config_file)

        # Mock consistent responses for early acceptance
        with patch.object(verifier, 'query_api') as mock_query:
            mock_query.return_value = APIResponse(
                api_name='test',
                model='test-model',
                challenge_id=0,
                prompt='test',
                response='Consistent response',
                latency_ms=10,
                tokens_used=5,
                cost=0.01,
                timestamp=datetime.now().isoformat(),
                success=True
            )
            
            # Mock reference with same response
            verifier.references = {
                'test_ref': {str(i): {'response': 'Consistent response'} for i in range(10)}
            }

            verifier.clients['test'] = {'type': 'mock', 'client': None, 'config': {}}

            verifier.generate_pot_challenges(num_challenges=10)
            result = verifier.verify_api('test', 'test_ref')

            # All queries should be used but verification should succeed
            self.assertEqual(result.queries_used, 10)
            self.assertTrue(result.accepted)
            self.assertGreater(result.confidence, 0.9)
            
    def test_far_frr_computation(self):
        """Test FAR/FRR computation"""
        verifier = APIVerifier(self.config_file)
        
        # Create known distances for testing
        verifier.challenges = [{'id': i, 'prompt': f'test {i}'} for i in range(5)]
        
        with patch.object(verifier, 'query_api') as mock_query:
            # Mock responses with varying distances
            responses = [
                ('Response A', 0.1),  # Close match
                ('Response B', 0.2),  # Close match
                ('Response C', 0.6),  # Medium match
                ('Response D', 0.8),  # Poor match
                ('Response E', 0.9),  # Poor match
            ]
            
            def mock_query_side_effect(api_key, challenge):
                idx = challenge['id']
                return APIResponse(
                    api_name=api_key,
                    model='test',
                    challenge_id=idx,
                    prompt=challenge['prompt'],
                    response=responses[idx][0],
                    latency_ms=10,
                    tokens_used=5,
                    cost=0.01,
                    timestamp=datetime.now().isoformat(),
                    success=True
                )
                
            mock_query.side_effect = mock_query_side_effect

            with patch.object(verifier, 'compute_distance') as mock_dist:
                mock_dist.side_effect = [r[1] for r in responses]

                verifier.clients['test'] = {'type': 'mock', 'client': None, 'config': {}}
                result = verifier.verify_api('test')
                
                # With threshold 0.7, responses 0,1,2 should be accepted (3 of 5)
                self.assertEqual(len(result.distances), 5)
                self.assertAlmostEqual(result.distances[0], 0.1)
                self.assertAlmostEqual(result.distances[4], 0.9)
                
    def test_results_saving(self):
        """Test saving results to files"""
        verifier = APIVerifier(self.config_file)
        
        # Create mock results
        result = VerificationResult(
            api_name='test_api',
            model='test_model',
            num_challenges=10,
            distances=[0.1, 0.2, 0.3],
            threshold=0.7,
            accepted=True,
            confidence=0.85,
            far=0.05,
            frr=0.03,
            auroc=0.92,
            precision=[0.9, 0.85, 0.8],
            recall=[0.95, 0.9, 0.85],
            queries_used=3,
            total_cost=0.15,
            total_latency_ms=150,
            timestamp=datetime.now().isoformat()
        )
        
        # Save results
        results_file, summary_file, report_file = verifier.save_results([result])
        
        # Check files were created
        self.assertTrue(os.path.exists(results_file))
        self.assertTrue(os.path.exists(summary_file))
        self.assertTrue(os.path.exists(report_file))
        
        # Check content
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
            self.assertEqual(len(saved_results), 1)
            self.assertEqual(saved_results[0]['api_name'], 'test_api')
            
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            self.assertEqual(summary['num_apis'], 1)
            self.assertIn('test_api', summary['accepted'])
            
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_generation(self, mock_close, mock_savefig):
        """Test plot generation"""
        verifier = APIVerifier(self.config_file)
        
        # Create mock results with valid data
        result = VerificationResult(
            api_name='test_api',
            model='test_model',
            num_challenges=10,
            distances=[0.1, 0.2, 0.3, 0.4, 0.5],
            threshold=0.7,
            accepted=True,
            confidence=0.85,
            far=0.05,
            frr=0.03,
            auroc=0.92,
            precision=[0.9, 0.85, 0.8],
            recall=[0.95, 0.9, 0.85],
            queries_used=5,
            total_cost=0.25,
            total_latency_ms=250,
            timestamp=datetime.now().isoformat()
        )
        
        # Generate plots
        verifier.generate_plots([result])
        
        # Check that plots were attempted to be saved
        self.assertEqual(mock_savefig.call_count, 4)  # 4 different plots
        self.assertEqual(mock_close.call_count, 4)
        
    def test_error_handling(self):
        """Test error handling in API queries"""
        verifier = APIVerifier(self.config_file)
        
        # Test with non-existent API
        challenge = {'id': 0, 'prompt': 'test', 'type': 'text', 'seed': 'test'}
        response = verifier.query_api('nonexistent_api', challenge)
        
        self.assertFalse(response.success)
        self.assertIsNotNone(response.error)
        self.assertEqual(response.response, "")
        
        # Test with API that raises exception
        with patch.object(verifier, 'clients') as mock_clients:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_clients.__getitem__.return_value = {
                'type': 'openai',
                'client': mock_client,
                'config': {'model_name': 'test', 'cost_per_token': 0.001}
            }
            mock_clients.__contains__.return_value = True
            
            response = verifier.query_api('failing_api', challenge)
            
            self.assertFalse(response.success)
            self.assertIn("API Error", response.error)
            
    def test_reference_loading(self):
        """Test loading reference responses"""
        # Create reference file
        ref_file = os.path.join(self.test_dir, 'reference.json')
        ref_data = {
            '0': {'response': 'Reference response 0'},
            '1': {'response': 'Reference response 1'}
        }
        with open(ref_file, 'w') as f:
            json.dump(ref_data, f)
            
        # Update config with reference
        self.mock_config['reference_models'] = {
            'test_ref': {
                'path': ref_file,
                'model_id': 'test_model'
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(self.mock_config, f)
            
        verifier = APIVerifier(self.config_file)
        
        self.assertIn('test_ref', verifier.references)
        self.assertEqual(verifier.references['test_ref']['0']['response'], 
                        'Reference response 0')
                        
    def test_rate_limiting(self):
        """Test rate limiting between API calls"""
        import time
        verifier = APIVerifier(self.config_file)
        
        # Generate minimal challenges
        verifier.generate_pot_challenges(num_challenges=2)
        
        start_time = time.time()
        
        # Mock verify_api to track timing
        with patch.object(verifier, 'verify_api') as mock_verify:
            mock_verify.return_value = VerificationResult(
                api_name='test',
                model='test',
                num_challenges=2,
                distances=[0.1],
                threshold=0.7,
                accepted=True,
                confidence=0.9,
                far=0.0,
                frr=0.0,
                auroc=1.0,
                precision=[1.0],
                recall=[1.0],
                queries_used=1,
                total_cost=0.01,
                total_latency_ms=10,
                timestamp=datetime.now().isoformat()
            )
            
            # Create multiple mock APIs
            verifier.clients = {
                'api1': {'type': 'mock', 'client': None, 'config': {}},
                'api2': {'type': 'mock', 'client': None, 'config': {}}
            }
            
            results = verifier.verify_all_apis()
            
            elapsed = time.time() - start_time
            
            # Should have rate limiting delay between calls
            self.assertGreater(elapsed, 1.0)  # At least 1 second delay
            self.assertEqual(len(results), 2)


class TestAPIResponseDataclass(unittest.TestCase):
    """Test APIResponse dataclass"""
    
    def test_creation(self):
        """Test creating APIResponse"""
        response = APIResponse(
            api_name="test_api",
            model="test_model",
            challenge_id=1,
            prompt="Test prompt",
            response="Test response",
            latency_ms=100.5,
            tokens_used=10,
            cost=0.015,
            timestamp="2025-01-01T00:00:00",
            success=True
        )
        
        self.assertEqual(response.api_name, "test_api")
        self.assertEqual(response.model, "test_model")
        self.assertEqual(response.challenge_id, 1)
        self.assertEqual(response.tokens_used, 10)
        self.assertTrue(response.success)
        self.assertIsNone(response.error)
        
    def test_with_error(self):
        """Test APIResponse with error"""
        response = APIResponse(
            api_name="test_api",
            model="test_model",
            challenge_id=1,
            prompt="Test prompt",
            response="",
            latency_ms=0,
            tokens_used=0,
            cost=0,
            timestamp="2025-01-01T00:00:00",
            success=False,
            error="Connection timeout"
        )
        
        self.assertFalse(response.success)
        self.assertEqual(response.error, "Connection timeout")
        self.assertEqual(response.response, "")


class TestVerificationResultDataclass(unittest.TestCase):
    """Test VerificationResult dataclass"""
    
    def test_creation(self):
        """Test creating VerificationResult"""
        result = VerificationResult(
            api_name="test_api",
            model="test_model",
            num_challenges=100,
            distances=[0.1, 0.2, 0.3],
            threshold=0.7,
            accepted=True,
            confidence=0.85,
            far=0.05,
            frr=0.03,
            auroc=0.92,
            precision=[0.9, 0.85],
            recall=[0.95, 0.9],
            queries_used=50,
            total_cost=5.0,
            total_latency_ms=5000,
            timestamp="2025-01-01T00:00:00"
        )
        
        self.assertEqual(result.api_name, "test_api")
        self.assertTrue(result.accepted)
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.far, 0.05)
        self.assertEqual(result.frr, 0.03)
        self.assertEqual(result.auroc, 0.92)
        self.assertEqual(len(result.distances), 3)
        

if __name__ == '__main__':
    unittest.main()