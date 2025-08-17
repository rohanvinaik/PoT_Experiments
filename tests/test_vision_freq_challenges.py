"""Unit tests for vision:freq challenge generation."""

import pytest
import numpy as np
from pot.core.challenge import ChallengeConfig, generate_challenges, Challenge

# Add parent directory to path for pot imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



class TestVisionFreqChallenges:
    """Test suite for vision:freq sine grating challenge generation."""
    
    def test_generate_basic_challenges(self):
        """Test basic challenge generation for vision:freq family."""
        config = ChallengeConfig(
            master_key_hex="a" * 64,
            session_nonce_hex="b" * 32,
            n=10,
            family="vision:freq",
            params={
                "freq_range": [0.5, 8.0],
                "contrast_range": [0.2, 0.9]
            }
        )
        
        result = generate_challenges(config)
        
        # Check structure
        assert result['family'] == "vision:freq"
        assert 'challenge_id' in result
        assert 'salt' in result
        assert 'items' in result
        assert 'challenges' in result
        
        # Check challenges
        assert len(result['challenges']) == 10
        for i, challenge in enumerate(result['challenges']):
            assert isinstance(challenge, Challenge)
            assert challenge.index == i
            assert challenge.family == "vision:freq"
            assert 'freq' in challenge.parameters
            assert 'theta' in challenge.parameters
            assert 'phase' in challenge.parameters
            assert 'contrast' in challenge.parameters
    
    def test_parameter_ranges(self):
        """Test that parameters are within specified ranges."""
        config = ChallengeConfig(
            master_key_hex="1234567890abcdef" * 4,  # 64 hex chars = 32 bytes
            session_nonce_hex="fedcba9876543210" * 2,  # 32 hex chars = 16 bytes
            n=100,
            family="vision:freq",
            params={
                "freq_range": [1.0, 5.0],
                "contrast_range": [0.3, 0.7]
            }
        )
        
        result = generate_challenges(config)
        
        for challenge in result['challenges']:
            params = challenge.parameters
            # Check frequency range
            assert 1.0 <= params['freq'] < 5.0
            # Check theta in degrees
            assert 0 <= params['theta'] < 180
            # Check theta in radians
            assert 0 <= params['theta_rad'] < np.pi
            # Check phase
            assert 0 <= params['phase'] < 2 * np.pi
            # Check contrast
            assert 0.3 <= params['contrast'] < 0.7
    
    def test_determinism(self):
        """Test that same config produces same challenges."""
        config = ChallengeConfig(
            master_key_hex="abcdef1234567890" * 4,
            session_nonce_hex="1234567890abcdef" * 2,
            n=5,
            family="vision:freq",
            params={
                "freq_range": [2.0, 4.0],
                "contrast_range": [0.4, 0.6]
            },
            model_id="test_model"
        )
        
        result1 = generate_challenges(config)
        result2 = generate_challenges(config)
        
        # Check overall determinism
        assert result1['challenge_id'] == result2['challenge_id']
        assert result1['salt'] == result2['salt']
        
        # Check individual challenges
        for c1, c2 in zip(result1['challenges'], result2['challenges']):
            assert c1.challenge_id == c2.challenge_id
            assert c1.parameters == c2.parameters
    
    def test_model_id_influence(self):
        """Test that model_id affects challenge generation."""
        base_config = {
            "master_key_hex": "deadbeefcafebabe" * 4,
            "session_nonce_hex": "0123456789abcdef" * 2,
            "n": 3,
            "family": "vision:freq",
            "params": {
                "freq_range": [1.0, 10.0],
                "contrast_range": [0.1, 1.0]
            }
        }
        
        config1 = ChallengeConfig(**base_config, model_id="model_a")
        config2 = ChallengeConfig(**base_config, model_id="model_b")
        
        result1 = generate_challenges(config1)
        result2 = generate_challenges(config2)
        
        # Different model_id should produce different challenges
        assert result1['challenge_id'] != result2['challenge_id']
        
        # At least some challenge parameters should differ
        params_differ = False
        for c1, c2 in zip(result1['challenges'], result2['challenges']):
            if c1.parameters != c2.parameters:
                params_differ = True
                break
        assert params_differ
    
    def test_unique_challenge_ids(self):
        """Test that each challenge has a unique ID."""
        config = ChallengeConfig(
            master_key_hex="0011223344556677" * 4,
            session_nonce_hex="8899aabbccddeeff" * 2,
            n=50,
            family="vision:freq",
            params={
                "freq_range": [0.1, 20.0],
                "contrast_range": [0.0, 1.0]
            }
        )
        
        result = generate_challenges(config)
        
        # Collect all challenge IDs
        challenge_ids = [c.challenge_id for c in result['challenges']]
        
        # All should be unique
        assert len(challenge_ids) == len(set(challenge_ids))
    
    def test_backward_compatibility(self):
        """Test that 'items' field maintains backward compatibility."""
        config = ChallengeConfig(
            master_key_hex="fedcba9876543210" * 4,
            session_nonce_hex="0123456789abcdef" * 2,
            n=5,
            family="vision:freq",
            params={
                "freq_range": [1.0, 5.0],
                "contrast_range": [0.2, 0.8]
            }
        )
        
        result = generate_challenges(config)
        
        # Check items field exists
        assert 'items' in result
        assert len(result['items']) == 5
        
        # Verify items match challenge parameters
        for item, challenge in zip(result['items'], result['challenges']):
            assert item == challenge.parameters
    
    def test_theta_radians_conversion(self):
        """Test that theta is correctly provided in both degrees and radians."""
        config = ChallengeConfig(
            master_key_hex="aabbccddeeff0011" * 4,
            session_nonce_hex="2233445566778899" * 2,
            n=20,
            family="vision:freq",
            params={
                "freq_range": [1.0, 2.0],
                "contrast_range": [0.5, 0.6]  # Need different values for valid range
            }
        )
        
        result = generate_challenges(config)
        
        for challenge in result['challenges']:
            params = challenge.parameters
            theta_deg = params['theta']
            theta_rad = params['theta_rad']
            
            # Check conversion is correct
            expected_rad = np.deg2rad(theta_deg)
            assert abs(theta_rad - expected_rad) < 1e-10
            
            # Check ranges
            assert 0 <= theta_deg < 180
            assert 0 <= theta_rad < np.pi
    
    def test_challenge_to_dict(self):
        """Test Challenge.to_dict() method."""
        config = ChallengeConfig(
            master_key_hex="1122334455667788" * 4,
            session_nonce_hex="99aabbccddeeff00" * 2,
            n=1,
            family="vision:freq",
            params={
                "freq_range": [3.0, 3.1],  # Need different values for valid range
                "contrast_range": [0.5, 0.51]  # Need different values for valid range
            }
        )
        
        result = generate_challenges(config)
        challenge = result['challenges'][0]
        
        # Convert to dict
        challenge_dict = challenge.to_dict()
        
        # Check structure
        assert 'challenge_id' in challenge_dict
        assert 'index' in challenge_dict
        assert 'family' in challenge_dict
        assert 'parameters' in challenge_dict
        
        # Check values
        assert challenge_dict['challenge_id'] == challenge.challenge_id
        assert challenge_dict['index'] == challenge.index
        assert challenge_dict['family'] == challenge.family
        assert challenge_dict['parameters'] == challenge.parameters