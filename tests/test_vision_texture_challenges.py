"""Unit tests for vision:texture challenge generation."""

import pytest
import numpy as np
from pot.core.challenge import ChallengeConfig, generate_challenges, Challenge


class TestVisionTextureChallenges:
    """Test suite for vision:texture challenge generation."""
    
    def test_generate_basic_texture_challenges(self):
        """Test basic texture challenge generation."""
        config = ChallengeConfig(
            master_key_hex="a" * 64,
            session_nonce_hex="b" * 32,
            n=10,
            family="vision:texture",
            params={}  # Use defaults
        )
        
        result = generate_challenges(config)
        
        # Check structure
        assert result['family'] == "vision:texture"
        assert 'challenge_id' in result
        assert 'salt' in result
        assert 'items' in result
        assert 'challenges' in result
        
        # Check challenges
        assert len(result['challenges']) == 10
        for i, challenge in enumerate(result['challenges']):
            assert isinstance(challenge, Challenge)
            assert challenge.index == i
            assert challenge.family == "vision:texture"
            assert 'texture_type' in challenge.parameters
            assert challenge.parameters['texture_type'] in ['perlin', 'gabor', 'checkerboard']
    
    def test_perlin_noise_parameters(self):
        """Test Perlin noise texture generation with specific parameters."""
        config = ChallengeConfig(
            master_key_hex="1234567890abcdef" * 4,
            session_nonce_hex="fedcba9876543210" * 2,
            n=5,
            family="vision:texture",
            params={
                "texture_types": ["perlin"],  # Only Perlin noise
                "perlin": {
                    "octaves": [2, 4],
                    "persistence": [0.4, 0.6],
                    "scale": [0.02, 0.08]
                }
            }
        )
        
        result = generate_challenges(config)
        
        for challenge in result['challenges']:
            params = challenge.parameters
            assert params['texture_type'] == 'perlin'
            assert 'octaves' in params
            assert 'persistence' in params
            assert 'scale' in params
            assert 'seed' in params
            
            # Check parameter ranges
            assert 2 <= params['octaves'] <= 4
            assert 0.4 <= params['persistence'] <= 0.6
            assert 0.02 <= params['scale'] <= 0.08
            assert isinstance(params['seed'], int)
    
    def test_gabor_filter_parameters(self):
        """Test Gabor filter texture generation with specific parameters."""
        config = ChallengeConfig(
            master_key_hex="abcdef1234567890" * 4,
            session_nonce_hex="1234567890abcdef" * 2,
            n=5,
            family="vision:texture",
            params={
                "texture_types": ["gabor"],  # Only Gabor filters
                "gabor": {
                    "wavelength": [10.0, 20.0],
                    "orientation": [0, 90],
                    "phase": [0, np.pi],
                    "sigma": [5.0, 10.0],
                    "aspect_ratio": [0.5, 0.8]
                }
            }
        )
        
        result = generate_challenges(config)
        
        for challenge in result['challenges']:
            params = challenge.parameters
            assert params['texture_type'] == 'gabor'
            assert 'wavelength' in params
            assert 'orientation' in params
            assert 'orientation_rad' in params
            assert 'phase' in params
            assert 'sigma' in params
            assert 'aspect_ratio' in params
            
            # Check parameter ranges
            assert 10.0 <= params['wavelength'] <= 20.0
            assert 0 <= params['orientation'] <= 90
            assert 0 <= params['phase'] <= np.pi
            assert 5.0 <= params['sigma'] <= 10.0
            assert 0.5 <= params['aspect_ratio'] <= 0.8
            
            # Check orientation conversion
            assert abs(params['orientation_rad'] - np.deg2rad(params['orientation'])) < 1e-6
    
    def test_checkerboard_parameters(self):
        """Test checkerboard pattern generation with specific parameters."""
        config = ChallengeConfig(
            master_key_hex="deadbeefcafebabe" * 4,
            session_nonce_hex="0123456789abcdef" * 2,
            n=5,
            family="vision:texture",
            params={
                "texture_types": ["checkerboard"],  # Only checkerboards
                "checkerboard": {
                    "square_size": [16, 24],
                    "contrast": [0.5, 0.8],
                    "rotation": [0, 30]
                }
            }
        )
        
        result = generate_challenges(config)
        
        for challenge in result['challenges']:
            params = challenge.parameters
            assert params['texture_type'] == 'checkerboard'
            assert 'square_size' in params
            assert 'contrast' in params
            assert 'rotation' in params
            assert 'rotation_rad' in params
            assert 'phase_x' in params
            assert 'phase_y' in params
            
            # Check parameter ranges
            assert 16 <= params['square_size'] <= 24
            assert 0.5 <= params['contrast'] <= 0.8
            assert 0 <= params['rotation'] <= 30
            
            # Check rotation conversion
            assert abs(params['rotation_rad'] - np.deg2rad(params['rotation'])) < 1e-6
            
            # Check phase offsets
            assert 0 <= params['phase_x'] <= params['square_size']
            assert 0 <= params['phase_y'] <= params['square_size']
    
    def test_mixed_texture_types(self):
        """Test generation with multiple texture types."""
        config = ChallengeConfig(
            master_key_hex="0011223344556677" * 4,
            session_nonce_hex="8899aabbccddeeff" * 2,
            n=30,
            family="vision:texture",
            params={
                "texture_types": ["perlin", "gabor", "checkerboard"]
            }
        )
        
        result = generate_challenges(config)
        
        # Count texture types
        type_counts = {"perlin": 0, "gabor": 0, "checkerboard": 0}
        for challenge in result['challenges']:
            texture_type = challenge.parameters['texture_type']
            type_counts[texture_type] += 1
        
        # Should have at least some of each type (probabilistically)
        assert type_counts["perlin"] > 0
        assert type_counts["gabor"] > 0
        assert type_counts["checkerboard"] > 0
    
    def test_determinism(self):
        """Test that same config produces same texture challenges."""
        config = ChallengeConfig(
            master_key_hex="fedcba9876543210" * 4,
            session_nonce_hex="0123456789abcdef" * 2,
            n=10,
            family="vision:texture",
            params={
                "texture_types": ["perlin", "gabor", "checkerboard"],
                "perlin": {"octaves": [1, 5], "persistence": [0.3, 0.7]},
                "gabor": {"wavelength": [5.0, 30.0], "orientation": [0, 180]},
                "checkerboard": {"square_size": [8, 32], "contrast": [0.3, 1.0]}
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
        """Test that model_id affects texture challenge generation."""
        base_config = {
            "master_key_hex": "aabbccddeeff0011" * 4,
            "session_nonce_hex": "2233445566778899" * 2,
            "n": 5,
            "family": "vision:texture",
            "params": {
                "texture_types": ["perlin", "gabor", "checkerboard"]
            }
        }
        
        config1 = ChallengeConfig(**base_config, model_id="model_a")
        config2 = ChallengeConfig(**base_config, model_id="model_b")
        
        result1 = generate_challenges(config1)
        result2 = generate_challenges(config2)
        
        # Different model_id should produce different challenges
        assert result1['challenge_id'] != result2['challenge_id']
        
        # At least some parameters should differ
        params_differ = False
        for c1, c2 in zip(result1['challenges'], result2['challenges']):
            if c1.parameters != c2.parameters:
                params_differ = True
                break
        assert params_differ
    
    def test_challenge_id_uniqueness(self):
        """Test that challenge IDs are unique."""
        config = ChallengeConfig(
            master_key_hex="1122334455667788" * 4,
            session_nonce_hex="99aabbccddeeff00" * 2,
            n=20,
            family="vision:texture",
            params={
                "texture_types": ["perlin", "gabor", "checkerboard"]
            }
        )
        
        result = generate_challenges(config)
        
        # Collect all challenge IDs
        challenge_ids = [c.challenge_id for c in result['challenges']]
        
        # All should be unique
        assert len(challenge_ids) == len(set(challenge_ids))
        
        # All should be valid hex strings
        for cid in challenge_ids:
            assert len(cid) == 16  # xxhash.xxh3_64_hexdigest produces 16 chars
            int(cid, 16)  # Should be valid hex
    
    def test_metadata_inclusion(self):
        """Test that metadata is included in parameters."""
        config = ChallengeConfig(
            master_key_hex="deadbeef" * 8,
            session_nonce_hex="cafebabe" * 4,
            n=3,
            family="vision:texture",
            params={
                "texture_types": ["perlin"],
                "perlin": {
                    "octaves": [2, 3],
                    "persistence": [0.5, 0.6],
                    "scale": [0.05, 0.06]
                }
            }
        )
        
        result = generate_challenges(config)
        
        for challenge in result['challenges']:
            params = challenge.parameters
            # Should include range metadata
            assert 'octaves_range' in params
            assert 'persistence_range' in params
            assert 'scale_range' in params
            assert params['octaves_range'] == [2, 3]
            assert params['persistence_range'] == [0.5, 0.6]
            assert params['scale_range'] == [0.05, 0.06]
    
    def test_unknown_texture_type_fallback(self):
        """Test fallback behavior for unknown texture types."""
        config = ChallengeConfig(
            master_key_hex="0123456789abcdef" * 4,
            session_nonce_hex="fedcba9876543210" * 2,
            n=2,
            family="vision:texture",
            params={
                "texture_types": ["unknown_type"]
            }
        )
        
        result = generate_challenges(config)
        
        # Should still generate challenges with fallback parameters
        assert len(result['challenges']) == 2
        for challenge in result['challenges']:
            params = challenge.parameters
            assert params['texture_type'] == 'unknown_type'
            # Fallback should provide basic parameters
            assert 'octaves' in params
            assert 'scale' in params
    
    def test_backward_compatibility(self):
        """Test backward compatibility with items field."""
        config = ChallengeConfig(
            master_key_hex="abcd" * 16,
            session_nonce_hex="1234" * 8,
            n=5,
            family="vision:texture",
            params={}
        )
        
        result = generate_challenges(config)
        
        # Check items field exists for backward compatibility
        assert 'items' in result
        assert len(result['items']) == 5
        
        # Verify items match challenge parameters
        for item, challenge in zip(result['items'], result['challenges']):
            assert item == challenge.parameters
    
    def test_parameter_types(self):
        """Test that parameters have correct types."""
        config = ChallengeConfig(
            master_key_hex="ffeeddccbbaa9988" * 4,
            session_nonce_hex="7766554433221100" * 2,
            n=10,
            family="vision:texture",
            params={
                "texture_types": ["perlin", "gabor", "checkerboard"]
            }
        )
        
        result = generate_challenges(config)
        
        for challenge in result['challenges']:
            params = challenge.parameters
            texture_type = params['texture_type']
            
            if texture_type == 'perlin':
                assert isinstance(params['octaves'], int)
                assert isinstance(params['persistence'], float)
                assert isinstance(params['scale'], float)
                assert isinstance(params['seed'], int)
            
            elif texture_type == 'gabor':
                assert isinstance(params['wavelength'], float)
                assert isinstance(params['orientation'], float)
                assert isinstance(params['orientation_rad'], float)
                assert isinstance(params['phase'], float)
                assert isinstance(params['sigma'], float)
                assert isinstance(params['aspect_ratio'], float)
            
            elif texture_type == 'checkerboard':
                assert isinstance(params['square_size'], int)
                assert isinstance(params['contrast'], float)
                assert isinstance(params['rotation'], float)
                assert isinstance(params['rotation_rad'], float)
                assert isinstance(params['phase_x'], int)
                assert isinstance(params['phase_y'], int)