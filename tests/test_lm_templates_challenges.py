"""Unit tests for lm:templates challenge generation."""

import pytest
from pot.core.challenge import ChallengeConfig, generate_challenges, Challenge

# Add parent directory to path for pot imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



class TestLMTemplatesChallenges:
    """Test suite for lm:templates text challenge generation."""
    
    def test_generate_basic_challenges(self):
        """Test basic challenge generation for lm:templates family."""
        config = ChallengeConfig(
            master_key_hex="a" * 64,
            session_nonce_hex="b" * 32,
            n=10,
            family="lm:templates",
            params={
                "templates": [
                    "The {adjective} {subject} {verb} the {object}.",
                    "A {subject} {verb} {object}."
                ],
                "slots": {
                    "subject": ["cat", "dog", "bird"],
                    "verb": ["chases", "sees", "finds"],
                    "object": ["ball", "toy", "food"],
                    "adjective": ["happy", "clever", "curious"]
                }
            }
        )
        
        result = generate_challenges(config)
        
        # Check structure
        assert result['family'] == "lm:templates"
        assert 'challenge_id' in result
        assert 'salt' in result
        assert 'items' in result
        assert 'challenges' in result
        
        # Check challenges
        assert len(result['challenges']) == 10
        for i, challenge in enumerate(result['challenges']):
            assert isinstance(challenge, Challenge)
            assert challenge.index == i
            assert challenge.family == "lm:templates"
            assert 'template' in challenge.parameters
            assert 'slot_values' in challenge.parameters
            assert 'prompt' in challenge.parameters
    
    def test_prompt_generation(self):
        """Test that prompts are correctly generated from templates and slots."""
        config = ChallengeConfig(
            master_key_hex="1234567890abcdef" * 4,
            session_nonce_hex="fedcba9876543210" * 2,
            n=5,
            family="lm:templates",
            params={
                "templates": ["The {adjective} {subject} {verb} the {object}."],
                "slots": {
                    "subject": ["cat"],
                    "verb": ["chases"],
                    "object": ["mouse"],
                    "adjective": ["clever"]
                }
            }
        )
        
        result = generate_challenges(config)
        
        # With single options, all prompts should be the same
        for challenge in result['challenges']:
            assert challenge.parameters['prompt'] == "The clever cat chases the mouse."
            assert challenge.parameters['template'] == "The {adjective} {subject} {verb} the {object}."
            assert challenge.parameters['slot_values']['subject'] == "cat"
            assert challenge.parameters['slot_values']['verb'] == "chases"
            assert challenge.parameters['slot_values']['object'] == "mouse"
            assert challenge.parameters['slot_values']['adjective'] == "clever"
    
    def test_determinism(self):
        """Test that same config produces same challenges."""
        config = ChallengeConfig(
            master_key_hex="abcdef1234567890" * 4,
            session_nonce_hex="1234567890abcdef" * 2,
            n=5,
            family="lm:templates",
            params={
                "templates": [
                    "The {subject} {verb} {object}.",
                    "{adjective} {subject} {verb}.",
                    "A {subject} will {verb} the {object}."
                ],
                "slots": {
                    "subject": ["cat", "dog", "bird", "robot"],
                    "verb": ["runs", "jumps", "flies", "walks"],
                    "object": ["ball", "stick", "toy"],
                    "adjective": ["happy", "sad", "clever"]
                }
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
            assert c1.parameters['prompt'] == c2.parameters['prompt']
    
    def test_model_id_influence(self):
        """Test that model_id affects challenge generation."""
        base_config = {
            "master_key_hex": "deadbeefcafebabe" * 4,
            "session_nonce_hex": "0123456789abcdef" * 2,
            "n": 3,
            "family": "lm:templates",
            "params": {
                "templates": [
                    "The {subject} {verb} the {object}.",
                    "A {adjective} {subject} {verb}."
                ],
                "slots": {
                    "subject": ["cat", "dog", "bird"],
                    "verb": ["sees", "hears", "finds"],
                    "object": ["ball", "toy"],
                    "adjective": ["clever", "happy"]
                }
            }
        }
        
        config1 = ChallengeConfig(**base_config, model_id="model_a")
        config2 = ChallengeConfig(**base_config, model_id="model_b")
        
        result1 = generate_challenges(config1)
        result2 = generate_challenges(config2)
        
        # Different model_id should produce different challenges
        assert result1['challenge_id'] != result2['challenge_id']
        
        # At least some prompts should differ
        prompts_differ = False
        for c1, c2 in zip(result1['challenges'], result2['challenges']):
            if c1.parameters['prompt'] != c2.parameters['prompt']:
                prompts_differ = True
                break
        assert prompts_differ
    
    def test_diversity(self):
        """Test that diverse prompts are generated with multiple options."""
        config = ChallengeConfig(
            master_key_hex="0011223344556677" * 4,
            session_nonce_hex="8899aabbccddeeff" * 2,
            n=50,
            family="lm:templates",
            params={
                "templates": [
                    "The {adjective} {subject} {verb} the {object}.",
                    "{subject} {verb} {adjective} {object}.",
                    "A {subject} will {verb} the {object}."
                ],
                "slots": {
                    "subject": ["cat", "dog", "bird", "robot", "scientist"],
                    "verb": ["chases", "observes", "creates", "discovers"],
                    "object": ["ball", "puzzle", "painting", "equation"],
                    "adjective": ["clever", "curious", "mysterious", "elegant"]
                }
            }
        )
        
        result = generate_challenges(config)
        
        # Collect unique prompts
        unique_prompts = set()
        unique_templates = set()
        for challenge in result['challenges']:
            unique_prompts.add(challenge.parameters['prompt'])
            unique_templates.add(challenge.parameters['template'])
        
        # Should have good diversity
        assert len(unique_prompts) > 10, f"Only {len(unique_prompts)} unique prompts in 50 challenges"
        assert len(unique_templates) >= 2, "Should use multiple templates"
    
    def test_default_slots(self):
        """Test that default slots are provided when not specified."""
        config = ChallengeConfig(
            master_key_hex="fedcba9876543210" * 4,
            session_nonce_hex="0123456789abcdef" * 2,
            n=5,
            family="lm:templates",
            params={}  # No templates or slots specified
        )
        
        result = generate_challenges(config)
        
        # Should use default templates and slots
        assert len(result['challenges']) == 5
        for challenge in result['challenges']:
            assert 'prompt' in challenge.parameters
            assert 'template' in challenge.parameters
            assert 'slot_values' in challenge.parameters
            # Check that prompt is not empty and contains actual words
            assert len(challenge.parameters['prompt']) > 10
            assert '[' not in challenge.parameters['prompt']  # No unfilled slots
    
    def test_backward_compatibility(self):
        """Test backward compatibility with old format."""
        config = ChallengeConfig(
            master_key_hex="aabbccddeeff0011" * 4,
            session_nonce_hex="2233445566778899" * 2,
            n=3,
            family="lm:templates",
            params={
                "templates": ["Template {slot1} and {slot2}"],
                "slots": {
                    "slot1": ["value1", "value2"],
                    "slot2": ["valueA", "valueB"]
                }
            }
        )
        
        result = generate_challenges(config)
        
        # Check items field exists for backward compatibility
        assert 'items' in result
        assert len(result['items']) == 3
        
        # Verify items match challenge parameters
        for item, challenge in zip(result['items'], result['challenges']):
            assert item == challenge.parameters
    
    def test_challenge_id_from_prompt(self):
        """Test that challenge IDs are generated from complete prompts."""
        config = ChallengeConfig(
            master_key_hex="1122334455667788" * 4,
            session_nonce_hex="99aabbccddeeff00" * 2,
            n=5,
            family="lm:templates",
            params={
                "templates": ["The {subject} {verb} the {object}."],
                "slots": {
                    "subject": ["cat", "dog"],
                    "verb": ["sees", "chases"],
                    "object": ["mouse", "ball"]
                }
            }
        )
        
        result = generate_challenges(config)
        
        # Collect prompt to ID mapping
        prompt_to_id = {}
        for challenge in result['challenges']:
            prompt = challenge.parameters['prompt']
            if prompt in prompt_to_id:
                # Same prompt should have same ID
                assert prompt_to_id[prompt] == challenge.challenge_id
            else:
                prompt_to_id[prompt] = challenge.challenge_id
        
        # All challenge IDs should be unique (unless prompts are identical)
        unique_ids = set(c.challenge_id for c in result['challenges'])
        unique_prompts = set(c.parameters['prompt'] for c in result['challenges'])
        assert len(unique_ids) == len(unique_prompts)
    
    def test_missing_slot_handling(self):
        """Test handling of templates with undefined slots."""
        config = ChallengeConfig(
            master_key_hex="deadbeef" * 8,
            session_nonce_hex="cafebabe" * 4,
            n=2,
            family="lm:templates",
            params={
                "templates": ["The {subject} {verb} the {undefined_slot}."],
                "slots": {
                    "subject": ["cat"],
                    "verb": ["sees"]
                    # Note: undefined_slot is not provided
                }
            }
        )
        
        result = generate_challenges(config)
        
        # Should handle missing slots gracefully
        for challenge in result['challenges']:
            prompt = challenge.parameters['prompt']
            # Should have placeholder for undefined slot
            assert "[undefined_slot]" in prompt
            assert challenge.parameters['slot_values']['undefined_slot'] == "[undefined_slot]"
    
    def test_complex_templates(self):
        """Test complex templates with multiple slots."""
        config = ChallengeConfig(
            master_key_hex="0123456789abcdef" * 4,
            session_nonce_hex="fedcba9876543210" * 2,
            n=10,
            family="lm:templates",
            params={
                "templates": [
                    "When the {subject} {verb}, the {object} becomes {adjective}.",
                    "The {object} was {verb_past} by the {adjective} {subject}.",
                    "{subject} and {subject2} {verb} the {adjective} {object}."
                ],
                "slots": {
                    "subject": ["scientist", "artist", "robot"],
                    "subject2": ["engineer", "designer", "assistant"],
                    "verb": ["analyzes", "creates", "discovers"],
                    "verb_past": ["analyzed", "created", "discovered"],
                    "object": ["pattern", "solution", "artifact"],
                    "adjective": ["complex", "elegant", "mysterious"]
                }
            }
        )
        
        result = generate_challenges(config)
        
        # Check that complex templates are filled correctly
        for challenge in result['challenges']:
            prompt = challenge.parameters['prompt']
            template = challenge.parameters['template']
            
            # No unfilled slots should remain
            assert '{' not in prompt
            assert '}' not in prompt
            
            # All slots in template should be in slot_values
            import re
            slot_pattern = r'\{([^}]+)\}'
            template_slots = re.findall(slot_pattern, template)
            for slot in template_slots:
                assert slot in challenge.parameters['slot_values']
    
    def test_template_metadata(self):
        """Test that template metadata is included in parameters."""
        config = ChallengeConfig(
            master_key_hex="abcd" * 16,
            session_nonce_hex="1234" * 8,
            n=5,
            family="lm:templates",
            params={
                "templates": [
                    "Template A: {slot1}",
                    "Template B: {slot2}",
                    "Template C: {slot1} and {slot2}"
                ],
                "slots": {
                    "slot1": ["value1"],
                    "slot2": ["value2"]
                }
            }
        )
        
        result = generate_challenges(config)
        
        for challenge in result['challenges']:
            params = challenge.parameters
            # Should include metadata
            assert 'available_slots' in params
            assert 'template_index' in params
            assert params['template_index'] >= 0
            assert params['template_index'] < 3