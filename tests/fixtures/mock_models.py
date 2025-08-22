#!/usr/bin/env python3
"""
Mock Models for Testing

Provides mock model implementations for testing without requiring
actual model downloads or GPU resources.
"""

import random
import time
import hashlib
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime


class MockTokenizer:
    """Mock tokenizer for testing"""
    
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.vocab = {f'token_{i}': i for i in range(vocab_size)}
        self.inverse_vocab = {i: f'token_{i}' for i in range(vocab_size)}
    
    def encode(self, text: str) -> List[int]:
        """Mock encode method"""
        # Simple hash-based encoding for consistency
        text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        random.seed(text_hash)
        
        # Generate consistent token sequence based on text length
        num_tokens = min(max(len(text.split()) + random.randint(-2, 2), 1), 20)
        tokens = [random.randint(0, self.vocab_size - 1) for _ in range(num_tokens)]
        
        # Reset random seed
        random.seed()
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Mock decode method"""
        words = [self.inverse_vocab.get(token, f'unk_{token}') for token in tokens]
        return ' '.join(words)


class MockModel:
    """Base mock model class"""
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or self._default_config()
        self.tokenizer = MockTokenizer(self.config.get('vocab_size', 50257))
        self.generation_count = 0
        
        # Model-specific response patterns
        self.response_patterns = self._get_response_patterns()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default model configuration"""
        return {
            'vocab_size': 50257,
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'max_position_embeddings': 1024,
            'model_type': 'gpt2'
        }
    
    def _get_response_patterns(self) -> List[str]:
        """Get model-specific response patterns"""
        if 'gpt2' in self.model_name.lower():
            return [
                "The {topic} is a fascinating subject that has captured the attention of researchers.",
                "In recent years, {topic} has become increasingly important in various fields.",
                "Understanding {topic} requires careful analysis and consideration of multiple factors.",
                "The implications of {topic} extend far beyond what was initially anticipated.",
                "Research into {topic} continues to reveal new insights and applications."
            ]
        elif 'distil' in self.model_name.lower():
            return [
                "The {topic} represents an interesting area of study with practical applications.",
                "Recent developments in {topic} have shown promising results for future research.",
                "Analyzing {topic} helps us understand complex relationships and patterns.",
                "The significance of {topic} becomes clear when examined from multiple perspectives.",
                "Studies of {topic} continue to provide valuable insights for researchers."
            ]
        elif 'pythia' in self.model_name.lower():
            return [
                "{topic} is a concept that requires systematic investigation and analysis.",
                "The study of {topic} involves examining various interconnected components.",
                "Researchers have identified {topic} as a key factor in understanding complex systems.",
                "Investigation into {topic} reveals important patterns and relationships.",
                "The analysis of {topic} contributes to our broader understanding of the field."
            ]
        else:
            return [
                "This is a response about {topic} from a language model.",
                "The concept of {topic} is important in many contexts.",
                "Understanding {topic} helps in various applications.",
                "Research on {topic} continues to evolve.",
                "{topic} represents an interesting area of study."
            ]
    
    def generate(self, prompt: str, max_length: int = 50, **kwargs) -> str:
        """Generate text response (mock implementation)"""
        # Simulate processing time
        time.sleep(random.uniform(0.01, 0.05))
        
        self.generation_count += 1
        
        # Extract topic from prompt for consistent responses
        prompt_words = prompt.lower().split()
        topic = 'the subject'
        for word in prompt_words:
            if len(word) > 4 and word.isalpha():
                topic = word
                break
        
        # Select response pattern based on prompt hash for consistency
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        pattern_idx = prompt_hash % len(self.response_patterns)
        
        response_template = self.response_patterns[pattern_idx]
        response = response_template.format(topic=topic)
        
        # Add some variation based on generation count
        if self.generation_count % 10 == 0:
            response += " This represents a milestone in our analysis."
        elif self.generation_count % 5 == 0:
            response += " Further investigation may be warranted."
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'config': self.config,
            'generation_count': self.generation_count,
            'vocab_size': self.tokenizer.vocab_size
        }


class DeterministicMockModel(MockModel):
    """Deterministic mock model for reproducible testing"""
    
    def __init__(self, model_name: str, seed: int = 42):
        super().__init__(model_name)
        self.seed = seed
        self.response_cache = {}
    
    def generate(self, prompt: str, max_length: int = 50, **kwargs) -> str:
        """Generate deterministic response based on prompt"""
        # Use cached response if available
        if prompt in self.response_cache:
            return self.response_cache[prompt]
        
        # Generate deterministic response
        prompt_hash = hashlib.md5(f"{prompt}_{self.seed}".encode()).hexdigest()
        
        # Use hash to select response pattern
        hash_int = int(prompt_hash[:8], 16)
        pattern_idx = hash_int % len(self.response_patterns)
        
        # Extract consistent topic
        words = prompt.lower().split()
        topic = next((word for word in words if len(word) > 4 and word.isalpha()), 'the subject')
        
        response = self.response_patterns[pattern_idx].format(topic=topic)
        
        # Add deterministic variation
        if hash_int % 100 < 20:  # 20% chance
            response += " This conclusion is based on extensive analysis."
        elif hash_int % 100 < 40:  # 20% chance  
            response += " These findings have significant implications."
        
        # Cache the response
        self.response_cache[prompt] = response
        self.generation_count += 1
        
        return response


class BehaviorVariantMockModel(MockModel):
    """Mock model with configurable behavioral variants"""
    
    def __init__(self, model_name: str, behavior_variant: str = 'standard'):
        super().__init__(model_name)
        self.behavior_variant = behavior_variant
        self._configure_behavior()
    
    def _configure_behavior(self):
        """Configure behavior based on variant"""
        if self.behavior_variant == 'verbose':
            # Longer, more detailed responses
            self.response_patterns = [
                "The comprehensive analysis of {topic} reveals a multifaceted phenomenon that requires careful examination across multiple dimensions and perspectives.",
                "In conducting a thorough investigation of {topic}, researchers have identified numerous interconnected factors that contribute to its complexity.",
                "The systematic study of {topic} demonstrates the importance of considering both theoretical frameworks and practical applications in understanding its implications."
            ]
        elif self.behavior_variant == 'concise':
            # Shorter, more direct responses
            self.response_patterns = [
                "{topic} is significant.",
                "Research shows {topic} matters.",
                "{topic} requires analysis.",
                "Studies examine {topic}."
            ]
        elif self.behavior_variant == 'technical':
            # More technical/formal responses
            self.response_patterns = [
                "The empirical investigation of {topic} employs rigorous methodological approaches to ensure validity and reliability.",
                "Quantitative analysis of {topic} indicates statistically significant correlations with established theoretical constructs.",
                "The systematic evaluation of {topic} utilizes standardized protocols and measurement instruments."
            ]
        elif self.behavior_variant == 'creative':
            # More creative/varied responses
            self.response_patterns = [
                "Imagine if {topic} could unlock the secrets of understanding complex phenomena in ways we never thought possible.",
                "The journey of exploring {topic} takes us through landscapes of knowledge filled with unexpected discoveries.",
                "Like a puzzle waiting to be solved, {topic} reveals its mysteries one piece at a time."
            ]
    
    def get_behavior_info(self) -> Dict[str, Any]:
        """Get behavior variant information"""
        return {
            'behavior_variant': self.behavior_variant,
            'pattern_count': len(self.response_patterns),
            'model_info': self.get_model_info()
        }


class CorrelatedMockModels:
    """Pair of mock models with configurable correlation"""
    
    def __init__(self, model1_name: str, model2_name: str, correlation: float = 0.8):
        """
        Create correlated mock models
        
        Args:
            model1_name: Name of first model
            model2_name: Name of second model  
            correlation: Correlation strength (0.0 = uncorrelated, 1.0 = identical)
        """
        self.model1 = DeterministicMockModel(model1_name, seed=42)
        self.model2 = DeterministicMockModel(model2_name, seed=43)
        self.correlation = correlation
        
        # Shared response patterns for correlated responses
        if correlation > 0.5:
            shared_patterns = [
                "The analysis of {topic} demonstrates important principles for understanding complex systems.",
                "Research into {topic} continues to provide valuable insights for the scientific community.",
                "The examination of {topic} reveals significant patterns that warrant further investigation."
            ]
            
            # Model 1 gets shared + unique patterns
            self.model1.response_patterns = shared_patterns + self.model1.response_patterns[:2]
            
            # Model 2 gets shared + slightly different patterns
            if correlation > 0.8:
                # High correlation - very similar patterns
                self.model2.response_patterns = shared_patterns + [
                    "The investigation of {topic} shows important principles for understanding complex systems.",
                    "Studies of {topic} continue to provide valuable insights for the research community."
                ]
            else:
                # Medium correlation - some shared patterns
                self.model2.response_patterns = shared_patterns[:2] + self.model2.response_patterns[:3]
    
    def generate_pair(self, prompt: str) -> tuple[str, str]:
        """Generate responses from both models"""
        response1 = self.model1.generate(prompt)
        
        if self.correlation > 0.9:
            # Very high correlation - almost identical responses
            response2 = response1.replace('analysis', 'examination').replace('demonstrates', 'shows')
        elif self.correlation > 0.7:
            # High correlation - similar responses with variations
            prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:4], 16)
            if prompt_hash % 10 < 7:  # 70% similar
                response2 = response1.replace('The', 'This').replace('important', 'significant')
            else:
                response2 = self.model2.generate(prompt)
        else:
            # Lower correlation - independent responses
            response2 = self.model2.generate(prompt)
        
        return response1, response2
    
    def get_correlation_info(self) -> Dict[str, Any]:
        """Get correlation information"""
        return {
            'model1': self.model1.model_name,
            'model2': self.model2.model_name,
            'correlation': self.correlation,
            'model1_patterns': len(self.model1.response_patterns),
            'model2_patterns': len(self.model2.response_patterns)
        }


# Factory functions for common test scenarios
def create_same_model_pair(model_name: str = "test_model") -> CorrelatedMockModels:
    """Create a pair of identical models for SAME testing"""
    return CorrelatedMockModels(model_name, model_name, correlation=1.0)


def create_different_model_pair(model1_name: str = "model_a", model2_name: str = "model_b") -> CorrelatedMockModels:
    """Create a pair of different models for DIFFERENT testing"""
    return CorrelatedMockModels(model1_name, model2_name, correlation=0.1)


def create_size_fraud_models() -> CorrelatedMockModels:
    """Create models for size fraud detection testing"""
    small_model = BehaviorVariantMockModel("pythia-70m", "concise")
    large_model = BehaviorVariantMockModel("pythia-160m", "verbose")
    
    # Manually create correlated pair with different sizes
    pair = CorrelatedMockModels("pythia-70m", "pythia-160m", correlation=0.3)
    pair.model1 = small_model
    pair.model2 = large_model
    
    return pair


def create_distillation_models() -> CorrelatedMockModels:
    """Create models for distillation detection testing"""
    teacher_model = BehaviorVariantMockModel("gpt2", "verbose")
    student_model = BehaviorVariantMockModel("distilgpt2", "concise")
    
    # High correlation but different verbosity
    pair = CorrelatedMockModels("gpt2", "distilgpt2", correlation=0.85)
    pair.model1 = teacher_model
    pair.model2 = student_model
    
    return pair


if __name__ == '__main__':
    # Example usage
    print("Testing mock models...")
    
    # Test basic model
    model = MockModel("test_gpt2")
    response = model.generate("What is machine learning?")
    print(f"Basic model response: {response}")
    
    # Test deterministic model
    det_model = DeterministicMockModel("test_deterministic")
    response1 = det_model.generate("What is AI?")
    response2 = det_model.generate("What is AI?")
    print(f"Deterministic responses match: {response1 == response2}")
    
    # Test correlated models
    pair = create_same_model_pair()
    resp_a, resp_b = pair.generate_pair("Explain neural networks")
    print(f"Correlated responses:\n  A: {resp_a}\n  B: {resp_b}")
    
    print("Mock models test completed.")