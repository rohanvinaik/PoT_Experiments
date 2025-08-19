#!/usr/bin/env python3
"""
KDF-based Deterministic Prompt Generator for LLM Verification
Maps cryptographic seeds to reproducible text prompts
"""

import hashlib
import hmac
from typing import List, Tuple, Optional
import struct

class KDFPromptGenerator:
    """Generate deterministic prompts from KDF seeds for LLM verification."""
    
    def __init__(self, master_key: str = "pot_verification_2024", namespace: str = "llm"):
        """
        Initialize the KDF prompt generator.
        
        Args:
            master_key: Master key for HMAC
            namespace: Namespace for this generator (e.g., 'llm', 'vision', 'semantic')
        """
        self.master_key = master_key.encode()
        self.namespace = namespace.encode()
        
        # Template banks - deterministic prompt structures
        self.templates = [
            "Complete the following: The {noun} is {adjective} because",
            "Explain why {noun} exhibits {property}:",
            "The relationship between {concept1} and {concept2} is",
            "Given that {premise}, we can conclude",
            "Analysis of {subject} reveals that",
            "The primary characteristic of {noun} is",
            "When {condition}, the result is",
            "The difference between {item1} and {item2} is",
            "{noun} can be described as {adjective} due to",
            "The process of {action} involves"
        ]
        
        # Filler word banks - deterministic vocabulary
        self.nouns = [
            "system", "model", "structure", "process", "mechanism",
            "framework", "algorithm", "network", "pattern", "configuration"
        ]
        
        self.adjectives = [
            "complex", "efficient", "robust", "adaptive", "optimal",
            "dynamic", "stable", "scalable", "reliable", "consistent"
        ]
        
        self.properties = [
            "resilience", "efficiency", "adaptability", "stability",
            "scalability", "robustness", "consistency", "reliability"
        ]
        
        self.concepts = [
            "entropy", "optimization", "convergence", "distribution",
            "variance", "correlation", "inference", "prediction"
        ]
        
        self.actions = [
            "optimization", "transformation", "computation", "evaluation",
            "classification", "regression", "clustering", "encoding"
        ]
        
        self.conditions = [
            "the input is normalized", "the system converges",
            "the threshold is exceeded", "the model is trained",
            "the data is processed", "the parameters are optimized"
        ]
        
        self.premises = [
            "all inputs are valid", "the system is stable",
            "the model has converged", "the data is normalized",
            "the process is complete", "the threshold is met"
        ]

    def _derive_seed(self, index: int) -> bytes:
        """
        Derive a deterministic seed using HMAC.
        
        Args:
            index: Challenge index
            
        Returns:
            32-byte seed
        """
        message = self.namespace + struct.pack('>I', index)
        return hmac.new(self.master_key, message, hashlib.sha256).digest()
    
    def _seed_to_indices(self, seed: bytes, num_indices: int) -> List[int]:
        """
        Convert seed bytes to indices for selection.
        
        Args:
            seed: Seed bytes
            num_indices: Number of indices to generate
            
        Returns:
            List of indices
        """
        indices = []
        for i in range(num_indices):
            # Use 2 bytes per index for good distribution
            if i * 2 + 1 < len(seed):
                idx = struct.unpack('>H', seed[i*2:i*2+2])[0]
                indices.append(idx)
            else:
                # If we run out of seed bytes, hash to get more
                extended_seed = hashlib.sha256(seed + struct.pack('>I', i)).digest()
                idx = struct.unpack('>H', extended_seed[:2])[0]
                indices.append(idx)
        return indices
    
    def generate_prompt(self, index: int) -> str:
        """
        Generate a deterministic prompt from an index.
        
        Args:
            index: Challenge index
            
        Returns:
            Deterministic prompt string
        """
        # Derive seed
        seed = self._derive_seed(index)
        
        # Get indices for selections
        indices = self._seed_to_indices(seed, 10)
        
        # Select template
        template_idx = indices[0] % len(self.templates)
        template = self.templates[template_idx]
        
        # Select fillers based on template requirements
        replacements = {}
        
        if '{noun}' in template:
            replacements['noun'] = self.nouns[indices[1] % len(self.nouns)]
        if '{adjective}' in template:
            replacements['adjective'] = self.adjectives[indices[2] % len(self.adjectives)]
        if '{property}' in template:
            replacements['property'] = self.properties[indices[3] % len(self.properties)]
        if '{concept1}' in template:
            replacements['concept1'] = self.concepts[indices[4] % len(self.concepts)]
        if '{concept2}' in template:
            replacements['concept2'] = self.concepts[indices[5] % len(self.concepts)]
        if '{subject}' in template:
            replacements['subject'] = self.nouns[indices[6] % len(self.nouns)]
        if '{condition}' in template:
            replacements['condition'] = self.conditions[indices[7] % len(self.conditions)]
        if '{item1}' in template:
            replacements['item1'] = self.nouns[indices[8] % len(self.nouns)]
        if '{item2}' in template:
            replacements['item2'] = self.nouns[(indices[8] + 1) % len(self.nouns)]
        if '{action}' in template:
            replacements['action'] = self.actions[indices[9] % len(self.actions)]
        if '{premise}' in template:
            replacements['premise'] = self.premises[indices[9] % len(self.premises)]
        
        # Generate final prompt
        prompt = template.format(**replacements)
        return prompt
    
    def generate_batch(self, start_idx: int, count: int) -> List[str]:
        """
        Generate a batch of deterministic prompts.
        
        Args:
            start_idx: Starting index
            count: Number of prompts to generate
            
        Returns:
            List of prompts
        """
        return [self.generate_prompt(start_idx + i) for i in range(count)]
    
    def generate_with_metadata(self, index: int) -> dict:
        """
        Generate prompt with metadata for verification.
        
        Args:
            index: Challenge index
            
        Returns:
            Dictionary with prompt and metadata
        """
        seed = self._derive_seed(index)
        prompt = self.generate_prompt(index)
        
        return {
            'index': index,
            'prompt': prompt,
            'seed_hash': hashlib.sha256(seed).hexdigest()[:16],
            'namespace': self.namespace.decode(),
            'length': len(prompt),
            'deterministic': True
        }


class SeededParaphraser:
    """Generate seeded paraphrases for additional variation."""
    
    def __init__(self, seed: bytes):
        """Initialize with a seed for deterministic paraphrasing."""
        self.seed = seed
        self.rng_state = int.from_bytes(seed[:4], 'big')
    
    def _next_random(self) -> float:
        """Simple deterministic PRNG."""
        self.rng_state = (self.rng_state * 1103515245 + 12345) & 0x7fffffff
        return self.rng_state / 0x7fffffff
    
    def paraphrase(self, text: str) -> str:
        """
        Apply deterministic paraphrasing to text.
        
        Args:
            text: Original text
            
        Returns:
            Paraphrased text
        """
        # Simple deterministic transformations
        transformations = [
            ("is", ["is", "remains", "appears to be", "can be described as"]),
            ("because", ["because", "since", "as", "due to the fact that"]),
            ("reveals", ["reveals", "shows", "demonstrates", "indicates"]),
            ("involves", ["involves", "includes", "comprises", "consists of"]),
            ("The", ["The", "This", "That", "A"]),
        ]
        
        result = text
        for original, alternatives in transformations:
            if original in result:
                # Deterministically choose alternative
                idx = int(self._next_random() * len(alternatives))
                result = result.replace(original, alternatives[idx], 1)
        
        return result


def create_challenge_set(
    num_challenges: int = 100,
    master_key: str = "pot_verification_2024",
    namespace: str = "llm",
    use_paraphrasing: bool = False
) -> List[dict]:
    """
    Create a complete set of KDF-based challenges.
    
    Args:
        num_challenges: Number of challenges to generate
        master_key: Master key for KDF
        namespace: Namespace for challenges
        use_paraphrasing: Whether to apply seeded paraphrasing
        
    Returns:
        List of challenge dictionaries
    """
    generator = KDFPromptGenerator(master_key, namespace)
    challenges = []
    
    for i in range(num_challenges):
        challenge = generator.generate_with_metadata(i)
        
        if use_paraphrasing:
            # Apply deterministic paraphrasing
            seed = generator._derive_seed(i)
            paraphraser = SeededParaphraser(seed)
            challenge['paraphrased'] = paraphraser.paraphrase(challenge['prompt'])
        
        challenges.append(challenge)
    
    return challenges


# Example usage and testing
if __name__ == "__main__":
    print("KDF-Based Prompt Generator Test")
    print("=" * 60)
    
    # Create generator
    generator = KDFPromptGenerator()
    
    # Generate some examples
    print("\nDeterministic Prompt Examples:")
    print("-" * 40)
    
    for i in range(5):
        prompt = generator.generate_prompt(i)
        print(f"{i}: {prompt}")
    
    # Verify determinism
    print("\nDeterminism Verification:")
    print("-" * 40)
    
    prompt1 = generator.generate_prompt(42)
    prompt2 = generator.generate_prompt(42)
    print(f"Same index (42) generates same prompt: {prompt1 == prompt2}")
    print(f"Prompt: {prompt1}")
    
    # Generate with metadata
    print("\nPrompt with Metadata:")
    print("-" * 40)
    
    challenge = generator.generate_with_metadata(0)
    for key, value in challenge.items():
        print(f"  {key}: {value}")
    
    # Test paraphrasing
    print("\nSeeded Paraphrasing Test:")
    print("-" * 40)
    
    seed = generator._derive_seed(0)
    paraphraser = SeededParaphraser(seed)
    
    original = "The system is efficient because it optimizes resources"
    paraphrased = paraphraser.paraphrase(original)
    print(f"Original:    {original}")
    print(f"Paraphrased: {paraphrased}")
    
    # Create challenge set
    print("\nChallenge Set Generation:")
    print("-" * 40)
    
    challenges = create_challenge_set(num_challenges=10, use_paraphrasing=True)
    print(f"Generated {len(challenges)} challenges")
    
    for i, challenge in enumerate(challenges[:3]):
        print(f"\nChallenge {i}:")
        print(f"  Prompt: {challenge['prompt']}")
        if 'paraphrased' in challenge:
            print(f"  Alt:    {challenge['paraphrased']}")
        print(f"  Hash:   {challenge['seed_hash']}")