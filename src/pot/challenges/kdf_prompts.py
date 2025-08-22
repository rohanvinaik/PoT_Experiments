"""
KDF-based Prompt Generation for Enhanced Verification

Provides deterministic yet unpredictable prompt generation using
cryptographic key derivation functions.
"""

import hashlib
import json
from typing import Callable, List, Optional
import numpy as np

class KDFPromptGenerator:
    """Generate prompts using KDF for reproducibility and unpredictability"""
    
    def __init__(self, prf_key: bytes, namespace: str = "enhanced:v2"):
        """
        Initialize prompt generator.
        
        Args:
            prf_key: PRF key for deterministic generation
            namespace: Namespace for domain separation
        """
        self.prf_key = prf_key
        self.namespace = namespace
        self.counter = 0
        
        # Template prompts for generation
        self.templates = [
            "Explain the concept of {topic} in simple terms.",
            "What are the main differences between {item1} and {item2}?",
            "Describe the process of {process} step by step.",
            "How does {system} work and what are its key components?",
            "What are the advantages and disadvantages of {technology}?",
            "Provide a brief overview of {subject} and its importance.",
            "Compare and contrast {concept1} with {concept2}.",
            "What are the best practices for {activity}?",
            "Explain why {phenomenon} occurs and its implications.",
            "How can {problem} be solved effectively?"
        ]
        
        self.topics = [
            "machine learning", "quantum computing", "blockchain", "neural networks",
            "cryptography", "distributed systems", "cloud computing", "data structures",
            "algorithms", "artificial intelligence", "cybersecurity", "databases"
        ]
        
        self.items = [
            ("TCP", "UDP"), ("Python", "Java"), ("REST", "GraphQL"),
            ("SQL", "NoSQL"), ("CPU", "GPU"), ("VM", "Container"),
            ("Sync", "Async"), ("Stack", "Queue"), ("BFS", "DFS")
        ]
    
    def _kdf(self, input_data: bytes) -> bytes:
        """Apply KDF to generate deterministic output"""
        return hashlib.sha256(
            self.prf_key + 
            self.namespace.encode() + 
            input_data
        ).digest()
    
    def generate_prompt(self) -> str:
        """Generate next prompt deterministically"""
        # Get deterministic random state
        seed_bytes = self._kdf(f"prompt:{self.counter}".encode())
        seed = int.from_bytes(seed_bytes[:4], 'big')
        rng = np.random.RandomState(seed)
        
        # Select template
        template = rng.choice(self.templates)
        
        # Fill template based on required variables
        if "{topic}" in template or "{subject}" in template or "{technology}" in template:
            topic = rng.choice(self.topics)
            prompt = template.format(
                topic=topic, 
                subject=topic, 
                technology=topic,
                system=topic,
                activity=f"implementing {topic}",
                phenomenon=f"{topic} performance",
                problem=f"scaling {topic}"
            )
        elif "{item1}" in template:
            pair_idx = rng.choice(len(self.items))
            item1, item2 = self.items[pair_idx]
            prompt = template.format(item1=item1, item2=item2)
        elif "{concept1}" in template:
            concepts = rng.choice(self.topics, size=2, replace=False)
            prompt = template.format(concept1=concepts[0], concept2=concepts[1])
        elif "{process}" in template:
            process = rng.choice([
                "training a neural network",
                "deploying a web application",
                "optimizing database queries",
                "implementing authentication",
                "debugging production issues"
            ])
            prompt = template.format(process=process)
        else:
            # Generic fill
            prompt = template
        
        self.counter += 1
        return prompt
    
    def __call__(self) -> str:
        """Make generator callable"""
        return self.generate_prompt()

def make_prompt_generator(prf_key: bytes, 
                         namespace: str = "enhanced:v2") -> Callable[[], str]:
    """
    Create a prompt generator function.
    
    Args:
        prf_key: PRF key for deterministic generation
        namespace: Namespace for domain separation
        
    Returns:
        Callable that generates prompts
    """
    generator = KDFPromptGenerator(prf_key, namespace)
    return generator