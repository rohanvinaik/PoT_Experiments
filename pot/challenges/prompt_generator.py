import hmac
import hashlib
import random
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    template: str
    category: str
    slots: Dict[str, List[str]]

class DeterministicPromptGenerator:
    """Prompt generator with unified API"""
    
    def __init__(self, master_key: bytes):
        self.master_key = master_key
        self.templates = self._init_templates()
    
    def _init_templates(self) -> List[PromptTemplate]:
        """Initialize prompt templates"""
        return [
            PromptTemplate(
                "Explain {topic} in {style}.",
                "factual",
                {
                    "topic": ["evolution", "photosynthesis", "gravity", "democracy"],
                    "style": ["simple terms", "one sentence", "technical detail"]
                }
            ),
            PromptTemplate(
                "What is {concept}?",
                "knowledge",
                {
                    "concept": ["machine learning", "quantum computing", "blockchain"]
                }
            ),
            PromptTemplate(
                "Translate to {language}: {phrase}",
                "translation",
                {
                    "language": ["French", "Spanish", "German"],
                    "phrase": ["hello", "good morning", "thank you"]
                }
            )
        ]
    
    def _sample_slots(self, rng: random.Random, template: PromptTemplate) -> Dict[str, str]:
        """Sample slot values"""
        result = {}
        for slot, options in template.slots.items():
            result[slot] = rng.choice(options)
        return result
    
    def generate_challenges(self,
                           ref_model_id: str,
                           cand_model_id: str,
                           *,
                           n: int,
                           namespace: str,
                           seed: int) -> List[Dict[str, Any]]:
        """Generate challenges with unified signature"""
        
        # Create deterministic RNG
        seed_data = f"{namespace}:{seed}:{ref_model_id}:{cand_model_id}".encode()
        seed_bytes = hmac.new(self.master_key, seed_data, hashlib.sha256).digest()
        rng = random.Random(int.from_bytes(seed_bytes[:8], 'big'))
        
        challenges = []
        
        for i in range(n):
            # Select template
            template = rng.choice(self.templates)
            
            # Fill slots
            slots = self._sample_slots(rng, template)
            prompt = template.template.format(**slots)
            
            challenges.append({
                "prompt": prompt,
                "family": template.category,
                "idx": i,
                "ref_model": ref_model_id,
                "cand_model": cand_model_id,
                "namespace": namespace
            })
        
        return challenges
    
    # Backward compatibility wrapper
    def __call__(self) -> str:
        """Generate single prompt for backward compatibility"""
        challenges = self.generate_challenges(
            "default", "default",
            n=1, namespace="default", seed=random.randint(0, 2**32)
        )
        return challenges[0]["prompt"] if challenges else ""
    

# Factory function for backward compatibility
def make_prompt_generator(master_key: bytes, namespace: str = "default"):
    """Create prompt generator"""
    gen = DeterministicPromptGenerator(master_key)
    
    # Return a callable that generates single prompts
    def prompt_fn():
        return gen()
    
    # Attach the full generator for access to all methods
    prompt_fn.generator = gen
    
    return prompt_fn


def create_prompt_challenges(master_key: bytes, 
                            family: str,
                            params: Dict[str, Any],
                            n_challenges: int) -> List[Dict[str, Any]]:
    """Create deterministic prompt challenges for POT verification (legacy API)"""
    
    generator = DeterministicPromptGenerator(master_key)
    namespace = f"{family}:{json.dumps(params, sort_keys=True)}"
    
    # Extract model IDs from params if available
    ref_model = params.get("ref_model", "default")
    cand_model = params.get("cand_model", "default") 
    seed = params.get("seed", 42)
    
    challenges = generator.generate_challenges(
        ref_model, cand_model,
        n=n_challenges,
        namespace=namespace,
        seed=seed
    )
    
    # Convert to legacy format
    legacy_challenges = []
    for challenge in challenges:
        legacy_challenge = {
            "id": f"{family}_{challenge['idx']:06d}",
            "type": "prompt",
            "content": challenge["prompt"],
            "metadata": {
                "family": challenge["family"],
                "index": challenge["idx"],
                "namespace": challenge["namespace"]
            }
        }
        legacy_challenges.append(legacy_challenge)
    
    return legacy_challenges