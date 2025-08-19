import hmac
import hashlib
import random
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    template: str  # e.g., "What is {concept}? {filler}"
    slots: List[str]  # e.g., ["concept", "filler"]
    category: str  # e.g., "factual", "reasoning", "creative"

class DeterministicPromptGenerator:
    def __init__(self, master_key: bytes):
        self.master_key = master_key
        self.templates = self._init_template_bank()
        self.filler_banks = self._init_filler_banks()
        self.concept_banks = self._init_concept_banks()
    
    def derive_seed(self, namespace: str, idx: int) -> bytes:
        """Derive deterministic seed from namespace and index"""
        message = f"{namespace}||{idx}".encode('utf-8')
        return hmac.new(self.master_key, message, hashlib.sha256).digest()
    
    def seed_to_prompt(self, seed: bytes, max_length: int = 100) -> str:
        """Convert seed bytes to deterministic prompt"""
        # Use seed to initialize deterministic RNG
        rng = random.Random(int.from_bytes(seed[:8], 'big'))
        
        # Select template deterministically
        template = rng.choice(self.templates)
        
        # Fill template slots deterministically
        filled_slots = {}
        for slot in template.slots:
            if slot == "filler":
                filled_slots[slot] = self._select_filler(rng, template.category)
            elif slot == "concept":
                filled_slots[slot] = self._select_concept(rng, template.category)
            else:
                filled_slots[slot] = self._generate_slot_content(rng, slot)
        
        # Generate final prompt
        prompt = template.template.format(**filled_slots)
        
        # Truncate if needed while keeping it valid
        if len(prompt) > max_length:
            prompt = self._smart_truncate(prompt, max_length)
        
        return prompt
    
    def batch_generate(self, namespace: str, start_idx: int, count: int) -> List[str]:
        """Generate batch of deterministic prompts"""
        prompts = []
        for i in range(start_idx, start_idx + count):
            seed = self.derive_seed(namespace, i)
            prompt = self.seed_to_prompt(seed)
            prompts.append(prompt)
        return prompts
    
    def _init_template_bank(self) -> List[PromptTemplate]:
        """Initialize diverse prompt templates"""
        return [
            # Factual templates
            PromptTemplate(
                "What is {concept}? {filler}",
                ["concept", "filler"],
                "factual"
            ),
            PromptTemplate(
                "Explain {concept} in simple terms. {filler}",
                ["concept", "filler"],
                "factual"
            ),
            PromptTemplate(
                "Define {concept}. {filler}",
                ["concept", "filler"],
                "factual"
            ),
            
            # Reasoning templates
            PromptTemplate(
                "If {premise}, then what about {question}? {filler}",
                ["premise", "question", "filler"],
                "reasoning"
            ),
            PromptTemplate(
                "Compare {item1} and {item2}. {filler}",
                ["item1", "item2", "filler"],
                "reasoning"
            ),
            
            # Creative templates
            PromptTemplate(
                "Write a short {genre} about {topic}. {filler}",
                ["genre", "topic", "filler"],
                "creative"
            ),
            PromptTemplate(
                "Describe {scene} vividly. {filler}",
                ["scene", "filler"],
                "creative"
            ),
            
            # Instruction templates
            PromptTemplate(
                "List {number} ways to {action}. {filler}",
                ["number", "action", "filler"],
                "instruction"
            ),
            PromptTemplate(
                "How do you {task}? {filler}",
                ["task", "filler"],
                "instruction"
            ),
            
            # Analysis templates
            PromptTemplate(
                "What are the implications of {statement}? {filler}",
                ["statement", "filler"],
                "analysis"
            ),
            PromptTemplate(
                "Analyze the following: {content}. {filler}",
                ["content", "filler"],
                "analysis"
            )
        ]
    
    def _init_filler_banks(self) -> Dict[str, List[str]]:
        """Initialize category-specific filler phrases"""
        return {
            "factual": [
                "Please be specific.",
                "Include key details.",
                "Be concise.",
                "Focus on main points.",
                "Provide clear information."
            ],
            "reasoning": [
                "Show your logic.",
                "Explain step by step.",
                "Consider all aspects.",
                "Be thorough.",
                "Think carefully."
            ],
            "creative": [
                "Be imaginative.",
                "Keep it engaging.",
                "Be original.",
                "Make it interesting.",
                "Be descriptive."
            ],
            "instruction": [
                "Be practical.",
                "Keep it simple.",
                "Be clear.",
                "Make it actionable.",
                "Be specific."
            ],
            "analysis": [
                "Be comprehensive.",
                "Consider context.",
                "Be objective.",
                "Think critically.",
                "Be detailed."
            ]
        }
    
    def _init_concept_banks(self) -> Dict[str, List[str]]:
        """Initialize concept/content banks for different categories"""
        return {
            "factual": [
                "photosynthesis", "gravity", "democracy", "evolution",
                "artificial intelligence", "quantum mechanics", "climate change",
                "DNA", "black holes", "renewable energy"
            ],
            "reasoning": [
                "correlation and causation", "risk and reward",
                "efficiency and effectiveness", "theory and practice",
                "individual and collective", "short-term and long-term"
            ],
            "creative": [
                "story", "poem", "dialogue", "description",
                "narrative", "scene", "character", "setting"
            ],
            "instruction": [
                "solve a problem", "learn a skill", "improve efficiency",
                "save money", "stay organized", "manage time"
            ],
            "analysis": [
                "technological advancement", "social change",
                "economic growth", "environmental impact",
                "cultural shift", "policy change"
            ]
        }
    
    def _select_filler(self, rng: random.Random, category: str) -> str:
        """Deterministically select filler based on category"""
        fillers = self.filler_banks.get(category, self.filler_banks["factual"])
        return rng.choice(fillers)
    
    def _select_concept(self, rng: random.Random, category: str) -> str:
        """Deterministically select concept based on category"""
        concepts = self.concept_banks.get(category, self.concept_banks["factual"])
        return rng.choice(concepts)
    
    def _generate_slot_content(self, rng: random.Random, slot: str) -> str:
        """Generate content for custom slots"""
        slot_generators = {
            "number": lambda: str(rng.randint(3, 7)),
            "genre": lambda: rng.choice(["story", "tale", "account", "narrative"]),
            "topic": lambda: rng.choice(["adventure", "discovery", "friendship", "challenge"]),
            "scene": lambda: rng.choice(["sunset", "city street", "forest", "ocean"]),
            "premise": lambda: rng.choice([
                "technology advances", "resources are limited",
                "time is valuable", "change is constant"
            ]),
            "question": lambda: rng.choice([
                "the future", "society", "individuals", "progress"
            ]),
            "item1": lambda: rng.choice(["efficiency", "quality", "speed", "cost"]),
            "item2": lambda: rng.choice(["reliability", "flexibility", "simplicity", "scalability"]),
            "action": lambda: rng.choice(["improve", "optimize", "organize", "learn"]),
            "task": lambda: rng.choice([
                "solve complex problems", "make decisions",
                "manage resources", "build systems"
            ]),
            "statement": lambda: rng.choice([
                "automation increases", "data grows exponentially",
                "connectivity improves", "complexity rises"
            ]),
            "content": lambda: rng.choice([
                "current trends", "recent developments",
                "emerging patterns", "key indicators"
            ])
        }
        
        generator = slot_generators.get(slot, lambda: f"[{slot}]")
        return generator()
    
    def _smart_truncate(self, prompt: str, max_length: int) -> str:
        """Truncate prompt intelligently at sentence boundary"""
        if len(prompt) <= max_length:
            return prompt
        
        # Try to cut at sentence end
        truncated = prompt[:max_length]
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclaim = truncated.rfind('!')
        
        cut_point = max(last_period, last_question, last_exclaim)
        if cut_point > max_length * 0.7:  # If we found a sentence end in last 30%
            return truncated[:cut_point + 1]
        
        # Otherwise cut at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:
            return truncated[:last_space] + "."
        
        return truncated[:max_length-3] + "..."
    
    def generate_paraphrased_set(self, base_seed: bytes, count: int = 5) -> List[str]:
        """Generate deterministic paraphrases of same semantic content"""
        rng = random.Random(int.from_bytes(base_seed[:8], 'big'))
        
        # Base semantic content
        base_concept = rng.choice(list(self.concept_banks["factual"]))
        
        paraphrase_templates = [
            f"What is {base_concept}?",
            f"Explain {base_concept}.",
            f"Tell me about {base_concept}.",
            f"Describe {base_concept}.",
            f"Define {base_concept}.",
            f"What do you know about {base_concept}?",
            f"Can you explain {base_concept}?",
            f"Provide information on {base_concept}."
        ]
        
        # Select deterministic subset
        selected = []
        indices = list(range(len(paraphrase_templates)))
        rng.shuffle(indices)
        
        for i in indices[:min(count, len(paraphrase_templates))]:
            template = paraphrase_templates[i]
            # Add deterministic filler
            filler = rng.choice(self.filler_banks["factual"])
            selected.append(f"{template} {filler}")
        
        return selected


def create_prompt_challenges(master_key: bytes, 
                            family: str,
                            params: Dict[str, Any],
                            n_challenges: int) -> List[Dict[str, Any]]:
    """Create deterministic prompt challenges for POT verification"""
    
    generator = DeterministicPromptGenerator(master_key)
    namespace = f"{family}:{json.dumps(params, sort_keys=True)}"
    
    challenges = []
    for i in range(n_challenges):
        seed = generator.derive_seed(namespace, i)
        prompt = generator.seed_to_prompt(seed, max_length=100)
        
        challenge = {
            "id": f"{family}_{i:06d}",
            "type": "prompt",
            "content": prompt,
            "seed": seed.hex(),
            "metadata": {
                "family": family,
                "index": i,
                "namespace": namespace,
                "length": len(prompt)
            }
        }
        challenges.append(challenge)
    
    return challenges