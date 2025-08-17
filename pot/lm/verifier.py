"""
Language Model Verifier for Proof-of-Training
Implements verification protocol from paper Section 3
"""

import torch
import numpy as np
import difflib
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
import time

from transformers import AutoTokenizer
import torch.nn as nn
from typing import Union

from ..core.stats import empirical_bernstein_bound, t_statistic
from ..core.sequential import (
    SequentialTester, SPRTResult, sequential_verify,
    SequentialState, welford_update, compute_empirical_variance
)
from ..core.challenge import generate_challenges, ChallengeConfig
from ..core.fingerprint import fingerprint_run, FingerprintConfig, FingerprintResult, compare_jacobian_sketches
from ..core.canonicalize import canonicalize_text, canonicalize_logits
from pot.security.token_space_normalizer import TokenSpaceNormalizer
from .fuzzy_hash import FuzzyHasher
from .models import LM
from .lm_config import LMVerifierConfig
from .template_challenges import TemplateChallenger as NewTemplateChallenger, ChallengeEvaluator
from .sequential_tester import SequentialTester as NewSequentialTester, SequentialVerificationSession

# Semantic verification imports (optional)
try:
    from ..semantic.library import ConceptLibrary
    from ..semantic.match import SemanticMatcher
    from ..semantic.utils import extract_embeddings_from_logits, normalize_embeddings
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False


@dataclass
class LMVerificationResult:
    """Result of language model verification"""
    accepted: bool
    distance: float
    confidence_radius: float
    n_challenges: int
    fuzzy_similarity: float
    time_elapsed: float
    fingerprint: Optional[FingerprintResult]  # Behavioral fingerprint
    fingerprint_match: Optional[float]  # Similarity score if reference fingerprint exists
    sequential_result: Optional[SPRTResult]  # Sequential verification result with trajectory
    semantic_score: Optional[float] = None  # Semantic verification score
    combined_score: Optional[float] = None  # Combined distance and semantic score
    metadata: Dict[str, Any] = None


class TemplateChallenger:
    """Generate template-based challenges for language models"""
    
    def __init__(self):
        self.templates = {
            'completion': [
                "The capital of {country} is [MASK]",
                "Water freezes at [MASK] degrees {unit}",
                "The formula for the area of a {shape} is [MASK]",
                "{number} plus {number2} equals [MASK]",
                "The {ordinal} president of {country} was [MASK]"
            ],
            'qa': [
                "Q: What is {concept}? A:",
                "Q: How does {process} work? A:",
                "Q: Why is {phenomenon} important? A:",
                "Q: When did {event} happen? A:",
                "Q: Where is {location} located? A:"
            ],
            'reasoning': [
                "If {premise}, then [MASK]",
                "Given that {fact1} and {fact2}, we can conclude [MASK]",
                "The opposite of {concept} is [MASK]",
                "Compare {item1} and {item2}: [MASK]",
                "{cause} leads to [MASK]"
            ],
            'translation': [
                "Translate to {language}: {text}",
                "'{phrase}' in {language} means [MASK]",
                "The {language} word for '{word}' is [MASK]"
            ]
        }
        
        self.slot_values = {
            'country': ['France', 'Japan', 'Brazil', 'Canada', 'Egypt'],
            'unit': ['Celsius', 'Fahrenheit', 'Kelvin'],
            'shape': ['circle', 'square', 'triangle', 'rectangle'],
            'number': ['two', 'five', 'ten', 'twenty'],
            'number2': ['three', 'seven', 'eight', 'fifteen'],
            'ordinal': ['first', 'second', 'third', 'fourth'],
            'concept': ['gravity', 'democracy', 'evolution', 'entropy', 'recursion'],
            'process': ['photosynthesis', 'digestion', 'combustion', 'fermentation'],
            'phenomenon': ['climate change', 'quantum entanglement', 'natural selection'],
            'event': ['World War II', 'the Renaissance', 'the Industrial Revolution'],
            'location': ['Mount Everest', 'the Amazon River', 'the Sahara Desert'],
            'premise': ['all birds can fly', 'it is raining', 'the sun rises in the east'],
            'fact1': ['water is H2O', 'plants need sunlight', 'metals conduct electricity'],
            'fact2': ['ice is frozen water', 'photosynthesis produces oxygen', 'copper is a metal'],
            'item1': ['apples', 'cats', 'summer', 'books'],
            'item2': ['oranges', 'dogs', 'winter', 'movies'],
            'cause': ['Heavy rain', 'Increased temperature', 'Strong wind'],
            'language': ['French', 'Spanish', 'German', 'Italian', 'Japanese'],
            'text': ['Hello world', 'Good morning', 'Thank you'],
            'phrase': ['goodbye', 'please', 'excuse me'],
            'word': ['cat', 'house', 'water', 'love']
        }
    
    def generate_challenges(self, challenge_type: str, num: int, difficulty: str = 'medium') -> List[Dict[str, Any]]:
        """Generate challenges of specified type"""
        import random
        
        if challenge_type not in self.templates and challenge_type != 'mixed':
            challenge_type = 'mixed'
        
        challenges = []
        
        if challenge_type == 'mixed':
            # Mix all types
            all_templates = []
            for templates in self.templates.values():
                all_templates.extend(templates)
            selected_templates = random.choices(all_templates, k=num)
        else:
            selected_templates = random.choices(self.templates[challenge_type], k=num)
        
        for i, template in enumerate(selected_templates):
            # Fill in slots
            filled_template = template
            slots_in_template = {}
            
            import re
            slot_pattern = re.compile(r'\{(\w+)\}')
            slots = slot_pattern.findall(template)
            
            for slot in slots:
                if slot in self.slot_values:
                    value = random.choice(self.slot_values[slot])
                    slots_in_template[slot] = value
                    filled_template = filled_template.replace(f'{{{slot}}}', value)
            
            challenges.append({
                'template': template,
                'filled_template': filled_template,
                'slot_values': slots_in_template,
                'difficulty': difficulty,
                'type': challenge_type,
                'index': i
            })
        
        return challenges


class DistanceMetrics:
    """Distance metrics for comparing model outputs"""
    
    def __init__(self):
        self.eps = 1e-8  # For numerical stability
    
    def compute_distance(self, output1: Dict[str, Any], output2: Dict[str, Any],
                        metric: str = 'combined') -> float:
        """
        Compute distance between two model outputs.
        
        Args:
            output1: First model output dictionary
            output2: Second model output dictionary
            metric: 'edit', 'logits_kl', 'embedding_cosine', 'combined'
        
        Returns:
            Distance score (0 = identical, higher = more different)
        """
        if metric == 'edit':
            text1 = output1.get('generated_text', [''])[0] if isinstance(output1.get('generated_text'), list) else output1.get('generated_text', '')
            text2 = output2.get('generated_text', [''])[0] if isinstance(output2.get('generated_text'), list) else output2.get('generated_text', '')
            return self.edit_distance(text1, text2)
        
        elif metric == 'logits_kl':
            logits1 = output1.get('logits')
            logits2 = output2.get('logits')
            if logits1 is not None and logits2 is not None:
                return self.kl_divergence(logits1, logits2)
            return 1.0  # Max distance if logits unavailable
        
        elif metric == 'embedding_cosine':
            emb1 = output1.get('hidden_states')
            emb2 = output2.get('hidden_states')
            if emb1 is not None and emb2 is not None:
                return self.cosine_distance(emb1, emb2)
            return 1.0  # Max distance if embeddings unavailable
        
        elif metric == 'combined':
            return self.combined_distance(output1, output2)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def edit_distance(self, text1: str, text2: str, normalized: bool = True) -> float:
        """Compute Levenshtein distance between texts."""
        import difflib
        
        if not text1 and not text2:
            return 0.0
        if not text1 or not text2:
            return 1.0
        
        # Use SequenceMatcher for normalized edit distance
        ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
        distance = 1.0 - ratio  # Convert similarity to distance
        
        if not normalized:
            # Approximate unnormalized distance
            max_len = max(len(text1), len(text2))
            distance = distance * max_len
        
        return distance
    
    def kl_divergence(self, logits1: torch.Tensor, logits2: torch.Tensor) -> float:
        """Compute KL divergence between logit distributions."""
        if logits1 is None or logits2 is None:
            return 1.0
        
        # Handle list of tensors (from generation)
        if isinstance(logits1, (list, tuple)):
            logits1 = torch.stack(logits1) if len(logits1) > 0 else None
        if isinstance(logits2, (list, tuple)):
            logits2 = torch.stack(logits2) if len(logits2) > 0 else None
        
        if logits1 is None or logits2 is None:
            return 1.0
        
        # Ensure same shape
        min_len = min(logits1.shape[0], logits2.shape[0])
        logits1 = logits1[:min_len]
        logits2 = logits2[:min_len]
        
        # Convert to probabilities with numerical stability
        p = torch.nn.functional.softmax(logits1.float(), dim=-1)
        q = torch.nn.functional.softmax(logits2.float(), dim=-1)
        
        # Add small epsilon for numerical stability
        p = p + self.eps
        q = q + self.eps
        
        # Compute KL divergence
        kl = (p * (p.log() - q.log())).sum(dim=-1).mean()
        
        # Normalize to [0, 1] range (KL can be unbounded)
        # Using sigmoid-like transformation
        normalized_kl = 2.0 / (1.0 + torch.exp(-kl)) - 1.0
        
        return float(normalized_kl.item())
    
    def cosine_distance(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> float:
        """Compute cosine distance between embeddings."""
        if embeddings1 is None or embeddings2 is None:
            return 1.0
        
        # Handle different formats
        if isinstance(embeddings1, (list, tuple)):
            if len(embeddings1) > 0 and isinstance(embeddings1[0], torch.Tensor):
                # List of tensors - take mean of last hidden states
                embeddings1 = torch.stack([e.mean(dim=0) if e.dim() > 1 else e for e in embeddings1]).mean(dim=0)
        
        if isinstance(embeddings2, (list, tuple)):
            if len(embeddings2) > 0 and isinstance(embeddings2[0], torch.Tensor):
                embeddings2 = torch.stack([e.mean(dim=0) if e.dim() > 1 else e for e in embeddings2]).mean(dim=0)
        
        # Flatten if needed
        if embeddings1.dim() > 1:
            embeddings1 = embeddings1.flatten()
        if embeddings2.dim() > 1:
            embeddings2 = embeddings2.flatten()
        
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            embeddings1.unsqueeze(0),
            embeddings2.unsqueeze(0)
        )
        
        # Convert to distance (1 - similarity)
        distance = 1.0 - cos_sim.item()
        
        return float(distance)
    
    def combined_distance(self, output1: Dict[str, Any], output2: Dict[str, Any],
                         weights: Dict[str, float] = None) -> float:
        """Compute weighted combination of multiple distance metrics."""
        if weights is None:
            weights = {
                'edit': 0.5,
                'logits_kl': 0.3,
                'embedding_cosine': 0.2
            }
        
        total_distance = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            try:
                dist = self.compute_distance(output1, output2, metric)
                if dist < 1.0 or metric == 'edit':  # Edit distance is always valid
                    total_distance += weight * dist
                    total_weight += weight
            except Exception:
                # Skip metrics that fail
                continue
        
        if total_weight > 0:
            return total_distance / total_weight
        else:
            # Fallback to edit distance
            return self.compute_distance(output1, output2, 'edit')


class TokenSpaceNormalizer:
    """Extended normalizer for enhanced functionality"""
    
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
    
    def normalize_tokens(self, tokens: List[int]) -> List[int]:
        """Normalize token sequence"""
        # Basic normalization - could be extended
        return tokens
    
    def compute_distance(self, tokens1: List[int], tokens2: List[int], method: str = 'fuzzy') -> float:
        """Compute distance between token sequences"""
        if method == 'exact':
            return 0.0 if tokens1 == tokens2 else 1.0
        
        # Fuzzy matching using Jaccard distance
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        if not set1 and not set2:
            return 0.0
        if not set1 or not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        jaccard_sim = intersection / union
        return 1.0 - jaccard_sim


class LMVerifier:
    """
    Language Model Verifier
    Implements the verification protocol for language models from paper Section 3
    """
    
    def __init__(self, 
                 reference_model: Optional[Union[LM, nn.Module, str]] = None,
                 tokenizer: Optional[Any] = None,
                 config: Optional[Union[Dict[str, Any], LMVerifierConfig]] = None,
                 delta: float = 0.01, 
                 use_sequential: bool = True, sequential_mode: str = 'legacy',
                 use_fingerprinting: bool = True,
                 fingerprint_config: Optional[FingerprintConfig] = None,
                 semantic_library: Optional['ConceptLibrary'] = None,
                 semantic_weight: float = 0.3):
        """
        Initialize LM verifier
        
        Args:
            reference_model: Reference language model f* (LM object, nn.Module, or path string)
            tokenizer: Tokenizer instance (optional if reference_model has one)
            config: Verification configuration dictionary
            delta: Confidence parameter (1-delta confidence)
            use_sequential: Whether to use sequential testing for early stopping
            sequential_mode: 'legacy' for old SPRT, 'enhanced' for new EB-based sequential verification
            use_fingerprinting: Whether to compute behavioral fingerprints
            fingerprint_config: Configuration for fingerprinting (uses default LM config if None)
            semantic_library: Optional ConceptLibrary for semantic verification
            semantic_weight: Weight for semantic score in combined verification (1-weight for distance)
        """
        # Handle different model input types
        self.model = self._load_model(reference_model) if reference_model else None
        self.reference_model = self.model  # For backward compatibility
        
        # Set tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif hasattr(self.model, 'tok'):
            self.tokenizer = self.model.tok
        elif hasattr(self.model, 'tokenizer'):
            self.tokenizer = self.model.tokenizer
        else:
            self.tokenizer = None
        
        # Configuration handling
        if config is None:
            self.lm_config = LMVerifierConfig()
        elif isinstance(config, LMVerifierConfig):
            self.lm_config = config
        elif isinstance(config, dict):
            self.lm_config = LMVerifierConfig.from_dict(config)
        else:
            raise ValueError(f"Config must be LMVerifierConfig, dict, or None, got {type(config)}")
        
        # Validate configuration
        issues = self.lm_config.validate()
        if issues:
            raise ValueError(f"Invalid configuration: {'; '.join(issues)}")
        
        # Backward compatibility - merge old parameters with config
        self.config = self.lm_config.to_dict()  # For legacy code
        self.delta = delta
        self.use_sequential = use_sequential or (self.lm_config.verification_method == 'sequential')
        self.sequential_mode = sequential_mode
        self.use_fingerprinting = use_fingerprinting
        
        # Set up fingerprinting configuration
        if use_fingerprinting:
            if fingerprint_config is None:
                # Use default language model configuration
                self.fingerprint_config = FingerprintConfig.for_language_model(
                    compute_jacobian=False,  # Jacobians less useful for text
                    include_timing=True
                )
            else:
                self.fingerprint_config = fingerprint_config
            
            # Will compute reference fingerprint lazily
            self.reference_fingerprint = None
        
        # Initialize components based on configuration
        if self.tokenizer:
            self.token_normalizer = TokenSpaceNormalizer(
                self.tokenizer, 
                mode=self.lm_config.normalization_mode
            )
            self.normalizer = self.token_normalizer  # Alias for compatibility
        else:
            self.token_normalizer = None
            self.normalizer = None
        
        # Initialize challenge generators
        self.challenge_generator = TemplateChallenger()  # Legacy generator
        self.new_challenge_generator = NewTemplateChallenger(
            difficulty_curve=self.lm_config.difficulty_curve
        )
        self.challenge_evaluator = ChallengeEvaluator(
            fuzzy_threshold=self.lm_config.fuzzy_threshold
        )
        
        self.distance_metrics = DistanceMetrics()
        
        # Initialize fuzzy hasher with config
        try:
            from .fuzzy_hash import FuzzyHasher
            self.fuzzy_hasher = FuzzyHasher(
                hash_type=self.lm_config.hash_type
            )
        except ImportError:
            self.fuzzy_hasher = None
        
        # Try to use advanced fuzzy hashing if available
        try:
            self.advanced_hasher = AdvancedFuzzyHasher()
        except ImportError:
            self.advanced_hasher = None
        
        # Sequential tester if enabled
        if use_sequential or self.lm_config.verification_method == 'sequential':
            self.sequential_tester = NewSequentialTester(
                alpha=self.lm_config.sprt_alpha,
                beta=self.lm_config.sprt_beta,
                p0=self.lm_config.sprt_p0,
                p1=self.lm_config.sprt_p1,
                max_trials=self.lm_config.max_trials,
                min_trials=self.lm_config.min_trials
            )
        else:
            self.sequential_tester = None
        
        # Initialize semantic verification if library provided
        self.semantic_library = semantic_library
        self.semantic_matcher = None
        self.semantic_weight = semantic_weight
        
        if semantic_library is not None and SEMANTIC_AVAILABLE:
            try:
                self.semantic_matcher = SemanticMatcher(
                    library=semantic_library,
                    threshold=0.7  # Default threshold for semantic matching
                )
            except Exception as e:
                print(f"Warning: Could not initialize semantic matcher: {e}")
                self.semantic_matcher = None
    
    def verify_enhanced(self, 
                       model: Optional[Union[LM, nn.Module]] = None,
                       challenges: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Enhanced verification using new template challenges and sequential testing.
        
        Args:
            model: Model to verify (uses self.model if None)  
            challenges: Pre-generated challenges (generates if None)
            
        Returns:
            Comprehensive verification result
        """
        if model is None:
            model = self.model
        if model is None:
            raise ValueError("No model to verify")
        
        start_time = time.time()
        
        # Generate challenges if not provided
        if challenges is None:
            challenges = self.new_challenge_generator.generate_challenge_set(
                num_challenges=self.lm_config.num_challenges,
                categories=self.lm_config.challenge_types,
                min_difficulty=1,
                max_difficulty=3
            )
        
        # Mock model runner for this model
        def model_runner(prompt: str) -> str:
            if isinstance(model, LM):
                return model.generate(prompt, max_new_tokens=64)
            else:
                result = self.run_model(prompt)
                return result.get('generated_text', [''])[0] if result else ''
        
        # Create verification session based on config
        if self.lm_config.verification_method == 'sequential' and self.sequential_tester:
            session = SequentialVerificationSession(
                tester=self.sequential_tester,
                challenger=self.new_challenge_generator,
                evaluator=self.challenge_evaluator,
                model_runner=model_runner
            )
            
            # Run sequential verification
            result = session.run_verification(
                max_challenges=self.lm_config.num_challenges,
                early_stop=True
            )
            
            # Add configuration info
            result['config'] = self.lm_config.to_dict()
            result['method'] = 'sequential'
            result['enhanced'] = True
            
        else:
            # Batch verification
            results = []
            total_score = 0.0
            
            for i, challenge in enumerate(challenges):
                # Get model response
                prompt = challenge.get('prompt', '')
                response = model_runner(prompt)
                
                # Evaluate response
                eval_result = self.challenge_evaluator.evaluate_response(response, challenge)
                
                results.append({
                    'challenge_id': i,
                    'challenge': challenge,
                    'response': response,
                    'success': eval_result.success,
                    'score': eval_result.score,
                    'details': eval_result.details
                })
                
                total_score += eval_result.score
            
            # Compute overall metrics
            num_successes = sum(1 for r in results if r['success'])
            success_rate = num_successes / len(results) if results else 0.0
            avg_score = total_score / len(results) if results else 0.0
            
            # Determine verification result
            threshold = self.lm_config.distance_threshold
            verified = success_rate >= (1.0 - threshold)
            
            result = {
                'verified': verified,
                'decision': 'reject' if verified else 'accept',  # reject H0 = genuine
                'success_rate': success_rate,
                'avg_score': avg_score,
                'num_trials': len(results),
                'num_successes': num_successes,
                'confidence': success_rate,
                'results': results,
                'method': 'batch',
                'enhanced': True,
                'config': self.lm_config.to_dict()
            }
        
        # Add timing
        result['duration'] = time.time() - start_time
        
        return result

    def _default_config(self) -> Dict[str, Any]:
        """Return default verification configuration"""
        return {
            'max_new_tokens': 64,
            'temperature': 1.0,
            'do_sample': False,
            'batch_size': 8,
            'challenge_types': ['mixed'],
            'distance_method': 'fuzzy',
            'early_stopping': True,
            'verbose': False
        }
    
    def _load_model(self, model: Union[LM, nn.Module, str]) -> Union[LM, nn.Module]:
        """Load model from various input types"""
        if isinstance(model, str):
            # Path to model - try to load
            try:
                from transformers import AutoModelForCausalLM
                loaded = AutoModelForCausalLM.from_pretrained(model)
                return loaded
            except Exception as e:
                print(f"Warning: Could not load model from path {model}: {e}")
                return None
        elif isinstance(model, (LM, nn.Module)):
            return model
        else:
            return model
    
    def generate_template_challenges(self, 
                                    challenge_type: str = 'mixed',
                                    num_challenges: int = 10,
                                    difficulty: str = 'adaptive',
                                    n: Optional[int] = None,
                                    master_key: Optional[str] = None,
                                    session_nonce: Optional[str] = None,
                                    model_id: Optional[str] = None) -> Union[List[Dict[str, Any]], Tuple[List[str], List[Dict[str, Any]]]]:
        """
        Generate template-based challenges for verification.
        
        Args:
            challenge_type: 'completion', 'qa', 'reasoning', 'mixed'
            num_challenges: Number of challenges to generate
            difficulty: 'easy', 'medium', 'hard', 'adaptive'
            n: Alternative to num_challenges for backward compatibility
            master_key: Master key for deterministic generation
            session_nonce: Session nonce for uniqueness
            model_id: Model identifier
        
        Returns:
            List of challenge dictionaries or tuple of (prompts, metadata)
        """
        # Handle backward compatibility
        if n is not None:
            num_challenges = n
        
        # If cryptographic parameters provided, use original method
        if master_key and session_nonce:
            return self._generate_crypto_challenges(num_challenges, master_key, session_nonce, model_id)
        
        # Otherwise use new template-based generation
        challenges = self.challenge_generator.generate_challenges(
            challenge_type, num_challenges, difficulty
        )
        
        # Adaptive difficulty adjustment
        if difficulty == 'adaptive':
            challenges = self._adjust_difficulty(challenges)
        
        return challenges
    
    def _generate_crypto_challenges(self, n: int, master_key: str, session_nonce: str, 
                                   model_id: Optional[str] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Generate challenges using cryptographic method (original)"""
        # Define template families for crypto generation
        templates = [
            "Complete the following: The {object} is {attribute}",
            "Q: What is {concept}? A:",
            "Translate to {language}: {text}",
            "Summarize: {passage}",
            "Continue the story: {beginning}",
        ]
        
        # Slot values for variation
        slots = {
            "object": ["cat", "house", "tree", "computer", "ocean"],
            "attribute": ["large", "blue", "ancient", "mysterious", "simple"],
            "concept": ["gravity", "democracy", "evolution", "entropy", "recursion"],
            "language": ["French", "Spanish", "German", "Italian", "Portuguese"],
            "text": ["Hello world", "Good morning", "Thank you", "How are you"],
            "passage": ["The quick brown fox...", "Once upon a time...", 
                       "In the beginning...", "It was the best of times..."],
            "beginning": ["The door creaked open", "She looked at the stars",
                         "The letter arrived", "He woke up suddenly"]
        }
        
        config = ChallengeConfig(
            master_key_hex=master_key,
            session_nonce_hex=session_nonce,
            n=n,
            family="lm:templates",
            params={"templates": templates, "slots": slots},
            model_id=model_id
        )
        
        result = generate_challenges(config)
        prompts = []
        metadata = []
        
        # Use Challenge objects if available
        challenges = result.get("challenges", None)
        if challenges:
            # New format with Challenge objects
            for challenge in challenges:
                params = challenge.parameters
                # Use the generated prompt
                prompt = params.get("prompt", None)
                if not prompt:
                    # Fallback: construct prompt from template and slots
                    template = params["template"]
                    slot_values = params["slot_values"]
                    prompt = template
                    for slot_name, slot_value in slot_values.items():
                        prompt = prompt.replace(f"{{{slot_name}}}", slot_value)
                
                prompts.append(prompt)
                
                # Store metadata including challenge ID
                metadata.append({
                    "challenge_id": challenge.challenge_id,
                    "index": challenge.index,
                    "template": params.get("template", ""),
                    "slot_values": params.get("slot_values", {}),
                    "prompt": prompt
                })
        else:
            # Backward compatibility: use items
            for idx, item in enumerate(result["items"]):
                # Construct prompt from template if needed
                if "prompt" in item:
                    prompt = item["prompt"]
                else:
                    template = item.get("template", "")
                    slot_values = item.get("slot_values", {})
                    prompt = template
                    for slot_name, slot_value in slot_values.items():
                        prompt = prompt.replace(f"{{{slot_name}}}", slot_value)
                
                prompts.append(prompt)
                metadata.append({
                    "challenge_id": f"legacy_{idx}",
                    "index": idx,
                    "template": item.get("template", ""),
                    "slot_values": item.get("slot_values", {}),
                    "prompt": prompt
                })
        
        return prompts, metadata
    
    
    def _adjust_difficulty(self, challenges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adjust challenge difficulty adaptively"""
        # Simple adaptive strategy - could be enhanced
        import random
        
        for challenge in challenges:
            # Randomly adjust difficulty based on index
            if challenge['index'] < len(challenges) // 3:
                challenge['difficulty'] = 'easy'
            elif challenge['index'] < 2 * len(challenges) // 3:
                challenge['difficulty'] = 'medium'
            else:
                challenge['difficulty'] = 'hard'
        
        return challenges
    
    def run_model(self, 
                  inputs: Union[str, List[str], torch.Tensor],
                  generation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run model on inputs and collect comprehensive outputs.
        
        Args:
            inputs: Text inputs (string or list of strings) or tensor
            generation_config: Generation configuration dictionary
        
        Returns:
            Dictionary with tokens, logits, embeddings, timing
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        generation_config = generation_config or {
            'max_length': 100,
            'max_new_tokens': 64,
            'temperature': 1.0,
            'do_sample': False,
            'return_dict_in_generate': True,
            'output_scores': True,
            'output_hidden_states': True
        }
        
        # Ensure inputs are list
        if isinstance(inputs, str):
            inputs = [inputs]
        elif isinstance(inputs, torch.Tensor):
            # Assume it's already tokenized
            pass
        
        # Tokenize if needed
        if self.tokenizer and not isinstance(inputs, torch.Tensor):
            encoded = self.tokenizer(inputs, return_tensors='pt', padding=True)
        else:
            encoded = {'input_ids': inputs} if isinstance(inputs, torch.Tensor) else None
        
        if encoded is None:
            raise ValueError("Could not tokenize inputs")
        
        # Run inference with timing
        start_time = time.perf_counter()
        
        # Check if model is LM type with generate method
        if hasattr(self.model, 'generate'):
            # LM-style model
            if isinstance(self.model, LM):
                # Use LM's generate method for each input
                outputs = []
                for text in inputs:
                    output = self.model.generate(text, max_new_tokens=generation_config.get('max_new_tokens', 64))
                    outputs.append(output)
                
                inference_time = time.perf_counter() - start_time
                
                return {
                    'generated_text': outputs,
                    'inference_time': inference_time,
                    'tokens_per_second': len(outputs) / inference_time if inference_time > 0 else 0
                }
            else:
                # PyTorch model with generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **encoded,
                        **generation_config
                    )
        else:
            # Forward pass only
            with torch.no_grad():
                outputs = self.model(**encoded)
        
        inference_time = time.perf_counter() - start_time
        
        # Process outputs
        result = {
            'inference_time': inference_time
        }
        
        if hasattr(outputs, 'sequences'):
            result['generated_ids'] = outputs.sequences
            if self.tokenizer:
                result['generated_text'] = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            result['tokens_per_second'] = outputs.sequences.shape[1] / inference_time if inference_time > 0 else 0
        
        if hasattr(outputs, 'scores'):
            result['logits'] = outputs.scores
        elif hasattr(outputs, 'logits'):
            result['logits'] = outputs.logits
        
        if hasattr(outputs, 'hidden_states'):
            result['hidden_states'] = outputs.hidden_states
        
        return result
    
    def verify_session(self,
                      model: Optional[Union[LM, nn.Module]] = None,
                      num_challenges: int = 10,
                      threshold: Optional[float] = None,
                      sequential_testing: bool = True,
                      challenge_type: str = 'mixed') -> Dict[str, Any]:
        """
        Run complete verification session.
        
        Args:
            model: Model to verify (uses self.model if None)
            num_challenges: Number of challenges to run
            threshold: Distance threshold for acceptance
            sequential_testing: Whether to use sequential testing
            challenge_type: Type of challenges to generate
        
        Returns:
            Verification result with confidence and metrics
        """
        if model is None:
            model = self.model
        if model is None:
            raise ValueError("No model to verify")
        
        # Default threshold
        if threshold is None:
            threshold = 0.1
        
        # Generate challenges
        challenges = self.generate_template_challenges(
            challenge_type=challenge_type,
            num_challenges=num_challenges
        )
        
        if sequential_testing:
            return self._sequential_verification(model, challenges, threshold)
        else:
            return self._batch_verification(model, challenges, threshold)
    
    def _sequential_verification(self, model: Union[LM, nn.Module], 
                                challenges: List[Dict[str, Any]],
                                threshold: float) -> Dict[str, Any]:
        """Implement SPRT-based sequential testing with early stopping."""
        from .sequential_tester import SequentialTester as SPRTTester, SequentialVerificationSession
        from .template_challenges import ChallengeEvaluator
        
        distances = []
        stopped_early = False
        n_evaluated = 0
        
        # Create distance stream generator
        def distance_stream():
            nonlocal n_evaluated
            for challenge in challenges:
                # Get prompt from challenge
                prompt = challenge.get('filled_template', challenge.get('prompt', ''))
                
                # Run both models
                if isinstance(model, LM):
                    model_output = model.generate(prompt, max_new_tokens=64)
                else:
                    model_result = self.run_model(prompt)
                    model_output = model_result.get('generated_text', [''])[0]
                
                if isinstance(self.reference_model, LM):
                    ref_output = self.reference_model.generate(prompt, max_new_tokens=64)
                else:
                    ref_result = self.run_model(prompt)
                    ref_output = ref_result.get('generated_text', [''])[0]
                
                # Compute distance
                distance = self.compute_output_distance(model_output, ref_output)
                distances.append(distance)
                n_evaluated += 1
                
                yield distance
        
        # Run sequential verification
        result = sequential_verify(
            stream=distance_stream(),
            tau=threshold,
            alpha=self.delta,
            beta=self.delta,
            max_samples=len(challenges),
            compute_p_value=True
        )
        
        # Check if stopped early
        if result.stopped_at < len(challenges):
            stopped_early = True
        
        return {
            'accepted': result.decision == 'H0',
            'distances': distances,
            'mean_distance': float(np.mean(distances)) if distances else 0.0,
            'n_evaluated': n_evaluated,
            'n_total': len(challenges),
            'stopped_early': stopped_early,
            'threshold': threshold,
            'confidence': 1.0 - self.delta,
            'sequential_result': result,
            'p_value': result.p_value
        }
    
    def _batch_verification(self, model: Union[LM, nn.Module],
                           challenges: List[Dict[str, Any]],
                           threshold: float) -> Dict[str, Any]:
        """Batch verification without sequential testing."""
        distances = []
        
        for challenge in challenges:
            # Get prompt from challenge
            prompt = challenge.get('filled_template', challenge.get('prompt', ''))
            
            # Run both models
            if isinstance(model, LM):
                model_output = model.generate(prompt, max_new_tokens=64)
            else:
                model_result = self.run_model(prompt)
                model_output = model_result.get('generated_text', [''])[0]
            
            if isinstance(self.reference_model, LM):
                ref_output = self.reference_model.generate(prompt, max_new_tokens=64)
            else:
                ref_result = self.run_model(prompt)
                ref_output = ref_result.get('generated_text', [''])[0]
            
            # Compute distance
            distance = self.compute_output_distance(model_output, ref_output)
            distances.append(distance)
        
        # Compute statistics
        distances_array = np.array(distances)
        mean_distance = float(np.mean(distances_array))
        std_distance = float(np.std(distances_array))
        
        # Statistical test
        from ..core.stats import empirical_bernstein_bound
        conf_radius = empirical_bernstein_bound(distances_array, self.delta)
        
        # Decision
        accepted = (mean_distance + conf_radius) <= threshold
        
        return {
            'accepted': accepted,
            'distances': distances,
            'mean_distance': mean_distance,
            'std_distance': std_distance,
            'confidence_radius': float(conf_radius),
            'n_evaluated': len(challenges),
            'n_total': len(challenges),
            'stopped_early': False,
            'threshold': threshold,
            'confidence': 1.0 - self.delta
        }
    
    def _sequential_verification_enhanced(self, 
                                         model: Union[LM, nn.Module],
                                         challenges: Optional[List[Dict]] = None,
                                         num_challenges: int = 100) -> Dict[str, Any]:
        """
        Enhanced sequential verification using new SPRT implementation.
        
        Args:
            model: Model to verify
            challenges: Optional pre-generated challenges
            num_challenges: Number of challenges if generating
            
        Returns:
            Verification results
        """
        from .sequential_tester import SequentialTester, SequentialVerificationSession
        from .template_challenges import TemplateChallenger, ChallengeEvaluator
        
        # Initialize components
        tester = SequentialTester(
            alpha=self.config.get('alpha', 0.05),
            beta=self.config.get('beta', 0.05),
            p0=self.config.get('p0', 0.5),
            p1=self.config.get('p1', 0.8),
            max_trials=num_challenges,
            min_trials=self.config.get('min_trials', 5)
        )
        
        challenger = TemplateChallenger(difficulty_curve='adaptive')
        evaluator = ChallengeEvaluator(fuzzy_threshold=0.85)
        
        # Create model runner function
        def model_runner(prompt: str) -> str:
            if isinstance(model, LM):
                return model.generate(prompt, max_new_tokens=64)
            else:
                result = self.run_model(prompt)
                return result.get('generated_text', [''])[0] if result else ''
        
        # Create verification session
        session = SequentialVerificationSession(
            tester=tester,
            challenger=challenger,
            evaluator=evaluator,
            model_runner=model_runner
        )
        
        # Run verification
        results = session.run_verification(
            max_challenges=num_challenges,
            early_stop=self.config.get('early_stopping', True)
        )
        
        return results
    
    def compute_output_distance(self, output1: str, output2: str,
                               method: str = 'fuzzy') -> float:
        """
        Compute distance between two model outputs.

        Args:
            output1: First model output
            output2: Second model output
            method: Distance computation method. Supported values:
                - ``'fuzzy'`` (default): Token-level fuzzy Jaccard distance
                - ``'exact'``: Exact token match
                - ``'weighted'``: Weighted n-gram distance
                - ``'edit'``: Normalized Levenshtein edit distance
                - ``'embedding'``: Cosine distance between token-count embeddings

        Returns:
            Distance in ``[0, 1]`` where 0 indicates identical outputs.
        """
        if method in {"fuzzy", "exact", "weighted"}:
            tokens1 = self.tokenizer.encode(output1, add_special_tokens=False)
            tokens2 = self.tokenizer.encode(output2, add_special_tokens=False)
            distance = self.normalizer.compute_distance(tokens1, tokens2, method=method)
            return distance

        if method == "edit":
            # Normalized edit distance using SequenceMatcher
            return 1.0 - difflib.SequenceMatcher(None, output1, output2).ratio()

        if method == "embedding":
            tokens1 = self.tokenizer.encode(output1, add_special_tokens=False)
            tokens2 = self.tokenizer.encode(output2, add_special_tokens=False)
            norm1 = self.normalizer.normalize_tokens(tokens1)
            norm2 = self.normalizer.normalize_tokens(tokens2)

            vec1 = Counter(norm1)
            vec2 = Counter(norm2)
            all_tokens = set(vec1) | set(vec2)
            if not all_tokens:
                return 0.0

            v1 = np.array([vec1[t] for t in all_tokens], dtype=float)
            v2 = np.array([vec2[t] for t in all_tokens], dtype=float)

            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                return 1.0

            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return 1.0 - cos_sim

        raise ValueError(f"Unknown method: {method}")
    
    def evaluate_challenge(self, model: LM, challenge: Dict[str, Any],
                            method: str = 'fuzzy') -> Tuple[str, float]:
        """
        Evaluate a single challenge on the model.

        Args:
            model: Model to query
            challenge: Challenge specification
            method: Distance computation method (default ``'fuzzy'``)

        Returns:
            Tuple of ``(model_output, distance_from_reference)``
        """
        # Construct prompt from challenge
        if "template" in challenge and "slot_values" in challenge:
            prompt = challenge["template"]
            for slot, value in challenge["slot_values"].items():
                prompt = prompt.replace(f"{{{slot}}}", value)
        else:
            # Fallback for simple text challenges
            prompt = challenge.get("prompt", "Hello")
        
        # Get outputs from both models
        model_output = model.generate(prompt, max_new_tokens=64)
        reference_output = self.reference_model.generate(prompt, max_new_tokens=64)
        
        # Compute distance
        distance = self.compute_output_distance(model_output, reference_output, method=method)
        
        return model_output, distance
    
    def _compute_semantic_score(self, output_embedding: torch.Tensor) -> float:
        """
        Compute semantic verification score for an output embedding.
        
        Args:
            output_embedding: Model output embedding
            
        Returns:
            Semantic similarity score in [0, 1]
        """
        if self.semantic_matcher is None:
            return 0.5  # Neutral score if no semantic verification
        
        try:
            # Normalize embedding
            normalized = normalize_embeddings(output_embedding, method='l2')
            
            # Match to library concepts
            matches = self.semantic_matcher.match_to_library(normalized)
            
            if not matches:
                return 0.0
            
            # Return best match score
            best_score = max(matches.values())
            return float(best_score)
            
        except Exception as e:
            print(f"Warning: Semantic scoring failed: {e}")
            return 0.5  # Neutral score on error
    
    def verify(self, model: LM, challenges: List[Dict[str, Any]],
              tolerance: float = 0.1, method: str = 'fuzzy',
              compute_reference_fingerprint: bool = False,
              alpha: float = None, beta: float = None) -> LMVerificationResult:
        """
        Verify a language model against reference.

        Args:
            model: Model to verify (f)
            challenges: List of challenges to evaluate
            tolerance: Maximum acceptable average distance (tau threshold)
            method: Distance computation method (default ``'fuzzy'``)
            compute_reference_fingerprint: Whether to compute reference fingerprint if not already done
            alpha: Type I error rate for sequential testing (defaults to self.delta)
            beta: Type II error rate for sequential testing (defaults to self.delta)

        Returns:
            LMVerificationResult with verification outcome
        """
        start_time = time.time()
        distances = []
        fuzzy_similarities = []
        
        # Set error rates if not provided
        if alpha is None:
            alpha = self.delta
        if beta is None:
            beta = self.delta
        
        # Compute behavioral fingerprint if enabled
        model_fingerprint = None
        fingerprint_similarity = None
        
        if self.use_fingerprinting:
            # Convert challenges to prompts for fingerprinting
            prompts = [self._challenge_to_prompt(ch) for ch in challenges]
            
            # Compute reference fingerprint if needed
            if compute_reference_fingerprint or self.reference_fingerprint is None:
                self.compute_reference_fingerprint(prompts)
            
            # Create a wrapper function for the model being verified
            def model_wrapper(prompt_batch):
                if isinstance(prompt_batch, str):
                    prompt_batch = [prompt_batch]
                
                outputs = []
                for prompt in prompt_batch:
                    output = model.generate(prompt, max_new_tokens=64)
                    # Canonicalize the text output
                    canonical_output = canonicalize_text(output)
                    outputs.append(canonical_output)
                
                return outputs if len(outputs) > 1 else outputs[0]
            
            # Compute fingerprint for the model being verified
            fingerprint_start = time.time()
            model_fingerprint = fingerprint_run(
                model_wrapper,
                prompts,
                self.fingerprint_config
            )
            fingerprint_time = time.time() - fingerprint_start
            
            # Compare fingerprints if reference exists
            if self.reference_fingerprint is not None:
                fingerprint_similarity = self.compute_fingerprint_similarity(
                    self.reference_fingerprint,
                    model_fingerprint
                )
                
                # Log fingerprint metrics for debugging
                print(f"[LM Fingerprint] IO Hash Match: {model_fingerprint.io_hash == self.reference_fingerprint.io_hash}")
                print(f"[LM Fingerprint] Similarity: {fingerprint_similarity:.4f}")
                print(f"[LM Fingerprint] Time: {fingerprint_time:.3f}s")
                
                # Early rejection based on fingerprint (optional, with conservative threshold)
                if fingerprint_similarity < 0.3:  # Very low threshold for text
                    print("[LM Fingerprint] Warning: Low similarity detected")
        
        # Initialize sequential testing based on mode
        sequential_result = None
        
        if self.use_sequential and self.sequential_mode == 'enhanced':
            # Use new EB-based sequential verification
            def distance_stream():
                for challenge in challenges:
                    model_output, distance = self.evaluate_challenge(model, challenge, method=method)
                    distances.append(distance)
                    
                    # Compute fuzzy similarity for additional validation
                    ref_output = self.reference_model.generate(
                        self._challenge_to_prompt(challenge), 
                        max_new_tokens=64
                    )
                    
                    tokens_model = self.tokenizer.encode(model_output, add_special_tokens=False)
                    tokens_ref = self.tokenizer.encode(ref_output, add_special_tokens=False)
                    
                    hash_model = self.fuzzy_hasher.compute_fuzzy_hash(tokens_model)
                    hash_ref = self.fuzzy_hasher.compute_fuzzy_hash(tokens_ref)
                    
                    similarity = self.fuzzy_hasher.jaccard_similarity(hash_model, hash_ref)
                    fuzzy_similarities.append(similarity)
                    
                    yield distance
            
            # Run sequential verification
            sequential_result = sequential_verify(
                stream=distance_stream(),
                tau=tolerance,
                alpha=alpha,
                beta=beta,
                max_samples=len(challenges),
                compute_p_value=True
            )
            
            # Early stopping achieved
            if sequential_result.stopped_at < len(challenges):
                print(f"[LM Sequential] Early stopping at n={sequential_result.stopped_at}")
                
        elif self.use_sequential and self.sequential_mode == 'legacy':
            # Use legacy SPRT
            self.sequential_tester = SequentialTester(
                alpha=alpha, beta=beta,
                tau0=tolerance/2, tau1=tolerance*2
            )
            
            for i, challenge in enumerate(challenges):
                # Evaluate challenge
                model_output, distance = self.evaluate_challenge(model, challenge, method=method)
                distances.append(distance)
                
                # Compute fuzzy similarity for additional validation
                ref_output = self.reference_model.generate(
                    self._challenge_to_prompt(challenge), 
                    max_new_tokens=64
                )
                
                tokens_model = self.tokenizer.encode(model_output, add_special_tokens=False)
                tokens_ref = self.tokenizer.encode(ref_output, add_special_tokens=False)
                
                hash_model = self.fuzzy_hasher.compute_fuzzy_hash(tokens_model)
                hash_ref = self.fuzzy_hasher.compute_fuzzy_hash(tokens_ref)
                
                similarity = self.fuzzy_hasher.jaccard_similarity(hash_model, hash_ref)
                fuzzy_similarities.append(similarity)
                
                # Sequential testing for early stopping
                result = self.sequential_tester.update(distance)
                if result.decision != 'continue':
                    # Early stopping
                    break
        else:
            # No sequential testing - evaluate all challenges
            for challenge in challenges:
                model_output, distance = self.evaluate_challenge(model, challenge, method=method)
                distances.append(distance)
                
                # Compute fuzzy similarity for additional validation
                ref_output = self.reference_model.generate(
                    self._challenge_to_prompt(challenge), 
                    max_new_tokens=64
                )
                
                tokens_model = self.tokenizer.encode(model_output, add_special_tokens=False)
                tokens_ref = self.tokenizer.encode(ref_output, add_special_tokens=False)
                
                hash_model = self.fuzzy_hasher.compute_fuzzy_hash(tokens_model)
                hash_ref = self.fuzzy_hasher.compute_fuzzy_hash(tokens_ref)
                
                similarity = self.fuzzy_hasher.jaccard_similarity(hash_model, hash_ref)
                fuzzy_similarities.append(similarity)
        
        # Compute test statistic and confidence radius
        distances_array = np.array(distances)
        test_statistic = t_statistic(distances_array)
        conf_radius = empirical_bernstein_bound(distances_array, self.delta)
        
        # Compute semantic score if enabled
        semantic_score = None
        combined_score = test_statistic
        
        if self.semantic_matcher is not None and SEMANTIC_AVAILABLE:
            try:
                # Extract embeddings from model outputs (using last hidden states if available)
                # For LMs, we'll use the average of token embeddings as the output embedding
                semantic_scores = []
                for challenge in challenges[:min(10, len(challenges))]:  # Sample for efficiency
                    prompt = self._challenge_to_prompt(challenge)
                    # Get model hidden states if available
                    if hasattr(model, 'get_hidden_states'):
                        hidden_states = model.get_hidden_states(prompt)
                        if hidden_states is not None:
                            # Average pooling over sequence length
                            output_embedding = torch.mean(hidden_states, dim=1) if hidden_states.dim() > 1 else hidden_states
                            sem_score = self._compute_semantic_score(output_embedding)
                            semantic_scores.append(sem_score)
                
                if semantic_scores:
                    semantic_score = float(np.mean(semantic_scores))
                    # Combine distance and semantic scores
                    # Lower distance is better, higher semantic score is better
                    # So we invert semantic score for combination
                    semantic_distance = 1.0 - semantic_score
                    combined_score = (1 - self.semantic_weight) * test_statistic + self.semantic_weight * semantic_distance
            except Exception as e:
                print(f"Warning: Semantic scoring failed: {e}")
                semantic_score = None
                combined_score = test_statistic
        
        # Decision logic based on sequential mode
        if sequential_result is not None:
            # Use enhanced sequential result
            accepted = sequential_result.decision == 'H0'
            test_statistic = sequential_result.final_mean
            conf_radius = sequential_result.confidence_radius
        elif self.use_sequential and self.sequential_mode == 'legacy' and hasattr(self, 'sequential_tester'):
            # Use legacy SPRT decision
            if self.sequential_tester.decided():
                accepted = self.sequential_tester.accept()
        else:
            # Fixed-sample decision: accept if combined score + radius <= tolerance
            # Use combined score if semantic verification is enabled
            decision_score = combined_score if semantic_score is not None else test_statistic
            accepted = (decision_score + conf_radius) <= tolerance
        
        elapsed = time.time() - start_time
        
        metadata = {
            "test_statistic": float(test_statistic),
            "tolerance": tolerance,
            "n_evaluated": len(distances),
            "distance_stats": {
                "mean": float(np.mean(distances_array)),
                "std": float(np.std(distances_array)),
                "min": float(np.min(distances_array)),
                "max": float(np.max(distances_array))
            },
            "fuzzy_stats": {
                "mean": float(np.mean(fuzzy_similarities)),
                "std": float(np.std(fuzzy_similarities))
            }
        }
        
        # Add fingerprint information to metadata if available
        if model_fingerprint is not None:
            metadata['fingerprint'] = {
                'io_hash': model_fingerprint.io_hash,
                'has_jacobian': model_fingerprint.jacobian_sketch is not None,
                'similarity': fingerprint_similarity if fingerprint_similarity is not None else 'N/A',
                'num_outputs': len(model_fingerprint.raw_outputs),
                'config': model_fingerprint.metadata.get('fingerprint_config', {})
            }
            if self.reference_fingerprint is not None:
                metadata['fingerprint']['reference_io_hash'] = self.reference_fingerprint.io_hash
        
        # Add advanced fuzzy hashing results if available
        if self.advanced_hasher:
            sample_idx = min(5, len(challenges))  # Sample a few for advanced hashing
            advanced_results = []
            
            for idx in range(sample_idx):
                challenge = challenges[idx]
                model_out, _ = self.evaluate_challenge(model, challenge)
                ref_out = self.reference_model.generate(
                    self._challenge_to_prompt(challenge), max_new_tokens=64
                )
                
                tokens_m = self.tokenizer.encode(model_out, add_special_tokens=False)
                tokens_r = self.tokenizer.encode(ref_out, add_special_tokens=False)
                
                sim_scores = self.advanced_hasher.compute_combined_similarity(
                    tokens_m, tokens_r
                )
                advanced_results.append(sim_scores)
            
            metadata["advanced_fuzzy"] = advanced_results
        
        # Add sequential result info to metadata if available
        if sequential_result is not None:
            metadata['sequential'] = {
                'mode': 'enhanced',
                'stopped_at': sequential_result.stopped_at,
                'decision': sequential_result.decision,
                'p_value': sequential_result.p_value,
                'trajectory_length': len(sequential_result.trajectory) if sequential_result.trajectory else 0,
                'forced_stop': sequential_result.forced_stop
            }
        elif self.use_sequential and self.sequential_mode == 'legacy':
            metadata['sequential'] = {
                'mode': 'legacy',
                'stopped_at': len(distances)
            }
        
        return LMVerificationResult(
            accepted=accepted,
            distance=float(test_statistic),
            confidence_radius=float(conf_radius),
            n_challenges=len(distances),
            fuzzy_similarity=float(np.mean(fuzzy_similarities)) if fuzzy_similarities else 0.0,
            time_elapsed=elapsed,
            fingerprint=model_fingerprint,
            fingerprint_match=fingerprint_similarity,
            sequential_result=sequential_result,
            semantic_score=semantic_score,
            combined_score=float(combined_score) if semantic_score is not None else None,
            metadata=metadata
        )
    
    def _challenge_to_prompt(self, challenge: Dict[str, Any]) -> str:
        """Convert challenge dict to prompt string"""
        if "template" in challenge and "slot_values" in challenge:
            prompt = challenge["template"]
            for slot, value in challenge["slot_values"].items():
                prompt = prompt.replace(f"{{{slot}}}", value)
            return prompt
        return challenge.get("prompt", "Hello")
    
    def compute_reference_fingerprint(self, prompts: List[str]) -> FingerprintResult:
        """
        Compute and store reference model fingerprint
        
        Args:
            prompts: Text prompts to use for fingerprinting
            
        Returns:
            FingerprintResult for the reference model
        """
        if not self.use_fingerprinting:
            return None
        
        # Create a wrapper function for the reference model
        def model_wrapper(prompt_batch):
            # Handle both single prompts and batches
            if isinstance(prompt_batch, str):
                prompt_batch = [prompt_batch]
            
            outputs = []
            for prompt in prompt_batch:
                output = self.reference_model.generate(prompt, max_new_tokens=64)
                # Canonicalize the text output
                canonical_output = canonicalize_text(output)
                outputs.append(canonical_output)
            
            return outputs if len(outputs) > 1 else outputs[0]
        
        # Compute fingerprint
        self.reference_fingerprint = fingerprint_run(
            model_wrapper,
            prompts,
            self.fingerprint_config
        )
        
        return self.reference_fingerprint
    
    def compute_fingerprint_similarity(self, fingerprint1: FingerprintResult,
                                      fingerprint2: FingerprintResult) -> float:
        """
        Compute similarity between two fingerprints
        
        Args:
            fingerprint1: First fingerprint
            fingerprint2: Second fingerprint
            
        Returns:
            Similarity score between 0 and 1 (1 = identical)
        """
        if fingerprint1 is None or fingerprint2 is None:
            return 0.0
        
        # Compare IO hashes (exact match)
        if fingerprint1.io_hash == fingerprint2.io_hash:
            return 1.0
        
        # If Jacobian sketches exist, compare them (though less useful for text)
        if fingerprint1.jacobian_sketch and fingerprint2.jacobian_sketch:
            sketch1_bytes = bytes.fromhex(fingerprint1.jacobian_sketch)
            sketch2_bytes = bytes.fromhex(fingerprint2.jacobian_sketch)
            similarity = compare_jacobian_sketches(sketch1_bytes, sketch2_bytes, method='hamming')
            return similarity
        
        # Compare raw outputs using fuzzy text matching
        if fingerprint1.raw_outputs and fingerprint2.raw_outputs:
            similarities = []
            for out1, out2 in zip(fingerprint1.raw_outputs, fingerprint2.raw_outputs):
                # Convert to strings if needed
                s1 = str(out1) if not isinstance(out1, str) else out1
                s2 = str(out2) if not isinstance(out2, str) else out2
                
                # Use SequenceMatcher for similarity
                import difflib
                similarity = difflib.SequenceMatcher(None, s1, s2).ratio()
                similarities.append(similarity)
            
            if similarities:
                return float(np.mean(similarities))
        
        # Default: different fingerprints
        return 0.0
    
    def verify_with_time_tolerance(self, model: LM,
                                  challenges: List[Dict[str, Any]],
                                  base_tolerance: float = 0.1,
                                  days_elapsed: int = 0,
                                  drift_rate: float = 0.001,
                                  drift_model: str = "linear",
                                  max_tolerance: Optional[float] = None) -> LMVerificationResult:
        """
        Verify with time-aware tolerance for version drift
        From paper Section 5 on handling model updates
        
        Args:
            model: Model to verify
            challenges: Verification challenges
            base_tolerance: Base tolerance at time 0
            days_elapsed: Days since reference model snapshot
            drift_rate: Expected drift parameter
            drift_model: Drift adjustment model ('linear', 'quadratic', 'exponential')
            max_tolerance: Optional cap on adjusted tolerance
            
        Returns:
            Verification result with adjusted tolerance
        """
        # Adjust tolerance based on time elapsed using specified drift model
        if drift_model == "linear":
            adjusted_tolerance = base_tolerance + drift_rate * days_elapsed
        elif drift_model == "quadratic":
            adjusted_tolerance = base_tolerance + drift_rate * (days_elapsed ** 2)
        elif drift_model == "exponential":
            adjusted_tolerance = base_tolerance * ((1 + drift_rate) ** days_elapsed)
        else:
            raise ValueError(f"Unknown drift_model: {drift_model}")

        cap_applied = False
        if max_tolerance is not None and adjusted_tolerance > max_tolerance:
            adjusted_tolerance = max_tolerance
            cap_applied = True
        
        # Run verification with adjusted tolerance (pass through fingerprinting)
        result = self.verify(model, challenges, adjusted_tolerance, 
                           compute_reference_fingerprint=True)
        
        # Add time tolerance info to metadata
        result.metadata["time_tolerance"] = {
            "base_tolerance": base_tolerance,
            "days_elapsed": days_elapsed,
            "drift_rate": drift_rate,
            "drift_model": drift_model,
            "max_tolerance": max_tolerance,
            "adjusted_tolerance": adjusted_tolerance,
            "justification": (
                f"Tolerance adjusted using {drift_model} model with rate {drift_rate} "
                f"over {days_elapsed} days" +
                ("; capped at maximum tolerance " + str(max_tolerance) if cap_applied else "")
            )
        }
        
        return result


class BatchLMVerifier:
    """
    Batch verification for multiple language models
    Implements efficient batch processing from paper
    """
    
    def __init__(self, reference_model: Optional[Union[LM, nn.Module, str]] = None,
                 tokenizer: Optional[Any] = None,
                 delta: float = 0.01,
                 use_fingerprinting: bool = True, 
                 fingerprint_config: Optional[FingerprintConfig] = None):
        self.verifier = LMVerifier(
            reference_model=reference_model,
            tokenizer=tokenizer,
            delta=delta, 
            use_fingerprinting=use_fingerprinting,
            fingerprint_config=fingerprint_config
        )
    
    def verify_batch(self, models: List[LM], 
                    challenges: List[Dict[str, Any]],
                    tolerance: float = 0.1) -> List[LMVerificationResult]:
        """
        Verify multiple models in batch
        
        Args:
            models: List of models to verify
            challenges: Common challenge set
            tolerance: Acceptance threshold
            
        Returns:
            List of verification results
        """
        results = []
        
        for i, model in enumerate(models):
            print(f"Verifying model {i+1}/{len(models)}...")
            result = self.verifier.verify(model, challenges, tolerance)
            results.append(result)
            
            # Early termination if too many failures
            failures = sum(1 for r in results if not r.accepted)
            if failures > len(models) * 0.5:
                print(f"High failure rate ({failures}/{i+1}), stopping batch")
                break
        
        return results
    
    def adaptive_verify(self, model: LM, 
                       min_challenges: int = 10,
                       max_challenges: int = 100,
                       tolerance: float = 0.1,
                       master_key: str = None,
                       session_nonce: str = None) -> LMVerificationResult:
        """
        Adaptive verification with dynamic challenge count
        Uses sequential testing to determine when to stop
        """
        if master_key is None:
            master_key = "0" * 64  # Default for testing
        if session_nonce is None:
            session_nonce = "1" * 32
        
        # Start with minimum challenges
        challenges = self.verifier.generate_template_challenges(
            min_challenges, master_key, session_nonce
        )
        
        # Initial verification
        result = self.verifier.verify(model, challenges, tolerance)
        
        # If inconclusive, add more challenges
        while (not result.accepted and 
               result.n_challenges < max_challenges and
               abs(result.distance - tolerance) < result.confidence_radius):
            
            # Generate additional challenges
            additional = min(min_challenges, max_challenges - result.n_challenges)
            new_challenges = self.verifier.generate_template_challenges(
                additional, master_key, session_nonce + str(result.n_challenges)
            )
            
            challenges.extend(new_challenges)
            
            # Re-verify with expanded challenge set
            result = self.verifier.verify(model, challenges, tolerance)
        
        return result