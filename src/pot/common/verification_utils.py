"""
Common verification utilities shared across the PoT framework.
Provides challenge generation, response evaluation, and verification helpers.
"""

import numpy as np
import torch
import hashlib
import json
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import time
import logging
from scipy import stats


logger = logging.getLogger(__name__)


class VerificationResult(Enum):
    """Verification result status"""
    SAME = "same"
    DIFFERENT = "different"
    UNDECIDED = "undecided"
    ERROR = "error"


@dataclass
class VerificationConfig:
    """Configuration for verification"""
    confidence_level: float = 0.95
    max_samples: int = 1000
    min_samples: int = 30
    tolerance: float = 1e-6
    timeout: float = 300.0  # seconds
    device: str = "cpu"
    seed: int = 42
    use_enhanced: bool = True  # Use enhanced diff decision framework
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class Challenge:
    """Represents a verification challenge"""
    id: str
    type: str
    data: Any
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'type': self.type,
            'data': self.data if not isinstance(self.data, (np.ndarray, torch.Tensor)) else str(type(self.data)),
            'metadata': self.metadata
        }


class ChallengeGenerator:
    """
    Generates challenges for verification.
    Consolidates challenge generation logic from multiple files.
    """
    
    def __init__(self, config: Optional[VerificationConfig] = None):
        """
        Initialize challenge generator.
        
        Args:
            config: Verification configuration
        """
        self.config = config or VerificationConfig()
        self._set_seed(self.config.seed)
        self.challenge_count = 0
    
    def _set_seed(self, seed: int):
        """Set random seed"""
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate(self, challenge_type: str, **kwargs) -> Challenge:
        """
        Generate a challenge of specified type.
        
        Args:
            challenge_type: Type of challenge to generate
            **kwargs: Type-specific parameters
            
        Returns:
            Generated challenge
        """
        self.challenge_count += 1
        challenge_id = f"{challenge_type}_{self.challenge_count:06d}"
        
        if challenge_type == "random_input":
            data = self._generate_random_input(**kwargs)
        elif challenge_type == "adversarial":
            data = self._generate_adversarial(**kwargs)
        elif challenge_type == "boundary":
            data = self._generate_boundary(**kwargs)
        elif challenge_type == "semantic":
            data = self._generate_semantic(**kwargs)
        elif challenge_type == "structural":
            data = self._generate_structural(**kwargs)
        else:
            raise ValueError(f"Unknown challenge type: {challenge_type}")
        
        return Challenge(
            id=challenge_id,
            type=challenge_type,
            data=data,
            metadata=kwargs
        )
    
    def _generate_random_input(self, 
                               shape: Tuple[int, ...] = (1, 10),
                               distribution: str = "normal",
                               **kwargs) -> np.ndarray:
        """Generate random input data"""
        if distribution == "normal":
            return np.random.randn(*shape).astype(np.float32)
        elif distribution == "uniform":
            low = kwargs.get('low', 0.0)
            high = kwargs.get('high', 1.0)
            return np.random.uniform(low, high, shape).astype(np.float32)
        elif distribution == "binary":
            return np.random.randint(0, 2, shape).astype(np.float32)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    def _generate_adversarial(self,
                             base_input: Optional[np.ndarray] = None,
                             epsilon: float = 0.01,
                             **kwargs) -> np.ndarray:
        """Generate adversarial input"""
        if base_input is None:
            base_input = self._generate_random_input(**kwargs)
        
        # Add adversarial perturbation
        perturbation = np.random.randn(*base_input.shape).astype(np.float32)
        perturbation = epsilon * np.sign(perturbation)
        
        return base_input + perturbation
    
    def _generate_boundary(self,
                          shape: Tuple[int, ...] = (1, 10),
                          boundary_type: str = "extreme",
                          **kwargs) -> np.ndarray:
        """Generate boundary case input"""
        if boundary_type == "extreme":
            # Mix of extreme values
            data = np.random.choice([-1e6, -1.0, 0.0, 1.0, 1e6], shape)
        elif boundary_type == "zero":
            data = np.zeros(shape)
        elif boundary_type == "ones":
            data = np.ones(shape)
        elif boundary_type == "nan":
            data = np.full(shape, np.nan)
        elif boundary_type == "inf":
            data = np.full(shape, np.inf)
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")
        
        return data.astype(np.float32)
    
    def _generate_semantic(self,
                          template: str = "default",
                          **kwargs) -> Dict[str, Any]:
        """Generate semantic challenge"""
        templates = {
            "default": {
                "prompt": "What is 2 + 2?",
                "expected": "4",
                "type": "arithmetic"
            },
            "reasoning": {
                "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
                "expected": "no",
                "type": "logic"
            },
            "factual": {
                "prompt": "What is the capital of France?",
                "expected": "Paris",
                "type": "knowledge"
            }
        }
        
        if template in templates:
            return templates[template]
        else:
            # Generate random semantic challenge
            return {
                "prompt": f"Test prompt {self.challenge_count}",
                "expected": f"Response {self.challenge_count}",
                "type": "generated"
            }
    
    def _generate_structural(self,
                           structure_type: str = "sequence",
                           length: int = 100,
                           **kwargs) -> Any:
        """Generate structural challenge"""
        if structure_type == "sequence":
            return np.random.randint(0, 100, length)
        elif structure_type == "tree":
            # Simple tree structure
            return {
                "root": 0,
                "children": {
                    0: [1, 2],
                    1: [3, 4],
                    2: [5, 6]
                }
            }
        elif structure_type == "graph":
            # Random graph edges
            num_nodes = kwargs.get('num_nodes', 10)
            num_edges = kwargs.get('num_edges', 15)
            edges = []
            for _ in range(num_edges):
                src = np.random.randint(0, num_nodes)
                dst = np.random.randint(0, num_nodes)
                if src != dst:
                    edges.append((src, dst))
            return {"nodes": num_nodes, "edges": edges}
        else:
            raise ValueError(f"Unknown structure type: {structure_type}")
    
    def generate_batch(self,
                      challenge_type: str,
                      batch_size: int,
                      **kwargs) -> List[Challenge]:
        """
        Generate a batch of challenges.
        
        Args:
            challenge_type: Type of challenges
            batch_size: Number of challenges
            **kwargs: Type-specific parameters
            
        Returns:
            List of challenges
        """
        return [self.generate(challenge_type, **kwargs) for _ in range(batch_size)]


class ResponseEvaluator:
    """
    Evaluates model responses to challenges.
    Consolidates evaluation logic from multiple verification modules.
    """
    
    def __init__(self, config: Optional[VerificationConfig] = None):
        """
        Initialize response evaluator.
        
        Args:
            config: Verification configuration
        """
        self.config = config or VerificationConfig()
    
    def evaluate(self,
                response1: Any,
                response2: Any,
                challenge: Optional[Challenge] = None) -> Dict[str, Any]:
        """
        Evaluate two responses for equivalence.
        
        Args:
            response1: First response
            response2: Second response
            challenge: Optional challenge that generated responses
            
        Returns:
            Evaluation results
        """
        result = {
            'equivalent': False,
            'difference': None,
            'confidence': 0.0,
            'method': 'unknown'
        }
        
        # Handle different response types
        if isinstance(response1, (np.ndarray, torch.Tensor)):
            result.update(self._evaluate_numeric(response1, response2))
        elif isinstance(response1, str):
            result.update(self._evaluate_text(response1, response2))
        elif isinstance(response1, dict):
            result.update(self._evaluate_structured(response1, response2))
        elif isinstance(response1, (int, float)):
            result.update(self._evaluate_scalar(response1, response2))
        else:
            result['method'] = 'type_comparison'
            result['equivalent'] = type(response1) == type(response2) and response1 == response2
        
        return result
    
    def _evaluate_numeric(self,
                         arr1: Union[np.ndarray, torch.Tensor],
                         arr2: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """Evaluate numeric arrays"""
        # Convert to numpy if needed
        if isinstance(arr1, torch.Tensor):
            arr1 = arr1.detach().cpu().numpy()
        if isinstance(arr2, torch.Tensor):
            arr2 = arr2.detach().cpu().numpy()
        
        # Check shapes
        if arr1.shape != arr2.shape:
            return {
                'equivalent': False,
                'difference': 'shape_mismatch',
                'confidence': 1.0,
                'method': 'shape_comparison',
                'details': f"Shapes: {arr1.shape} vs {arr2.shape}"
            }
        
        # Compute differences
        abs_diff = np.abs(arr1 - arr2)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        
        # Check if within tolerance
        equivalent = max_diff <= self.config.tolerance
        
        return {
            'equivalent': equivalent,
            'difference': float(mean_diff),
            'confidence': 1.0 if max_diff == 0 else max(0, 1.0 - max_diff),
            'method': 'numeric_comparison',
            'details': {
                'max_diff': float(max_diff),
                'mean_diff': float(mean_diff),
                'tolerance': self.config.tolerance
            }
        }
    
    def _evaluate_text(self, text1: str, text2: str) -> Dict[str, Any]:
        """Evaluate text responses"""
        # Exact match
        if text1 == text2:
            return {
                'equivalent': True,
                'difference': 0.0,
                'confidence': 1.0,
                'method': 'exact_match'
            }
        
        # Normalized match (case-insensitive, stripped)
        norm1 = text1.lower().strip()
        norm2 = text2.lower().strip()
        
        if norm1 == norm2:
            return {
                'equivalent': True,
                'difference': 0.0,
                'confidence': 0.95,
                'method': 'normalized_match'
            }
        
        # Compute similarity
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        return {
            'equivalent': similarity > 0.9,
            'difference': 1.0 - similarity,
            'confidence': similarity,
            'method': 'text_similarity',
            'details': {'similarity': similarity}
        }
    
    def _evaluate_structured(self, dict1: dict, dict2: dict) -> Dict[str, Any]:
        """Evaluate structured responses"""
        # Deep comparison
        def deep_compare(d1, d2):
            if type(d1) != type(d2):
                return False
            
            if isinstance(d1, dict):
                if set(d1.keys()) != set(d2.keys()):
                    return False
                return all(deep_compare(d1[k], d2[k]) for k in d1.keys())
            elif isinstance(d1, (list, tuple)):
                if len(d1) != len(d2):
                    return False
                return all(deep_compare(a, b) for a, b in zip(d1, d2))
            elif isinstance(d1, (np.ndarray, torch.Tensor)):
                return self._evaluate_numeric(d1, d2)['equivalent']
            else:
                return d1 == d2
        
        equivalent = deep_compare(dict1, dict2)
        
        return {
            'equivalent': equivalent,
            'difference': 0.0 if equivalent else 1.0,
            'confidence': 1.0 if equivalent else 0.0,
            'method': 'structural_comparison'
        }
    
    def _evaluate_scalar(self, val1: float, val2: float) -> Dict[str, Any]:
        """Evaluate scalar values"""
        diff = abs(val1 - val2)
        equivalent = diff <= self.config.tolerance
        
        # Compute relative difference
        if val1 != 0:
            rel_diff = diff / abs(val1)
        elif val2 != 0:
            rel_diff = diff / abs(val2)
        else:
            rel_diff = 0.0
        
        return {
            'equivalent': equivalent,
            'difference': diff,
            'confidence': 1.0 if equivalent else max(0, 1.0 - rel_diff),
            'method': 'scalar_comparison',
            'details': {
                'absolute_diff': diff,
                'relative_diff': rel_diff
            }
        }


class VerificationHelper:
    """
    Helper class for verification tasks.
    Provides common verification utilities.
    """
    
    @staticmethod
    def compute_confidence(results: List[Dict[str, Any]],
                         method: str = "binomial") -> Tuple[float, float]:
        """
        Compute confidence interval from verification results.
        
        Args:
            results: List of evaluation results
            method: Confidence computation method
            
        Returns:
            Tuple of (mean, confidence)
        """
        if not results:
            return 0.0, 0.0
        
        # Extract equivalence results
        equivalences = [r.get('equivalent', False) for r in results]
        n_same = sum(equivalences)
        n_total = len(equivalences)
        
        if n_total == 0:
            return 0.0, 0.0
        
        proportion = n_same / n_total
        
        if method == "binomial":
            # Binomial confidence interval
            if n_total < 30:
                # Use exact binomial
                confidence = stats.binom.interval(0.95, n_total, proportion)
                return proportion, (confidence[1] - confidence[0]) / 2
            else:
                # Use normal approximation
                std_error = np.sqrt(proportion * (1 - proportion) / n_total)
                confidence = 1.96 * std_error  # 95% confidence
                return proportion, confidence
        
        elif method == "bootstrap":
            # Bootstrap confidence interval
            n_bootstrap = 1000
            bootstrap_props = []
            
            for _ in range(n_bootstrap):
                sample = np.random.choice(equivalences, n_total, replace=True)
                bootstrap_props.append(np.mean(sample))
            
            confidence_interval = np.percentile(bootstrap_props, [2.5, 97.5])
            confidence = (confidence_interval[1] - confidence_interval[0]) / 2
            
            return proportion, confidence
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple verification results.
        
        Args:
            results: List of verification results
            
        Returns:
            Aggregated results
        """
        if not results:
            return {
                'verdict': VerificationResult.ERROR.value,
                'confidence': 0.0,
                'n_samples': 0
            }
        
        # Count outcomes
        n_same = sum(1 for r in results if r.get('equivalent', False))
        n_different = len(results) - n_same
        n_total = len(results)
        
        # Compute statistics
        differences = [r.get('difference', 0.0) for r in results if 'difference' in r]
        
        if differences:
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            max_diff = np.max(differences)
        else:
            mean_diff = std_diff = max_diff = 0.0
        
        # Determine verdict
        proportion_same = n_same / n_total
        
        if proportion_same >= 0.95:
            verdict = VerificationResult.SAME
        elif proportion_same <= 0.05:
            verdict = VerificationResult.DIFFERENT
        else:
            verdict = VerificationResult.UNDECIDED
        
        # Compute confidence
        mean_confidence, interval = VerificationHelper.compute_confidence(results)
        
        return {
            'verdict': verdict.value,
            'confidence': mean_confidence,
            'confidence_interval': interval,
            'n_samples': n_total,
            'n_same': n_same,
            'n_different': n_different,
            'statistics': {
                'mean_difference': mean_diff,
                'std_difference': std_diff,
                'max_difference': max_diff,
                'proportion_same': proportion_same
            }
        }
    
    @staticmethod
    def create_verification_report(aggregated: Dict[str, Any],
                                 config: VerificationConfig,
                                 elapsed_time: float) -> Dict[str, Any]:
        """
        Create a comprehensive verification report.
        
        Args:
            aggregated: Aggregated results
            config: Verification configuration
            elapsed_time: Total elapsed time
            
        Returns:
            Verification report
        """
        report = {
            'summary': {
                'verdict': aggregated['verdict'],
                'confidence': f"{aggregated['confidence']:.2%}",
                'samples_tested': aggregated['n_samples'],
                'execution_time': f"{elapsed_time:.2f}s"
            },
            'details': {
                'same_count': aggregated['n_same'],
                'different_count': aggregated['n_different'],
                'proportion_same': f"{aggregated['statistics']['proportion_same']:.4f}",
                'mean_difference': f"{aggregated['statistics']['mean_difference']:.6f}",
                'std_difference': f"{aggregated['statistics']['std_difference']:.6f}",
                'max_difference': f"{aggregated['statistics']['max_difference']:.6f}"
            },
            'configuration': config.to_dict(),
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'framework': 'PoT Verification',
                'enhanced_mode': config.use_enhanced
            }
        }
        
        # Add recommendations
        if aggregated['verdict'] == VerificationResult.UNDECIDED.value:
            report['recommendations'] = [
                f"Increase sample size (current: {aggregated['n_samples']})",
                "Consider adjusting confidence level",
                "Review difference threshold settings"
            ]
        
        return report


def compute_confidence(successes: int, 
                      trials: int,
                      confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute confidence interval for success rate.
    
    Args:
        successes: Number of successful trials
        trials: Total number of trials
        confidence_level: Desired confidence level
        
    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    if trials == 0:
        return 0.0, 0.0, 0.0
    
    proportion = successes / trials
    
    # Use Wilson score interval for better coverage
    z = stats.norm.ppf((1 + confidence_level) / 2)
    denominator = 1 + z**2 / trials
    
    center = (proportion + z**2 / (2 * trials)) / denominator
    
    margin = z * np.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) / denominator
    
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    
    return proportion, lower, upper


def run_verification(model1: Any,
                    model2: Any,
                    generator: ChallengeGenerator,
                    evaluator: ResponseEvaluator,
                    config: Optional[VerificationConfig] = None) -> Dict[str, Any]:
    """
    Run a complete verification process.
    
    Args:
        model1: First model
        model2: Second model
        generator: Challenge generator
        evaluator: Response evaluator
        config: Verification configuration
        
    Returns:
        Verification report
    """
    config = config or VerificationConfig()
    results = []
    start_time = time.time()
    
    logger.info("Starting verification process")
    
    for i in range(config.max_samples):
        # Check timeout
        if time.time() - start_time > config.timeout:
            logger.warning(f"Verification timeout after {i} samples")
            break
        
        # Generate challenge
        challenge = generator.generate("random_input")
        
        # Get responses
        try:
            if hasattr(model1, 'forward'):
                response1 = model1.forward(challenge.data)
            else:
                response1 = model1(challenge.data)
            
            if hasattr(model2, 'forward'):
                response2 = model2.forward(challenge.data)
            else:
                response2 = model2(challenge.data)
        except Exception as e:
            logger.error(f"Error getting responses: {e}")
            continue
        
        # Evaluate responses
        result = evaluator.evaluate(response1, response2, challenge)
        results.append(result)
        
        # Check for early stopping
        if i >= config.min_samples:
            aggregated = VerificationHelper.aggregate_results(results)
            if aggregated['verdict'] != VerificationResult.UNDECIDED.value:
                logger.info(f"Early stopping after {i+1} samples")
                break
    
    # Final aggregation
    aggregated = VerificationHelper.aggregate_results(results)
    elapsed_time = time.time() - start_time
    
    # Create report
    report = VerificationHelper.create_verification_report(
        aggregated, config, elapsed_time
    )
    
    logger.info(f"Verification completed: {aggregated['verdict']}")
    
    return report