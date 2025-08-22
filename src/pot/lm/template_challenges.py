"""
Template Challenge System for Language Model Verification
Implements comprehensive challenge generation with adaptive difficulty
"""

import re
import random
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import time

# Import fuzzy matching if available
try:
    from .fuzzy import FuzzyMatcher
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False


@dataclass
class ChallengeResult:
    """Result of challenge evaluation"""
    success: bool
    score: float
    match_type: str
    response: str
    expected: Any
    details: Dict[str, Any] = field(default_factory=dict)
    time_taken: float = 0.0
    difficulty: int = 1


class TemplateChallenger:
    """
    Template-based challenge generator with adaptive difficulty.
    Generates diverse challenges for language model verification.
    """
    
    def __init__(self, difficulty_curve: str = 'adaptive', seed: Optional[int] = None):
        """
        Initialize template challenge generator.
        
        Args:
            difficulty_curve: 'linear', 'exponential', 'adaptive', 'random'
            seed: Random seed for reproducibility
        """
        self.difficulty_curve = difficulty_curve
        self.challenge_history = []
        self.templates = self._load_templates()
        self.current_difficulty = 1
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _load_templates(self) -> Dict[str, List[Dict]]:
        """Load challenge templates by category."""
        return {
            'factual': [
                {
                    'prompt': 'The chemical formula for water is [MASK]',
                    'expected': r'H2O|H₂O',
                    'type': 'regex',
                    'difficulty': 1,
                    'tags': ['chemistry', 'basic']
                },
                {
                    'prompt': 'The speed of light in vacuum is approximately [MASK] m/s',
                    'expected': r'299[,.]?792[,.]?458|3\.?0?\s*[x×]\s*10\^8|c',
                    'type': 'regex',
                    'difficulty': 2,
                    'tags': ['physics', 'constant']
                },
                {
                    'prompt': 'The capital of France is [MASK]',
                    'expected': 'Paris',
                    'type': 'exact',
                    'difficulty': 1,
                    'tags': ['geography', 'capital']
                },
                {
                    'prompt': 'The largest planet in our solar system is [MASK]',
                    'expected': 'Jupiter',
                    'type': 'exact',
                    'difficulty': 1,
                    'tags': ['astronomy', 'solar_system']
                },
                {
                    'prompt': 'DNA stands for [MASK]',
                    'expected': r'[Dd]eoxyribonucleic [Aa]cid|DNA',
                    'type': 'regex',
                    'difficulty': 2,
                    'tags': ['biology', 'acronym']
                },
                {
                    'prompt': 'The boiling point of water at sea level is [MASK] degrees Celsius',
                    'expected': r'100|one hundred',
                    'type': 'regex',
                    'difficulty': 1,
                    'tags': ['physics', 'temperature']
                }
            ],
            'reasoning': [
                {
                    'prompt': 'If all roses are flowers and all flowers need water, then roses [MASK]',
                    'expected': ['need water', 'require water', 'need watering', 'must have water'],
                    'type': 'semantic',
                    'difficulty': 3,
                    'tags': ['logic', 'syllogism']
                },
                {
                    'prompt': 'If it is raining, the ground gets wet. The ground is wet. Therefore, [MASK]',
                    'expected': ['it might be raining', 'it could be raining', 'it may have rained'],
                    'type': 'semantic',
                    'difficulty': 4,
                    'tags': ['logic', 'inference']
                },
                {
                    'prompt': 'A is taller than B. B is taller than C. Therefore, A is [MASK] than C',
                    'expected': 'taller',
                    'type': 'exact',
                    'difficulty': 2,
                    'tags': ['logic', 'transitivity']
                },
                {
                    'prompt': 'All mammals are warm-blooded. Whales are mammals. Therefore, whales are [MASK]',
                    'expected': ['warm-blooded', 'warm blooded'],
                    'type': 'semantic',
                    'difficulty': 2,
                    'tags': ['logic', 'deduction']
                }
            ],
            'arithmetic': [
                {
                    'prompt': 'Calculate: 7 × 8 = [MASK]',
                    'expected': '56',
                    'type': 'exact',
                    'difficulty': 1,
                    'tags': ['math', 'multiplication']
                },
                {
                    'prompt': 'What is 15% of 240? [MASK]',
                    'expected': '36',
                    'type': 'exact',
                    'difficulty': 2,
                    'tags': ['math', 'percentage']
                },
                {
                    'prompt': 'Calculate: 123 + 456 = [MASK]',
                    'expected': '579',
                    'type': 'exact',
                    'difficulty': 1,
                    'tags': ['math', 'addition']
                },
                {
                    'prompt': 'What is the square root of 144? [MASK]',
                    'expected': '12',
                    'type': 'exact',
                    'difficulty': 2,
                    'tags': ['math', 'square_root']
                },
                {
                    'prompt': 'Calculate: 1000 ÷ 25 = [MASK]',
                    'expected': '40',
                    'type': 'exact',
                    'difficulty': 2,
                    'tags': ['math', 'division']
                },
                {
                    'prompt': 'What is 2^10? [MASK]',
                    'expected': '1024',
                    'type': 'exact',
                    'difficulty': 3,
                    'tags': ['math', 'exponent']
                }
            ],
            'completion': [
                {
                    'prompt': 'Roses are red, violets are [MASK]',
                    'expected': 'blue',
                    'type': 'exact',
                    'difficulty': 1,
                    'tags': ['poetry', 'rhyme']
                },
                {
                    'prompt': 'To be or not to be, that is the [MASK]',
                    'expected': 'question',
                    'type': 'exact',
                    'difficulty': 2,
                    'tags': ['literature', 'shakespeare']
                },
                {
                    'prompt': 'Once upon a [MASK]',
                    'expected': 'time',
                    'type': 'exact',
                    'difficulty': 1,
                    'tags': ['narrative', 'opening']
                },
                {
                    'prompt': 'The early bird catches the [MASK]',
                    'expected': 'worm',
                    'type': 'exact',
                    'difficulty': 1,
                    'tags': ['proverb', 'idiom']
                },
                {
                    'prompt': 'Better late than [MASK]',
                    'expected': 'never',
                    'type': 'exact',
                    'difficulty': 1,
                    'tags': ['proverb', 'idiom']
                }
            ],
            'coding': [
                {
                    'prompt': 'In Python, to print "Hello World", you write: print([MASK])',
                    'expected': r'"Hello World"|\'Hello World\'',
                    'type': 'regex',
                    'difficulty': 2,
                    'tags': ['programming', 'python']
                },
                {
                    'prompt': 'The boolean value for truth in Python is [MASK]',
                    'expected': 'True',
                    'type': 'exact',
                    'difficulty': 1,
                    'tags': ['programming', 'python', 'boolean']
                },
                {
                    'prompt': 'To define a function in Python, you use the keyword [MASK]',
                    'expected': 'def',
                    'type': 'exact',
                    'difficulty': 1,
                    'tags': ['programming', 'python', 'function']
                }
            ],
            'pattern': [
                {
                    'prompt': 'Complete the sequence: 2, 4, 6, 8, [MASK]',
                    'expected': '10',
                    'type': 'exact',
                    'difficulty': 1,
                    'tags': ['sequence', 'arithmetic']
                },
                {
                    'prompt': 'Complete the sequence: 1, 1, 2, 3, 5, 8, [MASK]',
                    'expected': '13',
                    'type': 'exact',
                    'difficulty': 3,
                    'tags': ['sequence', 'fibonacci']
                },
                {
                    'prompt': 'Complete the pattern: A, B, C, D, [MASK]',
                    'expected': 'E',
                    'type': 'exact',
                    'difficulty': 1,
                    'tags': ['sequence', 'alphabet']
                }
            ]
        }
    
    def generate_challenge_set(self, 
                              num_challenges: int = 10,
                              categories: Optional[List[str]] = None,
                              min_difficulty: int = 1,
                              max_difficulty: int = 5) -> List[Dict]:
        """
        Generate a set of challenges with balanced difficulty.
        
        Args:
            num_challenges: Number of challenges to generate
            categories: List of categories to include (None for all)
            min_difficulty: Minimum difficulty level
            max_difficulty: Maximum difficulty level
            
        Returns:
            List of challenge dictionaries
        """
        categories = categories or list(self.templates.keys())
        challenges = []
        
        for i in range(num_challenges):
            difficulty = self._compute_difficulty(i, num_challenges, min_difficulty, max_difficulty)
            category = self._select_category(categories, i)
            challenge = self._select_challenge(category, difficulty)
            
            if challenge:
                # Add metadata
                challenge['id'] = self._generate_challenge_id(challenge, i)
                challenge['category'] = category
                challenge['index'] = i
                challenges.append(challenge)
        
        return challenges
    
    def _compute_difficulty(self, index: int, total: int, 
                           min_diff: int, max_diff: int) -> int:
        """
        Compute difficulty based on curve type.
        
        Args:
            index: Current challenge index
            total: Total number of challenges
            min_diff: Minimum difficulty
            max_diff: Maximum difficulty
            
        Returns:
            Difficulty level
        """
        if self.difficulty_curve == 'linear':
            # Linear progression from min to max
            progress = index / max(total - 1, 1)
            diff = min_diff + int(progress * (max_diff - min_diff))
            
        elif self.difficulty_curve == 'exponential':
            # Exponential growth
            progress = index / max(total - 1, 1)
            diff = min_diff + int(np.exp(progress * np.log(max_diff - min_diff + 1)) - 1)
            
        elif self.difficulty_curve == 'adaptive':
            # Adapt based on history
            diff = self._adaptive_difficulty(min_diff, max_diff)
            
        elif self.difficulty_curve == 'random':
            # Random difficulty
            diff = random.randint(min_diff, max_diff)
            
        else:
            # Default to middle difficulty
            diff = (min_diff + max_diff) // 2
        
        return max(min_diff, min(max_diff, diff))
    
    def _adaptive_difficulty(self, min_diff: int = 1, max_diff: int = 5) -> int:
        """
        Adapt difficulty based on performance history.
        
        Args:
            min_diff: Minimum difficulty
            max_diff: Maximum difficulty
            
        Returns:
            Adapted difficulty level
        """
        if not self.challenge_history:
            return min_diff
        
        # Look at recent performance
        recent = self.challenge_history[-10:]  # Last 10 challenges
        if len(recent) < 3:
            return self.current_difficulty
        
        # Calculate success rate
        success_rate = sum(r.success for r in recent) / len(recent)
        
        # Adjust difficulty based on performance
        if success_rate > 0.8:
            # Too easy, increase difficulty
            self.current_difficulty = min(max_diff, self.current_difficulty + 1)
        elif success_rate < 0.4:
            # Too hard, decrease difficulty
            self.current_difficulty = max(min_diff, self.current_difficulty - 1)
        # else keep current difficulty
        
        return self.current_difficulty
    
    def _select_category(self, categories: List[str], index: int) -> str:
        """
        Select category for challenge.
        
        Args:
            categories: Available categories
            index: Challenge index
            
        Returns:
            Selected category
        """
        # Round-robin through categories for balance
        return categories[index % len(categories)]
    
    def _select_challenge(self, category: str, difficulty: int) -> Optional[Dict]:
        """
        Select a challenge from category with target difficulty.
        
        Args:
            category: Challenge category
            difficulty: Target difficulty level
            
        Returns:
            Challenge dictionary or None
        """
        if category not in self.templates:
            return None
        
        candidates = self.templates[category]
        
        # Filter by difficulty (with tolerance)
        exact_matches = [c for c in candidates if c['difficulty'] == difficulty]
        if exact_matches:
            return random.choice(exact_matches).copy()
        
        # If no exact match, find closest
        close_matches = [c for c in candidates 
                        if abs(c['difficulty'] - difficulty) <= 1]
        if close_matches:
            return random.choice(close_matches).copy()
        
        # Fallback to any challenge from category
        return random.choice(candidates).copy() if candidates else None
    
    def _generate_challenge_id(self, challenge: Dict, index: int) -> str:
        """
        Generate unique ID for challenge.
        
        Args:
            challenge: Challenge dictionary
            index: Challenge index
            
        Returns:
            Unique challenge ID
        """
        content = f"{challenge['prompt']}_{index}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def add_to_history(self, result: ChallengeResult):
        """
        Add challenge result to history for adaptive difficulty.
        
        Args:
            result: Challenge evaluation result
        """
        self.challenge_history.append(result)
        
        # Keep history bounded
        if len(self.challenge_history) > 100:
            self.challenge_history = self.challenge_history[-100:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from challenge history.
        
        Returns:
            Dictionary of statistics
        """
        if not self.challenge_history:
            return {
                'total': 0,
                'success_rate': 0.0,
                'avg_score': 0.0,
                'avg_time': 0.0
            }
        
        total = len(self.challenge_history)
        successes = sum(r.success for r in self.challenge_history)
        avg_score = sum(r.score for r in self.challenge_history) / total
        avg_time = sum(r.time_taken for r in self.challenge_history) / total
        
        # Breakdown by difficulty
        by_difficulty = defaultdict(lambda: {'total': 0, 'success': 0})
        for result in self.challenge_history:
            diff = result.difficulty
            by_difficulty[diff]['total'] += 1
            if result.success:
                by_difficulty[diff]['success'] += 1
        
        return {
            'total': total,
            'success_rate': successes / total,
            'avg_score': avg_score,
            'avg_time': avg_time,
            'by_difficulty': dict(by_difficulty),
            'current_difficulty': self.current_difficulty
        }


class ChallengeEvaluator:
    """
    Evaluates model responses against challenge expectations.
    Supports multiple evaluation methods and fuzzy matching.
    """
    
    def __init__(self, fuzzy_threshold: float = 0.85, 
                 strict_mode: bool = False):
        """
        Initialize challenge evaluator.
        
        Args:
            fuzzy_threshold: Threshold for fuzzy matching success
            strict_mode: If True, require exact matches for 'exact' type
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.strict_mode = strict_mode
        
        if FUZZY_AVAILABLE:
            self.fuzzy_matcher = FuzzyMatcher(threshold=fuzzy_threshold)
        else:
            self.fuzzy_matcher = None
    
    def evaluate_response(self, 
                         response: str,
                         challenge: Dict,
                         timeout: Optional[float] = None) -> ChallengeResult:
        """
        Evaluate response against challenge expectations.
        
        Args:
            response: Model's response
            challenge: Challenge dictionary with expected answer
            timeout: Optional timeout for evaluation
            
        Returns:
            ChallengeResult with evaluation details
        """
        start_time = time.time()
        
        # Clean response
        response = self._clean_response(response)
        
        # Initialize result
        result = ChallengeResult(
            success=False,
            score=0.0,
            match_type=challenge.get('type', 'exact'),
            response=response,
            expected=challenge.get('expected'),
            difficulty=challenge.get('difficulty', 1)
        )
        
        # Evaluate based on type
        if challenge['type'] == 'exact':
            result = self._evaluate_exact(response, challenge, result)
            
        elif challenge['type'] == 'regex':
            result = self._evaluate_regex(response, challenge, result)
            
        elif challenge['type'] == 'semantic':
            result = self._evaluate_semantic(response, challenge, result)
            
        elif challenge['type'] == 'contains':
            result = self._evaluate_contains(response, challenge, result)
            
        elif challenge['type'] == 'numeric':
            result = self._evaluate_numeric(response, challenge, result)
            
        else:
            # Unknown type, default to exact
            result = self._evaluate_exact(response, challenge, result)
        
        # Record time taken
        result.time_taken = time.time() - start_time
        
        return result
    
    def _clean_response(self, response: str) -> str:
        """
        Clean and normalize response.
        
        Args:
            response: Raw response
            
        Returns:
            Cleaned response
        """
        if not response:
            return ""
        
        # Remove leading/trailing whitespace
        response = response.strip()
        
        # Remove common markers
        markers = ['[MASK]', '<mask>', '[BLANK]', '___']
        for marker in markers:
            response = response.replace(marker, '').strip()
        
        # Normalize quotes
        response = response.replace('"', '"').replace('"', '"')
        response = response.replace(''', "'").replace(''', "'")
        
        return response
    
    def _evaluate_exact(self, response: str, challenge: Dict, 
                       result: ChallengeResult) -> ChallengeResult:
        """
        Evaluate exact match.
        
        Args:
            response: Cleaned response
            challenge: Challenge dictionary
            result: Result object to update
            
        Returns:
            Updated result
        """
        expected = challenge['expected']
        
        if self.strict_mode:
            # Strict exact match
            result.success = response == expected
            result.score = 1.0 if result.success else 0.0
        else:
            # Case-insensitive match
            result.success = response.lower() == str(expected).lower()
            result.score = 1.0 if result.success else 0.0
            
            # Try fuzzy match if available and failed exact
            if not result.success and self.fuzzy_matcher:
                fuzzy_score = self.fuzzy_matcher.fuzzy_match(
                    response, str(expected), method='ratio'
                )
                result.score = fuzzy_score
                result.details['fuzzy_score'] = fuzzy_score
        
        return result
    
    def _evaluate_regex(self, response: str, challenge: Dict,
                       result: ChallengeResult) -> ChallengeResult:
        """
        Evaluate regex pattern match.
        
        Args:
            response: Cleaned response
            challenge: Challenge dictionary
            result: Result object to update
            
        Returns:
            Updated result
        """
        pattern_str = challenge['expected']
        
        try:
            # Compile pattern (case-insensitive unless strict)
            flags = 0 if self.strict_mode else re.IGNORECASE
            pattern = re.compile(pattern_str, flags)
            
            # Check for match
            match = pattern.search(response)
            result.success = bool(match)
            result.score = 1.0 if result.success else 0.0
            
            if match:
                result.details['match'] = match.group()
                result.details['span'] = match.span()
            
        except re.error as e:
            # Invalid regex
            result.details['error'] = str(e)
            result.score = 0.0
            result.success = False
        
        return result
    
    def _evaluate_semantic(self, response: str, challenge: Dict,
                          result: ChallengeResult) -> ChallengeResult:
        """
        Evaluate semantic similarity.
        
        Args:
            response: Cleaned response
            challenge: Challenge dictionary
            result: Result object to update
            
        Returns:
            Updated result
        """
        expected_list = challenge['expected']
        if not isinstance(expected_list, list):
            expected_list = [expected_list]
        
        if not self.fuzzy_matcher:
            # Fallback to simple contains check
            for expected in expected_list:
                if str(expected).lower() in response.lower():
                    result.success = True
                    result.score = 1.0
                    result.details['matched'] = expected
                    break
        else:
            # Use fuzzy matching
            best_score = 0.0
            best_match = None
            
            for expected in expected_list:
                score = self.fuzzy_matcher.fuzzy_match(
                    response, str(expected), method='token_set_ratio'
                )
                if score > best_score:
                    best_score = score
                    best_match = expected
            
            result.score = best_score
            result.success = best_score >= self.fuzzy_threshold
            result.details['best_match'] = best_match
            result.details['best_score'] = best_score
        
        return result
    
    def _evaluate_contains(self, response: str, challenge: Dict,
                          result: ChallengeResult) -> ChallengeResult:
        """
        Evaluate if response contains required terms.
        
        Args:
            response: Cleaned response
            challenge: Challenge dictionary
            result: Result object to update
            
        Returns:
            Updated result
        """
        required_terms = challenge['expected']
        if not isinstance(required_terms, list):
            required_terms = [required_terms]
        
        response_lower = response.lower()
        found_terms = []
        
        for term in required_terms:
            if str(term).lower() in response_lower:
                found_terms.append(term)
        
        result.score = len(found_terms) / len(required_terms) if required_terms else 0.0
        result.success = result.score >= 0.8  # 80% threshold
        result.details['required_terms'] = required_terms
        result.details['found_terms'] = found_terms
        result.details['missing_terms'] = [t for t in required_terms if t not in found_terms]
        
        return result
    
    def _evaluate_numeric(self, response: str, challenge: Dict,
                         result: ChallengeResult) -> ChallengeResult:
        """
        Evaluate numeric answer with tolerance.
        
        Args:
            response: Cleaned response
            challenge: Challenge dictionary
            result: Result object to update
            
        Returns:
            Updated result
        """
        try:
            # Extract number from response
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if not numbers:
                result.success = False
                result.score = 0.0
                return result
            
            response_num = float(numbers[0])
            expected_num = float(challenge['expected'])
            
            # Check with tolerance
            tolerance = challenge.get('tolerance', 0.01)
            diff = abs(response_num - expected_num)
            relative_diff = diff / max(abs(expected_num), 1e-10)
            
            result.success = relative_diff <= tolerance
            result.score = max(0, 1 - relative_diff)
            result.details['response_value'] = response_num
            result.details['expected_value'] = expected_num
            result.details['relative_error'] = relative_diff
            
        except (ValueError, TypeError) as e:
            result.success = False
            result.score = 0.0
            result.details['error'] = str(e)
        
        return result
    
    def batch_evaluate(self, responses: List[str], 
                       challenges: List[Dict]) -> List[ChallengeResult]:
        """
        Evaluate multiple responses in batch.
        
        Args:
            responses: List of model responses
            challenges: List of challenges
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for response, challenge in zip(responses, challenges):
            result = self.evaluate_response(response, challenge)
            results.append(result)
        
        return results


class DynamicChallengeGenerator:
    """
    Generates challenges dynamically based on topics and difficulty.
    Can create novel challenges beyond templates.
    """
    
    def __init__(self, knowledge_base: Optional[Dict] = None,
                 seed: Optional[int] = None):
        """
        Initialize dynamic generator.
        
        Args:
            knowledge_base: Optional knowledge base for generating challenges
            seed: Random seed
        """
        self.knowledge_base = knowledge_base or self._default_knowledge()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _default_knowledge(self) -> Dict:
        """Create default knowledge base."""
        return {
            'math_operations': ['+', '-', '*', '/', '%', '**'],
            'math_constants': {'pi': 3.14159, 'e': 2.71828, 'phi': 1.618},
            'logic_connectives': ['and', 'or', 'not', 'implies', 'iff'],
            'countries': ['USA', 'UK', 'France', 'Germany', 'Japan', 'China'],
            'colors': ['red', 'blue', 'green', 'yellow', 'orange', 'purple'],
            'animals': ['cat', 'dog', 'bird', 'fish', 'lion', 'elephant'],
            'programming_languages': ['Python', 'Java', 'C++', 'JavaScript', 'Go', 'Rust']
        }
    
    def generate_dynamic_challenge(self, 
                                  topic: str,
                                  difficulty: int = 2,
                                  subtype: Optional[str] = None) -> Dict:
        """
        Generate challenge dynamically based on topic.
        
        Args:
            topic: Topic area (math, logic, knowledge, coding, etc.)
            difficulty: Difficulty level (1-5)
            subtype: Optional subtype for more specific generation
            
        Returns:
            Generated challenge dictionary
        """
        if topic == 'math':
            return self._generate_math_challenge(difficulty, subtype)
        elif topic == 'logic':
            return self._generate_logic_challenge(difficulty, subtype)
        elif topic == 'knowledge':
            return self._generate_knowledge_challenge(difficulty, subtype)
        elif topic == 'coding':
            return self._generate_coding_challenge(difficulty, subtype)
        elif topic == 'pattern':
            return self._generate_pattern_challenge(difficulty, subtype)
        else:
            # Fallback to random topic
            topic = random.choice(['math', 'logic', 'knowledge', 'pattern'])
            return self.generate_dynamic_challenge(topic, difficulty, subtype)
    
    def _generate_math_challenge(self, difficulty: int, 
                                subtype: Optional[str] = None) -> Dict:
        """Generate arithmetic challenge."""
        if difficulty == 1:
            # Simple arithmetic
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            op = random.choice(['+', '-', '*'])
            answer = str(eval(f"{a}{op}{b}"))
            prompt = f"Calculate: {a} {op} {b} = [MASK]"
            
        elif difficulty == 2:
            # Percentage or division
            if random.choice([True, False]):
                # Percentage
                a = random.randint(10, 200)
                b = random.choice([5, 10, 15, 20, 25, 50])
                answer = str(a * b // 100)
                prompt = f"What is {b}% of {a}? [MASK]"
            else:
                # Division
                answer = random.randint(2, 20)
                b = random.randint(2, 10)
                a = answer * b
                prompt = f"Calculate: {a} ÷ {b} = [MASK]"
                
        elif difficulty == 3:
            # Square roots or powers
            if random.choice([True, False]):
                # Square root
                n = random.choice([4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144])
                answer = str(int(n ** 0.5))
                prompt = f"What is the square root of {n}? [MASK]"
            else:
                # Powers
                base = random.randint(2, 5)
                exp = random.randint(2, 4)
                answer = str(base ** exp)
                prompt = f"Calculate: {base}^{exp} = [MASK]"
                
        elif difficulty == 4:
            # Multi-step problems
            a = random.randint(10, 50)
            b = random.randint(5, 20)
            c = random.randint(2, 10)
            answer = str((a + b) * c)
            prompt = f"Calculate: ({a} + {b}) × {c} = [MASK]"
            
        else:  # difficulty 5
            # Complex expressions
            a = random.randint(100, 500)
            b = random.randint(10, 50)
            c = random.randint(2, 9)
            answer = str((a - b * c) // 10)
            prompt = f"Calculate: ({a} - {b} × {c}) ÷ 10 = [MASK]"
        
        return {
            'prompt': prompt,
            'expected': answer,
            'type': 'exact',
            'difficulty': difficulty,
            'category': 'math',
            'generated': True
        }
    
    def _generate_logic_challenge(self, difficulty: int,
                                 subtype: Optional[str] = None) -> Dict:
        """Generate logic challenge."""
        if difficulty <= 2:
            # Simple syllogism
            subjects = ['cats', 'dogs', 'birds', 'flowers', 'trees']
            properties = ['animals', 'living things', 'pets', 'plants', 'organisms']
            actions = ['need water', 'grow', 'breathe', 'move', 'reproduce']
            
            A = random.choice(subjects)
            B = random.choice(properties)
            C = random.choice(actions)
            
            prompt = f"If all {A} are {B} and all {B} {C}, then {A} [MASK]"
            expected = [C, f"also {C}", f"must {C}"]
            
            return {
                'prompt': prompt,
                'expected': expected,
                'type': 'semantic',
                'difficulty': difficulty,
                'category': 'logic',
                'generated': True
            }
            
        elif difficulty == 3:
            # Comparison logic
            items = ['A', 'B', 'C', 'D']
            random.shuffle(items)
            
            prompt = f"{items[0]} is larger than {items[1]}. {items[1]} is larger than {items[2]}. "
            prompt += f"Therefore, {items[0]} is [MASK] than {items[2]}"
            
            return {
                'prompt': prompt,
                'expected': 'larger',
                'type': 'exact',
                'difficulty': difficulty,
                'category': 'logic',
                'generated': True
            }
            
        else:
            # Complex inference
            premises = [
                "If it rains, the street gets wet",
                "If the sun shines, it's warm",
                "If it's winter, it's cold"
            ]
            
            premise = random.choice(premises)
            parts = premise.split(', ')
            
            prompt = f"{premise}. {parts[1].capitalize()}. What can we conclude? [MASK]"
            expected = ["It might be raining", "It could be raining", "Possibly it rained"]
            
            return {
                'prompt': prompt,
                'expected': expected,
                'type': 'semantic',
                'difficulty': difficulty,
                'category': 'logic',
                'generated': True
            }
    
    def _generate_knowledge_challenge(self, difficulty: int,
                                     subtype: Optional[str] = None) -> Dict:
        """Generate knowledge-based challenge."""
        if difficulty == 1:
            # Simple facts
            facts = [
                ("The largest ocean on Earth", "Pacific"),
                ("The smallest continent", "Australia"),
                ("The longest river", "Nile"),
                ("The highest mountain", "Everest")
            ]
            
            fact = random.choice(facts)
            prompt = f"{fact[0]} is [MASK]"
            
            return {
                'prompt': prompt,
                'expected': fact[1],
                'type': 'exact',
                'difficulty': difficulty,
                'category': 'knowledge',
                'generated': True
            }
            
        else:
            # Use knowledge base
            topic = random.choice(['countries', 'colors', 'animals'])
            items = self.knowledge_base[topic]
            item = random.choice(items)
            
            prompts = {
                'countries': f"The country code for {item} is [MASK]",
                'colors': f"The complementary color of {item} is [MASK]",
                'animals': f"A baby {item} is called a [MASK]"
            }
            
            # Simple placeholder answers
            answers = {
                'countries': item[:2].upper(),
                'colors': 'varies',
                'animals': 'baby'
            }
            
            return {
                'prompt': prompts[topic],
                'expected': answers[topic],
                'type': 'semantic',
                'difficulty': difficulty,
                'category': 'knowledge',
                'generated': True
            }
    
    def _generate_coding_challenge(self, difficulty: int,
                                  subtype: Optional[str] = None) -> Dict:
        """Generate programming challenge."""
        lang = subtype or 'Python'
        
        if difficulty == 1:
            # Basic syntax
            prompts = [
                f"In {lang}, comments start with [MASK]",
                f"To import a module in {lang}, use [MASK]",
                f"The null value in {lang} is [MASK]"
            ]
            
            prompt = random.choice(prompts)
            
            # Language-specific answers
            if 'Python' in lang:
                answers = {'comments': '#', 'import': 'import', 'null': 'None'}
            else:
                answers = {'comments': '//', 'import': 'import', 'null': 'null'}
            
            key = 'comments' if 'comments' in prompt else ('import' if 'import' in prompt else 'null')
            
            return {
                'prompt': prompt,
                'expected': answers[key],
                'type': 'exact',
                'difficulty': difficulty,
                'category': 'coding',
                'generated': True
            }
            
        else:
            # Code completion
            templates = [
                "for i in range([MASK]):",
                "if x [MASK] 0:",
                "def function([MASK]):"
            ]
            
            template = random.choice(templates)
            
            if 'range' in template:
                answer = random.choice(['10', '5', 'n', 'len(arr)'])
            elif 'if x' in template:
                answer = random.choice(['>', '<', '==', '!='])
            else:
                answer = random.choice(['x', 'args', '*args', 'x, y'])
            
            return {
                'prompt': template,
                'expected': answer,
                'type': 'semantic',
                'difficulty': difficulty,
                'category': 'coding',
                'generated': True
            }
    
    def _generate_pattern_challenge(self, difficulty: int,
                                   subtype: Optional[str] = None) -> Dict:
        """Generate pattern recognition challenge."""
        if difficulty == 1:
            # Simple arithmetic sequence
            start = random.randint(1, 10)
            step = random.randint(1, 5)
            seq = [start + i * step for i in range(4)]
            answer = str(seq[3] + step)
            
            prompt = f"Complete the sequence: {', '.join(map(str, seq))}, [MASK]"
            
        elif difficulty == 2:
            # Geometric sequence
            start = random.randint(1, 5)
            ratio = random.randint(2, 3)
            seq = [start * (ratio ** i) for i in range(4)]
            answer = str(seq[3] * ratio)
            
            prompt = f"Complete the sequence: {', '.join(map(str, seq))}, [MASK]"
            
        elif difficulty == 3:
            # Fibonacci-like
            a, b = 1, 1
            seq = [a, b]
            for _ in range(3):
                a, b = b, a + b
                seq.append(b)
            answer = str(seq[-1] + seq[-2])
            
            prompt = f"Complete the sequence: {', '.join(map(str, seq))}, [MASK]"
            
        else:
            # Complex pattern
            seq = [i**2 + i for i in range(1, 5)]
            answer = str(5**2 + 5)
            
            prompt = f"Complete the sequence: {', '.join(map(str, seq))}, [MASK]"
        
        return {
            'prompt': prompt,
            'expected': answer,
            'type': 'exact',
            'difficulty': difficulty,
            'category': 'pattern',
            'generated': True
        }
    
    def generate_batch(self, num_challenges: int = 10,
                       topics: Optional[List[str]] = None,
                       difficulty_range: Tuple[int, int] = (1, 5)) -> List[Dict]:
        """
        Generate batch of dynamic challenges.
        
        Args:
            num_challenges: Number to generate
            topics: List of topics (None for all)
            difficulty_range: Min and max difficulty
            
        Returns:
            List of generated challenges
        """
        topics = topics or ['math', 'logic', 'knowledge', 'coding', 'pattern']
        challenges = []
        
        for i in range(num_challenges):
            topic = topics[i % len(topics)]
            difficulty = random.randint(*difficulty_range)
            
            challenge = self.generate_dynamic_challenge(topic, difficulty)
            challenge['index'] = i
            challenge['id'] = hashlib.md5(
                f"{challenge['prompt']}_{i}".encode()
            ).hexdigest()[:12]
            
            challenges.append(challenge)
        
        return challenges


# Utility functions
def create_challenge_suite(num_challenges: int = 20,
                          use_dynamic: bool = True,
                          difficulty_curve: str = 'adaptive',
                          categories: Optional[List[str]] = None) -> List[Dict]:
    """
    Create a comprehensive challenge suite.
    
    Args:
        num_challenges: Total number of challenges
        use_dynamic: Whether to include dynamically generated challenges
        difficulty_curve: Difficulty progression type
        categories: Categories to include
        
    Returns:
        List of challenges
    """
    challenges = []
    
    # Template challenges
    template_gen = TemplateChallenger(difficulty_curve=difficulty_curve)
    num_template = num_challenges // 2 if use_dynamic else num_challenges
    
    template_challenges = template_gen.generate_challenge_set(
        num_challenges=num_template,
        categories=categories
    )
    challenges.extend(template_challenges)
    
    # Dynamic challenges
    if use_dynamic:
        dynamic_gen = DynamicChallengeGenerator()
        num_dynamic = num_challenges - num_template
        
        dynamic_challenges = dynamic_gen.generate_batch(
            num_challenges=num_dynamic,
            topics=categories
        )
        challenges.extend(dynamic_challenges)
    
    # Shuffle for variety
    random.shuffle(challenges)
    
    # Re-index
    for i, challenge in enumerate(challenges):
        challenge['index'] = i
    
    return challenges


def evaluate_model_performance(model_responses: List[str],
                              challenges: List[Dict],
                              fuzzy_threshold: float = 0.85) -> Dict[str, Any]:
    """
    Evaluate model performance on challenge suite.
    
    Args:
        model_responses: List of model responses
        challenges: List of challenges
        fuzzy_threshold: Threshold for fuzzy matching
        
    Returns:
        Performance statistics
    """
    evaluator = ChallengeEvaluator(fuzzy_threshold=fuzzy_threshold)
    results = evaluator.batch_evaluate(model_responses, challenges)
    
    # Calculate statistics
    total = len(results)
    successes = sum(r.success for r in results)
    avg_score = sum(r.score for r in results) / total if total > 0 else 0
    
    # By category
    by_category = defaultdict(lambda: {'total': 0, 'success': 0, 'score': 0})
    for result, challenge in zip(results, challenges):
        cat = challenge.get('category', 'unknown')
        by_category[cat]['total'] += 1
        if result.success:
            by_category[cat]['success'] += 1
        by_category[cat]['score'] += result.score
    
    # Calculate per-category averages
    for cat in by_category:
        total_cat = by_category[cat]['total']
        if total_cat > 0:
            by_category[cat]['success_rate'] = by_category[cat]['success'] / total_cat
            by_category[cat]['avg_score'] = by_category[cat]['score'] / total_cat
    
    return {
        'total': total,
        'successes': successes,
        'success_rate': successes / total if total > 0 else 0,
        'avg_score': avg_score,
        'by_category': dict(by_category),
        'results': results
    }