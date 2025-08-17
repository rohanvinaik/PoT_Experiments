"""
Tests for Template Challenge System
"""

import unittest
import random
import re
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

# Import the classes to test
from pot.lm.template_challenges import (
    TemplateChallenger,
    ChallengeEvaluator,
    DynamicChallengeGenerator,
    ChallengeResult,
    create_challenge_suite,
    evaluate_model_performance
)


class TestTemplateChallenger(unittest.TestCase):
    """Test suite for TemplateChallenger"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.challenger = TemplateChallenger(difficulty_curve='linear', seed=42)
    
    def test_initialization(self):
        """Test challenger initialization"""
        self.assertEqual(self.challenger.difficulty_curve, 'linear')
        self.assertIsNotNone(self.challenger.templates)
        self.assertEqual(len(self.challenger.challenge_history), 0)
    
    def test_template_loading(self):
        """Test that templates are loaded correctly"""
        templates = self.challenger.templates
        
        # Check categories exist
        expected_categories = ['factual', 'reasoning', 'arithmetic', 'completion']
        for category in expected_categories:
            self.assertIn(category, templates)
            self.assertIsInstance(templates[category], list)
            self.assertGreater(len(templates[category]), 0)
        
        # Check template structure
        for category, template_list in templates.items():
            for template in template_list:
                self.assertIn('prompt', template)
                self.assertIn('expected', template)
                self.assertIn('type', template)
                self.assertIn('difficulty', template)
    
    def test_generate_challenge_set(self):
        """Test challenge set generation"""
        challenges = self.challenger.generate_challenge_set(
            num_challenges=10,
            categories=['arithmetic', 'factual']
        )
        
        self.assertEqual(len(challenges), 10)
        
        for challenge in challenges:
            self.assertIn('prompt', challenge)
            self.assertIn('expected', challenge)
            self.assertIn('category', challenge)
            self.assertIn('id', challenge)
            self.assertIn('index', challenge)
            self.assertIn(challenge['category'], ['arithmetic', 'factual'])
    
    def test_linear_difficulty_curve(self):
        """Test linear difficulty progression"""
        challenger = TemplateChallenger(difficulty_curve='linear')
        difficulties = []
        
        for i in range(10):
            diff = challenger._compute_difficulty(i, 10, 1, 5)
            difficulties.append(diff)
        
        # Should progress from min to max
        self.assertEqual(difficulties[0], 1)
        self.assertEqual(difficulties[-1], 5)
        
        # Should be non-decreasing
        for i in range(1, len(difficulties)):
            self.assertGreaterEqual(difficulties[i], difficulties[i-1])
    
    def test_exponential_difficulty_curve(self):
        """Test exponential difficulty progression"""
        challenger = TemplateChallenger(difficulty_curve='exponential')
        difficulties = []
        
        for i in range(10):
            diff = challenger._compute_difficulty(i, 10, 1, 5)
            difficulties.append(diff)
        
        # Should start low and increase
        self.assertEqual(difficulties[0], 1)
        self.assertGreaterEqual(difficulties[-1], 3)
    
    def test_random_difficulty_curve(self):
        """Test random difficulty selection"""
        challenger = TemplateChallenger(difficulty_curve='random', seed=42)
        difficulties = []
        
        for i in range(10):
            diff = challenger._compute_difficulty(i, 10, 1, 5)
            difficulties.append(diff)
        
        # All should be within range
        for diff in difficulties:
            self.assertGreaterEqual(diff, 1)
            self.assertLessEqual(diff, 5)
    
    def test_adaptive_difficulty(self):
        """Test adaptive difficulty adjustment"""
        challenger = TemplateChallenger(difficulty_curve='adaptive')
        
        # Add successful results
        for _ in range(5):
            result = ChallengeResult(
                success=True, score=1.0, match_type='exact',
                response='correct', expected='correct', difficulty=2
            )
            challenger.add_to_history(result)
        
        # Difficulty should increase
        new_diff = challenger._adaptive_difficulty(1, 5)
        self.assertGreater(new_diff, 2)
        
        # Add failed results
        for _ in range(5):
            result = ChallengeResult(
                success=False, score=0.0, match_type='exact',
                response='wrong', expected='correct', difficulty=3
            )
            challenger.add_to_history(result)
        
        # Difficulty should decrease
        new_diff = challenger._adaptive_difficulty(1, 5)
        self.assertLess(new_diff, 3)
    
    def test_challenge_selection(self):
        """Test challenge selection by category and difficulty"""
        challenge = self.challenger._select_challenge('arithmetic', 1)
        
        self.assertIsNotNone(challenge)
        self.assertIn('prompt', challenge)
        self.assertLessEqual(abs(challenge['difficulty'] - 1), 1)
    
    def test_statistics(self):
        """Test statistics calculation"""
        # Add some results
        for i in range(10):
            result = ChallengeResult(
                success=i % 2 == 0,
                score=0.5 if i % 2 == 0 else 0.0,
                match_type='exact',
                response='test',
                expected='test',
                difficulty=(i % 3) + 1,
                time_taken=0.1
            )
            self.challenger.add_to_history(result)
        
        stats = self.challenger.get_statistics()
        
        self.assertEqual(stats['total'], 10)
        self.assertEqual(stats['success_rate'], 0.5)
        self.assertIn('by_difficulty', stats)
        self.assertAlmostEqual(stats['avg_time'], 0.1, places=5)


class TestChallengeEvaluator(unittest.TestCase):
    """Test suite for ChallengeEvaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = ChallengeEvaluator(fuzzy_threshold=0.85)
    
    def test_exact_evaluation(self):
        """Test exact match evaluation"""
        challenge = {
            'prompt': 'Test prompt',
            'expected': 'correct answer',
            'type': 'exact',
            'difficulty': 1
        }
        
        # Exact match
        result = self.evaluator.evaluate_response('correct answer', challenge)
        self.assertTrue(result.success)
        self.assertEqual(result.score, 1.0)
        
        # Wrong answer
        result = self.evaluator.evaluate_response('wrong answer', challenge)
        self.assertFalse(result.success)
        self.assertEqual(result.score, 0.0)
        
        # Case insensitive (non-strict mode)
        result = self.evaluator.evaluate_response('CORRECT ANSWER', challenge)
        self.assertTrue(result.success)
    
    def test_regex_evaluation(self):
        """Test regex pattern evaluation"""
        challenge = {
            'prompt': 'Water formula',
            'expected': r'H2O|H₂O',
            'type': 'regex',
            'difficulty': 1
        }
        
        # Matching patterns
        for response in ['H2O', 'H₂O', 'h2o']:
            result = self.evaluator.evaluate_response(response, challenge)
            self.assertTrue(result.success)
            self.assertEqual(result.score, 1.0)
        
        # Non-matching
        result = self.evaluator.evaluate_response('H2O2', challenge)
        self.assertFalse(result.success)
    
    def test_semantic_evaluation(self):
        """Test semantic similarity evaluation"""
        challenge = {
            'prompt': 'Roses need',
            'expected': ['need water', 'require water', 'need watering'],
            'type': 'semantic',
            'difficulty': 2
        }
        
        # Should match semantically similar responses
        result = self.evaluator.evaluate_response('need water', challenge)
        self.assertTrue(result.success)
        
        # Without fuzzy matcher, falls back to contains
        if not self.evaluator.fuzzy_matcher:
            result = self.evaluator.evaluate_response('they need water', challenge)
            self.assertTrue(result.success)
    
    def test_contains_evaluation(self):
        """Test contains evaluation"""
        challenge = {
            'prompt': 'List items',
            'expected': ['apple', 'banana', 'orange'],
            'type': 'contains',
            'difficulty': 2
        }
        
        # Contains all terms
        result = self.evaluator.evaluate_response(
            'I have an apple, banana, and orange',
            challenge
        )
        self.assertTrue(result.success)
        self.assertEqual(result.score, 1.0)
        
        # Contains some terms
        result = self.evaluator.evaluate_response(
            'I have an apple and banana',
            challenge
        )
        self.assertAlmostEqual(result.score, 2/3, places=5)
        self.assertFalse(result.success)  # Below 80% threshold
        
        # Contains none
        result = self.evaluator.evaluate_response(
            'I have grapes',
            challenge
        )
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.success)
    
    def test_numeric_evaluation(self):
        """Test numeric evaluation with tolerance"""
        challenge = {
            'prompt': 'Calculate',
            'expected': '100',
            'type': 'numeric',
            'tolerance': 0.01,
            'difficulty': 2
        }
        
        # Exact match
        result = self.evaluator.evaluate_response('100', challenge)
        self.assertTrue(result.success)
        self.assertEqual(result.score, 1.0)
        
        # Within tolerance
        result = self.evaluator.evaluate_response('100.5', challenge)
        self.assertTrue(result.success)
        self.assertGreater(result.score, 0.99)
        
        # Outside tolerance
        result = self.evaluator.evaluate_response('110', challenge)
        self.assertFalse(result.success)
        
        # Extract number from text
        result = self.evaluator.evaluate_response('The answer is 100', challenge)
        self.assertTrue(result.success)
    
    def test_response_cleaning(self):
        """Test response cleaning"""
        # Remove markers
        cleaned = self.evaluator._clean_response('[MASK] answer ')
        self.assertEqual(cleaned, 'answer')
        
        # Normalize quotes
        cleaned = self.evaluator._clean_response('"test"')
        self.assertEqual(cleaned, '"test"')
    
    def test_batch_evaluation(self):
        """Test batch evaluation"""
        challenges = [
            {'prompt': 'Q1', 'expected': 'A1', 'type': 'exact', 'difficulty': 1},
            {'prompt': 'Q2', 'expected': 'A2', 'type': 'exact', 'difficulty': 1}
        ]
        responses = ['A1', 'wrong']
        
        results = self.evaluator.batch_evaluate(responses, challenges)
        
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)
    
    def test_strict_mode(self):
        """Test strict mode evaluation"""
        evaluator_strict = ChallengeEvaluator(strict_mode=True)
        
        challenge = {
            'prompt': 'Answer',
            'expected': 'Test',
            'type': 'exact',
            'difficulty': 1
        }
        
        # Case sensitive in strict mode
        result = evaluator_strict.evaluate_response('test', challenge)
        self.assertFalse(result.success)
        
        result = evaluator_strict.evaluate_response('Test', challenge)
        self.assertTrue(result.success)


class TestDynamicChallengeGenerator(unittest.TestCase):
    """Test suite for DynamicChallengeGenerator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = DynamicChallengeGenerator(seed=42)
    
    def test_math_challenge_generation(self):
        """Test math challenge generation"""
        for difficulty in range(1, 6):
            challenge = self.generator._generate_math_challenge(difficulty)
            
            self.assertIn('prompt', challenge)
            self.assertIn('expected', challenge)
            self.assertIn('[MASK]', challenge['prompt'])
            self.assertEqual(challenge['difficulty'], difficulty)
            self.assertEqual(challenge['category'], 'math')
            
            # Verify answer is correct (for simple cases)
            if difficulty == 1 and '+' in challenge['prompt']:
                # Extract numbers and verify
                numbers = re.findall(r'\d+', challenge['prompt'])
                if len(numbers) >= 2:
                    a, b = int(numbers[0]), int(numbers[1])
                    if '+' in challenge['prompt']:
                        self.assertEqual(challenge['expected'], str(a + b))
    
    def test_logic_challenge_generation(self):
        """Test logic challenge generation"""
        for difficulty in [1, 2, 3]:
            challenge = self.generator._generate_logic_challenge(difficulty)
            
            self.assertIn('prompt', challenge)
            self.assertIn('expected', challenge)
            self.assertIn('[MASK]', challenge['prompt'])
            self.assertEqual(challenge['category'], 'logic')
    
    def test_knowledge_challenge_generation(self):
        """Test knowledge challenge generation"""
        challenge = self.generator._generate_knowledge_challenge(1)
        
        self.assertIn('prompt', challenge)
        self.assertIn('expected', challenge)
        self.assertIn('[MASK]', challenge['prompt'])
        self.assertEqual(challenge['category'], 'knowledge')
    
    def test_coding_challenge_generation(self):
        """Test coding challenge generation"""
        challenge = self.generator._generate_coding_challenge(1, 'Python')
        
        self.assertIn('prompt', challenge)
        self.assertIn('expected', challenge)
        self.assertIn('[MASK]', challenge['prompt'])
        self.assertEqual(challenge['category'], 'coding')
    
    def test_pattern_challenge_generation(self):
        """Test pattern challenge generation"""
        for difficulty in range(1, 5):
            challenge = self.generator._generate_pattern_challenge(difficulty)
            
            self.assertIn('prompt', challenge)
            self.assertIn('expected', challenge)
            self.assertIn('[MASK]', challenge['prompt'])
            self.assertIn('sequence', challenge['prompt'].lower())
            self.assertEqual(challenge['category'], 'pattern')
    
    def test_dynamic_challenge_generation(self):
        """Test main dynamic challenge generation"""
        topics = ['math', 'logic', 'knowledge', 'coding', 'pattern']
        
        for topic in topics:
            challenge = self.generator.generate_dynamic_challenge(topic, 2)
            
            self.assertIn('prompt', challenge)
            self.assertIn('expected', challenge)
            self.assertEqual(challenge['category'], topic)
            self.assertTrue(challenge.get('generated', False))
    
    def test_batch_generation(self):
        """Test batch challenge generation"""
        challenges = self.generator.generate_batch(
            num_challenges=20,
            topics=['math', 'logic'],
            difficulty_range=(1, 3)
        )
        
        self.assertEqual(len(challenges), 20)
        
        for challenge in challenges:
            self.assertIn('id', challenge)
            self.assertIn('index', challenge)
            self.assertIn(challenge['category'], ['math', 'logic'])
            self.assertGreaterEqual(challenge['difficulty'], 1)
            self.assertLessEqual(challenge['difficulty'], 3)
    
    def test_reproducibility(self):
        """Test that same seed produces same challenges"""
        gen1 = DynamicChallengeGenerator(seed=123)
        gen2 = DynamicChallengeGenerator(seed=123)
        
        challenge1 = gen1.generate_dynamic_challenge('math', 2)
        challenge2 = gen2.generate_dynamic_challenge('math', 2)
        
        self.assertEqual(challenge1['prompt'], challenge2['prompt'])
        self.assertEqual(challenge1['expected'], challenge2['expected'])


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_create_challenge_suite(self):
        """Test challenge suite creation"""
        suite = create_challenge_suite(
            num_challenges=20,
            use_dynamic=True,
            difficulty_curve='linear',
            categories=['math', 'logic']
        )
        
        self.assertEqual(len(suite), 20)
        
        # Check mix of template and dynamic
        has_template = any(not c.get('generated', False) for c in suite)
        has_dynamic = any(c.get('generated', False) for c in suite)
        
        self.assertTrue(has_template or has_dynamic)
        
        # Check indexing
        for i, challenge in enumerate(suite):
            self.assertEqual(challenge['index'], i)
    
    def test_create_challenge_suite_template_only(self):
        """Test template-only suite creation"""
        suite = create_challenge_suite(
            num_challenges=10,
            use_dynamic=False,
            difficulty_curve='linear'
        )
        
        self.assertEqual(len(suite), 10)
        
        # Should all be template challenges
        for challenge in suite:
            self.assertFalse(challenge.get('generated', False))
    
    def test_evaluate_model_performance(self):
        """Test model performance evaluation"""
        challenges = [
            {'prompt': 'Q1', 'expected': 'A1', 'type': 'exact', 
             'category': 'test', 'difficulty': 1},
            {'prompt': 'Q2', 'expected': 'A2', 'type': 'exact',
             'category': 'test', 'difficulty': 1},
            {'prompt': 'Q3', 'expected': 'A3', 'type': 'exact',
             'category': 'other', 'difficulty': 2}
        ]
        
        responses = ['A1', 'wrong', 'A3']
        
        performance = evaluate_model_performance(responses, challenges)
        
        self.assertEqual(performance['total'], 3)
        self.assertEqual(performance['successes'], 2)
        self.assertAlmostEqual(performance['success_rate'], 2/3, places=5)
        
        # Check category breakdown
        self.assertIn('test', performance['by_category'])
        self.assertIn('other', performance['by_category'])
        
        test_cat = performance['by_category']['test']
        self.assertEqual(test_cat['total'], 2)
        self.assertEqual(test_cat['success'], 1)
        self.assertEqual(test_cat['success_rate'], 0.5)


class TestChallengeResult(unittest.TestCase):
    """Test ChallengeResult dataclass"""
    
    def test_challenge_result_creation(self):
        """Test ChallengeResult creation"""
        result = ChallengeResult(
            success=True,
            score=0.95,
            match_type='semantic',
            response='test response',
            expected='expected response',
            details={'key': 'value'},
            time_taken=0.5,
            difficulty=3
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.match_type, 'semantic')
        self.assertEqual(result.response, 'test response')
        self.assertEqual(result.expected, 'expected response')
        self.assertEqual(result.details['key'], 'value')
        self.assertEqual(result.time_taken, 0.5)
        self.assertEqual(result.difficulty, 3)
    
    def test_default_values(self):
        """Test default values in ChallengeResult"""
        result = ChallengeResult(
            success=False,
            score=0.0,
            match_type='exact',
            response='',
            expected=''
        )
        
        self.assertEqual(result.details, {})
        self.assertEqual(result.time_taken, 0.0)
        self.assertEqual(result.difficulty, 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_full_challenge_workflow(self):
        """Test complete challenge generation and evaluation workflow"""
        # Generate challenges
        challenger = TemplateChallenger(difficulty_curve='linear')
        challenges = challenger.generate_challenge_set(num_challenges=5)
        
        # Create mock responses
        evaluator = ChallengeEvaluator()
        responses = []
        
        for challenge in challenges:
            # Simulate model response (50% correct)
            if random.random() > 0.5:
                responses.append(str(challenge['expected']))
            else:
                responses.append('wrong answer')
        
        # Evaluate responses
        results = evaluator.batch_evaluate(responses, challenges)
        
        # Add to history for adaptive difficulty
        for result in results:
            challenger.add_to_history(result)
        
        # Check statistics
        stats = challenger.get_statistics()
        self.assertEqual(stats['total'], 5)
        self.assertGreaterEqual(stats['success_rate'], 0.0)
        self.assertLessEqual(stats['success_rate'], 1.0)
    
    def test_mixed_suite_evaluation(self):
        """Test evaluation of mixed template and dynamic challenges"""
        suite = create_challenge_suite(
            num_challenges=10,
            use_dynamic=True,
            difficulty_curve='adaptive'
        )
        
        # Generate perfect responses
        responses = []
        for challenge in suite:
            if isinstance(challenge['expected'], list):
                responses.append(challenge['expected'][0])
            else:
                responses.append(str(challenge['expected']))
        
        # Evaluate
        performance = evaluate_model_performance(responses, suite)
        
        # Should have high success rate
        self.assertGreater(performance['success_rate'], 0.8)
        
        # Check all categories are evaluated
        for category in performance['by_category']:
            cat_stats = performance['by_category'][category]
            self.assertGreater(cat_stats['total'], 0)


if __name__ == '__main__':
    unittest.main()