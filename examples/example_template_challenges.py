#!/usr/bin/env python3
"""
Example usage of the Template Challenge System for Language Model Verification
"""

import json
from pot.lm.template_challenges import (
    TemplateChallenger,
    ChallengeEvaluator, 
    DynamicChallengeGenerator,
    create_challenge_suite,
    evaluate_model_performance
)


def main():
    print("=" * 60)
    print("Template Challenge System Demo")
    print("=" * 60)
    
    # 1. Create template-based challenges
    print("\n1. Template-Based Challenges:")
    print("-" * 40)
    
    challenger = TemplateChallenger(difficulty_curve='linear')
    challenges = challenger.generate_challenge_set(
        num_challenges=5,
        categories=['arithmetic', 'factual', 'completion']
    )
    
    for i, challenge in enumerate(challenges, 1):
        print(f"\nChallenge {i}:")
        print(f"  Category: {challenge['category']}")
        print(f"  Difficulty: {challenge['difficulty']}")
        print(f"  Prompt: {challenge['prompt']}")
        print(f"  Expected: {challenge['expected']}")
        print(f"  Type: {challenge['type']}")
    
    # 2. Dynamic challenge generation
    print("\n\n2. Dynamically Generated Challenges:")
    print("-" * 40)
    
    dynamic_gen = DynamicChallengeGenerator()
    
    for topic in ['math', 'logic', 'pattern']:
        for difficulty in [1, 3]:
            challenge = dynamic_gen.generate_dynamic_challenge(topic, difficulty)
            print(f"\n{topic.capitalize()} (Difficulty {difficulty}):")
            print(f"  Prompt: {challenge['prompt']}")
            print(f"  Expected: {challenge['expected']}")
    
    # 3. Challenge evaluation
    print("\n\n3. Challenge Evaluation:")
    print("-" * 40)
    
    evaluator = ChallengeEvaluator(fuzzy_threshold=0.85)
    
    test_challenges = [
        {
            'prompt': 'The capital of France is [MASK]',
            'expected': 'Paris',
            'type': 'exact',
            'difficulty': 1
        },
        {
            'prompt': 'Calculate: 7 × 8 = [MASK]',
            'expected': '56',
            'type': 'exact',
            'difficulty': 1
        },
        {
            'prompt': 'The chemical formula for water is [MASK]',
            'expected': r'H2O|H₂O',
            'type': 'regex',
            'difficulty': 1
        }
    ]
    
    # Simulate model responses
    model_responses = ['Paris', '56', 'H2O']
    
    print("\nEvaluating model responses:")
    for challenge, response in zip(test_challenges, model_responses):
        result = evaluator.evaluate_response(response, challenge)
        print(f"\nPrompt: {challenge['prompt']}")
        print(f"Response: {response}")
        print(f"Success: {result.success}")
        print(f"Score: {result.score:.2f}")
    
    # 4. Adaptive difficulty
    print("\n\n4. Adaptive Difficulty Adjustment:")
    print("-" * 40)
    
    adaptive_challenger = TemplateChallenger(difficulty_curve='adaptive')
    
    print("\nInitial difficulty:", adaptive_challenger.current_difficulty)
    
    # Simulate successful challenges
    print("\nAdding 5 successful results...")
    for _ in range(5):
        result = evaluator.evaluate_response('correct', {
            'expected': 'correct', 'type': 'exact', 'difficulty': 2
        })
        adaptive_challenger.add_to_history(result)
    
    new_diff = adaptive_challenger._adaptive_difficulty(1, 5)
    print(f"New difficulty after successes: {new_diff}")
    
    # Simulate failed challenges
    print("\nAdding 5 failed results...")
    for _ in range(5):
        result = evaluator.evaluate_response('wrong', {
            'expected': 'correct', 'type': 'exact', 'difficulty': new_diff
        })
        adaptive_challenger.add_to_history(result)
    
    final_diff = adaptive_challenger._adaptive_difficulty(1, 5)
    print(f"Final difficulty after failures: {final_diff}")
    
    # 5. Complete challenge suite
    print("\n\n5. Complete Challenge Suite:")
    print("-" * 40)
    
    suite = create_challenge_suite(
        num_challenges=10,
        use_dynamic=True,
        difficulty_curve='linear',
        categories=['math', 'logic', 'factual']
    )
    
    print(f"\nGenerated {len(suite)} challenges:")
    
    # Count by category
    category_counts = {}
    for challenge in suite:
        cat = challenge.get('category', 'unknown')
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nChallenge distribution:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count}")
    
    # 6. Performance evaluation
    print("\n\n6. Model Performance Evaluation:")
    print("-" * 40)
    
    # Simulate model responses (80% correct)
    responses = []
    for challenge in suite[:5]:  # Test on first 5
        if isinstance(challenge['expected'], list):
            # Semantic challenge - pick first option
            responses.append(challenge['expected'][0] if len(responses) < 4 else 'wrong')
        else:
            # Exact/regex - use correct answer most of the time
            responses.append(str(challenge['expected']) if len(responses) < 4 else 'wrong')
    
    performance = evaluate_model_performance(
        responses, 
        suite[:5],
        fuzzy_threshold=0.85
    )
    
    print(f"\nPerformance on {performance['total']} challenges:")
    print(f"  Success rate: {performance['success_rate']:.1%}")
    print(f"  Average score: {performance['avg_score']:.2f}")
    
    if performance['by_category']:
        print("\nBy category:")
        for cat, stats in performance['by_category'].items():
            if stats['total'] > 0:
                print(f"  {cat}: {stats.get('success_rate', 0):.1%} "
                      f"({stats['success']}/{stats['total']})")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()