#!/usr/bin/env python3
"""
Example demonstrating the complete LM Verification integration with configuration
"""

import json
from pot.lm.lm_config import LMVerifierConfig, PresetConfigs
from pot.lm.verifier import LMVerifier
from pot.lm.template_challenges import TemplateChallenger, ChallengeEvaluator
from pot.lm.sequential_tester import SequentialTester, SequentialVerificationSession


def demo_configuration_system():
    """Demonstrate the configuration system"""
    print("=" * 60)
    print("1. Configuration System Demo")
    print("=" * 60)
    
    # Show default configuration
    default_config = LMVerifierConfig()
    print(f"\nDefault Configuration:")
    print(f"  Challenges: {default_config.num_challenges}")
    print(f"  Method: {default_config.verification_method}")
    print(f"  Threshold: {default_config.distance_threshold}")
    print(f"  SPRT Parameters: α={default_config.sprt_alpha}, β={default_config.sprt_beta}")
    
    # Show preset configurations
    print(f"\nPreset Configurations:")
    presets = ['quick_test', 'standard_verification', 'comprehensive_verification']
    for preset_name in presets:
        config = getattr(PresetConfigs, preset_name)()
        print(f"  {preset_name}: {config.num_challenges} challenges, {config.verification_method} method")
    
    # Create custom configuration
    custom_config = LMVerifierConfig(
        num_challenges=15,
        verification_method='sequential',
        sprt_alpha=0.03,
        sprt_beta=0.03,
        difficulty_curve='linear',
        fuzzy_threshold=0.9
    )
    
    print(f"\nCustom Configuration:")
    print(f"  Challenges: {custom_config.num_challenges}")
    print(f"  Higher precision: α={custom_config.sprt_alpha}, β={custom_config.sprt_beta}")
    print(f"  Stricter fuzzy matching: {custom_config.fuzzy_threshold}")
    print(f"  Valid: {custom_config.is_valid()}")
    
    return custom_config


def demo_enhanced_verification(config):
    """Demonstrate enhanced verification with configuration"""
    print("\n" + "=" * 60)
    print("2. Enhanced Verification Demo")
    print("=" * 60)
    
    # Create mock model
    class MockLanguageModel:
        def __init__(self, name, accuracy=0.75):
            self.name = name
            self.accuracy = accuracy
        
        def generate(self, prompt, max_new_tokens=64):
            import random
            random.seed(hash(prompt) % 1000)  # Deterministic based on prompt
            
            # Simple heuristic responses with specified accuracy
            if random.random() < self.accuracy:
                # Correct responses
                if "capital" in prompt.lower():
                    if "france" in prompt.lower():
                        return "Paris"
                    elif "japan" in prompt.lower():
                        return "Tokyo"
                elif any(op in prompt for op in ['+', '×', '=', 'calculate']):
                    if "7 × 8" in prompt:
                        return "56"
                    elif "2^10" in prompt:
                        return "1024"
                elif "water" in prompt.lower() and "formula" in prompt.lower():
                    return "H2O"
                elif "question" in prompt.lower():
                    return "question"
                
            # Default/wrong responses
            return "unknown"
    
    # Create models with different performance levels
    models = [
        MockLanguageModel("High-Performance Model", 0.85),
        MockLanguageModel("Medium-Performance Model", 0.65),
        MockLanguageModel("Low-Performance Model", 0.45)
    ]
    
    print(f"Testing {len(models)} mock models with configuration:")
    print(f"  Method: {config.verification_method}")
    print(f"  Challenges: {config.num_challenges}")
    print(f"  Early stopping: {config.verification_method == 'sequential'}")
    
    results = []
    
    for model in models:
        print(f"\nVerifying {model.name} (Expected accuracy: {model.accuracy:.0%})...")
        
        # Create verification components
        challenger = TemplateChallenger(difficulty_curve=config.difficulty_curve)
        evaluator = ChallengeEvaluator(fuzzy_threshold=config.fuzzy_threshold)
        
        if config.verification_method == 'sequential':
            tester = SequentialTester(
                alpha=config.sprt_alpha,
                beta=config.sprt_beta,
                p0=config.sprt_p0,
                p1=config.sprt_p1,
                max_trials=config.max_trials,
                min_trials=config.min_trials
            )
            
            # Model runner function
            def model_runner(prompt):
                return model.generate(prompt)
            
            # Create verification session
            session = SequentialVerificationSession(
                tester=tester,
                challenger=challenger,
                evaluator=evaluator,
                model_runner=model_runner
            )
            
            # Run verification
            result = session.run_verification(
                max_challenges=config.num_challenges,
                early_stop=True
            )
            
        else:
            # Batch verification
            challenges = challenger.generate_challenge_set(
                num_challenges=config.num_challenges,
                categories=config.challenge_types
            )
            
            total_score = 0.0
            challenge_results = []
            
            for challenge in challenges:
                response = model.generate(challenge['prompt'])
                eval_result = evaluator.evaluate_response(response, challenge)
                challenge_results.append(eval_result)
                total_score += eval_result.score
            
            success_rate = sum(1 for r in challenge_results if r.success) / len(challenge_results)
            
            result = {
                'verified': success_rate >= (1.0 - config.distance_threshold),
                'success_rate': success_rate,
                'avg_score': total_score / len(challenge_results),
                'num_trials': len(challenges),
                'method': 'batch'
            }
        
        # Store and display result
        result['model_name'] = model.name
        result['expected_accuracy'] = model.accuracy
        results.append(result)
        
        verified = result.get('verified', False)
        success_rate = result.get('success_rate', 0.0)
        trials = result.get('num_trials', 0)
        
        print(f"  Result: {'✓ GENUINE' if verified else '✗ SUSPICIOUS'}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Trials used: {trials}")
        
        if config.verification_method == 'sequential':
            early_stopped = result.get('early_stopped', False)
            print(f"  Early stopped: {early_stopped}")
    
    return results


def demo_configuration_persistence():
    """Demonstrate configuration file operations"""
    print("\n" + "=" * 60)
    print("3. Configuration Persistence Demo")
    print("=" * 60)
    
    # Create configurations for different scenarios
    scenarios = {
        'development': PresetConfigs.quick_test(),
        'testing': PresetConfigs.standard_verification(),
        'production': PresetConfigs.high_security()
    }
    
    print("Saving configurations for different scenarios:")
    
    for scenario, config in scenarios.items():
        filename = f"config_{scenario}.json"
        config.save_to_file(filename)
        print(f"  ✓ {scenario}: {filename}")
        
        # Verify we can load it back
        loaded = LMVerifierConfig.from_file(filename)
        assert loaded.to_dict() == config.to_dict(), f"Round-trip failed for {scenario}"
    
    print("\nConfiguration files created and verified!")
    
    # Show how to modify configurations
    print("\nModifying configuration example:")
    base_config = PresetConfigs.standard_verification()
    
    # Create a stricter version
    strict_config = LMVerifierConfig.from_dict(base_config.to_dict())
    strict_config.sprt_alpha = 0.01
    strict_config.sprt_beta = 0.01
    strict_config.fuzzy_threshold = 0.95
    strict_config.distance_threshold = 0.05
    
    print(f"  Base config: α={base_config.sprt_alpha}, threshold={base_config.distance_threshold}")
    print(f"  Strict config: α={strict_config.sprt_alpha}, threshold={strict_config.distance_threshold}")
    
    strict_config.save_to_file("config_strict.yaml")
    print("  ✓ Strict configuration saved to YAML")


def demo_cli_integration():
    """Demonstrate CLI integration"""
    print("\n" + "=" * 60)
    print("4. CLI Integration Demo")
    print("=" * 60)
    
    print("The CLI provides several commands for practical usage:")
    print()
    
    cli_examples = [
        ("List presets", "python pot/lm/cli.py list-presets"),
        ("Create config", "python pot/lm/cli.py create-config --preset quick_test --output my_config.yaml"),
        ("Validate config", "python pot/lm/cli.py validate-config --config my_config.yaml"),
        ("Show config", "python pot/lm/cli.py show-config my_config.yaml"),
        ("Run verification", "python pot/lm/cli.py verify --model gpt2 --config quick_test --output results.json"),
        ("Compare models", "python pot/lm/cli.py compare --model1 gpt2 --model2 distilgpt2 --config standard_verification")
    ]
    
    for description, command in cli_examples:
        print(f"  {description:20s}: {command}")
    
    print("\nThe CLI supports:")
    print("  • Multiple configuration formats (JSON, YAML)")
    print("  • Preset configurations for common scenarios")
    print("  • Configuration validation and inspection")
    print("  • Model comparison capabilities")
    print("  • Detailed result output and logging")


def main():
    print("=" * 80)
    print("LM Verification Integration and Configuration Demo")
    print("=" * 80)
    
    # Run demonstrations
    config = demo_configuration_system()
    results = demo_enhanced_verification(config)
    demo_configuration_persistence()
    demo_cli_integration()
    
    # Summary
    print("\n" + "=" * 80)
    print("INTEGRATION SUMMARY")
    print("=" * 80)
    
    print("\nVerification Results Summary:")
    for result in results:
        model_name = result['model_name']
        verified = result.get('verified', False)
        expected = result['expected_accuracy']
        actual = result.get('success_rate', 0.0)
        print(f"  {model_name:25s}: {'✓' if verified else '✗'} "
              f"(Expected: {expected:.0%}, Actual: {actual:.0%})")
    
    print("\nKey Integration Features:")
    print("  ✓ Flexible configuration system with presets")
    print("  ✓ Enhanced template challenge generation") 
    print("  ✓ Sequential testing with early stopping")
    print("  ✓ Batch verification for comprehensive testing")
    print("  ✓ Configuration file I/O (JSON/YAML)")
    print("  ✓ CLI interface for practical usage")
    print("  ✓ Validation and error checking")
    print("  ✓ Modular and extensible design")
    
    print("\nThe system is now ready for production use!")
    print("Use the CLI or Python API to verify language models with confidence.")


if __name__ == "__main__":
    main()