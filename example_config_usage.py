#!/usr/bin/env python3
"""
Demonstration of configuration usage for attack resistance module.
"""

import yaml
import json
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Any
import sys

# Add path
sys.path.insert(0, str(Path(__file__).parent))

from pot.core.attack_suites import AttackConfig, StandardAttackSuite
from pot.core.defenses import DefenseConfig, IntegratedDefenseSystem, MockBaseVerifier
from pot.eval.attack_benchmarks import AttackBenchmark


def load_config(config_path: str = "pot/config/attack_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def demonstrate_attack_config():
    """Demonstrate using attack configuration."""
    print("="*60)
    print("ATTACK CONFIGURATION DEMO")
    print("="*60)
    
    # Load configuration
    config = load_config()
    
    # Access distillation settings
    distillation_config = config['attacks']['distillation']
    print("\nDistillation Configuration:")
    print(f"  Temperatures: {distillation_config['temperatures']}")
    print(f"  Alpha values: {distillation_config['alpha_values']}")
    print(f"  Default epochs: {distillation_config['default_epochs']}")
    print(f"  Learning rates: {distillation_config['learning_rates']}")
    
    # Access compression settings
    compression_config = config['attacks']['compression']
    print("\nCompression Configuration:")
    print(f"  Pruning ratios: {compression_config['pruning_ratios']}")
    print(f"  Quantization bits: {compression_config['quantization_bits']}")
    print(f"  Fine-tuning epochs: {compression_config['fine_tuning_epochs']}")
    
    # Access budget settings
    budgets = config['attacks']['budgets']
    print("\nAttack Budgets:")
    for strength, budget in budgets.items():
        print(f"  {strength.capitalize()}:")
        print(f"    - Queries: {budget['queries']}")
        print(f"    - Compute hours: {budget['compute_hours']}")
        if 'memory_gb' in budget:
            print(f"    - Memory: {budget['memory_gb']} GB")


def create_attack_from_config():
    """Create attack configuration from loaded settings."""
    print("\n" + "="*60)
    print("CREATE ATTACK FROM CONFIG")
    print("="*60)
    
    # Load configuration
    config = load_config()
    
    # Create distillation attack with strong settings
    distillation = config['attacks']['distillation']
    budget = config['attacks']['budgets']['strong']
    
    attack_config = AttackConfig(
        name="config_based_distillation",
        attack_type="distillation",
        budget=budget,
        strength='strong',
        success_metrics={'accuracy_drop': 0.15},
        parameters={
            'temperature': distillation['temperatures'][3],  # Use 4th temperature
            'alpha': distillation['alpha_values'][3],  # Use 4th alpha
            'epochs': distillation['default_epochs'],
            'learning_rate': distillation['learning_rates']['strong'],
            'optimizer': 'adam',
            'optimizer_config': distillation['optimizer_configs']['adam']
        }
    )
    
    print(f"\nCreated attack: {attack_config.name}")
    print(f"  Type: {attack_config.attack_type}")
    print(f"  Temperature: {attack_config.parameters['temperature']}")
    print(f"  Alpha: {attack_config.parameters['alpha']}")
    print(f"  Budget queries: {attack_config.budget['queries']}")
    print(f"  Compute hours: {attack_config.budget['compute_hours']}")
    
    return attack_config


def demonstrate_defense_config():
    """Demonstrate defense configuration usage."""
    print("\n" + "="*60)
    print("DEFENSE CONFIGURATION DEMO")
    print("="*60)
    
    # Load configuration
    config = load_config()
    defense_config = config['defense']
    
    # Adaptive defense settings
    adaptive = defense_config['adaptive']
    print("\nAdaptive Defense Configuration:")
    print(f"  Update frequency: {adaptive['update_frequency']}")
    print(f"  Learning rate: {adaptive['learning_rate']}")
    print(f"  History size: {adaptive['history_size']}")
    print(f"  Threshold method: {adaptive['threshold_adjustment']['method']}")
    
    # Filtering settings
    filtering = defense_config['filtering']
    print("\nInput Filtering Configuration:")
    print(f"  Detection methods: {filtering['detection_methods']['statistical']['methods']}")
    print(f"  Outlier methods: {filtering['detection_methods']['outlier']['methods']}")
    print(f"  Sanitization - Blur kernels: {filtering['sanitization']['gaussian_blur']['kernel_sizes']}")
    print(f"  Quantization levels: {filtering['sanitization']['quantization']['levels']}")
    
    # Randomization settings
    randomization = defense_config['randomization']
    print("\nRandomization Configuration:")
    print(f"  Noise levels: {randomization['levels']}")
    print(f"  Distributions: {randomization['methods']['noise_injection']['distributions']}")
    print(f"  Smoothing samples: {randomization['methods']['smoothing']['num_samples']}")
    print(f"  Ensemble models: {randomization['methods']['ensemble']['num_models']}")


def create_benchmark_from_config():
    """Create benchmark configuration from settings."""
    print("\n" + "="*60)
    print("BENCHMARK CONFIGURATION DEMO")
    print("="*60)
    
    # Load configuration
    config = load_config()
    benchmark_config = config['benchmarking']
    
    print("\nBenchmark Settings:")
    print(f"  Enabled: {benchmark_config['standard_suite']['enabled']}")
    print(f"  Number of attacks: {len(benchmark_config['standard_suite']['attacks'])}")
    print(f"  Metrics tracked: {benchmark_config['metrics']}")
    print(f"  Report formats: {benchmark_config['reporting']['formats']}")
    
    # Create benchmark with config
    if benchmark_config['standard_suite']['enabled']:
        benchmark = AttackBenchmark(
            device='cpu',
            verbose=False,
            save_results='json' in benchmark_config['reporting']['formats']
        )
        
        # Override standard attacks with config
        benchmark.STANDARD_ATTACKS = benchmark_config['standard_suite']['attacks']
        
        print(f"\nConfigured benchmark with {len(benchmark.STANDARD_ATTACKS)} attacks:")
        for i, attack in enumerate(benchmark.STANDARD_ATTACKS[:5], 1):
            print(f"  {i}. {attack}")
        if len(benchmark.STANDARD_ATTACKS) > 5:
            print(f"  ... and {len(benchmark.STANDARD_ATTACKS) - 5} more")
    
    return benchmark


def demonstrate_vision_attacks():
    """Demonstrate vision-specific attack configuration."""
    print("\n" + "="*60)
    print("VISION ATTACK CONFIGURATION")
    print("="*60)
    
    # Load configuration
    config = load_config()
    vision_config = config['attacks']['vision']
    
    # Adversarial patch settings
    patch_config = vision_config['adversarial_patch']
    print("\nAdversarial Patch Configuration:")
    print(f"  Patch sizes: {patch_config['patch_sizes']}")
    print(f"  Epsilon values: {patch_config['epsilon_values']}")
    print(f"  Optimization steps: {patch_config['optimization_steps']}")
    print(f"  Optimizers: {patch_config['optimizers']}")
    print(f"  Loss functions: {patch_config['loss_functions']}")
    
    # Universal perturbation settings
    universal_config = vision_config['universal_perturbation']
    print("\nUniversal Perturbation Configuration:")
    print(f"  Epsilon: {universal_config['epsilon']}")
    print(f"  Max iterations: {universal_config['max_iterations']}")
    print(f"  Overshoot: {universal_config['overshoot']}")
    print(f"  Delta: {universal_config['delta']}")
    
    # Model extraction settings
    extraction_config = vision_config['model_extraction']
    print("\nModel Extraction Configuration:")
    print(f"  Methods: {extraction_config['methods']}")
    print(f"  Query strategies: {extraction_config['query_strategies']}")
    print(f"  Substitute architectures: {extraction_config['substitute_architectures']}")


def save_custom_config():
    """Demonstrate saving custom configuration."""
    print("\n" + "="*60)
    print("CUSTOM CONFIGURATION")
    print("="*60)
    
    # Create custom configuration
    custom_config = {
        'experiment_name': 'custom_robustness_test',
        'attacks': {
            'selected': ['distillation_moderate', 'compression_pruning_50'],
            'custom_parameters': {
                'distillation_moderate': {
                    'temperature': 5.0,
                    'epochs': 30
                },
                'compression_pruning_50': {
                    'fine_tuning_lr': 0.0001,
                    'recovery_epochs': 15
                }
            }
        },
        'defense': {
            'profile': 'high_security',
            'adaptive_enabled': True,
            'filtering_strength': 0.8,
            'randomization_level': 0.15
        },
        'evaluation': {
            'metrics': ['robustness_score', 'far', 'frr', 'accuracy'],
            'save_results': True,
            'generate_report': True,
            'create_dashboard': True
        }
    }
    
    # Save as YAML
    yaml_path = Path("custom_config.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(custom_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Saved custom configuration to {yaml_path}")
    
    # Save as JSON
    json_path = Path("custom_config.json")
    with open(json_path, 'w') as f:
        json.dump(custom_config, f, indent=2)
    
    print(f"Saved custom configuration to {json_path}")
    
    # Display configuration
    print("\nCustom Configuration:")
    print(yaml.dump(custom_config, default_flow_style=False, sort_keys=False))
    
    return custom_config


def run_configured_experiment():
    """Run experiment using configuration."""
    print("\n" + "="*60)
    print("CONFIGURED EXPERIMENT")
    print("="*60)
    
    # Load main configuration
    config = load_config()
    
    # Create simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(784, 10)
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))
    
    model = SimpleModel()
    
    # Create attack from config
    attack_config = AttackConfig(
        name="configured_attack",
        attack_type="distillation",
        budget=config['attacks']['budgets']['moderate'],
        strength='moderate',
        success_metrics={'accuracy_drop': 0.1},
        parameters={
            'temperature': config['attacks']['distillation']['temperatures'][2],
            'epochs': config['attacks']['distillation']['default_epochs']
        }
    )
    
    # Create defense from config
    defense_config = DefenseConfig(
        adaptive_threshold=config['defense']['adaptive']['threshold_adjustment']['initial_threshold'],
        input_filter_strength=0.5,
        randomization_scale=config['defense']['randomization']['levels'][1],
        update_frequency=config['defense']['adaptive']['update_frequency']
    )
    
    print(f"\nExperiment Configuration:")
    print(f"  Attack: {attack_config.name} ({attack_config.attack_type})")
    print(f"  Defense threshold: {defense_config.adaptive_threshold}")
    print(f"  Randomization scale: {defense_config.randomization_scale}")
    print(f"  Budget queries: {attack_config.budget['queries']}")
    
    # Would run actual experiment here
    print("\n✓ Configuration successfully loaded and applied")


def main():
    """Run all configuration demonstrations."""
    print("="*60)
    print("CONFIGURATION USAGE DEMONSTRATION")
    print("="*60)
    
    # Check if config file exists
    config_path = Path("pot/config/attack_config.yaml")
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        print("Please ensure the configuration file exists.")
        return
    
    # Run demonstrations
    demonstrate_attack_config()
    attack_config = create_attack_from_config()
    demonstrate_defense_config()
    benchmark = create_benchmark_from_config()
    demonstrate_vision_attacks()
    custom_config = save_custom_config()
    run_configured_experiment()
    
    # Cleanup
    for file in ['custom_config.yaml', 'custom_config.json']:
        if Path(file).exists():
            Path(file).unlink()
    
    print("\n" + "="*60)
    print("CONFIGURATION DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Takeaways:")
    print("• Configuration enables consistent experiments")
    print("• Settings can be version controlled")
    print("• Easy to modify without code changes")
    print("• Supports multiple formats (YAML, JSON)")
    print("• Integrates with all module components")


if __name__ == "__main__":
    main()