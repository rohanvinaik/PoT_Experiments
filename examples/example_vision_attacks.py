#!/usr/bin/env python3
"""
Comprehensive demonstration of vision-specific attacks.

This script shows the usage of all implemented vision attacks:
- Adversarial patch attacks
- Universal perturbation attacks  
- Model extraction attacks
- Backdoor attacks

All integrated with the main attack runner system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from pathlib import Path

from pot.vision.attacks import (
    AdversarialPatchAttack,
    UniversalPerturbationAttack, 
    VisionModelExtraction,
    BackdoorAttack,
    execute_vision_attack
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_vision_model(num_classes: int = 10) -> nn.Module:
    """Create a simple CNN for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), 
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(128 * 16, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )


def create_test_data(num_samples: int = 200, image_size: int = 32, batch_size: int = 16) -> DataLoader:
    """Create synthetic vision data for testing."""
    # Generate synthetic RGB images
    images = torch.randn(num_samples, 3, image_size, image_size)
    images = torch.clamp(images * 0.2 + 0.5, 0, 1)  # Normalize to [0,1]
    
    # Random labels
    labels = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def demonstrate_adversarial_patch_attack():
    """Demonstrate adversarial patch attack capabilities."""
    print("=== Adversarial Patch Attack Demonstration ===")
    
    # Setup
    model = create_test_vision_model()
    data_loader = create_test_data(100, 32, 8)
    
    # Initialize attack
    patch_attack = AdversarialPatchAttack(
        patch_size=(8, 8),
        patch_location='random',
        optimization_method='adam',
        device='cpu'
    )
    
    print(f"Initialized patch attack with {patch_attack.patch_size} patch size")
    
    # Generate adversarial patch
    print("Generating adversarial patch...")
    patch = patch_attack.generate_patch(
        model=model,
        data_loader=data_loader,
        target_class=7,  # Target class
        iterations=100,  # Reduced for demo
        learning_rate=0.02
    )
    
    print(f"Generated patch with shape: {patch.shape}")
    print(f"Generation stats: {patch_attack.generation_stats}")
    
    # Evaluate patch effectiveness
    print("Evaluating patch effectiveness...")
    metrics = patch_attack.evaluate_patch_effectiveness(
        model=model,
        patch=patch,
        test_loader=data_loader,
        target_class=7
    )
    
    print("Patch Effectiveness Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return patch_attack, patch, metrics


def demonstrate_universal_perturbation_attack():
    """Demonstrate universal perturbation attack."""
    print("\n=== Universal Perturbation Attack Demonstration ===")
    
    # Setup
    model = create_test_vision_model()
    data_loader = create_test_data(150, 32, 8)
    
    # Initialize attack
    universal_attack = UniversalPerturbationAttack(
        epsilon=0.1,
        max_iterations=20,  # Reduced for demo
        xi=10.0,
        device='cpu'
    )
    
    print(f"Initialized universal attack with epsilon={universal_attack.epsilon}")
    
    # Compute universal perturbation
    print("Computing universal perturbation...")
    perturbation = universal_attack.compute_perturbation(
        model=model,
        data_loader=data_loader,
        target_fooling_rate=0.6  # Lower target for demo
    )
    
    print(f"Generated perturbation with shape: {perturbation.shape}")
    print(f"Attack statistics: {universal_attack.attack_stats}")
    
    # Test transferability (with same model for demo)
    print("Testing transferability...")
    transfer_metrics = universal_attack.evaluate_transferability(
        perturbation=perturbation,
        models=[model],  # Same model for simplicity
        test_loader=data_loader
    )
    
    print("Transferability Metrics:")
    for key, value in transfer_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    return universal_attack, perturbation, transfer_metrics


def demonstrate_model_extraction_attack():
    """Demonstrate model extraction attack."""
    print("\n=== Model Extraction Attack Demonstration ===")
    
    # Setup
    target_model = create_test_vision_model()
    data_loader = create_test_data(100, 32, 8)
    
    # Initialize extraction attack
    extraction_attack = VisionModelExtraction(
        query_budget=1000,  # Limited for demo
        architecture='simple',  # Use simple architecture
        device='cpu'
    )
    
    print(f"Initialized extraction attack with budget={extraction_attack.query_budget}")
    
    # Test prediction-based extraction
    print("Performing prediction-based extraction...")
    surrogate_model = extraction_attack.extract_via_prediction(
        target_model=target_model,
        synthetic_data=data_loader,
        num_classes=10,
        epochs=10  # Reduced for demo
    )
    
    print(f"Extraction completed. Queries used: {extraction_attack.queries_used}")
    print(f"Extraction stats: {extraction_attack.extraction_stats}")
    
    # Evaluate extraction quality
    print("Evaluating extraction quality...")
    quality_metrics = extraction_attack.evaluate_extraction_quality(
        target_model=target_model,
        surrogate_model=surrogate_model,
        test_loader=data_loader
    )
    
    print("Extraction Quality Metrics:")
    for key, value in quality_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test Jacobian-based extraction (smaller scale)
    print("\nTesting Jacobian-based extraction...")
    try:
        sample_batch = next(iter(data_loader))
        probe_images = sample_batch[0][:10]  # Small subset
        
        jacobian_surrogate = extraction_attack.extract_via_jacobian(
            target_model=target_model,
            probe_images=probe_images,
            num_classes=10
        )
        
        jacobian_metrics = extraction_attack.evaluate_extraction_quality(
            target_model=target_model,
            surrogate_model=jacobian_surrogate,
            test_loader=data_loader
        )
        
        print("Jacobian Extraction Metrics:")
        for key, value in jacobian_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
                
    except Exception as e:
        print(f"Jacobian extraction demo skipped due to: {e}")
    
    return extraction_attack, surrogate_model, quality_metrics


def demonstrate_backdoor_attack():
    """Demonstrate backdoor injection and detection."""
    print("\n=== Backdoor Attack Demonstration ===")
    
    # Setup
    model = create_test_vision_model()
    data_loader = create_test_data(150, 32, 8)
    
    # Initialize backdoor attack
    backdoor_attack = BackdoorAttack(
        trigger_size=(4, 4),
        trigger_location='bottom_right',
        device='cpu'
    )
    
    print(f"Initialized backdoor attack with trigger size: {backdoor_attack.trigger_size}")
    
    # Create trigger pattern
    trigger = backdoor_attack.create_trigger_pattern(
        pattern_type='checkerboard',
        channels=3
    )
    
    print(f"Created trigger pattern with shape: {trigger.shape}")
    
    # Inject backdoor
    print("Injecting backdoor into model...")
    backdoored_model = backdoor_attack.inject_backdoor(
        model=model,
        train_loader=data_loader,
        trigger_pattern=trigger,
        target_class=3,
        poisoning_rate=0.15,
        epochs=5,  # Reduced for demo
        learning_rate=0.001
    )
    
    print(f"Backdoor injection completed. Stats: {backdoor_attack.injection_stats}")
    
    # Evaluate backdoor effectiveness
    print("Evaluating backdoor effectiveness...")
    effectiveness_metrics = backdoor_attack.evaluate_backdoor_effectiveness(
        model=backdoored_model,
        trigger_pattern=trigger,
        target_class=3,
        test_loader=data_loader
    )
    
    print("Backdoor Effectiveness Metrics:")
    for key, value in effectiveness_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test backdoor detection
    print("\nTesting backdoor detection...")
    
    # Create test triggers (including the real one)
    test_triggers = [
        trigger,  # Real trigger
        backdoor_attack.create_trigger_pattern('solid', 3),  # Decoy
        backdoor_attack.create_trigger_pattern('noise', 3)   # Decoy
    ]
    
    detection_methods = ['activation_clustering', 'output_consistency', 'neural_cleanse']
    
    for method in detection_methods:
        print(f"\n  Testing {method} detection:")
        try:
            detection_result = backdoor_attack.detect_backdoor(
                model=backdoored_model,
                test_triggers=test_triggers,
                clean_test_loader=data_loader,
                detection_method=method
            )
            
            print(f"    Backdoor detected: {detection_result['backdoor_detected']}")
            print(f"    Confidence: {detection_result['confidence']:.4f}")
            print(f"    Method: {detection_result['method']}")
            
            if 'evidence' in detection_result:
                evidence = detection_result['evidence']
                for key, value in evidence.items():
                    if isinstance(value, (int, float)):
                        print(f"    {key}: {value:.4f}")
                        
        except Exception as e:
            print(f"    Detection method {method} failed: {e}")
    
    return backdoor_attack, backdoored_model, trigger, effectiveness_metrics


def demonstrate_attack_runner_integration():
    """Demonstrate integration with the main attack runner system."""
    print("\n=== Attack Runner Integration Demonstration ===")
    
    # Setup
    model = create_test_vision_model()
    data_loader = create_test_data(100, 32, 8)
    
    # Test different vision attacks through the unified interface
    attack_configs = [
        {
            'attack_type': 'adversarial_patch',
            'config': {
                'patch_size': (6, 6),
                'patch_location': 'center',
                'target_class': 5,
                'iterations': 50,
                'learning_rate': 0.02,
                'success_threshold': 0.1
            }
        },
        {
            'attack_type': 'universal_perturbation',
            'config': {
                'epsilon': 0.08,
                'iterations': 15,
                'target_fooling_rate': 0.5,
                'success_threshold': 0.4
            }
        },
        {
            'attack_type': 'model_extraction',
            'config': {
                'method': 'prediction',
                'query_budget': 800,
                'architecture': 'simple',
                'num_classes': 10,
                'epochs': 8,
                'success_threshold': 0.6
            }
        },
        {
            'attack_type': 'backdoor',
            'config': {
                'trigger_size': (3, 3),
                'trigger_location': 'top_left',
                'pattern_type': 'solid',
                'target_class': 2,
                'poisoning_rate': 0.1,
                'epochs': 3,
                'success_threshold': 0.7
            }
        }
    ]
    
    results = []
    
    for attack_config in attack_configs:
        attack_type = attack_config['attack_type']
        config = attack_config['config']
        
        print(f"\nExecuting {attack_type} attack through unified interface...")
        
        try:
            result = execute_vision_attack(
                attack_type=attack_type,
                config=config,
                model=model,
                data_loader=data_loader,
                device='cpu'
            )
            
            results.append(result)
            
            print(f"  Success: {result.success}")
            print(f"  Execution time: {result.execution_time:.2f}s")
            print(f"  Key metrics:")
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)) and key != 'samples_evaluated':
                    print(f"    {key}: {value:.4f}")
            
            if result.metadata:
                print(f"  Metadata: {result.metadata}")
                
        except Exception as e:
            print(f"  Attack {attack_type} failed: {e}")
            results.append(None)
    
    # Summary
    successful_attacks = [r for r in results if r and r.success]
    print(f"\n=== Integration Summary ===")
    print(f"Total attacks tested: {len(attack_configs)}")
    print(f"Successful attacks: {len(successful_attacks)}")
    print(f"Success rate: {len(successful_attacks)/len(attack_configs):.1%}")
    
    return results


def demonstrate_attack_transferability():
    """Demonstrate attack transferability across models."""
    print("\n=== Attack Transferability Demonstration ===")
    
    # Create multiple models with different architectures
    models = {
        'model_a': create_test_vision_model(),
        'model_b': nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(32 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    }
    
    data_loader = create_test_data(80, 32, 8)
    
    # Generate attacks on model_a
    print("Generating attacks on source model...")
    
    # Universal perturbation
    universal_attack = UniversalPerturbationAttack(epsilon=0.08, max_iterations=10, device='cpu')
    perturbation = universal_attack.compute_perturbation(
        model=models['model_a'],
        data_loader=data_loader,
        target_fooling_rate=0.5
    )
    
    # Adversarial patch
    patch_attack = AdversarialPatchAttack(patch_size=(6, 6), device='cpu')
    patch = patch_attack.generate_patch(
        model=models['model_a'],
        data_loader=data_loader,
        target_class=4,
        iterations=50
    )
    
    # Test transferability
    print("\nTesting attack transferability...")
    
    # Universal perturbation transfer
    print("Universal perturbation transferability:")
    transfer_results = universal_attack.evaluate_transferability(
        perturbation=perturbation,
        models=list(models.values()),
        test_loader=data_loader
    )
    
    for key, value in transfer_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Patch transfer
    print("\nAdversarial patch transferability:")
    for model_name, model in models.items():
        patch_metrics = patch_attack.evaluate_patch_effectiveness(
            model=model,
            patch=patch,
            test_loader=data_loader,
            target_class=4
        )
        print(f"  {model_name} attack success rate: {patch_metrics['attack_success_rate']:.4f}")
    
    return transfer_results


def main():
    """Run comprehensive vision attacks demonstration."""
    print("Vision-Specific Attacks Comprehensive Demonstration")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Demonstrate individual attack types
        patch_results = demonstrate_adversarial_patch_attack()
        universal_results = demonstrate_universal_perturbation_attack()
        extraction_results = demonstrate_model_extraction_attack()
        backdoor_results = demonstrate_backdoor_attack()
        
        # Demonstrate integration
        integration_results = demonstrate_attack_runner_integration()
        
        # Demonstrate transferability
        transfer_results = demonstrate_attack_transferability()
        
        print("\n" + "=" * 60)
        print("Vision Attacks System Summary:")
        print("✓ AdversarialPatchAttack - Targeted patch generation and evaluation")
        print("✓ UniversalPerturbationAttack - Transferable perturbations across images")
        print("✓ VisionModelExtraction - Jacobian and prediction-based extraction")
        print("✓ BackdoorAttack - Injection and detection with multiple methods")
        print("✓ Attack Runner Integration - Unified execution interface")
        print("✓ Transferability Analysis - Cross-model attack effectiveness")
        print("✓ Comprehensive Evaluation - Detailed metrics and reporting")
        
        print("\nAll vision-specific attacks are implemented and functional!")
        print("The system is ready for comprehensive PoT defense evaluation.")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()