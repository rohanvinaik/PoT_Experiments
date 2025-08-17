#!/usr/bin/env python3
"""
Simple integration test for attack and defense systems.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pot.core.attack_suites import (
    AttackConfig,
    StandardAttackSuite,
    AttackRunner
)
from pot.core.defenses import (
    DefenseConfig,
    IntegratedDefenseSystem,
    MockBaseVerifier
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
        
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def test_attack_suite():
    """Test attack suite functionality."""
    print("Testing Attack Suite...")
    
    # Create attack config with all required fields
    config = AttackConfig(
        name="test_distillation",
        attack_type="distillation",
        budget={'queries': 1000, 'compute_time': 60},
        strength='moderate',
        success_metrics={'accuracy_drop': 0.1},
        parameters={'temperature': 5.0}
    )
    
    print(f"✓ Created attack config: {config.name}")
    
    # Create standard suite
    suite = StandardAttackSuite()
    configs = suite.get_distillation_configs()
    print(f"✓ Standard suite has {len(configs)} distillation configs")
    
    # Create attack runner
    runner = AttackRunner(device='cpu')
    print("✓ Created attack runner")
    
    # Test metrics calculation
    results = [
        {'success': True, 'confidence': 0.8},
        {'success': False, 'confidence': 0.3}
    ]
    metrics = runner.calculate_metrics(results)
    print(f"✓ Calculated metrics: {metrics['success_rate']:.2f} success rate")
    
    return True


def test_defense_system():
    """Test defense system functionality."""
    print("\nTesting Defense System...")
    
    # Create defense config
    config = DefenseConfig(
        adaptive_threshold=0.05,
        input_filter_strength=0.5,
        randomization_scale=0.1
    )
    print(f"✓ Created defense config")
    
    # Create base verifier
    base_verifier = MockBaseVerifier()
    
    # Create integrated defense system
    defense = IntegratedDefenseSystem(base_verifier)
    print("✓ Created integrated defense system")
    
    # Test with sample input
    model = SimpleModel()
    test_input = torch.randn(10, 784)
    
    result = defense.comprehensive_defense(
        test_input,
        model,
        threat_level=0.5
    )
    
    print(f"✓ Defense pipeline executed")
    print(f"  - Threat level: {result['threat_assessment']['threat_level']:.2f}")
    print(f"  - Final decision: {result['final_decision']['verified']}")
    
    return True


def test_vision_attacks():
    """Test vision attack imports and basic functionality."""
    print("\nTesting Vision Attacks...")
    
    try:
        from pot.vision.attacks import (
            AdversarialPatchAttack,
            execute_vision_attack
        )
        
        # Create patch attack
        attack = AdversarialPatchAttack(
            patch_size=8,
            epsilon=0.03
        )
        print("✓ Created adversarial patch attack")
        
        # Test execution wrapper
        model = SimpleModel()
        data = torch.randn(10, 3, 32, 32)
        labels = torch.randint(0, 10, (10,))
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=5)
        
        result = execute_vision_attack(
            attack_type='adversarial_patch',
            config={'patch_size': 8, 'epsilon': 0.03},
            model=model,
            data_loader=loader,
            device='cpu'
        )
        
        print(f"✓ Vision attack executed: {result['attack_type']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Vision attack test failed: {e}")
        return False


def test_attack_defense_integration():
    """Test integration between attacks and defenses."""
    print("\nTesting Attack-Defense Integration...")
    
    # Create model and data
    model = SimpleModel()
    data = torch.randn(50, 784)
    labels = torch.randint(0, 10, (50,))
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=10)
    
    # Run attack
    runner = AttackRunner(device='cpu')
    attack_config = AttackConfig(
        name="integration_test",
        attack_type="distillation",
        budget={'queries': 100},
        strength='weak',
        success_metrics={'accuracy_drop': 0.05},
        parameters={'temperature': 3.0}
    )
    
    attack_result = runner.run_single_attack(model, attack_config, loader)
    print(f"✓ Attack executed: success={attack_result['success']}")
    
    # Apply defense
    base_verifier = MockBaseVerifier()
    defense = IntegratedDefenseSystem(base_verifier)
    
    defense_result = defense.comprehensive_defense(
        data[:10],
        model,
        threat_level=0.6 if attack_result['success'] else 0.3
    )
    
    print(f"✓ Defense applied: verified={defense_result['final_decision']['verified']}")
    
    # Check if defense detected the attack
    if attack_result['success'] and not defense_result['final_decision']['verified']:
        print("✓ Defense successfully detected attack")
    elif not attack_result['success'] and defense_result['final_decision']['verified']:
        print("✓ Defense correctly verified benign input")
    else:
        print("⚠ Defense response may need tuning")
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Attack-Defense Integration Tests")
    print("="*60)
    
    all_passed = True
    
    # Run tests
    tests = [
        ("Attack Suite", test_attack_suite),
        ("Defense System", test_defense_system),
        ("Vision Attacks", test_vision_attacks),
        ("Integration", test_attack_defense_integration)
    ]
    
    for name, test_func in tests:
        try:
            passed = test_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n✗ {name} test failed with error: {e}")
            all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())